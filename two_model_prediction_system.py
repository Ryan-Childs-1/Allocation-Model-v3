from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from data_io import dataframe_to_csv_bytes, read_allocation_file
from features_two import build_feature_frame
from schema import build_column_map

EPS = 1e-6


@dataclass
class TwoPredictionConfig:
    allocation_threshold: float | None = None
    review_threshold: float | None = None
    demand_cap_extra_flm: float = 1.0
    alloc_rec_influence: str = "balanced"
    prefer_left_dc: bool = True
    allow_partial_leftover_below_flm: bool = True
    review_priority_weight: float = 0.50
    review_probability_weight: float = 0.25
    review_need_weight: float = 0.25
    review_partial_leftover_to_single_row: bool = True
    review_partial_leftover_min_priority: float = 0.50
    prediction_chunk_size: int = 2500
    use_dc_optimizer: bool = True
    use_quantity_correction: bool = True
    use_override_detector: bool = True


# -----------------------------------------------------------------------------
# NumPy-only model loader/inference. No scikit-learn is imported in this file.
# -----------------------------------------------------------------------------
def _joblib_part_paths(path: str | Path) -> list[Path]:
    p = Path(path)
    return sorted(p.parent.glob(p.name + ".part*"))


def _load_joblib_with_split_support(path: str | Path):
    """Load a NumPy-only joblib bundle, including split .partXX files.

    This function intentionally catches ModuleNotFoundError for sklearn. That
    means if an older sklearn-pickled model file is accidentally left in the
    repo, the loader can skip it and continue looking for the NumPy-only model
    instead of crashing the app.
    """
    p = Path(path)
    try:
        if p.exists():
            return joblib.load(p)
        parts = _joblib_part_paths(p)
        if parts:
            return joblib.load(io.BytesIO(b"".join(part.read_bytes() for part in parts)))
    except ModuleNotFoundError as exc:
        if "sklearn" in str(exc).lower():
            raise ValueError(
                f"{p.name} appears to be an old sklearn-pickled model. "
                "This app requires the NumPy-only converted model files. "
                "Use base_allocation_numpy_model.joblib and base_review_numpy_model.joblib."
            ) from exc
        raise
    raise FileNotFoundError(f"Could not find model file {p} or split parts like {p.name}.part01")


def _is_supported_numpy_bundle(bundle: dict) -> bool:
    """Accept both exported NumPy bundle formats used by Allocation AI.

    v1 was exported from sklearn MLP weights converted to NumPy.
    v2 / keras_to_numpy was exported directly from Keras-trained dense layers.
    Both run with NumPy only and require no TensorFlow or scikit-learn at runtime.
    """
    if not isinstance(bundle, dict):
        return False
    if bundle.get("format") == "allocation_ai_numpy_mlp_v1":
        return True
    if str(bundle.get("bundle_type", "")).startswith("allocation_ai_keras_numpy"):
        return True
    if bundle.get("metadata", {}).get("backend") == "keras_to_numpy":
        return True
    return False


def _load_bundle(path: str | Path) -> dict:
    bundle = _load_joblib_with_split_support(path)
    if not _is_supported_numpy_bundle(bundle):
        raise ValueError(
            f"{path} is not a supported NumPy-only Allocation AI model bundle. "
            "This Streamlit package intentionally does not use scikit-learn or TensorFlow. "
            "Use the converted NumPy model bundles exported by the trainer."
        )
    return bundle


def load_two_models(folder_or_zip: str | Path | None = None) -> Dict[str, dict]:
    if folder_or_zip is None:
        return _load_models_from_dir(Path("."))
    p = Path(folder_or_zip)
    if p.suffix.lower() == ".zip":
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(root)
        models = _load_models_from_dir(root)
        models["__tempdir"] = td
        return models
    return _load_models_from_dir(p)


def _load_models_from_dir(root: Path) -> Dict[str, dict]:
    """Load the built-in NumPy-only Base Allocation and Base Review models.

    The preferred filenames deliberately include `_numpy_` so they do not collide
    with older sklearn-pickled model artifacts that may remain in a GitHub repo
    after an update. Older filenames are tried only as a fallback and are skipped
    if they require sklearn.
    """
    candidate_paths = {
        "Base Allocation": [
            Path(root) / "base_allocation_numpy_model.joblib",
            Path(root) / "base_allocation_model.joblib",
        ],
        "Base Review": [
            Path(root) / "base_review_numpy_model.joblib",
            Path(root) / "base_review_model.joblib",
        ],
    }
    models: Dict[str, dict] = {}
    load_errors: Dict[str, list[str]] = {}
    for label, paths in candidate_paths.items():
        load_errors[label] = []
        for path in paths:
            if not (path.exists() or _joblib_part_paths(path)):
                continue
            try:
                bundle = _load_bundle(path)
                bundle["__model_file"] = str(path)
                models[label] = bundle
                break
            except Exception as exc:
                load_errors[label].append(f"{path.name}: {type(exc).__name__}: {exc}")
        if label not in models:
            load_errors[label].append("No usable NumPy-only model file found.")
    missing = [k for k in candidate_paths if k not in models]
    if missing:
        details = "; ".join(f"{k} -> " + " | ".join(load_errors.get(k, [])) for k in missing)
        raise FileNotFoundError(
            f"Missing required NumPy-only model(s): {missing}. Details: {details}. "
            "Confirm the repo includes base_allocation_numpy_model.joblib and base_review_numpy_model.joblib."
        )
    return models


def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _num(df: pd.DataFrame, cmap: dict, field: str, default: float = 0.0) -> pd.Series:
    c = cmap.get(field)
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _txt(df: pd.DataFrame, cmap: dict, field: str) -> pd.Series:
    c = cmap.get(field)
    if c in df.columns:
        return df[c].astype(str).fillna("").str.strip()
    return pd.Series("", index=df.index, dtype="object")


def _flag_masks(flag: pd.Series):
    f = flag.astype(str).str.upper().fillna("")
    is_z = ((f.str.contains("NO") & f.str.contains("ALLOC")) | f.str.startswith("Z - NO") | f.str.startswith("Z NO"))
    is_review = f.str.contains("REVIEW")
    is_alloc = f.str.contains("ALLOC") & ~is_z
    return is_alloc, is_review, is_z


def _sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(x, -60, 60))
    return e / np.maximum(e.sum(axis=1, keepdims=True), EPS)


def _activation(x, name: str):
    if name == "relu":
        return np.maximum(x, 0)
    if name == "logistic":
        return _sigmoid(x)
    if name == "tanh":
        return np.tanh(x)
    if name == "identity":
        return x
    return np.maximum(x, 0)


def _mlp_raw(X: np.ndarray, model: dict) -> np.ndarray:
    h = X.astype(np.float32, copy=False)
    coefs = model["coefs"]
    intercepts = model["intercepts"]
    hidden_activation = model.get("activation", "relu")
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        h = h @ w + b
        if i < len(coefs) - 1:
            h = _activation(h, hidden_activation)
    return h


def _mlp_predict(model: dict, X: np.ndarray) -> np.ndarray:
    # Keras-to-NumPy direct layer format.
    if "layers" in model:
        raw = _keras_layers_raw(X, model)
        output_type = model.get("output_type", "regression")
        if output_type == "class":
            return np.argmax(raw, axis=1).astype(int)
        if output_type == "binary":
            return (_sigmoid(raw).reshape(-1) >= 0.5).astype(int)
        return raw.reshape(-1)

    raw = _mlp_raw(X, model)
    out_activation = model.get("out_activation", "identity")
    classes = model.get("classes")
    if classes is not None:
        classes = np.asarray(classes)
        if out_activation == "softmax":
            p = _softmax(raw)
            return classes[np.argmax(p, axis=1)]
        if out_activation == "logistic":
            p1 = _sigmoid(raw).reshape(-1)
            idx = (p1 >= 0.5).astype(int)
            return classes[np.minimum(idx, len(classes)-1)]
        return classes[np.argmax(raw, axis=1)]
    return raw.reshape(-1)


def _mlp_predict_proba_positive(model: dict, X: np.ndarray) -> np.ndarray:
    # Keras-to-NumPy direct layer format.
    if "layers" in model:
        raw = _keras_layers_raw(X, model)
        output_type = model.get("output_type", "regression")
        if output_type == "binary":
            return _sigmoid(raw).reshape(-1)
        if output_type == "class":
            p = _softmax(raw)
            return p[:, 1] if p.shape[1] > 1 else np.zeros(len(X), dtype=np.float32)
        return np.clip(raw.reshape(-1), 0, 1)

    raw = _mlp_raw(X, model)
    classes = model.get("classes")
    out_activation = model.get("out_activation", "identity")
    if classes is None:
        return np.clip(raw.reshape(-1), 0, 1)
    classes = list(np.asarray(classes).tolist())
    if out_activation == "softmax":
        p = _softmax(raw)
        return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X), dtype=np.float32)
    if out_activation == "logistic":
        p1 = _sigmoid(raw).reshape(-1)
        return p1 if 1 in classes else 1.0 - p1
    return np.zeros(len(X), dtype=np.float32)


def _transform_numpy_preprocessor(X: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Transform feature frame using the NumPy-only preprocessor stored in the bundle.

    Supports:
      - allocation_ai_numpy_mlp_v1: sklearn-export style keys {numeric, categorical}
      - keras_to_numpy bundles: direct keys {num_cols, cat_cols, median, mean, scale, categories}
    """
    pre = bundle["preprocessor"]

    # New Keras-to-NumPy export format.
    if "num_cols" in pre or "cat_cols" in pre:
        parts = []
        ncols = list(pre.get("num_cols", []))
        if ncols:
            arr = np.zeros((len(X), len(ncols)), dtype=np.float32)
            med_src = pre.get("median", {})
            mean_src = pre.get("mean", {})
            scale_src = pre.get("scale", {})
            if isinstance(med_src, dict):
                med = np.asarray([med_src.get(c, 0.0) for c in ncols], dtype=np.float32)
            else:
                med = np.asarray(med_src if med_src is not None else np.zeros(len(ncols)), dtype=np.float32)
            if isinstance(mean_src, dict):
                mean = np.asarray([mean_src.get(c, 0.0) for c in ncols], dtype=np.float32)
            else:
                mean = np.asarray(mean_src if mean_src is not None else np.zeros(len(ncols)), dtype=np.float32)
            if isinstance(scale_src, dict):
                scale = np.asarray([scale_src.get(c, 1.0) for c in ncols], dtype=np.float32)
            else:
                scale = np.asarray(scale_src if scale_src is not None else np.ones(len(ncols)), dtype=np.float32)
            scale = np.where(np.abs(scale) < EPS, 1.0, scale)
            for j, c in enumerate(ncols):
                if c in X.columns:
                    vals = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
                else:
                    vals = np.full(len(X), np.nan, dtype=np.float32)
                vals = np.where(np.isfinite(vals), vals, med[j])
                arr[:, j] = (vals - mean[j]) / scale[j]
            parts.append(arr)

        ccols = list(pre.get("cat_cols", []))
        categories = pre.get("categories", {}) or {}
        if ccols:
            widths = []
            maps = []
            for c in ccols:
                cmap = categories.get(c, {})
                if isinstance(cmap, dict):
                    maps.append({str(k): int(v) for k, v in cmap.items()})
                    widths.append((max([int(v) for v in cmap.values()] + [-1]) + 1))
                else:
                    vals = list(cmap)
                    maps.append({str(v): i for i, v in enumerate(vals)})
                    widths.append(len(vals))
            out = np.zeros((len(X), int(sum(widths))), dtype=np.float32)
            offset = 0
            for j, c in enumerate(ccols):
                lookup = maps[j]
                width = widths[j]
                vals = X[c].astype(str).fillna("").str.strip().tolist() if c in X.columns else [""] * len(X)
                for i, v in enumerate(vals):
                    idx = lookup.get(str(v))
                    if idx is not None and 0 <= idx < width:
                        out[i, offset + idx] = 1.0
                offset += width
            parts.append(out)

        return np.concatenate(parts, axis=1).astype(np.float32, copy=False) if parts else np.zeros((len(X), 0), dtype=np.float32)

    # Original v1 sklearn-export-to-NumPy format.
    num = pre.get("numeric", {})
    cat = pre.get("categorical", {})
    parts = []

    ncols = list(num.get("columns", []))
    if ncols:
        arr = np.zeros((len(X), len(ncols)), dtype=np.float32)
        stats = np.asarray(num.get("statistics", np.zeros(len(ncols))), dtype=np.float32)
        mean = np.asarray(num.get("mean", np.zeros(len(ncols))), dtype=np.float32)
        scale = np.asarray(num.get("scale", np.ones(len(ncols))), dtype=np.float32)
        scale = np.where(np.abs(scale) < EPS, 1.0, scale)
        for j, c in enumerate(ncols):
            if c in X.columns:
                s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                vals = s.to_numpy(dtype=np.float32)
            else:
                vals = np.full(len(X), np.nan, dtype=np.float32)
            vals = np.where(np.isfinite(vals), vals, stats[j])
            arr[:, j] = (vals - mean[j]) / scale[j]
        parts.append(arr)

    ccols = list(cat.get("columns", []))
    if ccols:
        nouts = list(map(int, cat.get("n_features_outs", [])))
        total = int(sum(nouts))
        out = np.zeros((len(X), total), dtype=np.float32)
        cat_to_idx = cat.get("category_to_index", [])
        maps = cat.get("maps", [])
        fill = str(cat.get("fill_value", "") or "")
        offset = 0
        for j, c in enumerate(ccols):
            width = nouts[j]
            lookup = cat_to_idx[j] if j < len(cat_to_idx) else {}
            mapping = np.asarray(maps[j], dtype=np.int32) if j < len(maps) else None
            vals = X[c].astype(str).fillna(fill).str.strip().tolist() if c in X.columns else [fill] * len(X)
            for i, v in enumerate(vals):
                orig_idx = lookup.get(str(v))
                if orig_idx is None:
                    continue
                mapped = int(mapping[orig_idx]) if mapping is not None and orig_idx < len(mapping) else int(orig_idx)
                if 0 <= mapped < width:
                    out[i, offset + mapped] = 1.0
            offset += width
        parts.append(out)

    if not parts:
        return np.zeros((len(X), 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _keras_layers_raw(X: np.ndarray, model: dict) -> np.ndarray:
    h = X.astype(np.float32, copy=False)
    layers = model.get("layers", [])
    for i, layer in enumerate(layers):
        h = h @ np.asarray(layer["W"], dtype=np.float32) + np.asarray(layer["b"], dtype=np.float32)
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h

def _predict_model(df: pd.DataFrame, bundle: dict, chunk_size: int = 2500, use_quantity_correction: bool = True, use_override_detector: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict units/probability/priority with optional v7 auxiliary heads.

    The v7 trainer may export two extra NumPy heads:
      - quantity_correction_model: predicts a -3..+3 FLM adjustment
      - override_model: estimates when the workbook Alloc. Rec. is likely overridden

    Older NumPy bundles do not have those heads, so this remains backward-compatible.
    """
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    X_all = build_feature_frame(df)
    cols = list(bundle["feature_columns"])
    for c in cols:
        if c not in X_all.columns:
            X_all[c] = 0.0 if c.startswith("num__") else ""
    all_units, all_prob, all_priority, all_override = [], [], [], []
    chunk_size = max(250, int(chunk_size or 2500))
    for start in range(0, len(X_all), chunk_size):
        end = min(len(X_all), start + chunk_size)
        X = X_all.iloc[start:end][cols].replace([np.inf, -np.inf], np.nan)
        Xt = _transform_numpy_preprocessor(X, bundle)
        units = np.asarray(_mlp_predict(bundle["unit_model"], Xt), dtype=int)
        prob = np.asarray(_mlp_predict_proba_positive(bundle["prob_model"], Xt), dtype=float)
        priority = np.asarray(_mlp_predict(bundle["priority_model"], Xt), dtype=float).clip(0, 1)
        override_prob = np.zeros(len(X), dtype=float)

        # Optional v7 quantity correction model. It is a 7-class model where
        # class 0..6 maps to delta -3..+3 FLM units.
        q_model = bundle.get("quantity_correction_model")
        if use_quantity_correction and isinstance(q_model, dict):
            try:
                q_raw = _mlp_predict(q_model, Xt)
                # _mlp_predict returns class index for Keras output_type='class'.
                delta = np.asarray(q_raw, dtype=int) - 3
                units = np.clip(units + delta, 0, 9999).astype(int)
            except Exception:
                pass

        # Optional v7 override detector model.
        o_model = bundle.get("override_model")
        if use_override_detector and isinstance(o_model, dict):
            try:
                override_prob = np.asarray(_mlp_predict_proba_positive(o_model, Xt), dtype=float)
            except Exception:
                override_prob = np.zeros(len(X), dtype=float)

        all_units.append(units); all_prob.append(prob); all_priority.append(priority); all_override.append(override_prob)
    return np.concatenate(all_units), np.concatenate(all_prob), np.concatenate(all_priority), np.concatenate(all_override)

def _threshold(bundle: dict, fallback: float = 0.35) -> float:
    try:
        meta = bundle.get("metadata", {})
        th = meta.get("best_threshold") or meta.get("best_validation_metrics", {}).get("threshold")
        if th is not None:
            th = float(th)
            if 0 < th <= 1:
                return th
    except Exception:
        pass
    return fallback


def _round_alloc(capped: float, flm: int, allow_partial: bool = True) -> int:
    flm = max(int(round(flm or 1)), 1)
    capped = max(float(capped), 0.0)
    if capped <= 0:
        return 0
    if capped < flm and allow_partial:
        return int(np.floor(capped))
    if capped < flm:
        return 0
    return int(np.floor(capped / flm) * flm)


def predict_allocation_file(df: pd.DataFrame, models: Dict[str, dict], cfg: TwoPredictionConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = cfg or TwoPredictionConfig()
    raw = df.copy()
    cmap = build_column_map(raw)
    n = len(raw)
    item = _txt(raw, cmap, "item").replace("", "__missing_item__")
    flag = _txt(raw, cmap, "flag")
    is_alloc, is_review, is_z = _flag_masks(flag)

    flm = _num(raw, cmap, "flm", 1).fillna(1).where(lambda s: s > 0, 1).round().astype(int)
    supply = _num(raw, cmap, "supply", 0).fillna(0).clip(lower=0)
    d60 = _num(raw, cmap, "d60", 0).fillna(0).clip(lower=0)
    d30 = _num(raw, cmap, "d30", 0).fillna(0).clip(lower=0)
    l30 = _num(raw, cmap, "l30", 0).fillna(0).clip(lower=0)
    lw = _num(raw, cmap, "lw", 0).fillna(0).clip(lower=0)
    ttm = _num(raw, cmap, "ttm", 0).fillna(0).clip(lower=0)
    proj = _num(raw, cmap, "proj_demand", 0).fillna(0).clip(lower=0)
    alloc_rec = _num(raw, cmap, "alloc_rec", 0).fillna(0).clip(lower=0)
    dc_avail = _num(raw, cmap, "dc_avail", 0).fillna(0).clip(lower=0)
    left_dc = _num(raw, cmap, "left_dc", np.nan)

    demand_basis = pd.Series(np.maximum.reduce([d60.values, proj.values, (l30*2).values, (d30*2).values, (ttm/6).values, (lw*8).values]), index=raw.index)

    remaining = {}
    for it, idxs in item.groupby(item, sort=False).groups.items():
        idxs = list(idxs)
        ld = left_dc.loc[idxs].dropna().clip(lower=0)
        da = dc_avail.loc[idxs].dropna().clip(lower=0)
        if cfg.prefer_left_dc and len(ld) and ld.max() > 0:
            remaining[it] = float(ld.max())
        elif len(da) and da.max() > 0:
            remaining[it] = float(da.max())
        elif len(ld):
            remaining[it] = float(ld.max())
        else:
            remaining[it] = 0.0

    final = pd.Series("", index=raw.index, dtype=object)
    audit_rows = []
    predictions = {}
    for label, mask in [("Base Allocation", is_alloc), ("Base Review", is_review)]:
        idxs = raw.index[mask]
        if len(idxs):
            units, prob, priority, override_prob = _predict_model(
                raw.loc[idxs],
                models[label],
                cfg.prediction_chunk_size,
                use_quantity_correction=bool(getattr(cfg, "use_quantity_correction", True)),
                use_override_detector=bool(getattr(cfg, "use_override_detector", True)),
            )
            predictions[label] = pd.DataFrame({"units": units, "prob": prob, "priority": priority, "override_prob": override_prob}, index=idxs)
        else:
            predictions[label] = pd.DataFrame(columns=["units", "prob", "priority"])

    work = []
    for idx in raw.index[is_alloc]:
        pred = predictions["Base Allocation"].loc[idx]
        work.append((0, idx, float(pred["priority"]), "Base Allocation"))
    for idx in raw.index[is_review]:
        pred = predictions["Base Review"].loc[idx]
        f = max(int(flm.loc[idx]), 1)
        need_units = max(0.0, float(demand_basis.loc[idx] - supply.loc[idx]) / (f + EPS))
        score = cfg.review_priority_weight * float(pred["priority"]) + cfg.review_probability_weight * float(pred["prob"]) + cfg.review_need_weight * np.tanh(need_units / 3.0)
        work.append((1, idx, score, "Base Review"))
    if bool(getattr(cfg, "use_dc_optimizer", True)):
        work = sorted(work, key=lambda x: (x[0], -x[2], int(raw.loc[x[1], "__row_order"]) if "__row_order" in raw.columns else 0))
    else:
        work = sorted(work, key=lambda x: (x[0], int(raw.loc[x[1], "__row_order"]) if "__row_order" in raw.columns else 0))

    for pass_order, idx, section_score, model_label in work:
        it = item.loc[idx]
        f = max(int(flm.loc[idx]), 1)
        pred = predictions[model_label].loc[idx]
        units = max(int(pred["units"]), 0)
        prob = float(pred["prob"])
        priority = float(pred["priority"])
        override_probability = float(pred.get("override_prob", 0.0))
        threshold = cfg.allocation_threshold if model_label == "Base Allocation" else cfg.review_threshold
        if threshold is None:
            threshold = _threshold(models[model_label], 0.35)
        raw_alloc = units * f
        left_before = float(remaining.get(it, 0.0))
        cap = max(0.0, float(demand_basis.loc[idx]) + cfg.demand_cap_extra_flm * f - float(supply.loc[idx]))
        if cfg.alloc_rec_influence == "hard_cap" and alloc_rec.loc[idx] > 0:
            cap = min(cap, float(alloc_rec.loc[idx]))
        elif cfg.alloc_rec_influence == "balanced" and alloc_rec.loc[idx] > 0:
            cap = min(cap, max(float(alloc_rec.loc[idx]) + f, f))
        elif cfg.alloc_rec_influence == "soft_cap" and alloc_rec.loc[idx] > 0:
            cap = min(cap, max(float(alloc_rec.loc[idx]) + 2*f, f))
        reason = []
        partial_review_leftover = False
        partial_leftover_qty = 0
        if model_label == "Base Review" and cfg.allow_partial_leftover_below_flm and cfg.review_partial_leftover_to_single_row and 0 < left_before < f:
            partial_leftover_qty = int(np.floor(left_before))
            need_units_for_row = max(0.0, float(demand_basis.loc[idx] - supply.loc[idx]) / (f + EPS))
            review_support = prob >= threshold or priority >= float(cfg.review_partial_leftover_min_priority) or float(alloc_rec.loc[idx]) > 0 or need_units_for_row > 0
            cap_support = cap >= partial_leftover_qty > 0
            if review_support and cap_support:
                raw_alloc = max(int(raw_alloc), partial_leftover_qty)
                partial_review_leftover = True
                reason.append("review_partial_leftover_assigned_to_top_row")
        if prob < threshold and not partial_review_leftover:
            raw_alloc = 0
            reason.append("below_section_threshold")
        if left_before <= 0:
            raw_alloc = 0
            reason.append("no_left_dc")
        capped = min(float(raw_alloc), left_before, cap)
        alloc = _round_alloc(capped, f, cfg.allow_partial_leftover_below_flm)
        if alloc > 0:
            final.loc[idx] = int(alloc)
            remaining[it] = max(0.0, left_before - alloc)
            reason.append("approved_section_allocation" if alloc >= raw_alloc else "capped_by_dc_demand_or_alloc_rec")
        else:
            final.loc[idx] = ""
            remaining[it] = left_before
            if not reason:
                reason.append("rounded_or_capped_to_blank")
        audit_rows.append({
            "row_order": int(raw.get("__row_order", pd.Series(range(n), index=raw.index)).loc[idx]),
            "excel_row": int(raw.get("__excel_row", pd.Series(range(2, n+2), index=raw.index)).loc[idx]),
            "model_used": model_label,
            "flag": str(flag.loc[idx]),
            "item": str(it),
            "section_score": float(section_score),
            "probability": prob,
            "priority": priority,
            "override_probability": override_probability,
            "predicted_units": units,
            "flm": f,
            "raw_alloc": int(raw_alloc),
            "left_dc_before": left_before,
            "partial_leftover_below_flm": bool(0 < left_before < f),
            "review_partial_leftover_to_single_row": bool(partial_review_leftover),
            "review_partial_leftover_qty": int(partial_leftover_qty),
            "final_alloc": int(alloc) if alloc > 0 else "",
            "left_dc_after": float(remaining[it]),
            "demand_basis": float(demand_basis.loc[idx]),
            "alloc_rec": float(alloc_rec.loc[idx]),
            "reason": "; ".join(reason),
        })

    for idx in raw.index[~(is_alloc | is_review)]:
        audit_rows.append({
            "row_order": int(raw.get("__row_order", pd.Series(range(n), index=raw.index)).loc[idx]),
            "excel_row": int(raw.get("__excel_row", pd.Series(range(2, n+2), index=raw.index)).loc[idx]),
            "model_used": "Ignored",
            "flag": str(flag.loc[idx]),
            "item": str(item.loc[idx]),
            "section_score": 0.0,
            "probability": 0.0,
            "priority": 0.0,
            "predicted_units": 0,
            "flm": int(flm.loc[idx]),
            "raw_alloc": 0,
            "left_dc_before": float(remaining.get(item.loc[idx], 0.0)),
            "partial_leftover_below_flm": False,
            "review_partial_leftover_to_single_row": False,
            "review_partial_leftover_qty": 0,
            "final_alloc": "",
            "left_dc_after": float(remaining.get(item.loc[idx], 0.0)),
            "demand_basis": float(demand_basis.loc[idx]),
            "alloc_rec": float(alloc_rec.loc[idx]),
            "reason": "ignored_not_allocate_or_review",
        })

    out = raw.copy()
    final_col = cmap.get("final_alloc") or "Final Alloc."
    if final_col not in out.columns:
        out[final_col] = ""
    out[final_col] = final.values
    if "__row_order" in out.columns:
        out = out.sort_values("__row_order")
    audit = pd.DataFrame(audit_rows).sort_values("row_order") if audit_rows else pd.DataFrame()
    final_numeric = pd.to_numeric(audit.get("final_alloc", pd.Series(dtype=float)), errors="coerce").fillna(0)
    review_partial_rows = int(audit.get("review_partial_leftover_to_single_row", pd.Series(False)).astype(bool).sum()) if not audit.empty else 0
    review_partial_units = int(pd.to_numeric(audit.loc[audit.get("review_partial_leftover_to_single_row", pd.Series(False)).astype(bool), "final_alloc"], errors="coerce").fillna(0).sum()) if review_partial_rows else 0
    summary = {
        "rows": int(len(out)),
        "allocated_rows": int((final_numeric > 0).sum()),
        "total_final_alloc": int(final_numeric.sum()),
        "models": ["Base Allocation", "Base Review"],
        "ignored_no_alloc_rows": int((~(is_alloc | is_review)).sum()),
        "section_rows": {"Base Allocation": int(is_alloc.sum()), "Base Review": int(is_review.sum()), "Ignored": int((~(is_alloc | is_review)).sum())},
        "review_partial_leftover_rows": review_partial_rows,
        "review_partial_leftover_units": review_partial_units,
        "allocation_threshold": float(cfg.allocation_threshold if cfg.allocation_threshold is not None else _threshold(models["Base Allocation"], 0.35)),
        "review_threshold": float(cfg.review_threshold if cfg.review_threshold is not None else _threshold(models["Base Review"], 0.35)),
        "inference_backend": "numpy_only_no_sklearn",
        "use_dc_optimizer": bool(getattr(cfg, "use_dc_optimizer", True)),
        "use_quantity_correction": bool(getattr(cfg, "use_quantity_correction", True)),
        "use_override_detector": bool(getattr(cfg, "use_override_detector", True)),
    }
    return out, audit, summary


# -----------------------------------------------------------------------------
# Explanation helpers
# -----------------------------------------------------------------------------
def _simplify_feature_name(name: str) -> str:
    text = str(name)
    for prefix in ("num__", "cat__"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.startswith("num__"):
        return text[len("num__"):]
    if text.startswith("cat__"):
        rest = text[len("cat__"):]
        parts = rest.split("_")
        return parts[0] if parts else rest
    return text


def feature_family(name: str) -> str:
    n = _simplify_feature_name(name).lower()
    if "alloc_rec" in n:
        return "Alloc. Rec."
    if "left_dc" in n or "dc_" in n or "to_dc" in n:
        return "DC / Left DC"
    if any(k in n for k in ["demand", "d60", "d30", "l30", "ttm", "lw", "velocity"]):
        return "Demand / velocity"
    if any(k in n for k in ["need", "gap", "pressure", "scarcity"]):
        return "Need / scarcity"
    if "supply" in n or "qoh" in n:
        return "Supply"
    if any(k in n for k in ["review", "flag", "section"]):
        return "Flag / section"
    if any(k in n for k in ["rank", "share", "total", "cum", "remaining"]):
        return "Section/group ranking"
    if any(k in n for k in ["retail", "cost", "margin", "gm"]):
        return "Retail / margin"
    if n in {"item", "site", "upc", "description"}:
        return "Categorical identity"
    return "Other"


def _transformed_names_from_bundle(bundle: dict, input_dim: int | None = None) -> list[str]:
    names = list(bundle.get("transformed_feature_names", []) or [])
    if names:
        return names
    pre = bundle.get("preprocessor", {})
    out = []
    if "num_cols" in pre or "cat_cols" in pre:
        out.extend(list(pre.get("num_cols", [])))
        cats = pre.get("categories", {}) or {}
        for c in list(pre.get("cat_cols", [])):
            cmap = cats.get(c, {})
            if isinstance(cmap, dict):
                # Sort by encoded index so names align with the one-hot columns.
                vals = sorted(cmap.items(), key=lambda kv: int(kv[1]))
                out.extend([f"{c}_{str(v)}" for v, _idx in vals])
            else:
                out.extend([f"{c}_{str(v)}" for v in list(cmap)])
    else:
        num = pre.get("numeric", {})
        cat = pre.get("categorical", {})
        out.extend(list(num.get("columns", [])))
        ccols = list(cat.get("columns", []))
        categories = cat.get("categories", [])
        nouts = list(map(int, cat.get("n_features_outs", []))) if cat.get("n_features_outs") is not None else []
        for j, c in enumerate(ccols):
            vals = list(categories[j]) if j < len(categories) else [str(i) for i in range(nouts[j] if j < len(nouts) else 0)]
            if j < len(nouts):
                vals = vals[:nouts[j]]
            out.extend([f"{c}_{str(v)}" for v in vals])
    if input_dim is not None and len(out) != int(input_dim):
        return [f"transformed_feature_{i}" for i in range(int(input_dim))]
    return out


def model_feature_importance(bundle: dict, model_label: str, top_n: int = 80) -> pd.DataFrame:
    rows = []
    for head_name, key in [("unit", "unit_model"), ("probability", "prob_model"), ("priority", "priority_model"), ("quantity_correction", "quantity_correction_model"), ("override_detector", "override_model")]:
        model = bundle.get(key)
        if not model:
            continue
        if model.get("coefs"):
            w = np.asarray(model["coefs"][0])
        elif model.get("layers"):
            w = np.asarray(model["layers"][0]["W"])
        else:
            continue
        imp = np.mean(np.abs(w), axis=1)
        names = _transformed_names_from_bundle(bundle, len(imp))
        for fname, val in zip(names, imp):
            base = _simplify_feature_name(fname)
            rows.append({
                "model": model_label,
                "head": head_name,
                "transformed_feature": str(fname),
                "base_feature": base,
                "feature_family": feature_family(base),
                "importance": float(val),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    grouped = df.groupby(["model", "feature_family", "base_feature"], as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    return grouped.head(top_n).reset_index(drop=True)


def prediction_feature_relationships(df: pd.DataFrame, audit_df: pd.DataFrame, top_n: int = 60) -> pd.DataFrame:
    try:
        X = build_feature_frame(df)
    except Exception:
        return pd.DataFrame()
    target_prob = pd.to_numeric(audit_df.get("probability", pd.Series(0, index=audit_df.index)), errors="coerce").fillna(0)
    target_alloc = pd.to_numeric(audit_df.get("final_alloc", pd.Series(0, index=audit_df.index)), errors="coerce").fillna(0)
    rows = []
    for c in [c for c in X.columns if c.startswith("num__")]:
        s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().mean() == 0 or s.nunique(dropna=True) <= 1:
            continue
        sf = s.fillna(s.median())
        def corr(a, b):
            try:
                m = min(len(a), len(b))
                v = float(np.corrcoef(a[:m], b[:m])[0, 1])
                return 0.0 if not np.isfinite(v) else v
            except Exception:
                return 0.0
        cp = corr(sf.values, target_prob.values)
        ca = corr(sf.values, target_alloc.values)
        base = c.replace("num__", "")
        rows.append({
            "feature": base,
            "feature_family": feature_family(base),
            "corr_with_probability": cp,
            "corr_with_final_alloc": ca,
            "relationship_strength": max(abs(cp), abs(ca)),
        })
    return pd.DataFrame(rows).sort_values("relationship_strength", ascending=False).head(top_n).reset_index(drop=True) if rows else pd.DataFrame()


def predict_file_to_csvs(file_path: str | Path, model_folder_or_zip: str | Path, output_dir: str | Path = "prediction_outputs") -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = read_allocation_file(file_path)
    models = load_two_models(model_folder_or_zip)
    out, audit, summary = predict_allocation_file(df, models)
    (output_dir / "completed_allocation.csv").write_bytes(dataframe_to_csv_bytes(out))
    (output_dir / "allocation_audit.csv").write_bytes(dataframe_to_csv_bytes(audit))
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
