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
    alloc_rec_influence: str = "balanced"  # feature_only, soft_cap, balanced, hard_cap
    prefer_left_dc: bool = True
    allow_partial_leftover_below_flm: bool = True
    review_priority_weight: float = 0.50
    review_probability_weight: float = 0.25
    review_need_weight: float = 0.25
    review_partial_leftover_to_single_row: bool = True
    review_partial_leftover_min_priority: float = 0.50
    prediction_chunk_size: int = 2500


# -----------------------------------------------------------------------------
# scikit-learn compatibility repairs
# -----------------------------------------------------------------------------
def _walk_estimator_tree(obj: Any, seen: set[int] | None = None):
    if obj is None:
        return
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    yield obj
    steps = getattr(obj, "steps", None)
    if steps:
        for _name, step in steps:
            yield from _walk_estimator_tree(step, seen)
    transformers = getattr(obj, "transformers_", None) or getattr(obj, "transformers", None)
    if transformers:
        for item in transformers:
            if not item or len(item) < 2:
                continue
            trans = item[1]
            if trans in (None, "drop", "passthrough"):
                continue
            yield from _walk_estimator_tree(trans, seen)
    for attr in ("estimator", "estimator_", "base_estimator", "base_estimator_", "calibrated_classifiers_"):
        try:
            val = getattr(obj, attr, None)
        except Exception:
            val = None
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            for child in val:
                yield from _walk_estimator_tree(child, seen)
        else:
            yield from _walk_estimator_tree(val, seen)


def repair_sklearn_pickle_compat(bundle: dict) -> dict:
    if not isinstance(bundle, dict):
        return bundle
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        SimpleImputer = None
    try:
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        OneHotEncoder = None
    repairs = []
    for root in [bundle.get("preprocessor"), bundle.get("unit_model"), bundle.get("prob_model"), bundle.get("priority_model")]:
        for est in _walk_estimator_tree(root):
            cls = est.__class__.__name__
            if cls == "ColumnTransformer":
                for attr, val in [("force_int_remainder_cols", "deprecated"), ("verbose_feature_names_out", True)]:
                    if not hasattr(est, attr):
                        try:
                            setattr(est, attr, val)
                            repairs.append(f"ColumnTransformer.{attr}")
                        except Exception:
                            pass
            if (SimpleImputer is not None and isinstance(est, SimpleImputer)) or cls == "SimpleImputer":
                if not hasattr(est, "_fill_dtype"):
                    fill_dtype = getattr(est, "_fit_dtype", None)
                    stats = getattr(est, "statistics_", None)
                    if fill_dtype is None and stats is not None:
                        fill_dtype = getattr(stats, "dtype", None)
                    if fill_dtype is None:
                        fill_dtype = object if getattr(est, "strategy", None) == "constant" else float
                    try:
                        est._fill_dtype = fill_dtype
                        repairs.append("SimpleImputer._fill_dtype")
                    except Exception:
                        pass
                for attr, val in [("keep_empty_features", False), ("indicator_", None), ("add_indicator", False)]:
                    if not hasattr(est, attr):
                        try:
                            setattr(est, attr, val)
                            repairs.append(f"SimpleImputer.{attr}")
                        except Exception:
                            pass
            if (OneHotEncoder is not None and isinstance(est, OneHotEncoder)) or cls == "OneHotEncoder":
                if not hasattr(est, "sparse_output"):
                    try:
                        est.sparse_output = getattr(est, "sparse", True)
                        repairs.append("OneHotEncoder.sparse_output")
                    except Exception:
                        pass
                for attr, val in [("_infrequent_enabled", False), ("feature_name_combiner", "concat")]:
                    if not hasattr(est, attr):
                        try:
                            setattr(est, attr, val)
                            repairs.append(f"OneHotEncoder.{attr}")
                        except Exception:
                            pass
    bundle["__compat_repairs"] = list(dict.fromkeys(bundle.get("__compat_repairs", []) + repairs))
    return bundle


def _load_joblib_with_split_support(path: str | Path):
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    parts = sorted(p.parent.glob(p.name + ".part*"))
    if parts:
        data = b"".join(part.read_bytes() for part in parts)
        return joblib.load(io.BytesIO(data))
    raise FileNotFoundError(f"Could not find model file {p} or split parts like {p.name}.part01")


def _load_bundle(path: str | Path) -> dict:
    return repair_sklearn_pickle_compat(_load_joblib_with_split_support(path))


def load_two_models(folder_or_zip: str | Path | None = None) -> Dict[str, dict]:
    """Load the two packaged section models from the app folder or an artifact zip."""
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
    paths = {
        "Base Allocation": Path(root) / "base_allocation_model.joblib",
        "Base Review": Path(root) / "base_review_model.joblib",
    }
    models = {}
    for label, path in paths.items():
        if path.exists() or sorted(path.parent.glob(path.name + ".part*")):
            bundle = _load_bundle(path)
            bundle["__model_file"] = str(path)
            models[label] = bundle
    missing = [k for k in paths if k not in models]
    if missing:
        raise FileNotFoundError(f"Missing required model(s): {missing}. Expected Base Allocation and Base Review model files in {root}.")
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


def _predict_model(df: pd.DataFrame, bundle: dict, chunk_size: int = 2500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(df) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    X_all = build_feature_frame(df)
    cols = list(bundle["feature_columns"])
    for c in cols:
        if c not in X_all.columns:
            X_all[c] = 0.0 if c.startswith("num__") else ""
    all_units, all_prob, all_priority = [], [], []
    chunk_size = max(250, int(chunk_size or 2500))
    for start in range(0, len(X_all), chunk_size):
        end = min(len(X_all), start + chunk_size)
        X = X_all.iloc[start:end][cols].replace([np.inf, -np.inf], np.nan)
        Xt = bundle["preprocessor"].transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        Xt = np.asarray(Xt, dtype=np.float32)
        units = np.asarray(bundle["unit_model"].predict(Xt), dtype=int)
        prob_model = bundle["prob_model"]
        if hasattr(prob_model, "predict_proba"):
            classes = list(prob_model.classes_)
            proba = prob_model.predict_proba(Xt)
            prob = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(Xt))
        else:
            prob = np.asarray(prob_model.predict(Xt), dtype=float)
        priority = np.asarray(bundle["priority_model"].predict(Xt), dtype=float).clip(0, 1)
        all_units.append(units); all_prob.append(prob); all_priority.append(priority)
    return np.concatenate(all_units), np.concatenate(all_prob), np.concatenate(all_priority)


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
            units, prob, priority = _predict_model(raw.loc[idxs], models[label], cfg.prediction_chunk_size)
            predictions[label] = pd.DataFrame({"units": units, "prob": prob, "priority": priority}, index=idxs)
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
    work = sorted(work, key=lambda x: (x[0], -x[2], int(raw.loc[x[1], "__row_order"]) if "__row_order" in raw.columns else 0))

    for pass_order, idx, section_score, model_label in work:
        it = item.loc[idx]
        f = max(int(flm.loc[idx]), 1)
        pred = predictions[model_label].loc[idx]
        units = max(int(pred["units"]), 0)
        prob = float(pred["prob"])
        priority = float(pred["priority"])
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

    # Append ignored rows to audit with no prediction for visibility.
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


def model_feature_importance(bundle: dict, model_label: str, top_n: int = 80) -> pd.DataFrame:
    try:
        names = list(bundle["preprocessor"].get_feature_names_out())
    except Exception:
        names = []
    rows = []
    for head_name, key in [("unit", "unit_model"), ("probability", "prob_model"), ("priority", "priority_model")]:
        model = bundle.get(key)
        coefs = getattr(model, "coefs_", None)
        if not coefs:
            continue
        w = np.asarray(coefs[0])
        imp = np.mean(np.abs(w), axis=1)
        local_names = names if len(names) == len(imp) else [f"transformed_feature_{i}" for i in range(len(imp))]
        for fname, val in zip(local_names, imp):
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
                v = float(np.corrcoef(a, b)[0, 1])
                return 0.0 if not np.isfinite(v) else v
            except Exception:
                return 0.0
        cp = corr(sf.values, target_prob.values[:len(sf)])
        ca = corr(sf.values, target_alloc.values[:len(sf)])
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
