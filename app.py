from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data_io import dataframe_to_csv_bytes, read_allocation_file, save_upload
from schema import ColumnDiagnostics, build_column_map
from two_model_prediction_system import (
    TwoPredictionConfig,
    load_two_models_from_paths,
    model_feature_importance,
    predict_allocation_file,
    prediction_feature_relationships,
    read_json,
)

st.set_page_config(
    page_title="Allocation AI · Base Model v1/v2",
    page_icon="🎯",
    layout="wide",
)

APP_TITLE = "🎯 Allocation AI · Base Model v1 / v2 Predictor"
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")

MODEL_PROFILES = {
    "base_model_v1": {
        "label": "Base Model v1 · Existing uploaded model",
        "short_label": "Base Model v1",
        "description": "Existing NumPy-only two-model package using Base Allocation + Base Review v7 artifacts.",
        "allocate_model": "base_model_v1_allocate_model.joblib",
        "review_model": "base_model_v1_review_model.joblib",
        "allocate_metadata": "base_model_v1_allocate_model_metadata.json",
        "review_metadata": "base_model_v1_review_model_metadata.json",
        "summary": "base_model_v1_allocation_review_summary.json",
        "allocate_sweep": "base_model_v1_allocate_threshold_sweep.csv",
        "review_sweep": "base_model_v1_review_threshold_sweep.csv",
        "allocate_progress": "base_model_v1_allocate_training_progress.csv",
        "review_progress": "base_model_v1_review_training_progress.csv",
        "allocate_validation": "base_model_v1_allocate_validation_predictions.csv",
        "review_validation": "base_model_v1_review_validation_predictions.csv",
        "allocate_backtest": "base_model_v1_allocate_workbook_backtest.json",
        "review_backtest": "base_model_v1_review_workbook_backtest.json",
    },
    "base_model_v2": {
        "label": "Base Model v2 · New v8 recall/scarcity/memory model",
        "short_label": "Base Model v2",
        "description": "Newer v8 NumPy-only model with counterfactual augmentation, store behavior memory, DC scarcity, reason codes, and Review recall recovery.",
        "allocate_model": "base_model_v2_allocate_model.joblib",
        "review_model": "base_model_v2_review_model.joblib",
        "allocate_metadata": "base_model_v2_allocate_model_metadata.json",
        "review_metadata": "base_model_v2_review_model_metadata.json",
        "summary": "base_model_v2_allocation_review_summary.json",
        "allocate_sweep": "base_model_v2_allocate_threshold_sweep.csv",
        "review_sweep": "base_model_v2_review_threshold_sweep.csv",
        "allocate_progress": "base_model_v2_allocate_training_progress.csv",
        "review_progress": "base_model_v2_review_training_progress.csv",
        "allocate_validation": "base_model_v2_allocate_validation_predictions.csv",
        "review_validation": "base_model_v2_review_validation_predictions.csv",
        "allocate_backtest": "base_model_v2_allocate_workbook_backtest.json",
        "review_backtest": "base_model_v2_review_workbook_backtest.json",
    },
}


def _profile_path(profile_key: str, key: str) -> Path:
    return BASE_DIR / MODEL_PROFILES[profile_key][key]


@st.cache_resource(show_spinner=False)
def cached_models(profile_key: str):
    p = MODEL_PROFILES[profile_key]
    return load_two_models_from_paths(BASE_DIR / p["allocate_model"], BASE_DIR / p["review_model"])


@st.cache_data(show_spinner=False)
def load_profile_metadata(profile_key: str):
    return {
        "Base Allocation": read_json(_profile_path(profile_key, "allocate_metadata")),
        "Base Review": read_json(_profile_path(profile_key, "review_metadata")),
    }, read_json(_profile_path(profile_key, "summary"))


@st.cache_data(show_spinner=False)
def load_all_profile_metadata():
    out = {}
    for key in MODEL_PROFILES:
        out[key] = load_profile_metadata(key)
    return out


def _fmt(x, digits=3, default="—"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default


def _metric_dict(meta: dict) -> dict:
    best = meta.get("best_validation_metrics", {}) if isinstance(meta, dict) else {}
    priority = meta.get("priority_validation_summary", {}) if isinstance(meta, dict) else {}
    backtest = meta.get("workbook_like_backtest", {}) if isinstance(meta, dict) else {}
    return {
        "rows_total": meta.get("rows_total"),
        "rows_train": meta.get("rows_train"),
        "rows_validation": meta.get("rows_validation"),
        "positive_rows_total": meta.get("positive_rows_total"),
        "negative_rows_total": meta.get("negative_rows_total"),
        "best_epoch": meta.get("best_epoch"),
        "best_threshold": meta.get("best_threshold", best.get("threshold")),
        "f1": best.get("f1"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "unit_accuracy": best.get("unit_accuracy"),
        "positive_unit_accuracy": best.get("positive_unit_accuracy"),
        "unit_mae": best.get("unit_mae"),
        "false_positive_rate": best.get("false_positive_rate"),
        "predicted_positive_rows": best.get("predicted_positive_rows"),
        "actual_positive_rows": best.get("actual_positive_rows"),
        "row_exact_match": backtest.get("row_exact_match"),
        "row_near_match": backtest.get("row_near_match"),
        "total_units_error": backtest.get("total_units_error"),
        "priority_mean_positive": priority.get("priority_mean_positive"),
        "priority_mean_negative": priority.get("priority_mean_negative"),
        "model_file_mb": meta.get("model_file_mb"),
        "quantity_correction_enabled": meta.get("quantity_correction_enabled"),
        "override_detector_enabled": meta.get("override_detector_enabled"),
        "review_recall_recovery_enabled": meta.get("review_recall_recovery_enabled"),
        "dc_scarcity_model_enabled": meta.get("dc_scarcity_model_enabled"),
        "reason_code_model_enabled": meta.get("reason_code_model_enabled"),
        "store_behavior_memory_enabled": meta.get("store_behavior_memory_enabled"),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _safe_model_feature_importance(models: dict) -> pd.DataFrame:
    rows = []
    for label in ["Base Allocation", "Base Review"]:
        try:
            rows.append(model_feature_importance(models[label], label, top_n=120))
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _feature_family_summary(fi: pd.DataFrame) -> pd.DataFrame:
    if fi.empty:
        return pd.DataFrame()
    return fi.groupby(["model", "feature_family"], as_index=False)["importance"].sum().sort_values("importance", ascending=False)


all_meta = load_all_profile_metadata()

with st.sidebar:
    st.header("Model version")
    profile_key = st.selectbox(
        "Choose base model",
        list(MODEL_PROFILES.keys()),
        index=1,
        format_func=lambda k: MODEL_PROFILES[k]["label"],
    )
    profile = MODEL_PROFILES[profile_key]
    metadata, summary_meta = all_meta[profile_key]
    st.success(f"Selected: {profile['short_label']}")
    st.caption(profile["description"])

    with st.expander("Model files used", expanded=False):
        st.code(
            "\n".join([
                profile["allocate_model"],
                profile["review_model"],
                profile["allocate_metadata"],
                profile["review_metadata"],
                profile["summary"],
            ]),
            language="text",
        )

    st.header("Prediction controls")
    use_model_thresholds = st.checkbox("Use selected model thresholds", value=True)
    alloc_default = float(_metric_dict(metadata.get("Base Allocation", {})).get("best_threshold") or 0.90)
    review_default = float(_metric_dict(metadata.get("Base Review", {})).get("best_threshold") or 0.05)
    allocation_threshold = st.slider("Base Allocation threshold", 0.01, 0.99, min(max(alloc_default, 0.01), 0.99), 0.01, disabled=use_model_thresholds)
    review_threshold = st.slider("Base Review threshold", 0.01, 0.99, min(max(review_default, 0.01), 0.99), 0.01, disabled=use_model_thresholds)

    demand_extra = st.slider("Demand cap extra FLM", 0.0, 8.0, 1.0, 0.25)
    alloc_rec_influence = st.selectbox("Alloc. Rec. influence", ["feature_only", "soft_cap", "balanced", "hard_cap"], index=2)
    prefer_left_dc = st.checkbox("Prefer Left DC over DC Avail", value=True)
    allow_partial = st.checkbox("Allow below-FLM leftover allocation", value=True)

    st.header("Review ranking")
    review_priority_weight = st.slider("Priority model weight", 0.0, 1.0, 0.50, 0.05)
    review_probability_weight = st.slider("Probability weight", 0.0, 1.0, 0.25, 0.05)
    review_need_weight = st.slider("Need weight", 0.0, 1.0, 0.25, 0.05)
    partial_single = st.checkbox("Give below-FLM Review leftover to one top row", value=True)
    partial_min_priority = st.slider("Minimum Review priority for below-FLM leftover", 0.0, 1.0, 0.50, 0.05)

    st.header("Advanced controls")
    chunk_size = st.select_slider("Prediction chunk size", options=[500, 1000, 2500, 5000, 10000], value=2500)
    use_dc_optimizer = st.checkbox("Use DC allocation optimizer", value=True)
    use_quantity_correction = st.checkbox("Use quantity correction model", value=True)
    use_override_detector = st.checkbox("Use override detector", value=True)

st.title(APP_TITLE)
st.caption("Upload an allocation workbook and select Base Model v1 or Base Model v2. Both model versions are included with unique filenames to avoid collisions.")

tabs = st.tabs([
    "Predict Allocation",
    "Prediction Insights",
    "Model Comparison",
    "Selected Model Metrics",
    "How It Works",
])

with tabs[0]:
    st.markdown("## 1. Upload allocation file")
    uploaded = st.file_uploader("Upload `.xlsb`, `.xlsx`, or `.csv` allocation file", type=["xlsb", "xlsx", "csv"])
    sheet_name = st.text_input("Excel sheet name", value="3.3 Working Table")

    if uploaded is not None:
        try:
            tmp_path = save_upload(uploaded, suffix=Path(uploaded.name).suffix)
            with st.spinner("Reading allocation file and preserving row order..."):
                df = read_allocation_file(tmp_path, sheet_name=sheet_name)
            st.success(f"Loaded `{uploaded.name}` with {len(df):,} rows and {len(df.columns):,} columns.")
            with st.expander("Detected workbook column mapping", expanded=False):
                diag = ColumnDiagnostics(rows=len(df), columns=len(df.columns), header_map=build_column_map(df))
                st.dataframe(pd.DataFrame(diag.as_rows()), use_container_width=True, height=360)

            st.markdown("## 2. Run prediction")
            st.info(f"Prediction will use **{profile['short_label']}**.")
            if st.button("Predict Final Alloc", type="primary"):
                cfg = TwoPredictionConfig(
                    allocation_threshold=None if use_model_thresholds else float(allocation_threshold),
                    review_threshold=None if use_model_thresholds else float(review_threshold),
                    demand_cap_extra_flm=float(demand_extra),
                    alloc_rec_influence=str(alloc_rec_influence),
                    prefer_left_dc=bool(prefer_left_dc),
                    allow_partial_leftover_below_flm=bool(allow_partial),
                    review_priority_weight=float(review_priority_weight),
                    review_probability_weight=float(review_probability_weight),
                    review_need_weight=float(review_need_weight),
                    review_partial_leftover_to_single_row=bool(partial_single),
                    review_partial_leftover_min_priority=float(partial_min_priority),
                    prediction_chunk_size=int(chunk_size),
                    use_dc_optimizer=bool(use_dc_optimizer),
                    use_quantity_correction=bool(use_quantity_correction),
                    use_override_detector=bool(use_override_detector),
                )
                with st.spinner(f"Loading {profile['short_label']}, scoring Allocate/Review sections, and simulating DC availability..."):
                    models = cached_models(profile_key)
                    out_df, audit_df, run_summary = predict_allocation_file(df, models, cfg)
                    run_summary["selected_model_profile"] = profile["short_label"]
                    run_summary["selected_model_key"] = profile_key
                try:
                    fi = _safe_model_feature_importance(models)
                except Exception:
                    fi = pd.DataFrame()
                try:
                    rel = prediction_feature_relationships(df, audit_df, top_n=80)
                except Exception:
                    rel = pd.DataFrame()

                st.session_state["selected_profile_key"] = profile_key
                st.session_state["input_df"] = df
                st.session_state["out_df"] = out_df
                st.session_state["audit_df"] = audit_df
                st.session_state["run_summary"] = run_summary
                st.session_state["feature_importance"] = fi
                st.session_state["feature_relationships"] = rel
                st.success(f"Prediction complete using {profile['short_label']}. Final Alloc values are integer quantities or blank.")
        except Exception as exc:
            st.error("File loading or prediction setup failed.")
            st.exception(exc)

    if "out_df" in st.session_state:
        out_df = st.session_state["out_df"]
        audit_df = st.session_state["audit_df"]
        run_summary = st.session_state["run_summary"]

        st.markdown("## 3. Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{run_summary.get('rows', 0):,}")
        c2.metric("Allocated rows", f"{run_summary.get('allocated_rows', 0):,}")
        c3.metric("Total Final Alloc", f"{run_summary.get('total_final_alloc', 0):,}")
        c4.metric("Ignored rows", f"{run_summary.get('ignored_no_alloc_rows', 0):,}")
        c5.metric("Model used", run_summary.get("selected_model_profile", "—"))

        st.info(
            f"Thresholds used — Base Allocation: `{run_summary.get('allocation_threshold')}` · "
            f"Base Review: `{run_summary.get('review_threshold')}`"
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Completed allocation preview")
            st.dataframe(out_df.head(250), use_container_width=True, height=420)
        with right:
            st.subheader("Audit preview")
            st.dataframe(audit_df.head(250), use_container_width=True, height=420)

        completed_csv = dataframe_to_csv_bytes(out_df)
        audit_csv = dataframe_to_csv_bytes(audit_df)
        summary_bytes = json.dumps(run_summary, indent=2, default=str).encode("utf-8")
        feature_imp_csv = dataframe_to_csv_bytes(st.session_state.get("feature_importance", pd.DataFrame()))
        rel_csv = dataframe_to_csv_bytes(st.session_state.get("feature_relationships", pd.DataFrame()))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("completed_allocation.csv", completed_csv)
            z.writestr("allocation_audit.csv", audit_csv)
            z.writestr("prediction_summary.json", summary_bytes)
            z.writestr("model_feature_importance.csv", feature_imp_csv)
            z.writestr("prediction_feature_relationships.csv", rel_csv)
        zip_bytes = zip_buffer.getvalue()

        st.markdown("## 4. Downloads")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download completed CSV", completed_csv, "completed_allocation.csv", "text/csv")
        d2.download_button("Download audit CSV", audit_csv, "allocation_audit.csv", "text/csv")
        d3.download_button("Download output ZIP", zip_bytes, "allocation_ai_output.zip", "application/zip")
    else:
        st.info("Upload a file and run prediction to generate completed allocation outputs.")

with tabs[1]:
    st.markdown("## Prediction Insights")
    if "audit_df" not in st.session_state:
        st.info("Run a prediction first to populate this page.")
    else:
        audit = st.session_state["audit_df"].copy()
        fi = st.session_state.get("feature_importance", pd.DataFrame()).copy()
        rel = st.session_state.get("feature_relationships", pd.DataFrame()).copy()
        audit["final_alloc_numeric"] = pd.to_numeric(audit.get("final_alloc", 0), errors="coerce").fillna(0)
        audit["allocated"] = audit["final_alloc_numeric"] > 0

        st.markdown("### Allocation by model section")
        section_mix = audit.groupby("model_used", dropna=False).agg(
            rows=("model_used", "size"),
            allocated_rows=("allocated", "sum"),
            total_alloc=("final_alloc_numeric", "sum"),
            avg_probability=("probability", "mean"),
            avg_priority=("priority", "mean"),
        ).reset_index().sort_values("total_alloc", ascending=False)
        st.dataframe(section_mix, use_container_width=True, height=240)
        if not section_mix.empty:
            st.bar_chart(section_mix.set_index("model_used")[["rows", "allocated_rows"]])
            st.bar_chart(section_mix.set_index("model_used")["total_alloc"])

        st.markdown("### Top allocated items")
        item_mix = audit.groupby("item", dropna=False).agg(
            rows=("item", "size"),
            allocated_rows=("allocated", "sum"),
            total_alloc=("final_alloc_numeric", "sum"),
            avg_section_score=("section_score", "mean"),
            avg_probability=("probability", "mean"),
        ).reset_index().sort_values("total_alloc", ascending=False).head(30)
        st.dataframe(item_mix, use_container_width=True, height=350)
        if not item_mix.empty:
            st.bar_chart(item_mix.set_index("item")["total_alloc"])

        st.markdown("### Decision reasons")
        reasons = audit.get("reason", pd.Series("", index=audit.index)).astype(str).str.split("; ").explode().replace("", np.nan).dropna().value_counts().head(25).rename_axis("reason").reset_index(name="rows")
        st.dataframe(reasons, use_container_width=True, height=300)
        if not reasons.empty:
            st.bar_chart(reasons.set_index("reason")["rows"])

        st.markdown("### Model feature usage")
        if fi.empty:
            st.info("No feature importance information was available.")
        else:
            family = _feature_family_summary(fi)
            st.dataframe(family, use_container_width=True, height=330)
            pivot = family.pivot_table(index="feature_family", columns="model", values="importance", aggfunc="sum", fill_value=0)
            st.bar_chart(pivot)
            st.dataframe(fi.head(60), use_container_width=True, height=420)

        st.markdown("### Run-specific feature relationships")
        if rel.empty:
            st.info("No run-specific feature relationship table was available.")
        else:
            st.dataframe(rel.head(60), use_container_width=True, height=420)
            st.bar_chart(rel.head(25).set_index("feature")["relationship_strength"])

with tabs[2]:
    st.markdown("## Base Model v1 vs Base Model v2")
    rows = []
    for key, prof in MODEL_PROFILES.items():
        meta_pair, summary = all_meta[key]
        for model_label in ["Base Allocation", "Base Review"]:
            m = _metric_dict(meta_pair.get(model_label, {}))
            rows.append({"Version": prof["short_label"], "Model": model_label, **m})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    if not comp.empty:
        chart_cols = [c for c in ["f1", "precision", "recall", "unit_mae", "row_near_match", "total_units_error"] if c in comp.columns]
        for col in chart_cols:
            st.markdown(f"### {col}")
            st.bar_chart(comp.pivot_table(index="Version", columns="Model", values=col, aggfunc="first"))

    st.markdown("### File status")
    status_rows = []
    for key, prof in MODEL_PROFILES.items():
        for artifact_key in [
            "allocate_model", "review_model", "allocate_metadata", "review_metadata", "summary",
            "allocate_sweep", "review_sweep", "allocate_progress", "review_progress",
            "allocate_validation", "review_validation", "allocate_backtest", "review_backtest",
        ]:
            path = BASE_DIR / prof[artifact_key]
            status_rows.append({"Version": prof["short_label"], "Artifact": artifact_key, "Filename": prof[artifact_key], "Found": path.exists(), "Size MB": round(path.stat().st_size / (1024*1024), 3) if path.exists() else None})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown(f"## Selected Model Metrics: {profile['short_label']}")
    summary_rows = []
    for label in ["Base Allocation", "Base Review"]:
        meta = metadata.get(label, {})
        summary_rows.append({"Model": label, **_metric_dict(meta)})
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for label, prefix in [("Base Allocation", "allocate"), ("Base Review", "review")]:
        meta = metadata.get(label, {})
        m = _metric_dict(meta)
        st.divider()
        st.markdown(f"### {label}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows train/valid", f"{int(m.get('rows_train') or 0):,} / {int(m.get('rows_validation') or 0):,}")
        c2.metric("Best epoch", f"{int(m.get('best_epoch') or 0):,}")
        c3.metric("Threshold", _fmt(m.get("best_threshold"), 2))
        c4.metric("F1", _fmt(m.get("f1")))
        c5.metric("Unit MAE", _fmt(m.get("unit_mae"), 4))

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Precision", _fmt(m.get("precision")))
        x2.metric("Recall", _fmt(m.get("recall")))
        x3.metric("Near match", _fmt(m.get("row_near_match")))
        x4.metric("Total units error", f"{int(m.get('total_units_error') or 0):,}")

        flags = pd.DataFrame([
            {"Feature": "Quantity correction", "Enabled": bool(m.get("quantity_correction_enabled"))},
            {"Feature": "Override detector", "Enabled": bool(m.get("override_detector_enabled"))},
            {"Feature": "Review recall recovery", "Enabled": bool(m.get("review_recall_recovery_enabled"))},
            {"Feature": "DC scarcity model", "Enabled": bool(m.get("dc_scarcity_model_enabled"))},
            {"Feature": "Reason-code model", "Enabled": bool(m.get("reason_code_model_enabled"))},
            {"Feature": "Store behavior memory", "Enabled": bool(m.get("store_behavior_memory_enabled"))},
        ])
        st.dataframe(flags, use_container_width=True, hide_index=True)

        backtest = read_json(_profile_path(profile_key, f"{prefix}_backtest"))
        if backtest:
            st.markdown("#### Workbook-like backtest")
            st.json(backtest)

        sweep = _read_csv(_profile_path(profile_key, f"{prefix}_sweep"))
        if not sweep.empty:
            st.markdown("#### Threshold sweep")
            st.dataframe(sweep, use_container_width=True, height=260)
            cols = [c for c in ["f1", "precision", "recall"] if c in sweep.columns]
            if "threshold" in sweep.columns and cols:
                st.line_chart(sweep.set_index("threshold")[cols])

        progress = _read_csv(_profile_path(profile_key, f"{prefix}_progress"))
        if not progress.empty:
            st.markdown("#### Training progress")
            st.dataframe(progress.tail(25), use_container_width=True, height=300)
            cols = [c for c in ["val_f1", "val_precision", "val_recall", "val_unit_mae"] if c in progress.columns]
            if "epoch" in progress.columns and cols:
                st.line_chart(progress.set_index("epoch")[cols])

        with st.expander(f"Full {label} metadata", expanded=False):
            st.json(meta)

    if summary_meta:
        st.divider()
        st.markdown("### Selected two-model system summary")
        st.json({k: v for k, v in summary_meta.items() if k != "models"})

with tabs[4]:
    st.markdown("## How the multi-model app works")
    st.markdown(
        """
        This app includes two complete Base Allocation + Base Review model pairs.

        1. Select **Base Model v1** or **Base Model v2** in the sidebar.
        2. Upload an allocation workbook as `.xlsb`, `.xlsx`, or `.csv`.
        3. Rows marked **Allocate** are scored by the selected version's Allocation model.
        4. Rows marked **Review** are scored by the selected version's Review model.
        5. Rows that are not Allocate or Review are ignored and left blank.
        6. The prediction system simulates remaining DC by item and writes integer Final Alloc values or blanks.
        7. Output downloads include the completed CSV, audit CSV, summary JSON, feature importance CSV, and feature relationship CSV.

        **Base Model v1** is the existing uploaded model package.  
        **Base Model v2** is the new v8 package with newer auxiliary heads and memory features when available.
        """
    )

    st.markdown("### Exact non-duplicated model naming convention")
    names = []
    for key, prof in MODEL_PROFILES.items():
        for artifact_key in ["allocate_model", "review_model", "allocate_metadata", "review_metadata", "summary"]:
            names.append({"Version": prof["short_label"], "Artifact": artifact_key, "Filename": prof[artifact_key]})
    st.dataframe(pd.DataFrame(names), use_container_width=True, hide_index=True)
