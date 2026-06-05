from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from data_io import dataframe_to_csv_bytes, read_allocation_file, save_upload
from schema import ColumnDiagnostics, build_column_map
from two_model_prediction_system import (
    TwoPredictionConfig,
    load_two_models,
    model_feature_importance,
    predict_allocation_file,
    prediction_feature_relationships,
    read_json,
)

st.set_page_config(page_title="Allocation AI · Base Allocation + Base Review · sklearn-free", page_icon="🎯", layout="wide")

APP_TITLE = "🎯 Allocation AI · Two-Model Predictor"
ARTIFACT_NAME = "Base Allocation + Base Review"
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")

MODEL_FILES = {
    "Base Allocation": BASE_DIR / "base_allocation_model.joblib",
    "Base Review": BASE_DIR / "base_review_model.joblib",
}
METADATA_FILES = {
    "Base Allocation": BASE_DIR / "base_allocation_model_metadata.json",
    "Base Review": BASE_DIR / "base_review_model_metadata.json",
}
SWEEP_FILES = {
    "Base Allocation": BASE_DIR / "base_allocation_threshold_sweep.csv",
    "Base Review": BASE_DIR / "base_review_threshold_sweep.csv",
}
PROGRESS_FILES = {
    "Base Allocation": BASE_DIR / "base_allocation_training_progress.csv",
    "Base Review": BASE_DIR / "base_review_training_progress.csv",
}
VALIDATION_FILES = {
    "Base Allocation": BASE_DIR / "base_allocation_validation_predictions.csv",
    "Base Review": BASE_DIR / "base_review_validation_predictions.csv",
}
SUMMARY_FILE = BASE_DIR / "base_allocation_base_review_summary.json"


@st.cache_resource(show_spinner=False)
def cached_models():
    return load_two_models(BASE_DIR)


@st.cache_data(show_spinner=False)
def load_metadata():
    return {label: read_json(path) for label, path in METADATA_FILES.items()}, read_json(SUMMARY_FILE)


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
    return {
        "rows_total": meta.get("rows_total"),
        "rows_train": meta.get("rows_train"),
        "rows_validation": meta.get("rows_validation"),
        "positive_rows_total": meta.get("positive_rows_total"),
        "best_epoch": meta.get("best_epoch"),
        "best_threshold": meta.get("best_threshold", best.get("threshold")),
        "f1": best.get("f1"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "unit_accuracy": best.get("unit_accuracy"),
        "positive_unit_accuracy": best.get("positive_unit_accuracy"),
        "unit_mae": best.get("unit_mae"),
        "false_positive_rate": best.get("false_positive_rate"),
        "priority_spread": priority.get("priority_spread_pos_minus_neg"),
        "model_file_mb": meta.get("model_file_mb"),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _feature_family_summary(fi: pd.DataFrame) -> pd.DataFrame:
    if fi.empty:
        return pd.DataFrame()
    return fi.groupby(["model", "feature_family"], as_index=False)["importance"].sum().sort_values("importance", ascending=False)


def _safe_model_feature_importance(models: dict) -> pd.DataFrame:
    rows = []
    for label in ["Base Allocation", "Base Review"]:
        try:
            rows.append(model_feature_importance(models[label], label, top_n=120))
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


metadata, summary_meta = load_metadata()

with st.sidebar:
    st.header("Model system")
    st.success("Built-in: Base Allocation + Base Review")
    st.caption("Allocate rows and Review rows are handled by separate packaged MLP models. No Alloc / Z rows are ignored and left blank.")

    st.header("Prediction controls")
    use_model_thresholds = st.checkbox("Use trained model thresholds", value=True)
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

    st.header("Performance")
    chunk_size = st.select_slider("Prediction chunk size", options=[500, 1000, 2500, 5000, 10000], value=2500)

st.title(APP_TITLE)
st.caption("Upload an allocation workbook and return the same rows with Final Alloc filled by the two-model section-aware MLP system.")

predict_tab, insights_tab, model_tab, process_tab = st.tabs([
    "Predict Allocation",
    "Prediction Insights",
    "Model Metrics",
    "How It Works + Feature Guide",
])

with predict_tab:
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
                )
                with st.spinner("Loading models, scoring Allocate/Review sections, and simulating DC availability..."):
                    models = cached_models()
                    out_df, audit_df, run_summary = predict_allocation_file(df, models, cfg)
                try:
                    fi = _safe_model_feature_importance(models)
                except Exception:
                    fi = pd.DataFrame()
                try:
                    rel = prediction_feature_relationships(df, audit_df, top_n=80)
                except Exception:
                    rel = pd.DataFrame()

                st.session_state["input_df"] = df
                st.session_state["out_df"] = out_df
                st.session_state["audit_df"] = audit_df
                st.session_state["run_summary"] = run_summary
                st.session_state["feature_importance"] = fi
                st.session_state["feature_relationships"] = rel
                st.success("Prediction complete. Final Alloc values are integer quantities or blank.")
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
        c4.metric("Ignored non-Alloc/Review", f"{run_summary.get('ignored_no_alloc_rows', 0):,}")
        c5.metric("Review partial leftover units", f"{run_summary.get('review_partial_leftover_units', 0):,}")

        st.info(
            f"Thresholds used — Base Allocation: `{run_summary.get('allocation_threshold')}` · "
            f"Base Review: `{run_summary.get('review_threshold')}`"
        )

        section_rows = pd.DataFrame([{"section": k, "rows": v} for k, v in run_summary.get("section_rows", {}).items()])
        if not section_rows.empty:
            st.markdown("### Section rows")
            st.bar_chart(section_rows.set_index("section"))

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
        d3.download_button("Download output ZIP", zip_bytes, "allocation_ai_two_model_output.zip", "application/zip")
    else:
        st.info("Upload a file and run prediction to generate completed allocation outputs.")

with insights_tab:
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
        st.caption("Approximate feature usage from neural-network first-layer weight magnitudes. This is an inspection view, not causal proof.")
        if fi.empty:
            st.info("No feature importance information was available.")
        else:
            family = _feature_family_summary(fi)
            st.subheader("Feature family usage")
            st.dataframe(family, use_container_width=True, height=330)
            if not family.empty:
                pivot = family.pivot_table(index="feature_family", columns="model", values="importance", aggfunc="sum", fill_value=0)
                st.bar_chart(pivot)
            st.subheader("Top base features")
            st.dataframe(fi.head(60), use_container_width=True, height=420)

        st.markdown("### Run-specific feature relationships")
        st.caption("Correlations between engineered numeric features and this run's predicted probability / final allocation.")
        if rel.empty:
            st.info("No run-specific feature relationship table was available.")
        else:
            st.dataframe(rel.head(60), use_container_width=True, height=420)
            st.bar_chart(rel.head(25).set_index("feature")["relationship_strength"])

with model_tab:
    st.markdown("## Model Metrics")
    st.caption("The app uses two separately trained MLP model bundles: one for Allocate rows and one for Review rows.")

    summary_rows = []
    for label in ["Base Allocation", "Base Review"]:
        meta = metadata.get(label, {})
        m = _metric_dict(meta)
        summary_rows.append({"Model": label, **m})
    model_summary = pd.DataFrame(summary_rows)
    st.dataframe(model_summary, use_container_width=True, hide_index=True)

    for label in ["Base Allocation", "Base Review"]:
        meta = metadata.get(label, {})
        m = _metric_dict(meta)
        st.divider()
        st.markdown(f"### {label}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows trained/valid", f"{int(m.get('rows_train') or 0):,} / {int(m.get('rows_validation') or 0):,}")
        c2.metric("Best epoch", f"{int(m.get('best_epoch') or 0):,}")
        c3.metric("Threshold", _fmt(m.get("best_threshold"), 2))
        c4.metric("Model size MB", _fmt(m.get("model_file_mb"), 2))

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("F1", _fmt(m.get("f1")))
        p2.metric("Precision", _fmt(m.get("precision")))
        p3.metric("Recall", _fmt(m.get("recall")))
        p4.metric("Unit accuracy", _fmt(m.get("unit_accuracy")))
        p5.metric("Unit MAE", _fmt(m.get("unit_mae"), 4))

        sweep = _read_csv(SWEEP_FILES[label])
        if not sweep.empty:
            st.markdown("#### Threshold sweep")
            st.dataframe(sweep, use_container_width=True, height=260)
            cols = [c for c in ["f1", "precision", "recall"] if c in sweep.columns]
            if "threshold" in sweep.columns and cols:
                st.line_chart(sweep.set_index("threshold")[cols])

        progress = _read_csv(PROGRESS_FILES[label])
        if not progress.empty:
            st.markdown("#### Training progress")
            st.dataframe(progress.tail(25), use_container_width=True, height=300)
            cols = [c for c in ["val_f1", "val_precision", "val_recall", "val_unit_mae"] if c in progress.columns]
            if "epoch" in progress.columns and cols:
                st.line_chart(progress.set_index("epoch")[cols])

        val = _read_csv(VALIDATION_FILES[label])
        if not val.empty:
            st.markdown("#### Validation prediction sample")
            st.dataframe(val.head(100), use_container_width=True, height=260)

        with st.expander(f"Full {label} metadata", expanded=False):
            st.json(meta)

    if summary_meta:
        st.divider()
        st.markdown("### Full two-model system summary")
        st.json({k: v for k, v in summary_meta.items() if k != "models"})

with process_tab:
    st.markdown("## How the two-model system works")
    st.markdown(
        """
        This Streamlit app is a prediction-only interface for a section-aware allocation model.

        **End-to-end flow**

        1. Upload an allocation workbook as `.xlsb`, `.xlsx`, or `.csv`.
        2. The app detects key workbook columns, including item, site, demand, supply, Final Alloc, Left DC, Alloc. Rec., FLM, flag, Demand Check, and Helper.
        3. Rows marked **Allocate** are scored by the **Base Allocation** model.
        4. Rows marked **Review** are scored by the **Base Review** model.
        5. Rows that are not Allocate or Review are ignored and left blank.
        6. The app simulates remaining DC by item while filling Final Alloc.
        7. Review rows are ranked by a blend of model priority, model probability, and need so the most important review rows consume scarce DC first.
        8. The output is a completed CSV plus an audit CSV and explanation tables.
        """
    )

    st.markdown("### Core model features")
    feature_groups = pd.DataFrame([
        {"Feature group": "Demand / velocity", "Examples": "L30, D30, D60, LW, TTM, projected demand, recent velocity blend"},
        {"Feature group": "Supply / DC", "Examples": "QOH, supply, DC available, Left DC, reconstructed DC before Final Alloc"},
        {"Feature group": "Allocation recommendation", "Examples": "Alloc. Rec., Alloc. Rec. units, Alloc. Rec. to need/DC/projection ratios"},
        {"Feature group": "Need / scarcity", "Examples": "Need gap, demand cap, need-to-DC pressure, DC-to-need ratio"},
        {"Feature group": "Section/group context", "Examples": "Item totals, site totals, department/class totals, item-section totals"},
        {"Feature group": "Ranking", "Examples": "Within-item rank by need, demand, Alloc. Rec., DC-before, and velocity"},
        {"Feature group": "Workbook helper logic", "Examples": "Demand Check, Helper, Final Supply, FLM, partial leftover indicators"},
        {"Feature group": "Categorical identity", "Examples": "Item, UPC, site, description, department, class, region, flag"},
    ])
    st.dataframe(feature_groups, use_container_width=True, hide_index=True)

    st.markdown("### What makes this section-aware")
    st.markdown(
        """
        The model is not only looking row by row. The training process rebuilt each row in context:

        - How much demand exists for the item across all stores.
        - How much need exists within the current section.
        - How scarce DC is for the item.
        - Where the row ranks among competing rows for the same item.
        - How much DC likely existed **before** historical Final Alloc values were entered.

        That section context is especially important for Review rows, where the model should prioritize rows that need product most.
        """
    )

    st.markdown("### Model roles")
    roles = pd.DataFrame([
        {"Model": "Base Allocation", "Rows handled": "Allocate", "Main purpose": "Fill normal allocation rows with integer Final Alloc quantities."},
        {"Model": "Base Review", "Rows handled": "Review", "Main purpose": "Rank and allocate Review rows, especially when DC is scarce."},
        {"Model": "Ignored", "Rows handled": "No Alloc / Z / other", "Main purpose": "Left blank by design in this version."},
    ])
    st.dataframe(roles, use_container_width=True, hide_index=True)
