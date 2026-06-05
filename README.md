# Allocation AI Streamlit App · Base Model v1 + Base Model v2

This package is a flat Streamlit deployment folder that includes two selectable model versions:

- **Base Model v1**: the existing uploaded NumPy-only Base Allocation + Base Review model pair.
- **Base Model v2**: the new uploaded v8 Base Allocation + Base Review model pair with counterfactual augmentation, store behavior memory, DC scarcity, reason-code heads, and Review recall recovery where present.

The app intentionally does **not** require scikit-learn, TensorFlow, Keras, or SciPy at runtime. It uses NumPy-only joblib model bundles.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud entrypoint

Use:

```text
app.py
```

## Model selection

Use the sidebar dropdown named **Choose base model** to select:

- `Base Model v1 · Existing uploaded model`
- `Base Model v2 · New v8 recall/scarcity/memory model`

## File naming

All model artifacts have unique prefixes so v1 and v2 do not overwrite each other:

- `base_model_v1_*`
- `base_model_v2_*`

See `MODEL_FILE_MANIFEST.md` for the complete upload list.

## Outputs

After prediction, the app provides:

- `completed_allocation.csv`
- `allocation_audit.csv`
- `prediction_summary.json`
- `model_feature_importance.csv`
- `prediction_feature_relationships.csv`
- a combined output ZIP
