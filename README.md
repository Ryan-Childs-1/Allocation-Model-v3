# Allocation AI Two-Model Predictor — NumPy-only runtime

This Streamlit app uses two converted NumPy-only MLP bundles:

- `base_allocation_numpy_model.joblib`
- `base_review_numpy_model.joblib`

It does **not** require scikit-learn or scipy at runtime. The app loads Excel/CSV allocation files, scores Allocate and Review rows with separate models, simulates remaining DC by item, and outputs completed allocation CSV/audit files.

## Streamlit entry point

`app.py`

## Requirements

```text
streamlit
pandas
numpy
joblib
pyxlsb
openpyxl
```

## Notes

- Rows not flagged Allocate or Review are ignored and left blank.
- Review rows are ranked using priority/probability/need settings.
- Review rows may receive below-FLM leftover DC when enabled.
- The sidebar includes explanations for every setting.
