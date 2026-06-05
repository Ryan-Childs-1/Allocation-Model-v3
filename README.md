# Allocation AI Two-Model Streamlit App

This Streamlit app uses two packaged MLP models:

- **Base Allocation** for rows flagged Allocate
- **Base Review** for rows flagged Review

Rows not flagged Allocate or Review are intentionally ignored and left blank.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Output

The app returns:

- `completed_allocation.csv`
- `allocation_audit.csv`
- `prediction_summary.json`
- `model_feature_importance.csv`
- `prediction_feature_relationships.csv`

## Files

The package is flat and includes the two trained joblib models plus their metadata, threshold sweeps, training progress, and validation prediction files.
