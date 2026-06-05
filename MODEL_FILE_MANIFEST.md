# Model File Manifest

Upload every file in this folder to your Streamlit/GitHub repo root. Do not rename them unless you also update `MODEL_PROFILES` in `app.py`.

## Python / runtime files

- `app.py`
- `two_model_prediction_system.py`
- `features_two.py`
- `schema.py`
- `data_io.py`
- `requirements.txt`
- `runtime.txt`
- `README.md`
- `SKLEARN_FREE_MODEL_NOTE.md`
- `MODEL_FILE_MANIFEST.md`

## Base Model v1 files

- `base_model_v1_allocate_model.joblib`
- `base_model_v1_review_model.joblib`
- `base_model_v1_allocate_model_metadata.json`
- `base_model_v1_review_model_metadata.json`
- `base_model_v1_allocation_review_summary.json`
- `base_model_v1_allocate_threshold_sweep.csv`
- `base_model_v1_review_threshold_sweep.csv`
- `base_model_v1_allocate_training_progress.csv`
- `base_model_v1_review_training_progress.csv`
- `base_model_v1_allocate_validation_predictions.csv`
- `base_model_v1_review_validation_predictions.csv`
- `base_model_v1_allocate_workbook_backtest.json`
- `base_model_v1_review_workbook_backtest.json`

## Base Model v2 files

- `base_model_v2_allocate_model.joblib`
- `base_model_v2_review_model.joblib`
- `base_model_v2_allocate_model_metadata.json`
- `base_model_v2_review_model_metadata.json`
- `base_model_v2_allocation_review_summary.json`
- `base_model_v2_allocate_threshold_sweep.csv`
- `base_model_v2_review_threshold_sweep.csv`
- `base_model_v2_allocate_training_progress.csv`
- `base_model_v2_review_training_progress.csv`
- `base_model_v2_allocate_validation_predictions.csv`
- `base_model_v2_review_validation_predictions.csv`
- `base_model_v2_allocate_workbook_backtest.json`
- `base_model_v2_review_workbook_backtest.json`

## Naming rule used

The app uses this non-duplicated convention:

```text
base_model_v<version>_<section>_<artifact>.<ext>
```

Examples:

```text
base_model_v1_allocate_model.joblib
base_model_v1_review_model.joblib
base_model_v2_allocate_model.joblib
base_model_v2_review_model.joblib
```
