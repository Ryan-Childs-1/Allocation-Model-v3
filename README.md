# Allocation AI Two-Model Streamlit App

This Streamlit app runs the latest Base Allocation and Base Review NumPy-only models exported from the Keras trainer with negative-row mixing and expanded demand features.

## Runtime dependencies

The app intentionally does **not** require scikit-learn, TensorFlow, Keras, or SciPy.

## Included models

- `base_allocation_numpy_model.joblib` — Base Allocation model for rows flagged Allocate.
- `base_review_numpy_model.joblib` — Base Review model for rows flagged Review.

Both models are NumPy-only bundles and can run on Streamlit Cloud without ML framework installs.

## Outputs

- completed allocation CSV
- allocation audit CSV
- prediction summary JSON
- feature importance CSV
- feature relationship CSV

## Notes

The app includes the newer demand-focused feature engineering around L30, D30, D60, LW, TTM, projected demand, demand acceleration, demand consistency, and demand-to-supply coverage.
