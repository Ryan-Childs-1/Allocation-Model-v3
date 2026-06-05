# Allocation AI Two-Model Streamlit App

This is a Streamlit prediction app for allocation workbooks. It uses two built-in NumPy-only model bundles:

- **Base Allocation** for rows flagged Allocate
- **Base Review** for rows flagged Review, including priority/ranking logic

The app intentionally does **not** require scikit-learn, TensorFlow, or Keras at runtime.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Upload all files in this flat folder to Streamlit Cloud / GitHub. Do not remove the `.joblib`, `.json`, or `.csv` model artifacts.

## Output

The app returns:

- completed allocation CSV
- allocation audit CSV
- prediction summary JSON
- feature importance CSV
- feature relationship CSV
