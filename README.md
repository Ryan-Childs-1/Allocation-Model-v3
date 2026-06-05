# Allocation AI Two-Model Predictor — sklearn-free runtime

This Streamlit app runs the Base Allocation and Base Review models without installing scikit-learn. The trained sklearn MLPs were converted into NumPy-only bundles containing preprocessing parameters and neural-network weights.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Included runtime dependencies

- streamlit
- pandas
- numpy
- joblib
- pyxlsb
- openpyxl

No `scikit-learn`, TensorFlow, Torch, or JAX is required at prediction time.
