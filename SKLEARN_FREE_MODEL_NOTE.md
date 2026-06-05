# Runtime Model Format

The Streamlit app uses NumPy-only model bundles. The models were trained outside the app and exported to lightweight joblib dictionaries containing preprocessing values and MLP weights.

The hosted Streamlit app does not install or import:

- scikit-learn
- TensorFlow
- Keras
- scipy

This avoids wheel build failures on Streamlit Cloud and keeps deployment simpler.
