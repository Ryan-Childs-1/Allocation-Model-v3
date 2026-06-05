# sklearn-free runtime note

The original Base Allocation and Base Review sklearn MLP models were converted into NumPy-only model bundles. The bundles contain:

- numeric imputer/scaler parameters
- one-hot category mappings, including infrequent-category grouping
- MLP weights and biases
- model metadata and thresholds

This allows Streamlit prediction without installing `scikit-learn`. Training still requires a training environment, but deployment/prediction does not.
