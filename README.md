# Allocation AI — Base Model v1/v2 Streamlit App

Completed Streamlit app for selecting and running **Base Model v1** or **Base Model v2** on Sportsman's Warehouse allocation files.

## What this app does

- Upload `.csv`, `.xlsx`, `.xlsm`, or `.xlsb`
- Select **Base Model v1** or **Base Model v2**
- Fill **Final Alloc.** only
- Preserve blank allocations for non-candidate rows
- Include `Allocate`, `Review`, and `Z - No Alloc` rows as model candidates
- Run three allocation passes
- Round to FLM/pack size
- Allow allocation of remaining DC units when `Left DC < FLM`
- Download output as XLSX with audit tabs or CSV

## Files

- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit Cloud configuration

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Upload this flat project folder to GitHub, then deploy `app.py`.

## Model definitions

### Base Model v1
Conservative baseline. Better when the priority is avoiding over-allocation.

### Base Model v2
Improved baseline. Uses a fuller demand blend and stronger BM/BN signals, while still protecting against demand-overstock issues.

## Expected column logic

The app detects columns by name and by fallback Excel positions:

- `BR` = Flag
- `BS` = Final Alloc.
- `BN` = Alloc. Rec.
- `BM` = Proj. Demand
- `BT` = Left DC
- `BU` = Final Supply
- `BK` = FLM
- `AP` = QOH
- `AQ` = Supply
- `AK/AL/AM/AN/AO` = L30/D30/D60/LW/TTM
- `CA` = Demand Discount
- `CB` = New Base Demand
- `O` = group/item key fallback

## Important note

This app is self-contained and does not require TensorFlow, Keras, PyTorch, or external model artifacts. That avoids Streamlit Cloud Python-version dependency failures while preserving the Base Model v1/v2 selection workflow.
