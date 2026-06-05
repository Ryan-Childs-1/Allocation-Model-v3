from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from schema import detect_header_row, unique_columns


def _read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # Read raw without header first to detect true header row.
    raw = pd.read_csv(path, header=None, dtype=object, encoding_errors="replace", low_memory=False)
    h = detect_header_row(raw)
    headers = unique_columns(raw.iloc[h].fillna("").astype(str).tolist())
    df = raw.iloc[h + 1 :].copy().reset_index(drop=True)
    df.columns = headers[: len(df.columns)]
    df = df.dropna(how="all").reset_index(drop=True)
    df["__row_order"] = range(len(df))
    df["__excel_row"] = h + 2 + df["__row_order"]
    return df


def _read_xlsx(path: str | Path, sheet_name: str = "3.3 Working Table") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object, engine="openpyxl")
    h = detect_header_row(raw)
    headers = unique_columns(raw.iloc[h].fillna("").astype(str).tolist())
    df = raw.iloc[h + 1 :].copy().reset_index(drop=True)
    df.columns = headers[: len(df.columns)]
    df = df.dropna(how="all").reset_index(drop=True)
    df["__row_order"] = range(len(df))
    df["__excel_row"] = h + 2 + df["__row_order"]
    return df


def _read_xlsb(path: str | Path, sheet_name: str = "3.3 Working Table") -> pd.DataFrame:
    try:
        from pyxlsb import open_workbook
    except Exception as exc:
        raise ImportError("pyxlsb is required to read .xlsb files. Install with: pip install pyxlsb") from exc

    # pyxlsb can iterate over a huge formatted range even after real data ends.
    # Stop after a long streak of truly blank rows once meaningful rows have appeared.
    rows = []
    max_col = 0
    blank_streak = 0
    seen_nonblank = False
    trailing_blank_limit = 2000
    with open_workbook(str(path)) as wb:
        sheet_names = list(wb.sheets)
        sname = sheet_name if sheet_name in sheet_names else sheet_names[0]
        with wb.get_sheet(sname) as sh:
            for row in sh.rows():
                vals = {}
                row_nonblank = False
                for cell in row:
                    c = int(getattr(cell, "c", 0))
                    v = cell.v
                    vals[c] = v
                    if c > max_col:
                        max_col = c
                    if v is not None and str(v).strip() != "":
                        row_nonblank = True
                if row_nonblank:
                    seen_nonblank = True
                    blank_streak = 0
                elif seen_nonblank:
                    blank_streak += 1
                rows.append([vals.get(i, None) for i in range(max_col + 1)])
                if seen_nonblank and blank_streak >= trailing_blank_limit:
                    break

    # Trim trailing blank rows collected before the stop condition.
    while rows and not any(v is not None and str(v).strip() != "" for v in rows[-1]):
        rows.pop()
    raw = pd.DataFrame(rows)
    h = detect_header_row(raw)
    headers = unique_columns(raw.iloc[h].fillna("").astype(str).tolist())
    df = raw.iloc[h + 1 :].copy().reset_index(drop=True)
    df.columns = headers[: len(df.columns)]
    df = df.dropna(how="all").reset_index(drop=True)
    df["__row_order"] = range(len(df))
    df["__excel_row"] = h + 2 + df["__row_order"]
    return df


def read_allocation_file(path: str | Path, sheet_name: str = "3.3 Working Table") -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        return _read_csv_flexible(path)
    if suf == ".xlsx":
        return _read_xlsx(path, sheet_name=sheet_name)
    if suf == ".xlsb":
        return _read_xlsb(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_upload(uploaded_file: Any, suffix: str = "") -> Path:
    suffix = suffix or Path(getattr(uploaded_file, "name", "upload")).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
