from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


def normalize_header(x) -> str:
    text = "" if x is None else str(x)
    text = text.replace("\n", " ").replace("\r", " ").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def excel_col_to_idx(col: str) -> int:
    col = col.upper().strip()
    n = 0
    for ch in col:
        if "A" <= ch <= "Z":
            n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def unique_columns(cols: Iterable) -> List[str]:
    out, seen = [], {}
    for c in cols:
        base = str(c).strip() if str(c).strip() else "blank"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]+1}")
    return out


def detect_header_row(df: pd.DataFrame, max_scan: int = 40) -> int:
    # Prefer rows containing a cluster of known allocation headers.
    expected = {"item", "site", "flag", "final_alloc", "alloc_rec", "left_dc", "flm", "d60", "ttm"}
    best_i, best_score = 0, -1
    for i in range(min(max_scan, len(df))):
        vals = [normalize_header(v) for v in df.iloc[i].tolist()]
        joined = set(vals)
        score = 0
        for e in expected:
            if e in joined or any(e in v for v in vals):
                score += 1
        nonblank = sum(bool(v) for v in vals)
        score += min(nonblank, 20) / 100.0
        if score > best_score:
            best_i, best_score = i, score
    return best_i


SYNONYMS: Dict[str, List[str]] = {
    "vendor": ["vendor"],
    "vendor_site_id": ["vendor_site_id", "vendor_site", "vendor_siteid"],
    "brand": ["brand"],
    "department_id": ["department_id", "dept_id", "department"],
    "class_id": ["class_id", "class"],
    "line_id": ["line_id", "line"],
    "product_id": ["product_id", "product"],
    "item": ["item", "sku", "sku_id"],
    "description": ["description", "desc", "pcode_description"],
    "upc": ["upc"],
    "site": ["site", "store", "store_id"],
    "site_name": ["site_name", "store_name"],
    "state": ["state"],
    "region": ["region"],
    "zone": ["zone"],
    "store_size": ["store_size", "store_size_band", "size"],
    "rank": ["rank"],
    "flag": ["flag", "alloc_flag", "allocation_flag"],
    "mil": ["mil", "minimum_inventory_level", "min_inventory_level"],
    "flm": ["flm", "case_pack", "pack", "pack_size"],
    "dc_flm": ["dc_flm", "dc_pack"],
    "cost": ["cost"],
    "retail": ["retail", "atg_retail"],
    "gm_pct": ["gm_pct", "gm", "gross_margin_pct"],
    "l30": ["l30", "last_30", "sales_l30"],
    "d30": ["d30", "demand_30", "demand30"],
    "d60": ["d60", "demand_60", "demand60"],
    "lw": ["lw", "last_week", "last_wk"],
    "ttm": ["ttm", "trailing_twelve_months"],
    "qoh": ["qoh", "quantity_on_hand", "qty_on_hand"],
    "supply": ["supply", "supply_on_hand", "soh"],
    "dc_avail": ["dc_avail", "dc_available", "dc_avail_units", "dc_avail_", "dc_avail_qty", "dc_availabile", "dc_availabl", "dc_availablity", "dc_avail_"],
    "proj_demand": ["proj_demand", "projected_demand", "proj_dmd"],
    "alloc_rec": ["alloc_rec", "allocation_rec", "alloc_recommendation", "alloc_rec_"],
    "final_alloc": ["final_alloc", "final_allocate", "final_allocation", "final_alloc_"],
    "left_dc": ["left_dc", "left_in_dc", "left_dc_", "left_in_dc_"],
    "final_supply": ["final_supply", "final_supp"],
    "demand_check": ["demand_check"],
    "helper": ["helper"],
    "allocated": ["allocated"],
    "intrans": ["intrans", "in_transit"],
    "store_transfer": ["store_transfer"],
    "qty_reserve": ["qty_reserve", "reserve"],
    "store_po_qty": ["store_po_qty", "store_po"],
    "dc_qoh": ["dc_qoh"],
    "dc_staged": ["dc_staged"],
    "dc_rv": ["dc_rv"],
    "dc_po_qty": ["dc_po_qty", "dc_po"],
    "avg_woc": ["avg_woc", "average_woc"],
    "days": ["days"],
}

# Fixed fallback positions from your allocation workbooks. These include hidden columns.
LETTER_FALLBACKS = {
    "item": "O",
    "site": "X",
    "l30": "AK",
    "d30": "AL",
    "d60": "AM",
    "lw": "AN",
    "ttm": "AO",
    "qoh": "AP",
    "supply": "AQ",
    "dc_avail": "AX",
    "flm": "BK",
    "proj_demand": "BM",
    "alloc_rec": "BN",
    "flag": "BR",
    "final_alloc": "BS",
    "left_dc": "BT",
    "final_supply": "BU",
    "demand_check": "BZ",
    "helper": "CA",
}


def build_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    norm_to_cols: Dict[str, List[str]] = {}
    for c in df.columns:
        norm_to_cols.setdefault(normalize_header(c), []).append(c)
    cmap: Dict[str, Optional[str]] = {}
    for field, names in SYNONYMS.items():
        found = None
        for name in names:
            n = normalize_header(name)
            if n in norm_to_cols:
                found = norm_to_cols[n][0]
                break
        if found is None:
            # fuzzy contains match
            for norm, cols in norm_to_cols.items():
                if any(normalize_header(name) == norm or normalize_header(name) in norm for name in names):
                    found = cols[0]
                    break
        if found is None and field in LETTER_FALLBACKS:
            idx = excel_col_to_idx(LETTER_FALLBACKS[field])
            if 0 <= idx < len(df.columns):
                found = df.columns[idx]
        cmap[field] = found
    return cmap


@dataclass
class ColumnDiagnostics:
    rows: int
    columns: int
    header_map: Dict[str, Optional[str]]

    def as_rows(self):
        return [{"field": k, "column": v or ""} for k, v in self.header_map.items()]
