from __future__ import annotations

import numpy as np
import pandas as pd
from schema import build_column_map

EPS = 1e-6


def _num(df: pd.DataFrame, cmap: dict, field: str, default: float = 0.0) -> pd.Series:
    c = cmap.get(field)
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _txt(df: pd.DataFrame, cmap: dict, field: str) -> pd.Series:
    c = cmap.get(field)
    if c in df.columns:
        return df[c].astype(str).fillna("").str.strip()
    return pd.Series("", index=df.index, dtype="object")


def _safe_div(a, b):
    a = pd.Series(a) if not isinstance(a, pd.Series) else a
    b = pd.Series(b, index=a.index) if not isinstance(b, pd.Series) else b
    return (a.astype(float) / (b.astype(float).replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _flag_masks(flag: pd.Series):
    f = flag.astype(str).str.upper().fillna("")
    is_z = ((f.str.contains("NO") & f.str.contains("ALLOC")) | f.str.startswith("Z - NO") | f.str.startswith("Z NO"))
    is_review = f.str.contains("REVIEW")
    is_alloc = f.str.contains("ALLOC") & ~is_z
    return is_alloc, is_review, is_z


def reconstruct_dc_before(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """Estimate item-level DC available before historical Final Alloc was assigned.

    Workbook Left DC is often already reduced after allocations. For training we need
    the pre-allocation state so the model can learn decisions as if it were seeing the
    file before Final Alloc is entered. This reconstructs an item section pool and a
    row-level before/after sequence in original row order.
    """
    item = _txt(df, cmap, "item").replace("", "__missing_item__")
    dc_avail = _num(df, cmap, "dc_avail", 0).fillna(0).clip(lower=0)
    left_dc = _num(df, cmap, "left_dc", np.nan)
    final_alloc = _num(df, cmap, "final_alloc", 0).fillna(0).clip(lower=0)

    before = pd.Series(0.0, index=df.index)
    after = pd.Series(0.0, index=df.index)
    start_pool = pd.Series(0.0, index=df.index)
    section_remaining_ratio = pd.Series(0.0, index=df.index)

    for it, idxs in item.groupby(item, sort=False).groups.items():
        idxs = list(idxs)
        if not idxs:
            continue
        da_max = float(dc_avail.loc[idxs].max())
        total_alloc = float(final_alloc.loc[idxs].sum())
        left_non_na = left_dc.loc[idxs].dropna().clip(lower=0)
        left_plus_current_max = float((left_dc.loc[idxs].fillna(0).clip(lower=0) + final_alloc.loc[idxs]).max())
        min_left_plus_total = float(left_non_na.min() + total_alloc) if len(left_non_na) else total_alloc
        pool = max(da_max, left_plus_current_max, min_left_plus_total, total_alloc)
        remaining = pool
        for idx in idxs:
            before.loc[idx] = max(remaining, 0.0)
            alloc = float(final_alloc.loc[idx]) if np.isfinite(final_alloc.loc[idx]) else 0.0
            remaining = max(0.0, remaining - alloc)
            after.loc[idx] = remaining
            start_pool.loc[idx] = pool
            section_remaining_ratio.loc[idx] = remaining / (pool + EPS)
    return pd.DataFrame({
        "num__dc_start_pool_estimated": start_pool,
        "num__dc_before_row": before,
        "num__dc_after_row_historical": after,
        "num__dc_before_left_plus_final": (left_dc.fillna(0).clip(lower=0) + final_alloc).clip(lower=0),
        "num__section_remaining_ratio_after_hist": section_remaining_ratio,
    }, index=df.index)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    cmap = build_column_map(df)
    out = pd.DataFrame(index=df.index)

    numeric_fields = [
        "department_id", "class_id", "line_id", "site", "mil", "flm", "dc_flm",
        "cost", "retail", "gm_pct", "l30", "d30", "d60", "lw", "ttm", "qoh", "supply",
        "allocated", "intrans", "store_transfer", "qty_reserve", "store_po_qty", "dc_qoh",
        "dc_avail", "dc_staged", "dc_rv", "dc_po_qty", "avg_woc", "days", "proj_demand",
        "alloc_rec", "left_dc", "final_supply", "demand_check", "helper",
    ]
    for f in numeric_fields:
        out[f"num__{f}"] = _num(df, cmap, f, 0).fillna(0)

    flm = out["num__flm"].replace(0, np.nan).fillna(1).clip(lower=1).round()
    supply = out["num__supply"].fillna(0).clip(lower=0)
    qoh = out["num__qoh"].fillna(0).clip(lower=0)
    d60 = out["num__d60"].fillna(0).clip(lower=0)
    d30 = out["num__d30"].fillna(0).clip(lower=0)
    l30 = out["num__l30"].fillna(0).clip(lower=0)
    lw = out["num__lw"].fillna(0).clip(lower=0)
    ttm = out["num__ttm"].fillna(0).clip(lower=0)
    proj = out["num__proj_demand"].fillna(0).clip(lower=0)
    alloc_rec = out["num__alloc_rec"].fillna(0).clip(lower=0)
    dc_avail = out["num__dc_avail"].fillna(0).clip(lower=0)
    left_dc = out["num__left_dc"].fillna(0).clip(lower=0)

    demand_basis = np.maximum.reduce([
        d60.values, proj.values, (l30 * 2.0).values, (d30 * 2.0).values,
        (ttm / 6.0).values, (lw * 8.0).values,
    ])
    out["num__demand_basis"] = demand_basis
    out["num__need_gap"] = np.maximum(0, demand_basis - supply)
    out["num__need_units"] = out["num__need_gap"] / (flm + EPS)
    out["num__demand_cap"] = np.maximum(0, out["num__demand_basis"] + flm - supply)
    out["num__demand_cap_units"] = out["num__demand_cap"] / (flm + EPS)
    out["num__alloc_rec_units"] = alloc_rec / (flm + EPS)
    out["num__dc_units_available"] = dc_avail / (flm + EPS)
    out["num__left_dc_units"] = left_dc / (flm + EPS)
    out["num__leftover_below_flm"] = ((left_dc > 0) & (left_dc < flm)).astype(float)
    out["num__leftover_units_ratio"] = left_dc / (flm + EPS)

    # Demand/velocity relationships.
    out["num__supply_to_demand"] = _safe_div(supply, demand_basis)
    out["num__qoh_to_demand"] = _safe_div(qoh, demand_basis)
    out["num__dc_to_need"] = _safe_div(dc_avail, out["num__need_gap"])
    out["num__need_to_dc"] = _safe_div(out["num__need_gap"], dc_avail)
    out["num__proj_to_d60"] = _safe_div(proj, d60)
    out["num__l30_to_d60"] = _safe_div(l30, d60)
    out["num__d30_to_d60"] = _safe_div(d30, d60)
    out["num__lw_to_l30"] = _safe_div(lw, l30)
    out["num__ttm_monthly"] = ttm / 12.0
    out["num__ttm_60day"] = ttm / 6.0
    out["num__recent_velocity_blend"] = 0.45 * l30 + 0.25 * d30 + 0.20 * (d60 / 2.0) + 0.10 * (lw * 4.29)

    # Allocation recommendation relationships.
    out["num__alloc_rec_to_need"] = _safe_div(alloc_rec, out["num__need_gap"])
    out["num__alloc_rec_to_dc"] = _safe_div(alloc_rec, dc_avail)
    out["num__alloc_rec_to_proj"] = _safe_div(alloc_rec, proj)
    out["num__supply_after_alloc_rec"] = supply + alloc_rec
    out["num__over_demand_after_alloc_rec"] = np.maximum(0, out["num__supply_after_alloc_rec"] - demand_basis)
    out["num__margin_dollars"] = out["num__retail"].fillna(0) - out["num__cost"].fillna(0)
    out["num__alloc_rec_retail_value"] = alloc_rec * out["num__retail"].fillna(0)
    out["num__alloc_rec_margin_value"] = alloc_rec * out["num__margin_dollars"].fillna(0)

    # Flags.
    flag = _txt(df, cmap, "flag")
    is_alloc, is_review, is_z = _flag_masks(flag)
    out["num__is_allocate_flag"] = is_alloc.astype(float)
    out["num__is_review_flag"] = is_review.astype(float)
    out["num__is_z_no_alloc_flag"] = is_z.astype(float)

    # Pre-allocation DC features.
    out = pd.concat([out, reconstruct_dc_before(df, cmap)], axis=1)
    out["num__dc_before_units"] = out["num__dc_before_row"] / (flm + EPS)
    out["num__need_to_dc_before"] = _safe_div(out["num__need_gap"], out["num__dc_before_row"])
    out["num__alloc_rec_to_dc_before"] = _safe_div(alloc_rec, out["num__dc_before_row"])
    out["num__section_has_partial_dc_before"] = ((out["num__dc_before_row"] > 0) & (out["num__dc_before_row"] < flm)).astype(float)

    # Categorical identity fields.
    cat_fields = [
        "vendor", "vendor_site_id", "brand", "department_id", "class_id", "line_id", "product_id",
        "item", "description", "upc", "site", "site_name", "state", "region", "zone", "store_size", "rank", "flag",
    ]
    for f in cat_fields:
        out[f"cat__{f}"] = _txt(df, cmap, f)

    item = out["cat__item"].replace("", "__missing_item__")
    site = out["cat__site"].replace("", "__missing_site__")
    dept = out["cat__department_id"].replace("", "__missing_department__")
    cls = out["cat__class_id"].replace("", "__missing_class__")
    flag_section = pd.Series(np.where(is_review, "review", np.where(is_z, "z_no_alloc", np.where(is_alloc, "allocate", "other"))), index=df.index)
    out["cat__section_type"] = flag_section
    out["cat__item_site_section"] = item.astype(str) + "|" + site.astype(str) + "|" + flag_section.astype(str)
    out["cat__dept_class_section"] = dept.astype(str) + "|" + cls.astype(str) + "|" + flag_section.astype(str)

    def add_group(prefix: str, key: pd.Series):
        g = out.groupby(key, sort=False)
        rows = g["num__demand_basis"].transform("size").astype(float)
        tot_demand = g["num__demand_basis"].transform("sum")
        tot_need = g["num__need_gap"].transform("sum")
        tot_rec = g["num__alloc_rec"].transform("sum")
        tot_supply = g["num__supply"].transform("sum")
        tot_dc_before = g["num__dc_before_row"].transform("max")
        out[f"num__{prefix}_rows"] = rows
        out[f"num__{prefix}_total_demand"] = tot_demand
        out[f"num__{prefix}_share_demand"] = _safe_div(out["num__demand_basis"], tot_demand)
        out[f"num__{prefix}_total_need"] = tot_need
        out[f"num__{prefix}_share_need"] = _safe_div(out["num__need_gap"], tot_need)
        out[f"num__{prefix}_total_alloc_rec"] = tot_rec
        out[f"num__{prefix}_share_alloc_rec"] = _safe_div(out["num__alloc_rec"], tot_rec)
        out[f"num__{prefix}_total_supply"] = tot_supply
        out[f"num__{prefix}_share_supply"] = _safe_div(out["num__supply"], tot_supply)
        out[f"num__{prefix}_total_dc_before"] = tot_dc_before
        out[f"num__{prefix}_need_to_dc_pressure"] = _safe_div(tot_need, tot_dc_before)
        out[f"num__{prefix}_dc_to_need_ratio"] = _safe_div(tot_dc_before, tot_need)
        out[f"num__{prefix}_alloc_rec_to_dc_pressure"] = _safe_div(tot_rec, tot_dc_before)

    # Section-level context: these are the key shift away from row-by-row training.
    add_group("item", item)
    add_group("site", site)
    add_group("department", dept)
    add_group("class", cls)
    add_group("item_site", item.astype(str) + "|" + site.astype(str))
    add_group("dept_class", dept.astype(str) + "|" + cls.astype(str))
    add_group("item_section", item.astype(str) + "|" + flag_section.astype(str))
    add_group("item_site_section", item.astype(str) + "|" + site.astype(str) + "|" + flag_section.astype(str))
    add_group("dept_class_section", dept.astype(str) + "|" + cls.astype(str) + "|" + flag_section.astype(str))

    # Item/section ranking and cumulative pressure in original row order.
    def add_rank_cum(prefix: str, key: pd.Series):
        for metric, source in [
            ("need", out["num__need_gap"]),
            ("demand", out["num__demand_basis"]),
            ("alloc_rec", out["num__alloc_rec"]),
            ("dc_before", out["num__dc_before_row"]),
            ("recent_velocity", out["num__recent_velocity_blend"]),
        ]:
            out[f"num__{prefix}_rank_{metric}_descending"] = source.groupby(key, sort=False).rank(method="first", ascending=False)
            out[f"num__{prefix}_pct_rank_{metric}"] = source.groupby(key, sort=False).rank(method="first", ascending=False, pct=True).fillna(1.0)
            out[f"num__{prefix}_cum_{metric}_before"] = source.groupby(key, sort=False).cumsum() - source
            total = source.groupby(key, sort=False).transform("sum")
            out[f"num__{prefix}_remaining_{metric}_after"] = total - source.groupby(key, sort=False).cumsum()

    add_rank_cum("item", item)
    add_rank_cum("item_section", item.astype(str) + "|" + flag_section.astype(str))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.copy()  # defragment after extensive feature construction


def build_targets(df: pd.DataFrame, max_units: int = 240) -> pd.DataFrame:
    cmap = build_column_map(df)
    final_alloc = _num(df, cmap, "final_alloc", 0).fillna(0).clip(lower=0)
    flm = _num(df, cmap, "flm", 1).fillna(1).where(lambda s: s > 0, 1).round()
    units = np.rint(final_alloc / (flm + EPS)).astype(int).clip(0, max_units)
    alloc = (final_alloc > 0).astype(int)
    flag = _txt(df, cmap, "flag")
    is_alloc, is_review, is_z = _flag_masks(flag)
    feats = build_feature_frame(df)

    need_units = pd.to_numeric(feats.get("num__need_units", 0), errors="coerce").fillna(0)
    rec_units = pd.to_numeric(feats.get("num__alloc_rec_units", 0), errors="coerce").fillna(0)
    dc_before_units = pd.to_numeric(feats.get("num__dc_before_units", 0), errors="coerce").fillna(0)
    pct_rank_need = pd.to_numeric(feats.get("num__item_section_pct_rank_need", 1), errors="coerce").fillna(1)

    # Priority target for section-by-section allocation. Positive historical rows rise
    # first; among all rows, high need/alloc-rec and high within-section rank get credit.
    priority = (
        0.58 * alloc.astype(float)
        + 0.18 * np.tanh(need_units / 3.0)
        + 0.14 * np.tanh(rec_units / 3.0)
        + 0.06 * np.tanh(dc_before_units / 5.0)
        + 0.04 * (1.0 - pct_rank_need.clip(0, 1))
    ).clip(0, 1)

    return pd.DataFrame({
        "__target_final_alloc": final_alloc.astype(float),
        "__target_units": units.astype(int),
        "__target_alloc": alloc.astype(int),
        "__target_priority": np.asarray(priority, dtype=float),
        "__is_allocate": is_alloc.astype(int),
        "__is_review": is_review.astype(int),
        "__is_z_no_alloc": is_z.astype(int),
        "__section_type": np.where(is_review, "review", np.where(is_z, "z_no_alloc", np.where(is_alloc, "allocate", "other"))),
    }, index=df.index)


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c.startswith("num__") or c.startswith("cat__")]


def split_three_model_datasets(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "Base Allocation": data[data["__is_allocate"].astype(bool)].reset_index(drop=True),
        "Base Review": data[data["__is_review"].astype(bool)].reset_index(drop=True),
        "Base Z No Alloc": data[data["__is_z_no_alloc"].astype(bool)].reset_index(drop=True),
    }
