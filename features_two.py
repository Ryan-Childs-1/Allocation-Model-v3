from __future__ import annotations

import numpy as np
import pandas as pd
from schema import build_column_map

NUMERIC_FIELDS = [
    'department_id','class_id','line_id','site','mil','flm','dc_flm','cost','retail','gm_pct',
    'l30','d30','d60','lw','ttm','qoh','supply','allocated','intrans','dc_qoh','dc_avail',
    'avg_woc','days','proj_demand','alloc_rec','left_dc','final_supply','demand_check','helper'
]
CATEGORICAL_FIELDS = [
    'vendor','vendor_site_id','brand','department_id','class_id','line_id','product_id','item','description',
    'upc','site','site_name','state','region','zone','store_size','rank','flag'
]

def _num(df: pd.DataFrame, cmap: dict, field: str, default=0.0) -> pd.Series:
    col = cmap.get(field)
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(default, index=df.index, dtype='float64')

def _txt(df: pd.DataFrame, cmap: dict, field: str, default='') -> pd.Series:
    col = cmap.get(field)
    if col in df.columns:
        return df[col].astype(str).fillna(default)
    return pd.Series(default, index=df.index, dtype='object')

def safe_div(a, b, default=0.0):
    a = pd.Series(a)
    b = pd.Series(b)
    out = a.astype(float) / b.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)

def _flag_section(flag: pd.Series) -> pd.Series:
    f = flag.astype(str).str.upper()
    return np.select(
        [f.str.contains('REVIEW', na=False), f.str.contains('NO.*ALLOC|Z - NO|Z NO', regex=True, na=False), f.str.contains('ALLOC', na=False)],
        ['review','z_no_alloc','allocate'],
        default='other'
    )

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a dense feature frame before preprocessing.

    The trainer later fits the preprocessor. The Streamlit runtime uses the exported
    preprocessing values/mappings and the same feature column names.
    """
    cmap = build_column_map(df)
    X = pd.DataFrame(index=df.index)

    # Direct numeric/categorical signals.
    for f in NUMERIC_FIELDS:
        X[f'num__{f}'] = _num(df, cmap, f, 0).fillna(0)
    for f in CATEGORICAL_FIELDS:
        X[f'cat__{f}'] = _txt(df, cmap, f, '').fillna('').astype(str)

    flag = _txt(df, cmap, 'flag')
    section = pd.Series(_flag_section(flag), index=df.index)
    X['cat__section_type'] = section.astype(str)
    X['num__is_allocate_flag'] = (section == 'allocate').astype(float)
    X['num__is_review_flag'] = (section == 'review').astype(float)
    X['num__is_z_no_alloc_flag'] = (section == 'z_no_alloc').astype(float)

    flm = _num(df, cmap, 'flm', 1).fillna(1).clip(lower=1)
    d60 = _num(df, cmap, 'd60', 0).fillna(0)
    d30 = _num(df, cmap, 'd30', 0).fillna(0)
    l30 = _num(df, cmap, 'l30', 0).fillna(0)
    lw = _num(df, cmap, 'lw', 0).fillna(0)
    ttm = _num(df, cmap, 'ttm', 0).fillna(0)
    proj = _num(df, cmap, 'proj_demand', 0).fillna(0)
    supply = _num(df, cmap, 'supply', 0).fillna(0)
    qoh = _num(df, cmap, 'qoh', 0).fillna(0)
    alloc_rec = _num(df, cmap, 'alloc_rec', 0).fillna(0)
    left_dc = _num(df, cmap, 'left_dc', 0).fillna(0)
    dc_avail = _num(df, cmap, 'dc_avail', 0).fillna(0)
    final_alloc = _num(df, cmap, 'final_alloc', 0).fillna(0)
    cost = _num(df, cmap, 'cost', 0).fillna(0)
    retail = _num(df, cmap, 'retail', 0).fillna(0)

    demand_basis = pd.Series(np.maximum.reduce([d60, proj, l30*2, d30*2, ttm/6, lw*8]), index=df.index).fillna(0)
    need_gap = (demand_basis - supply).clip(lower=0)
    dc_start = pd.Series(np.maximum(left_dc + final_alloc, dc_avail), index=df.index).fillna(0)

    X['num__demand_basis'] = demand_basis
    X['num__need_gap'] = need_gap
    X['num__need_units'] = safe_div(need_gap, flm)
    X['num__demand_cap'] = (demand_basis + flm - supply).clip(lower=0)
    X['num__demand_cap_units'] = safe_div(X['num__demand_cap'], flm)
    X['num__alloc_rec_units'] = safe_div(alloc_rec, flm)
    X['num__dc_units_available'] = safe_div(dc_start, flm)
    X['num__left_dc_units'] = safe_div(left_dc, flm)
    X['num__leftover_below_flm'] = ((left_dc > 0) & (left_dc < flm)).astype(float)
    X['num__leftover_units_ratio'] = safe_div(left_dc, flm)
    X['num__supply_to_demand'] = safe_div(supply, demand_basis)
    X['num__qoh_to_demand'] = safe_div(qoh, demand_basis)
    X['num__dc_to_need'] = safe_div(dc_start, need_gap)
    X['num__need_to_dc'] = safe_div(need_gap, dc_start)
    X['num__proj_to_d60'] = safe_div(proj, d60)
    X['num__l30_to_d60'] = safe_div(l30, d60)
    X['num__d30_to_d60'] = safe_div(d30, d60)
    X['num__lw_to_l30'] = safe_div(lw, l30)
    X['num__ttm_monthly'] = ttm / 12.0
    X['num__ttm_60day'] = ttm / 6.0

    # ------------------------------------------------------------------
    # Demand-focused feature engineering
    # ------------------------------------------------------------------
    # These features intentionally expand the relationships among the demand
    # columns most relevant to allocation decisions: L30, D30, D60, LW and TTM.
    # They help the model distinguish true sustained demand from short spikes,
    # recent acceleration, seasonality, and demand consistency.
    d30_per_30 = d30
    d60_per_30 = d60 / 2.0
    ttm_per_30 = ttm / 12.0
    lw_monthly = lw * 4.29
    lw_60day = lw * 8.58
    l30_60day = l30 * 2.0
    d30_60day = d30 * 2.0

    X['num__demand_l30_per_30'] = l30
    X['num__demand_d30_per_30'] = d30_per_30
    X['num__demand_d60_per_30'] = d60_per_30
    X['num__demand_lw_monthly'] = lw_monthly
    X['num__demand_ttm_per_30'] = ttm_per_30
    X['num__demand_lw_60day_equiv'] = lw_60day
    X['num__demand_l30_60day_equiv'] = l30_60day
    X['num__demand_d30_60day_equiv'] = d30_60day

    # Recent trend / acceleration signals. Positive values indicate recent demand
    # is running ahead of longer-window demand.
    X['num__demand_l30_minus_d60_per30'] = l30 - d60_per_30
    X['num__demand_d30_minus_d60_per30'] = d30_per_30 - d60_per_30
    X['num__demand_lw_monthly_minus_l30'] = lw_monthly - l30
    X['num__demand_l30_minus_ttm_monthly'] = l30 - ttm_per_30
    X['num__demand_d60_per30_minus_ttm_monthly'] = d60_per_30 - ttm_per_30
    X['num__demand_recent_acceleration'] = (l30 - d60_per_30) + 0.5 * (lw_monthly - l30)
    X['num__demand_recent_vs_baseline_delta'] = ((l30 + d30_per_30 + lw_monthly) / 3.0) - ((d60_per_30 + ttm_per_30) / 2.0)

    # Ratios across horizons. These are clipped later by the preprocessing path
    # after inf/nan cleanup.
    X['num__demand_l30_to_d30'] = safe_div(l30, d30_per_30)
    X['num__demand_l30_to_d60_per30'] = safe_div(l30, d60_per_30)
    X['num__demand_d30_to_d60_per30'] = safe_div(d30_per_30, d60_per_30)
    X['num__demand_lw_monthly_to_l30'] = safe_div(lw_monthly, l30)
    X['num__demand_lw_monthly_to_d60_per30'] = safe_div(lw_monthly, d60_per_30)
    X['num__demand_l30_to_ttm_monthly'] = safe_div(l30, ttm_per_30)
    X['num__demand_d60_per30_to_ttm_monthly'] = safe_div(d60_per_30, ttm_per_30)

    # Consensus / stability.  High consistency means the demand windows broadly
    # agree; high volatility/spike scores mean the allocation should be handled
    # more carefully.
    demand_matrix = np.vstack([
        np.asarray(l30, dtype=float),
        np.asarray(d30_per_30, dtype=float),
        np.asarray(d60_per_30, dtype=float),
        np.asarray(lw_monthly, dtype=float),
        np.asarray(ttm_per_30, dtype=float),
    ]).T
    X['num__demand_window_mean'] = np.nanmean(demand_matrix, axis=1)
    X['num__demand_window_median'] = np.nanmedian(demand_matrix, axis=1)
    X['num__demand_window_max'] = np.nanmax(demand_matrix, axis=1)
    X['num__demand_window_min'] = np.nanmin(demand_matrix, axis=1)
    X['num__demand_window_range'] = X['num__demand_window_max'] - X['num__demand_window_min']
    X['num__demand_window_std'] = np.nanstd(demand_matrix, axis=1)
    X['num__demand_window_cv'] = safe_div(X['num__demand_window_std'], X['num__demand_window_mean'])
    X['num__demand_consistency_score'] = safe_div(X['num__demand_window_mean'], X['num__demand_window_mean'] + X['num__demand_window_std'])
    X['num__demand_spike_score'] = safe_div(X['num__demand_window_max'] - X['num__demand_window_median'], X['num__demand_window_median'])
    X['num__demand_zero_window_count'] = (pd.DataFrame(demand_matrix, index=df.index).fillna(0) <= 0).sum(axis=1).astype(float)

    # Weighted demand consensus. Recent windows matter most, but TTM/D60 keep
    # the model anchored to sustained demand.
    X['num__demand_recent_velocity_blend'] = (0.35*l30 + 0.25*d30_per_30 + 0.25*lw_monthly + 0.10*d60_per_30 + 0.05*ttm_per_30)
    X['num__recent_velocity_blend'] = X['num__demand_recent_velocity_blend']
    X['num__demand_balanced_consensus'] = (0.30*l30 + 0.25*d30_per_30 + 0.20*d60_per_30 + 0.15*lw_monthly + 0.10*ttm_per_30)
    X['num__demand_conservative_consensus'] = (0.20*l30 + 0.20*d30_per_30 + 0.30*d60_per_30 + 0.05*lw_monthly + 0.25*ttm_per_30)
    X['num__demand_aggressive_consensus'] = (0.40*l30 + 0.25*d30_per_30 + 0.25*lw_monthly + 0.10*d60_per_30)

    # Demand-to-inventory/pack-size features. These make the model more aware of
    # how many FLM packs are actually justified by each demand horizon.
    for _name, _metric in {
        'l30': l30,
        'd30': d30_per_30,
        'd60_per30': d60_per_30,
        'lw_monthly': lw_monthly,
        'ttm_monthly': ttm_per_30,
        'balanced_consensus': X['num__demand_balanced_consensus'],
        'aggressive_consensus': X['num__demand_aggressive_consensus'],
    }.items():
        X[f'num__{_name}_demand_units'] = safe_div(_metric, flm)
        X[f'num__{_name}_demand_gap'] = (_metric - supply).clip(lower=0)
        X[f'num__{_name}_demand_gap_units'] = safe_div(X[f'num__{_name}_demand_gap'], flm)
        X[f'num__{_name}_supply_coverage'] = safe_div(supply, _metric)
        X[f'num__{_name}_qoh_coverage'] = safe_div(qoh, _metric)

    X['num__alloc_rec_to_need'] = safe_div(alloc_rec, need_gap)
    X['num__alloc_rec_to_dc'] = safe_div(alloc_rec, dc_start)
    X['num__alloc_rec_to_proj'] = safe_div(alloc_rec, proj)
    X['num__supply_after_alloc_rec'] = supply + alloc_rec
    X['num__over_demand_after_alloc_rec'] = ((supply + alloc_rec) - demand_basis).clip(lower=0)
    X['num__margin_dollars'] = retail - cost
    X['num__alloc_rec_retail_value'] = alloc_rec * retail
    X['num__alloc_rec_margin_value'] = alloc_rec * (retail - cost)

    # Pre-allocation DC reconstruction: historical left DC was already after Final Alloc; add back final_alloc.
    X['num__dc_start_pool_estimated'] = dc_start
    X['num__dc_before_left_plus_final'] = left_dc + final_alloc
    X['num__dc_before_units'] = safe_div(dc_start, flm)
    X['num__need_to_dc_before'] = safe_div(need_gap, dc_start)
    X['num__alloc_rec_to_dc_before'] = safe_div(alloc_rec, dc_start)
    X['num__section_has_partial_dc_before'] = ((dc_start > 0) & (dc_start < flm)).astype(float)

    # Group/section features.
    item = _txt(df, cmap, 'item').replace('', '__missing_item__')
    site = _txt(df, cmap, 'site').replace('', '__missing_site__')
    dept = _txt(df, cmap, 'department_id').replace('', '__missing_dept__')
    cls = _txt(df, cmap, 'class_id').replace('', '__missing_class__')
    item_site = item.astype(str) + '|' + site.astype(str)
    dept_class = dept.astype(str) + '|' + cls.astype(str)
    item_section = item.astype(str) + '|' + section.astype(str)
    X['cat__item_site_section'] = item_site + '|' + section.astype(str)
    X['cat__dept_class_section'] = dept_class + '|' + section.astype(str)

    def add_group(prefix: str, key: pd.Series):
        grp = pd.DataFrame({'key': key.astype(str), 'demand': demand_basis, 'need': need_gap, 'alloc_rec': alloc_rec, 'supply': supply, 'dc': dc_start}, index=df.index)
        g = grp.groupby('key')
        rows = g['key'].transform('size').astype(float)
        X[f'num__{prefix}_rows'] = rows
        for col in ['demand','need','alloc_rec','supply','dc']:
            total = g[col].transform('sum').replace(0, np.nan)
            X[f'num__{prefix}_total_{col if col != "dc" else "dc_before"}'] = total.fillna(0)
            X[f'num__{prefix}_share_{col if col != "dc" else "dc"}'] = safe_div(grp[col], total)
        X[f'num__{prefix}_need_to_dc_pressure'] = safe_div(g['need'].transform('sum'), g['dc'].transform('sum'))
        X[f'num__{prefix}_dc_to_need_ratio'] = safe_div(g['dc'].transform('sum'), g['need'].transform('sum'))
        X[f'num__{prefix}_alloc_rec_to_dc_pressure'] = safe_div(g['alloc_rec'].transform('sum'), g['dc'].transform('sum'))

    add_group('item', item)
    add_group('site', site)
    add_group('department', dept)
    add_group('class', cls)
    add_group('item_site', item_site)
    add_group('dept_class', dept_class)
    add_group('item_section', item_section)

    # Ranking/cumulative features within item and within item-section.
    def add_rank(prefix: str, key: pd.Series, metric_name: str, metric: pd.Series, descending=True):
        tmp = pd.DataFrame({'key': key.astype(str), 'metric': metric.fillna(0), 'order': df.get('__row_order', pd.Series(range(len(df)), index=df.index))}, index=df.index)
        ascending = not descending
        # rank 1 = largest need/demand/etc.
        X[f'num__{prefix}_rank_{metric_name}_descending'] = tmp.groupby('key')['metric'].rank(method='first', ascending=False)
        X[f'num__{prefix}_pct_rank_{metric_name}'] = tmp.groupby('key')['metric'].rank(pct=True, method='average')
        X[f'num__{prefix}_cum_{metric_name}_before'] = tmp.groupby('key')['metric'].cumsum() - tmp['metric']
        X[f'num__{prefix}_remaining_{metric_name}_after'] = tmp.groupby('key')['metric'].transform('sum') - tmp.groupby('key')['metric'].cumsum()

    for keyname, key in [('item', item), ('item_section', item_section)]:
        add_rank(keyname, key, 'need', need_gap)
        add_rank(keyname, key, 'demand', demand_basis)
        add_rank(keyname, key, 'alloc_rec', alloc_rec)
        add_rank(keyname, key, 'dc_before', dc_start)
        add_rank(keyname, key, 'recent_velocity', X['num__recent_velocity_blend'])
        add_rank(keyname, key, 'balanced_consensus', X['num__demand_balanced_consensus'])
        add_rank(keyname, key, 'demand_acceleration', X['num__demand_recent_acceleration'])
        add_rank(keyname, key, 'demand_consistency', X['num__demand_consistency_score'])

    # Ensure no infs leak into model preprocessing.
    for c in X.columns:
        if c.startswith('num__'):
            X[c] = pd.to_numeric(X[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            X[c] = X[c].fillna('').astype(str)
    return X

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    cmap = build_column_map(df)
    flm = _num(df, cmap, 'flm', 1).fillna(1).clip(lower=1)
    final_alloc = _num(df, cmap, 'final_alloc', 0).fillna(0).clip(lower=0)
    units = np.floor(final_alloc / flm).clip(lower=0).astype(int)
    flag = _txt(df, cmap, 'flag')
    section = pd.Series(_flag_section(flag), index=df.index)
    # Priority target for ranking: actual units + demand pressure. Positive rows dominate.
    X = build_feature_frame(df)
    priority = (units > 0).astype(float) * 0.7 + pd.to_numeric(X.get('num__need_to_dc_before', 0), errors='coerce').fillna(0).clip(0, 3) / 3 * 0.3
    return pd.DataFrame({
        '__target_units': units.astype(int),
        '__target_allocated': (final_alloc > 0).astype(int),
        '__target_priority': priority.astype(float),
        '__section_type': section.astype(str),
    }, index=df.index)
