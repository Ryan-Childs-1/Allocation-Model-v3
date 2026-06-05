
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from schema import build_column_map

try:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass

NUMERIC_FIELDS = [
    'department_id','class_id','line_id','site','mil','flm','dc_flm','cost','retail','gm_pct',
    'l30','d30','d60','lw','ttm','qoh','supply','allocated','intrans','store_transfer','qty_reserve',
    'store_po_qty','dc_qoh','dc_avail','dc_staged','dc_rv','dc_po_qty','avg_woc','days',
    'proj_demand','alloc_rec','left_dc','final_supply','demand_check','helper'
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
    return pd.Series(np.select(
        [f.str.contains('REVIEW', na=False), f.str.contains('NO.*ALLOC|Z - NO|Z NO', regex=True, na=False), f.str.contains('ALLOC', na=False)],
        ['review','z_no_alloc','allocate'],
        default='other'
    ), index=flag.index)

def _pct_rank_desc(s: pd.Series) -> pd.Series:
    try:
        return s.rank(method='average', ascending=False, pct=True).fillna(1.0)
    except Exception:
        return pd.Series(1.0, index=s.index)

def _add_group_features(num: dict, key: pd.Series, prefix: str, metrics: dict[str, pd.Series]):
    # Build group features into dict to avoid pandas fragmentation warnings.
    key = key.fillna('__missing__').astype(str)
    rows = key.groupby(key).transform('size').astype(float)
    num[f'{prefix}_rows'] = rows
    for name, s in metrics.items():
        s = pd.to_numeric(s, errors='coerce').fillna(0)
        total = s.groupby(key).transform('sum').fillna(0)
        total_name = 'dc_before' if name == 'dc' else name
        num[f'{prefix}_total_{total_name}'] = total
        num[f'{prefix}_share_{name}'] = safe_div(s, total)
    # Derived pressure ratios.
    if 'need' in metrics and 'dc' in metrics:
        num[f'{prefix}_need_to_dc_pressure'] = safe_div(num[f'{prefix}_total_need'], num[f'{prefix}_total_dc_before'])
        num[f'{prefix}_dc_to_need_ratio'] = safe_div(num[f'{prefix}_total_dc_before'], num[f'{prefix}_total_need'])
    if 'alloc_rec' in metrics and 'dc' in metrics:
        num[f'{prefix}_alloc_rec_to_dc_pressure'] = safe_div(num[f'{prefix}_total_alloc_rec'], num[f'{prefix}_total_dc_before'])

def _add_rank_features(num: dict, key: pd.Series, prefix: str, metric_name: str, metric: pd.Series):
    key = key.fillna('__missing__').astype(str)
    m = pd.to_numeric(metric, errors='coerce').fillna(0)
    rank = m.groupby(key).rank(method='first', ascending=False).fillna(0)
    pct = m.groupby(key).transform(lambda s: _pct_rank_desc(s))
    csum_before = m.groupby(key).cumsum() - m
    total = m.groupby(key).transform('sum').fillna(0)
    num[f'{prefix}_rank_{metric_name}_descending'] = rank
    num[f'{prefix}_pct_rank_{metric_name}'] = pct
    num[f'{prefix}_cum_{metric_name}_before'] = csum_before.fillna(0)
    num[f'{prefix}_remaining_{metric_name}_after'] = (total - csum_before - m).clip(lower=0).fillna(0)

def _demand_pattern_labels(l30, d30, d60_per30, lw_monthly, ttm_per30) -> pd.Series:
    recent_avg = (l30 + d30 + lw_monthly) / 3.0
    baseline = (d60_per30 + ttm_per30) / 2.0
    labels = pd.Series('mixed_demand', index=l30.index, dtype='object')
    labels[(recent_avg <= 0) & (ttm_per30 <= 0)] = 'dead_item'
    labels[(l30 <= 0) & (d30 <= 0) & (ttm_per30 > 0)] = 'long_tail_no_recent'
    labels[(recent_avg > baseline * 1.75) & (recent_avg > 0)] = 'recent_spike'
    labels[(recent_avg < baseline * 0.45) & (baseline > 0)] = 'recent_decline'
    labels[(lw_monthly > l30 * 1.8) & (lw_monthly > 0)] = 'weekly_acceleration'
    labels[(ttm_per30 > 0) & (l30 > ttm_per30 * 1.4) & (d60_per30 > ttm_per30)] = 'seasonal_recovery'
    labels[(ttm_per30 <= 0) & (recent_avg > 0)] = 'new_recent_activity'
    # Consistent demand overrides mixed when windows agree.
    mx = pd.concat([l30, d30, d60_per30, lw_monthly, ttm_per30], axis=1).max(axis=1)
    mn = pd.concat([l30, d30, d60_per30, lw_monthly, ttm_per30], axis=1).min(axis=1)
    labels[(mx > 0) & ((mx - mn) / mx.replace(0, np.nan) < 0.35)] = 'consistent_demand'
    return labels.fillna('mixed_demand')

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features in batched dictionaries to avoid DataFrame fragmentation.

    This v7 feature builder adds the five improvement themes requested:
    1. optimizer-ready section/item DC features,
    2. quantity-correction features,
    3. workbook/backtest-oriented outcome features,
    4. demand-window pattern features,
    5. human override / Alloc. Rec. trust features.
    """
    cmap = build_column_map(df)
    idx = df.index
    num: dict[str, pd.Series | np.ndarray | float] = {}
    cat: dict[str, pd.Series | np.ndarray | str] = {}

    # Direct numeric/categorical signals.
    for f in NUMERIC_FIELDS:
        num[f] = _num(df, cmap, f, 0).fillna(0)
    for f in CATEGORICAL_FIELDS:
        cat[f] = _txt(df, cmap, f, '').fillna('').astype(str)

    flag = _txt(df, cmap, 'flag')
    section = _flag_section(flag)
    cat['section_type'] = section.astype(str)
    num['is_allocate_flag'] = (section == 'allocate').astype(float)
    num['is_review_flag'] = (section == 'review').astype(float)
    num['is_z_no_alloc_flag'] = (section == 'z_no_alloc').astype(float)

    flm = _num(df, cmap, 'flm', 1).fillna(1).clip(lower=1)
    l30 = _num(df, cmap, 'l30', 0).fillna(0)
    d30 = _num(df, cmap, 'd30', 0).fillna(0)
    d60 = _num(df, cmap, 'd60', 0).fillna(0)
    lw = _num(df, cmap, 'lw', 0).fillna(0)
    ttm = _num(df, cmap, 'ttm', 0).fillna(0)
    proj = _num(df, cmap, 'proj_demand', 0).fillna(0)
    supply = _num(df, cmap, 'supply', 0).fillna(0)
    qoh = _num(df, cmap, 'qoh', 0).fillna(0)
    alloc_rec = _num(df, cmap, 'alloc_rec', 0).fillna(0)
    left_dc = _num(df, cmap, 'left_dc', 0).fillna(0)
    dc_avail = _num(df, cmap, 'dc_avail', 0).fillna(0)
    final_alloc = _num(df, cmap, 'final_alloc', 0).fillna(0).clip(lower=0)
    cost = _num(df, cmap, 'cost', 0).fillna(0)
    retail = _num(df, cmap, 'retail', 0).fillna(0)

    # Core demand horizons.
    d30_per30 = d30
    d60_per30 = d60 / 2.0
    ttm_per30 = ttm / 12.0
    lw_monthly = lw * 4.29
    lw_60day = lw * 8.58
    l30_60day = l30 * 2.0
    d30_60day = d30 * 2.0

    demand_basis = pd.Series(np.maximum.reduce([d60, proj, l30*2, d30*2, ttm/6, lw*8.58]), index=idx).fillna(0)
    need_gap = (demand_basis - supply).clip(lower=0)
    dc_start = pd.Series(np.maximum(left_dc + final_alloc, dc_avail), index=idx).fillna(0)

    # Baseline need/DC features.
    num.update({
        'demand_basis': demand_basis,
        'need_gap': need_gap,
        'need_units': safe_div(need_gap, flm),
        'demand_cap': (demand_basis + flm - supply).clip(lower=0),
        'demand_cap_units': safe_div((demand_basis + flm - supply).clip(lower=0), flm),
        'alloc_rec_units': safe_div(alloc_rec, flm),
        'dc_units_available': safe_div(dc_start, flm),
        'left_dc_units': safe_div(left_dc, flm),
        'leftover_below_flm': ((left_dc > 0) & (left_dc < flm)).astype(float),
        'leftover_units_ratio': safe_div(left_dc, flm),
        'supply_to_demand': safe_div(supply, demand_basis),
        'qoh_to_demand': safe_div(qoh, demand_basis),
        'dc_to_need': safe_div(dc_start, need_gap),
        'need_to_dc': safe_div(need_gap, dc_start),
        'proj_to_d60': safe_div(proj, d60),
        'l30_to_d60': safe_div(l30, d60),
        'd30_to_d60': safe_div(d30, d60),
        'lw_to_l30': safe_div(lw, l30),
        'ttm_monthly': ttm_per30,
        'ttm_60day': ttm / 6.0,
        'dc_start_pool_estimated': dc_start,
        'dc_before_row': dc_start,
        'dc_after_row_historical': (dc_start - final_alloc).clip(lower=0),
        'dc_before_left_plus_final': left_dc + final_alloc,
        'section_remaining_ratio_after_hist': safe_div((dc_start - final_alloc).clip(lower=0), dc_start),
        'dc_before_units': safe_div(dc_start, flm),
        'need_to_dc_before': safe_div(need_gap, dc_start),
        'alloc_rec_to_dc_before': safe_div(alloc_rec, dc_start),
        'section_has_partial_dc_before': ((dc_start > 0) & (dc_start < flm)).astype(float),
        'margin_dollars': retail - cost,
        'alloc_rec_retail_value': alloc_rec * retail,
        'alloc_rec_margin_value': alloc_rec * (retail - cost),
    })

    # Demand-window features and pattern classes.
    demand_df = pd.DataFrame({
        'l30': l30, 'd30': d30_per30, 'd60_per30': d60_per30, 'lw_monthly': lw_monthly, 'ttm_monthly': ttm_per30
    }, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0)
    recent_avg = demand_df[['l30','d30','lw_monthly']].mean(axis=1)
    baseline = demand_df[['d60_per30','ttm_monthly']].mean(axis=1)
    window_mean = demand_df.mean(axis=1)
    window_med = demand_df.median(axis=1)
    window_max = demand_df.max(axis=1)
    window_min = demand_df.min(axis=1)
    window_std = demand_df.std(axis=1).fillna(0)
    pattern = _demand_pattern_labels(l30, d30_per30, d60_per30, lw_monthly, ttm_per30)
    cat['demand_pattern'] = pattern
    for name, metric in {
        'l30': l30,
        'd30_per30': d30_per30,
        'd60_per30': d60_per30,
        'lw_monthly': lw_monthly,
        'ttm_monthly': ttm_per30,
        'lw_60day_equiv': lw_60day,
        'l30_60day_equiv': l30_60day,
        'd30_60day_equiv': d30_60day,
    }.items():
        num[f'demand_{name}'] = metric
    num.update({
        'demand_l30_minus_d60_per30': l30 - d60_per30,
        'demand_d30_minus_d60_per30': d30_per30 - d60_per30,
        'demand_lw_monthly_minus_l30': lw_monthly - l30,
        'demand_l30_minus_ttm_monthly': l30 - ttm_per30,
        'demand_d60_per30_minus_ttm_monthly': d60_per30 - ttm_per30,
        'demand_recent_acceleration': (l30 - d60_per30) + 0.5 * (lw_monthly - l30),
        'demand_recent_vs_baseline_delta': recent_avg - baseline,
        'demand_recent_vs_baseline_ratio': safe_div(recent_avg, baseline),
        'demand_l30_to_d30': safe_div(l30, d30_per30),
        'demand_l30_to_d60_per30': safe_div(l30, d60_per30),
        'demand_d30_to_d60_per30': safe_div(d30_per30, d60_per30),
        'demand_lw_monthly_to_l30': safe_div(lw_monthly, l30),
        'demand_lw_monthly_to_d60_per30': safe_div(lw_monthly, d60_per30),
        'demand_l30_to_ttm_monthly': safe_div(l30, ttm_per30),
        'demand_d60_per30_to_ttm_monthly': safe_div(d60_per30, ttm_per30),
        'demand_window_mean': window_mean,
        'demand_window_median': window_med,
        'demand_window_max': window_max,
        'demand_window_min': window_min,
        'demand_window_range': window_max - window_min,
        'demand_window_std': window_std,
        'demand_window_cv': safe_div(window_std, window_mean),
        'demand_consistency_score': safe_div(window_mean, window_mean + window_std),
        'demand_spike_score': safe_div(window_max - window_med, window_med),
        'demand_zero_window_count': (demand_df <= 0).sum(axis=1).astype(float),
        'demand_recent_velocity_blend': (0.35*l30 + 0.25*d30_per30 + 0.25*lw_monthly + 0.10*d60_per30 + 0.05*ttm_per30),
        'recent_velocity_blend': (0.35*l30 + 0.25*d30_per30 + 0.25*lw_monthly + 0.10*d60_per30 + 0.05*ttm_per30),
        'demand_balanced_consensus': (0.30*l30 + 0.25*d30_per30 + 0.20*d60_per30 + 0.15*lw_monthly + 0.10*ttm_per30),
        'demand_conservative_consensus': (0.20*l30 + 0.20*d30_per30 + 0.30*d60_per30 + 0.05*lw_monthly + 0.25*ttm_per30),
        'demand_aggressive_consensus': (0.40*l30 + 0.25*d30_per30 + 0.25*lw_monthly + 0.10*d60_per30),
    })
    # Pattern indicator features for both MLP and display.
    for patt in ['dead_item','long_tail_no_recent','recent_spike','recent_decline','weekly_acceleration','seasonal_recovery','new_recent_activity','consistent_demand','mixed_demand']:
        num[f'demand_pattern_{patt}'] = (pattern == patt).astype(float)

    for name, metric in {
        'l30': l30,
        'd30': d30_per30,
        'd60_per30': d60_per30,
        'lw_monthly': lw_monthly,
        'ttm_monthly': ttm_per30,
        'balanced_consensus': num['demand_balanced_consensus'],
        'aggressive_consensus': num['demand_aggressive_consensus'],
        'conservative_consensus': num['demand_conservative_consensus'],
    }.items():
        num[f'{name}_demand_units'] = safe_div(metric, flm)
        gap = (pd.Series(metric, index=idx) - supply).clip(lower=0)
        num[f'{name}_demand_gap'] = gap
        num[f'{name}_demand_gap_units'] = safe_div(gap, flm)
        num[f'{name}_supply_coverage'] = safe_div(supply, metric)
        num[f'{name}_qoh_coverage'] = safe_div(qoh, metric)

    # Alloc Rec trust / human override features. At prediction time final_alloc is usually blank;
    # these become neutral, while during training they provide useful non-output diagnostics.
    alloc_rec_units = safe_div(alloc_rec, flm)
    final_units = safe_div(final_alloc, flm)
    num.update({
        'alloc_rec_to_need': safe_div(alloc_rec, need_gap),
        'alloc_rec_to_dc': safe_div(alloc_rec, dc_start),
        'alloc_rec_to_proj': safe_div(alloc_rec, proj),
        'supply_after_alloc_rec': supply + alloc_rec,
        'over_demand_after_alloc_rec': (supply + alloc_rec - demand_basis).clip(lower=0),
        'alloc_rec_units_minus_need_units': alloc_rec_units - safe_div(need_gap, flm),
        'alloc_rec_units_minus_demand_cap_units': alloc_rec_units - safe_div((demand_basis + flm - supply).clip(lower=0), flm),
        'alloc_rec_trust_signal': safe_div(alloc_rec, (need_gap + flm)),
        'historical_override_any': ((final_alloc != alloc_rec) & ((final_alloc > 0) | (alloc_rec > 0))).astype(float),
        'historical_override_cut': ((alloc_rec > 0) & (final_alloc < alloc_rec)).astype(float),
        'historical_override_add': ((final_alloc > alloc_rec) & (final_alloc > 0)).astype(float),
        'historical_override_zeroed_rec': ((alloc_rec > 0) & (final_alloc <= 0)).astype(float),
        'historical_override_missed_by_rec': ((alloc_rec <= 0) & (final_alloc > 0)).astype(float),
        'historical_override_units_delta': final_units - alloc_rec_units,
        'predicted_quantity_correction_context': safe_div(need_gap, flm) - alloc_rec_units,
    })

    item = _txt(df, cmap, 'item').replace('', '__missing_item__').astype(str)
    site = _txt(df, cmap, 'site').replace('', '__missing_site__').astype(str)
    dept = _txt(df, cmap, 'department_id').replace('', '__missing_dept__').astype(str)
    cls = _txt(df, cmap, 'class_id').replace('', '__missing_class__').astype(str)
    item_site = item + '|' + site
    dept_class = dept + '|' + cls
    item_section = item + '|' + section.astype(str)
    item_site_section = item + '|' + site + '|' + section.astype(str)
    dept_class_section = dept + '|' + cls + '|' + section.astype(str)
    cat['item_site_section'] = item_site_section
    cat['dept_class_section'] = dept_class_section

    group_metrics = {
        'demand': demand_basis,
        'need': need_gap,
        'alloc_rec': alloc_rec,
        'supply': supply,
        'dc': dc_start,
        'recent_velocity': num['recent_velocity_blend'],
        'consensus': num['demand_balanced_consensus'],
    }
    for key, prefix in [
        (item, 'item'), (site, 'site'), (dept, 'department'), (cls, 'class'),
        (item_site, 'item_site'), (dept_class, 'dept_class'),
        (item_section, 'item_section'), (item_site_section, 'item_site_section'), (dept_class_section, 'dept_class_section')
    ]:
        _add_group_features(num, key, prefix, group_metrics)
    for key, prefix in [(item, 'item'), (item_section, 'item_section')]:
        for mname, metric in {
            'need': need_gap,
            'demand': demand_basis,
            'alloc_rec': alloc_rec,
            'dc_before': dc_start,
            'recent_velocity': num['recent_velocity_blend'],
            'demand_acceleration': num['demand_recent_acceleration'],
            'demand_consensus': num['demand_balanced_consensus'],
            'demand_consistency': num['demand_consistency_score'],
        }.items():
            _add_rank_features(num, key, prefix, mname, metric)

    # Build one consolidated dataframe, avoiding repeated insert/fragmentation.
    num_df = pd.DataFrame({f'num__{k}': pd.Series(v, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0) for k, v in num.items()}, index=idx)
    cat_df = pd.DataFrame({f'cat__{k}': pd.Series(v, index=idx).fillna('').astype(str) for k, v in cat.items()}, index=idx)
    return pd.concat([num_df, cat_df], axis=1).copy()

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    cmap = build_column_map(df)
    flm = _num(df, cmap, 'flm', 1).fillna(1).clip(lower=1)
    final_alloc = _num(df, cmap, 'final_alloc', 0).fillna(0).clip(lower=0)
    alloc_rec = _num(df, cmap, 'alloc_rec', 0).fillna(0).clip(lower=0)
    units = np.floor(final_alloc / flm).clip(lower=0).astype(int)
    alloc_rec_units = np.floor(alloc_rec / flm).clip(lower=0).astype(int)
    flag = _txt(df, cmap, 'flag')
    section = _flag_section(flag)
    X = build_feature_frame(df)
    priority = (units > 0).astype(float) * 0.65 + pd.to_numeric(X.get('num__need_to_dc_before', 0), errors='coerce').fillna(0).clip(0, 3) / 3 * 0.25 + pd.to_numeric(X.get('num__demand_consistency_score', 0), errors='coerce').fillna(0).clip(0,1) * 0.10
    correction = (units - alloc_rec_units).clip(-3, 3).astype(int) + 3
    override_any = ((final_alloc != alloc_rec) & ((final_alloc > 0) | (alloc_rec > 0))).astype(int)
    return pd.DataFrame({
        '__target_units': units.astype(int),
        '__target_allocated': (final_alloc > 0).astype(int),
        '__target_priority': priority.astype(float),
        '__section_type': section.astype(str),
        '__target_quantity_correction': correction.astype(int),
        '__target_override_any': override_any.astype(int),
        '__target_override_delta_units': (units - alloc_rec_units).astype(int),
    }, index=df.index)
