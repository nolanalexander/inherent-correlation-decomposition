"""
Regenerate the four CSV tables that generate_rc_decomp_figures.py emits:
  - portfolio_weights.csv
  - last_rc_lag_adj_data.csv
  - last_asset_class_rc_lag_adj_data.csv
  - last_rc_comparison.csv

This focused script computes only the last-date values needed for the tables
(via the renamed risk_calc functions: get_rc, get_rc_parts, get_ivol), so it
finishes in seconds instead of the ~18 min the full figure-generation script
takes.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import risk_calc as rc


alpha = 0.99
lookback = 126
leverage = 10
roll_code = 'N:05_0_R'

weights_by_ticker = {
    'CL1 Comdty': -0.05,
    'NG1 Comdty': 0.05,
    'C 1 Comdty': -0.1,
    'ES1 Index': 0.05,
    'CF1 Index': -0.2,
    'XU1 Index': 0.1,
    'TY1 Comdty': 0.9,
    'RX1 Comdty': -0.9,
    'XM1 Comdty': 0.8,
    'CADUSD Curncy': 0.1,
    'EURUSD Curncy': 0.15,
    'JPYUSD Curncy': 0.1,
}

asset_class_by_ticker = {
    'CL1 Comdty': 'Commodity',
    'NG1 Comdty': 'Commodity',
    'C 1 Comdty': 'Commodity',
    'ES1 Index': 'Equity',
    'CF1 Index': 'Equity',
    'XU1 Index': 'Equity',
    'TY1 Comdty': 'Fixed Income',
    'RX1 Comdty': 'Fixed Income',
    'XM1 Comdty': 'Fixed Income',
    'CADUSD Curncy': 'FX',
    'EURUSD Curncy': 'FX',
    'JPYUSD Curncy': 'FX',
}

data_dir = os.path.join('..', 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
proc_data_dir = os.path.join(data_dir, 'processed')

# Load returns
price_df = pd.read_csv(os.path.join(raw_data_dir, 'icd_prices.csv'), index_col=0, parse_dates=True)
price_df.columns = price_df.columns.str.replace(roll_code + ' ', '')
price_df.columns.name = 'ticker'
rets_df = price_df.pct_change(1).iloc[1:]
rets_df = rets_df[rets_df.index <= '2025-08-01']

# Build the weights table (Table 1 / portfolio_weights.csv).
# RP weights are 1/sigma normalised over the most recent `lookback` days.
weights_df = pd.DataFrame({'Asset Class': pd.Series(asset_class_by_ticker)})
weights_df.index.name = 'Ticker'
weights_df = weights_df.reset_index()
weights_df['LS Weight'] = weights_df['Ticker'].map(weights_by_ticker)
weights_df['Eq Weight'] = 1 / len(weights_df.index)
rp_weights_by_ticker = 1 / rets_df.iloc[-lookback:].std()
rp_weights_by_ticker = rp_weights_by_ticker / rp_weights_by_ticker.sum()
weights_df['RP Weight'] = weights_df['Ticker'].map(rp_weights_by_ticker)


def build_weighted_rets(weight_col_name):
    """Build a per-asset weighted-PnL DataFrame keyed by (asset_class, ticker)."""
    weight_map = weights_df.set_index('Ticker')[weight_col_name]
    stacked_df = pd.DataFrame({'return': rets_df.stack()}).reset_index()
    stacked_df['asset_class'] = stacked_df['ticker'].map(asset_class_by_ticker)
    stacked_df['weight'] = stacked_df['ticker'].map(weight_map) * leverage
    stacked_df['weighted_return'] = stacked_df['weight'] * stacked_df['return']
    stacked_df = stacked_df.set_index(['asset_class', 'ticker', 'date']).sort_index()
    return stacked_df['weighted_return'].unstack(['asset_class', 'ticker'])


def last_date_rc(weight_col_name):
    """Compute (RC, Inh RC, Corr RC) per asset for the last `lookback` days, in %."""
    weighted_rets_df = build_weighted_rets(weight_col_name)
    lookback_df = weighted_rets_df.iloc[-lookback:].copy()
    rc_inh_srs, rc_corr_srs = rc.get_rc_parts(lookback_df, alpha=alpha, lag_adj=True)
    rc_inh_srs = rc_inh_srs * 100
    rc_corr_srs = rc_corr_srs * 100
    rc_total_srs = rc_inh_srs + rc_corr_srs
    return rc_total_srs, rc_inh_srs, rc_corr_srs, lookback_df


# Compute per-portfolio RC decomposition at the last date.
ls_rc_srs, ls_inh_srs, ls_corr_srs, ls_lookback_df = last_date_rc('LS Weight')
eq_rc_srs, eq_inh_srs, eq_corr_srs, _ = last_date_rc('Eq Weight')
rp_rc_srs, rp_inh_srs, rp_corr_srs, _ = last_date_rc('RP Weight')

# Table 1: portfolio_weights.csv
out_weights_df = weights_df[['Asset Class', 'Ticker', 'LS Weight', 'Eq Weight', 'RP Weight']].copy()
out_weights_path = os.path.join(proc_data_dir, 'portfolio_weights.csv')
out_weights_df.to_csv(out_weights_path, index=False)
print(f'Wrote {out_weights_path}')

# Table 2: last_rc_lag_adj_data.csv — RC decomposition per asset for LS, Eq, RP
def to_df(rc_srs, inh_srs, corr_srs, prefix):
    df = pd.DataFrame({
        f'{prefix} RC': rc_srs,
        f'{prefix} Inh': inh_srs,
        f'{prefix} Corr': corr_srs,
    }).reset_index()
    return df

ls_df = to_df(ls_rc_srs, ls_inh_srs, ls_corr_srs, 'LS')
eq_df = to_df(eq_rc_srs, eq_inh_srs, eq_corr_srs, 'Eq')[['ticker', 'Eq RC', 'Eq Inh', 'Eq Corr']]
rp_df = to_df(rp_rc_srs, rp_inh_srs, rp_corr_srs, 'RP')[['ticker', 'RP RC', 'RP Inh', 'RP Corr']]

rc_by_asset_df = ls_df.merge(eq_df, on='ticker').merge(rp_df, on='ticker')
rc_by_asset_df = rc_by_asset_df[['asset_class', 'ticker',
                                 'LS RC', 'LS Inh', 'LS Corr',
                                 'Eq RC', 'Eq Inh', 'Eq Corr',
                                 'RP RC', 'RP Inh', 'RP Corr']]
out_rc_by_asset_path = os.path.join(proc_data_dir, 'last_rc_lag_adj_data.csv')
rc_by_asset_df.to_csv(out_rc_by_asset_path, index=False)
print(f'Wrote {out_rc_by_asset_path}')

# Table 3: last_asset_class_rc_lag_adj_data.csv — same, rolled up by asset class
agg_cols = ['LS RC', 'LS Inh', 'LS Corr', 'Eq RC', 'Eq Inh', 'Eq Corr', 'RP RC', 'RP Inh', 'RP Corr']
rc_by_asset_class_df = rc_by_asset_df.groupby('asset_class')[agg_cols].sum().reset_index()
total_row_df = pd.DataFrame([['Total'] + rc_by_asset_class_df[agg_cols].sum().tolist()],
                            columns=['asset_class'] + agg_cols)
rc_by_asset_class_df = pd.concat([rc_by_asset_class_df, total_row_df], ignore_index=True)
out_rc_by_class_path = os.path.join(proc_data_dir, 'last_asset_class_rc_lag_adj_data.csv')
rc_by_asset_class_df.to_csv(out_rc_by_class_path, index=False)
print(f'Wrote {out_rc_by_class_path}')

# Table 4: last_rc_comparison.csv — LS portfolio RC vs non-additive iVol vs MDD
ivol_srs_ls = rc.get_ivol(ls_lookback_df, alpha=alpha, lag_adj=True) * 100
comparison_df = pd.DataFrame({
    'RC': ls_rc_srs,
    'Inh. RC': ls_inh_srs,
    'Corr. RC': ls_corr_srs,
    'iVol': ivol_srs_ls,
}).droplevel('asset_class').reset_index()

mdd_dict = {}
for ticker in comparison_df['ticker']:
    if ticker in rets_df.columns:
        cumret = (1 + rets_df[ticker]).cumprod()
        drawdown = (cumret - cumret.cummax()) / cumret.cummax()
        mdd_dict[ticker] = drawdown.min() * 100
    else:
        mdd_dict[ticker] = 0
comparison_df['MDD'] = comparison_df['ticker'].map(mdd_dict)
comparison_df = comparison_df[['ticker', 'RC', 'Inh. RC', 'Corr. RC', 'iVol', 'MDD']]

out_comparison_path = os.path.join(proc_data_dir, 'last_rc_comparison.csv')
comparison_df.to_csv(out_comparison_path, index=False)
print(f'Wrote {out_comparison_path}')

print()
print('=== LS Comparison Table ===')
print(comparison_df.to_string(index=False))
print()
print(f"Sum of RC (≈ portfolio volatility, additive): {comparison_df['RC'].sum():.6f}")
print(f"Sum of iVol (NOT equal — non-additive):       {comparison_df['iVol'].sum():.6f}")
print(f"Max |RC − iVol| across assets:                {(comparison_df['RC'] - comparison_df['iVol']).abs().max():.6f}")
