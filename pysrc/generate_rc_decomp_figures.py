"""
Generate Risk Contribution Decomposition Figures and Tables

This script generates all the figures and tables for the risk contribution decomposition paper:
- Figure 2: Convergence of RC with inherent and correlation components
- Figure 3: Tornado chart showing inherent and correlation RC decomposition
- Figure 4: Individual asset RC history
- Figures 5, 6, 7: Three-panel plots (RC, inherent RC, correlation RC) for each portfolio
- Tables: portfolio_weights.csv, last_rc_lag_adj_data.csv, last_asset_class_rc_lag_adj_data.csv, last_rc_comparison.csv
"""

import pandas as pd
import numpy as np
import os
import logging
import time
import traceback
import matplotlib.pyplot as plt
from tqdm import tqdm

import helpers as hlp
import risk_model_history as rmh
import risk_calc as rc

hlp.setup_logging('generate_rc_decomp_figures')

# Setup directories
data_dir = os.path.join('..', 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
proc_data_dir = os.path.join(data_dir, 'processed')
output_dir = proc_data_dir

# Create output directory if it doesn't exist
hlp.mkdir(output_dir)

# Track start time
start_time = time.time()

# Parameters
alpha = 0.99
lookback = 126
leverage = 10
roll_code = 'N:05_0_R'

# Asset configuration
tickers = [
    f'CL1 {roll_code} Comdty',
    f'NG1 {roll_code} Comdty',
    f'C 1 {roll_code} Comdty',
    f'ES1 {roll_code} Index',
    f'CF1 {roll_code} Index',
    f'XU1 {roll_code} Index',
    f'TY1 {roll_code} Comdty',
    f'RX1 {roll_code} Comdty',
    f'XM1 {roll_code} Comdty',
    'CADUSD Curncy',
    'EURUSD Curncy',
    'JPYUSD Curncy',
]

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

try:
    # Load price data
    logging.info("Loading price data...")
    price_df = pd.read_csv(os.path.join(raw_data_dir, 'icd_prices.csv'), index_col=0, parse_dates=True)
    price_df.columns = price_df.columns.str.replace(roll_code + ' ', '')
    price_df.columns.name = 'ticker'
    rets_df = price_df.pct_change(1).iloc[1:]
    rets_df = rets_df[rets_df.index <= '2025-08-01']
    logging.info(f"Loaded price data with shape {price_df.shape}")
    logging.info(f"Returns data shape: {rets_df.shape}, date range: {rets_df.index.min()} to {rets_df.index.max()}")

    # Prepare portfolio data
    logging.info("Preparing portfolio data...")
    weights_df = pd.DataFrame({'Asset Class': pd.Series(asset_class_by_ticker)})
    weights_df.index.name = 'Ticker'
    weights_df = weights_df.reset_index()
    weights_df['LS Weight'] = weights_df['Ticker'].map(weights_by_ticker)
    weights_df['Eq Weight'] = 1 / len(weights_df.index)
    rp_weights_by_ticker = 1 / rets_df.iloc[-126:].std()
    rp_weights_by_ticker = rp_weights_by_ticker / rp_weights_by_ticker.sum()
    weights_df['RP Weight'] = weights_df['Ticker'].map(rp_weights_by_ticker)
    logging.info(f"Weights dataframe:\n{weights_df.tail()}")

    # Create stacked returns dataframes
    rets_df_stacked_by_weight = {}
    for weight_name in ['LS', 'Eq', 'RP']:
        rets_df_stacked = pd.DataFrame({'return': rets_df.stack()}).reset_index()
        rets_df_stacked['asset_class'] = rets_df_stacked['ticker'].map(asset_class_by_ticker)
        rets_df_stacked['weight'] = rets_df_stacked['ticker'].map(
            weights_df[['Ticker', weight_name + ' Weight']].set_index('Ticker')[weight_name + ' Weight']
        ) * leverage
        rets_df_stacked['weighted_return'] = rets_df_stacked['weight'] * rets_df_stacked['return']
        rets_df_stacked = rets_df_stacked.set_index(['asset_class', 'ticker', 'date']).sort_index()
        rets_df_stacked_by_weight[weight_name] = rets_df_stacked

    # Calculate risk history for all portfolios
    logging.info("Calculating risk history for LS portfolio...")
    risk_hist_df_ls = rmh.get_risk_metrics_cur_pos(
        rets_df_stacked_by_weight['LS'],
        pd.to_datetime('1990-01-01') + pd.offsets.BDay(70),
        end_date=rets_df.index.max(),
        lookback=lookback,
        halflife=None,
        alpha=alpha,
        yesterday_price_only=False,
        use_dv01_pnl_approx=False,
        risk_column_names=['rc_lag_adj'],
        agg_col=None,
        save_table=False,
        hist_calc_freq_days=1,
        filter_func=None,
        index_cols=['asset_class', 'ticker'],
        non_index_pnl_cols=[],
        pnl_col='weighted_return'
    )
    logging.info(f"LS risk history shape: {risk_hist_df_ls.shape}")

    logging.info("Calculating risk history for Eq portfolio...")
    risk_hist_df_eq = rmh.get_risk_metrics_cur_pos(
        rets_df_stacked_by_weight['Eq'],
        pd.to_datetime('1990-01-01') + pd.offsets.BDay(70),
        end_date=rets_df.index.max(),
        lookback=lookback,
        halflife=None,
        alpha=alpha,
        yesterday_price_only=False,
        use_dv01_pnl_approx=False,
        risk_column_names=['rc_lag_adj'],
        agg_col=None,
        save_table=False,
        hist_calc_freq_days=1,
        filter_func=None,
        index_cols=['asset_class', 'ticker'],
        non_index_pnl_cols=[],
        pnl_col='weighted_return'
    )
    logging.info(f"Eq risk history shape: {risk_hist_df_eq.shape}")

    logging.info("Calculating risk history for RP portfolio...")
    risk_hist_df_rp = rmh.get_risk_metrics_cur_pos(
        rets_df_stacked_by_weight['RP'],
        pd.to_datetime('1990-01-01') + pd.offsets.BDay(70),
        end_date=rets_df.index.max(),
        lookback=lookback,
        halflife=None,
        alpha=alpha,
        yesterday_price_only=False,
        use_dv01_pnl_approx=False,
        risk_column_names=['rc_lag_adj'],
        agg_col=None,
        save_table=False,
        hist_calc_freq_days=1,
        filter_func=None,
        index_cols=['asset_class', 'ticker'],
        non_index_pnl_cols=[],
        pnl_col='weighted_return'
    )
    logging.info(f"RP risk history shape: {risk_hist_df_rp.shape}")

    # Prepare RC distribution data
    logging.info("Preparing RC distribution data...")
    rc_dist_lag_adj_df_stacked = risk_hist_df_ls[['asset_class', 'ticker', 'date', 'rc_lag_adj']].set_index(['asset_class', 'ticker', 'date'])['rc_lag_adj'].copy()
    rc_dist_lag_adj_df = rc_dist_lag_adj_df_stacked.unstack(['asset_class', 'ticker'], sort=False)
    asset_class_rc_dist_lag_adj_df_stacked = risk_hist_df_ls[['asset_class', 'ticker', 'date', 'rc_lag_adj']].groupby(['asset_class', 'date'], sort=False)['rc_lag_adj'].sum().copy()
    asset_class_rc_dist_lag_adj_df = asset_class_rc_dist_lag_adj_df_stacked.unstack('asset_class', sort=False)

    # Equal weight
    rc_dist_lag_adj_df_stacked_eq = risk_hist_df_eq[['asset_class', 'ticker', 'date', 'rc_lag_adj']].set_index(['asset_class', 'ticker', 'date'])['rc_lag_adj'].copy()
    rc_dist_lag_adj_df_eq = rc_dist_lag_adj_df_stacked_eq.unstack(['asset_class', 'ticker'], sort=False)
    asset_class_rc_dist_lag_adj_df_stacked_eq = risk_hist_df_eq[['asset_class', 'ticker', 'date', 'rc_lag_adj']].groupby(['asset_class', 'date'], sort=False)['rc_lag_adj'].sum().copy()
    asset_class_rc_dist_lag_adj_df_eq = asset_class_rc_dist_lag_adj_df_stacked_eq.unstack('asset_class', sort=False)

    # Risk parity
    rc_dist_lag_adj_df_stacked_rp = risk_hist_df_rp[['asset_class', 'ticker', 'date', 'rc_lag_adj']].set_index(['asset_class', 'ticker', 'date'])['rc_lag_adj'].copy()
    rc_dist_lag_adj_df_rp = rc_dist_lag_adj_df_stacked_rp.unstack(['asset_class', 'ticker'], sort=False)
    asset_class_rc_dist_lag_adj_df_stacked_rp = risk_hist_df_rp[['asset_class', 'ticker', 'date', 'rc_lag_adj']].groupby(['asset_class', 'date'], sort=False)['rc_lag_adj'].sum().copy()
    asset_class_rc_dist_lag_adj_df_rp = asset_class_rc_dist_lag_adj_df_stacked_rp.unstack('asset_class', sort=False)

    # Calculate RC decomposition for all portfolios
    logging.info("Calculating RC decomposition for all portfolios...")

    # LS Portfolio decomposition
    ls_rets_df = rets_df_stacked_by_weight['LS']['weighted_return'].unstack(['asset_class', 'ticker'])
    ls_rets_df_lookback = ls_rets_df.iloc[-lookback:].copy()
    icd_inher_srs_ls, icd_corr_srs_ls = rc.get_rc_parts(ls_rets_df_lookback, alpha=alpha, lag_adj=True)
    icd_parts_df_ls = pd.DataFrame({
        'Inherent RC': icd_inher_srs_ls,
        'Correlation RC': icd_corr_srs_ls,
    })
    icd_parts_df_ls['RC'] = icd_parts_df_ls['Inherent RC'] + icd_parts_df_ls['Correlation RC']
    icd_parts_df_display_ls = (icd_parts_df_ls * 100).copy()
    logging.info(f"LS RC decomposition:\n{icd_parts_df_display_ls.tail()}")

    # Eq Portfolio decomposition
    eq_rets_df = rets_df_stacked_by_weight['Eq']['weighted_return'].unstack(['asset_class', 'ticker'])
    eq_rets_df_lookback = eq_rets_df.iloc[-lookback:].copy()
    icd_inher_srs_eq, icd_corr_srs_eq = rc.get_rc_parts(eq_rets_df_lookback, alpha=alpha, lag_adj=True)
    icd_parts_df_eq = pd.DataFrame({
        'Inherent RC': icd_inher_srs_eq,
        'Correlation RC': icd_corr_srs_eq,
    })
    icd_parts_df_eq['RC'] = icd_parts_df_eq['Inherent RC'] + icd_parts_df_eq['Correlation RC']
    icd_parts_df_display_eq = (icd_parts_df_eq * 100).copy()
    logging.info(f"Eq RC decomposition:\n{icd_parts_df_display_eq.tail()}")

    # RP Portfolio decomposition
    rp_rets_df = rets_df_stacked_by_weight['RP']['weighted_return'].unstack(['asset_class', 'ticker'])
    rp_rets_df_lookback = rp_rets_df.iloc[-lookback:].copy()
    icd_inher_srs_rp, icd_corr_srs_rp = rc.get_rc_parts(rp_rets_df_lookback, alpha=alpha, lag_adj=True)
    icd_parts_df_rp = pd.DataFrame({
        'Inherent RC': icd_inher_srs_rp,
        'Correlation RC': icd_corr_srs_rp,
    })
    icd_parts_df_rp['RC'] = icd_parts_df_rp['Inherent RC'] + icd_parts_df_rp['Correlation RC']
    icd_parts_df_display_rp = (icd_parts_df_rp * 100).copy()
    logging.info(f"RP RC decomposition:\n{icd_parts_df_display_rp.tail()}")

    # Keep backward compatibility aliases for LS
    icd_inher_srs = icd_inher_srs_ls
    icd_corr_srs = icd_corr_srs_ls
    icd_parts_df = icd_parts_df_ls
    icd_parts_df_display = icd_parts_df_display_ls

    # Sort by RC for display
    icd_parts_df_display_sorted = icd_parts_df_display.sort_values('RC', ascending=True)

    # ============================================================================
    # FIGURE 2: Convergence of RC with inherent and correlation components
    # ============================================================================
    logging.info("Generating Figure 2 (RC Convergence)...")

    # Simulate convergence using synthetic data (matching notebook approach)
    n_samples = 500
    n_dim = 5

    mean = np.zeros(n_dim)
    cov = np.full((n_dim, n_dim), 0.75)
    np.fill_diagonal(cov, 1)

    np.random.seed(0)  # For reproducibility
    full_samp_df = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=n_samples),
                                columns=[f'Asset_{i}' for i in range(1, n_dim+1)])

    # Calculate RC with expanding window (matching notebook: alpha=1)
    rc_conv_df = pd.DataFrame(columns=full_samp_df.columns)
    inherent_conv_df = pd.DataFrame(columns=full_samp_df.columns)
    correlation_conv_df = pd.DataFrame(columns=full_samp_df.columns)

    for i in tqdm(range(5, n_samples), desc="Calculating convergence"):
        cur_full_samp_df = full_samp_df.iloc[:i]
        try:
            # RC using additive method (matching notebook)
            cur_rc = rc.get_rc(cur_full_samp_df, alpha=1)
            rc_conv_df.loc[i] = cur_rc

            # Decomposition into inherent and correlation
            cur_inher, cur_corr = rc.get_rc_parts(cur_full_samp_df, alpha=1, lag_adj=False)
            inherent_conv_df.loc[i] = pd.Series(cur_inher).values
            correlation_conv_df.loc[i] = pd.Series(cur_corr).values
        except:
            continue

    # Calculate theoretical population targets for reference lines
    # Formula from notebook: target = np.sqrt(cov.sum().sum()) / n_dim
    # For symmetric cov with var=1, corr=rho: sum(cov) = n + n*(n-1)*rho = 5 + 5*4*0.75 = 20
    # sigma_p = sqrt(20) = 4.472, target = 4.472/5 = 0.894 per asset
    target = np.sqrt(cov.sum().sum()) / n_dim

    # Inherent RC per asset: w_i^2 * var_i / sigma_p = 1 * 1 / sigma_p
    sigma_p = np.sqrt(cov.sum().sum())
    inherent_target = 1 / sigma_p

    # Correlation RC per asset: Total RC - Inherent RC
    corr_target = target - inherent_target

    # Create 2x2 grid convergence figure
    # Top left: RC, Top right: Inherent RC, Bottom left: empty, Bottom right: Correlation RC
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Total RC convergence (stacked area plot)
    ax1 = axes[0, 0]
    rc_conv_df.plot.area(ax=ax1, alpha=0.8, xlim=(rc_conv_df.index.min(), rc_conv_df.index.max()))
    ax1.set_title('RC Convergence for 5 Synthetic Assets', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total RC', fontsize=11)
    ax1.set_xlabel('Sample Count', fontsize=11)
    for i in range(n_dim):
        if i == 0:
            ax1.axhline(y=target*(i+1), color='black', linestyle='--', label='Population RC')
        else:
            ax1.axhline(y=target*(i+1), color='black', linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)

    # Top right: Inherent RC convergence (stacked area plot)
    ax2 = axes[0, 1]
    inherent_conv_df.plot.area(ax=ax2, alpha=0.8, xlim=(inherent_conv_df.index.min(), inherent_conv_df.index.max()))
    ax2.set_title('Inherent RC Convergence for 5 Synthetic Assets', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Inherent RC', fontsize=11)
    ax2.set_xlabel('Sample Count', fontsize=11)
    for i in range(n_dim):
        if i == 0:
            ax2.axhline(y=inherent_target*(i+1), color='black', linestyle='--', label='Population Inherent RC')
        else:
            ax2.axhline(y=inherent_target*(i+1), color='black', linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)

    # Bottom left: Empty
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Bottom right: Correlation RC convergence (stacked area plot)
    ax4 = axes[1, 1]
    correlation_conv_df.plot.area(ax=ax4, alpha=0.8, xlim=(correlation_conv_df.index.min(), correlation_conv_df.index.max()))
    ax4.set_title('Correlation RC Convergence for 5 Synthetic Assets', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Total Correlation RC', fontsize=11)
    ax4.set_xlabel('Sample Count', fontsize=11)
    for i in range(n_dim):
        if i == 0:
            ax4.axhline(y=corr_target*(i+1), color='black', linestyle='--', label='Population Correlation RC')
        else:
            ax4.axhline(y=corr_target*(i+1), color='black', linestyle='--')
    ax4.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rc_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {os.path.join(output_dir, 'rc_convergence.png')}")

    # ============================================================================
    # FIGURE 3: Tornado chart (Inherent and Correlation RC bar charts)
    # ============================================================================
    logging.info("Generating Figure 3 (RC Tornado/Decomposition)...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Inherent RC chart
    inherent_data = icd_parts_df_display_sorted['Inherent RC'].droplevel('asset_class')
    ax1.barh(range(len(inherent_data)), inherent_data.values, color='cornflowerblue')
    ax1.set_yticks(range(len(inherent_data)))
    ax1.set_yticklabels(inherent_data.index)
    ax1.set_xlabel('Inherent Risk Contribution (%)')
    ax1.set_title('Inherent Risk Contribution by Asset', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)

    # Correlation RC chart
    corr_data = icd_parts_df_display_sorted['Correlation RC'].droplevel('asset_class')
    colors = ['red' if x < 0 else 'orange' for x in corr_data.values]
    ax2.barh(range(len(corr_data)), corr_data.values, color=colors)
    ax2.set_yticks(range(len(corr_data)))
    ax2.set_yticklabels(corr_data.index)
    ax2.set_xlabel('Correlation Risk Contribution (%)')
    ax2.set_title('Correlation Risk Contribution by Asset', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rc_tornado.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {os.path.join(output_dir, 'rc_tornado.png')}")

    # ============================================================================
    # FIGURE 4: Individual Asset RC History
    # ============================================================================
    logging.info("Generating Figure 4 (Individual Asset RC History)...")

    fig, ax = plt.subplots(figsize=(12, 7))
    rc_dist_lag_adj_df_pct = rc_dist_lag_adj_df.droplevel('asset_class', axis=1) * 100

    for col in rc_dist_lag_adj_df_pct.columns:
        ax.plot(rc_dist_lag_adj_df_pct.index, rc_dist_lag_adj_df_pct[col], label=col, alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RC (%)', fontsize=12)
    ax.set_title('Risk Contribution by Asset Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xlim(rc_dist_lag_adj_df_pct.index.min(), rc_dist_lag_adj_df_pct.index.max())
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ls_rc_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {os.path.join(output_dir, 'ls_rc_history.png')}")

    # ============================================================================
    # Helper function to calculate RC decomposition over time (OPTIMIZED)
    # ============================================================================
    def calculate_rc_decomp_history(rets_df_stacked, lookback, alpha, calc_freq_days=5):
        """
        Calculate inherent and correlation RC over time - OPTIMIZED VERSION

        Parameters:
        -----------
        calc_freq_days : int
            Calculate decomposition every N days (default 5) to reduce computation time
        """
        # Pre-unstack the data once (much faster than doing it in the loop)
        logging.info("  Preparing data for efficient calculation...")
        rets_unstacked_full = rets_df_stacked['weighted_return'].unstack(['asset_class', 'ticker'])

        # Get available dates and sample them
        all_dates = rets_unstacked_full.index
        dates_to_calc = all_dates[lookback-1::calc_freq_days]  # Calculate every calc_freq_days, starting after lookback

        inherent_history = []
        correlation_history = []

        logging.info(f"  Calculating decomposition for {len(dates_to_calc)} time points (every {calc_freq_days} days)...")
        for date in tqdm(dates_to_calc, desc="Calculating RC decomposition"):
            # Get the position of this date in the full dataset
            date_pos = all_dates.get_loc(date)

            # Slice the lookback window efficiently using position
            start_pos = max(0, date_pos - lookback + 1)
            rets_window = rets_unstacked_full.iloc[start_pos:date_pos+1]

            if len(rets_window) < lookback:
                continue

            try:
                # Calculate decomposition
                icd_inher, icd_corr = rc.get_rc_parts(rets_window, alpha=alpha, lag_adj=True)

                # Aggregate by asset class
                icd_inher_by_class = pd.Series(icd_inher).groupby(level='asset_class').sum()
                icd_corr_by_class = pd.Series(icd_corr).groupby(level='asset_class').sum()

                inherent_history.append({'date': date, **icd_inher_by_class.to_dict()})
                correlation_history.append({'date': date, **icd_corr_by_class.to_dict()})
            except Exception as e:
                logging.warning(f"  Error at date {date}: {str(e)}")
                logging.debug(traceback.format_exc())
                continue

        inherent_df = pd.DataFrame(inherent_history).set_index('date')
        correlation_df = pd.DataFrame(correlation_history).set_index('date')

        return inherent_df, correlation_df

    # ============================================================================
    # FIGURES 5, 6, 7: Three-panel plots for each portfolio
    # ============================================================================
    portfolios_config = [
        ('LS', risk_hist_df_ls, rets_df_stacked_by_weight['LS'], 'Long-Short', 'ls'),
        ('Eq', risk_hist_df_eq, rets_df_stacked_by_weight['Eq'], 'Equal-Weight', 'eq'),
        ('RP', risk_hist_df_rp, rets_df_stacked_by_weight['RP'], 'Risk Parity', 'rp'),
    ]

    figure_numbers = [5, 6, 7]

    for (port_code, risk_hist_df, rets_stacked, port_name, port_file_code), fig_num in zip(portfolios_config, figure_numbers):
        logging.info(f"Generating Figure {fig_num} ({port_name} Portfolio)...")

        # Calculate RC by asset class over time
        asset_class_rc = risk_hist_df[['asset_class', 'ticker', 'date', 'rc_lag_adj']].groupby(['asset_class', 'date'], sort=False)['rc_lag_adj'].sum().copy()
        asset_class_rc_df = asset_class_rc.unstack('asset_class', sort=False) * 100

        # Calculate decomposition history
        logging.info(f"  Calculating RC decomposition history for {port_name}...")
        inherent_df, correlation_df = calculate_rc_decomp_history(rets_stacked, lookback, alpha)
        inherent_df = inherent_df * 100
        correlation_df = correlation_df * 100
        logging.info(f"  Inherent RC history shape: {inherent_df.shape}")
        logging.info(f"  Correlation RC history shape: {correlation_df.shape}")

        # Align columns
        common_cols = sorted(set(asset_class_rc_df.columns) & set(inherent_df.columns) & set(correlation_df.columns))
        asset_class_rc_df = asset_class_rc_df[common_cols]
        inherent_df = inherent_df[common_cols]
        correlation_df = correlation_df[common_cols]

        # Create three-panel figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))

        # Top panel: Total RC - title is just "Portfolio Name - Risk Contribution"
        ax1 = axes[0]
        for col in asset_class_rc_df.columns:
            ax1.plot(asset_class_rc_df.index, asset_class_rc_df[col], label=col, alpha=0.8, linewidth=1.5)
        ax1.set_title(f'{port_name} Portfolio - Risk Contribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RC (%)', fontsize=11)
        ax1.set_xlabel('')
        ax1.set_xlim(asset_class_rc_df.index.min(), asset_class_rc_df.index.max())
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

        # Middle panel: Inherent RC
        ax2 = axes[1]
        for col in inherent_df.columns:
            ax2.plot(inherent_df.index, inherent_df[col], label=col, alpha=0.8, linewidth=1.5)
        ax2.set_title(f'{port_name} Portfolio - Inherent Risk Contribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Inherent RC (%)', fontsize=11)
        ax2.set_xlabel('')
        ax2.set_xlim(inherent_df.index.min(), inherent_df.index.max())
        ax2.grid(alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

        # Bottom panel: Correlation RC
        ax3 = axes[2]
        for col in correlation_df.columns:
            ax3.plot(correlation_df.index, correlation_df[col], label=col, alpha=0.8, linewidth=1.5)
        ax3.set_title(f'{port_name} Portfolio - Correlation Risk Contribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Correlation RC (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_xlim(correlation_df.index.min(), correlation_df.index.max())
        ax3.grid(alpha=0.3)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{port_file_code}_rc_history_asset_group.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"  Saved: {output_path}")

    # ============================================================================
    # TABLES: Generate all required tables
    # ============================================================================
    logging.info("Generating tables...")

    # Table 1: Portfolio Weights
    logging.info("  Generating portfolio_weights.csv...")
    portfolio_weights_df = weights_df[['Asset Class', 'Ticker', 'LS Weight', 'Eq Weight', 'RP Weight']].copy()
    portfolio_weights_df.to_csv(os.path.join(output_dir, 'portfolio_weights.csv'), index=False)
    logging.info(f"  Saved: {os.path.join(output_dir, 'portfolio_weights.csv')}")

    # Table 2: RC by Asset (last_rc_lag_adj_data.csv)
    logging.info("  Generating last_rc_lag_adj_data.csv...")

    # Get RC for each portfolio at the last date
    last_date = rets_df.index.max()

    # LS RC
    ls_rc_last = risk_hist_df_ls[risk_hist_df_ls['date'] == last_date][['asset_class', 'ticker', 'rc_lag_adj']].copy()
    ls_rc_last = ls_rc_last.rename(columns={'rc_lag_adj': 'LS RC'})
    ls_rc_last['LS RC'] = ls_rc_last['LS RC'] * 100

    # Add LS Inherent and Correlation RC from decomposition
    ls_decomp = icd_parts_df_display_ls.reset_index()
    ls_decomp.columns = ['asset_class', 'ticker', 'LS Inh', 'LS Corr', 'LS RC_check']
    ls_rc_last = ls_rc_last.merge(ls_decomp[['ticker', 'LS Inh', 'LS Corr']], on='ticker')

    # Eq RC
    eq_rc_last = risk_hist_df_eq[risk_hist_df_eq['date'] == last_date][['asset_class', 'ticker', 'rc_lag_adj']].copy()
    eq_rc_last = eq_rc_last.rename(columns={'rc_lag_adj': 'Eq RC'})
    eq_rc_last['Eq RC'] = eq_rc_last['Eq RC'] * 100

    # Add Eq Inherent and Correlation RC from decomposition
    eq_decomp = icd_parts_df_display_eq.reset_index()
    eq_decomp.columns = ['asset_class', 'ticker', 'Eq Inh', 'Eq Corr', 'Eq RC_check']
    eq_rc_last = eq_rc_last.merge(eq_decomp[['ticker', 'Eq Inh', 'Eq Corr']], on='ticker')

    # RP RC
    rp_rc_last = risk_hist_df_rp[risk_hist_df_rp['date'] == last_date][['asset_class', 'ticker', 'rc_lag_adj']].copy()
    rp_rc_last = rp_rc_last.rename(columns={'rc_lag_adj': 'RP RC'})
    rp_rc_last['RP RC'] = rp_rc_last['RP RC'] * 100

    # Add RP Inherent and Correlation RC from decomposition
    rp_decomp = icd_parts_df_display_rp.reset_index()
    rp_decomp.columns = ['asset_class', 'ticker', 'RP Inh', 'RP Corr', 'RP RC_check']
    rp_rc_last = rp_rc_last.merge(rp_decomp[['ticker', 'RP Inh', 'RP Corr']], on='ticker')

    # Merge all portfolios
    rc_by_asset = ls_rc_last.merge(eq_rc_last[['ticker', 'Eq RC', 'Eq Inh', 'Eq Corr']], on='ticker')
    rc_by_asset = rc_by_asset.merge(rp_rc_last[['ticker', 'RP RC', 'RP Inh', 'RP Corr']], on='ticker')

    # Reorder columns: asset_class, ticker, then for each portfolio: RC, Inh, Corr
    rc_by_asset = rc_by_asset[['asset_class', 'ticker',
                               'LS RC', 'LS Inh', 'LS Corr',
                               'Eq RC', 'Eq Inh', 'Eq Corr',
                               'RP RC', 'RP Inh', 'RP Corr']]
    rc_by_asset.to_csv(os.path.join(output_dir, 'last_rc_lag_adj_data.csv'), index=False)
    logging.info(f"  Saved: {os.path.join(output_dir, 'last_rc_lag_adj_data.csv')}")

    # Table 3: RC by Asset Class (last_asset_class_rc_lag_adj_data.csv)
    logging.info("  Generating last_asset_class_rc_lag_adj_data.csv...")

    # Aggregate all RC columns by asset class
    agg_cols = ['LS RC', 'LS Inh', 'LS Corr', 'Eq RC', 'Eq Inh', 'Eq Corr', 'RP RC', 'RP Inh', 'RP Corr']
    rc_by_asset_class = rc_by_asset.groupby('asset_class')[agg_cols].sum().reset_index()

    # Add total row
    total_row = pd.DataFrame({
        'asset_class': ['Total'],
        'LS RC': [rc_by_asset_class['LS RC'].sum()],
        'LS Inh': [rc_by_asset_class['LS Inh'].sum()],
        'LS Corr': [rc_by_asset_class['LS Corr'].sum()],
        'Eq RC': [rc_by_asset_class['Eq RC'].sum()],
        'Eq Inh': [rc_by_asset_class['Eq Inh'].sum()],
        'Eq Corr': [rc_by_asset_class['Eq Corr'].sum()],
        'RP RC': [rc_by_asset_class['RP RC'].sum()],
        'RP Inh': [rc_by_asset_class['RP Inh'].sum()],
        'RP Corr': [rc_by_asset_class['RP Corr'].sum()]
    })
    rc_by_asset_class = pd.concat([rc_by_asset_class, total_row], ignore_index=True)
    rc_by_asset_class.to_csv(os.path.join(output_dir, 'last_asset_class_rc_lag_adj_data.csv'), index=False)
    logging.info(f"  Saved: {os.path.join(output_dir, 'last_asset_class_rc_lag_adj_data.csv')}")

    # Table 4: RC Comparison (last_rc_comparison.csv)
    logging.info("  Generating last_rc_comparison.csv...")

    # Get decomposition for LS portfolio
    rc_comparison = icd_parts_df_display.copy()
    rc_comparison = rc_comparison.droplevel('asset_class')
    rc_comparison = rc_comparison.reset_index()
    rc_comparison = rc_comparison.rename(columns={
        'ticker': 'ticker',
        'RC': 'RC',
        'Inherent RC': 'Inh. RC',
        'Correlation RC': 'Corr. RC'
    })

    # Compute non-additive iVol (paper eq. 1) per asset for the LS portfolio.
    # iVol(a) = sigma_p - sigma_{p\{a}}; it does NOT sum to portfolio volatility,
    # so its values differ from the additive RC computed above.
    ivol_srs_ls = rc.get_ivol(ls_rets_df_lookback, alpha=alpha, lag_adj=True) * 100
    ivol_by_ticker = ivol_srs_ls.droplevel('asset_class') if isinstance(ivol_srs_ls.index, pd.MultiIndex) else ivol_srs_ls
    rc_comparison['iVol'] = rc_comparison['ticker'].map(ivol_by_ticker)

    # Add MDD placeholder (would need actual drawdown calculation)
    # For now, use a simple rolling max drawdown approximation
    mdd_dict = {}
    for ticker in rc_comparison['ticker']:
        if ticker in rets_df.columns:
            cumret = (1 + rets_df[ticker]).cumprod()
            rolling_max = cumret.cummax()
            drawdown = (cumret - rolling_max) / rolling_max
            mdd_dict[ticker] = drawdown.min() * 100
        else:
            mdd_dict[ticker] = 0
    rc_comparison['MDD'] = rc_comparison['ticker'].map(mdd_dict)

    # Reorder columns to match paper Table 4: Ticker, RC, Inherent RC, Correlation RC, iVol, MDD
    rc_comparison = rc_comparison[['ticker', 'RC', 'Inh. RC', 'Corr. RC', 'iVol', 'MDD']]
    rc_comparison.to_csv(os.path.join(output_dir, 'last_rc_comparison.csv'), index=False)
    logging.info(f"  Saved: {os.path.join(output_dir, 'last_rc_comparison.csv')}")

    logging.info("=" * 80)
    logging.info("All figures and tables generated successfully!")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 80)
    hlp.print_completed_time(start_time, "Figure and table generation")

except Exception as e:
    logging.error(f"Error during execution: {str(e)}")
    logging.error(traceback.format_exc())
    raise
