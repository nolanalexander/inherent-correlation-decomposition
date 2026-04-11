"""
Minimal risk model history functions for figure generation only.
Contains only the functions needed to generate RC decomposition figures.
"""

import pandas as pd
import logging
from tqdm import tqdm

import risk_calc


def get_risk_metrics(pos_pnl_calc_df, position_date, lookback, run_date=None, halflife=None, alpha=None, yesterday_price_only=True,
                     use_dv01_pnl_approx=False, minimal=False, agg_col=None,
                     pos_pnl_df=None, use_rank_corr=False, index_cols=[], date_col='date',
                     pnl_col='pnl_usd'):
    """
    Calculate risk metrics for a portfolio at a specific date.
    For figure generation, this is always called with minimal=True.
    """
    logging.info('begin get_risk_metrics()')

    pos_pnl_calc_df = pos_pnl_calc_df.copy()

    if use_dv01_pnl_approx and 'pnl_usd_dv01_approx' in pos_pnl_calc_df.columns:
        pnl_col = 'pnl_usd_dv01_approx'
    pos_pnl_calc_df[pnl_col] = pos_pnl_calc_df[pnl_col].astype(float)

    if yesterday_price_only:
        yesterday_date = pd.to_datetime(position_date) - pd.offsets.BDay(1)
        pos_pnl_calc_df = pos_pnl_calc_df[pos_pnl_calc_df.index.get_level_values(date_col) <= yesterday_date]
        pos_pnl_calc_df = pos_pnl_calc_df.sort_index()

    if pos_pnl_df is None:
        if agg_col is not None:
            pos_pnl_calc_df = pos_pnl_calc_df.set_index(index_cols)
            pos_pnl_calc_df = pos_pnl_calc_df[[pnl_col]]
            pos_pnl_calc_df = pos_pnl_calc_df.groupby([agg_col, date_col]).sum()
            pos_pnl_df = pos_pnl_calc_df[pnl_col].unstack([agg_col]).sort_index()
        else:
            pos_pnl_df = pos_pnl_calc_df[pnl_col].unstack(index_cols).sort_index()

    if pos_pnl_df.iloc[0].isna().all():
        pos_pnl_df = pos_pnl_df.iloc[1:]

    if run_date is not None:
        pos_pnl_df = pos_pnl_df[pos_pnl_df.index <= run_date]

    if lookback is not None:
        pos_pnl_df = pos_pnl_df.iloc[-lookback:]

    # For figure generation, we only use minimal=True
    if minimal:
        weighted_ivols_lag_adj = risk_calc.get_additive_ivol(pos_pnl_df, halflife=halflife, alpha=alpha, lead_adj=False, lag_adj=True, use_rank_corr=use_rank_corr)
        risk_df = pd.DataFrame({
            'ivol_lag_adj' : weighted_ivols_lag_adj,
        })
    else:
        # This branch is not used for figure generation
        raise NotImplementedError("Only minimal=True is supported in this streamlined version")

    logging.info('resetting index for risk df')
    if agg_col is not None:
        risk_df = risk_df.reset_index().rename(columns={agg_col: agg_col})
    else:
        risk_df = risk_df.reset_index()

    return risk_df


def get_risk_metrics_cur_pos(pos_pnl_calc_df, start_date, end_date='today', lookback=126,
                                     halflife=None, alpha=0.99,
                                     yesterday_price_only=True,
                                     use_dv01_pnl_approx=False,
                                     risk_column_names=['ivol_lag_adj'],
                                     agg_col=None,
                                     hist_calc_freq_days=1, filter_func=None,
                                     index_cols=[], non_index_pnl_cols=[],
                                     pnl_col='pnl_usd'):
    """
    Calculate historical risk metrics using the current day's positions with past price data.
    """
    risk_hist_df = pd.DataFrame()
    pbar = tqdm(pd.bdate_range(start_date, end_date)[::hist_calc_freq_days])
    updated_at = pd.to_datetime('now')

    if filter_func is not None:
        pos_pnl_calc_df = filter_func(pos_pnl_calc_df)

    if use_dv01_pnl_approx and 'pnl_usd_dv01_approx' in pos_pnl_calc_df.columns:
        pnl_col = 'pnl_usd_dv01_approx'

    if agg_col is not None:
        # Note: position_groupings import removed, assuming data already has required columns
        pos_pnl_calc_df = pos_pnl_calc_df[[agg_col] + non_index_pnl_cols + [pnl_col]]
        pos_pnl_calc_df[non_index_pnl_cols + [pnl_col]] = pos_pnl_calc_df[non_index_pnl_cols + [pnl_col]].astype(float)
        pos_pnl_calc_df = pos_pnl_calc_df.groupby([agg_col, 'date']).sum()
        pos_pnl_df = pos_pnl_calc_df[pnl_col].unstack([agg_col]).sort_index()
    else:
        pos_pnl_df = pos_pnl_calc_df[pnl_col].unstack(index_cols).sort_index().fillna(0)

    table_name = 'risk_metrics_hist_cur_pos'
    if agg_col is not None:
        table_name += '_' + agg_col

    for date in pbar:
        pbar.set_description(f'{date.date()}')
        cur_risk_df = get_risk_metrics(pos_pnl_calc_df, end_date, lookback, run_date=date, halflife=halflife, alpha=alpha,
                                       yesterday_price_only=yesterday_price_only,
                                       use_dv01_pnl_approx=use_dv01_pnl_approx,
                                       agg_col=agg_col, minimal=True,
                                       pos_pnl_df=pos_pnl_df, pnl_col=pnl_col)
        cur_risk_df['date'] = date
        cur_risk_df['updated_at'] = updated_at
        id_cols = index_cols if (agg_col is None) else [agg_col]
        cur_risk_df = cur_risk_df[id_cols + ['date', 'updated_at'] + risk_column_names]

        if cur_risk_df.empty:
            raise ValueError(f'cur_risk_df is empty on {date}')

        risk_hist_df = pd.concat([risk_hist_df, cur_risk_df])

    return risk_hist_df

def get_risk_metrics_historical_pos(pos_pnl_calc_df, start_date, end_date='today', lookback=126,
                                     halflife=None, alpha=0.99,
                                     yesterday_price_only=True,
                                     use_dv01_pnl_approx=False,
                                     risk_column_names=['ivol_lag_adj'],
                                     agg_col=None,
                                     index_cols=[],
                                     pnl_col='pnl_usd'):
    """
    Calculate historical risk metrics using historical positions and price data.

    This function differs from get_risk_metrics_cur_pos in that it uses the actual
    positions held on each historical date, rather than using current positions with
    historical prices.

    Parameters:
    -----------
    pos_pnl_calc_df : pd.DataFrame
        DataFrame with historical position and PnL data indexed by date
    start_date : str or pd.Timestamp
        Start date for historical calculation
    end_date : str or pd.Timestamp
        End date for historical calculation (default 'today')
    lookback : int
        Number of days to look back for risk calculation (default 126)
    halflife : float, optional
        Halflife for exponential weighting (alternative to alpha)
    alpha : float
        Alpha parameter for exponential weighting (default 0.99)
    yesterday_price_only : bool
        Whether to use only yesterday's prices (default True)
    use_dv01_pnl_approx : bool
        Whether to use DV01 approximation for PnL (default False)
    risk_column_names : list
        List of risk metric column names to include (default ['ivol_lag_adj'])
    agg_col : str, optional
        Column to aggregate by (e.g., 'asset_class')
    index_cols : list
        Index columns for non-aggregated data
    pnl_col : str
        PnL column name (default 'pnl_usd')

    Returns:
    --------
    pd.DataFrame
        Historical risk metrics with columns: [id_cols, date, updated_at, risk_column_names]
    """
    logging.info('begin get_risk_metrics_historical_pos()')

    risk_hist_df = pd.DataFrame()
    pbar = tqdm(pd.bdate_range(start_date, end_date))
    updated_at = pd.to_datetime('now')

    if use_dv01_pnl_approx and 'pnl_usd_dv01_approx' in pos_pnl_calc_df.columns:
        pnl_col = 'pnl_usd_dv01_approx'

    for date in pbar:
        pbar.set_description(f'{date.date()}')

        # Get risk metrics for this specific date
        cur_risk_df = get_risk_metrics(pos_pnl_calc_df, date, lookback, run_date=date,
                                       halflife=halflife, alpha=alpha,
                                       yesterday_price_only=yesterday_price_only,
                                       use_dv01_pnl_approx=use_dv01_pnl_approx,
                                       agg_col=agg_col, minimal=True,
                                       index_cols=index_cols, pnl_col=pnl_col)

        cur_risk_df['date'] = date
        cur_risk_df['updated_at'] = updated_at

        id_cols = index_cols if (agg_col is None) else [agg_col]
        cur_risk_df = cur_risk_df[id_cols + ['date', 'updated_at'] + risk_column_names]

        if cur_risk_df.empty:
            logging.warning(f'cur_risk_df is empty on {date}')
            continue

        risk_hist_df = pd.concat([risk_hist_df, cur_risk_df])

    logging.info(f'Completed historical risk calculation. Result shape: {risk_hist_df.shape}')
    return risk_hist_df