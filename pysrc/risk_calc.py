"""
Minimal risk calculation functions for figure generation only.
Contains only the functions needed to generate RC decomposition figures.
"""

import pandas as pd
import numpy as np
import logging

def calc_alpha(halflife):
    """
    Characterizes relationship between halflife and alpha.
    Alternative form is 1 - alpha = exp(-ln(2) / halflife)
    """
    return 1 - np.exp(-np.log(2)/halflife)


def cov(X, aweights, use_bias_correction=True, lead_adj=False, lag_adj=False, use_rank_corr=False):
    """Weighted covariance calculation with optional lag adjustments."""
    if use_rank_corr:
        X = pd.DataFrame(X).rank().values
    X = X.astype(float)
    w = aweights
    X_demeaned = X - np.dot(w, X)
    bias_correction = sum(w)/(sum(w)**2 - sum(w**2)) if use_bias_correction else 1
    cov_mat = np.dot((w * X_demeaned.T), X_demeaned) / sum(w) * bias_correction

    if lead_adj or lag_adj and X.shape[1] == 2:
        X_demeaned_shift_left  = pd.DataFrame(pd.concat((pd.DataFrame(X_demeaned).iloc[:,0].shift(),
                                                         pd.DataFrame(X_demeaned).iloc[:,1]), axis=1)).iloc[1:].to_numpy()
        X_demeaned_shift_right = pd.DataFrame(pd.concat((pd.DataFrame(X_demeaned).iloc[:,0],
                                                         pd.DataFrame(X_demeaned).iloc[:,1].shift()), axis=1)).iloc[1:].to_numpy()
        w = w[1:]/w[1:].sum()
        adj_cov_mat = cov_mat

        if lead_adj and X.shape[1] == 2:
            cov_mat_lag_left = np.dot((w * X_demeaned_shift_left.T), X_demeaned_shift_left) / sum(w) * bias_correction
            np.fill_diagonal(cov_mat_lag_left, 0)
            adj_cov_mat += cov_mat_lag_left

        if lag_adj and X.shape[1] == 2:
            cov_mat_lag_right = np.dot((w * X_demeaned_shift_right.T), X_demeaned_shift_right) / sum(w) * bias_correction
            np.fill_diagonal(cov_mat_lag_right, 0)
            adj_cov_mat += cov_mat_lag_right

        cov_mat = adj_cov_mat

    return cov_mat


def weighted_cov_no_rank_corr(X, halflife=None, alpha=None, fillna_val=0, lead_adj=False, lag_adj=False, use_rank_corr=False):
    """Weighted covariance without rank correction."""
    if fillna_val is not None:
        X = X.fillna(fillna_val).copy()

    if halflife is None and alpha is None:
        raise ValueError('halflife and alpha are both None for weighted_cov()')
    else:
        alpha = calc_alpha(halflife) if alpha is None else alpha
        samp_weights = alpha ** np.arange(len(X.index)-1, 0-1, step=-1)
        samp_weights = samp_weights / samp_weights.sum()
        weighted_cov_df = pd.DataFrame(cov(X.values, aweights=samp_weights, lead_adj=lead_adj, lag_adj=lag_adj, use_rank_corr=use_rank_corr),
                                        index=X.columns, columns=X.columns)
    return weighted_cov_df


def weighted_corr(X, halflife=None, alpha=None, fillna_val=0, lead_adj=False, lag_adj=False, use_rank_corr=False):
    """Weighted correlation."""
    weighted_cov_df = weighted_cov_no_rank_corr(X, halflife=halflife, alpha=alpha, fillna_val=fillna_val, lead_adj=lead_adj, lag_adj=lag_adj,
                                                      use_rank_corr=use_rank_corr)
    weighted_var_srs = pd.Series(np.diag(weighted_cov_df), index=weighted_cov_df.index)
    weighted_sd_srs = np.sqrt(weighted_var_srs)
    weighted_corr_df = weighted_cov_df.multiply(1/weighted_sd_srs, axis=0).multiply(1/weighted_sd_srs, axis=1)
    return weighted_corr_df


def weighted_sd(x, halflife=None, alpha=None, fillna_val=0, use_bias_correction=True):
    """Weighted standard deviation."""
    if fillna_val is not None:
        x = x.fillna(fillna_val).copy()

    if halflife is None and alpha is None:
        raise ValueError('halflife and alpha are both none for weighted_sd()')
    else:
        alpha = calc_alpha(halflife) if alpha is None else alpha
        samp_weights = alpha ** np.arange(len(x.index)-1, 0-1, step=-1)
        samp_weights = samp_weights / samp_weights.sum()

        w = samp_weights
        x_demeaned = x - np.dot(w, x)
        bias_correction = sum(w)/(sum(w)**2 - sum(w**2)) if use_bias_correction else 1
        var_val = np.dot((w * x_demeaned.T), x_demeaned) / sum(w) * bias_correction
        sd_val = np.sqrt(var_val)
        return sd_val


def weighted_cov(X, halflife=None, alpha=None, fillna_val=0, lead_adj=False, lag_adj=False, use_rank_corr=False):
    """Weighted covariance with optional rank correlation."""
    if use_rank_corr:
        port_corr_mat = weighted_corr(X, halflife=halflife, alpha=alpha, fillna_val=fillna_val, lead_adj=lead_adj, lag_adj=lag_adj,
                                      use_rank_corr=use_rank_corr)
        port_weighted_sd_srs = weighted_sd(X, halflife=halflife, alpha=alpha)
        port_cov_mat = port_corr_mat.multiply(port_weighted_sd_srs, axis=0).multiply(port_weighted_sd_srs, axis=1)
    else:
        port_cov_mat = weighted_cov_no_rank_corr(X, halflife=halflife, alpha=alpha, fillna_val=fillna_val, lead_adj=lead_adj, lag_adj=lag_adj)

    return port_cov_mat

def get_ivol(pos_pnl_usd_df, halflife=None, alpha=None, lead_adj=False, lag_adj=False, use_rank_corr=False):
    """
    Compute non-additive Incremental Volatility (iVol) per asset.

    iVol(a) = sigma_p - sigma_{p\\{a}}   (paper eq. 1)

    where sigma_{p\\{a}} = sqrt(sigma_p^2 + Var(P_a) - 2*Cov(P_a, P_p)) is the
    volatility of the portfolio with asset a removed. iVol does not sum to
    portfolio volatility (see Appendix A of the paper).

    Parameters
    ----------
    pos_pnl_usd_df : pd.DataFrame
        Per-asset weighted PnL (one column per asset, one row per date).
    halflife, alpha : float, optional
        Exponential weighting parameters for the covariance estimate.
    lead_adj, lag_adj : bool
        Asynchronous-covariance adjustments (paper eq. 12).
    use_rank_corr : bool
        Use Spearman rank correlation when estimating covariance.

    Returns
    -------
    pd.Series
        iVol per asset, indexed by the columns of pos_pnl_usd_df.
    """
    pos_pnl_usd_df = pos_pnl_usd_df.astype(float)
    port_cov_mat = weighted_cov(pos_pnl_usd_df, halflife=halflife, alpha=alpha,
                                lead_adj=lead_adj, lag_adj=lag_adj, use_rank_corr=use_rank_corr)
    port_vol = weighted_sd(pos_pnl_usd_df.sum(1), halflife=halflife, alpha=alpha)

    pnl_cov_rowsum_srs = port_cov_mat.sum(axis=1)
    asset_pnl_vol_srs = pd.Series(np.sqrt(np.diag(port_cov_mat)), index=port_cov_mat.index)
    # sigma_{p\{a}} = sqrt(sigma_p^2 + Var(P_a) - 2 * Cov(P_a, P_p))
    excl_port_vol_srs = np.sqrt(port_vol**2 + asset_pnl_vol_srs**2 - 2 * pnl_cov_rowsum_srs)
    ivol_srs = port_vol - excl_port_vol_srs
    return ivol_srs


def get_rc_weight(pos_pnl_usd_df, halflife=None, alpha=None, port_cov_mat=None,
                  universe=None, lead_adj=False, lag_adj=False, cov_contribution=False, port_vol=None,
                  use_rank_corr=False):
    """
    For a given position pnl dataframe, compute each asset's share of portfolio
    variance, decomposed into inherent and correlation parts (paper eq. 9):
        weight_inh(a) = w_a^2 * sigma_a^2 / sigma_p^2
        weight_cor(a) = w_a * (1 - w_a) * cov(r_a, r_{p\\{a}}) / sigma_p^2
        weight(a)     = weight_inh(a) + weight_cor(a)
    Multiplying weight(a) by sigma_p gives RC(a).
    """
    if universe is not None:
        pos_pnl_usd_df = pos_pnl_usd_df[universe].copy()

    if port_cov_mat is None:
        port_cov_mat = weighted_cov(pos_pnl_usd_df, halflife=halflife, alpha=alpha, lead_adj=lead_adj, lag_adj=lag_adj,
                                    use_rank_corr=use_rank_corr)

    if cov_contribution:
        cov_rowsum = port_cov_mat.sum(1)
        cov_contribution = cov_rowsum / cov_rowsum.sum()
        weights_df = pd.DataFrame({'weight' : cov_contribution})
        return weights_df

    port_pnl = pos_pnl_usd_df.sum(1)
    if port_vol is None:
        port_vol = weighted_sd(pos_pnl_usd_df.sum(1), halflife=halflife, alpha=alpha)

    asset_sds = pd.Series(np.sqrt(np.diag(port_cov_mat)), index=port_cov_mat.index)

    weights = pd.Series(index=pos_pnl_usd_df.columns, name='bloomberg_code_long')
    weight_inh_srs = pd.Series(index=pos_pnl_usd_df.columns, name='bloomberg_code_long')
    weight_cor_srs = pd.Series(index=pos_pnl_usd_df.columns, name='bloomberg_code_long')

    for col in asset_sds.index[asset_sds.notna()]:
        cur_pos_pnl = pos_pnl_usd_df[col].fillna(0)
        excl_and_port_pnls = pd.DataFrame(np.vstack((cur_pos_pnl, port_pnl - cur_pos_pnl)).T,
                                          columns=['position', 'portfolio_ex'])
        excl_and_port_cov_mat = weighted_cov(excl_and_port_pnls, halflife=halflife, alpha=alpha,
                                             lead_adj=lead_adj, lag_adj=lag_adj, use_rank_corr=use_rank_corr)
        cov = excl_and_port_cov_mat.iloc[0,1]
        asset_excluded_sd = asset_sds[col]

        weight_inh_srs[col] = asset_excluded_sd**2 / port_vol**2
        weight_cor_srs[col] = cov / port_vol**2
        weights[col] = weight_inh_srs[col] + weight_cor_srs[col]

    weights_df = pd.DataFrame({
        'weight' : weights,
        'weight_inh_part' : weight_inh_srs,
        'weight_cor_part' : weight_cor_srs,
    })

    return weights_df


def get_rc(pos_pnl_usd_df, halflife=None, alpha=None,
           universe=None, lead_adj=False, lag_adj=False, cov_contribution=False,
           use_rank_corr=False):
    """
    Compute additive Risk Contribution (RC) per asset (paper eq. 7):
        RC(a) = (w_a^2 * sigma_a^2 + w_a * (1 - w_a) * cov(r_a, r_{p\\{a}})) / sigma_p
    Sum of RC(a) over assets equals portfolio volatility sigma_p.
    """
    pos_pnl_usd_df = pos_pnl_usd_df.astype(float)
    port_cov_mat = weighted_cov(pos_pnl_usd_df, halflife=halflife, alpha=alpha, lead_adj=lead_adj, lag_adj=lag_adj, use_rank_corr=use_rank_corr)
    port_vol = weighted_sd(pos_pnl_usd_df.sum(1), halflife=halflife, alpha=alpha)

    weight_df = get_rc_weight(pos_pnl_usd_df, port_cov_mat=port_cov_mat,
                              halflife=halflife, alpha=alpha, universe=universe,
                              lead_adj=lead_adj, lag_adj=lag_adj, cov_contribution=cov_contribution, port_vol=port_vol)

    rc_srs = port_vol * weight_df['weight']
    logging.debug('rc_srs: \n' + str(rc_srs.head()))
    return rc_srs


def get_rc_parts(pos_pnl_usd_df, halflife=None, alpha=None, cov_contribution=False, lead_adj=False, lag_adj=False,
                 use_rank_corr=False):
    """
    Decompose RC into its inherent and correlation components (paper eq. 9):
        RC_inh(a)  = w_a^2 * sigma_a^2 / sigma_p
        RC_corr(a) = w_a * (1 - w_a) * cov(r_a, r_{p\\{a}}) / sigma_p
    Returns (rc_inh_srs, rc_corr_srs); both sum (component-wise) to RC.
    """
    port_cov_mat = weighted_cov(pos_pnl_usd_df, halflife=halflife, alpha=alpha, lead_adj=lead_adj, lag_adj=lag_adj,
                                use_rank_corr=use_rank_corr)
    port_vol = weighted_sd(pos_pnl_usd_df.sum(1), halflife=halflife, alpha=alpha)

    weight_df = get_rc_weight(pos_pnl_usd_df, port_cov_mat=port_cov_mat, halflife=halflife, alpha=alpha,
                              lead_adj=lead_adj, lag_adj=lag_adj, cov_contribution=cov_contribution)
    rc_inh_srs = weight_df['weight_inh_part'] * port_vol
    rc_corr_srs = weight_df['weight_cor_part'] * port_vol
    return rc_inh_srs, rc_corr_srs
