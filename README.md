# Risk Contribution Decomposition - Figure & Table Generation

This project generates all figures and tables for the paper "On the Structure of Risk Contribution: A Leave-One-Out Decomposition into Inherent and Correlation Risk" by Nolan Alexander and Frank Fabozzi.

## Quick Start

To generate all figures and tables:

```bash
cd pysrc
python generate_rc_decomp_figures.py
```

All outputs will be saved to `data/processed/`

**Runtime:** ~18-20 minutes on standard hardware

## Outputs Generated

### Figures (PNG, 300 DPI)

1. **rc_convergence.png** - Figure 2: RC convergence with synthetic data showing total RC, inherent RC, and correlation RC components
2. **rc_tornado.png** - Figure 3: Tornado chart showing inherent and correlation RC decomposition by asset
3. **ls_rc_history.png** - Figure 4: Individual asset RC history over time
4. **ls_rc_history_asset_group.png** - Figure 5: Long-Short portfolio three-panel RC decomposition
5. **eq_rc_history_asset_group.png** - Figure 6: Equal-Weight portfolio three-panel RC decomposition
6. **rp_rc_history_asset_group.png** - Figure 7: Risk Parity portfolio three-panel RC decomposition

### Tables (CSV)

1. **portfolio_weights.csv** - Portfolio weights for LS, Eq, and RP portfolios
2. **last_rc_lag_adj_data.csv** - RC decomposition by individual asset (RC, Inherent, Correlation)
3. **last_asset_class_rc_lag_adj_data.csv** - RC decomposition aggregated by asset class
4. **last_rc_comparison.csv** - RC metrics comparison for LS portfolio

## Project Structure

```
.
├── data/
│   ├── raw/                    # Input data (icd_prices.csv)
│   └── processed/              # Generated figures and tables
├── pysrc/                      # Python source code
│   ├── generate_rc_decomp_figures.py  # Main script
│   ├── helpers.py              # Utility functions
│   ├── risk_calc.py            # Risk calculation functions
│   └── risk_model_history.py   # Historical risk metrics
├── log/                        # Execution logs
├── documentation/              # Project documentation
├── PARAMS.yaml                 # Configuration file
└── README.md                   # This file
```

## Configuration

Key parameters in `PARAMS.yaml`:

```yaml
risk_params:
  alpha: 0.99          # Exponential decay parameter
  lookback: 126        # 6-month lookback window
  leverage: 10         # Portfolio leverage
  roll_code: "N:05_0_R"  # Bloomberg roll code
```

## Data Requirements

The script requires only one input file: `data/raw/icd_prices.csv`

**Format:**
- Date index (daily frequency)
- Price columns for 12 instruments (futures and FX)
- Data spanning 1990-2025

**Note:** All unnecessary data files have been removed. The `data/` directory contains only:
- `data/raw/icd_prices.csv` - Input price data (908 KB)
- `data/processed/` - Generated outputs (6 figures + 4 tables)

## Dependencies

**Standard libraries:** pandas, numpy, matplotlib, tqdm, yaml

**Custom modules (4 files, ~1,370 lines total):**
- **generate_rc_decomp_figures.py** (675 lines) - Main script
- **helpers.py** (303 lines) - Utility functions (logging, directory management, parameter loading)
- **risk_calc.py** (189 lines) - Core risk calculation and decomposition functions
- **risk_model_history.py** (204 lines) - Historical risk metric calculations

This is a **minimal, self-contained codebase** with only the code necessary to generate the paper's figures and tables. All database connections, unused functions, and unnecessary modules have been removed.

## Methodology

The script implements the Inherent-Correlation Decomposition (ICD):

```
RC(a) = RC_inherent(a) + RC_correlation(a)

Where:
- RC_inherent(a) = w_a² σ_a² / σ_p
- RC_correlation(a) = w_a(1-w_a) cov(r_a, r_{p\{a}}) / σ_p
```

Key properties:
- ✓ Strictly additive: Sum of all RCs equals portfolio volatility
- ✓ Inherent component is always non-negative
- ✓ Correlation component can be positive or negative
- ✓ Position reduces risk when: -RC_corr > RC_inherent

## Performance Notes

- Processes ~35 years of daily data (9,282 days)
- Calculates risk for 12 instruments across 3 portfolios
- Time-varying decomposition uses 5-day intervals for efficiency (1,832 calculation points)
- Total runtime: ~18-20 minutes

## Code Conventions

Following CLAUDE.md guidelines:
- Variables: `_df` (DataFrames), `_srs` (Series), `_arr` (arrays), `_list` (lists)
- Logging instead of print statements
- Configuration in PARAMS.yaml
- Cross-platform paths with os.path.join
- Type hints and docstrings for all functions

## Troubleshooting

**Issue:** Script fails with "file not found"
- **Solution:** Ensure you run from `pysrc/` directory

**Issue:** Missing parameters error
- **Solution:** Check that PARAMS.yaml exists in root directory

**Issue:** Out of memory
- **Solution:** Increase calc_freq_days in PARAMS.yaml to reduce calculation points

## References

Alexander, N., and F. Fabozzi. 2026. "On the Structure of Risk Contribution: A Leave-One-Out Decomposition into Inherent and Correlation Risk."

## License

Research and educational use only.
