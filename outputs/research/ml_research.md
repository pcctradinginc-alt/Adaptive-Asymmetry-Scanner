# ML-Research (Shadow) – 2026-09-29T14:02:46+00:00

Panel: 315876 Zeilen, 513 Ticker, 2014-06-06..2026-09-28 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | 0.00352 | 1.01 | 0.4 | -0.21179 | 0.571 | 0.00184 | 0.22 | 0.00052 | 0.02851 | 0 | None | 1.241/1.211 | None |
| enet_xs20_v1 | valid | 0.00732 | 1.6 | 0.63 | -0.1953 | 0.857 | 0.02559 | 1.09 | 0.00432 | 0.03908 | 0 | None | 1.297/1.213 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00798 | 1.9 | 0.75 | -0.1869 | 0.571 | 0.02297 | 1.28 | 0.00498 | 0.01505 | 0 | None | 1.35/1.213 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00312 | 1.03 | 0.41 | -0.18746 | 0.429 | 0.0233 | 1.34 | 0.00012 | 0.01176 | 0 | None | 1.308/1.213 | rejected_so_far |

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Basic Materials 0.0255 (t 1.39), Communication Services 0.02437 (t 1.47), Energy 0.01113 (t 0.58), Industrials 0.00806 (t 0.62), Consumer Cyclical -0.00097 (t -0.06), Real Estate -0.00395 (t -0.21), Healthcare -0.00414 (t -0.3), Technology -0.00678 (t -0.55), Consumer Defensive -0.00726 (t -0.47), Utilities -0.00781 (t -0.53), Financial Services -0.01526 (t -0.85)
- Regime (netto): vix_lt_20 0.00147 (n 199), vix_ge_20 0.0065 (n 136), spy_uptrend 0.00324 (n 260), spy_downtrend 0.00448 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00732 mit t=1.6 (< 2.0)
- Rank-IC 0.02559 (t=1.09) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00384, 'risk': 0.01706, 'liquidity': 0.00162, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('vol_60', 0.01342), ('rev_1m', 0.00637), ('max_ret_21', 0.00492), ('log_dollar_vol', 0.00162), ('vol_20', 0.00153), ('ret_5d', 0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Industrials 0.05426 (t 4.0), Financial Services 0.02994 (t 1.74), Real Estate 0.02414 (t 1.29), Consumer Cyclical 0.02259 (t 1.52), Technology 0.01786 (t 1.28), Utilities 0.01008 (t 0.64), Healthcare 0.00583 (t 0.44), Basic Materials -0.00056 (t -0.03), Energy -0.00226 (t -0.12), Consumer Defensive -0.0061 (t -0.43), Communication Services -0.0331 (t -1.9)
- Regime (netto): vix_lt_20 0.00046 (n 199), vix_ge_20 0.01736 (n 136), spy_uptrend 0.00328 (n 260), spy_downtrend 0.02136 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00798 mit t=1.9 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02297 (t=1.28) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0055, 'risk': 0.00623, 'liquidity': 0.0077, 'market': 0.00058, 'macro': 0.0203}
- Wichtigste Features: [('spy_mom_63', 0.01135), ('vol_60', 0.00866), ('usd_63d_chg', 0.00798), ('rev_1m', 0.00793), ('log_dollar_vol', 0.00743), ('fed_assets_13w_chg', 0.00617)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.0372 (t 2.56), Consumer Cyclical 0.03472 (t 2.65), Communication Services 0.02627 (t 1.77), Technology 0.02205 (t 1.95), Healthcare 0.01934 (t 1.74), Energy 0.01815 (t 1.03), Consumer Defensive 0.01686 (t 1.21), Industrials 0.00827 (t 0.68), Real Estate -0.00538 (t -0.32), Basic Materials -0.00657 (t -0.39), Utilities -0.01258 (t -0.92)
- Regime (netto): vix_lt_20 0.002 (n 199), vix_ge_20 0.01674 (n 136), spy_uptrend 0.00369 (n 260), spy_downtrend 0.02287 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00312 mit t=1.03 (< 2.0)
- nur 0.429 der Testjahre positiv
- Rank-IC 0.0233 (t=1.34) nicht signifikant
- Sharpe 0.41 nicht > Benchmark 0.4 + Marge
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0067, 'risk': 0.01974, 'liquidity': 0.00461, 'market': 0.01459, 'macro': 0.01043}
- Wichtigste Features: [('vol_60', 0.01037), ('spy_trend_200', 0.01025), ('usd_63d_chg', 0.00948), ('beta_126', 0.00795), ('rev_1m', 0.00639), ('log_dollar_vol', 0.00447)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.04367 (t 2.93), Financial Services 0.0354 (t 2.27), Consumer Cyclical 0.02753 (t 2.18), Healthcare 0.02728 (t 2.28), Technology 0.00895 (t 0.96), Industrials 0.00621 (t 0.53), Real Estate 0.0026 (t 0.15), Consumer Defensive 4e-05 (t 0.0), Energy -0.00892 (t -0.56), Basic Materials -0.0099 (t -0.6), Utilities -0.01044 (t -0.77)
- Regime (netto): vix_lt_20 -0.00114 (n 199), vix_ge_20 0.00936 (n 136), spy_uptrend 0.0013 (n 260), spy_downtrend 0.00941 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
