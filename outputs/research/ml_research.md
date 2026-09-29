# ML-Research (Shadow) – 2026-09-29T14:17:55+00:00

Panel: 315232 Zeilen, 512 Ticker, 2014-06-06..2026-09-28 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | 0.00355 | 1.02 | 0.4 | -0.20608 | 0.571 | 0.0012 | 0.19 | 0.00055 | 0.02845 | 0 | None | 1.242/1.21 | None |
| enet_xs20_v1 | valid | 0.00744 | 1.63 | 0.64 | -0.19874 | 0.857 | 0.02507 | 1.08 | 0.00444 | 0.03949 | 0 | None | 1.3/1.212 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00833 | 1.89 | 0.75 | -0.19204 | 0.714 | 0.02721 | 1.53 | 0.00533 | 0.0138 | 0 | None | 1.351/1.212 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00404 | 1.38 | 0.55 | -0.17735 | 0.571 | 0.02886 | 1.67 | 0.00104 | 0.0096 | 0 | None | 1.334/1.212 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00972 | 3.03 | 1.2 | -0.10393 | 0.857 | 0.01414 | 1.19 | 0.00672 | 0.01375 | 0 | None | 1.36/1.212 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00784 | 1.62 | 0.64 | -0.27936 | 0.571 | 0.0182 | 0.75 | 0.00484 | 0.0193 | 0 | None | 1.337/1.212 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt 0.6459 ab (Soll 0.8) -> NICHT kalibriert
- P(Rendite_60 > +10 %): Brier 0.23048 vs. Basisrate 0.21797 (Skill -0.0574; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Basic Materials 0.0255 (t 1.39), Communication Services 0.02437 (t 1.47), Energy 0.01113 (t 0.58), Industrials 0.00806 (t 0.62), Consumer Cyclical -0.00097 (t -0.06), Real Estate -0.00395 (t -0.21), Healthcare -0.00414 (t -0.3), Technology -0.00678 (t -0.55), Consumer Defensive -0.00726 (t -0.47), Utilities -0.00778 (t -0.53), Financial Services -0.02019 (t -1.12)
- Regime (netto): vix_lt_20 0.00143 (n 199), vix_ge_20 0.00665 (n 136), spy_uptrend 0.00327 (n 260), spy_downtrend 0.0045 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00744 mit t=1.63 (< 2.0)
- Rank-IC 0.02507 (t=1.08) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00289, 'risk': 0.01798, 'liquidity': 0.00102, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('vol_60', 0.01373), ('rev_1m', 0.00562), ('max_ret_21', 0.00549), ('vol_20', 0.00115), ('log_dollar_vol', 0.00104), ('ret_5d', 5e-05)]
- Sektoren (Rank-IC innerhalb Sektor): Industrials 0.05455 (t 4.03), Real Estate 0.02493 (t 1.34), Financial Services 0.02466 (t 1.43), Consumer Cyclical 0.02231 (t 1.5), Technology 0.01811 (t 1.3), Utilities 0.01031 (t 0.66), Healthcare 0.00563 (t 0.43), Basic Materials -9e-05 (t -0.01), Energy -0.00355 (t -0.19), Consumer Defensive -0.0061 (t -0.43), Communication Services -0.0332 (t -1.92)
- Regime (netto): vix_lt_20 0.00069 (n 199), vix_ge_20 0.01732 (n 136), spy_uptrend 0.00342 (n 260), spy_downtrend 0.02138 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00833 mit t=1.89 (< 2.0)
- Rank-IC 0.02721 (t=1.53) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00343, 'risk': 0.0167, 'liquidity': 0.00804, 'market': 0.00601, 'macro': 0.02331}
- Wichtigste Features: [('spy_mom_63', 0.01126), ('usd_63d_chg', 0.0087), ('beta_126', 0.00795), ('log_dollar_vol', 0.00777), ('vol_60', 0.00756), ('fed_assets_13w_chg', 0.00679)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04633 (t 3.25), Consumer Cyclical 0.03533 (t 2.75), Consumer Defensive 0.03241 (t 2.46), Technology 0.0264 (t 2.3), Communication Services 0.02108 (t 1.39), Healthcare 0.01641 (t 1.44), Energy 0.01537 (t 0.87), Industrials 0.01196 (t 1.0), Real Estate 0.00513 (t 0.31), Basic Materials -0.00491 (t -0.3), Utilities -0.0123 (t -0.91)
- Regime (netto): vix_lt_20 0.0013 (n 199), vix_ge_20 0.01861 (n 136), spy_uptrend 0.00382 (n 260), spy_downtrend 0.02393 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00404 mit t=1.38 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02886 (t=1.67) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00539, 'risk': 0.02954, 'liquidity': 0.0037, 'market': 0.01703, 'macro': 0.00875}
- Wichtigste Features: [('beta_126', 0.01937), ('spy_trend_200', 0.01497), ('usd_63d_chg', 0.00875), ('vol_60', 0.0075), ('rev_1m', 0.00365), ('log_dollar_vol', 0.00352)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.04867 (t 3.29), Financial Services 0.0427 (t 2.76), Consumer Cyclical 0.02251 (t 1.82), Healthcare 0.02224 (t 1.81), Consumer Defensive 0.0172 (t 1.26), Industrials 0.01267 (t 1.12), Technology 0.01252 (t 1.32), Real Estate 0.00744 (t 0.42), Basic Materials 0.00166 (t 0.1), Energy -0.00267 (t -0.17), Utilities -0.00388 (t -0.28)
- Regime (netto): vix_lt_20 -0.00033 (n 199), vix_ge_20 0.01043 (n 136), spy_uptrend 0.00225 (n 260), spy_downtrend 0.01026 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Rank-IC 0.01414 (t=1.19) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.01256, 'risk': 0.0, 'liquidity': 0.00391, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('dist_52w_high', 0.01076), ('log_dollar_vol', 0.00409), ('rev_1m', 0.00305), ('ret_5d', 0.00182), ('rs_63', 0.0), ('relvol_5_60', -0.00018)]
- Sektoren (Rank-IC innerhalb Sektor): Technology 0.03067 (t 3.53), Communication Services 0.03056 (t 2.11), Healthcare 0.02404 (t 2.33), Real Estate 0.02338 (t 1.48), Financial Services 0.01155 (t 1.12), Industrials 0.00295 (t 0.33), Consumer Cyclical 0.00064 (t 0.06), Utilities -0.00844 (t -0.64), Basic Materials -0.00934 (t -0.6), Energy -0.01173 (t -0.76), Consumer Defensive -0.02475 (t -1.99)
- Regime (netto): vix_lt_20 0.00559 (n 199), vix_ge_20 0.01576 (n 136), spy_uptrend 0.00651 (n 260), spy_downtrend 0.02086 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00784 mit t=1.62 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.0182 (t=0.75) nicht signifikant
- Max-DD -0.27936 schlechter als Benchmark -0.20608
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.00588, 'liquidity': 0.0, 'market': 0.0038, 'macro': 0.0102}
- Wichtigste Features: [('vol_60', 0.00941), ('spy_mom_63', 0.00793), ('usd_63d_chg', 0.00598), ('fed_assets_13w_chg', 0.00409), ('vol_20', 0.00352), ('spy_trend_200', 0.00174)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05177 (t 2.76), Consumer Cyclical 0.04297 (t 2.9), Consumer Defensive 0.03151 (t 2.41), Industrials 0.02654 (t 1.85), Technology 0.02158 (t 1.64), Energy 0.02138 (t 1.22), Healthcare 0.00919 (t 0.77), Utilities -0.00136 (t -0.09), Communication Services -0.00202 (t -0.12), Basic Materials -0.00304 (t -0.18), Real Estate -0.01321 (t -0.76)
- Regime (netto): vix_lt_20 -0.00158 (n 199), vix_ge_20 0.02163 (n 136), spy_uptrend 0.0036 (n 260), spy_downtrend 0.02253 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
