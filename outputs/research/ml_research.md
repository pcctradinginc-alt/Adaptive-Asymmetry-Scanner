# ML-Research (Shadow) – 2026-10-02T18:54:43+00:00

Panel: 316389 Zeilen, 513 Ticker, 2014-06-06..2026-10-01 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | 0.00352 | 1.01 | 0.4 | -0.21179 | 0.571 | 0.00184 | 0.22 | 0.00052 | 0.02851 | 0 | None | 1.241/1.211 | None |
| enet_xs20_v1 | valid | 0.00734 | 1.6 | 0.63 | -0.19538 | 0.857 | 0.02557 | 1.09 | 0.00434 | 0.03908 | 0 | None | 1.297/1.213 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00737 | 1.75 | 0.69 | -0.21666 | 0.714 | 0.02537 | 1.43 | 0.00437 | 0.01405 | 0 | None | 1.34/1.213 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00364 | 1.35 | 0.53 | -0.12707 | 0.714 | 0.02604 | 1.6 | 0.00064 | 0.01255 | 0 | None | 1.326/1.213 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00905 | 2.91 | 1.15 | -0.12078 | 0.857 | 0.01437 | 1.22 | 0.00605 | 0.01463 | 0 | None | 1.351/1.213 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00873 | 1.83 | 0.72 | -0.23469 | 0.714 | 0.0181 | 0.74 | 0.00573 | 0.01486 | 0 | None | 1.346/1.213 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6483 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7919 (schlechtestes Jahr ±0.1425) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1389
- P(Rendite_60 > +10 %): Brier 0.23009 vs. Basisrate 0.21814 (Skill -0.0548; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Basic Materials 0.0255 (t 1.39), Communication Services 0.02437 (t 1.47), Energy 0.01113 (t 0.58), Industrials 0.00806 (t 0.62), Consumer Cyclical -0.00097 (t -0.06), Real Estate -0.00396 (t -0.21), Healthcare -0.00414 (t -0.3), Technology -0.00678 (t -0.55), Consumer Defensive -0.00726 (t -0.47), Utilities -0.00781 (t -0.53), Financial Services -0.01526 (t -0.85)
- Regime (netto): vix_lt_20 0.00147 (n 199), vix_ge_20 0.0065 (n 136), spy_uptrend 0.00324 (n 260), spy_downtrend 0.00448 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00734 mit t=1.6 (< 2.0)
- Rank-IC 0.02557 (t=1.09) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00385, 'risk': 0.01707, 'liquidity': 0.00161, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('vol_60', 0.01343), ('rev_1m', 0.00637), ('max_ret_21', 0.00492), ('log_dollar_vol', 0.00161), ('vol_20', 0.00153), ('ret_5d', 0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Industrials 0.05428 (t 4.0), Financial Services 0.0299 (t 1.74), Real Estate 0.02405 (t 1.29), Consumer Cyclical 0.02261 (t 1.52), Technology 0.01788 (t 1.28), Utilities 0.01019 (t 0.65), Healthcare 0.00594 (t 0.45), Basic Materials -0.00061 (t -0.03), Energy -0.00233 (t -0.13), Consumer Defensive -0.0062 (t -0.44), Communication Services -0.03319 (t -1.91)
- Regime (netto): vix_lt_20 0.00046 (n 199), vix_ge_20 0.01741 (n 136), spy_uptrend 0.00327 (n 260), spy_downtrend 0.02142 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00737 mit t=1.75 (< 2.0)
- Rank-IC 0.02537 (t=1.43) nicht signifikant
- Max-DD -0.21666 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00537, 'risk': 0.01069, 'liquidity': 0.00871, 'market': 0.00485, 'macro': 0.02178}
- Wichtigste Features: [('spy_mom_63', 0.01217), ('log_dollar_vol', 0.00849), ('usd_63d_chg', 0.00842), ('vol_60', 0.00715), ('rev_1m', 0.00689), ('fed_assets_13w_chg', 0.0061)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.03858 (t 2.67), Consumer Defensive 0.03667 (t 2.73), Consumer Cyclical 0.03481 (t 2.71), Healthcare 0.03021 (t 2.66), Communication Services 0.02615 (t 1.79), Technology 0.02525 (t 2.3), Energy 0.02432 (t 1.42), Industrials 0.00512 (t 0.42), Real Estate -0.00903 (t -0.55), Basic Materials -0.01054 (t -0.65), Utilities -0.01811 (t -1.31)
- Regime (netto): vix_lt_20 0.00034 (n 199), vix_ge_20 0.01765 (n 136), spy_uptrend 0.00272 (n 260), spy_downtrend 0.02346 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00364 mit t=1.35 (< 2.0)
- Rank-IC 0.02604 (t=1.6) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': -0.00024, 'risk': 0.02484, 'liquidity': 0.00466, 'market': 0.0067, 'macro': 0.00428}
- Wichtigste Features: [('beta_126', 0.01587), ('spy_trend_200', 0.00941), ('usd_63d_chg', 0.00734), ('vol_60', 0.00491), ('log_dollar_vol', 0.00459), ('wti_63d_chg', 0.00227)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.04445 (t 3.0), Financial Services 0.03874 (t 2.6), Consumer Cyclical 0.02807 (t 2.25), Healthcare 0.02723 (t 2.28), Industrials 0.01291 (t 1.17), Technology 0.00953 (t 1.02), Consumer Defensive 0.00358 (t 0.26), Real Estate 0.00238 (t 0.13), Utilities 0.00154 (t 0.12), Energy -0.00618 (t -0.39), Basic Materials -0.0102 (t -0.61)
- Regime (netto): vix_lt_20 0.00123 (n 199), vix_ge_20 0.00716 (n 136), spy_uptrend 0.00211 (n 260), spy_downtrend 0.00893 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Rank-IC 0.01437 (t=1.22) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.01088, 'risk': 0.0, 'liquidity': 0.00532, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('dist_52w_high', 0.01212), ('log_dollar_vol', 0.0042), ('rev_1m', 0.0032), ('relvol_5_60', 0.00112), ('ret_5d', 9e-05), ('rs_63', 0.0)]
- Sektoren (Rank-IC innerhalb Sektor): Technology 0.02958 (t 3.49), Healthcare 0.02571 (t 2.44), Real Estate 0.02437 (t 1.55), Communication Services 0.02106 (t 1.45), Financial Services 0.01407 (t 1.39), Consumer Cyclical 0.0025 (t 0.24), Industrials 0.00231 (t 0.26), Basic Materials 0.00137 (t 0.09), Utilities -0.01075 (t -0.83), Energy -0.01848 (t -1.25), Consumer Defensive -0.02685 (t -2.19)
- Regime (netto): vix_lt_20 0.00495 (n 199), vix_ge_20 0.01506 (n 136), spy_uptrend 0.00602 (n 260), spy_downtrend 0.01958 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00873 mit t=1.83 (< 2.0)
- Rank-IC 0.0181 (t=0.74) nicht signifikant
- Max-DD -0.23469 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.00682, 'liquidity': 0.0, 'market': 0.00329, 'macro': 0.01451}
- Wichtigste Features: [('spy_mom_63', 0.00993), ('usd_63d_chg', 0.00905), ('vol_60', 0.00804), ('spy_trend_200', 0.00354), ('vol_20', 0.00319), ('fed_assets_13w_chg', 0.00238)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04775 (t 2.54), Consumer Defensive 0.0384 (t 2.92), Consumer Cyclical 0.03742 (t 2.61), Energy 0.03104 (t 1.87), Industrials 0.02559 (t 1.85), Technology 0.01633 (t 1.24), Utilities 0.01037 (t 0.66), Basic Materials 0.01011 (t 0.64), Healthcare 0.00934 (t 0.8), Communication Services -0.00059 (t -0.04), Real Estate -0.01072 (t -0.64)
- Regime (netto): vix_lt_20 -0.00078 (n 199), vix_ge_20 0.02263 (n 136), spy_uptrend 0.00394 (n 260), spy_downtrend 0.02532 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
