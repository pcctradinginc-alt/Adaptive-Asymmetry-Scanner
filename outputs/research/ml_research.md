# ML-Research (Shadow) – 2026-09-29T15:02:05+00:00

Panel: 315876 Zeilen, 513 Ticker, 2014-06-06..2026-09-28 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | 0.00352 | 1.01 | 0.4 | -0.21179 | 0.571 | 0.00184 | 0.22 | 0.00052 | 0.02851 | 0 | None | 1.241/1.211 | None |
| enet_xs20_v1 | valid | 0.00734 | 1.61 | 0.63 | -0.1953 | 0.857 | 0.02558 | 1.09 | 0.00434 | 0.03908 | 0 | None | 1.297/1.213 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00688 | 1.57 | 0.62 | -0.22777 | 0.714 | 0.01841 | 0.96 | 0.00388 | 0.0143 | 0 | None | 1.338/1.213 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00337 | 1.09 | 0.43 | -0.17637 | 0.714 | 0.02116 | 1.23 | 0.00037 | 0.01229 | 0 | None | 1.31/1.213 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00884 | 2.81 | 1.11 | -0.12436 | 0.857 | 0.01396 | 1.17 | 0.00584 | 0.01496 | 0 | None | 1.348/1.213 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00867 | 1.8 | 0.71 | -0.23204 | 0.714 | 0.02611 | 1.17 | 0.00567 | 0.01311 | 0 | None | 1.341/1.213 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.649 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7875 (schlechtestes Jahr ±0.1603) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1421
- P(Rendite_60 > +10 %): Brier 0.22969 vs. Basisrate 0.21814 (Skill -0.0529; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Basic Materials 0.0255 (t 1.39), Communication Services 0.02437 (t 1.47), Energy 0.01113 (t 0.58), Industrials 0.00806 (t 0.62), Consumer Cyclical -0.00097 (t -0.06), Real Estate -0.00395 (t -0.21), Healthcare -0.00414 (t -0.3), Technology -0.00678 (t -0.55), Consumer Defensive -0.00726 (t -0.47), Utilities -0.00781 (t -0.53), Financial Services -0.01525 (t -0.85)
- Regime (netto): vix_lt_20 0.00147 (n 199), vix_ge_20 0.0065 (n 136), spy_uptrend 0.00324 (n 260), spy_downtrend 0.00448 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00734 mit t=1.61 (< 2.0)
- Rank-IC 0.02558 (t=1.09) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00385, 'risk': 0.01706, 'liquidity': 0.00162, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('vol_60', 0.01343), ('rev_1m', 0.00637), ('max_ret_21', 0.00492), ('log_dollar_vol', 0.00162), ('vol_20', 0.00153), ('ret_5d', 0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Industrials 0.05427 (t 4.0), Financial Services 0.02993 (t 1.74), Real Estate 0.02414 (t 1.29), Consumer Cyclical 0.02261 (t 1.52), Technology 0.01786 (t 1.28), Utilities 0.01009 (t 0.64), Healthcare 0.00581 (t 0.44), Basic Materials -0.00056 (t -0.03), Energy -0.00231 (t -0.13), Consumer Defensive -0.00608 (t -0.43), Communication Services -0.03309 (t -1.9)
- Regime (netto): vix_lt_20 0.00049 (n 199), vix_ge_20 0.01736 (n 136), spy_uptrend 0.00328 (n 260), spy_downtrend 0.02142 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00688 mit t=1.57 (< 2.0)
- Rank-IC 0.01841 (t=0.96) nicht signifikant
- Max-DD -0.22777 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00303, 'risk': 0.0077, 'liquidity': 0.00724, 'market': 0.00145, 'macro': 0.02262}
- Wichtigste Features: [('spy_mom_63', 0.01193), ('usd_63d_chg', 0.00907), ('log_dollar_vol', 0.00714), ('rev_1m', 0.00676), ('vol_60', 0.00654), ('fed_assets_13w_chg', 0.0054)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.0384 (t 2.5), Consumer Cyclical 0.03087 (t 2.29), Healthcare 0.02836 (t 2.54), Consumer Defensive 0.02771 (t 2.0), Communication Services 0.01565 (t 1.03), Technology 0.01261 (t 1.07), Energy 0.01146 (t 0.66), Industrials 0.00487 (t 0.39), Real Estate -0.01396 (t -0.87), Basic Materials -0.01659 (t -0.97), Utilities -0.01813 (t -1.3)
- Regime (netto): vix_lt_20 0.00069 (n 199), vix_ge_20 0.01594 (n 136), spy_uptrend 0.00246 (n 260), spy_downtrend 0.02219 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00337 mit t=1.09 (< 2.0)
- Rank-IC 0.02116 (t=1.23) nicht signifikant
- Sharpe 0.43 nicht > Benchmark 0.4 + Marge
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00137, 'risk': 0.0202, 'liquidity': 0.00431, 'market': 0.01194, 'macro': 0.00812}
- Wichtigste Features: [('vol_60', 0.01004), ('spy_trend_200', 0.00979), ('usd_63d_chg', 0.00884), ('beta_126', 0.00765), ('rev_1m', 0.00442), ('log_dollar_vol', 0.00413)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.0315 (t 2.1), Financial Services 0.03093 (t 2.01), Consumer Cyclical 0.02692 (t 2.14), Healthcare 0.02043 (t 1.71), Technology 0.00852 (t 0.91), Industrials 0.00537 (t 0.46), Consumer Defensive 0.00162 (t 0.11), Real Estate 0.00124 (t 0.07), Energy 0.00036 (t 0.02), Basic Materials -0.00967 (t -0.58), Utilities -0.01324 (t -1.0)
- Regime (netto): vix_lt_20 -0.00041 (n 199), vix_ge_20 0.00892 (n 136), spy_uptrend 0.00165 (n 260), spy_downtrend 0.00936 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Rank-IC 0.01396 (t=1.17) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00941, 'risk': 0.0, 'liquidity': 0.00522, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('dist_52w_high', 0.01121), ('log_dollar_vol', 0.00436), ('rev_1m', 0.00299), ('relvol_5_60', 0.00086), ('ret_5d', 0.0001), ('rs_63', 0.0)]
- Sektoren (Rank-IC innerhalb Sektor): Healthcare 0.02635 (t 2.52), Technology 0.02544 (t 3.0), Real Estate 0.02322 (t 1.47), Communication Services 0.02221 (t 1.51), Financial Services 0.01607 (t 1.58), Consumer Cyclical 0.00317 (t 0.31), Industrials 0.00287 (t 0.33), Basic Materials -0.00281 (t -0.18), Utilities -0.00961 (t -0.75), Energy -0.01574 (t -1.04), Consumer Defensive -0.02799 (t -2.24)
- Regime (netto): vix_lt_20 0.00496 (n 199), vix_ge_20 0.01451 (n 136), spy_uptrend 0.00571 (n 260), spy_downtrend 0.01969 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00867 mit t=1.8 (< 2.0)
- Rank-IC 0.02611 (t=1.17) nicht signifikant
- Max-DD -0.23204 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0109, 'liquidity': 0.0, 'market': 0.0019, 'macro': 0.01429}
- Wichtigste Features: [('spy_mom_63', 0.00925), ('usd_63d_chg', 0.00748), ('fed_assets_13w_chg', 0.00653), ('vol_60', 0.00651), ('spy_trend_200', 0.00349), ('beta_126', 0.00267)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05646 (t 3.19), Consumer Cyclical 0.04336 (t 3.12), Consumer Defensive 0.03605 (t 2.76), Industrials 0.0295 (t 2.23), Energy 0.02659 (t 1.61), Technology 0.02472 (t 1.93), Basic Materials 0.01224 (t 0.78), Healthcare 0.01178 (t 1.0), Utilities 0.01146 (t 0.75), Communication Services -0.00332 (t -0.21), Real Estate -0.01787 (t -1.07)
- Regime (netto): vix_lt_20 -0.00094 (n 199), vix_ge_20 0.02273 (n 136), spy_uptrend 0.00384 (n 260), spy_downtrend 0.02543 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
