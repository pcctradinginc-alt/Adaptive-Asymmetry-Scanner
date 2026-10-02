# ML-Research (Shadow) – 2026-10-02T14:46:09+00:00

Panel: 316389 Zeilen, 513 Ticker, 2014-06-06..2026-10-01 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | 0.00352 | 1.01 | 0.4 | -0.21179 | 0.571 | 0.00184 | 0.22 | 0.00052 | 0.02851 | 0 | None | 1.241/1.211 | None |
| enet_xs20_v1 | valid | 0.00733 | 1.6 | 0.63 | -0.19538 | 0.857 | 0.02557 | 1.09 | 0.00433 | 0.03908 | 0 | None | 1.297/1.213 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00679 | 1.57 | 0.62 | -0.19135 | 0.714 | 0.02448 | 1.43 | 0.00379 | 0.01239 | 0 | None | 1.329/1.213 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00349 | 1.23 | 0.49 | -0.15107 | 0.571 | 0.02283 | 1.36 | 0.00049 | 0.00616 | 0 | None | 1.32/1.213 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00853 | 2.75 | 1.08 | -0.1116 | 0.857 | 0.01309 | 1.09 | 0.00553 | 0.01412 | 0 | None | 1.342/1.213 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00846 | 1.73 | 0.68 | -0.27194 | 0.714 | 0.02215 | 0.9 | 0.00546 | 0.01935 | 0 | None | 1.339/1.213 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6512 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7944 (schlechtestes Jahr ±0.1376) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1502
- P(Rendite_60 > +10 %): Brier 0.23025 vs. Basisrate 0.21814 (Skill -0.0555; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Basic Materials 0.0255 (t 1.39), Communication Services 0.02437 (t 1.47), Energy 0.01113 (t 0.58), Industrials 0.00806 (t 0.62), Consumer Cyclical -0.00097 (t -0.06), Real Estate -0.00396 (t -0.21), Healthcare -0.00414 (t -0.3), Technology -0.00678 (t -0.55), Consumer Defensive -0.00726 (t -0.47), Utilities -0.00781 (t -0.53), Financial Services -0.01525 (t -0.85)
- Regime (netto): vix_lt_20 0.00147 (n 199), vix_ge_20 0.0065 (n 136), spy_uptrend 0.00324 (n 260), spy_downtrend 0.00448 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00733 mit t=1.6 (< 2.0)
- Rank-IC 0.02557 (t=1.09) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00385, 'risk': 0.01707, 'liquidity': 0.00161, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('vol_60', 0.01343), ('rev_1m', 0.00637), ('max_ret_21', 0.00492), ('log_dollar_vol', 0.00161), ('vol_20', 0.00153), ('ret_5d', 0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Industrials 0.05428 (t 4.0), Financial Services 0.02989 (t 1.74), Real Estate 0.02404 (t 1.28), Consumer Cyclical 0.0226 (t 1.52), Technology 0.01788 (t 1.28), Utilities 0.01019 (t 0.65), Healthcare 0.00591 (t 0.44), Basic Materials -0.0006 (t -0.03), Energy -0.0024 (t -0.13), Consumer Defensive -0.00622 (t -0.44), Communication Services -0.03318 (t -1.91)
- Regime (netto): vix_lt_20 0.00046 (n 199), vix_ge_20 0.01738 (n 136), spy_uptrend 0.00326 (n 260), spy_downtrend 0.02142 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00679 mit t=1.57 (< 2.0)
- Rank-IC 0.02448 (t=1.43) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0058, 'risk': 0.0109, 'liquidity': 0.0082, 'market': 0.00252, 'macro': 0.01677}
- Wichtigste Features: [('spy_mom_63', 0.01407), ('vol_60', 0.01048), ('log_dollar_vol', 0.00807), ('usd_63d_chg', 0.00693), ('rev_1m', 0.00674), ('fed_assets_13w_chg', 0.00609)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.03869 (t 2.77), Consumer Defensive 0.03465 (t 2.57), Consumer Cyclical 0.03294 (t 2.6), Healthcare 0.02647 (t 2.28), Technology 0.02131 (t 1.9), Communication Services 0.0183 (t 1.24), Energy 0.01563 (t 0.9), Industrials 0.00713 (t 0.6), Basic Materials -0.00739 (t -0.45), Real Estate -0.01454 (t -0.88), Utilities -0.01607 (t -1.17)
- Regime (netto): vix_lt_20 -8e-05 (n 199), vix_ge_20 0.01683 (n 136), spy_uptrend 0.00218 (n 260), spy_downtrend 0.02275 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00349 mit t=1.23 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02283 (t=1.36) nicht signifikant
- Sharpe 0.49 nicht > Benchmark 0.4 + Marge
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00339, 'risk': 0.02239, 'liquidity': 0.00447, 'market': 0.01089, 'macro': 0.01067}
- Wichtigste Features: [('spy_trend_200', 0.01159), ('vol_60', 0.01095), ('usd_63d_chg', 0.00983), ('beta_126', 0.00957), ('rev_1m', 0.00451), ('log_dollar_vol', 0.00433)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.03892 (t 2.62), Financial Services 0.03414 (t 2.25), Consumer Cyclical 0.03104 (t 2.48), Healthcare 0.02117 (t 1.74), Industrials 0.00945 (t 0.83), Technology 0.00918 (t 0.97), Real Estate 0.00108 (t 0.06), Consumer Defensive 0.00101 (t 0.07), Basic Materials -0.00081 (t -0.05), Utilities -0.00483 (t -0.36), Energy -0.00866 (t -0.54)
- Regime (netto): vix_lt_20 -0.00037 (n 199), vix_ge_20 0.00914 (n 136), spy_uptrend 0.00149 (n 260), spy_downtrend 0.01043 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Rank-IC 0.01309 (t=1.09) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00867, 'risk': 0.0, 'liquidity': 0.00455, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('dist_52w_high', 0.01131), ('log_dollar_vol', 0.00374), ('rev_1m', 0.00269), ('relvol_5_60', 0.00081), ('rs_63', 0.0), ('ret_5d', -0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Healthcare 0.02587 (t 2.46), Technology 0.02584 (t 2.93), Communication Services 0.02189 (t 1.49), Real Estate 0.02013 (t 1.28), Financial Services 0.01502 (t 1.48), Consumer Cyclical 0.00196 (t 0.19), Industrials 0.00109 (t 0.12), Utilities -0.00487 (t -0.38), Basic Materials -0.00728 (t -0.47), Energy -0.02061 (t -1.37), Consumer Defensive -0.02607 (t -2.09)
- Regime (netto): vix_lt_20 0.00479 (n 199), vix_ge_20 0.014 (n 136), spy_uptrend 0.00537 (n 260), spy_downtrend 0.01946 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00846 mit t=1.73 (< 2.0)
- Rank-IC 0.02215 (t=0.9) nicht signifikant
- Max-DD -0.27194 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.01003, 'liquidity': 0.0, 'market': 0.00545, 'macro': 0.0169}
- Wichtigste Features: [('vol_60', 0.01024), ('spy_mom_63', 0.01009), ('usd_63d_chg', 0.00875), ('fed_assets_13w_chg', 0.00777), ('vol_20', 0.00239), ('spy_trend_200', 0.00185)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05756 (t 2.97), Consumer Cyclical 0.04213 (t 2.91), Energy 0.04193 (t 2.49), Consumer Defensive 0.03721 (t 2.75), Industrials 0.03484 (t 2.46), Basic Materials 0.02637 (t 1.59), Healthcare 0.01589 (t 1.36), Technology 0.01484 (t 1.12), Utilities -0.00079 (t -0.05), Communication Services -0.00434 (t -0.27), Real Estate -0.00487 (t -0.28)
- Regime (netto): vix_lt_20 -0.00119 (n 199), vix_ge_20 0.02257 (n 136), spy_uptrend 0.00362 (n 260), spy_downtrend 0.02525 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
