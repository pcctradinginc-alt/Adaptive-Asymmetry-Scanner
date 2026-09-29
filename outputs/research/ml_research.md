# ML-Research (Shadow) – 2026-09-29T16:09:09+00:00

Panel: 315876 Zeilen, 513 Ticker, 2014-06-06..2026-09-28 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | 0.00352 | 1.01 | 0.4 | -0.21179 | 0.571 | 0.00184 | 0.22 | 0.00052 | 0.02851 | 0 | None | 1.241/1.211 | None |
| enet_xs20_v1 | valid | 0.00732 | 1.6 | 0.63 | -0.1955 | 0.857 | 0.02558 | 1.09 | 0.00432 | 0.03908 | 0 | None | 1.297/1.213 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.0066 | 1.53 | 0.6 | -0.23173 | 0.571 | 0.02083 | 1.13 | 0.0036 | 0.0154 | 0 | None | 1.323/1.213 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00426 | 1.39 | 0.55 | -0.13212 | 0.571 | 0.02449 | 1.44 | 0.00126 | 0.01303 | 0 | None | 1.327/1.213 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00893 | 2.87 | 1.13 | -0.12466 | 0.714 | 0.01331 | 1.11 | 0.00593 | 0.01456 | 0 | None | 1.351/1.213 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.0073 | 1.5 | 0.59 | -0.29478 | 0.571 | 0.01628 | 0.68 | 0.0043 | 0.01068 | 0 | None | 1.335/1.213 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6461 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7899 (schlechtestes Jahr ±0.1671) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.132
- P(Rendite_60 > +10 %): Brier 0.23003 vs. Basisrate 0.21814 (Skill -0.0545; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Basic Materials 0.0255 (t 1.39), Communication Services 0.02437 (t 1.47), Energy 0.01113 (t 0.58), Industrials 0.00806 (t 0.62), Consumer Cyclical -0.00097 (t -0.06), Real Estate -0.00395 (t -0.21), Healthcare -0.00414 (t -0.3), Technology -0.00678 (t -0.55), Consumer Defensive -0.00726 (t -0.47), Utilities -0.0078 (t -0.53), Financial Services -0.01525 (t -0.85)
- Regime (netto): vix_lt_20 0.00147 (n 199), vix_ge_20 0.0065 (n 136), spy_uptrend 0.00324 (n 260), spy_downtrend 0.00448 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00732 mit t=1.6 (< 2.0)
- Rank-IC 0.02558 (t=1.09) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00385, 'risk': 0.01706, 'liquidity': 0.00162, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('vol_60', 0.01342), ('rev_1m', 0.00637), ('max_ret_21', 0.00492), ('log_dollar_vol', 0.00162), ('vol_20', 0.00153), ('ret_5d', 0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Industrials 0.05425 (t 4.0), Financial Services 0.02994 (t 1.74), Real Estate 0.02413 (t 1.29), Consumer Cyclical 0.02261 (t 1.52), Technology 0.01786 (t 1.28), Utilities 0.01007 (t 0.64), Healthcare 0.00582 (t 0.44), Basic Materials -0.00056 (t -0.03), Energy -0.0023 (t -0.13), Consumer Defensive -0.0061 (t -0.43), Communication Services -0.0331 (t -1.9)
- Regime (netto): vix_lt_20 0.00046 (n 199), vix_ge_20 0.01735 (n 136), spy_uptrend 0.00328 (n 260), spy_downtrend 0.02134 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.0066 mit t=1.53 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02083 (t=1.13) nicht signifikant
- Max-DD -0.23173 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00465, 'risk': 0.00558, 'liquidity': 0.00722, 'market': 0.0002, 'macro': 0.01949}
- Wichtigste Features: [('spy_mom_63', 0.01214), ('vol_60', 0.00733), ('rev_1m', 0.0071), ('log_dollar_vol', 0.00692), ('usd_63d_chg', 0.00678), ('fed_assets_13w_chg', 0.00617)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04076 (t 2.8), Consumer Defensive 0.0291 (t 2.14), Consumer Cyclical 0.02874 (t 2.16), Technology 0.02111 (t 1.84), Energy 0.01754 (t 0.99), Communication Services 0.01625 (t 1.09), Healthcare 0.01476 (t 1.31), Industrials 0.00582 (t 0.48), Basic Materials -0.0092 (t -0.54), Real Estate -0.01049 (t -0.63), Utilities -0.02252 (t -1.7)
- Regime (netto): vix_lt_20 0.00021 (n 199), vix_ge_20 0.01595 (n 136), spy_uptrend 0.00217 (n 260), spy_downtrend 0.02197 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00426 mit t=1.39 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02449 (t=1.44) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00244, 'risk': 0.02008, 'liquidity': 0.00486, 'market': 0.01103, 'macro': 0.00911}
- Wichtigste Features: [('spy_trend_200', 0.01196), ('vol_60', 0.00966), ('usd_63d_chg', 0.0091), ('beta_126', 0.0078), ('log_dollar_vol', 0.00469), ('rev_1m', 0.00322)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.03678 (t 2.44), Financial Services 0.03414 (t 2.22), Consumer Cyclical 0.03162 (t 2.55), Healthcare 0.0239 (t 1.98), Technology 0.012 (t 1.26), Industrials 0.00788 (t 0.66), Energy 0.00771 (t 0.49), Consumer Defensive 0.00069 (t 0.05), Basic Materials -0.00316 (t -0.19), Real Estate -0.00664 (t -0.38), Utilities -0.01293 (t -0.96)
- Regime (netto): vix_lt_20 0.0012 (n 199), vix_ge_20 0.00873 (n 136), spy_uptrend 0.00313 (n 260), spy_downtrend 0.00817 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Rank-IC 0.01331 (t=1.11) nicht signifikant
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00933, 'risk': 0.0, 'liquidity': 0.00487, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('dist_52w_high', 0.0107), ('log_dollar_vol', 0.00412), ('rev_1m', 0.0034), ('relvol_5_60', 0.00076), ('ret_5d', 0.00054), ('rs_63', 0.0)]
- Sektoren (Rank-IC innerhalb Sektor): Healthcare 0.02735 (t 2.59), Technology 0.02478 (t 2.85), Real Estate 0.02229 (t 1.42), Communication Services 0.01959 (t 1.34), Financial Services 0.0156 (t 1.54), Consumer Cyclical 0.00357 (t 0.35), Industrials 0.00218 (t 0.25), Basic Materials -0.00267 (t -0.17), Utilities -0.0074 (t -0.58), Energy -0.01601 (t -1.05), Consumer Defensive -0.02799 (t -2.23)
- Regime (netto): vix_lt_20 0.00539 (n 199), vix_ge_20 0.01409 (n 136), spy_uptrend 0.00621 (n 260), spy_downtrend 0.01834 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.0073 mit t=1.5 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.01628 (t=0.68) nicht signifikant
- Max-DD -0.29478 schlechter als Benchmark -0.21179
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.00123, 'liquidity': 0.0, 'market': -0.00016, 'macro': 0.01335}
- Wichtigste Features: [('vol_60', 0.00875), ('usd_63d_chg', 0.00763), ('spy_mom_63', 0.00627), ('fed_assets_13w_chg', 0.00353), ('vol_20', 0.00228), ('spy_trend_200', 0.00168)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04227 (t 2.28), Consumer Cyclical 0.04123 (t 2.87), Energy 0.02533 (t 1.51), Consumer Defensive 0.02321 (t 1.76), Industrials 0.02294 (t 1.64), Utilities 0.01817 (t 1.17), Technology 0.01738 (t 1.35), Healthcare 0.0095 (t 0.8), Communication Services 0.00342 (t 0.21), Real Estate -0.0068 (t -0.41), Basic Materials -0.00792 (t -0.5)
- Regime (netto): vix_lt_20 -0.00209 (n 199), vix_ge_20 0.02104 (n 136), spy_uptrend 0.00306 (n 260), spy_downtrend 0.022 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
