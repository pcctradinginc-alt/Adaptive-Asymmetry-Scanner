# ML-Research (Shadow) – 2026-10-02T21:09:01+00:00

Panel: 292515 Zeilen, 600 Ticker, 2014-06-06..2026-10-02 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: heutige Indexliste (Querschnittsvergleich dämpft den Bias). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | -0.0016 | -0.49 | -0.2 | -0.22412 | 0.571 | -0.00141 | 0.06 | -0.0046 | 0.02165 | 0 | None | 1.121/1.153 | None |
| enet_xs20_v1 | valid | 0.00341 | 1.01 | 0.4 | -0.15694 | 0.714 | 0.01842 | 1.03 | 0.00041 | 0.00863 | 0 | None | 1.249/1.154 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00394 | 1.04 | 0.41 | -0.16448 | 0.714 | 0.0375 | 2.03 | 0.00094 | -0.00437 | 0 | None | 1.281/1.154 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00304 | 0.95 | 0.37 | -0.17591 | 0.571 | 0.02181 | 1.32 | 4e-05 | 0.00075 | 0 | None | 1.256/1.154 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00316 | 1.65 | 0.65 | -0.07974 | 0.857 | 0.01158 | 1.28 | 0.00016 | -0.00029 | 0 | None | 1.26/1.154 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.0029 | 0.78 | 0.31 | -0.26092 | 0.571 | 0.03106 | 1.63 | -0.0001 | 0.00116 | 0 | None | 1.253/1.154 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6584 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7843 (schlechtestes Jahr ±0.1939) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1491
- P(Rendite_60 > +10 %): Brier 0.22233 vs. Basisrate 0.21053 (Skill -0.0561; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Real Estate 0.01935 (t 1.03), Communication Services 0.01695 (t 1.01), Energy 0.01611 (t 0.81), Basic Materials 0.01324 (t 0.74), Industrials 0.00543 (t 0.39), Consumer Cyclical 0.00101 (t 0.07), Technology -0.0023 (t -0.17), Consumer Defensive -0.01077 (t -0.7), Healthcare -0.01198 (t -0.84), Utilities -0.01603 (t -1.09), Financial Services -0.02204 (t -1.15)
- Regime (netto): vix_lt_20 -0.00243 (n 199), vix_ge_20 -0.00038 (n 136), spy_uptrend -0.00193 (n 260), spy_downtrend -0.00045 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00341 mit t=1.01 (< 2.0)
- Rank-IC 0.01842 (t=1.03) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02719, 'risk': 0.0096, 'liquidity': 0.00166, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01681), ('beta_126', 0.01143), ('mom_12_1', 0.00961), ('log_dollar_vol', 0.00167), ('max_ret_21', 0.00084), ('mom_3m', 0.00072)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.0417 (t 2.44), Real Estate 0.0266 (t 1.72), Consumer Defensive 0.02358 (t 1.83), Technology 0.02284 (t 1.99), Consumer Cyclical 0.02281 (t 1.76), Utilities 0.02067 (t 1.44), Communication Services 0.00196 (t 0.11), Industrials -0.00206 (t -0.17), Healthcare -0.00333 (t -0.28), Basic Materials -0.01421 (t -0.89), Energy -0.01875 (t -1.11)
- Regime (netto): vix_lt_20 -0.00317 (n 199), vix_ge_20 0.01304 (n 136), spy_uptrend 0.00032 (n 260), spy_downtrend 0.01413 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00394 mit t=1.04 (< 2.0)
- Locked-Holdout netto -0.00437 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.01273, 'risk': 0.03304, 'liquidity': 0.00533, 'market': 0.01178, 'macro': 0.04006}
- Wichtigste Features: [('beta_126', 0.02073), ('usd_63d_chg', 0.0169), ('spy_mom_63', 0.01089), ('wti_63d_chg', 0.01015), ('vol_60', 0.00885), ('rev_1m', 0.00834)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.07349 (t 4.53), Consumer Defensive 0.05422 (t 4.04), Real Estate 0.05133 (t 2.95), Healthcare 0.04407 (t 3.55), Consumer Cyclical 0.04123 (t 3.06), Communication Services 0.03563 (t 2.14), Technology 0.02401 (t 2.26), Utilities 0.01852 (t 1.35), Industrials 0.01475 (t 1.06), Energy 0.01319 (t 0.75), Basic Materials -0.01522 (t -0.89)
- Regime (netto): vix_lt_20 -4e-05 (n 199), vix_ge_20 0.00976 (n 136), spy_uptrend 0.00153 (n 260), spy_downtrend 0.01229 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00304 mit t=0.95 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02181 (t=1.32) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': -0.00262, 'risk': 0.0229, 'liquidity': 0.00232, 'market': 0.01273, 'macro': -0.00018}
- Wichtigste Features: [('beta_126', 0.01552), ('spy_trend_200', 0.00884), ('vix_chg_21', 0.00388), ('vol_ratio', 0.00377), ('usd_63d_chg', 0.00305), ('vol_20', 0.00261)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05429 (t 3.46), Communication Services 0.04025 (t 2.39), Healthcare 0.03268 (t 2.66), Consumer Cyclical 0.02628 (t 2.08), Technology 0.02259 (t 2.17), Consumer Defensive 0.01308 (t 1.03), Energy -0.00462 (t -0.29), Industrials -0.00543 (t -0.41), Utilities -0.00808 (t -0.6), Real Estate -0.00853 (t -0.5), Basic Materials -0.02223 (t -1.36)
- Regime (netto): vix_lt_20 0.00029 (n 199), vix_ge_20 0.00706 (n 136), spy_uptrend 0.00098 (n 260), spy_downtrend 0.01016 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Walk-Forward netto 0.00316 mit t=1.65 (< 2.0)
- Rank-IC 0.01158 (t=1.28) nicht signifikant
- Locked-Holdout netto -0.00029 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.01835, 'risk': 0.0, 'liquidity': 0.00675, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.009), ('mom_12_1', 0.00617), ('log_dollar_vol', 0.00578), ('mom_3m', 0.00444), ('relvol_5_60', 0.00096), ('rs_63', 0.0)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.06884 (t 4.21), Financial Services 0.0252 (t 1.94), Technology 0.02386 (t 2.92), Utilities 0.01685 (t 1.29), Real Estate 0.01671 (t 1.19), Healthcare 0.01597 (t 1.74), Consumer Cyclical 0.01436 (t 1.54), Consumer Defensive 0.01419 (t 1.32), Basic Materials 0.00591 (t 0.4), Industrials -0.02264 (t -2.35), Energy -0.06034 (t -3.84)
- Regime (netto): vix_lt_20 0.0009 (n 199), vix_ge_20 0.00647 (n 136), spy_uptrend 0.00152 (n 260), spy_downtrend 0.00886 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.0029 mit t=0.78 (< 2.0)
- nur 0.571 der Testjahre positiv
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.03106 (t=1.63) nicht signifikant
- Max-DD -0.26092 schlechter als Benchmark -0.22412
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.03379, 'liquidity': 0.0, 'market': 0.00631, 'macro': 0.01714}
- Wichtigste Features: [('beta_126', 0.01664), ('usd_63d_chg', 0.01338), ('vol_60', 0.01238), ('spy_mom_63', 0.00996), ('vix', 0.00389), ('vol_20', 0.00299)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05206 (t 3.19), Consumer Cyclical 0.05136 (t 3.63), Technology 0.03273 (t 2.75), Healthcare 0.02129 (t 1.78), Energy 0.01755 (t 0.96), Industrials 0.01066 (t 0.76), Utilities 0.00699 (t 0.5), Consumer Defensive 0.00615 (t 0.51), Real Estate -0.00311 (t -0.19), Basic Materials -0.0059 (t -0.36), Communication Services -0.00996 (t -0.66)
- Regime (netto): vix_lt_20 -0.00179 (n 199), vix_ge_20 0.00976 (n 136), spy_uptrend 0.00022 (n 260), spy_downtrend 0.01219 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
