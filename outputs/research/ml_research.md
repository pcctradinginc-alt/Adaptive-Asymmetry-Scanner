# ML-Research (Shadow) – 2026-10-03T11:56:36+00:00

Panel: 292515 Zeilen, 600 Ticker, 2014-06-06..2026-10-02 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: PIT-Universum (758 Titel je Mitglied, 409 Änderungen; entfernte Titel mit Kursen 120/255 – Restbias: entfernte Titel ohne Yahoo-Kurse fehlen weiterhin). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | -0.0016 | -0.49 | -0.2 | -0.22412 | 0.571 | -0.00141 | 0.06 | -0.0046 | 0.02165 | 0 | None | 1.121/1.153 | None |
| enet_xs20_v1 | valid | 0.00341 | 1.01 | 0.4 | -0.15694 | 0.714 | 0.01842 | 1.03 | 0.00041 | 0.00863 | 0 | None | 1.249/1.154 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00346 | 0.89 | 0.35 | -0.16355 | 0.714 | 0.03605 | 2.0 | 0.00046 | -0.00437 | 0 | None | 1.271/1.154 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00043 | 0.07 | 0.03 | -0.1744 | 0.714 | 0.01058 | 0.54 | -0.00257 | 0.00075 | 0 | None | 1.212/1.154 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00335 | 1.78 | 0.7 | -0.07726 | 0.857 | 0.01307 | 1.42 | 0.00035 | -0.00029 | 0 | None | 1.265/1.154 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00386 | 1.07 | 0.42 | -0.22473 | 0.571 | 0.03072 | 1.53 | 0.00086 | 0.00116 | 0 | None | 1.266/1.154 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6655 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7839 (schlechtestes Jahr ±0.187) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1471
- P(Rendite_60 > +10 %): Brier 0.22218 vs. Basisrate 0.21053 (Skill -0.0553; <= 0 heißt: keine Information über die Basisrate hinaus)

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
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02719, 'risk': 0.00959, 'liquidity': 0.00166, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01681), ('beta_126', 0.01143), ('mom_12_1', 0.00961), ('log_dollar_vol', 0.00167), ('max_ret_21', 0.00084), ('mom_3m', 0.00072)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04171 (t 2.44), Real Estate 0.02659 (t 1.72), Consumer Defensive 0.02359 (t 1.83), Consumer Cyclical 0.02283 (t 1.76), Technology 0.02283 (t 1.98), Utilities 0.02072 (t 1.44), Communication Services 0.00196 (t 0.11), Industrials -0.00209 (t -0.17), Healthcare -0.0033 (t -0.27), Basic Materials -0.01422 (t -0.89), Energy -0.01891 (t -1.12)
- Regime (netto): vix_lt_20 -0.00316 (n 199), vix_ge_20 0.01303 (n 136), spy_uptrend 0.00032 (n 260), spy_downtrend 0.01413 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00346 mit t=0.89 (< 2.0)
- Locked-Holdout netto -0.00437 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.01068, 'risk': 0.03311, 'liquidity': 0.00527, 'market': 0.00991, 'macro': 0.03861}
- Wichtigste Features: [('beta_126', 0.01844), ('usd_63d_chg', 0.01787), ('wti_63d_chg', 0.0092), ('spy_mom_63', 0.00895), ('vol_60', 0.00832), ('rev_1m', 0.00745)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.07287 (t 4.61), Real Estate 0.05311 (t 3.11), Consumer Defensive 0.0487 (t 3.69), Consumer Cyclical 0.04362 (t 3.23), Communication Services 0.03488 (t 2.04), Healthcare 0.03462 (t 2.84), Technology 0.02469 (t 2.3), Utilities 0.01622 (t 1.18), Energy 0.01165 (t 0.67), Industrials 0.01154 (t 0.84), Basic Materials -0.02098 (t -1.22)
- Regime (netto): vix_lt_20 -0.0002 (n 199), vix_ge_20 0.00882 (n 136), spy_uptrend 0.0012 (n 260), spy_downtrend 0.01129 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00043 mit t=0.07 (< 2.0)
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.01058 (t=0.54) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00272, 'risk': 0.00072, 'liquidity': 0.00187, 'market': 0.01082, 'macro': -0.00384}
- Wichtigste Features: [('spy_trend_200', 0.00678), ('rev_1m', 0.00412), ('usd_63d_chg', 0.00322), ('vol_ratio', 0.00303), ('curve_10y_3m', 0.00195), ('log_dollar_vol', 0.00188)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.03847 (t 2.26), Healthcare 0.02517 (t 1.98), Communication Services 0.02205 (t 1.35), Technology 0.01356 (t 1.28), Consumer Cyclical 0.00704 (t 0.51), Consumer Defensive 0.00293 (t 0.23), Industrials -0.01301 (t -1.0), Utilities -0.01555 (t -1.14), Basic Materials -0.01684 (t -0.97), Real Estate -0.03237 (t -1.82), Energy -0.03423 (t -2.13)
- Regime (netto): vix_lt_20 -0.00233 (n 199), vix_ge_20 0.00446 (n 136), spy_uptrend -0.00116 (n 260), spy_downtrend 0.00594 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Walk-Forward netto 0.00335 mit t=1.78 (< 2.0)
- Rank-IC 0.01307 (t=1.42) nicht signifikant
- Locked-Holdout netto -0.00029 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02426, 'risk': 0.0, 'liquidity': 0.00794, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.00979), ('mom_12_1', 0.00699), ('log_dollar_vol', 0.00642), ('mom_3m', 0.00604), ('relvol_5_60', 0.00152), ('dist_52w_high', 0.00132)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.07208 (t 4.34), Technology 0.0279 (t 3.39), Financial Services 0.02519 (t 1.94), Healthcare 0.01771 (t 1.93), Consumer Defensive 0.01765 (t 1.62), Real Estate 0.01421 (t 1.03), Utilities 0.01374 (t 1.06), Consumer Cyclical 0.01239 (t 1.32), Basic Materials 0.01164 (t 0.81), Industrials -0.02195 (t -2.27), Energy -0.06135 (t -3.85)
- Regime (netto): vix_lt_20 0.00136 (n 199), vix_ge_20 0.00626 (n 136), spy_uptrend 0.00189 (n 260), spy_downtrend 0.00839 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00386 mit t=1.07 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.03072 (t=1.53) nicht signifikant
- Max-DD -0.22473 schlechter als Benchmark -0.22412
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.03478, 'liquidity': 0.0, 'market': 0.01001, 'macro': 0.01128}
- Wichtigste Features: [('beta_126', 0.01422), ('vol_60', 0.01372), ('spy_mom_63', 0.01147), ('usd_63d_chg', 0.0099), ('vol_20', 0.00488), ('tnx', 0.00298)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05344 (t 3.22), Consumer Cyclical 0.04645 (t 3.32), Technology 0.03485 (t 2.88), Energy 0.02335 (t 1.27), Healthcare 0.02322 (t 1.86), Industrials 0.01191 (t 0.84), Consumer Defensive 0.01189 (t 0.99), Utilities 0.00434 (t 0.31), Real Estate 0.0021 (t 0.13), Basic Materials -0.00958 (t -0.59), Communication Services -0.01872 (t -1.24)
- Regime (netto): vix_lt_20 -0.00157 (n 199), vix_ge_20 0.0118 (n 136), spy_uptrend 0.00103 (n 260), spy_downtrend 0.01367 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
