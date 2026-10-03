# ML-Research (Shadow) – 2026-10-03T09:16:54+00:00

Panel: 292515 Zeilen, 600 Ticker, 2014-06-06..2026-10-02 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: PIT-Universum (758 Titel je Mitglied, 409 Änderungen; entfernte Titel mit Kursen 120/255 – Restbias: entfernte Titel ohne Yahoo-Kurse fehlen weiterhin). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | -0.0016 | -0.49 | -0.2 | -0.22412 | 0.571 | -0.00141 | 0.06 | -0.0046 | 0.02165 | 0 | None | 1.121/1.153 | None |
| enet_xs20_v1 | valid | 0.00341 | 1.0 | 0.4 | -0.15694 | 0.714 | 0.01842 | 1.03 | 0.00041 | 0.00863 | 0 | None | 1.249/1.154 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00323 | 0.79 | 0.31 | -0.16955 | 0.714 | 0.03376 | 1.88 | 0.00023 | -0.00437 | 0 | None | 1.264/1.154 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00296 | 0.9 | 0.35 | -0.19623 | 0.857 | 0.02184 | 1.28 | -4e-05 | 0.00075 | 0 | None | 1.253/1.154 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00335 | 1.72 | 0.68 | -0.08714 | 0.857 | 0.01299 | 1.42 | 0.00035 | -0.00029 | 0 | None | 1.263/1.154 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00303 | 0.81 | 0.32 | -0.26367 | 0.571 | 0.02733 | 1.38 | 3e-05 | 0.00116 | 0 | None | 1.251/1.154 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6594 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7859 (schlechtestes Jahr ±0.1903) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1673
- P(Rendite_60 > +10 %): Brier 0.22215 vs. Basisrate 0.21053 (Skill -0.0552; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Real Estate 0.01935 (t 1.03), Communication Services 0.01695 (t 1.01), Energy 0.01611 (t 0.81), Basic Materials 0.01324 (t 0.74), Industrials 0.00543 (t 0.39), Consumer Cyclical 0.00101 (t 0.07), Technology -0.0023 (t -0.17), Consumer Defensive -0.01077 (t -0.7), Healthcare -0.01198 (t -0.84), Utilities -0.01604 (t -1.09), Financial Services -0.02205 (t -1.15)
- Regime (netto): vix_lt_20 -0.00243 (n 199), vix_ge_20 -0.00038 (n 136), spy_uptrend -0.00193 (n 260), spy_downtrend -0.00045 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00341 mit t=1.0 (< 2.0)
- Rank-IC 0.01842 (t=1.03) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02719, 'risk': 0.0096, 'liquidity': 0.00166, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01681), ('beta_126', 0.01143), ('mom_12_1', 0.00961), ('log_dollar_vol', 0.00167), ('max_ret_21', 0.00084), ('mom_3m', 0.00072)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.0417 (t 2.44), Real Estate 0.02657 (t 1.72), Consumer Defensive 0.02357 (t 1.83), Consumer Cyclical 0.02287 (t 1.77), Technology 0.02283 (t 1.98), Utilities 0.02066 (t 1.44), Communication Services 0.00189 (t 0.11), Industrials -0.0021 (t -0.17), Healthcare -0.00331 (t -0.27), Basic Materials -0.01416 (t -0.89), Energy -0.01881 (t -1.12)
- Regime (netto): vix_lt_20 -0.00317 (n 199), vix_ge_20 0.01303 (n 136), spy_uptrend 0.00031 (n 260), spy_downtrend 0.01412 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00323 mit t=0.79 (< 2.0)
- Rank-IC 0.03376 (t=1.88) nicht signifikant
- Locked-Holdout netto -0.00437 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00952, 'risk': 0.02805, 'liquidity': 0.00463, 'market': 0.00494, 'macro': 0.03802}
- Wichtigste Features: [('usd_63d_chg', 0.01854), ('beta_126', 0.01508), ('vol_60', 0.00815), ('wti_63d_chg', 0.00768), ('rev_1m', 0.00727), ('cpi_yoy', 0.00702)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.07251 (t 4.61), Real Estate 0.05242 (t 3.08), Consumer Defensive 0.05007 (t 3.8), Consumer Cyclical 0.04076 (t 3.05), Healthcare 0.034 (t 2.81), Communication Services 0.03275 (t 1.93), Technology 0.01913 (t 1.77), Utilities 0.01577 (t 1.15), Energy 0.01042 (t 0.6), Industrials 0.00907 (t 0.67), Basic Materials -0.02434 (t -1.44)
- Regime (netto): vix_lt_20 -0.00058 (n 199), vix_ge_20 0.0088 (n 136), spy_uptrend 0.00109 (n 260), spy_downtrend 0.01062 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00296 mit t=0.9 (< 2.0)
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.02184 (t=1.28) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': -0.00272, 'risk': 0.02178, 'liquidity': 0.00254, 'market': 0.0124, 'macro': 0.0006}
- Wichtigste Features: [('beta_126', 0.01541), ('spy_trend_200', 0.00873), ('vix_chg_21', 0.00399), ('vol_ratio', 0.0031), ('usd_63d_chg', 0.0028), ('log_dollar_vol', 0.00254)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.05709 (t 3.61), Communication Services 0.04125 (t 2.44), Healthcare 0.0362 (t 2.95), Consumer Cyclical 0.02807 (t 2.19), Technology 0.0226 (t 2.14), Consumer Defensive 0.01375 (t 1.08), Energy -0.00428 (t -0.27), Utilities -0.00504 (t -0.37), Industrials -0.0064 (t -0.48), Real Estate -0.00855 (t -0.49), Basic Materials -0.02343 (t -1.42)
- Regime (netto): vix_lt_20 0.00011 (n 199), vix_ge_20 0.00713 (n 136), spy_uptrend 0.001 (n 260), spy_downtrend 0.00975 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Walk-Forward netto 0.00335 mit t=1.72 (< 2.0)
- Rank-IC 0.01299 (t=1.42) nicht signifikant
- Locked-Holdout netto -0.00029 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02249, 'risk': 0.0, 'liquidity': 0.00754, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.00947), ('mom_12_1', 0.00693), ('log_dollar_vol', 0.00607), ('mom_3m', 0.00599), ('relvol_5_60', 0.00147), ('ret_5d', 0.00019)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.0718 (t 4.35), Technology 0.02737 (t 3.33), Financial Services 0.02445 (t 1.89), Real Estate 0.01709 (t 1.22), Consumer Defensive 0.01706 (t 1.58), Utilities 0.01634 (t 1.28), Healthcare 0.01442 (t 1.58), Consumer Cyclical 0.01422 (t 1.52), Basic Materials 0.0095 (t 0.66), Industrials -0.02192 (t -2.26), Energy -0.05668 (t -3.6)
- Regime (netto): vix_lt_20 0.00099 (n 199), vix_ge_20 0.00681 (n 136), spy_uptrend 0.00193 (n 260), spy_downtrend 0.00826 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00303 mit t=0.81 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.02733 (t=1.38) nicht signifikant
- Max-DD -0.26367 schlechter als Benchmark -0.22412
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.02656, 'liquidity': 0.0, 'market': 0.00608, 'macro': 0.01169}
- Wichtigste Features: [('usd_63d_chg', 0.01152), ('beta_126', 0.01006), ('vol_60', 0.00988), ('spy_mom_63', 0.00757), ('vol_20', 0.00461), ('tnx', 0.00361)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04654 (t 2.82), Consumer Cyclical 0.04186 (t 2.99), Technology 0.03257 (t 2.69), Healthcare 0.02556 (t 2.07), Energy 0.02447 (t 1.35), Consumer Defensive 0.00884 (t 0.73), Industrials 0.00856 (t 0.6), Utilities 0.0042 (t 0.3), Real Estate -0.00997 (t -0.61), Basic Materials -0.01044 (t -0.64), Communication Services -0.0205 (t -1.37)
- Regime (netto): vix_lt_20 -0.00239 (n 199), vix_ge_20 0.01096 (n 136), spy_uptrend 0.00014 (n 260), spy_downtrend 0.01305 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
