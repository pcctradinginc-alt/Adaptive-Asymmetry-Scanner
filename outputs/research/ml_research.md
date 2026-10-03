# ML-Research (Shadow) – 2026-10-03T07:14:37+00:00

Panel: 292515 Zeilen, 600 Ticker, 2014-06-06..2026-10-02 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: PIT-Universum (758 Titel je Mitglied, 409 Änderungen; entfernte Titel mit Kursen 120/255 – Restbias: entfernte Titel ohne Yahoo-Kurse fehlen weiterhin). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | -0.0016 | -0.49 | -0.2 | -0.22412 | 0.571 | -0.00141 | 0.06 | -0.0046 | 0.02165 | 0 | None | 1.121/1.153 | None |
| enet_xs20_v1 | valid | 0.00341 | 1.01 | 0.4 | -0.15694 | 0.714 | 0.01842 | 1.03 | 0.00041 | 0.00863 | 0 | None | 1.249/1.154 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.00277 | 0.67 | 0.26 | -0.16947 | 0.714 | 0.03182 | 1.79 | -0.00023 | -0.00437 | 0 | None | 1.255/1.154 | rejected_so_far |
| hgb_asym20_v1 | valid | 2e-05 | -0.08 | -0.03 | -0.22144 | 0.571 | 0.01092 | 0.58 | -0.00298 | 0.00075 | 0 | None | 1.212/1.154 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00369 | 1.95 | 0.77 | -0.08417 | 0.857 | 0.0145 | 1.6 | 0.00069 | -0.00029 | 0 | None | 1.265/1.154 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00285 | 0.76 | 0.3 | -0.27613 | 0.571 | 0.02674 | 1.34 | -0.00015 | 0.00116 | 0 | None | 1.249/1.154 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6627 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7854 (schlechtestes Jahr ±0.1942) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1829
- P(Rendite_60 > +10 %): Brier 0.22234 vs. Basisrate 0.21053 (Skill -0.0561; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Real Estate 0.01935 (t 1.03), Communication Services 0.01695 (t 1.01), Energy 0.01611 (t 0.81), Basic Materials 0.01324 (t 0.74), Industrials 0.00543 (t 0.39), Consumer Cyclical 0.00101 (t 0.07), Technology -0.0023 (t -0.17), Consumer Defensive -0.01077 (t -0.7), Healthcare -0.01198 (t -0.84), Utilities -0.01604 (t -1.09), Financial Services -0.02205 (t -1.15)
- Regime (netto): vix_lt_20 -0.00243 (n 199), vix_ge_20 -0.00038 (n 136), spy_uptrend -0.00193 (n 260), spy_downtrend -0.00045 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00341 mit t=1.01 (< 2.0)
- Rank-IC 0.01842 (t=1.03) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02717, 'risk': 0.00958, 'liquidity': 0.00166, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01681), ('beta_126', 0.01143), ('mom_12_1', 0.00961), ('log_dollar_vol', 0.00166), ('max_ret_21', 0.00084), ('mom_3m', 0.00072)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04169 (t 2.44), Real Estate 0.02657 (t 1.72), Consumer Defensive 0.02358 (t 1.83), Consumer Cyclical 0.02282 (t 1.76), Technology 0.02282 (t 1.98), Utilities 0.02069 (t 1.44), Communication Services 0.00191 (t 0.11), Industrials -0.00207 (t -0.17), Healthcare -0.00331 (t -0.27), Basic Materials -0.01422 (t -0.89), Energy -0.01884 (t -1.12)
- Regime (netto): vix_lt_20 -0.00316 (n 199), vix_ge_20 0.01303 (n 136), spy_uptrend 0.00032 (n 260), spy_downtrend 0.01413 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.00277 mit t=0.67 (< 2.0)
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.03182 (t=1.79) nicht signifikant
- Locked-Holdout netto -0.00437 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00951, 'risk': 0.02507, 'liquidity': 0.00421, 'market': 0.00699, 'macro': 0.03902}
- Wichtigste Features: [('usd_63d_chg', 0.02019), ('beta_126', 0.01386), ('spy_mom_63', 0.00821), ('wti_63d_chg', 0.00789), ('cpi_yoy', 0.00716), ('rev_1m', 0.007)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.07779 (t 4.91), Consumer Cyclical 0.04871 (t 3.57), Real Estate 0.04561 (t 2.66), Consumer Defensive 0.04504 (t 3.38), Communication Services 0.04118 (t 2.48), Healthcare 0.02729 (t 2.26), Utilities 0.02359 (t 1.72), Technology 0.02172 (t 1.95), Industrials 0.01254 (t 0.92), Energy 0.00889 (t 0.5), Basic Materials -0.02791 (t -1.63)
- Regime (netto): vix_lt_20 -0.00113 (n 199), vix_ge_20 0.00847 (n 136), spy_uptrend 0.00071 (n 260), spy_downtrend 0.00991 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 2e-05 mit t=-0.08 (< 2.0)
- nur 0.571 der Testjahre positiv
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.01092 (t=0.58) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00031, 'risk': 0.00324, 'liquidity': 0.00222, 'market': 0.0111, 'macro': -0.00194}
- Wichtigste Features: [('spy_trend_200', 0.00668), ('vol_ratio', 0.00385), ('rev_1m', 0.0029), ('usd_63d_chg', 0.00289), ('log_dollar_vol', 0.00221), ('curve_10y_3m', 0.00196)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.03485 (t 2.08), Communication Services 0.03415 (t 2.06), Healthcare 0.02859 (t 2.28), Technology 0.01709 (t 1.68), Consumer Cyclical 0.01067 (t 0.79), Consumer Defensive 0.00831 (t 0.66), Industrials -0.01296 (t -1.02), Energy -0.01746 (t -1.13), Basic Materials -0.01899 (t -1.11), Utilities -0.01925 (t -1.44), Real Estate -0.02164 (t -1.25)
- Regime (netto): vix_lt_20 -0.00252 (n 199), vix_ge_20 0.00373 (n 136), spy_uptrend -0.00151 (n 260), spy_downtrend 0.00531 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Walk-Forward netto 0.00369 mit t=1.95 (< 2.0)
- Rank-IC 0.0145 (t=1.6) nicht signifikant
- Locked-Holdout netto -0.00029 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02321, 'risk': 0.0, 'liquidity': 0.00817, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.0097), ('mom_12_1', 0.00722), ('log_dollar_vol', 0.00691), ('mom_3m', 0.00575), ('relvol_5_60', 0.00126), ('dist_52w_high', 0.00089)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.06976 (t 4.17), Technology 0.03054 (t 3.77), Financial Services 0.02699 (t 2.11), Real Estate 0.01971 (t 1.42), Utilities 0.01772 (t 1.36), Basic Materials 0.01638 (t 1.17), Healthcare 0.01551 (t 1.69), Consumer Defensive 0.01464 (t 1.37), Consumer Cyclical 0.01165 (t 1.24), Industrials -0.02096 (t -2.17), Energy -0.05627 (t -3.61)
- Regime (netto): vix_lt_20 0.00177 (n 199), vix_ge_20 0.00649 (n 136), spy_uptrend 0.00229 (n 260), spy_downtrend 0.00851 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00285 mit t=0.76 (< 2.0)
- nur 0.571 der Testjahre positiv
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.02674 (t=1.34) nicht signifikant
- Max-DD -0.27613 schlechter als Benchmark -0.22412
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.02736, 'liquidity': 0.0, 'market': 0.00787, 'macro': 0.01048}
- Wichtigste Features: [('beta_126', 0.01097), ('usd_63d_chg', 0.01072), ('vol_60', 0.01066), ('spy_mom_63', 0.00775), ('spy_trend_200', 0.0042), ('vix', 0.00387)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04762 (t 2.87), Consumer Cyclical 0.04373 (t 3.13), Technology 0.02936 (t 2.43), Healthcare 0.025 (t 2.01), Energy 0.0177 (t 0.97), Consumer Defensive 0.0141 (t 1.14), Industrials 0.00626 (t 0.45), Real Estate -0.00138 (t -0.08), Utilities -0.00224 (t -0.16), Basic Materials -0.01241 (t -0.77), Communication Services -0.02537 (t -1.68)
- Regime (netto): vix_lt_20 -0.00233 (n 199), vix_ge_20 0.01042 (n 136), spy_uptrend -3e-05 (n 260), spy_downtrend 0.01284 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
