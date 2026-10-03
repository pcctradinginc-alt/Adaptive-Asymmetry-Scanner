# ML-Research (Shadow) – 2026-10-03T12:51:46+00:00

Panel: 292515 Zeilen, 600 Ticker, 2014-06-06..2026-10-02 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: PIT-Universum (758 Titel je Mitglied, 409 Änderungen; entfernte Titel mit Kursen 120/255 – Restbias: entfernte Titel ohne Yahoo-Kurse fehlen weiterhin). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | -0.0016 | -0.49 | -0.2 | -0.22412 | 0.571 | -0.00141 | 0.06 | -0.0046 | 0.02165 | 0 | None | 1.121/1.153 | None |
| enet_xs20_v1 | valid | 0.00341 | 1.01 | 0.4 | -0.15694 | 0.714 | 0.01842 | 1.03 | 0.00041 | 0.00863 | 0 | None | 1.249/1.154 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.0041 | 1.09 | 0.43 | -0.14562 | 0.714 | 0.03799 | 2.04 | 0.0011 | -0.00437 | 0 | None | 1.283/1.154 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00045 | 0.08 | 0.03 | -0.18736 | 0.714 | 0.0077 | 0.37 | -0.00255 | 0.00075 | 0 | None | 1.214/1.154 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.0034 | 1.74 | 0.69 | -0.09595 | 0.857 | 0.01268 | 1.4 | 0.0004 | -0.00029 | 0 | None | 1.262/1.154 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00304 | 0.84 | 0.33 | -0.26745 | 0.571 | 0.0283 | 1.44 | 4e-05 | 0.00116 | 0 | None | 1.253/1.154 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6582 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7887 (schlechtestes Jahr ±0.1878) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.156
- P(Rendite_60 > +10 %): Brier 0.22215 vs. Basisrate 0.21053 (Skill -0.0552; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): Real Estate 0.01935 (t 1.03), Communication Services 0.01695 (t 1.01), Energy 0.01611 (t 0.81), Basic Materials 0.01324 (t 0.74), Industrials 0.00543 (t 0.39), Consumer Cyclical 0.00101 (t 0.07), Technology -0.0023 (t -0.17), Consumer Defensive -0.01077 (t -0.7), Healthcare -0.01198 (t -0.84), Utilities -0.01604 (t -1.09), Financial Services -0.02204 (t -1.15)
- Regime (netto): vix_lt_20 -0.00243 (n 199), vix_ge_20 -0.00038 (n 136), spy_uptrend -0.00193 (n 260), spy_downtrend -0.00045 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00341 mit t=1.01 (< 2.0)
- Rank-IC 0.01842 (t=1.03) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02717, 'risk': 0.00958, 'liquidity': 0.00166, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01681), ('beta_126', 0.01143), ('mom_12_1', 0.00961), ('log_dollar_vol', 0.00166), ('max_ret_21', 0.00084), ('mom_3m', 0.00072)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04171 (t 2.44), Real Estate 0.02654 (t 1.72), Consumer Defensive 0.02363 (t 1.83), Technology 0.02284 (t 1.99), Consumer Cyclical 0.02283 (t 1.76), Utilities 0.02066 (t 1.44), Communication Services 0.00192 (t 0.11), Industrials -0.00209 (t -0.17), Healthcare -0.00331 (t -0.27), Basic Materials -0.01416 (t -0.89), Energy -0.01891 (t -1.12)
- Regime (netto): vix_lt_20 -0.00316 (n 199), vix_ge_20 0.01304 (n 136), spy_uptrend 0.00032 (n 260), spy_downtrend 0.01413 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.0041 mit t=1.09 (< 2.0)
- Locked-Holdout netto -0.00437 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.01101, 'risk': 0.03446, 'liquidity': 0.00375, 'market': 0.01181, 'macro': 0.04022}
- Wichtigste Features: [('beta_126', 0.02063), ('usd_63d_chg', 0.01847), ('spy_mom_63', 0.01019), ('vol_60', 0.01013), ('wti_63d_chg', 0.0096), ('cpi_yoy', 0.00796)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.07864 (t 4.84), Consumer Cyclical 0.05312 (t 3.83), Real Estate 0.04902 (t 2.83), Consumer Defensive 0.04365 (t 3.36), Healthcare 0.03498 (t 2.84), Communication Services 0.03334 (t 2.04), Technology 0.03031 (t 2.78), Utilities 0.02556 (t 1.86), Energy 0.01724 (t 0.99), Industrials 0.01607 (t 1.15), Basic Materials -0.017 (t -0.98)
- Regime (netto): vix_lt_20 0.00074 (n 199), vix_ge_20 0.00902 (n 136), spy_uptrend 0.00252 (n 260), spy_downtrend 0.00956 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00045 mit t=0.08 (< 2.0)
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.0077 (t=0.37) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00075, 'risk': 4e-05, 'liquidity': 0.00262, 'market': 0.01057, 'macro': 1e-05}
- Wichtigste Features: [('spy_trend_200', 0.00767), ('vol_ratio', 0.0035), ('rev_1m', 0.00346), ('usd_63d_chg', 0.00339), ('log_dollar_vol', 0.00262), ('curve_10y_3m', 0.00194)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.03213 (t 1.9), Communication Services 0.03189 (t 1.96), Healthcare 0.02599 (t 2.07), Consumer Defensive 0.01125 (t 0.9), Technology 0.00904 (t 0.9), Consumer Cyclical 0.00795 (t 0.59), Industrials -0.01514 (t -1.18), Basic Materials -0.02034 (t -1.2), Utilities -0.02108 (t -1.59), Energy -0.02153 (t -1.37), Real Estate -0.02382 (t -1.37)
- Regime (netto): vix_lt_20 -0.00213 (n 199), vix_ge_20 0.00422 (n 136), spy_uptrend -0.00116 (n 260), spy_downtrend 0.00601 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Walk-Forward netto 0.0034 mit t=1.74 (< 2.0)
- Rank-IC 0.01268 (t=1.4) nicht signifikant
- Locked-Holdout netto -0.00029 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02102, 'risk': 0.0, 'liquidity': 0.00734, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.00957), ('mom_12_1', 0.00626), ('log_dollar_vol', 0.00625), ('mom_3m', 0.00481), ('relvol_5_60', 0.00108), ('dist_52w_high', 0.00065)]
- Sektoren (Rank-IC innerhalb Sektor): Communication Services 0.06233 (t 3.76), Technology 0.02672 (t 3.29), Financial Services 0.02572 (t 1.99), Real Estate 0.01789 (t 1.29), Healthcare 0.01662 (t 1.81), Consumer Defensive 0.01446 (t 1.33), Utilities 0.01233 (t 0.94), Consumer Cyclical 0.01089 (t 1.15), Basic Materials 0.00979 (t 0.68), Industrials -0.02054 (t -2.11), Energy -0.05724 (t -3.57)
- Regime (netto): vix_lt_20 0.00151 (n 199), vix_ge_20 0.00615 (n 136), spy_uptrend 0.00205 (n 260), spy_downtrend 0.00805 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00304 mit t=0.84 (< 2.0)
- nur 0.571 der Testjahre positiv
- Rank-IC 0.0283 (t=1.44) nicht signifikant
- Max-DD -0.26745 schlechter als Benchmark -0.22412
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 0/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.03274, 'liquidity': 0.0, 'market': 0.00516, 'macro': 0.00908}
- Wichtigste Features: [('beta_126', 0.01404), ('vol_60', 0.01262), ('spy_mom_63', 0.00852), ('usd_63d_chg', 0.00739), ('vol_20', 0.0034), ('vix', 0.00288)]
- Sektoren (Rank-IC innerhalb Sektor): Financial Services 0.04666 (t 2.82), Consumer Cyclical 0.04454 (t 3.21), Technology 0.0312 (t 2.63), Healthcare 0.02534 (t 2.07), Energy 0.01949 (t 1.07), Industrials 0.0075 (t 0.54), Utilities 0.00519 (t 0.37), Real Estate 0.00108 (t 0.07), Consumer Defensive 0.00046 (t 0.04), Basic Materials -0.00751 (t -0.46), Communication Services -0.01684 (t -1.11)
- Regime (netto): vix_lt_20 -0.00178 (n 199), vix_ge_20 0.0101 (n 136), spy_uptrend 0.00045 (n 260), spy_downtrend 0.01202 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
