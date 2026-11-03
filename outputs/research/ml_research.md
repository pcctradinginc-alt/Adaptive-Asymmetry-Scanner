# ML-Research (Shadow) – 2026-11-03T15:37:33+00:00

Panel: 294631 Zeilen, 600 Ticker, 2014-06-06..2026-11-02 · Locked-Holdout ab 2025-07-01 · Champion: — (keiner)
Bewertung: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto 10 bp/Seite. Survivorship: PIT-Universum (762 Titel je Mitglied, 414 Änderungen; entfernte Titel mit Kursen 122/259 – Restbias: entfernte Titel ohne Yahoo-Kurse fehlen weiterhin). Keine Produktionswirkung.

| Modell | Status | WF netto | t | Sharpe | Max-DD | Jahre + | IC | IC t | 25bp | Locked netto | Forward n | Forward netto | Asym Top/Univ | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum_12_1 | valid | -0.0016 | -0.5 | -0.2 | -0.22335 | 0.571 | -0.00175 | 0.04 | -0.0046 | 0.02165 | 2 | -0.12945 | 1.121/1.153 | None |
| enet_xs20_v1 | valid | 0.00318 | 0.92 | 0.36 | -0.15942 | 0.714 | 0.01805 | 1.0 | 0.00018 | 0.00863 | 2 | -0.17559 | 1.244/1.154 | rejected_so_far |
| hgb_xs20_v1 | valid | 0.0039 | 1.01 | 0.4 | -0.15131 | 0.714 | 0.03094 | 1.77 | 0.0009 | -0.00437 | 2 | -0.1367 | 1.27/1.154 | rejected_so_far |
| hgb_asym20_v1 | valid | 0.00204 | 0.73 | 0.29 | -0.17085 | 0.714 | 0.01903 | 1.12 | -0.00096 | 0.00075 | 2 | -0.01572 | 1.247/1.154 | rejected_so_far |
| hgb_xs20_momentum_v1 | valid | 0.00232 | 1.29 | 0.51 | -0.08615 | 0.857 | 0.01365 | 1.54 | -0.00068 | -0.00029 | 2 | -0.10356 | 1.246/1.154 | rejected_so_far |
| hgb_xs20_risk_regime_v1 | valid | 0.00342 | 0.99 | 0.39 | -0.24624 | 0.714 | 0.03146 | 1.59 | 0.00042 | 0.00116 | 2 | -0.16369 | 1.258/1.154 | rejected_so_far |

## Unsicherheit (Walk-Forward-Kalibrierung)

- 80-%-Intervall der 60-Tage-Rendite deckt roh 0.6691 ab (Soll 0.8); nach konformaler Vorjahres-Korrektur 0.7818 (schlechtestes Jahr ±0.2065) -> kalibriert
- P(>+10 %) nach isotonischer Vorjahres-Rekalibrierung: Skill -0.1344
- P(Rendite_60 > +10 %): Brier 0.22106 vs. Basisrate 0.21055 (Skill -0.0499; <= 0 heißt: keine Information über die Basisrate hinaus)

## momentum_12_1

- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.0, 'liquidity': 0.0, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: []
- Sektoren (Rank-IC innerhalb Sektor): unknown -0.00175 (t -0.14)
- Regime (netto): vix_lt_20 -0.00234 (n 199), vix_ge_20 -0.00053 (n 136), spy_uptrend -0.00194 (n 260), spy_downtrend -0.00042 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {}, '2020': {}, '2021': {}, '2022': {}, '2023': {}, '2024': {}, '2025': {}}

## enet_xs20_v1

- Walk-Forward netto 0.00318 mit t=0.92 (< 2.0)
- Rank-IC 0.01805 (t=1.0) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 2/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02718, 'risk': 0.00969, 'liquidity': 0.00127, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01655), ('beta_126', 0.01149), ('mom_12_1', 0.0098), ('log_dollar_vol', 0.00127), ('ret_5d', 0.00119), ('max_ret_21', 0.00075)]
- Sektoren (Rank-IC innerhalb Sektor): unknown 0.01805 (t 1.61)
- Regime (netto): vix_lt_20 -0.00376 (n 199), vix_ge_20 0.01333 (n 136), spy_uptrend -0.00013 (n 260), spy_downtrend 0.01464 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2020': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2021': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2022': {'alpha': 0.003, 'l1_ratio': 0.5}, '2023': {'alpha': 0.003, 'l1_ratio': 0.5}, '2024': {'alpha': 0.0003, 'l1_ratio': 0.5}, '2025': {'alpha': 0.003, 'l1_ratio': 0.5}}

## hgb_xs20_v1

- Walk-Forward netto 0.0039 mit t=1.01 (< 2.0)
- Rank-IC 0.03094 (t=1.77) nicht signifikant
- Locked-Holdout netto -0.00437 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 2/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.00866, 'risk': 0.01943, 'liquidity': 0.0051, 'market': 0.00446, 'macro': 0.03276}
- Wichtigste Features: [('usd_63d_chg', 0.01564), ('beta_126', 0.01327), ('wti_63d_chg', 0.00757), ('spy_mom_63', 0.00689), ('cpi_yoy', 0.00569), ('rev_1m', 0.0056)]
- Sektoren (Rank-IC innerhalb Sektor): unknown 0.03094 (t 2.8)
- Regime (netto): vix_lt_20 -0.00067 (n 199), vix_ge_20 0.01059 (n 136), spy_uptrend 0.00116 (n 260), spy_downtrend 0.01341 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_asym20_v1

- Walk-Forward netto 0.00204 mit t=0.73 (< 2.0)
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.01903 (t=1.12) nicht signifikant
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 2/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': -3e-05, 'risk': 0.01343, 'liquidity': 0.00353, 'market': 0.01525, 'macro': 0.00164}
- Wichtigste Features: [('spy_trend_200', 0.00936), ('beta_126', 0.0068), ('cpi_yoy', 0.00644), ('tnx', 0.00419), ('log_dollar_vol', 0.00353), ('rev_1m', 0.00353)]
- Sektoren (Rank-IC innerhalb Sektor): unknown 0.01903 (t 1.63)
- Regime (netto): vix_lt_20 0.00027 (n 199), vix_ge_20 0.00464 (n 136), spy_uptrend 0.00022 (n 260), spy_downtrend 0.00835 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_momentum_v1

- Walk-Forward netto 0.00232 mit t=1.29 (< 2.0)
- bei 25 bp/Seite nicht mehr positiv
- Rank-IC 0.01365 (t=1.54) nicht signifikant
- Locked-Holdout netto -0.00029 nicht > 0
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 2/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.02956, 'risk': 0.0, 'liquidity': 0.00795, 'market': 0.0, 'macro': 0.0}
- Wichtigste Features: [('rev_1m', 0.01036), ('mom_12_1', 0.00861), ('log_dollar_vol', 0.00651), ('mom_3m', 0.005), ('ret_5d', 0.00396), ('dist_52w_high', 0.00163)]
- Sektoren (Rank-IC innerhalb Sektor): unknown 0.01365 (t 2.11)
- Regime (netto): vix_lt_20 0.00078 (n 199), vix_ge_20 0.00458 (n 136), spy_uptrend 0.00126 (n 260), spy_downtrend 0.006 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}

## hgb_xs20_risk_regime_v1

- Walk-Forward netto 0.00342 mit t=0.99 (< 2.0)
- Rank-IC 0.03146 (t=1.59) nicht signifikant
- Max-DD -0.24624 schlechter als Benchmark -0.22335
- Locked-Holdout KONTAMINIERT – nur informativ; bindend ist der Forward-Shadow
- Forward-Shadow: 2/26 fertige Kohorten
- Feature-Gruppen (IC-Verlust bei Permutation): {'momentum': 0.0, 'risk': 0.02986, 'liquidity': 0.0, 'market': 0.00361, 'macro': 0.01255}
- Wichtigste Features: [('beta_126', 0.0153), ('vol_60', 0.01271), ('usd_63d_chg', 0.00841), ('spy_mom_63', 0.00775), ('spy_trend_200', 0.00493), ('wti_63d_chg', 0.00209)]
- Sektoren (Rank-IC innerhalb Sektor): unknown 0.03146 (t 2.55)
- Regime (netto): vix_lt_20 -0.00125 (n 199), vix_ge_20 0.01025 (n 136), spy_uptrend 0.00077 (n 260), spy_downtrend 0.0126 (n 75)
- Hyperparameter je Testjahr (innere Validierung): {'2019': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2020': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2021': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2022': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2023': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2024': {'max_iter': 150, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}, '2025': {'max_iter': 400, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 300}}
