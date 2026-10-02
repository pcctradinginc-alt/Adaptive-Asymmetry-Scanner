# Meta-Learning-Validierung – 2026-10-02T19:11:33+00:00

Version meta-v1 · Panel-Hash 4841a6db129b4a6f · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): REJECT** – nicht erfüllt: delta_sharpe_positive, bootstrap_ci_positive, years_positive, halves_positive, not_outlier_driven, locked_delta_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0206 | 0.22 | 0.237 | -0.1724 | 0.12 | 1.063 | 0.4735 | 0.08846 | -0.07483 | 1.182 | 0.0025 | 0.24948 | 0.00567 | 0.2642 | 0.1665 | 0.3624 | 11587 |
| best_single_ex_ante | -0.0142 | -0.046 | -0.05 | -0.2745 | -0.052 | 0.992 | 0.4621 | 0.0858 | -0.0743 | 1.155 | -0.00032 | 0.24997 | 0.02404 | 0.2527 | 0.1539 | 0.3095 | 11587 |
| trailing_ic_weighted | 0.0094 | 0.136 | 0.136 | -0.2225 | 0.042 | 1.026 | 0.4744 | 0.08621 | -0.07587 | 1.136 | 0.00102 | 0.24939 | 0.00276 | 0.2595 | 0.1623 | 0.3127 | 11587 |
| meta_regime_weights | -0.004 | 0.013 | 0.014 | -0.1819 | -0.022 | 1.013 | 0.4721 | 0.08242 | -0.07276 | 1.133 | 0.0005 | 0.24985 | 0.0148 | 0.253 | 0.1566 | 0.416 | 11587 |
| meta_regime_weights__no_regime | 0.0493 | 0.482 | 0.711 | -0.1603 | 0.307 | 1.139 | 0.4827 | 0.08484 | -0.06953 | 1.22 | 0.00499 | 0.24989 | 0.01584 | 0.257 | 0.1599 | 0.4169 | 11587 |
| meta_regime_weights__no_failure_memory | 0.0013 | 0.078 | 0.078 | -0.2477 | 0.005 | 1.031 | 0.4674 | 0.08863 | -0.07542 | 1.175 | 0.00126 | 0.24931 | 0.00513 | 0.2612 | 0.167 | 0.3233 | 11587 |
| meta_regime_weights__no_disagreement | 0.0023 | 0.072 | 0.078 | -0.1884 | 0.012 | 1.023 | 0.4716 | 0.08324 | -0.07264 | 1.146 | 0.00088 | 0.2498 | 0.01464 | 0.2541 | 0.1571 | 0.4187 | 11587 |
| meta_stacking | -0.0015 | 0.059 | 0.064 | -0.2382 | -0.006 | 1.027 | 0.4706 | 0.0939 | -0.08128 | 1.155 | 0.00116 | 0.24941 | 0.00256 | 0.274 | 0.1755 | 0.308 | 11587 |
| meta_stacking__no_regime | -0.004 | 0.037 | 0.041 | -0.2229 | -0.018 | 1.022 | 0.4672 | 0.09512 | -0.08164 | 1.165 | 0.00095 | 0.24936 | 0.00221 | 0.274 | 0.1767 | 0.3357 | 11587 |
| meta_stacking__no_disagreement | 0.0007 | 0.075 | 0.084 | -0.2319 | 0.003 | 1.03 | 0.4711 | 0.09465 | -0.0819 | 1.156 | 0.00128 | 0.24954 | 0.00283 | 0.2707 | 0.1739 | 0.3098 | 11587 |
| meta_stacking__no_failure_memory | -0.0654 | -0.427 | -0.445 | -0.3874 | -0.169 | 0.902 | 0.4495 | 0.09029 | -0.08173 | 1.105 | -0.00441 | 0.24954 | 0.00251 | 0.2548 | 0.1599 | 0.3308 | 11587 |
| meta_stacking__no_sector | -0.0094 | -0.001 | -0.001 | -0.2893 | -0.033 | 1.008 | 0.4652 | 0.0987 | -0.08519 | 1.159 | 0.00035 | 0.24932 | 0.00207 | 0.2762 | 0.1838 | 0.3266 | 11587 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.266, Δ Expectancy -0.00282, Δ MaxDD -0.1021, Δ PF -0.071, Δ Hit -0.0114, Δ Brier 0.00049, Δ ECE 0.01837, Δ Prec@K -0.0115 · Monats-Δ -0.002913 CI [-0.006354, 0.000626]
- **trailing_ic_weighted**: Δ Sharpe -0.084, Δ Expectancy -0.00148, Δ MaxDD -0.0501, Δ PF -0.037, Δ Hit 0.0009, Δ Brier -9e-05, Δ ECE -0.00291, Δ Prec@K -0.0047 · Monats-Δ -0.000931 CI [-0.006734, 0.00539]
- **meta_regime_weights**: Δ Sharpe -0.207, Δ Expectancy -0.002, Δ MaxDD -0.0095, Δ PF -0.05, Δ Hit -0.0014, Δ Brier 0.00037, Δ ECE 0.00913, Δ Prec@K -0.0112 · Monats-Δ -0.002295 CI [-0.007191, 0.002353]
- **meta_regime_weights__no_regime**: Δ Sharpe 0.262, Δ Expectancy 0.00249, Δ MaxDD 0.0121, Δ PF 0.076, Δ Hit 0.0092, Δ Brier 0.00041, Δ ECE 0.01017, Δ Prec@K -0.0072 · Monats-Δ 0.0021 CI [-0.003346, 0.008208]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe -0.142, Δ Expectancy -0.00124, Δ MaxDD -0.0753, Δ PF -0.032, Δ Hit -0.0061, Δ Brier -0.00017, Δ ECE -0.00054, Δ Prec@K -0.003 · Monats-Δ -0.001514 CI [-0.005753, 0.003149]
- **meta_regime_weights__no_disagreement**: Δ Sharpe -0.148, Δ Expectancy -0.00162, Δ MaxDD -0.016, Δ PF -0.04, Δ Hit -0.0019, Δ Brier 0.00032, Δ ECE 0.00897, Δ Prec@K -0.0101 · Monats-Δ -0.001799 CI [-0.00654, 0.003215]
- **meta_stacking**: Δ Sharpe -0.161, Δ Expectancy -0.00134, Δ MaxDD -0.0658, Δ PF -0.036, Δ Hit -0.0029, Δ Brier -7e-05, Δ ECE -0.00311, Δ Prec@K 0.0098 · Monats-Δ -0.001724 CI [-0.00828, 0.004788]
- **meta_stacking__no_regime**: Δ Sharpe -0.183, Δ Expectancy -0.00155, Δ MaxDD -0.0505, Δ PF -0.041, Δ Hit -0.0063, Δ Brier -0.00012, Δ ECE -0.00346, Δ Prec@K 0.0098 · Monats-Δ -0.001989 CI [-0.009001, 0.004852]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.145, Δ Expectancy -0.00122, Δ MaxDD -0.0595, Δ PF -0.033, Δ Hit -0.0024, Δ Brier 6e-05, Δ ECE -0.00284, Δ Prec@K 0.0065 · Monats-Δ -0.001515 CI [-0.007601, 0.004327]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.647, Δ Expectancy -0.00691, Δ MaxDD -0.215, Δ PF -0.161, Δ Hit -0.024, Δ Brier 6e-05, Δ ECE -0.00316, Δ Prec@K -0.0094 · Monats-Δ -0.007273 CI [-0.013512, -0.001242]
- **meta_stacking__no_sector**: Δ Sharpe -0.221, Δ Expectancy -0.00215, Δ MaxDD -0.1169, Δ PF -0.055, Δ Hit -0.0083, Δ Brier -0.00016, Δ ECE -0.0036, Δ Prec@K 0.012 · Monats-Δ -0.002423 CI [-0.008295, 0.003353]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ❌ delta_sharpe_positive: -0.207
- ❌ bootstrap_ci_positive: [-0.007191, 0.002353]
- ❌ years_positive: 0.4
- ❌ halves_positive: [0.00118, -0.00564]
- ✅ no_regime_collapse: {'vix_lt_20': -0.00385, 'vix_ge_20': 0.00091, 'spy_uptrend': -0.00354, 'spy_downtrend': 0.00292}
- ✅ calibration_not_worse: {'brier': [0.24985, 0.24948], 'ece': [0.0148, 0.00567]}
- ❌ not_outlier_driven: -0.004736
- ❌ locked_delta_positive: -0.00149

## Locked-Holdout

- static_equal: Expectancy 0.01953, Sharpe 1.499, Trades 3111
- best_single_ex_ante: Expectancy 0.01265, Sharpe 1.46, Trades 3111
- trailing_ic_weighted: Expectancy 0.01658, Sharpe 1.292, Trades 3111
- meta_regime_weights: Expectancy 0.01805, Sharpe 1.443, Trades 3111
- meta_regime_weights__no_regime: Expectancy 0.01613, Sharpe 1.636, Trades 3111
- meta_regime_weights__no_failure_memory: Expectancy 0.01798, Sharpe 1.413, Trades 3111
- meta_regime_weights__no_disagreement: Expectancy 0.01036, Sharpe 0.779, Trades 3111
- meta_stacking: Expectancy 0.02473, Sharpe 2.733, Trades 3111
- meta_stacking__no_regime: Expectancy 0.01056, Sharpe 1.595, Trades 3111
- meta_stacking__no_disagreement: Expectancy 0.02276, Sharpe 2.378, Trades 3111
- meta_stacking__no_failure_memory: Expectancy 0.02773, Sharpe 2.623, Trades 3111
- meta_stacking__no_sector: Expectancy 0.02288, Sharpe 2.252, Trades 3111

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': 0.00499, 'sharpe': 0.482, 'brier': 0.24989, 'delta_vs_full_expectancy': 0.00449}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00126, 'sharpe': 0.078, 'brier': 0.24931, 'delta_vs_full_expectancy': 0.00076}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00088, 'sharpe': 0.072, 'brier': 0.2498, 'delta_vs_full_expectancy': 0.00038}
- meta_stacking__no_regime: {'expectancy': 0.00095, 'sharpe': 0.037, 'brier': 0.24936, 'delta_vs_full_expectancy': -0.00021}
- meta_stacking__no_disagreement: {'expectancy': 0.00128, 'sharpe': 0.075, 'brier': 0.24954, 'delta_vs_full_expectancy': 0.00012}
- meta_stacking__no_failure_memory: {'expectancy': -0.00441, 'sharpe': -0.427, 'brier': 0.24954, 'delta_vs_full_expectancy': -0.00557}
- meta_stacking__no_sector: {'expectancy': 0.00035, 'sharpe': -0.001, 'brier': 0.24932, 'delta_vs_full_expectancy': -0.00081}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.0025, 'sharpe': 0.22}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00306, 'liquid_half': -0.00151, 'less_liquid_half': 0.00267}
- meta_stacking: {'rebalance_4w': -0.00096, 'liquid_half': -0.00064, 'less_liquid_half': 0.00363}
- static_equal: {'rebalance_4w': 0.00118, 'liquid_half': 0.00146, 'less_liquid_half': 0.00473}
- best_single_ex_ante: {'rebalance_4w': -0.00275, 'liquid_half': -0.00144, 'less_liquid_half': 0.0024}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': -0.00098, 't_months': -0.26, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': -0.00673, 't_months': -0.97, 'share_years_positive': 0.2, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': -0.00177, 't_months': -0.45, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00454, 't_months': -1.01, 'share_years_positive': 0.2, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.254
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: –; versagt: –
- **hgb_asym20_v1** – funktioniert: Communication Services (IC 0.05, t 2.26), Consumer Cyclical (IC 0.0441, t 2.39); versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: –
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 14934 | 0.4626 | 0.5102 | 0.00337 | -0.00757 | -0.07344 | 0.00337 | -0.0476 | calibrated |
| 55–60 % | 230 | 0.4913 | 0.5657 | -0.00346 | -0.00132 | -0.06344 | -0.00346 | -0.0744 | overconfident |
| 60–65 % | 104 | 0.4712 | 0.6078 | 0.00625 | -0.00215 | -0.04745 | 0.00625 | -0.1367 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 8252 | 0.433 | 0.5223 | -0.00583 | -0.00994 | -0.05232 | -0.00583 | -0.0893 | overconfident |
| 55–60 % | 226 | 0.5575 | 0.586 | 0.06183 | 0.03087 | -0.1237 | 0.06183 | -0.0284 | calibrated |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 61 | 0.541 | 0.6818 | 0.0726 | 0.0459 | -0.11826 | 0.0726 | -0.1408 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 8190 | 0.4889 | 0.5043 | 0.01215 | -0.00283 | -0.09057 | 0.01215 | -0.0155 | calibrated |
| 55–60 % | 398 | 0.5754 | 0.577 | 0.05199 | 0.02903 | -0.11785 | 0.05199 | -0.0016 | calibrated |
| 60–65 % | 182 | 0.4835 | 0.6157 | 0.02478 | -0.01326 | -0.10958 | 0.02478 | -0.1322 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.008, 'recent_ic': -0.1172, 'prior_ic': 0.0711, 'trend': 'deteriorating', 'trend_t': -2.65, 'calibration_slope_recent': -0.04374, 'meta_weight': 0.6912, 'contribution': -0.00065}
- enet_xs20_v1: {'oos_ic': 0.0322, 'recent_ic': -0.0052, 'prior_ic': 0.1107, 'trend': 'stable', 'trend_t': -1.82, 'calibration_slope_recent': 0.00555, 'meta_weight': 0.0825, 'contribution': 0.00093}
- hgb_xs20_v1: {'oos_ic': 0.0235, 'recent_ic': -0.0308, 'prior_ic': 0.0517, 'trend': 'stable', 'trend_t': -1.15, 'calibration_slope_recent': -0.01543, 'meta_weight': 0.0, 'contribution': -0.0011}
- hgb_asym20_v1: {'oos_ic': 0.0216, 'recent_ic': -0.0834, 'prior_ic': 0.0369, 'trend': 'deteriorating', 'trend_t': -2.08, 'calibration_slope_recent': -0.03025, 'meta_weight': 0.2264, 'contribution': -0.00051}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0176, 'recent_ic': 0.0644, 'prior_ic': 0.0288, 'trend': 'stable', 'trend_t': 1.29, 'calibration_slope_recent': 0.02639, 'meta_weight': 0.0, 'contribution': 0.00156}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.013, 'recent_ic': 0.0425, 'prior_ic': 0.0001, 'trend': 'stable', 'trend_t': 0.99, 'calibration_slope_recent': 0.00914, 'meta_weight': 0.0, 'contribution': -0.00194}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
