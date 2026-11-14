# Meta-Learning-Validierung – 2026-11-14T15:23:31+00:00

Version meta-v1 · Panel-Hash b04fd73fe2ee4d2c · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0048 | 0.097 | 0.111 | -0.1654 | 0.029 | 1.025 | 0.4873 | 0.06271 | -0.05815 | 1.078 | 0.00075 | 0.25026 | 0.01459 | 0.2254 | 0.124 | 0.4569 | 10986 |
| best_single_ex_ante | -0.0092 | -0.077 | -0.102 | -0.2167 | -0.043 | 0.984 | 0.4797 | 0.06152 | -0.05762 | 1.068 | -0.00047 | 0.2504 | 0.02102 | 0.2074 | 0.114 | 0.4924 | 10986 |
| trailing_ic_weighted | -0.0131 | -0.161 | -0.155 | -0.157 | -0.083 | 0.968 | 0.486 | 0.06093 | -0.05953 | 1.024 | -0.00099 | 0.24998 | 0.00336 | 0.2226 | 0.1196 | 0.4136 | 10986 |
| meta_regime_weights | 0.0113 | 0.163 | 0.198 | -0.1837 | 0.062 | 1.052 | 0.4897 | 0.06854 | -0.06252 | 1.096 | 0.00166 | 0.25021 | 0.01299 | 0.2395 | 0.142 | 0.4489 | 10986 |
| meta_regime_weights__no_regime | -0.0338 | -0.322 | -0.361 | -0.2683 | -0.126 | 0.937 | 0.4689 | 0.06649 | -0.06263 | 1.062 | -0.00209 | 0.25038 | 0.01315 | 0.2198 | 0.1272 | 0.4523 | 10986 |
| meta_regime_weights__no_failure_memory | 0.0141 | 0.188 | 0.193 | -0.1722 | 0.082 | 1.066 | 0.494 | 0.06586 | -0.06032 | 1.092 | 0.00202 | 0.24974 | 0.00595 | 0.2388 | 0.1349 | 0.3966 | 10986 |
| meta_regime_weights__no_disagreement | 0.0171 | 0.246 | 0.29 | -0.1716 | 0.1 | 1.064 | 0.4939 | 0.06707 | -0.06149 | 1.091 | 0.00201 | 0.24972 | 0.0085 | 0.2367 | 0.1395 | 0.4443 | 10986 |
| meta_stacking | 0.0042 | 0.09 | 0.107 | -0.2476 | 0.017 | 1.021 | 0.4802 | 0.07025 | -0.06359 | 1.105 | 0.00069 | 0.25019 | 0.01837 | 0.2405 | 0.1404 | 0.4785 | 10986 |
| meta_stacking__no_regime | -0.002 | 0.02 | 0.022 | -0.2619 | -0.007 | 1.001 | 0.485 | 0.07058 | -0.0664 | 1.063 | 3e-05 | 0.25006 | 0.01651 | 0.2466 | 0.1402 | 0.4681 | 10986 |
| meta_stacking__no_disagreement | -0.0176 | -0.138 | -0.176 | -0.3218 | -0.055 | 0.974 | 0.4675 | 0.07146 | -0.06442 | 1.109 | -0.0009 | 0.25038 | 0.00791 | 0.2342 | 0.138 | 0.4102 | 10986 |
| meta_stacking__no_failure_memory | 0.0025 | 0.07 | 0.086 | -0.159 | 0.016 | 1.028 | 0.4867 | 0.06949 | -0.06409 | 1.084 | 0.00093 | 0.24993 | 0.00198 | 0.2371 | 0.1425 | 0.5432 | 10986 |
| meta_stacking__no_sector | 0.0042 | 0.09 | 0.107 | -0.2476 | 0.017 | 1.021 | 0.4802 | 0.07025 | -0.06359 | 1.105 | 0.00069 | 0.25019 | 0.01837 | 0.2405 | 0.1404 | 0.4785 | 10986 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.174, Δ Expectancy -0.00122, Δ MaxDD -0.0513, Δ PF -0.041, Δ Hit -0.0076, Δ Brier 0.00014, Δ ECE 0.00643, Δ Prec@K -0.018 · Monats-Δ -0.001202 CI [-0.005773, 0.003227]
- **trailing_ic_weighted**: Δ Sharpe -0.258, Δ Expectancy -0.00174, Δ MaxDD 0.0084, Δ PF -0.057, Δ Hit -0.0013, Δ Brier -0.00028, Δ ECE -0.01123, Δ Prec@K -0.0028 · Monats-Δ -0.001597 CI [-0.005472, 0.002013]
- **meta_regime_weights**: Δ Sharpe 0.066, Δ Expectancy 0.00091, Δ MaxDD -0.0183, Δ PF 0.027, Δ Hit 0.0024, Δ Brier -5e-05, Δ ECE -0.0016, Δ Prec@K 0.0141 · Monats-Δ 0.000628 CI [-0.002647, 0.003971]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.419, Δ Expectancy -0.00284, Δ MaxDD -0.1029, Δ PF -0.088, Δ Hit -0.0184, Δ Brier 0.00012, Δ ECE -0.00144, Δ Prec@K -0.0056 · Monats-Δ -0.003196 CI [-0.006456, 5e-06]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.091, Δ Expectancy 0.00127, Δ MaxDD -0.0068, Δ PF 0.041, Δ Hit 0.0067, Δ Brier -0.00052, Δ ECE -0.00864, Δ Prec@K 0.0134 · Monats-Δ 0.000895 CI [-0.00289, 0.004618]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.149, Δ Expectancy 0.00126, Δ MaxDD -0.0062, Δ PF 0.039, Δ Hit 0.0066, Δ Brier -0.00054, Δ ECE -0.00609, Δ Prec@K 0.0113 · Monats-Δ 0.000998 CI [-0.002469, 0.004671]
- **meta_stacking**: Δ Sharpe -0.007, Δ Expectancy -6e-05, Δ MaxDD -0.0822, Δ PF -0.004, Δ Hit -0.0071, Δ Brier -7e-05, Δ ECE 0.00378, Δ Prec@K 0.0151 · Monats-Δ 2e-06 CI [-0.006401, 0.00647]
- **meta_stacking__no_regime**: Δ Sharpe -0.077, Δ Expectancy -0.00072, Δ MaxDD -0.0965, Δ PF -0.024, Δ Hit -0.0023, Δ Brier -0.0002, Δ ECE 0.00192, Δ Prec@K 0.0212 · Monats-Δ -0.000545 CI [-0.007216, 0.006021]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.235, Δ Expectancy -0.00165, Δ MaxDD -0.1564, Δ PF -0.051, Δ Hit -0.0198, Δ Brier 0.00012, Δ ECE -0.00668, Δ Prec@K 0.0088 · Monats-Δ -0.001799 CI [-0.008256, 0.004637]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.027, Δ Expectancy 0.00018, Δ MaxDD 0.0064, Δ PF 0.003, Δ Hit -0.0006, Δ Brier -0.00033, Δ ECE -0.01261, Δ Prec@K 0.0117 · Monats-Δ -0.000228 CI [-0.004716, 0.004613]
- **meta_stacking__no_sector**: Δ Sharpe -0.007, Δ Expectancy -6e-05, Δ MaxDD -0.0822, Δ PF -0.004, Δ Hit -0.0071, Δ Brier -7e-05, Δ ECE 0.00378, Δ Prec@K 0.0151 · Monats-Δ 2e-06 CI [-0.006401, 0.00647]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.066
- ❌ bootstrap_ci_positive: [-0.002647, 0.003971]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00307, 0.00419]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00012, 'vix_ge_20': 0.00215, 'spy_uptrend': 0.00069, 'spy_downtrend': 0.00159}
- ✅ calibration_not_worse: {'brier': [0.25021, 0.25026], 'ece': [0.01299, 0.01459]}
- ❌ not_outlier_driven: -0.000626
- ✅ locked_delta_positive: 0.00684

## Locked-Holdout

- static_equal: Expectancy -0.00286, Sharpe -0.175, Trades 3542
- best_single_ex_ante: Expectancy -0.0004, Sharpe 0.024, Trades 3542
- trailing_ic_weighted: Expectancy -0.01336, Sharpe -0.854, Trades 3542
- meta_regime_weights: Expectancy 0.00398, Sharpe 0.632, Trades 3542
- meta_regime_weights__no_regime: Expectancy -0.00072, Sharpe 0.095, Trades 3542
- meta_regime_weights__no_failure_memory: Expectancy -0.00411, Sharpe -0.331, Trades 3542
- meta_regime_weights__no_disagreement: Expectancy 0.00194, Sharpe 0.291, Trades 3542
- meta_stacking: Expectancy 0.0095, Sharpe 1.618, Trades 3542
- meta_stacking__no_regime: Expectancy 6e-05, Sharpe 0.192, Trades 3542
- meta_stacking__no_disagreement: Expectancy 0.00195, Sharpe 0.404, Trades 3542
- meta_stacking__no_failure_memory: Expectancy 0.00963, Sharpe 1.213, Trades 3542
- meta_stacking__no_sector: Expectancy 0.0095, Sharpe 1.618, Trades 3542

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00209, 'sharpe': -0.322, 'brier': 0.25038, 'delta_vs_full_expectancy': -0.00375}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00202, 'sharpe': 0.188, 'brier': 0.24974, 'delta_vs_full_expectancy': 0.00036}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00201, 'sharpe': 0.246, 'brier': 0.24972, 'delta_vs_full_expectancy': 0.00035}
- meta_stacking__no_regime: {'expectancy': 3e-05, 'sharpe': 0.02, 'brier': 0.25006, 'delta_vs_full_expectancy': -0.00066}
- meta_stacking__no_disagreement: {'expectancy': -0.0009, 'sharpe': -0.138, 'brier': 0.25038, 'delta_vs_full_expectancy': -0.00159}
- meta_stacking__no_failure_memory: {'expectancy': 0.00093, 'sharpe': 0.07, 'brier': 0.24993, 'delta_vs_full_expectancy': 0.00024}
- meta_stacking__no_sector: {'expectancy': 0.00069, 'sharpe': 0.09, 'brier': 0.25019, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00075, 'sharpe': 0.097}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00241, 'liquid_half': 0.00134, 'less_liquid_half': 0.00157}
- meta_stacking: {'rebalance_4w': -0.00225, 'liquid_half': 0.00174, 'less_liquid_half': 0.00019}
- static_equal: {'rebalance_4w': -0.00209, 'liquid_half': 0.00156, 'less_liquid_half': -0.00055}
- best_single_ex_ante: {'rebalance_4w': -0.00169, 'liquid_half': -0.00102, 'less_liquid_half': -0.00057}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00098, 't_months': 0.33, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00818, 't_months': 2.1, 'share_years_positive': 0.8, 'empirically_supported': True}
- dis_rank_range: {'low_minus_high_mean': 0.00175, 't_months': 0.6, 'share_years_positive': 0.8, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -4e-05, 't_months': -0.09, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: True
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2752
- current_level: HIGH

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: –; versagt: –
- **hgb_asym20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: –
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 30685 | 0.4695 | 0.5166 | -0.0024 | -0.00489 | -0.06045 | -0.0024 | -0.0472 | calibrated |
| 55–60 % | 156 | 0.4744 | 0.5616 | -0.00552 | -0.00683 | -0.05764 | -0.00552 | -0.0873 | low_n |
| 60–65 % | 317 | 0.4479 | 0.6249 | -0.00894 | -0.01306 | -0.0899 | -0.00894 | -0.177 | overconfident |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 23982 | 0.4774 | 0.5127 | -0.00077 | -0.00359 | -0.06085 | -0.00077 | -0.0353 | calibrated |
| 55–60 % | 440 | 0.4932 | 0.576 | 0.00335 | -0.00152 | -0.09316 | 0.00335 | -0.0828 | overconfident |
| 60–65 % | 22 | 0.6818 | 0.625 | 0.04036 | 0.08237 | -0.12524 | 0.04036 | 0.0568 | low_n |
| 65–70 % | 71 | 0.5493 | 0.6818 | 0.02982 | 0.0166 | -0.08648 | 0.02982 | -0.1325 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 23282 | 0.4766 | 0.5054 | -0.00157 | -0.00333 | -0.0588 | -0.00157 | -0.0287 | calibrated |
| 55–60 % | 2292 | 0.4974 | 0.5599 | 0.00259 | -0.00023 | -0.09307 | 0.00259 | -0.0625 | overconfident |
| 60–65 % | 4 | 0.0 | 0.6471 | -0.04139 | -0.04264 | -0.06697 | -0.04139 | -0.6471 | low_n |
| 65–70 % | 103 | 0.5437 | 0.6753 | 0.00824 | 0.00616 | -0.10733 | 0.00824 | -0.1316 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0004, 'recent_ic': -0.0644, 'prior_ic': 0.0331, 'trend': 'stable', 'trend_t': -1.09, 'calibration_slope_recent': -0.0404, 'meta_weight': 0.3107, 'contribution': -0.001}
- enet_xs20_v1: {'oos_ic': 0.015, 'recent_ic': -0.1912, 'prior_ic': 0.0503, 'trend': 'deteriorating', 'trend_t': -2.46, 'calibration_slope_recent': -0.06911, 'meta_weight': 0.0, 'contribution': -0.00022}
- hgb_xs20_v1: {'oos_ic': 0.0236, 'recent_ic': 0.1636, 'prior_ic': -0.0385, 'trend': 'improving', 'trend_t': 4.5, 'calibration_slope_recent': 0.0499, 'meta_weight': 0.4344, 'contribution': 0.00021}
- hgb_asym20_v1: {'oos_ic': 0.0159, 'recent_ic': 0.2046, 'prior_ic': -0.0433, 'trend': 'improving', 'trend_t': 5.89, 'calibration_slope_recent': 0.07063, 'meta_weight': 0.0, 'contribution': -0.00112}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0106, 'recent_ic': 0.0385, 'prior_ic': -0.012, 'trend': 'stable', 'trend_t': 1.41, 'calibration_slope_recent': 0.01024, 'meta_weight': 0.0, 'contribution': 8e-05}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0287, 'recent_ic': 0.0995, 'prior_ic': -0.0079, 'trend': 'stable', 'trend_t': 1.51, 'calibration_slope_recent': 0.05606, 'meta_weight': 0.2549, 'contribution': 0.00067}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 18, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
