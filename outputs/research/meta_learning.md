# Meta-Learning-Validierung – 2026-11-03T16:00:45+00:00

Version meta-v1 · Panel-Hash edf9fb6fa64a48b4 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): REJECT** – nicht erfüllt: delta_sharpe_positive, bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0156 | 0.218 | 0.255 | -0.1739 | 0.09 | 1.056 | 0.4881 | 0.064 | -0.05778 | 1.108 | 0.00166 | 0.25024 | 0.01576 | 0.2301 | 0.1261 | 0.4602 | 10986 |
| best_single_ex_ante | -0.0182 | -0.179 | -0.208 | -0.2475 | -0.074 | 0.961 | 0.4739 | 0.06283 | -0.0589 | 1.067 | -0.00122 | 0.2505 | 0.02029 | 0.21 | 0.1188 | 0.4838 | 10986 |
| trailing_ic_weighted | -0.0098 | -0.105 | -0.103 | -0.1544 | -0.064 | 0.977 | 0.487 | 0.06179 | -0.06002 | 1.03 | -0.0007 | 0.25003 | 0.00277 | 0.2252 | 0.1234 | 0.3998 | 10986 |
| meta_regime_weights | 0.0101 | 0.155 | 0.2 | -0.153 | 0.066 | 1.045 | 0.4836 | 0.06926 | -0.06209 | 1.115 | 0.00143 | 0.25028 | 0.01203 | 0.2345 | 0.142 | 0.4479 | 10986 |
| meta_regime_weights__no_regime | -0.0366 | -0.363 | -0.386 | -0.2703 | -0.136 | 0.93 | 0.465 | 0.06629 | -0.06195 | 1.07 | -0.00231 | 0.25046 | 0.01438 | 0.2175 | 0.1257 | 0.4452 | 10986 |
| meta_regime_weights__no_failure_memory | 0.0082 | 0.13 | 0.143 | -0.1311 | 0.062 | 1.049 | 0.4883 | 0.06581 | -0.05988 | 1.099 | 0.00149 | 0.24974 | 0.00641 | 0.233 | 0.132 | 0.4193 | 10986 |
| meta_regime_weights__no_disagreement | 0.0131 | 0.198 | 0.256 | -0.1545 | 0.085 | 1.05 | 0.4844 | 0.06885 | -0.06164 | 1.117 | 0.00158 | 0.25005 | 0.00994 | 0.2356 | 0.1413 | 0.4415 | 10986 |
| meta_stacking | -0.0056 | -0.014 | -0.018 | -0.2397 | -0.023 | 1.003 | 0.4799 | 0.07263 | -0.06681 | 1.087 | 0.0001 | 0.24991 | 0.00128 | 0.2457 | 0.145 | 0.4623 | 10986 |
| meta_stacking__no_regime | -0.0059 | -0.016 | -0.019 | -0.251 | -0.023 | 1.004 | 0.48 | 0.0723 | -0.06649 | 1.087 | 0.00013 | 0.25017 | 0.0156 | 0.2452 | 0.1443 | 0.4232 | 10986 |
| meta_stacking__no_disagreement | -0.0134 | -0.115 | -0.127 | -0.2614 | -0.051 | 0.981 | 0.4792 | 0.0658 | -0.06172 | 1.066 | -0.00062 | 0.24988 | 0.0103 | 0.2271 | 0.1259 | 0.4412 | 10986 |
| meta_stacking__no_failure_memory | 0.0086 | 0.152 | 0.173 | -0.1338 | 0.064 | 1.046 | 0.4843 | 0.0685 | -0.06148 | 1.114 | 0.00146 | 0.24996 | 0.00108 | 0.238 | 0.1397 | 0.5739 | 10986 |
| meta_stacking__no_sector | -0.0056 | -0.014 | -0.018 | -0.2397 | -0.023 | 1.003 | 0.4799 | 0.07263 | -0.06681 | 1.087 | 0.0001 | 0.24991 | 0.00128 | 0.2457 | 0.145 | 0.4623 | 10986 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.397, Δ Expectancy -0.00288, Δ MaxDD -0.0736, Δ PF -0.095, Δ Hit -0.0142, Δ Brier 0.00026, Δ ECE 0.00453, Δ Prec@K -0.0201 · Monats-Δ -0.002859 CI [-0.007105, 0.001674]
- **trailing_ic_weighted**: Δ Sharpe -0.323, Δ Expectancy -0.00236, Δ MaxDD 0.0195, Δ PF -0.079, Δ Hit -0.0011, Δ Brier -0.00021, Δ ECE -0.01299, Δ Prec@K -0.0049 · Monats-Δ -0.002227 CI [-0.006351, 0.00156]
- **meta_regime_weights**: Δ Sharpe -0.063, Δ Expectancy -0.00023, Δ MaxDD 0.0209, Δ PF -0.011, Δ Hit -0.0045, Δ Brier 4e-05, Δ ECE -0.00373, Δ Prec@K 0.0044 · Monats-Δ -0.000457 CI [-0.00408, 0.003174]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.581, Δ Expectancy -0.00397, Δ MaxDD -0.0964, Δ PF -0.126, Δ Hit -0.0231, Δ Brier 0.00022, Δ ECE -0.00138, Δ Prec@K -0.0126 · Monats-Δ -0.004378 CI [-0.008175, -0.000686]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe -0.088, Δ Expectancy -0.00017, Δ MaxDD 0.0428, Δ PF -0.007, Δ Hit 0.0002, Δ Brier -0.0005, Δ ECE -0.00935, Δ Prec@K 0.0029 · Monats-Δ -0.000531 CI [-0.004305, 0.00331]
- **meta_regime_weights__no_disagreement**: Δ Sharpe -0.02, Δ Expectancy -8e-05, Δ MaxDD 0.0194, Δ PF -0.006, Δ Hit -0.0037, Δ Brier -0.00019, Δ ECE -0.00582, Δ Prec@K 0.0055 · Monats-Δ -0.000257 CI [-0.00393, 0.003648]
- **meta_stacking**: Δ Sharpe -0.232, Δ Expectancy -0.00156, Δ MaxDD -0.0658, Δ PF -0.053, Δ Hit -0.0082, Δ Brier -0.00033, Δ ECE -0.01448, Δ Prec@K 0.0156 · Monats-Δ -0.001718 CI [-0.007438, 0.004106]
- **meta_stacking__no_regime**: Δ Sharpe -0.234, Δ Expectancy -0.00153, Δ MaxDD -0.0771, Δ PF -0.052, Δ Hit -0.0081, Δ Brier -7e-05, Δ ECE -0.00016, Δ Prec@K 0.0151 · Monats-Δ -0.001734 CI [-0.008269, 0.004993]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.333, Δ Expectancy -0.00228, Δ MaxDD -0.0875, Δ PF -0.075, Δ Hit -0.0089, Δ Brier -0.00036, Δ ECE -0.00546, Δ Prec@K -0.003 · Monats-Δ -0.002432 CI [-0.00757, 0.002719]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.066, Δ Expectancy -0.0002, Δ MaxDD 0.0401, Δ PF -0.01, Δ Hit -0.0038, Δ Brier -0.00028, Δ ECE -0.01468, Δ Prec@K 0.0079 · Monats-Δ -0.000681 CI [-0.00437, 0.00339]
- **meta_stacking__no_sector**: Δ Sharpe -0.232, Δ Expectancy -0.00156, Δ MaxDD -0.0658, Δ PF -0.053, Δ Hit -0.0082, Δ Brier -0.00033, Δ ECE -0.01448, Δ Prec@K 0.0156 · Monats-Δ -0.001718 CI [-0.007438, 0.004106]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ❌ delta_sharpe_positive: -0.063
- ❌ bootstrap_ci_positive: [-0.00408, 0.003174]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.0038, 0.00277]
- ✅ no_regime_collapse: {'vix_lt_20': -0.00051, 'vix_ge_20': 0.00021, 'spy_uptrend': 0.00012, 'spy_downtrend': -0.00134}
- ✅ calibration_not_worse: {'brier': [0.25028, 0.25024], 'ece': [0.01203, 0.01576]}
- ❌ not_outlier_driven: -0.002024
- ✅ locked_delta_positive: 0.00454

## Locked-Holdout

- static_equal: Expectancy 0.00285, Sharpe 0.26, Trades 3392
- best_single_ex_ante: Expectancy 0.00133, Sharpe 0.29, Trades 3392
- trailing_ic_weighted: Expectancy -0.00528, Sharpe -0.689, Trades 3392
- meta_regime_weights: Expectancy 0.00739, Sharpe 0.684, Trades 3392
- meta_regime_weights__no_regime: Expectancy -0.00022, Sharpe -0.145, Trades 3392
- meta_regime_weights__no_failure_memory: Expectancy 0.00221, Sharpe 0.162, Trades 3392
- meta_regime_weights__no_disagreement: Expectancy 0.00578, Sharpe 0.689, Trades 3392
- meta_stacking: Expectancy 0.00741, Sharpe 0.73, Trades 3392
- meta_stacking__no_regime: Expectancy 0.00879, Sharpe 1.021, Trades 3392
- meta_stacking__no_disagreement: Expectancy 0.01133, Sharpe 1.444, Trades 3392
- meta_stacking__no_failure_memory: Expectancy 0.01538, Sharpe 1.721, Trades 3392
- meta_stacking__no_sector: Expectancy 0.00741, Sharpe 0.73, Trades 3392

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00231, 'sharpe': -0.363, 'brier': 0.25046, 'delta_vs_full_expectancy': -0.00374}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00149, 'sharpe': 0.13, 'brier': 0.24974, 'delta_vs_full_expectancy': 6e-05}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00158, 'sharpe': 0.198, 'brier': 0.25005, 'delta_vs_full_expectancy': 0.00015}
- meta_stacking__no_regime: {'expectancy': 0.00013, 'sharpe': -0.016, 'brier': 0.25017, 'delta_vs_full_expectancy': 3e-05}
- meta_stacking__no_disagreement: {'expectancy': -0.00062, 'sharpe': -0.115, 'brier': 0.24988, 'delta_vs_full_expectancy': -0.00072}
- meta_stacking__no_failure_memory: {'expectancy': 0.00146, 'sharpe': 0.152, 'brier': 0.24996, 'delta_vs_full_expectancy': 0.00136}
- meta_stacking__no_sector: {'expectancy': 0.0001, 'sharpe': -0.014, 'brier': 0.24991, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00166, 'sharpe': 0.218}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00213, 'liquid_half': 0.00091, 'less_liquid_half': -0.00019}
- meta_stacking: {'rebalance_4w': -0.00201, 'liquid_half': -0.00078, 'less_liquid_half': -0.00096}
- static_equal: {'rebalance_4w': -0.00049, 'liquid_half': -7e-05, 'less_liquid_half': -0.00168}
- best_single_ex_ante: {'rebalance_4w': -0.0011, 'liquid_half': -0.00145, 'less_liquid_half': -0.00178}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.0005, 't_months': 0.17, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00413, 't_months': 1.13, 'share_years_positive': 0.8, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00065, 't_months': 0.21, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00034, 't_months': -0.16, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.1963
- current_level: LOW

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
| 50–55 % | 25470 | 0.4706 | 0.5162 | -0.00168 | -0.00448 | -0.05938 | -0.00168 | -0.0456 | calibrated |
| 55–60 % | 204 | 0.4657 | 0.5909 | 0.00149 | -0.01363 | -0.09703 | 0.00149 | -0.1252 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 52 | 0.3846 | 0.6875 | -0.01097 | -0.01378 | -0.05453 | -0.01097 | -0.3029 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 18413 | 0.4752 | 0.5194 | 0.00048 | -0.00399 | -0.06301 | 0.00048 | -0.0442 | calibrated |
| 55–60 % | 362 | 0.4696 | 0.5574 | 0.0003 | -0.01011 | -0.09753 | 0.0003 | -0.0878 | overconfident |
| 60–65 % | 68 | 0.5147 | 0.6364 | 0.02987 | 0.01233 | -0.09448 | 0.02987 | -0.1217 | low_n |
| 65–70 % | 104 | 0.5577 | 0.6562 | 0.0221 | 0.01786 | -0.1041 | 0.0221 | -0.0986 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 12199 | 0.4892 | 0.5135 | 0.00385 | -0.00203 | -0.08034 | 0.00385 | -0.0243 | calibrated |
| 55–60 % | 0 | None | None | None | None | None | None | None | empty |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0029, 'recent_ic': -0.0365, 'prior_ic': 0.0372, 'trend': 'stable', 'trend_t': -0.87, 'calibration_slope_recent': -0.01706, 'meta_weight': 0.1979, 'contribution': 0.00065}
- enet_xs20_v1: {'oos_ic': 0.0168, 'recent_ic': -0.1778, 'prior_ic': 0.0676, 'trend': 'deteriorating', 'trend_t': -2.5, 'calibration_slope_recent': -0.04937, 'meta_weight': 0.0, 'contribution': 0.0006}
- hgb_xs20_v1: {'oos_ic': 0.0228, 'recent_ic': 0.1041, 'prior_ic': -0.0476, 'trend': 'improving', 'trend_t': 2.92, 'calibration_slope_recent': 0.02472, 'meta_weight': 0.3383, 'contribution': 0.00152}
- hgb_asym20_v1: {'oos_ic': 0.0137, 'recent_ic': 0.1222, 'prior_ic': -0.0457, 'trend': 'improving', 'trend_t': 2.6, 'calibration_slope_recent': 0.03558, 'meta_weight': 0.1444, 'contribution': -0.00031}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0118, 'recent_ic': 0.0423, 'prior_ic': -0.0062, 'trend': 'stable', 'trend_t': 1.49, 'calibration_slope_recent': 0.01126, 'meta_weight': 0.0, 'contribution': 0.00105}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0296, 'recent_ic': 0.092, 'prior_ic': 0.0089, 'trend': 'stable', 'trend_t': 1.37, 'calibration_slope_recent': 0.0323, 'meta_weight': 0.3194, 'contribution': 0.00071}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
