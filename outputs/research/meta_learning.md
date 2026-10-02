# Meta-Learning-Validierung – 2026-10-02T15:03:11+00:00

Version meta-v1 · Panel-Hash e722d05e49dcc5d0 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, years_positive, not_outlier_driven, locked_delta_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0277 | 0.264 | 0.294 | -0.1794 | 0.155 | 1.08 | 0.4759 | 0.09037 | -0.07596 | 1.19 | 0.00319 | 0.24952 | 0.0062 | 0.2671 | 0.17 | 0.3529 | 11587 |
| best_single_ex_ante | -0.0265 | -0.137 | -0.147 | -0.3262 | -0.081 | 0.967 | 0.4581 | 0.08591 | -0.07508 | 1.144 | -0.00133 | 0.24993 | 0.02319 | 0.2477 | 0.1509 | 0.3189 | 11587 |
| trailing_ic_weighted | 0.0038 | 0.093 | 0.089 | -0.2142 | 0.018 | 1.013 | 0.4724 | 0.08616 | -0.07617 | 1.131 | 0.00052 | 0.24951 | 0.00238 | 0.2595 | 0.1616 | 0.3059 | 11587 |
| meta_regime_weights | 0.0477 | 0.472 | 0.539 | -0.1722 | 0.277 | 1.126 | 0.4836 | 0.08895 | -0.07397 | 1.202 | 0.00481 | 0.24972 | 0.01321 | 0.2695 | 0.1711 | 0.3694 | 11587 |
| meta_regime_weights__no_regime | 0.0467 | 0.445 | 0.605 | -0.1593 | 0.293 | 1.133 | 0.4812 | 0.08681 | -0.07109 | 1.221 | 0.0049 | 0.25001 | 0.01597 | 0.2601 | 0.16 | 0.402 | 11587 |
| meta_regime_weights__no_failure_memory | 0.0075 | 0.122 | 0.123 | -0.2375 | 0.031 | 1.045 | 0.4709 | 0.08936 | -0.07608 | 1.175 | 0.00182 | 0.24931 | 0.00296 | 0.267 | 0.17 | 0.3216 | 11587 |
| meta_regime_weights__no_disagreement | 0.045 | 0.455 | 0.521 | -0.1742 | 0.258 | 1.119 | 0.4812 | 0.08849 | -0.07335 | 1.206 | 0.00453 | 0.24988 | 0.01295 | 0.2675 | 0.1684 | 0.3936 | 11587 |
| meta_stacking | -0.0015 | 0.057 | 0.064 | -0.2343 | -0.006 | 1.029 | 0.4756 | 0.09073 | -0.07999 | 1.134 | 0.00121 | 0.24936 | 0.00204 | 0.2677 | 0.1719 | 0.3488 | 11587 |
| meta_stacking__no_regime | 0.0079 | 0.124 | 0.135 | -0.1992 | 0.04 | 1.048 | 0.4771 | 0.09264 | -0.08068 | 1.148 | 0.00201 | 0.24934 | 0.00224 | 0.2725 | 0.1763 | 0.3335 | 11587 |
| meta_stacking__no_disagreement | -0.0016 | 0.058 | 0.063 | -0.2157 | -0.007 | 1.027 | 0.4723 | 0.09202 | -0.08015 | 1.148 | 0.00116 | 0.24944 | 0.00672 | 0.2667 | 0.1716 | 0.3344 | 11587 |
| meta_stacking__no_failure_memory | -0.0297 | -0.147 | -0.15 | -0.2926 | -0.102 | 0.967 | 0.4595 | 0.0924 | -0.08124 | 1.137 | -0.00146 | 0.24935 | 0.00203 | 0.266 | 0.1696 | 0.3322 | 11587 |
| meta_stacking__no_sector | -0.0035 | 0.044 | 0.049 | -0.249 | -0.014 | 1.02 | 0.4688 | 0.09549 | -0.08265 | 1.155 | 0.00087 | 0.24961 | 0.01294 | 0.2712 | 0.177 | 0.3574 | 11587 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.401, Δ Expectancy -0.00452, Δ MaxDD -0.1468, Δ PF -0.113, Δ Hit -0.0178, Δ Brier 0.00041, Δ ECE 0.01699, Δ Prec@K -0.0194 · Monats-Δ -0.004602 CI [-0.008711, -0.000308]
- **trailing_ic_weighted**: Δ Sharpe -0.171, Δ Expectancy -0.00267, Δ MaxDD -0.0348, Δ PF -0.067, Δ Hit -0.0035, Δ Brier -1e-05, Δ ECE -0.00382, Δ Prec@K -0.0076 · Monats-Δ -0.002103 CI [-0.008243, 0.004393]
- **meta_regime_weights**: Δ Sharpe 0.208, Δ Expectancy 0.00162, Δ MaxDD 0.0072, Δ PF 0.046, Δ Hit 0.0077, Δ Brier 0.0002, Δ ECE 0.00701, Δ Prec@K 0.0024 · Monats-Δ 0.00131 CI [-0.004379, 0.00736]
- **meta_regime_weights__no_regime**: Δ Sharpe 0.181, Δ Expectancy 0.00171, Δ MaxDD 0.0201, Δ PF 0.053, Δ Hit 0.0053, Δ Brier 0.00049, Δ ECE 0.00977, Δ Prec@K -0.007 · Monats-Δ 0.001276 CI [-0.003945, 0.006869]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe -0.142, Δ Expectancy -0.00137, Δ MaxDD -0.0581, Δ PF -0.035, Δ Hit -0.005, Δ Brier -0.00021, Δ ECE -0.00324, Δ Prec@K -0.0001 · Monats-Δ -0.001609 CI [-0.006117, 0.003229]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.191, Δ Expectancy 0.00134, Δ MaxDD 0.0052, Δ PF 0.039, Δ Hit 0.0053, Δ Brier 0.00036, Δ ECE 0.00675, Δ Prec@K 0.0004 · Monats-Δ 0.001079 CI [-0.004819, 0.007334]
- **meta_stacking**: Δ Sharpe -0.207, Δ Expectancy -0.00198, Δ MaxDD -0.0549, Δ PF -0.051, Δ Hit -0.0003, Δ Brier -0.00016, Δ ECE -0.00416, Δ Prec@K 0.0006 · Monats-Δ -0.002408 CI [-0.008387, 0.003771]
- **meta_stacking__no_regime**: Δ Sharpe -0.14, Δ Expectancy -0.00118, Δ MaxDD -0.0198, Δ PF -0.032, Δ Hit 0.0012, Δ Brier -0.00018, Δ ECE -0.00396, Δ Prec@K 0.0054 · Monats-Δ -0.001739 CI [-0.008138, 0.004409]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.206, Δ Expectancy -0.00203, Δ MaxDD -0.0363, Δ PF -0.053, Δ Hit -0.0036, Δ Brier -8e-05, Δ ECE 0.00052, Δ Prec@K -0.0004 · Monats-Δ -0.002388 CI [-0.00813, 0.003137]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.411, Δ Expectancy -0.00465, Δ MaxDD -0.1132, Δ PF -0.113, Δ Hit -0.0164, Δ Brier -0.00017, Δ ECE -0.00417, Δ Prec@K -0.0011 · Monats-Δ -0.004792 CI [-0.010521, 0.000556]
- **meta_stacking__no_sector**: Δ Sharpe -0.22, Δ Expectancy -0.00232, Δ MaxDD -0.0696, Δ PF -0.06, Δ Hit -0.0071, Δ Brier 9e-05, Δ ECE 0.00674, Δ Prec@K 0.0041 · Monats-Δ -0.002558 CI [-0.008041, 0.003351]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.208
- ❌ bootstrap_ci_positive: [-0.004379, 0.00736]
- ❌ years_positive: 0.4
- ✅ halves_positive: [0.00171, 0.00092]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00122, 'vix_ge_20': 0.00222, 'spy_uptrend': 0.00136, 'spy_downtrend': 0.00244}
- ✅ calibration_not_worse: {'brier': [0.24972, 0.24952], 'ece': [0.01321, 0.0062]}
- ❌ not_outlier_driven: -0.00199
- ❌ locked_delta_positive: -0.01003

## Locked-Holdout

- static_equal: Expectancy 0.01947, Sharpe 1.504, Trades 3111
- best_single_ex_ante: Expectancy 0.03908, Sharpe 2.347, Trades 3111
- trailing_ic_weighted: Expectancy 0.01353, Sharpe 1.085, Trades 3111
- meta_regime_weights: Expectancy 0.00944, Sharpe 0.766, Trades 3111
- meta_regime_weights__no_regime: Expectancy 0.01163, Sharpe 1.185, Trades 3111
- meta_regime_weights__no_failure_memory: Expectancy 0.00987, Sharpe 0.788, Trades 3111
- meta_regime_weights__no_disagreement: Expectancy 0.00897, Sharpe 0.69, Trades 3111
- meta_stacking: Expectancy 0.02113, Sharpe 2.955, Trades 3111
- meta_stacking__no_regime: Expectancy 0.01263, Sharpe 2.47, Trades 3111
- meta_stacking__no_disagreement: Expectancy 0.0272, Sharpe 3.354, Trades 3111
- meta_stacking__no_failure_memory: Expectancy 0.02978, Sharpe 2.368, Trades 3111
- meta_stacking__no_sector: Expectancy 0.02133, Sharpe 2.224, Trades 3111

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': 0.0049, 'sharpe': 0.445, 'brier': 0.25001, 'delta_vs_full_expectancy': 9e-05}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00182, 'sharpe': 0.122, 'brier': 0.24931, 'delta_vs_full_expectancy': -0.00299}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00453, 'sharpe': 0.455, 'brier': 0.24988, 'delta_vs_full_expectancy': -0.00028}
- meta_stacking__no_regime: {'expectancy': 0.00201, 'sharpe': 0.124, 'brier': 0.24934, 'delta_vs_full_expectancy': 0.0008}
- meta_stacking__no_disagreement: {'expectancy': 0.00116, 'sharpe': 0.058, 'brier': 0.24944, 'delta_vs_full_expectancy': -5e-05}
- meta_stacking__no_failure_memory: {'expectancy': -0.00146, 'sharpe': -0.147, 'brier': 0.24935, 'delta_vs_full_expectancy': -0.00267}
- meta_stacking__no_sector: {'expectancy': 0.00087, 'sharpe': 0.044, 'brier': 0.24961, 'delta_vs_full_expectancy': -0.00034}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00319, 'sharpe': 0.264}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.0039, 'liquid_half': 0.00318, 'less_liquid_half': 0.00622}
- meta_stacking: {'rebalance_4w': -0.0002, 'liquid_half': -7e-05, 'less_liquid_half': 0.00325}
- static_equal: {'rebalance_4w': 0.00095, 'liquid_half': 0.00297, 'less_liquid_half': 0.00433}
- best_single_ex_ante: {'rebalance_4w': -0.00352, 'liquid_half': -0.00098, 'less_liquid_half': 0.00048}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00054, 't_months': 0.13, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': -0.00968, 't_months': -1.3, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': -0.00113, 't_months': -0.32, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00791, 't_months': -1.79, 'share_years_positive': 0.2, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.254
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: –; versagt: –
- **hgb_asym20_v1** – funktioniert: Communication Services (IC 0.055, t 2.35), Consumer Cyclical (IC 0.0489, t 2.41); versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: –
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15659 | 0.4606 | 0.5053 | 0.00177 | -0.0081 | -0.07411 | 0.00177 | -0.0447 | calibrated |
| 55–60 % | 356 | 0.4635 | 0.5717 | 0.00112 | -0.00581 | -0.06874 | 0.00112 | -0.1082 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 61 | 0.5738 | 0.6818 | 0.12717 | 0.0222 | -0.11435 | 0.12717 | -0.108 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15702 | 0.4462 | 0.5151 | -0.00311 | -0.00904 | -0.06244 | -0.00311 | -0.0689 | overconfident |
| 55–60 % | 670 | 0.4955 | 0.5785 | 0.02059 | -0.00268 | -0.09728 | 0.02059 | -0.083 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 144 | 0.5694 | 0.6806 | 0.07835 | 0.05488 | -0.12044 | 0.07835 | -0.1111 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 3542 | 0.5192 | 0.5258 | 0.01488 | 0.00631 | -0.10292 | 0.01488 | -0.0066 | calibrated |
| 55–60 % | 22 | 0.6818 | 0.5577 | 0.06109 | 0.03596 | -0.13446 | 0.06109 | 0.1241 | low_n |
| 60–65 % | 120 | 0.5667 | 0.6445 | 0.05465 | 0.04011 | -0.08543 | 0.05465 | -0.0778 | low_n |
| 65–70 % | 63 | 0.381 | 0.6806 | -0.01197 | -0.02772 | -0.10923 | -0.01197 | -0.2997 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.008, 'recent_ic': -0.1172, 'prior_ic': 0.0711, 'trend': 'deteriorating', 'trend_t': -2.65, 'calibration_slope_recent': -0.04374, 'meta_weight': 0.5104, 'contribution': 0.00014}
- enet_xs20_v1: {'oos_ic': 0.0322, 'recent_ic': -0.0052, 'prior_ic': 0.1107, 'trend': 'stable', 'trend_t': -1.82, 'calibration_slope_recent': 0.00555, 'meta_weight': 0.0654, 'contribution': 0.00136}
- hgb_xs20_v1: {'oos_ic': 0.0205, 'recent_ic': -0.0071, 'prior_ic': 0.0278, 'trend': 'stable', 'trend_t': -0.53, 'calibration_slope_recent': -0.00737, 'meta_weight': 0.0, 'contribution': -0.0002}
- hgb_asym20_v1: {'oos_ic': 0.0171, 'recent_ic': -0.0309, 'prior_ic': 0.0157, 'trend': 'stable', 'trend_t': -1.11, 'calibration_slope_recent': -0.01039, 'meta_weight': 0.4243, 'contribution': 2e-05}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0166, 'recent_ic': 0.0632, 'prior_ic': 0.0282, 'trend': 'stable', 'trend_t': 1.3, 'calibration_slope_recent': 0.02596, 'meta_weight': 0.0, 'contribution': 0.00222}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.019, 'recent_ic': -0.0263, 'prior_ic': 0.0377, 'trend': 'stable', 'trend_t': -1.13, 'calibration_slope_recent': -0.00638, 'meta_weight': 0.0, 'contribution': -0.00151}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
