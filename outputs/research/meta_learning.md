# Meta-Learning-Validierung – 2026-10-03T12:20:20+00:00

Version meta-v1 · Panel-Hash 3ca11bd4b9919d4a · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0074 | 0.127 | 0.155 | -0.1712 | 0.043 | 1.032 | 0.4861 | 0.06238 | -0.05718 | 1.091 | 0.00094 | 0.25028 | 0.01452 | 0.2225 | 0.1236 | 0.4631 | 11059 |
| best_single_ex_ante | -0.0086 | -0.062 | -0.077 | -0.2196 | -0.039 | 0.983 | 0.4766 | 0.06357 | -0.05889 | 1.079 | -0.00052 | 0.25055 | 0.02011 | 0.214 | 0.1227 | 0.4688 | 11059 |
| trailing_ic_weighted | -0.0138 | -0.173 | -0.173 | -0.1692 | -0.081 | 0.965 | 0.4859 | 0.06078 | -0.05953 | 1.021 | -0.00107 | 0.24989 | 0.00242 | 0.2195 | 0.118 | 0.4202 | 11059 |
| meta_regime_weights | 0.0274 | 0.366 | 0.454 | -0.1298 | 0.211 | 1.088 | 0.4918 | 0.06758 | -0.06013 | 1.124 | 0.00268 | 0.25026 | 0.01281 | 0.2379 | 0.141 | 0.4584 | 11059 |
| meta_regime_weights__no_regime | -0.0274 | -0.289 | -0.304 | -0.2298 | -0.119 | 0.95 | 0.4715 | 0.06547 | -0.06149 | 1.065 | -0.00163 | 0.25064 | 0.01619 | 0.2224 | 0.1271 | 0.457 | 11059 |
| meta_regime_weights__no_failure_memory | 0.0163 | 0.216 | 0.222 | -0.1251 | 0.13 | 1.072 | 0.4958 | 0.06441 | -0.0591 | 1.09 | 0.00214 | 0.24981 | 0.00966 | 0.2338 | 0.1315 | 0.4288 | 11059 |
| meta_regime_weights__no_disagreement | 0.0253 | 0.369 | 0.439 | -0.132 | 0.192 | 1.082 | 0.4925 | 0.06703 | -0.06011 | 1.115 | 0.00251 | 0.25006 | 0.00897 | 0.2374 | 0.1401 | 0.4418 | 11059 |
| meta_stacking | -0.0226 | -0.212 | -0.251 | -0.2353 | -0.096 | 0.963 | 0.4793 | 0.06753 | -0.06454 | 1.046 | -0.00124 | 0.2499 | 0.00243 | 0.23 | 0.1327 | 0.4391 | 11059 |
| meta_stacking__no_regime | -0.0073 | -0.034 | -0.037 | -0.1933 | -0.038 | 1.001 | 0.4831 | 0.06925 | -0.06464 | 1.071 | 4e-05 | 0.2499 | 0.00262 | 0.2374 | 0.1392 | 0.4252 | 11059 |
| meta_stacking__no_disagreement | -0.0122 | -0.079 | -0.094 | -0.2463 | -0.049 | 0.991 | 0.4838 | 0.06815 | -0.06445 | 1.057 | -0.0003 | 0.25026 | 0.00515 | 0.2353 | 0.1348 | 0.3962 | 11059 |
| meta_stacking__no_failure_memory | -0.0053 | -0.014 | -0.017 | -0.1546 | -0.034 | 1.019 | 0.4785 | 0.06759 | -0.06086 | 1.111 | 0.0006 | 0.24998 | 0.00077 | 0.2243 | 0.134 | 0.4979 | 11059 |
| meta_stacking__no_sector | -0.0492 | -0.556 | -0.581 | -0.3095 | -0.159 | 0.904 | 0.46 | 0.07026 | -0.06621 | 1.061 | -0.00344 | 0.24993 | 0.00126 | 0.2281 | 0.1325 | 0.5049 | 11059 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.189, Δ Expectancy -0.00146, Δ MaxDD -0.0484, Δ PF -0.049, Δ Hit -0.0095, Δ Brier 0.00027, Δ ECE 0.00559, Δ Prec@K -0.0085 · Monats-Δ -0.001344 CI [-0.004977, 0.002291]
- **trailing_ic_weighted**: Δ Sharpe -0.3, Δ Expectancy -0.00201, Δ MaxDD 0.002, Δ PF -0.067, Δ Hit -0.0002, Δ Brier -0.00039, Δ ECE -0.0121, Δ Prec@K -0.003 · Monats-Δ -0.001875 CI [-0.005569, 0.001731]
- **meta_regime_weights**: Δ Sharpe 0.239, Δ Expectancy 0.00174, Δ MaxDD 0.0414, Δ PF 0.056, Δ Hit 0.0057, Δ Brier -2e-05, Δ ECE -0.00171, Δ Prec@K 0.0154 · Monats-Δ 0.001626 CI [-0.002054, 0.00529]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.416, Δ Expectancy -0.00257, Δ MaxDD -0.0586, Δ PF -0.082, Δ Hit -0.0146, Δ Brier 0.00036, Δ ECE 0.00167, Δ Prec@K -0.0001 · Monats-Δ -0.002929 CI [-0.006475, 0.000524]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.089, Δ Expectancy 0.0012, Δ MaxDD 0.0461, Δ PF 0.04, Δ Hit 0.0097, Δ Brier -0.00047, Δ ECE -0.00486, Δ Prec@K 0.0113 · Monats-Δ 0.000805 CI [-0.002575, 0.004467]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.242, Δ Expectancy 0.00157, Δ MaxDD 0.0392, Δ PF 0.05, Δ Hit 0.0064, Δ Brier -0.00022, Δ ECE -0.00555, Δ Prec@K 0.0149 · Monats-Δ 0.001409 CI [-0.00268, 0.005703]
- **meta_stacking**: Δ Sharpe -0.339, Δ Expectancy -0.00218, Δ MaxDD -0.0641, Δ PF -0.069, Δ Hit -0.0068, Δ Brier -0.00038, Δ ECE -0.01209, Δ Prec@K 0.0075 · Monats-Δ -0.002485 CI [-0.00841, 0.003487]
- **meta_stacking__no_regime**: Δ Sharpe -0.161, Δ Expectancy -0.0009, Δ MaxDD -0.0221, Δ PF -0.031, Δ Hit -0.003, Δ Brier -0.00038, Δ ECE -0.0119, Δ Prec@K 0.0149 · Monats-Δ -0.001172 CI [-0.007251, 0.005088]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.206, Δ Expectancy -0.00124, Δ MaxDD -0.0751, Δ PF -0.041, Δ Hit -0.0023, Δ Brier -2e-05, Δ ECE -0.00937, Δ Prec@K 0.0128 · Monats-Δ -0.001546 CI [-0.007996, 0.00448]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.141, Δ Expectancy -0.00034, Δ MaxDD 0.0166, Δ PF -0.013, Δ Hit -0.0076, Δ Brier -0.0003, Δ ECE -0.01375, Δ Prec@K 0.0018 · Monats-Δ -0.001015 CI [-0.006635, 0.004606]
- **meta_stacking__no_sector**: Δ Sharpe -0.683, Δ Expectancy -0.00438, Δ MaxDD -0.1383, Δ PF -0.128, Δ Hit -0.0261, Δ Brier -0.00035, Δ ECE -0.01326, Δ Prec@K 0.0056 · Monats-Δ -0.004814 CI [-0.010055, 0.000653]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.239
- ❌ bootstrap_ci_positive: [-0.002054, 0.00529]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00194, 0.00506]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00169, 'vix_ge_20': 0.0018, 'spy_uptrend': 0.00199, 'spy_downtrend': 0.00092}
- ✅ calibration_not_worse: {'brier': [0.25026, 0.25028], 'ece': [0.01281, 0.01452]}
- ✅ not_outlier_driven: 1.2e-05
- ✅ locked_delta_positive: 0.01348

## Locked-Holdout

- static_equal: Expectancy -0.00245, Sharpe -0.216, Trades 3048
- best_single_ex_ante: Expectancy -0.00519, Sharpe -0.914, Trades 3048
- trailing_ic_weighted: Expectancy -0.00227, Sharpe -0.284, Trades 3048
- meta_regime_weights: Expectancy 0.01102, Sharpe 1.364, Trades 3048
- meta_regime_weights__no_regime: Expectancy 0.00268, Sharpe 0.455, Trades 3048
- meta_regime_weights__no_failure_memory: Expectancy 0.00285, Sharpe 0.216, Trades 3048
- meta_regime_weights__no_disagreement: Expectancy 0.00896, Sharpe 0.928, Trades 3048
- meta_stacking: Expectancy 0.01695, Sharpe 1.867, Trades 3048
- meta_stacking__no_regime: Expectancy 0.01785, Sharpe 1.393, Trades 3048
- meta_stacking__no_disagreement: Expectancy 0.02344, Sharpe 1.905, Trades 3048
- meta_stacking__no_failure_memory: Expectancy 0.01315, Sharpe 1.595, Trades 3048
- meta_stacking__no_sector: Expectancy 0.01365, Sharpe 1.834, Trades 3048

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00163, 'sharpe': -0.289, 'brier': 0.25064, 'delta_vs_full_expectancy': -0.00431}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00214, 'sharpe': 0.216, 'brier': 0.24981, 'delta_vs_full_expectancy': -0.00054}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00251, 'sharpe': 0.369, 'brier': 0.25006, 'delta_vs_full_expectancy': -0.00017}
- meta_stacking__no_regime: {'expectancy': 4e-05, 'sharpe': -0.034, 'brier': 0.2499, 'delta_vs_full_expectancy': 0.00128}
- meta_stacking__no_disagreement: {'expectancy': -0.0003, 'sharpe': -0.079, 'brier': 0.25026, 'delta_vs_full_expectancy': 0.00094}
- meta_stacking__no_failure_memory: {'expectancy': 0.0006, 'sharpe': -0.014, 'brier': 0.24998, 'delta_vs_full_expectancy': 0.00184}
- meta_stacking__no_sector: {'expectancy': -0.00344, 'sharpe': -0.556, 'brier': 0.24993, 'delta_vs_full_expectancy': -0.0022}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00094, 'sharpe': 0.127}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00179, 'liquid_half': 0.00183, 'less_liquid_half': 0.00248}
- meta_stacking: {'rebalance_4w': -0.00119, 'liquid_half': -0.00273, 'less_liquid_half': -0.00143}
- static_equal: {'rebalance_4w': -0.00174, 'liquid_half': 0.00123, 'less_liquid_half': -0.00034}
- best_single_ex_ante: {'rebalance_4w': -0.00209, 'liquid_half': -0.00138, 'less_liquid_half': -0.00254}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': -4e-05, 't_months': 0.02, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': -0.00205, 't_months': -0.8, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00092, 't_months': 0.47, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': 0.00215, 't_months': 0.71, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2592
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: Financial Services (IC 0.0637, t 2.49); versagt: –
- **hgb_asym20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: Energy (IC -0.0897, t -3.02)
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 28350 | 0.4644 | 0.52 | -0.00277 | -0.0055 | -0.06119 | -0.00277 | -0.0556 | overconfident |
| 55–60 % | 538 | 0.4275 | 0.5785 | -0.00532 | -0.01008 | -0.065 | -0.00532 | -0.151 | overconfident |
| 60–65 % | 104 | 0.4904 | 0.6038 | -0.00256 | -0.00146 | -0.04815 | -0.00256 | -0.1134 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15301 | 0.4809 | 0.5305 | 5e-05 | -0.00278 | -0.0584 | 5e-05 | -0.0496 | calibrated |
| 55–60 % | 5844 | 0.4771 | 0.5645 | 0.00537 | -0.00454 | -0.066 | 0.00537 | -0.0874 | overconfident |
| 60–65 % | 83 | 0.5422 | 0.6278 | 0.04004 | 0.01688 | -0.08879 | 0.04004 | -0.0857 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15637 | 0.4734 | 0.5078 | -0.00081 | -0.00464 | -0.07676 | -0.00081 | -0.0344 | calibrated |
| 55–60 % | 1965 | 0.5303 | 0.5834 | 0.02407 | 0.00598 | -0.07885 | 0.02407 | -0.0532 | overconfident |
| 60–65 % | 103 | 0.5049 | 0.6049 | -0.00259 | 0.00268 | -0.11158 | -0.00259 | -0.1001 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0047, 'recent_ic': -0.1045, 'prior_ic': 0.0652, 'trend': 'deteriorating', 'trend_t': -2.44, 'calibration_slope_recent': -0.0365, 'meta_weight': 0.3858, 'contribution': -4e-05}
- enet_xs20_v1: {'oos_ic': 0.0219, 'recent_ic': -0.0958, 'prior_ic': 0.0976, 'trend': 'deteriorating', 'trend_t': -2.76, 'calibration_slope_recent': -0.02426, 'meta_weight': 0.1776, 'contribution': -0.0003}
- hgb_xs20_v1: {'oos_ic': 0.0232, 'recent_ic': -0.0225, 'prior_ic': -0.0277, 'trend': 'stable', 'trend_t': 0.07, 'calibration_slope_recent': -0.01294, 'meta_weight': 0.2827, 'contribution': 0.00088}
- hgb_asym20_v1: {'oos_ic': 0.0049, 'recent_ic': 0.0257, 'prior_ic': -0.0233, 'trend': 'stable', 'trend_t': 1.13, 'calibration_slope_recent': 0.00544, 'meta_weight': 0.0, 'contribution': -0.00073}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0111, 'recent_ic': 0.0706, 'prior_ic': -0.0197, 'trend': 'improving', 'trend_t': 3.46, 'calibration_slope_recent': 0.02181, 'meta_weight': 0.0, 'contribution': 0.00095}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0215, 'recent_ic': -0.0466, 'prior_ic': 0.001, 'trend': 'stable', 'trend_t': -0.98, 'calibration_slope_recent': -0.02162, 'meta_weight': 0.154, 'contribution': 0.00036}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': {'prob': 0.55, 'agreement_sd': 9.0, 'lcb': -0.014873282338276294, 'n': 310, 'mean': -0.008694987204633934}, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
