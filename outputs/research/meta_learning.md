# Meta-Learning-Validierung – 2026-09-29T15:17:56+00:00

Version meta-v1 · Panel-Hash 776a327d3d005250 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): REJECT** – nicht erfüllt: delta_sharpe_positive, bootstrap_ci_positive, years_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0297 | 0.284 | 0.305 | -0.1603 | 0.186 | 1.085 | 0.4765 | 0.08898 | -0.07466 | 1.192 | 0.00331 | 0.25179 | 0.0463 | 0.2669 | 0.1688 | 0.3528 | 11587 |
| best_single_ex_ante | -0.0097 | -0.013 | -0.016 | -0.2822 | -0.034 | 0.999 | 0.4637 | 0.0869 | -0.07522 | 1.155 | -4e-05 | 0.25221 | 0.04735 | 0.2545 | 0.1582 | 0.3393 | 11587 |
| trailing_ic_weighted | 0.0002 | 0.064 | 0.063 | -0.2444 | 0.001 | 1.008 | 0.4698 | 0.08649 | -0.07608 | 1.137 | 0.0003 | 0.25176 | 0.04664 | 0.2585 | 0.1611 | 0.3062 | 11587 |
| meta_regime_weights | 0.024 | 0.257 | 0.273 | -0.1877 | 0.128 | 1.078 | 0.4729 | 0.08717 | -0.07258 | 1.201 | 0.00297 | 0.25186 | 0.04635 | 0.2582 | 0.1641 | 0.3921 | 11587 |
| meta_regime_weights__no_regime | 0.0418 | 0.428 | 0.66 | -0.176 | 0.237 | 1.12 | 0.4779 | 0.08546 | -0.06989 | 1.223 | 0.00436 | 0.25205 | 0.04718 | 0.2554 | 0.1597 | 0.3973 | 11587 |
| meta_regime_weights__no_failure_memory | 0.0169 | 0.188 | 0.194 | -0.2288 | 0.074 | 1.065 | 0.472 | 0.08945 | -0.07505 | 1.192 | 0.00259 | 0.25181 | 0.04668 | 0.2649 | 0.1711 | 0.3241 | 11587 |
| meta_regime_weights__no_disagreement | 0.0181 | 0.211 | 0.227 | -0.198 | 0.092 | 1.064 | 0.4707 | 0.08674 | -0.07249 | 1.197 | 0.00246 | 0.25188 | 0.04653 | 0.259 | 0.1632 | 0.4003 | 11587 |
| meta_stacking | -0.0189 | -0.052 | -0.056 | -0.2909 | -0.065 | 0.996 | 0.4657 | 0.09504 | -0.08318 | 1.143 | -0.00018 | 0.2517 | 0.04627 | 0.2674 | 0.1721 | 0.2996 | 11587 |
| meta_stacking__no_regime | 0.0082 | 0.126 | 0.138 | -0.1928 | 0.042 | 1.048 | 0.4746 | 0.09182 | -0.07914 | 1.16 | 0.002 | 0.25169 | 0.04592 | 0.2694 | 0.1727 | 0.339 | 11587 |
| meta_stacking__no_disagreement | -0.0105 | -0.0 | -0.001 | -0.2537 | -0.042 | 1.008 | 0.4684 | 0.09392 | -0.0821 | 1.144 | 0.00034 | 0.25191 | 0.04642 | 0.267 | 0.1692 | 0.2901 | 11587 |
| meta_stacking__no_failure_memory | -0.0405 | -0.207 | -0.214 | -0.3266 | -0.124 | 0.953 | 0.4584 | 0.09428 | -0.08367 | 1.127 | -0.00211 | 0.25185 | 0.04675 | 0.265 | 0.173 | 0.3157 | 11587 |
| meta_stacking__no_sector | 0.0068 | 0.116 | 0.133 | -0.232 | 0.029 | 1.042 | 0.4739 | 0.09625 | -0.08319 | 1.157 | 0.00184 | 0.25196 | 0.04674 | 0.2739 | 0.1786 | 0.3608 | 11587 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.297, Δ Expectancy -0.00335, Δ MaxDD -0.1219, Δ PF -0.086, Δ Hit -0.0128, Δ Brier 0.00042, Δ ECE 0.00105, Δ Prec@K -0.0124 · Monats-Δ -0.003323 CI [-0.007438, 0.000767]
- **trailing_ic_weighted**: Δ Sharpe -0.22, Δ Expectancy -0.00301, Δ MaxDD -0.0841, Δ PF -0.077, Δ Hit -0.0067, Δ Brier -3e-05, Δ ECE 0.00034, Δ Prec@K -0.0084 · Monats-Δ -0.002504 CI [-0.008298, 0.003453]
- **meta_regime_weights**: Δ Sharpe -0.027, Δ Expectancy -0.00034, Δ MaxDD -0.0274, Δ PF -0.007, Δ Hit -0.0036, Δ Brier 7e-05, Δ ECE 5e-05, Δ Prec@K -0.0087 · Monats-Δ -0.000628 CI [-0.005028, 0.004005]
- **meta_regime_weights__no_regime**: Δ Sharpe 0.144, Δ Expectancy 0.00105, Δ MaxDD -0.0157, Δ PF 0.035, Δ Hit 0.0014, Δ Brier 0.00026, Δ ECE 0.00088, Δ Prec@K -0.0115 · Monats-Δ 0.000702 CI [-0.005144, 0.007217]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe -0.096, Δ Expectancy -0.00072, Δ MaxDD -0.0685, Δ PF -0.02, Δ Hit -0.0045, Δ Brier 2e-05, Δ ECE 0.00038, Δ Prec@K -0.002 · Monats-Δ -0.000971 CI [-0.004812, 0.003367]
- **meta_regime_weights__no_disagreement**: Δ Sharpe -0.073, Δ Expectancy -0.00085, Δ MaxDD -0.0377, Δ PF -0.021, Δ Hit -0.0058, Δ Brier 9e-05, Δ ECE 0.00023, Δ Prec@K -0.0079 · Monats-Δ -0.001134 CI [-0.006007, 0.003947]
- **meta_stacking**: Δ Sharpe -0.336, Δ Expectancy -0.00349, Δ MaxDD -0.1306, Δ PF -0.089, Δ Hit -0.0108, Δ Brier -9e-05, Δ ECE -3e-05, Δ Prec@K 0.0005 · Monats-Δ -0.00383 CI [-0.010244, 0.002324]
- **meta_stacking__no_regime**: Δ Sharpe -0.158, Δ Expectancy -0.00131, Δ MaxDD -0.0325, Δ PF -0.037, Δ Hit -0.0019, Δ Brier -0.0001, Δ ECE -0.00038, Δ Prec@K 0.0025 · Monats-Δ -0.001807 CI [-0.007739, 0.004348]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.284, Δ Expectancy -0.00297, Δ MaxDD -0.0934, Δ PF -0.077, Δ Hit -0.0081, Δ Brier 0.00012, Δ ECE 0.00012, Δ Prec@K 0.0001 · Monats-Δ -0.003186 CI [-0.009348, 0.002718]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.491, Δ Expectancy -0.00542, Δ MaxDD -0.1663, Δ PF -0.132, Δ Hit -0.0181, Δ Brier 6e-05, Δ ECE 0.00045, Δ Prec@K -0.0019 · Monats-Δ -0.005729 CI [-0.011709, 0.000122]
- **meta_stacking__no_sector**: Δ Sharpe -0.168, Δ Expectancy -0.00147, Δ MaxDD -0.0717, Δ PF -0.043, Δ Hit -0.0026, Δ Brier 0.00017, Δ ECE 0.00044, Δ Prec@K 0.007 · Monats-Δ -0.001838 CI [-0.006689, 0.003491]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ❌ delta_sharpe_positive: -0.027
- ❌ bootstrap_ci_positive: [-0.005028, 0.004005]
- ❌ years_positive: 0.4
- ❌ halves_positive: [0.00104, -0.00224]
- ✅ no_regime_collapse: {'vix_lt_20': -0.00242, 'vix_ge_20': 0.00292, 'spy_uptrend': -0.00198, 'spy_downtrend': 0.0049}
- ✅ calibration_not_worse: {'brier': [0.25186, 0.25179], 'ece': [0.04635, 0.0463]}
- ❌ not_outlier_driven: -0.003026
- ✅ locked_delta_positive: 0.00461

## Locked-Holdout

- static_equal: Expectancy 0.01908, Sharpe 1.589, Trades 3111
- best_single_ex_ante: Expectancy 0.01332, Sharpe 1.448, Trades 3111
- trailing_ic_weighted: Expectancy 0.01318, Sharpe 1.023, Trades 3111
- meta_regime_weights: Expectancy 0.02369, Sharpe 1.901, Trades 3111
- meta_regime_weights__no_regime: Expectancy 0.01384, Sharpe 1.359, Trades 3111
- meta_regime_weights__no_failure_memory: Expectancy 0.01968, Sharpe 1.525, Trades 3111
- meta_regime_weights__no_disagreement: Expectancy 0.01223, Sharpe 0.971, Trades 3111
- meta_stacking: Expectancy 0.02026, Sharpe 2.669, Trades 3111
- meta_stacking__no_regime: Expectancy 0.03182, Sharpe 2.764, Trades 3111
- meta_stacking__no_disagreement: Expectancy 0.02414, Sharpe 3.394, Trades 3111
- meta_stacking__no_failure_memory: Expectancy 0.02814, Sharpe 2.65, Trades 3111
- meta_stacking__no_sector: Expectancy 0.02283, Sharpe 2.142, Trades 3111

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': 0.00436, 'sharpe': 0.428, 'brier': 0.25205, 'delta_vs_full_expectancy': 0.00139}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00259, 'sharpe': 0.188, 'brier': 0.25181, 'delta_vs_full_expectancy': -0.00038}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00246, 'sharpe': 0.211, 'brier': 0.25188, 'delta_vs_full_expectancy': -0.00051}
- meta_stacking__no_regime: {'expectancy': 0.002, 'sharpe': 0.126, 'brier': 0.25169, 'delta_vs_full_expectancy': 0.00218}
- meta_stacking__no_disagreement: {'expectancy': 0.00034, 'sharpe': -0.0, 'brier': 0.25191, 'delta_vs_full_expectancy': 0.00052}
- meta_stacking__no_failure_memory: {'expectancy': -0.00211, 'sharpe': -0.207, 'brier': 0.25185, 'delta_vs_full_expectancy': -0.00193}
- meta_stacking__no_sector: {'expectancy': 0.00184, 'sharpe': 0.116, 'brier': 0.25196, 'delta_vs_full_expectancy': 0.00202}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00331, 'sharpe': 0.284}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00093, 'liquid_half': 0.00142, 'less_liquid_half': 0.00433}
- meta_stacking: {'rebalance_4w': -0.00135, 'liquid_half': -0.00114, 'less_liquid_half': 0.00375}
- static_equal: {'rebalance_4w': 0.00231, 'liquid_half': 0.00277, 'less_liquid_half': 0.00386}
- best_single_ex_ante: {'rebalance_4w': -0.00172, 'liquid_half': -0.00066, 'less_liquid_half': 0.00132}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00037, 't_months': 0.07, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': -0.00181, 't_months': -0.51, 'share_years_positive': 0.2, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': -0.00151, 't_months': -0.42, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.01118, 't_months': -2.77, 'share_years_positive': 0.2, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2512
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: –; versagt: –
- **hgb_asym20_v1** – funktioniert: Consumer Cyclical (IC 0.0424, t 2.16); versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: –
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 30376 | 0.4745 | 0.5269 | 0.0017 | -0.00458 | -0.06416 | -0.0003 | -0.0524 | overconfident |
| 55–60 % | 7868 | 0.4235 | 0.5517 | -0.00872 | -0.01073 | -0.05338 | -0.01072 | -0.1282 | overconfident |
| 60–65 % | 468 | 0.4444 | 0.6255 | -0.00519 | -0.00769 | -0.05184 | -0.00719 | -0.1811 | overconfident |
| 65–70 % | 104 | 0.4327 | 0.6569 | -0.00871 | -0.00955 | -0.05445 | -0.01071 | -0.2242 | low_n |
| 70–75 % | 61 | 0.6066 | 0.7273 | 0.12354 | 0.06613 | -0.1008 | 0.12154 | -0.1207 | low_n |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 30958 | 0.4699 | 0.5303 | 0.00047 | -0.00512 | -0.0597 | -0.00153 | -0.0604 | overconfident |
| 55–60 % | 5471 | 0.4341 | 0.5688 | -0.00479 | -0.00874 | -0.05458 | -0.00679 | -0.1347 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 61 | 0.5902 | 0.7273 | 0.09135 | 0.04753 | -0.10304 | 0.08935 | -0.1371 | low_n |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 34916 | 0.4672 | 0.5334 | 0.00109 | -0.00555 | -0.06113 | -0.00091 | -0.0662 | overconfident |
| 55–60 % | 61 | 0.6066 | 0.563 | 0.04188 | 0.02652 | -0.08446 | 0.03988 | 0.0435 | low_n |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 60 | 0.6167 | 0.6819 | 0.05619 | 0.04511 | -0.10286 | 0.05419 | -0.0652 | low_n |
| 70–75 % | 62 | 0.5645 | 0.7265 | 0.04952 | 0.00752 | -0.09643 | 0.04752 | -0.162 | low_n |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.008, 'recent_ic': -0.1172, 'prior_ic': 0.0711, 'trend': 'deteriorating', 'trend_t': -2.65, 'calibration_slope_recent': -0.04374, 'meta_weight': 0.8468, 'contribution': 0.00028}
- enet_xs20_v1: {'oos_ic': 0.0322, 'recent_ic': -0.0052, 'prior_ic': 0.1107, 'trend': 'stable', 'trend_t': -1.82, 'calibration_slope_recent': 0.00555, 'meta_weight': 0.1532, 'contribution': 0.00079}
- hgb_xs20_v1: {'oos_ic': 0.0166, 'recent_ic': -0.0227, 'prior_ic': 0.0419, 'trend': 'stable', 'trend_t': -0.92, 'calibration_slope_recent': -0.01109, 'meta_weight': 0.0, 'contribution': -0.00099}
- hgb_asym20_v1: {'oos_ic': 0.0166, 'recent_ic': -0.0495, 'prior_ic': 0.0282, 'trend': 'stable', 'trend_t': -1.49, 'calibration_slope_recent': -0.00985, 'meta_weight': 0.0, 'contribution': 6e-05}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0176, 'recent_ic': 0.0674, 'prior_ic': 0.0307, 'trend': 'stable', 'trend_t': 1.22, 'calibration_slope_recent': 0.02771, 'meta_weight': 0.0, 'contribution': 0.00137}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0172, 'recent_ic': 0.0027, 'prior_ic': -0.01, 'trend': 'stable', 'trend_t': 0.25, 'calibration_slope_recent': -0.00655, 'meta_weight': 0.0, 'contribution': -0.00103}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': {'prob': 0.55, 'agreement_sd': 9.0, 'lcb': -0.008353031150829866, 'n': 2425, 'mean': -0.005868681352518407}, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
