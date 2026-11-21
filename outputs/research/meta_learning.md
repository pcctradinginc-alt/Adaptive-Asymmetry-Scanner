# Meta-Learning-Validierung – 2026-11-21T16:26:27+00:00

Version meta-v1 · Panel-Hash f2326edcf7425eb7 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0104 | 0.162 | 0.199 | -0.1667 | 0.062 | 1.04 | 0.4861 | 0.06267 | -0.05702 | 1.099 | 0.00116 | 0.25034 | 0.01627 | 0.2277 | 0.1246 | 0.4641 | 10986 |
| best_single_ex_ante | 0.0045 | 0.096 | 0.11 | -0.177 | 0.025 | 1.015 | 0.4866 | 0.05982 | -0.05586 | 1.071 | 0.00043 | 0.25073 | 0.01643 | 0.2108 | 0.1103 | 0.4191 | 10986 |
| trailing_ic_weighted | -0.019 | -0.254 | -0.232 | -0.1653 | -0.115 | 0.951 | 0.4821 | 0.0613 | -0.05997 | 1.022 | -0.00151 | 0.24998 | 0.00229 | 0.2244 | 0.1215 | 0.4179 | 10986 |
| meta_regime_weights | 0.0183 | 0.246 | 0.321 | -0.139 | 0.132 | 1.066 | 0.484 | 0.06779 | -0.05966 | 1.136 | 0.00202 | 0.25049 | 0.01525 | 0.2317 | 0.1388 | 0.4497 | 10986 |
| meta_regime_weights__no_regime | -0.0267 | -0.248 | -0.287 | -0.2435 | -0.109 | 0.957 | 0.4717 | 0.06498 | -0.06063 | 1.072 | -0.00138 | 0.25078 | 0.01976 | 0.2175 | 0.1228 | 0.4474 | 10986 |
| meta_regime_weights__no_failure_memory | 0.0093 | 0.142 | 0.14 | -0.1407 | 0.066 | 1.052 | 0.4886 | 0.06581 | -0.05976 | 1.101 | 0.00159 | 0.2498 | 0.00747 | 0.235 | 0.1334 | 0.4156 | 10986 |
| meta_regime_weights__no_disagreement | 0.0074 | 0.13 | 0.162 | -0.134 | 0.055 | 1.033 | 0.4802 | 0.06722 | -0.06012 | 1.118 | 0.00102 | 0.25028 | 0.01594 | 0.2307 | 0.1362 | 0.4498 | 10986 |
| meta_stacking | -0.0385 | -0.421 | -0.456 | -0.2685 | -0.143 | 0.929 | 0.4676 | 0.07207 | -0.06813 | 1.058 | -0.00257 | 0.24988 | 0.00207 | 0.2393 | 0.1387 | 0.4487 | 10986 |
| meta_stacking__no_regime | -0.0309 | -0.376 | -0.433 | -0.2404 | -0.128 | 0.94 | 0.4724 | 0.07076 | -0.06742 | 1.049 | -0.00214 | 0.25011 | 0.01246 | 0.2376 | 0.1384 | 0.4472 | 10986 |
| meta_stacking__no_disagreement | -0.0387 | -0.436 | -0.484 | -0.3038 | -0.127 | 0.92 | 0.4628 | 0.06883 | -0.06446 | 1.068 | -0.00278 | 0.25013 | 0.00477 | 0.2275 | 0.1313 | 0.4457 | 10986 |
| meta_stacking__no_failure_memory | 0.0041 | 0.089 | 0.115 | -0.1533 | 0.027 | 1.032 | 0.4802 | 0.06875 | -0.06156 | 1.117 | 0.00102 | 0.25015 | 0.01643 | 0.2335 | 0.1377 | 0.5538 | 10986 |
| meta_stacking__no_sector | -0.0385 | -0.421 | -0.456 | -0.2685 | -0.143 | 0.929 | 0.4676 | 0.07207 | -0.06813 | 1.058 | -0.00257 | 0.24988 | 0.00207 | 0.2393 | 0.1387 | 0.4487 | 10986 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.066, Δ Expectancy -0.00073, Δ MaxDD -0.0103, Δ PF -0.025, Δ Hit 0.0005, Δ Brier 0.00039, Δ ECE 0.00016, Δ Prec@K -0.0169 · Monats-Δ -0.00054 CI [-0.005483, 0.004413]
- **trailing_ic_weighted**: Δ Sharpe -0.416, Δ Expectancy -0.00267, Δ MaxDD 0.0014, Δ PF -0.089, Δ Hit -0.004, Δ Brier -0.00036, Δ ECE -0.01398, Δ Prec@K -0.0033 · Monats-Δ -0.002569 CI [-0.006688, 0.001206]
- **meta_regime_weights**: Δ Sharpe 0.084, Δ Expectancy 0.00086, Δ MaxDD 0.0277, Δ PF 0.026, Δ Hit -0.0021, Δ Brier 0.00015, Δ ECE -0.00102, Δ Prec@K 0.004 · Monats-Δ 0.000673 CI [-0.002557, 0.003953]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.41, Δ Expectancy -0.00254, Δ MaxDD -0.0768, Δ PF -0.083, Δ Hit -0.0144, Δ Brier 0.00044, Δ ECE 0.00349, Δ Prec@K -0.0102 · Monats-Δ -0.003063 CI [-0.006744, 0.000526]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe -0.02, Δ Expectancy 0.00043, Δ MaxDD 0.026, Δ PF 0.012, Δ Hit 0.0025, Δ Brier -0.00054, Δ ECE -0.0088, Δ Prec@K 0.0073 · Monats-Δ 2.4e-05 CI [-0.003739, 0.003915]
- **meta_regime_weights__no_disagreement**: Δ Sharpe -0.032, Δ Expectancy -0.00014, Δ MaxDD 0.0327, Δ PF -0.007, Δ Hit -0.0059, Δ Brier -6e-05, Δ ECE -0.00033, Δ Prec@K 0.003 · Monats-Δ -0.000277 CI [-0.003398, 0.003156]
- **meta_stacking**: Δ Sharpe -0.583, Δ Expectancy -0.00373, Δ MaxDD -0.1018, Δ PF -0.111, Δ Hit -0.0185, Δ Brier -0.00046, Δ ECE -0.0142, Δ Prec@K 0.0116 · Monats-Δ -0.004132 CI [-0.009399, 0.001441]
- **meta_stacking__no_regime**: Δ Sharpe -0.538, Δ Expectancy -0.0033, Δ MaxDD -0.0737, Δ PF -0.1, Δ Hit -0.0137, Δ Brier -0.00023, Δ ECE -0.00381, Δ Prec@K 0.0099 · Monats-Δ -0.003533 CI [-0.009767, 0.002614]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.598, Δ Expectancy -0.00394, Δ MaxDD -0.1371, Δ PF -0.12, Δ Hit -0.0233, Δ Brier -0.00021, Δ ECE -0.0115, Δ Prec@K -0.0002 · Monats-Δ -0.00416 CI [-0.00967, 0.00138]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.073, Δ Expectancy -0.00014, Δ MaxDD 0.0134, Δ PF -0.008, Δ Hit -0.0059, Δ Brier -0.00019, Δ ECE 0.00016, Δ Prec@K 0.0058 · Monats-Δ -0.00055 CI [-0.004768, 0.004038]
- **meta_stacking__no_sector**: Δ Sharpe -0.583, Δ Expectancy -0.00373, Δ MaxDD -0.1018, Δ PF -0.111, Δ Hit -0.0185, Δ Brier -0.00046, Δ ECE -0.0142, Δ Prec@K 0.0116 · Monats-Δ -0.004132 CI [-0.009399, 0.001441]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.084
- ❌ bootstrap_ci_positive: [-0.002557, 0.003953]
- ✅ years_positive: 0.6
- ❌ halves_positive: [-0.00178, 0.00304]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00087, 'vix_ge_20': 0.00085, 'spy_uptrend': 0.00109, 'spy_downtrend': 0.00014}
- ✅ calibration_not_worse: {'brier': [0.25049, 0.25034], 'ece': [0.01525, 0.01627]}
- ❌ not_outlier_driven: -0.000664
- ✅ locked_delta_positive: 0.00542

## Locked-Holdout

- static_equal: Expectancy -0.00084, Sharpe 0.137, Trades 3592
- best_single_ex_ante: Expectancy -0.00091, Sharpe 0.033, Trades 3592
- trailing_ic_weighted: Expectancy -0.01066, Sharpe -0.482, Trades 3592
- meta_regime_weights: Expectancy 0.00457, Sharpe 0.718, Trades 3592
- meta_regime_weights__no_regime: Expectancy -0.001, Sharpe 0.195, Trades 3592
- meta_regime_weights__no_failure_memory: Expectancy -0.00491, Sharpe -0.184, Trades 3592
- meta_regime_weights__no_disagreement: Expectancy 0.00301, Sharpe 0.474, Trades 3592
- meta_stacking: Expectancy 0.00802, Sharpe 0.835, Trades 3592
- meta_stacking__no_regime: Expectancy 0.00815, Sharpe 1.331, Trades 3592
- meta_stacking__no_disagreement: Expectancy 0.00658, Sharpe 0.762, Trades 3592
- meta_stacking__no_failure_memory: Expectancy 0.01022, Sharpe 1.028, Trades 3592
- meta_stacking__no_sector: Expectancy 0.00802, Sharpe 0.835, Trades 3592

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00138, 'sharpe': -0.248, 'brier': 0.25078, 'delta_vs_full_expectancy': -0.0034}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00159, 'sharpe': 0.142, 'brier': 0.2498, 'delta_vs_full_expectancy': -0.00043}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00102, 'sharpe': 0.13, 'brier': 0.25028, 'delta_vs_full_expectancy': -0.001}
- meta_stacking__no_regime: {'expectancy': -0.00214, 'sharpe': -0.376, 'brier': 0.25011, 'delta_vs_full_expectancy': 0.00043}
- meta_stacking__no_disagreement: {'expectancy': -0.00278, 'sharpe': -0.436, 'brier': 0.25013, 'delta_vs_full_expectancy': -0.00021}
- meta_stacking__no_failure_memory: {'expectancy': 0.00102, 'sharpe': 0.089, 'brier': 0.25015, 'delta_vs_full_expectancy': 0.00359}
- meta_stacking__no_sector: {'expectancy': -0.00257, 'sharpe': -0.421, 'brier': 0.24988, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00116, 'sharpe': 0.162}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00102, 'liquid_half': 0.00112, 'less_liquid_half': 0.00133}
- meta_stacking: {'rebalance_4w': -0.00504, 'liquid_half': -0.00182, 'less_liquid_half': -0.00456}
- static_equal: {'rebalance_4w': -0.00053, 'liquid_half': 0.00149, 'less_liquid_half': -0.00167}
- best_single_ex_ante: {'rebalance_4w': 0.00101, 'liquid_half': 0.00181, 'less_liquid_half': -0.00208}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00064, 't_months': 0.2, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00469, 't_months': 1.48, 'share_years_positive': 1.0, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00076, 't_months': 0.31, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': 0.00052, 't_months': 0.09, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.185
- current_level: LOW

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: –; versagt: –
- **hgb_asym20_v1** – funktioniert: vix_ge_20 (IC 0.0608, t 2.02); versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: –
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 26015 | 0.4748 | 0.5147 | -0.00149 | -0.00386 | -0.05997 | -0.00149 | -0.04 | calibrated |
| 55–60 % | 215 | 0.4651 | 0.5759 | -0.01148 | -0.01626 | -0.1076 | -0.01148 | -0.1108 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 104 | 0.4519 | 0.6725 | -0.00807 | -0.01187 | -0.05271 | -0.00807 | -0.2205 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 20748 | 0.4769 | 0.5151 | -0.00023 | -0.00361 | -0.06162 | -0.00023 | -0.0382 | calibrated |
| 55–60 % | 588 | 0.466 | 0.5615 | -0.00238 | -0.00837 | -0.0837 | -0.00238 | -0.0955 | overconfident |
| 60–65 % | 52 | 0.4808 | 0.6458 | 0.00897 | -0.0013 | -0.11364 | 0.00897 | -0.1651 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 143 | 0.4336 | 0.7083 | 0.00501 | -0.02178 | -0.1149 | 0.00501 | -0.2748 | low_n |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 10313 | 0.4902 | 0.5151 | 0.00318 | -0.00197 | -0.08352 | 0.00318 | -0.025 | calibrated |
| 55–60 % | 361 | 0.5291 | 0.5553 | 0.00202 | 0.00562 | -0.10119 | 0.00202 | -0.0262 | calibrated |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': -0.0004, 'recent_ic': -0.083, 'prior_ic': 0.0271, 'trend': 'stable', 'trend_t': -1.2, 'calibration_slope_recent': -0.04844, 'meta_weight': 0.1813, 'contribution': 0.00102}
- enet_xs20_v1: {'oos_ic': 0.014, 'recent_ic': -0.188, 'prior_ic': 0.0376, 'trend': 'deteriorating', 'trend_t': -2.3, 'calibration_slope_recent': -0.07378, 'meta_weight': 0.1074, 'contribution': 0.00044}
- hgb_xs20_v1: {'oos_ic': 0.0206, 'recent_ic': 0.1658, 'prior_ic': -0.0391, 'trend': 'improving', 'trend_t': 5.94, 'calibration_slope_recent': 0.05718, 'meta_weight': 0.2799, 'contribution': 0.00044}
- hgb_asym20_v1: {'oos_ic': 0.0227, 'recent_ic': 0.1719, 'prior_ic': -0.0457, 'trend': 'improving', 'trend_t': 3.81, 'calibration_slope_recent': 0.06463, 'meta_weight': 0.161, 'contribution': -5e-05}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0101, 'recent_ic': 0.0259, 'prior_ic': -0.0047, 'trend': 'stable', 'trend_t': 0.84, 'calibration_slope_recent': 0.00533, 'meta_weight': 0.0, 'contribution': 0.00108}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0274, 'recent_ic': 0.0559, 'prior_ic': 0.0115, 'trend': 'stable', 'trend_t': 0.62, 'calibration_slope_recent': 0.03221, 'meta_weight': 0.2704, 'contribution': 0.00105}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
