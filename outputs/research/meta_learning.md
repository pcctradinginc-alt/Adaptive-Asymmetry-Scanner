# Meta-Learning-Validierung – 2026-11-07T15:06:36+00:00

Version meta-v1 · Panel-Hash 9d99584f2c2fa5be · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0036 | 0.083 | 0.103 | -0.1698 | 0.021 | 1.018 | 0.4834 | 0.06233 | -0.05728 | 1.088 | 0.00054 | 0.2503 | 0.01376 | 0.2215 | 0.1212 | 0.4657 | 10986 |
| best_single_ex_ante | -0.0236 | -0.248 | -0.311 | -0.2641 | -0.089 | 0.947 | 0.4724 | 0.06134 | -0.05802 | 1.057 | -0.00164 | 0.25032 | 0.01815 | 0.2004 | 0.1125 | 0.4765 | 10986 |
| trailing_ic_weighted | -0.0193 | -0.251 | -0.224 | -0.1782 | -0.108 | 0.948 | 0.4812 | 0.06113 | -0.05981 | 1.022 | -0.00161 | 0.24998 | 0.00264 | 0.2206 | 0.1208 | 0.4016 | 10986 |
| meta_regime_weights | 0.0168 | 0.228 | 0.292 | -0.1731 | 0.097 | 1.064 | 0.485 | 0.06882 | -0.0609 | 1.13 | 0.00201 | 0.25025 | 0.01287 | 0.237 | 0.1421 | 0.4525 | 10986 |
| meta_regime_weights__no_regime | -0.0432 | -0.436 | -0.472 | -0.2924 | -0.148 | 0.916 | 0.4616 | 0.06513 | -0.06096 | 1.068 | -0.00276 | 0.25033 | 0.01507 | 0.2142 | 0.1228 | 0.458 | 10986 |
| meta_regime_weights__no_failure_memory | 0.0193 | 0.244 | 0.254 | -0.1195 | 0.162 | 1.08 | 0.4954 | 0.06525 | -0.05935 | 1.099 | 0.00238 | 0.24992 | 0.00915 | 0.2343 | 0.1346 | 0.3921 | 10986 |
| meta_regime_weights__no_disagreement | 0.0238 | 0.331 | 0.425 | -0.1721 | 0.139 | 1.08 | 0.4882 | 0.06823 | -0.06026 | 1.132 | 0.00246 | 0.24982 | 0.00801 | 0.2369 | 0.1398 | 0.4505 | 10986 |
| meta_stacking | -0.0227 | -0.242 | -0.252 | -0.3015 | -0.075 | 0.961 | 0.4791 | 0.06618 | -0.06334 | 1.045 | -0.00129 | 0.24999 | 0.00715 | 0.2254 | 0.1284 | 0.4998 | 10986 |
| meta_stacking__no_regime | -0.0053 | -0.011 | -0.014 | -0.2429 | -0.022 | 1.003 | 0.4797 | 0.0717 | -0.06592 | 1.088 | 0.0001 | 0.25004 | 0.01093 | 0.2408 | 0.1395 | 0.485 | 10986 |
| meta_stacking__no_disagreement | -0.0511 | -0.515 | -0.644 | -0.3568 | -0.143 | 0.898 | 0.4578 | 0.06914 | -0.06503 | 1.063 | -0.00361 | 0.25016 | 0.00177 | 0.2254 | 0.1317 | 0.4341 | 10986 |
| meta_stacking__no_failure_memory | -0.0131 | -0.13 | -0.163 | -0.186 | -0.07 | 0.991 | 0.4759 | 0.06837 | -0.06262 | 1.092 | -0.00028 | 0.24993 | 0.00211 | 0.2305 | 0.1337 | 0.5819 | 10986 |
| meta_stacking__no_sector | -0.0227 | -0.242 | -0.252 | -0.3015 | -0.075 | 0.961 | 0.4791 | 0.06618 | -0.06334 | 1.045 | -0.00129 | 0.24999 | 0.00715 | 0.2254 | 0.1284 | 0.4998 | 10986 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.331, Δ Expectancy -0.00218, Δ MaxDD -0.0943, Δ PF -0.071, Δ Hit -0.011, Δ Brier 2e-05, Δ ECE 0.00439, Δ Prec@K -0.0211 · Monats-Δ -0.00231 CI [-0.006688, 0.001877]
- **trailing_ic_weighted**: Δ Sharpe -0.334, Δ Expectancy -0.00215, Δ MaxDD -0.0084, Δ PF -0.07, Δ Hit -0.0022, Δ Brier -0.00032, Δ ECE -0.01112, Δ Prec@K -0.0009 · Monats-Δ -0.002025 CI [-0.006079, 0.001947]
- **meta_regime_weights**: Δ Sharpe 0.145, Δ Expectancy 0.00147, Δ MaxDD -0.0033, Δ PF 0.046, Δ Hit 0.0016, Δ Brier -5e-05, Δ ECE -0.00089, Δ Prec@K 0.0155 · Monats-Δ 0.001116 CI [-0.002015, 0.004051]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.519, Δ Expectancy -0.0033, Δ MaxDD -0.1226, Δ PF -0.102, Δ Hit -0.0218, Δ Brier 3e-05, Δ ECE 0.00131, Δ Prec@K -0.0073 · Monats-Δ -0.003926 CI [-0.007911, -0.000369]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.161, Δ Expectancy 0.00184, Δ MaxDD 0.0503, Δ PF 0.062, Δ Hit 0.012, Δ Brier -0.00038, Δ ECE -0.00461, Δ Prec@K 0.0128 · Monats-Δ 0.001384 CI [-0.002171, 0.005033]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.248, Δ Expectancy 0.00192, Δ MaxDD -0.0023, Δ PF 0.062, Δ Hit 0.0048, Δ Brier -0.00048, Δ ECE -0.00575, Δ Prec@K 0.0154 · Monats-Δ 0.001631 CI [-0.001888, 0.005338]
- **meta_stacking**: Δ Sharpe -0.325, Δ Expectancy -0.00183, Δ MaxDD -0.1317, Δ PF -0.057, Δ Hit -0.0043, Δ Brier -0.00031, Δ ECE -0.00661, Δ Prec@K 0.0039 · Monats-Δ -0.002237 CI [-0.007063, 0.002758]
- **meta_stacking__no_regime**: Δ Sharpe -0.094, Δ Expectancy -0.00044, Δ MaxDD -0.0731, Δ PF -0.015, Δ Hit -0.0037, Δ Brier -0.00026, Δ ECE -0.00283, Δ Prec@K 0.0193 · Monats-Δ -0.000681 CI [-0.007181, 0.006019]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.598, Δ Expectancy -0.00415, Δ MaxDD -0.187, Δ PF -0.12, Δ Hit -0.0256, Δ Brier -0.00014, Δ ECE -0.01199, Δ Prec@K 0.0039 · Monats-Δ -0.004608 CI [-0.009995, 0.001203]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.213, Δ Expectancy -0.00082, Δ MaxDD -0.0162, Δ PF -0.027, Δ Hit -0.0075, Δ Brier -0.00037, Δ ECE -0.01165, Δ Prec@K 0.009 · Monats-Δ -0.001445 CI [-0.005661, 0.003068]
- **meta_stacking__no_sector**: Δ Sharpe -0.325, Δ Expectancy -0.00183, Δ MaxDD -0.1317, Δ PF -0.057, Δ Hit -0.0043, Δ Brier -0.00031, Δ ECE -0.00661, Δ Prec@K 0.0039 · Monats-Δ -0.002237 CI [-0.007063, 0.002758]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.145
- ❌ bootstrap_ci_positive: [-0.002015, 0.004051]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00283, 0.00491]
- ✅ no_regime_collapse: {'vix_lt_20': 0.0018, 'vix_ge_20': 0.00094, 'spy_uptrend': 0.00141, 'spy_downtrend': 0.00166}
- ✅ calibration_not_worse: {'brier': [0.25025, 0.2503], 'ece': [0.01287, 0.01376]}
- ✅ not_outlier_driven: 6.5e-05
- ✅ locked_delta_positive: 0.00281

## Locked-Holdout

- static_equal: Expectancy 0.00352, Sharpe 0.411, Trades 3442
- best_single_ex_ante: Expectancy 0.0008, Sharpe 0.288, Trades 3442
- trailing_ic_weighted: Expectancy -0.0061, Sharpe -0.563, Trades 3442
- meta_regime_weights: Expectancy 0.00633, Sharpe 1.084, Trades 3442
- meta_regime_weights__no_regime: Expectancy 0.00281, Sharpe 0.375, Trades 3442
- meta_regime_weights__no_failure_memory: Expectancy 0.00166, Sharpe 0.162, Trades 3442
- meta_regime_weights__no_disagreement: Expectancy 0.00314, Sharpe 0.512, Trades 3442
- meta_stacking: Expectancy -0.0036, Sharpe -0.539, Trades 3442
- meta_stacking__no_regime: Expectancy -0.00199, Sharpe -0.216, Trades 3442
- meta_stacking__no_disagreement: Expectancy -0.00048, Sharpe -0.15, Trades 3442
- meta_stacking__no_failure_memory: Expectancy 0.00885, Sharpe 1.078, Trades 3442
- meta_stacking__no_sector: Expectancy -0.0036, Sharpe -0.539, Trades 3442

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00276, 'sharpe': -0.436, 'brier': 0.25033, 'delta_vs_full_expectancy': -0.00477}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00238, 'sharpe': 0.244, 'brier': 0.24992, 'delta_vs_full_expectancy': 0.00037}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00246, 'sharpe': 0.331, 'brier': 0.24982, 'delta_vs_full_expectancy': 0.00045}
- meta_stacking__no_regime: {'expectancy': 0.0001, 'sharpe': -0.011, 'brier': 0.25004, 'delta_vs_full_expectancy': 0.00139}
- meta_stacking__no_disagreement: {'expectancy': -0.00361, 'sharpe': -0.515, 'brier': 0.25016, 'delta_vs_full_expectancy': -0.00232}
- meta_stacking__no_failure_memory: {'expectancy': -0.00028, 'sharpe': -0.13, 'brier': 0.24993, 'delta_vs_full_expectancy': 0.00101}
- meta_stacking__no_sector: {'expectancy': -0.00129, 'sharpe': -0.242, 'brier': 0.24999, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00054, 'sharpe': 0.083}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00076, 'liquid_half': 0.00062, 'less_liquid_half': 0.0011}
- meta_stacking: {'rebalance_4w': -0.00456, 'liquid_half': -0.00039, 'less_liquid_half': -0.00351}
- static_equal: {'rebalance_4w': -0.00104, 'liquid_half': 0.00041, 'less_liquid_half': -0.00178}
- best_single_ex_ante: {'rebalance_4w': -0.00289, 'liquid_half': -0.00152, 'less_liquid_half': -0.00203}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00194, 't_months': 0.69, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00239, 't_months': 0.46, 'share_years_positive': 0.8, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00082, 't_months': 0.25, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': 0.00202, 't_months': 0.73, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2004
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
| 50–55 % | 28183 | 0.4726 | 0.5138 | -0.00162 | -0.00438 | -0.06083 | -0.00162 | -0.0412 | calibrated |
| 55–60 % | 52 | 0.4231 | 0.5833 | -0.00782 | -0.01065 | -0.0636 | -0.00782 | -0.1603 | low_n |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 104 | 0.5 | 0.6981 | 0.00092 | 0.0001 | -0.05128 | 0.00092 | -0.1981 | low_n |
| 70–75 % | 69 | 0.4783 | 0.7273 | 0.02422 | -0.00991 | -0.09632 | 0.02422 | -0.249 | low_n |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 25327 | 0.4745 | 0.5146 | -0.00027 | -0.00401 | -0.06067 | -0.00027 | -0.0401 | calibrated |
| 55–60 % | 387 | 0.4599 | 0.5651 | 0.00088 | -0.00683 | -0.07596 | 0.00088 | -0.1052 | overconfident |
| 60–65 % | 208 | 0.5144 | 0.625 | 0.00148 | 0.00208 | -0.07838 | 0.00148 | -0.1106 | overconfident |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 137 | 0.5036 | 0.75 | 0.02667 | 0.00557 | -0.09208 | 0.02667 | -0.2464 | low_n |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 6178 | 0.4752 | 0.5241 | -0.00227 | -0.00416 | -0.07213 | -0.00227 | -0.0489 | calibrated |
| 55–60 % | 0 | None | None | None | None | None | None | None | empty |
| 60–65 % | 52 | 0.4615 | 0.617 | -0.01336 | -0.01316 | -0.07179 | -0.01336 | -0.1555 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0019, 'recent_ic': -0.0394, 'prior_ic': 0.0316, 'trend': 'stable', 'trend_t': -0.83, 'calibration_slope_recent': -0.02501, 'meta_weight': 0.1891, 'contribution': 2e-05}
- enet_xs20_v1: {'oos_ic': 0.016, 'recent_ic': -0.1951, 'prior_ic': 0.069, 'trend': 'deteriorating', 'trend_t': -2.69, 'calibration_slope_recent': -0.06179, 'meta_weight': 0.0755, 'contribution': 0.00035}
- hgb_xs20_v1: {'oos_ic': 0.0238, 'recent_ic': 0.1478, 'prior_ic': -0.0319, 'trend': 'improving', 'trend_t': 3.55, 'calibration_slope_recent': 0.04595, 'meta_weight': 0.2561, 'contribution': -0.0001}
- hgb_asym20_v1: {'oos_ic': 0.0184, 'recent_ic': 0.1529, 'prior_ic': -0.038, 'trend': 'improving', 'trend_t': 3.0, 'calibration_slope_recent': 0.05551, 'meta_weight': 0.1972, 'contribution': -0.00073}
- hgb_xs20_momentum_v1: {'oos_ic': 0.01, 'recent_ic': 0.0444, 'prior_ic': -0.0038, 'trend': 'stable', 'trend_t': 1.49, 'calibration_slope_recent': 0.01255, 'meta_weight': 0.0, 'contribution': 0.00085}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0276, 'recent_ic': 0.0932, 'prior_ic': 0.0005, 'trend': 'stable', 'trend_t': 1.39, 'calibration_slope_recent': 0.04507, 'meta_weight': 0.2821, 'contribution': 0.00049}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
