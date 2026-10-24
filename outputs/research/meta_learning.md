# Meta-Learning-Validierung – 2026-10-24T13:03:43+00:00

Version meta-v1 · Panel-Hash f9747e705eb10b42 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | -0.0095 | -0.07 | -0.074 | -0.1985 | -0.048 | 0.988 | 0.4804 | 0.06221 | -0.05823 | 1.068 | -0.00037 | 0.25036 | 0.01644 | 0.2226 | 0.1201 | 0.4709 | 11006 |
| best_single_ex_ante | -0.0204 | -0.203 | -0.225 | -0.2564 | -0.08 | 0.953 | 0.4727 | 0.0636 | -0.05978 | 1.064 | -0.00147 | 0.2507 | 0.02269 | 0.2164 | 0.1213 | 0.4949 | 11006 |
| trailing_ic_weighted | -0.009 | -0.094 | -0.097 | -0.1671 | -0.054 | 0.978 | 0.4874 | 0.06118 | -0.0595 | 1.028 | -0.00068 | 0.2499 | 0.00185 | 0.2225 | 0.1201 | 0.4264 | 11006 |
| meta_regime_weights | 0.0267 | 0.329 | 0.42 | -0.1428 | 0.187 | 1.092 | 0.4916 | 0.06837 | -0.06052 | 1.13 | 0.00284 | 0.25028 | 0.01382 | 0.2362 | 0.1411 | 0.4505 | 11006 |
| meta_regime_weights__no_regime | -0.0266 | -0.218 | -0.243 | -0.2432 | -0.11 | 0.959 | 0.4709 | 0.06694 | -0.06211 | 1.078 | -0.00134 | 0.25059 | 0.01649 | 0.2222 | 0.1282 | 0.4417 | 11006 |
| meta_regime_weights__no_failure_memory | 0.0085 | 0.133 | 0.141 | -0.1451 | 0.059 | 1.053 | 0.4941 | 0.06547 | -0.06071 | 1.078 | 0.00163 | 0.24981 | 0.00621 | 0.238 | 0.1335 | 0.4229 | 11006 |
| meta_regime_weights__no_disagreement | 0.0098 | 0.154 | 0.18 | -0.1504 | 0.065 | 1.046 | 0.4828 | 0.06834 | -0.06098 | 1.121 | 0.00146 | 0.25028 | 0.01484 | 0.2351 | 0.1394 | 0.4482 | 11006 |
| meta_stacking | -0.0088 | -0.052 | -0.065 | -0.2463 | -0.036 | 0.996 | 0.484 | 0.07145 | -0.06728 | 1.062 | -0.00014 | 0.25004 | 0.01476 | 0.2418 | 0.1407 | 0.4796 | 11006 |
| meta_stacking__no_regime | -0.0023 | 0.013 | 0.015 | -0.2195 | -0.01 | 1.002 | 0.4869 | 0.06931 | -0.06567 | 1.056 | 6e-05 | 0.25028 | 0.01677 | 0.2395 | 0.1383 | 0.484 | 11006 |
| meta_stacking__no_disagreement | -0.018 | -0.145 | -0.161 | -0.2903 | -0.062 | 0.975 | 0.4831 | 0.06674 | -0.06398 | 1.043 | -0.00083 | 0.25011 | 0.0174 | 0.2324 | 0.1307 | 0.476 | 11006 |
| meta_stacking__no_failure_memory | -0.0138 | -0.133 | -0.176 | -0.1925 | -0.072 | 0.99 | 0.4833 | 0.06566 | -0.06201 | 1.059 | -0.00031 | 0.24996 | 0.00097 | 0.2299 | 0.133 | 0.5582 | 11006 |
| meta_stacking__no_sector | -0.0088 | -0.052 | -0.065 | -0.2463 | -0.036 | 0.996 | 0.484 | 0.07145 | -0.06728 | 1.062 | -0.00014 | 0.25004 | 0.01476 | 0.2418 | 0.1407 | 0.4796 | 11006 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.133, Δ Expectancy -0.0011, Δ MaxDD -0.0579, Δ PF -0.035, Δ Hit -0.0077, Δ Brier 0.00034, Δ ECE 0.00625, Δ Prec@K -0.0062 · Monats-Δ -0.000934 CI [-0.004237, 0.00243]
- **trailing_ic_weighted**: Δ Sharpe -0.024, Δ Expectancy -0.00031, Δ MaxDD 0.0314, Δ PF -0.01, Δ Hit 0.007, Δ Brier -0.00046, Δ ECE -0.01459, Δ Prec@K -0.0001 · Monats-Δ -5.4e-05 CI [-0.003688, 0.003574]
- **meta_regime_weights**: Δ Sharpe 0.399, Δ Expectancy 0.00321, Δ MaxDD 0.0557, Δ PF 0.104, Δ Hit 0.0112, Δ Brier -8e-05, Δ ECE -0.00262, Δ Prec@K 0.0136 · Monats-Δ 0.003037 CI [-0.000297, 0.006434]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.148, Δ Expectancy -0.00097, Δ MaxDD -0.0447, Δ PF -0.029, Δ Hit -0.0095, Δ Brier 0.00023, Δ ECE 5e-05, Δ Prec@K -0.0004 · Monats-Δ -0.001339 CI [-0.00553, 0.002889]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.203, Δ Expectancy 0.002, Δ MaxDD 0.0534, Δ PF 0.065, Δ Hit 0.0137, Δ Brier -0.00055, Δ ECE -0.01023, Δ Prec@K 0.0154 · Monats-Δ 0.001616 CI [-0.002099, 0.005437]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.224, Δ Expectancy 0.00183, Δ MaxDD 0.0481, Δ PF 0.058, Δ Hit 0.0024, Δ Brier -8e-05, Δ ECE -0.0016, Δ Prec@K 0.0125 · Monats-Δ 0.00162 CI [-0.001493, 0.004738]
- **meta_stacking**: Δ Sharpe 0.018, Δ Expectancy 0.00023, Δ MaxDD -0.0478, Δ PF 0.008, Δ Hit 0.0036, Δ Brier -0.00032, Δ ECE -0.00168, Δ Prec@K 0.0192 · Monats-Δ 0.000101 CI [-0.00505, 0.005602]
- **meta_stacking__no_regime**: Δ Sharpe 0.083, Δ Expectancy 0.00043, Δ MaxDD -0.021, Δ PF 0.014, Δ Hit 0.0065, Δ Brier -8e-05, Δ ECE 0.00033, Δ Prec@K 0.0169 · Monats-Δ 0.00059 CI [-0.004968, 0.006485]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.075, Δ Expectancy -0.00046, Δ MaxDD -0.0918, Δ PF -0.013, Δ Hit 0.0027, Δ Brier -0.00025, Δ ECE 0.00096, Δ Prec@K 0.0098 · Monats-Δ -0.000653 CI [-0.005653, 0.004155]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.063, Δ Expectancy 6e-05, Δ MaxDD 0.006, Δ PF 0.002, Δ Hit 0.0029, Δ Brier -0.0004, Δ ECE -0.01547, Δ Prec@K 0.0073 · Monats-Δ -0.0004 CI [-0.004374, 0.003804]
- **meta_stacking__no_sector**: Δ Sharpe 0.018, Δ Expectancy 0.00023, Δ MaxDD -0.0478, Δ PF 0.008, Δ Hit 0.0036, Δ Brier -0.00032, Δ ECE -0.00168, Δ Prec@K 0.0192 · Monats-Δ 0.000101 CI [-0.00505, 0.005602]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.399
- ❌ bootstrap_ci_positive: [-0.000297, 0.006434]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.0014, 0.00731]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00257, 'vix_ge_20': 0.00423, 'spy_uptrend': 0.00308, 'spy_downtrend': 0.00364}
- ✅ calibration_not_worse: {'brier': [0.25028, 0.25036], 'ece': [0.01382, 0.01644]}
- ✅ not_outlier_driven: 0.001576
- ✅ locked_delta_positive: 0.00932

## Locked-Holdout

- static_equal: Expectancy 0.00059, Sharpe 0.168, Trades 3247
- best_single_ex_ante: Expectancy 0.00066, Sharpe 0.184, Trades 3247
- trailing_ic_weighted: Expectancy -0.00055, Sharpe -0.038, Trades 3247
- meta_regime_weights: Expectancy 0.00991, Sharpe 1.127, Trades 3247
- meta_regime_weights__no_regime: Expectancy 0.00293, Sharpe 0.468, Trades 3247
- meta_regime_weights__no_failure_memory: Expectancy 0.00158, Sharpe 0.168, Trades 3247
- meta_regime_weights__no_disagreement: Expectancy 0.00311, Sharpe 0.302, Trades 3247
- meta_stacking: Expectancy 0.00458, Sharpe 0.982, Trades 3247
- meta_stacking__no_regime: Expectancy 0.0089, Sharpe 1.227, Trades 3247
- meta_stacking__no_disagreement: Expectancy 0.00046, Sharpe 0.126, Trades 3247
- meta_stacking__no_failure_memory: Expectancy 0.01043, Sharpe 1.391, Trades 3247
- meta_stacking__no_sector: Expectancy 0.00458, Sharpe 0.982, Trades 3247

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00134, 'sharpe': -0.218, 'brier': 0.25059, 'delta_vs_full_expectancy': -0.00418}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00163, 'sharpe': 0.133, 'brier': 0.24981, 'delta_vs_full_expectancy': -0.00121}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00146, 'sharpe': 0.154, 'brier': 0.25028, 'delta_vs_full_expectancy': -0.00138}
- meta_stacking__no_regime: {'expectancy': 6e-05, 'sharpe': 0.013, 'brier': 0.25028, 'delta_vs_full_expectancy': 0.0002}
- meta_stacking__no_disagreement: {'expectancy': -0.00083, 'sharpe': -0.145, 'brier': 0.25011, 'delta_vs_full_expectancy': -0.00069}
- meta_stacking__no_failure_memory: {'expectancy': -0.00031, 'sharpe': -0.133, 'brier': 0.24996, 'delta_vs_full_expectancy': -0.00017}
- meta_stacking__no_sector: {'expectancy': -0.00014, 'sharpe': -0.052, 'brier': 0.25004, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': -0.00037, 'sharpe': -0.07}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00079, 'liquid_half': 0.00325, 'less_liquid_half': 0.00105}
- meta_stacking: {'rebalance_4w': -0.00158, 'liquid_half': -0.00084, 'less_liquid_half': -0.00036}
- static_equal: {'rebalance_4w': -0.00329, 'liquid_half': -8e-05, 'less_liquid_half': -0.00223}
- best_single_ex_ante: {'rebalance_4w': -0.00243, 'liquid_half': -0.0009, 'less_liquid_half': -0.00356}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00097, 't_months': 0.37, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00423, 't_months': 1.44, 'share_years_positive': 0.8, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 5e-05, 't_months': 0.01, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.001, 't_months': -0.47, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2819
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
| 50–55 % | 12882 | 0.468 | 0.5204 | -0.00271 | -0.00456 | -0.05698 | -0.00271 | -0.0523 | overconfident |
| 55–60 % | 299 | 0.5017 | 0.5763 | 0.01582 | 0.00074 | -0.07238 | 0.01582 | -0.0747 | overconfident |
| 60–65 % | 104 | 0.4327 | 0.6021 | -0.01296 | -0.01315 | -0.05648 | -0.01296 | -0.1694 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 22181 | 0.4738 | 0.5156 | 0.0003 | -0.00405 | -0.06036 | 0.0003 | -0.0418 | calibrated |
| 55–60 % | 1209 | 0.4955 | 0.5836 | 0.01176 | -0.00084 | -0.08489 | 0.01176 | -0.0881 | overconfident |
| 60–65 % | 126 | 0.5635 | 0.6112 | 0.01158 | 0.00438 | -0.06552 | 0.01158 | -0.0477 | low_n |
| 65–70 % | 65 | 0.5692 | 0.6818 | 0.05569 | 0.01837 | -0.0789 | 0.05569 | -0.1126 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 28293 | 0.4778 | 0.5121 | -3e-05 | -0.00359 | -0.0638 | -3e-05 | -0.0343 | calibrated |
| 55–60 % | 92 | 0.4674 | 0.5812 | 0.01323 | -0.00496 | -0.13262 | 0.01323 | -0.1138 | low_n |
| 60–65 % | 61 | 0.541 | 0.6111 | -0.01114 | 0.0217 | -0.09807 | -0.01114 | -0.0701 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0048, 'recent_ic': -0.018, 'prior_ic': 0.0533, 'trend': 'stable', 'trend_t': -0.88, 'calibration_slope_recent': -0.00717, 'meta_weight': 0.5974, 'contribution': -0.00095}
- enet_xs20_v1: {'oos_ic': 0.0206, 'recent_ic': -0.0557, 'prior_ic': 0.0559, 'trend': 'stable', 'trend_t': -1.27, 'calibration_slope_recent': -0.008, 'meta_weight': 0.4026, 'contribution': -0.00041}
- hgb_xs20_v1: {'oos_ic': 0.0224, 'recent_ic': 0.0811, 'prior_ic': -0.0565, 'trend': 'improving', 'trend_t': 2.46, 'calibration_slope_recent': 0.01938, 'meta_weight': 0.0, 'contribution': -0.00015}
- hgb_asym20_v1: {'oos_ic': 0.0094, 'recent_ic': 0.032, 'prior_ic': -0.0342, 'trend': 'stable', 'trend_t': 1.23, 'calibration_slope_recent': 0.00316, 'meta_weight': 0.0, 'contribution': -0.00149}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0101, 'recent_ic': 0.064, 'prior_ic': -0.0113, 'trend': 'improving', 'trend_t': 2.41, 'calibration_slope_recent': 0.01748, 'meta_weight': 0.0, 'contribution': -0.00017}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0212, 'recent_ic': 0.0599, 'prior_ic': -0.0267, 'trend': 'stable', 'trend_t': 1.5, 'calibration_slope_recent': 0.01243, 'meta_weight': 0.0, 'contribution': -0.00068}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
