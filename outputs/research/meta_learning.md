# Meta-Learning-Validierung – 2026-10-31T14:49:39+00:00

Version meta-v1 · Panel-Hash 4c39bb7c6ce59193 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | -0.0001 | 0.04 | 0.046 | -0.1945 | -0.0 | 1.009 | 0.4834 | 0.06223 | -0.05769 | 1.079 | 0.00028 | 0.25031 | 0.0155 | 0.223 | 0.1216 | 0.4669 | 11006 |
| best_single_ex_ante | -0.0274 | -0.291 | -0.33 | -0.2597 | -0.105 | 0.934 | 0.4671 | 0.06419 | -0.06023 | 1.066 | -0.00211 | 0.25053 | 0.01952 | 0.2128 | 0.1241 | 0.4723 | 11006 |
| trailing_ic_weighted | -0.0065 | -0.057 | -0.055 | -0.1527 | -0.042 | 0.986 | 0.4892 | 0.06192 | -0.06014 | 1.03 | -0.00043 | 0.25009 | 0.008 | 0.2275 | 0.1233 | 0.3982 | 11006 |
| meta_regime_weights | 0.0105 | 0.159 | 0.184 | -0.1392 | 0.076 | 1.044 | 0.4813 | 0.06822 | -0.06062 | 1.125 | 0.00139 | 0.25018 | 0.01078 | 0.2335 | 0.1407 | 0.4779 | 11006 |
| meta_regime_weights__no_regime | -0.0405 | -0.387 | -0.412 | -0.2928 | -0.138 | 0.919 | 0.4648 | 0.0645 | -0.06099 | 1.058 | -0.00266 | 0.25039 | 0.01404 | 0.2103 | 0.1217 | 0.4585 | 11006 |
| meta_regime_weights__no_failure_memory | 0.0121 | 0.173 | 0.183 | -0.1429 | 0.085 | 1.059 | 0.4886 | 0.06558 | -0.05913 | 1.109 | 0.0018 | 0.24989 | 0.00866 | 0.2336 | 0.1326 | 0.4268 | 11006 |
| meta_regime_weights__no_disagreement | 0.0215 | 0.297 | 0.382 | -0.136 | 0.158 | 1.071 | 0.4876 | 0.068 | -0.06044 | 1.125 | 0.00219 | 0.24998 | 0.01032 | 0.2349 | 0.1424 | 0.4756 | 11006 |
| meta_stacking | 0.01 | 0.155 | 0.167 | -0.2701 | 0.037 | 1.038 | 0.4872 | 0.06906 | -0.06323 | 1.092 | 0.00122 | 0.25002 | 0.00906 | 0.2448 | 0.1389 | 0.5123 | 11006 |
| meta_stacking__no_regime | -0.0121 | -0.089 | -0.11 | -0.2703 | -0.045 | 0.985 | 0.4803 | 0.07015 | -0.06582 | 1.066 | -0.00052 | 0.25004 | 0.01348 | 0.2391 | 0.1379 | 0.5129 | 11006 |
| meta_stacking__no_disagreement | -0.0192 | -0.167 | -0.195 | -0.3109 | -0.062 | 0.969 | 0.4782 | 0.06953 | -0.06577 | 1.057 | -0.00107 | 0.24997 | 0.01051 | 0.2371 | 0.1383 | 0.4434 | 11006 |
| meta_stacking__no_failure_memory | -0.0149 | -0.16 | -0.191 | -0.1926 | -0.078 | 0.985 | 0.4766 | 0.06625 | -0.06126 | 1.081 | -0.00048 | 0.24987 | 0.00101 | 0.2252 | 0.1261 | 0.6052 | 11006 |
| meta_stacking__no_sector | 0.01 | 0.155 | 0.167 | -0.2701 | 0.037 | 1.038 | 0.4872 | 0.06906 | -0.06323 | 1.092 | 0.00122 | 0.25002 | 0.00906 | 0.2448 | 0.1389 | 0.5123 | 11006 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.331, Δ Expectancy -0.00239, Δ MaxDD -0.0652, Δ PF -0.075, Δ Hit -0.0163, Δ Brier 0.00022, Δ ECE 0.00402, Δ Prec@K -0.0102 · Monats-Δ -0.002309 CI [-0.005492, 0.000976]
- **trailing_ic_weighted**: Δ Sharpe -0.097, Δ Expectancy -0.00071, Δ MaxDD 0.0418, Δ PF -0.023, Δ Hit 0.0058, Δ Brier -0.00022, Δ ECE -0.0075, Δ Prec@K 0.0045 · Monats-Δ -0.000616 CI [-0.004124, 0.003078]
- **meta_regime_weights**: Δ Sharpe 0.119, Δ Expectancy 0.00111, Δ MaxDD 0.0553, Δ PF 0.035, Δ Hit -0.0021, Δ Brier -0.00013, Δ ECE -0.00472, Δ Prec@K 0.0105 · Monats-Δ 0.000917 CI [-0.001976, 0.003829]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.427, Δ Expectancy -0.00294, Δ MaxDD -0.0983, Δ PF -0.09, Δ Hit -0.0186, Δ Brier 8e-05, Δ ECE -0.00146, Δ Prec@K -0.0127 · Monats-Δ -0.003352 CI [-0.007126, 0.000122]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.133, Δ Expectancy 0.00152, Δ MaxDD 0.0516, Δ PF 0.05, Δ Hit 0.0052, Δ Brier -0.00042, Δ ECE -0.00684, Δ Prec@K 0.0106 · Monats-Δ 0.001097 CI [-0.002989, 0.005235]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.257, Δ Expectancy 0.00191, Δ MaxDD 0.0585, Δ PF 0.062, Δ Hit 0.0042, Δ Brier -0.00033, Δ ECE -0.00518, Δ Prec@K 0.0119 · Monats-Δ 0.001774 CI [-0.001377, 0.005337]
- **meta_stacking**: Δ Sharpe 0.115, Δ Expectancy 0.00094, Δ MaxDD -0.0756, Δ PF 0.029, Δ Hit 0.0038, Δ Brier -0.00029, Δ ECE -0.00644, Δ Prec@K 0.0218 · Monats-Δ 0.000882 CI [-0.004616, 0.006162]
- **meta_stacking__no_regime**: Δ Sharpe -0.129, Δ Expectancy -0.0008, Δ MaxDD -0.0758, Δ PF -0.024, Δ Hit -0.0031, Δ Brier -0.00027, Δ ECE -0.00202, Δ Prec@K 0.0161 · Monats-Δ -0.000958 CI [-0.006185, 0.004399]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.207, Δ Expectancy -0.00135, Δ MaxDD -0.1164, Δ PF -0.04, Δ Hit -0.0052, Δ Brier -0.00034, Δ ECE -0.00499, Δ Prec@K 0.0141 · Monats-Δ -0.001557 CI [-0.006914, 0.00384]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.2, Δ Expectancy -0.00076, Δ MaxDD 0.0019, Δ PF -0.024, Δ Hit -0.0068, Δ Brier -0.00044, Δ ECE -0.01449, Δ Prec@K 0.0022 · Monats-Δ -0.001301 CI [-0.004866, 0.002533]
- **meta_stacking__no_sector**: Δ Sharpe 0.115, Δ Expectancy 0.00094, Δ MaxDD -0.0756, Δ PF 0.029, Δ Hit 0.0038, Δ Brier -0.00029, Δ ECE -0.00644, Δ Prec@K 0.0218 · Monats-Δ 0.000882 CI [-0.004616, 0.006162]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.119
- ❌ bootstrap_ci_positive: [-0.001976, 0.003829]
- ✅ years_positive: 0.6
- ❌ halves_positive: [-0.00279, 0.00449]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00132, 'vix_ge_20': 0.00077, 'spy_uptrend': 0.00158, 'spy_downtrend': -0.00038}
- ✅ calibration_not_worse: {'brier': [0.25018, 0.25031], 'ece': [0.01078, 0.0155]}
- ❌ not_outlier_driven: -8.1e-05
- ✅ locked_delta_positive: 0.01146

## Locked-Holdout

- static_equal: Expectancy -0.00057, Sharpe -0.022, Trades 3397
- best_single_ex_ante: Expectancy 0.001, Sharpe 0.285, Trades 3397
- trailing_ic_weighted: Expectancy -0.00731, Sharpe -0.837, Trades 3397
- meta_regime_weights: Expectancy 0.01089, Sharpe 1.099, Trades 3397
- meta_regime_weights__no_regime: Expectancy -0.00141, Sharpe -0.286, Trades 3397
- meta_regime_weights__no_failure_memory: Expectancy 0.00202, Sharpe -0.03, Trades 3397
- meta_regime_weights__no_disagreement: Expectancy 0.00695, Sharpe 0.761, Trades 3397
- meta_stacking: Expectancy 0.00265, Sharpe 0.313, Trades 3397
- meta_stacking__no_regime: Expectancy -0.00052, Sharpe 0.16, Trades 3397
- meta_stacking__no_disagreement: Expectancy 0.00343, Sharpe 0.271, Trades 3397
- meta_stacking__no_failure_memory: Expectancy 0.01086, Sharpe 1.356, Trades 3397
- meta_stacking__no_sector: Expectancy 0.00265, Sharpe 0.313, Trades 3397

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00266, 'sharpe': -0.387, 'brier': 0.25039, 'delta_vs_full_expectancy': -0.00405}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.0018, 'sharpe': 0.173, 'brier': 0.24989, 'delta_vs_full_expectancy': 0.00041}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00219, 'sharpe': 0.297, 'brier': 0.24998, 'delta_vs_full_expectancy': 0.0008}
- meta_stacking__no_regime: {'expectancy': -0.00052, 'sharpe': -0.089, 'brier': 0.25004, 'delta_vs_full_expectancy': -0.00174}
- meta_stacking__no_disagreement: {'expectancy': -0.00107, 'sharpe': -0.167, 'brier': 0.24997, 'delta_vs_full_expectancy': -0.00229}
- meta_stacking__no_failure_memory: {'expectancy': -0.00048, 'sharpe': -0.16, 'brier': 0.24987, 'delta_vs_full_expectancy': -0.0017}
- meta_stacking__no_sector: {'expectancy': 0.00122, 'sharpe': 0.155, 'brier': 0.25002, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00028, 'sharpe': 0.04}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00212, 'liquid_half': 0.00134, 'less_liquid_half': -0.00025}
- meta_stacking: {'rebalance_4w': -0.00074, 'liquid_half': 0.00093, 'less_liquid_half': 0.00104}
- static_equal: {'rebalance_4w': -0.00037, 'liquid_half': 0.00039, 'less_liquid_half': -0.00126}
- best_single_ex_ante: {'rebalance_4w': -0.00293, 'liquid_half': -0.00135, 'less_liquid_half': -0.00433}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00208, 't_months': 0.98, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.001, 't_months': 0.33, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00279, 't_months': 1.51, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': 0.00037, 't_months': 0.1, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2185
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
| 50–55 % | 26844 | 0.4686 | 0.5141 | -0.00211 | -0.0048 | -0.05966 | -0.00211 | -0.0455 | calibrated |
| 55–60 % | 66 | 0.5606 | 0.5567 | 0.06073 | 0.03229 | -0.12215 | 0.06073 | 0.0039 | low_n |
| 60–65 % | 68 | 0.3971 | 0.6364 | -0.00839 | -0.02515 | -0.10968 | -0.00839 | -0.2393 | low_n |
| 65–70 % | 104 | 0.4135 | 0.661 | -0.00664 | -0.01367 | -0.05095 | -0.00664 | -0.2476 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 28143 | 0.4809 | 0.5134 | 0.00065 | -0.00314 | -0.06025 | 0.00065 | -0.0326 | calibrated |
| 55–60 % | 894 | 0.4911 | 0.5816 | 0.00663 | -0.0027 | -0.09024 | 0.00663 | -0.0905 | overconfident |
| 60–65 % | 52 | 0.4423 | 0.625 | -0.00665 | -0.01407 | -0.04979 | -0.00665 | -0.1827 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 14091 | 0.4884 | 0.5187 | 0.00118 | -0.00186 | -0.06665 | 0.00118 | -0.0303 | calibrated |
| 55–60 % | 480 | 0.5229 | 0.5799 | 0.00595 | 0.00438 | -0.09379 | 0.00595 | -0.057 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0029, 'recent_ic': -0.036, 'prior_ic': 0.0371, 'trend': 'stable', 'trend_t': -0.87, 'calibration_slope_recent': -0.01694, 'meta_weight': 0.1842, 'contribution': -0.00059}
- enet_xs20_v1: {'oos_ic': 0.0169, 'recent_ic': -0.1769, 'prior_ic': 0.0679, 'trend': 'deteriorating', 'trend_t': -2.5, 'calibration_slope_recent': -0.04902, 'meta_weight': 0.0, 'contribution': 0.00022}
- hgb_xs20_v1: {'oos_ic': 0.0223, 'recent_ic': 0.1573, 'prior_ic': -0.0475, 'trend': 'improving', 'trend_t': 3.4, 'calibration_slope_recent': 0.0467, 'meta_weight': 0.3959, 'contribution': 0.00085}
- hgb_asym20_v1: {'oos_ic': 0.012, 'recent_ic': 0.1248, 'prior_ic': -0.0451, 'trend': 'improving', 'trend_t': 3.11, 'calibration_slope_recent': 0.03815, 'meta_weight': 0.1128, 'contribution': -0.0001}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0102, 'recent_ic': 0.0507, 'prior_ic': -0.0062, 'trend': 'stable', 'trend_t': 1.64, 'calibration_slope_recent': 0.01363, 'meta_weight': 0.0, 'contribution': 0.00038}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0228, 'recent_ic': 0.1503, 'prior_ic': -0.0317, 'trend': 'improving', 'trend_t': 3.13, 'calibration_slope_recent': 0.04144, 'meta_weight': 0.307, 'contribution': 0.00047}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
