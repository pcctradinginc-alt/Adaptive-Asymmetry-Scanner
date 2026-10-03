# Meta-Learning-Validierung – 2026-10-03T13:15:09+00:00

Version meta-v1 · Panel-Hash 30753002d9156ed5 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.015 | 0.216 | 0.255 | -0.16 | 0.094 | 1.053 | 0.4877 | 0.06225 | -0.05627 | 1.106 | 0.00154 | 0.25022 | 0.01439 | 0.2233 | 0.1236 | 0.4704 | 11059 |
| best_single_ex_ante | -0.0036 | -0.016 | -0.018 | -0.2109 | -0.017 | 0.997 | 0.4839 | 0.06188 | -0.05818 | 1.063 | -8e-05 | 0.25006 | 0.01581 | 0.2134 | 0.1187 | 0.4672 | 11059 |
| trailing_ic_weighted | -0.0149 | -0.189 | -0.193 | -0.1845 | -0.081 | 0.959 | 0.4847 | 0.06025 | -0.05907 | 1.02 | -0.00124 | 0.24985 | 0.00364 | 0.2174 | 0.1173 | 0.4119 | 11059 |
| meta_regime_weights | 0.023 | 0.309 | 0.414 | -0.1252 | 0.184 | 1.077 | 0.488 | 0.06777 | -0.05996 | 1.13 | 0.00238 | 0.25029 | 0.01164 | 0.2347 | 0.1398 | 0.4495 | 11059 |
| meta_regime_weights__no_regime | -0.038 | -0.404 | -0.44 | -0.2621 | -0.145 | 0.921 | 0.4659 | 0.06345 | -0.06009 | 1.056 | -0.00254 | 0.25049 | 0.01579 | 0.2087 | 0.1182 | 0.4529 | 11059 |
| meta_regime_weights__no_failure_memory | 0.0168 | 0.221 | 0.222 | -0.121 | 0.139 | 1.072 | 0.4942 | 0.06388 | -0.05823 | 1.097 | 0.00211 | 0.24996 | 0.00964 | 0.2325 | 0.1305 | 0.415 | 11059 |
| meta_regime_weights__no_disagreement | 0.0286 | 0.402 | 0.473 | -0.1266 | 0.226 | 1.092 | 0.4914 | 0.06729 | -0.05955 | 1.13 | 0.00277 | 0.25006 | 0.01212 | 0.2368 | 0.1398 | 0.4405 | 11059 |
| meta_stacking | 0.0281 | 0.335 | 0.405 | -0.1043 | 0.269 | 1.088 | 0.502 | 0.07026 | -0.06508 | 1.08 | 0.00287 | 0.24989 | 0.01147 | 0.2502 | 0.1465 | 0.4435 | 11059 |
| meta_stacking__no_regime | 0.0107 | 0.16 | 0.178 | -0.1533 | 0.07 | 1.046 | 0.4887 | 0.0697 | -0.06368 | 1.095 | 0.0015 | 0.25 | 0.00405 | 0.2446 | 0.1415 | 0.3957 | 11059 |
| meta_stacking__no_disagreement | 0.0191 | 0.24 | 0.303 | -0.176 | 0.109 | 1.066 | 0.5042 | 0.06793 | -0.0648 | 1.048 | 0.00212 | 0.25017 | 0.00426 | 0.2454 | 0.1406 | 0.3869 | 11059 |
| meta_stacking__no_failure_memory | -0.0052 | -0.015 | -0.018 | -0.1712 | -0.03 | 1.016 | 0.4764 | 0.06955 | -0.06227 | 1.117 | 0.00054 | 0.24995 | 0.00172 | 0.235 | 0.1414 | 0.4784 | 11059 |
| meta_stacking__no_sector | 0.0107 | 0.164 | 0.208 | -0.154 | 0.07 | 1.042 | 0.4913 | 0.06974 | -0.06462 | 1.079 | 0.00139 | 0.24984 | 0.00683 | 0.2376 | 0.138 | 0.4999 | 11059 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.232, Δ Expectancy -0.00162, Δ MaxDD -0.0509, Δ PF -0.056, Δ Hit -0.0038, Δ Brier -0.00016, Δ ECE 0.00142, Δ Prec@K -0.0099 · Monats-Δ -0.001628 CI [-0.005957, 0.002751]
- **trailing_ic_weighted**: Δ Sharpe -0.405, Δ Expectancy -0.00278, Δ MaxDD -0.0245, Δ PF -0.094, Δ Hit -0.003, Δ Brier -0.00037, Δ ECE -0.01075, Δ Prec@K -0.0059 · Monats-Δ -0.002597 CI [-0.006287, 0.001032]
- **meta_regime_weights**: Δ Sharpe 0.093, Δ Expectancy 0.00084, Δ MaxDD 0.0348, Δ PF 0.024, Δ Hit 0.0003, Δ Brier 7e-05, Δ ECE -0.00275, Δ Prec@K 0.0114 · Monats-Δ 0.00065 CI [-0.003169, 0.004376]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.62, Δ Expectancy -0.00408, Δ MaxDD -0.1021, Δ PF -0.132, Δ Hit -0.0218, Δ Brier 0.00027, Δ ECE 0.0014, Δ Prec@K -0.0146 · Monats-Δ -0.004455 CI [-0.007767, -0.001192]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.005, Δ Expectancy 0.00057, Δ MaxDD 0.039, Δ PF 0.019, Δ Hit 0.0065, Δ Brier -0.00026, Δ ECE -0.00475, Δ Prec@K 0.0092 · Monats-Δ 0.000223 CI [-0.003207, 0.004053]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.186, Δ Expectancy 0.00123, Δ MaxDD 0.0334, Δ PF 0.039, Δ Hit 0.0037, Δ Brier -0.00016, Δ ECE -0.00227, Δ Prec@K 0.0135 · Monats-Δ 0.001061 CI [-0.002798, 0.005309]
- **meta_stacking**: Δ Sharpe 0.119, Δ Expectancy 0.00133, Δ MaxDD 0.0557, Δ PF 0.035, Δ Hit 0.0143, Δ Brier -0.00033, Δ ECE -0.00292, Δ Prec@K 0.0269 · Monats-Δ 0.001145 CI [-0.004319, 0.00695]
- **meta_stacking__no_regime**: Δ Sharpe -0.056, Δ Expectancy -4e-05, Δ MaxDD 0.0067, Δ PF -0.007, Δ Hit 0.001, Δ Brier -0.00022, Δ ECE -0.01034, Δ Prec@K 0.0213 · Monats-Δ -0.000294 CI [-0.006827, 0.006124]
- **meta_stacking__no_disagreement**: Δ Sharpe 0.024, Δ Expectancy 0.00058, Δ MaxDD -0.016, Δ PF 0.013, Δ Hit 0.0165, Δ Brier -5e-05, Δ ECE -0.01013, Δ Prec@K 0.0221 · Monats-Δ 0.000437 CI [-0.005528, 0.006109]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.231, Δ Expectancy -0.001, Δ MaxDD -0.0112, Δ PF -0.037, Δ Hit -0.0113, Δ Brier -0.00027, Δ ECE -0.01267, Δ Prec@K 0.0117 · Monats-Δ -0.001648 CI [-0.006896, 0.00392]
- **meta_stacking__no_sector**: Δ Sharpe -0.052, Δ Expectancy -0.00015, Δ MaxDD 0.006, Δ PF -0.011, Δ Hit 0.0036, Δ Brier -0.00038, Δ ECE -0.00756, Δ Prec@K 0.0143 · Monats-Δ -0.000335 CI [-0.005332, 0.005288]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.093
- ❌ bootstrap_ci_positive: [-0.003169, 0.004376]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00257, 0.00375]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00053, 'vix_ge_20': 0.00131, 'spy_uptrend': 0.00139, 'spy_downtrend': -0.00091}
- ✅ calibration_not_worse: {'brier': [0.25029, 0.25022], 'ece': [0.01164, 0.01439]}
- ❌ not_outlier_driven: -0.000933
- ✅ locked_delta_positive: 0.00804

## Locked-Holdout

- static_equal: Expectancy 0.00589, Sharpe 0.714, Trades 3048
- best_single_ex_ante: Expectancy 0.00964, Sharpe 1.341, Trades 3048
- trailing_ic_weighted: Expectancy 0.00389, Sharpe 0.427, Trades 3048
- meta_regime_weights: Expectancy 0.01393, Sharpe 1.618, Trades 3048
- meta_regime_weights__no_regime: Expectancy 0.00848, Sharpe 1.044, Trades 3048
- meta_regime_weights__no_failure_memory: Expectancy 0.00933, Sharpe 0.787, Trades 3048
- meta_regime_weights__no_disagreement: Expectancy 0.00733, Sharpe 0.686, Trades 3048
- meta_stacking: Expectancy 0.02523, Sharpe 2.001, Trades 3048
- meta_stacking__no_regime: Expectancy 0.01854, Sharpe 1.655, Trades 3048
- meta_stacking__no_disagreement: Expectancy 0.02083, Sharpe 2.379, Trades 3048
- meta_stacking__no_failure_memory: Expectancy 0.01769, Sharpe 1.836, Trades 3048
- meta_stacking__no_sector: Expectancy 0.01995, Sharpe 1.862, Trades 3048

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00254, 'sharpe': -0.404, 'brier': 0.25049, 'delta_vs_full_expectancy': -0.00492}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00211, 'sharpe': 0.221, 'brier': 0.24996, 'delta_vs_full_expectancy': -0.00027}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00277, 'sharpe': 0.402, 'brier': 0.25006, 'delta_vs_full_expectancy': 0.00039}
- meta_stacking__no_regime: {'expectancy': 0.0015, 'sharpe': 0.16, 'brier': 0.25, 'delta_vs_full_expectancy': -0.00137}
- meta_stacking__no_disagreement: {'expectancy': 0.00212, 'sharpe': 0.24, 'brier': 0.25017, 'delta_vs_full_expectancy': -0.00075}
- meta_stacking__no_failure_memory: {'expectancy': 0.00054, 'sharpe': -0.015, 'brier': 0.24995, 'delta_vs_full_expectancy': -0.00233}
- meta_stacking__no_sector: {'expectancy': 0.00139, 'sharpe': 0.164, 'brier': 0.24984, 'delta_vs_full_expectancy': -0.00148}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00154, 'sharpe': 0.216}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00061, 'liquid_half': 0.00151, 'less_liquid_half': 0.00115}
- meta_stacking: {'rebalance_4w': 0.00062, 'liquid_half': 0.00298, 'less_liquid_half': 0.00157}
- static_equal: {'rebalance_4w': -0.00084, 'liquid_half': 0.00095, 'less_liquid_half': -0.00114}
- best_single_ex_ante: {'rebalance_4w': -0.00034, 'liquid_half': -0.0014, 'less_liquid_half': -0.00117}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': -0.00085, 't_months': -0.36, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00128, 't_months': 0.29, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00011, 't_months': 0.08, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.0006, 't_months': -0.25, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2592
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: Consumer Cyclical (IC 0.0461, t 2.24), Financial Services (IC 0.0708, t 2.84); versagt: –
- **hgb_asym20_v1** – funktioniert: –; versagt: Utilities (IC -0.0448, t -2.29)
- **hgb_xs20_momentum_v1** – funktioniert: Technology (IC 0.0317, t 2.01); versagt: Energy (IC -0.0827, t -2.76)
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 29764 | 0.462 | 0.5186 | -0.00256 | -0.00594 | -0.06114 | -0.00256 | -0.0566 | overconfident |
| 55–60 % | 425 | 0.4447 | 0.5762 | -0.00083 | -0.00924 | -0.06035 | -0.00083 | -0.1315 | overconfident |
| 60–65 % | 104 | 0.5 | 0.6126 | -0.00392 | 0.00107 | -0.049 | -0.00392 | -0.1126 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 27934 | 0.4825 | 0.5251 | 0.00168 | -0.00291 | -0.06196 | 0.00168 | -0.0426 | calibrated |
| 55–60 % | 74 | 0.4865 | 0.5917 | 0.01473 | -0.00196 | -0.06714 | 0.01473 | -0.1052 | low_n |
| 60–65 % | 104 | 0.4808 | 0.6024 | -0.00619 | -0.00333 | -0.05271 | -0.00619 | -0.1216 | low_n |
| 65–70 % | 61 | 0.4918 | 0.6818 | 0.0281 | -0.01011 | -0.09125 | 0.0281 | -0.19 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 21717 | 0.4845 | 0.505 | -0.00111 | -0.0023 | -0.06269 | -0.00111 | -0.0205 | calibrated |
| 55–60 % | 2705 | 0.5327 | 0.5637 | 0.02566 | 0.0088 | -0.08791 | 0.02566 | -0.0309 | calibrated |
| 60–65 % | 104 | 0.4808 | 0.6296 | -0.00481 | -0.00333 | -0.0986 | -0.00481 | -0.1489 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0047, 'recent_ic': -0.1045, 'prior_ic': 0.0652, 'trend': 'deteriorating', 'trend_t': -2.44, 'calibration_slope_recent': -0.0365, 'meta_weight': 0.4383, 'contribution': 0.00044}
- enet_xs20_v1: {'oos_ic': 0.0219, 'recent_ic': -0.0958, 'prior_ic': 0.0976, 'trend': 'deteriorating', 'trend_t': -2.76, 'calibration_slope_recent': -0.02426, 'meta_weight': 0.1938, 'contribution': 0.00012}
- hgb_xs20_v1: {'oos_ic': 0.0355, 'recent_ic': 0.0194, 'prior_ic': 0.0445, 'trend': 'stable', 'trend_t': -0.36, 'calibration_slope_recent': 0.00598, 'meta_weight': 0.1539, 'contribution': 0.00202}
- hgb_asym20_v1: {'oos_ic': 0.0016, 'recent_ic': 0.0071, 'prior_ic': -0.0252, 'trend': 'stable', 'trend_t': 0.72, 'calibration_slope_recent': -0.00301, 'meta_weight': 0.0, 'contribution': 8e-05}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0107, 'recent_ic': 0.0729, 'prior_ic': -0.0206, 'trend': 'improving', 'trend_t': 3.51, 'calibration_slope_recent': 0.02264, 'meta_weight': 0.0, 'contribution': 0.00061}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0188, 'recent_ic': -0.0483, 'prior_ic': -0.0028, 'trend': 'stable', 'trend_t': -0.79, 'calibration_slope_recent': -0.01816, 'meta_weight': 0.214, 'contribution': 0.00112}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': {'prob': 0.55, 'agreement_sd': 9.0, 'lcb': -0.016666331182231254, 'n': 284, 'mean': -0.01062606202407421}, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
