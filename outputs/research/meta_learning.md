# Meta-Learning-Validierung – 2026-10-02T21:24:55+00:00

Version meta-v1 · Panel-Hash b1124bae4f858176 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0033 | 0.08 | 0.09 | -0.1869 | 0.018 | 1.02 | 0.4851 | 0.06165 | -0.05697 | 1.082 | 0.00058 | 0.25029 | 0.01514 | 0.2208 | 0.1209 | 0.4582 | 11059 |
| best_single_ex_ante | -0.0131 | -0.126 | -0.149 | -0.2349 | -0.056 | 0.968 | 0.4749 | 0.0627 | -0.05856 | 1.071 | -0.00097 | 0.25059 | 0.02097 | 0.2129 | 0.122 | 0.4742 | 11059 |
| trailing_ic_weighted | -0.0088 | -0.096 | -0.09 | -0.1799 | -0.049 | 0.972 | 0.4864 | 0.06068 | -0.0591 | 1.027 | -0.00084 | 0.24993 | 0.00317 | 0.2191 | 0.1178 | 0.4056 | 11059 |
| meta_regime_weights | 0.0055 | 0.106 | 0.124 | -0.1495 | 0.037 | 1.031 | 0.4806 | 0.06643 | -0.0596 | 1.115 | 0.00097 | 0.25055 | 0.01526 | 0.2281 | 0.1347 | 0.4487 | 11059 |
| meta_regime_weights__no_regime | -0.03 | -0.329 | -0.336 | -0.2314 | -0.13 | 0.943 | 0.4701 | 0.06462 | -0.0608 | 1.063 | -0.00184 | 0.25058 | 0.01773 | 0.2156 | 0.1227 | 0.4517 | 11059 |
| meta_regime_weights__no_failure_memory | 0.0101 | 0.153 | 0.152 | -0.1153 | 0.088 | 1.052 | 0.4906 | 0.06438 | -0.05893 | 1.092 | 0.00157 | 0.24995 | 0.00947 | 0.2303 | 0.1309 | 0.4183 | 11059 |
| meta_regime_weights__no_disagreement | 0.022 | 0.323 | 0.384 | -0.1342 | 0.164 | 1.074 | 0.4902 | 0.06572 | -0.05886 | 1.117 | 0.00221 | 0.25011 | 0.01069 | 0.2323 | 0.1368 | 0.433 | 11059 |
| meta_stacking | 0.0054 | 0.102 | 0.116 | -0.241 | 0.022 | 1.031 | 0.4938 | 0.06782 | -0.06418 | 1.057 | 0.001 | 0.25011 | 0.01063 | 0.2411 | 0.1412 | 0.4485 | 11059 |
| meta_stacking__no_regime | 0.0051 | 0.1 | 0.108 | -0.2306 | 0.022 | 1.03 | 0.4865 | 0.06931 | -0.06376 | 1.087 | 0.00098 | 0.25025 | 0.01019 | 0.2415 | 0.1402 | 0.4383 | 11059 |
| meta_stacking__no_disagreement | -0.0209 | -0.155 | -0.183 | -0.2793 | -0.075 | 0.972 | 0.4766 | 0.0685 | -0.06417 | 1.067 | -0.00094 | 0.25028 | 0.00404 | 0.234 | 0.1374 | 0.3906 | 11059 |
| meta_stacking__no_failure_memory | -0.0125 | -0.106 | -0.124 | -0.1905 | -0.065 | 0.991 | 0.4756 | 0.06718 | -0.06149 | 1.093 | -0.00029 | 0.25005 | 0.0013 | 0.2272 | 0.1321 | 0.5014 | 11059 |
| meta_stacking__no_sector | 0.0065 | 0.115 | 0.136 | -0.2514 | 0.026 | 1.037 | 0.4882 | 0.06969 | -0.06413 | 1.087 | 0.0012 | 0.25017 | 0.01576 | 0.238 | 0.1402 | 0.4901 | 11059 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.206, Δ Expectancy -0.00155, Δ MaxDD -0.048, Δ PF -0.052, Δ Hit -0.0102, Δ Brier 0.0003, Δ ECE 0.00583, Δ Prec@K -0.0079 · Monats-Δ -0.001421 CI [-0.005069, 0.00252]
- **trailing_ic_weighted**: Δ Sharpe -0.176, Δ Expectancy -0.00142, Δ MaxDD 0.007, Δ PF -0.048, Δ Hit 0.0013, Δ Brier -0.00036, Δ ECE -0.01197, Δ Prec@K -0.0017 · Monats-Δ -0.001126 CI [-0.005153, 0.002956]
- **meta_regime_weights**: Δ Sharpe 0.026, Δ Expectancy 0.00039, Δ MaxDD 0.0374, Δ PF 0.011, Δ Hit -0.0045, Δ Brier 0.00026, Δ ECE 0.00012, Δ Prec@K 0.0073 · Monats-Δ 0.000173 CI [-0.003414, 0.003698]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.409, Δ Expectancy -0.00242, Δ MaxDD -0.0445, Δ PF -0.077, Δ Hit -0.015, Δ Brier 0.00029, Δ ECE 0.00259, Δ Prec@K -0.0052 · Monats-Δ -0.002838 CI [-0.006918, 0.00093]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.073, Δ Expectancy 0.00099, Δ MaxDD 0.0716, Δ PF 0.032, Δ Hit 0.0055, Δ Brier -0.00034, Δ ECE -0.00567, Δ Prec@K 0.0095 · Monats-Δ 0.000618 CI [-0.002984, 0.004497]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.243, Δ Expectancy 0.00163, Δ MaxDD 0.0527, Δ PF 0.054, Δ Hit 0.0051, Δ Brier -0.00018, Δ ECE -0.00445, Δ Prec@K 0.0115 · Monats-Δ 0.001467 CI [-0.002655, 0.00565]
- **meta_stacking**: Δ Sharpe 0.022, Δ Expectancy 0.00042, Δ MaxDD -0.0541, Δ PF 0.011, Δ Hit 0.0087, Δ Brier -0.00018, Δ ECE -0.00451, Δ Prec@K 0.0203 · Monats-Δ 0.000255 CI [-0.005187, 0.005836]
- **meta_stacking__no_regime**: Δ Sharpe 0.02, Δ Expectancy 0.0004, Δ MaxDD -0.0437, Δ PF 0.01, Δ Hit 0.0014, Δ Brier -4e-05, Δ ECE -0.00495, Δ Prec@K 0.0207 · Monats-Δ 0.000192 CI [-0.005931, 0.006224]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.235, Δ Expectancy -0.00152, Δ MaxDD -0.0924, Δ PF -0.048, Δ Hit -0.0085, Δ Brier -1e-05, Δ ECE -0.0111, Δ Prec@K 0.0132 · Monats-Δ -0.001915 CI [-0.007752, 0.004137]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.186, Δ Expectancy -0.00087, Δ MaxDD -0.0036, Δ PF -0.029, Δ Hit -0.0095, Δ Brier -0.00024, Δ ECE -0.01384, Δ Prec@K 0.0064 · Monats-Δ -0.001332 CI [-0.006012, 0.003448]
- **meta_stacking__no_sector**: Δ Sharpe 0.035, Δ Expectancy 0.00062, Δ MaxDD -0.0645, Δ PF 0.017, Δ Hit 0.0031, Δ Brier -0.00012, Δ ECE 0.00062, Δ Prec@K 0.0172 · Monats-Δ 0.000304 CI [-0.004449, 0.005764]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.026
- ❌ bootstrap_ci_positive: [-0.003414, 0.003698]
- ✅ years_positive: 0.6
- ❌ halves_positive: [-0.00292, 0.00315]
- ✅ no_regime_collapse: {'vix_lt_20': 2e-05, 'vix_ge_20': 0.00099, 'spy_uptrend': -7e-05, 'spy_downtrend': 0.00187}
- ✅ calibration_not_worse: {'brier': [0.25055, 0.25029], 'ece': [0.01526, 0.01514]}
- ❌ not_outlier_driven: -0.001433
- ✅ locked_delta_positive: 0.00433

## Locked-Holdout

- static_equal: Expectancy 0.00186, Sharpe 0.319, Trades 3048
- best_single_ex_ante: Expectancy -0.0044, Sharpe -0.738, Trades 3048
- trailing_ic_weighted: Expectancy -0.0018, Sharpe -0.213, Trades 3048
- meta_regime_weights: Expectancy 0.00619, Sharpe 0.773, Trades 3048
- meta_regime_weights__no_regime: Expectancy 0.00542, Sharpe 0.842, Trades 3048
- meta_regime_weights__no_failure_memory: Expectancy -0.00166, Sharpe -0.153, Trades 3048
- meta_regime_weights__no_disagreement: Expectancy 0.00429, Sharpe 0.43, Trades 3048
- meta_stacking: Expectancy 0.01246, Sharpe 1.466, Trades 3048
- meta_stacking__no_regime: Expectancy 0.02295, Sharpe 2.169, Trades 3048
- meta_stacking__no_disagreement: Expectancy 0.01399, Sharpe 1.826, Trades 3048
- meta_stacking__no_failure_memory: Expectancy 0.00628, Sharpe 0.849, Trades 3048
- meta_stacking__no_sector: Expectancy 0.00806, Sharpe 1.447, Trades 3048

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00184, 'sharpe': -0.329, 'brier': 0.25058, 'delta_vs_full_expectancy': -0.00281}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00157, 'sharpe': 0.153, 'brier': 0.24995, 'delta_vs_full_expectancy': 0.0006}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00221, 'sharpe': 0.323, 'brier': 0.25011, 'delta_vs_full_expectancy': 0.00124}
- meta_stacking__no_regime: {'expectancy': 0.00098, 'sharpe': 0.1, 'brier': 0.25025, 'delta_vs_full_expectancy': -2e-05}
- meta_stacking__no_disagreement: {'expectancy': -0.00094, 'sharpe': -0.155, 'brier': 0.25028, 'delta_vs_full_expectancy': -0.00194}
- meta_stacking__no_failure_memory: {'expectancy': -0.00029, 'sharpe': -0.106, 'brier': 0.25005, 'delta_vs_full_expectancy': -0.00129}
- meta_stacking__no_sector: {'expectancy': 0.0012, 'sharpe': 0.115, 'brier': 0.25017, 'delta_vs_full_expectancy': 0.0002}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00058, 'sharpe': 0.08}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.0012, 'liquid_half': -0.00041, 'less_liquid_half': 0.00035}
- meta_stacking: {'rebalance_4w': -0.00161, 'liquid_half': 0.00174, 'less_liquid_half': -0.00112}
- static_equal: {'rebalance_4w': -0.00203, 'liquid_half': 0.0003, 'less_liquid_half': -0.00144}
- best_single_ex_ante: {'rebalance_4w': -0.00303, 'liquid_half': -0.00172, 'less_liquid_half': -0.00242}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00073, 't_months': 0.41, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00202, 't_months': 0.33, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00049, 't_months': 0.31, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': 0.00057, 't_months': 0.2, 'share_years_positive': 0.4, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2592
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: Consumer Defensive (IC 0.0486, t 2.21), Financial Services (IC 0.0641, t 2.3), Healthcare (IC 0.0475, t 2.01); versagt: –
- **hgb_asym20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: Energy (IC -0.0854, t -2.94)
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 26974 | 0.4639 | 0.5197 | -0.00279 | -0.00575 | -0.06038 | -0.00279 | -0.0558 | overconfident |
| 55–60 % | 364 | 0.4505 | 0.5689 | -0.0083 | -0.00712 | -0.05339 | -0.0083 | -0.1184 | overconfident |
| 60–65 % | 61 | 0.4918 | 0.6364 | 0.03065 | -0.00752 | -0.07945 | 0.03065 | -0.1446 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 16593 | 0.4708 | 0.5197 | -0.00039 | -0.00434 | -0.06032 | -0.00039 | -0.0489 | calibrated |
| 55–60 % | 911 | 0.4632 | 0.5647 | 0.0006 | -0.01002 | -0.08276 | 0.0006 | -0.1015 | overconfident |
| 60–65 % | 252 | 0.5635 | 0.6067 | 0.01572 | 0.01835 | -0.0866 | 0.01572 | -0.0432 | calibrated |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 60 | 0.4 | 0.7727 | 0.00224 | -0.03009 | -0.10799 | 0.00224 | -0.3727 | low_n |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 17631 | 0.4755 | 0.5249 | 0.00032 | -0.00393 | -0.06589 | 0.00032 | -0.0494 | calibrated |
| 55–60 % | 346 | 0.5347 | 0.5715 | 0.01646 | 0.01124 | -0.09222 | 0.01646 | -0.0369 | calibrated |
| 60–65 % | 269 | 0.5167 | 0.6369 | 0.01583 | 0.005 | -0.09159 | 0.01583 | -0.1202 | overconfident |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0047, 'recent_ic': -0.1045, 'prior_ic': 0.0652, 'trend': 'deteriorating', 'trend_t': -2.44, 'calibration_slope_recent': -0.0365, 'meta_weight': 0.3692, 'contribution': -0.00048}
- enet_xs20_v1: {'oos_ic': 0.0219, 'recent_ic': -0.0958, 'prior_ic': 0.0976, 'trend': 'deteriorating', 'trend_t': -2.76, 'calibration_slope_recent': -0.02426, 'meta_weight': 0.2072, 'contribution': -0.00055}
- hgb_xs20_v1: {'oos_ic': 0.0232, 'recent_ic': -0.0682, 'prior_ic': -0.0251, 'trend': 'stable', 'trend_t': -0.62, 'calibration_slope_recent': -0.02311, 'meta_weight': 0.1101, 'contribution': 0.00112}
- hgb_asym20_v1: {'oos_ic': 0.0166, 'recent_ic': 0.0333, 'prior_ic': -0.0046, 'trend': 'stable', 'trend_t': 1.01, 'calibration_slope_recent': 0.01324, 'meta_weight': 0.0998, 'contribution': -0.00036}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0097, 'recent_ic': 0.0766, 'prior_ic': -0.0226, 'trend': 'improving', 'trend_t': 3.71, 'calibration_slope_recent': 0.02253, 'meta_weight': 0.0, 'contribution': 0.00037}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0217, 'recent_ic': -0.0331, 'prior_ic': -0.0033, 'trend': 'stable', 'trend_t': -0.57, 'calibration_slope_recent': -0.01719, 'meta_weight': 0.2137, 'contribution': 0.00053}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': {'prob': 0.55, 'agreement_sd': 9.0, 'lcb': -0.013119132464450133, 'n': 211, 'mean': -0.006690535228670998}, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
