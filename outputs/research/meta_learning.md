# Meta-Learning-Validierung – 2026-09-29T16:26:01+00:00

Version meta-v1 · Panel-Hash 2646988555eefd2d · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): REJECT** – nicht erfüllt: delta_sharpe_positive, bootstrap_ci_positive, years_positive, halves_positive, not_outlier_driven, locked_delta_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0223 | 0.229 | 0.249 | -0.1875 | 0.119 | 1.068 | 0.4742 | 0.08871 | -0.07492 | 1.184 | 0.00267 | 0.24954 | 0.00309 | 0.2657 | 0.1677 | 0.3456 | 11587 |
| best_single_ex_ante | -0.0288 | -0.159 | -0.167 | -0.3266 | -0.088 | 0.959 | 0.459 | 0.08471 | -0.0749 | 1.131 | -0.00165 | 0.2499 | 0.01372 | 0.2461 | 0.1499 | 0.2752 | 11587 |
| trailing_ic_weighted | 0.0087 | 0.131 | 0.125 | -0.2097 | 0.041 | 1.025 | 0.4747 | 0.08569 | -0.07557 | 1.134 | 0.00098 | 0.24941 | 0.00243 | 0.2577 | 0.1619 | 0.3002 | 11587 |
| meta_regime_weights | 0.0156 | 0.193 | 0.197 | -0.1723 | 0.091 | 1.056 | 0.4723 | 0.08848 | -0.075 | 1.18 | 0.00222 | 0.2497 | 0.01007 | 0.2618 | 0.166 | 0.3524 | 11587 |
| meta_regime_weights__no_regime | 0.0573 | 0.554 | 0.837 | -0.148 | 0.387 | 1.152 | 0.482 | 0.08777 | -0.07087 | 1.238 | 0.00559 | 0.24965 | 0.01326 | 0.2631 | 0.166 | 0.379 | 11587 |
| meta_regime_weights__no_failure_memory | -0.0024 | 0.051 | 0.052 | -0.245 | -0.01 | 1.026 | 0.4661 | 0.08908 | -0.07582 | 1.175 | 0.00104 | 0.24936 | 0.00526 | 0.2605 | 0.1682 | 0.329 | 11587 |
| meta_regime_weights__no_disagreement | 0.0269 | 0.285 | 0.314 | -0.1847 | 0.146 | 1.082 | 0.4763 | 0.08899 | -0.0748 | 1.19 | 0.00322 | 0.2497 | 0.0106 | 0.2655 | 0.1685 | 0.3595 | 11587 |
| meta_stacking | 0.0103 | 0.141 | 0.158 | -0.2174 | 0.047 | 1.048 | 0.4739 | 0.09585 | -0.08241 | 1.163 | 0.00207 | 0.24953 | 0.00288 | 0.2772 | 0.1794 | 0.3214 | 11587 |
| meta_stacking__no_regime | 0.0241 | 0.244 | 0.271 | -0.2016 | 0.119 | 1.075 | 0.4824 | 0.09499 | -0.08235 | 1.153 | 0.00319 | 0.24933 | 0.00255 | 0.2821 | 0.182 | 0.3225 | 11587 |
| meta_stacking__no_disagreement | 0.0133 | 0.162 | 0.182 | -0.2062 | 0.065 | 1.056 | 0.4778 | 0.09589 | -0.08305 | 1.155 | 0.00244 | 0.24947 | 0.00269 | 0.2786 | 0.1788 | 0.3016 | 11587 |
| meta_stacking__no_failure_memory | -0.039 | -0.223 | -0.23 | -0.3241 | -0.12 | 0.95 | 0.4567 | 0.09377 | -0.08295 | 1.13 | -0.00224 | 0.24933 | 0.00213 | 0.2653 | 0.1717 | 0.3364 | 11587 |
| meta_stacking__no_sector | 0.0026 | 0.084 | 0.099 | -0.2399 | 0.011 | 1.031 | 0.4691 | 0.09839 | -0.08432 | 1.167 | 0.00139 | 0.24935 | 0.00356 | 0.2757 | 0.1821 | 0.3567 | 11587 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.388, Δ Expectancy -0.00432, Δ MaxDD -0.1391, Δ PF -0.109, Δ Hit -0.0152, Δ Brier 0.00036, Δ ECE 0.01063, Δ Prec@K -0.0196 · Monats-Δ -0.004321 CI [-0.00799, -0.000519]
- **trailing_ic_weighted**: Δ Sharpe -0.098, Δ Expectancy -0.00169, Δ MaxDD -0.0222, Δ PF -0.043, Δ Hit 0.0005, Δ Brier -0.00013, Δ ECE -0.00066, Δ Prec@K -0.008 · Monats-Δ -0.001189 CI [-0.006758, 0.004956]
- **meta_regime_weights**: Δ Sharpe -0.036, Δ Expectancy -0.00045, Δ MaxDD 0.0152, Δ PF -0.012, Δ Hit -0.0019, Δ Brier 0.00016, Δ ECE 0.00698, Δ Prec@K -0.0039 · Monats-Δ -0.000773 CI [-0.005461, 0.003925]
- **meta_regime_weights__no_regime**: Δ Sharpe 0.325, Δ Expectancy 0.00292, Δ MaxDD 0.0395, Δ PF 0.084, Δ Hit 0.0078, Δ Brier 0.00011, Δ ECE 0.01017, Δ Prec@K -0.0026 · Monats-Δ 0.002561 CI [-0.002596, 0.008445]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe -0.178, Δ Expectancy -0.00163, Δ MaxDD -0.0575, Δ PF -0.042, Δ Hit -0.0081, Δ Brier -0.00018, Δ ECE 0.00217, Δ Prec@K -0.0052 · Monats-Δ -0.001985 CI [-0.006296, 0.002814]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.056, Δ Expectancy 0.00055, Δ MaxDD 0.0028, Δ PF 0.014, Δ Hit 0.0021, Δ Brier 0.00016, Δ ECE 0.00751, Δ Prec@K -0.0002 · Monats-Δ 0.000181 CI [-0.004342, 0.004761]
- **meta_stacking**: Δ Sharpe -0.088, Δ Expectancy -0.0006, Δ MaxDD -0.0299, Δ PF -0.02, Δ Hit -0.0003, Δ Brier -1e-05, Δ ECE -0.00021, Δ Prec@K 0.0115 · Monats-Δ -0.000964 CI [-0.00768, 0.005548]
- **meta_stacking__no_regime**: Δ Sharpe 0.015, Δ Expectancy 0.00052, Δ MaxDD -0.0141, Δ PF 0.007, Δ Hit 0.0082, Δ Brier -0.00021, Δ ECE -0.00054, Δ Prec@K 0.0164 · Monats-Δ 0.000117 CI [-0.006646, 0.007353]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.067, Δ Expectancy -0.00023, Δ MaxDD -0.0187, Δ PF -0.012, Δ Hit 0.0036, Δ Brier -7e-05, Δ ECE -0.0004, Δ Prec@K 0.0129 · Monats-Δ -0.00063 CI [-0.006866, 0.005841]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.452, Δ Expectancy -0.00491, Δ MaxDD -0.1366, Δ PF -0.118, Δ Hit -0.0175, Δ Brier -0.00021, Δ ECE -0.00096, Δ Prec@K -0.0004 · Monats-Δ -0.005126 CI [-0.011064, 0.000316]
- **meta_stacking__no_sector**: Δ Sharpe -0.145, Δ Expectancy -0.00128, Δ MaxDD -0.0524, Δ PF -0.037, Δ Hit -0.0051, Δ Brier -0.00019, Δ ECE 0.00047, Δ Prec@K 0.01 · Monats-Δ -0.001645 CI [-0.008059, 0.005042]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ❌ delta_sharpe_positive: -0.036
- ❌ bootstrap_ci_positive: [-0.005461, 0.003925]
- ❌ years_positive: 0.4
- ❌ halves_positive: [0.00158, -0.00304]
- ✅ no_regime_collapse: {'vix_lt_20': -0.00225, 'vix_ge_20': 0.00237, 'spy_uptrend': -0.00155, 'spy_downtrend': 0.00307}
- ✅ calibration_not_worse: {'brier': [0.2497, 0.24954], 'ece': [0.01007, 0.00309]}
- ❌ not_outlier_driven: -0.002832
- ❌ locked_delta_positive: -0.00714

## Locked-Holdout

- static_equal: Expectancy 0.01883, Sharpe 1.475, Trades 3111
- best_single_ex_ante: Expectancy 0.03908, Sharpe 2.347, Trades 3111
- trailing_ic_weighted: Expectancy 0.01597, Sharpe 1.259, Trades 3111
- meta_regime_weights: Expectancy 0.01169, Sharpe 0.989, Trades 3111
- meta_regime_weights__no_regime: Expectancy 0.01198, Sharpe 1.018, Trades 3111
- meta_regime_weights__no_failure_memory: Expectancy 0.01806, Sharpe 1.415, Trades 3111
- meta_regime_weights__no_disagreement: Expectancy 0.00979, Sharpe 0.861, Trades 3111
- meta_stacking: Expectancy 0.02436, Sharpe 1.956, Trades 3111
- meta_stacking__no_regime: Expectancy 0.01995, Sharpe 1.753, Trades 3111
- meta_stacking__no_disagreement: Expectancy 0.02391, Sharpe 2.711, Trades 3111
- meta_stacking__no_failure_memory: Expectancy 0.02912, Sharpe 2.957, Trades 3111
- meta_stacking__no_sector: Expectancy 0.00793, Sharpe 1.417, Trades 3111

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': 0.00559, 'sharpe': 0.554, 'brier': 0.24965, 'delta_vs_full_expectancy': 0.00337}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00104, 'sharpe': 0.051, 'brier': 0.24936, 'delta_vs_full_expectancy': -0.00118}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00322, 'sharpe': 0.285, 'brier': 0.2497, 'delta_vs_full_expectancy': 0.001}
- meta_stacking__no_regime: {'expectancy': 0.00319, 'sharpe': 0.244, 'brier': 0.24933, 'delta_vs_full_expectancy': 0.00112}
- meta_stacking__no_disagreement: {'expectancy': 0.00244, 'sharpe': 0.162, 'brier': 0.24947, 'delta_vs_full_expectancy': 0.00037}
- meta_stacking__no_failure_memory: {'expectancy': -0.00224, 'sharpe': -0.223, 'brier': 0.24933, 'delta_vs_full_expectancy': -0.00431}
- meta_stacking__no_sector: {'expectancy': 0.00139, 'sharpe': 0.084, 'brier': 0.24935, 'delta_vs_full_expectancy': -0.00068}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00267, 'sharpe': 0.229}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00093, 'liquid_half': 0.00034, 'less_liquid_half': 0.00405}
- meta_stacking: {'rebalance_4w': -0.0025, 'liquid_half': 0.00017, 'less_liquid_half': 0.00571}
- static_equal: {'rebalance_4w': 0.00185, 'liquid_half': 0.00187, 'less_liquid_half': 0.00527}
- best_single_ex_ante: {'rebalance_4w': -0.00191, 'liquid_half': -0.00221, 'less_liquid_half': 0.00111}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00414, 't_months': 0.9, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00262, 't_months': 0.48, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.0031, 't_months': 0.69, 'share_years_positive': 0.8, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00918, 't_months': -2.65, 'share_years_positive': 0.2, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2512
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: –; versagt: –
- **hgb_asym20_v1** – funktioniert: Communication Services (IC 0.0505, t 2.06), Consumer Cyclical (IC 0.0502, t 2.65); versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: –
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: –

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15534 | 0.4591 | 0.5046 | 0.00105 | -0.00826 | -0.07418 | 0.00105 | -0.0455 | calibrated |
| 55–60 % | 367 | 0.5531 | 0.5551 | 0.05108 | 0.02248 | -0.12059 | 0.05108 | -0.002 | calibrated |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 165 | 0.4727 | 0.6646 | 0.03941 | -0.00176 | -0.07678 | 0.03941 | -0.1919 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 6329 | 0.4405 | 0.5134 | -0.00432 | -0.00907 | -0.05506 | -0.00432 | -0.0729 | overconfident |
| 55–60 % | 1223 | 0.4693 | 0.5601 | 0.00798 | -0.00519 | -0.0673 | 0.00798 | -0.0907 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 5189 | 0.5209 | 0.508 | 0.01784 | 0.00544 | -0.09044 | 0.01784 | 0.0129 | calibrated |
| 55–60 % | 1259 | 0.5226 | 0.5513 | 0.02873 | 0.00509 | -0.09939 | 0.02873 | -0.0287 | calibrated |
| 60–65 % | 122 | 0.5164 | 0.6307 | 0.04961 | 0.02022 | -0.09581 | 0.04961 | -0.1143 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.008, 'recent_ic': -0.1172, 'prior_ic': 0.0711, 'trend': 'deteriorating', 'trend_t': -2.65, 'calibration_slope_recent': -0.04374, 'meta_weight': 0.9385, 'contribution': -0.00041}
- enet_xs20_v1: {'oos_ic': 0.0322, 'recent_ic': -0.0052, 'prior_ic': 0.1107, 'trend': 'stable', 'trend_t': -1.82, 'calibration_slope_recent': 0.00555, 'meta_weight': 0.0615, 'contribution': 0.00098}
- hgb_xs20_v1: {'oos_ic': 0.0189, 'recent_ic': -0.0256, 'prior_ic': 0.0451, 'trend': 'stable', 'trend_t': -0.97, 'calibration_slope_recent': -0.01114, 'meta_weight': 0.0, 'contribution': -0.00127}
- hgb_asym20_v1: {'oos_ic': 0.02, 'recent_ic': -0.0645, 'prior_ic': 0.0361, 'trend': 'deteriorating', 'trend_t': -2.04, 'calibration_slope_recent': -0.01482, 'meta_weight': 0.0, 'contribution': -0.0005}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0169, 'recent_ic': 0.0621, 'prior_ic': 0.0302, 'trend': 'stable', 'trend_t': 1.15, 'calibration_slope_recent': 0.02401, 'meta_weight': 0.0, 'contribution': 0.00014}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0095, 'recent_ic': -0.0533, 'prior_ic': 0.0089, 'trend': 'stable', 'trend_t': -1.02, 'calibration_slope_recent': -0.0153, 'meta_weight': 0.0, 'contribution': -0.00255}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
