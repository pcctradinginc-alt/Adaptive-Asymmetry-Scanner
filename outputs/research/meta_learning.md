# Meta-Learning-Validierung – 2026-10-17T12:32:52+00:00

Version meta-v1 · Panel-Hash ec8f8b46a507b44f · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0008 | 0.052 | 0.054 | -0.2002 | 0.004 | 1.012 | 0.4865 | 0.06215 | -0.05819 | 1.068 | 0.00035 | 0.25041 | 0.01685 | 0.2263 | 0.1234 | 0.4588 | 11006 |
| best_single_ex_ante | -0.012 | -0.102 | -0.107 | -0.237 | -0.05 | 0.972 | 0.4768 | 0.06465 | -0.0606 | 1.067 | -0.00088 | 0.25071 | 0.02012 | 0.2208 | 0.1238 | 0.4932 | 11006 |
| trailing_ic_weighted | -0.0105 | -0.102 | -0.098 | -0.1779 | -0.059 | 0.974 | 0.4873 | 0.06155 | -0.06008 | 1.024 | -0.00082 | 0.25001 | 0.0087 | 0.2239 | 0.122 | 0.3967 | 11006 |
| meta_regime_weights | 0.0113 | 0.163 | 0.191 | -0.14 | 0.081 | 1.048 | 0.4873 | 0.06835 | -0.06197 | 1.103 | 0.00154 | 0.2504 | 0.01542 | 0.2361 | 0.1416 | 0.4367 | 11006 |
| meta_regime_weights__no_regime | -0.0115 | -0.076 | -0.083 | -0.1883 | -0.061 | 0.993 | 0.4812 | 0.06598 | -0.06165 | 1.07 | -0.00023 | 0.25059 | 0.01394 | 0.2242 | 0.1293 | 0.4531 | 11006 |
| meta_regime_weights__no_failure_memory | 0.0166 | 0.212 | 0.229 | -0.1171 | 0.142 | 1.072 | 0.4963 | 0.06552 | -0.06023 | 1.088 | 0.00218 | 0.24984 | 0.00622 | 0.2384 | 0.1347 | 0.4049 | 11006 |
| meta_regime_weights__no_disagreement | 0.0265 | 0.357 | 0.458 | -0.1339 | 0.198 | 1.089 | 0.4894 | 0.06799 | -0.05986 | 1.136 | 0.00271 | 0.25006 | 0.01115 | 0.2366 | 0.141 | 0.4535 | 11006 |
| meta_stacking | 0.0012 | 0.058 | 0.073 | -0.2379 | 0.005 | 1.02 | 0.4846 | 0.0704 | -0.06493 | 1.084 | 0.00066 | 0.25011 | 0.0032 | 0.2403 | 0.139 | 0.4748 | 11006 |
| meta_stacking__no_regime | -0.0144 | -0.128 | -0.149 | -0.2637 | -0.055 | 0.975 | 0.4823 | 0.06774 | -0.06471 | 1.047 | -0.00083 | 0.25023 | 0.01655 | 0.2326 | 0.1322 | 0.4637 | 11006 |
| meta_stacking__no_disagreement | -0.0109 | -0.069 | -0.082 | -0.2713 | -0.04 | 0.987 | 0.4783 | 0.06642 | -0.06169 | 1.077 | -0.00042 | 0.25023 | 0.00522 | 0.225 | 0.1285 | 0.4688 | 11006 |
| meta_stacking__no_failure_memory | -0.014 | -0.128 | -0.155 | -0.1764 | -0.08 | 0.993 | 0.48 | 0.06709 | -0.06237 | 1.076 | -0.00023 | 0.2499 | 0.00093 | 0.2296 | 0.134 | 0.554 | 11006 |
| meta_stacking__no_sector | 0.0012 | 0.058 | 0.073 | -0.2379 | 0.005 | 1.02 | 0.4846 | 0.0704 | -0.06493 | 1.084 | 0.00066 | 0.25011 | 0.0032 | 0.2403 | 0.139 | 0.4748 | 11006 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.154, Δ Expectancy -0.00123, Δ MaxDD -0.0368, Δ PF -0.04, Δ Hit -0.0097, Δ Brier 0.0003, Δ ECE 0.00327, Δ Prec@K -0.0055 · Monats-Δ -0.001095 CI [-0.004203, 0.002244]
- **trailing_ic_weighted**: Δ Sharpe -0.154, Δ Expectancy -0.00117, Δ MaxDD 0.0223, Δ PF -0.038, Δ Hit 0.0008, Δ Brier -0.0004, Δ ECE -0.00815, Δ Prec@K -0.0024 · Monats-Δ -0.001025 CI [-0.004382, 0.002504]
- **meta_regime_weights**: Δ Sharpe 0.111, Δ Expectancy 0.00119, Δ MaxDD 0.0602, Δ PF 0.036, Δ Hit 0.0008, Δ Brier -1e-05, Δ ECE -0.00143, Δ Prec@K 0.0098 · Monats-Δ 0.000928 CI [-0.002588, 0.004485]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.128, Δ Expectancy -0.00058, Δ MaxDD 0.0119, Δ PF -0.019, Δ Hit -0.0053, Δ Brier 0.00018, Δ ECE -0.00291, Δ Prec@K -0.0021 · Monats-Δ -0.00098 CI [-0.005011, 0.002963]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.16, Δ Expectancy 0.00183, Δ MaxDD 0.0831, Δ PF 0.06, Δ Hit 0.0098, Δ Brier -0.00057, Δ ECE -0.01063, Δ Prec@K 0.0121 · Monats-Δ 0.001402 CI [-0.002456, 0.005012]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.305, Δ Expectancy 0.00236, Δ MaxDD 0.0663, Δ PF 0.077, Δ Hit 0.0029, Δ Brier -0.00035, Δ ECE -0.0057, Δ Prec@K 0.0103 · Monats-Δ 0.002069 CI [-0.001796, 0.006143]
- **meta_stacking**: Δ Sharpe 0.006, Δ Expectancy 0.00031, Δ MaxDD -0.0377, Δ PF 0.008, Δ Hit -0.0019, Δ Brier -0.0003, Δ ECE -0.01365, Δ Prec@K 0.014 · Monats-Δ 6.7e-05 CI [-0.005952, 0.006571]
- **meta_stacking__no_regime**: Δ Sharpe -0.18, Δ Expectancy -0.00118, Δ MaxDD -0.0635, Δ PF -0.037, Δ Hit -0.0042, Δ Brier -0.00018, Δ ECE -0.0003, Δ Prec@K 0.0063 · Monats-Δ -0.001295 CI [-0.007882, 0.005408]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.121, Δ Expectancy -0.00077, Δ MaxDD -0.0711, Δ PF -0.025, Δ Hit -0.0082, Δ Brier -0.00018, Δ ECE -0.01163, Δ Prec@K -0.0013 · Monats-Δ -0.000929 CI [-0.006638, 0.004925]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.18, Δ Expectancy -0.00058, Δ MaxDD 0.0238, Δ PF -0.019, Δ Hit -0.0065, Δ Brier -0.00051, Δ ECE -0.01592, Δ Prec@K 0.0033 · Monats-Δ -0.001277 CI [-0.005832, 0.003294]
- **meta_stacking__no_sector**: Δ Sharpe 0.006, Δ Expectancy 0.00031, Δ MaxDD -0.0377, Δ PF 0.008, Δ Hit -0.0019, Δ Brier -0.0003, Δ ECE -0.01365, Δ Prec@K 0.014 · Monats-Δ 6.7e-05 CI [-0.005952, 0.006571]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.111
- ❌ bootstrap_ci_positive: [-0.002588, 0.004485]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.0031, 0.0048]
- ✅ no_regime_collapse: {'vix_lt_20': -0.00031, 'vix_ge_20': 0.00355, 'spy_uptrend': 0.00042, 'spy_downtrend': 0.00364}
- ✅ calibration_not_worse: {'brier': [0.2504, 0.25041], 'ece': [0.01542, 0.01685]}
- ❌ not_outlier_driven: -0.000517
- ✅ locked_delta_positive: 0.00509

## Locked-Holdout

- static_equal: Expectancy 0.00339, Sharpe 0.484, Trades 3197
- best_single_ex_ante: Expectancy 0.00068, Sharpe 0.205, Trades 3197
- trailing_ic_weighted: Expectancy 0.00307, Sharpe 0.425, Trades 3197
- meta_regime_weights: Expectancy 0.00847, Sharpe 0.904, Trades 3197
- meta_regime_weights__no_regime: Expectancy 0.00257, Sharpe 0.523, Trades 3197
- meta_regime_weights__no_failure_memory: Expectancy 0.00739, Sharpe 0.68, Trades 3197
- meta_regime_weights__no_disagreement: Expectancy 0.00965, Sharpe 0.904, Trades 3197
- meta_stacking: Expectancy 0.0032, Sharpe 0.471, Trades 3197
- meta_stacking__no_regime: Expectancy 0.02035, Sharpe 2.415, Trades 3197
- meta_stacking__no_disagreement: Expectancy 0.00782, Sharpe 1.095, Trades 3197
- meta_stacking__no_failure_memory: Expectancy 0.00983, Sharpe 1.2, Trades 3197
- meta_stacking__no_sector: Expectancy 0.0032, Sharpe 0.471, Trades 3197

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00023, 'sharpe': -0.076, 'brier': 0.25059, 'delta_vs_full_expectancy': -0.00177}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00218, 'sharpe': 0.212, 'brier': 0.24984, 'delta_vs_full_expectancy': 0.00064}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00271, 'sharpe': 0.357, 'brier': 0.25006, 'delta_vs_full_expectancy': 0.00117}
- meta_stacking__no_regime: {'expectancy': -0.00083, 'sharpe': -0.128, 'brier': 0.25023, 'delta_vs_full_expectancy': -0.00149}
- meta_stacking__no_disagreement: {'expectancy': -0.00042, 'sharpe': -0.069, 'brier': 0.25023, 'delta_vs_full_expectancy': -0.00108}
- meta_stacking__no_failure_memory: {'expectancy': -0.00023, 'sharpe': -0.128, 'brier': 0.2499, 'delta_vs_full_expectancy': -0.00089}
- meta_stacking__no_sector: {'expectancy': 0.00066, 'sharpe': 0.058, 'brier': 0.25011, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00035, 'sharpe': 0.052}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00191, 'liquid_half': 0.00035, 'less_liquid_half': 0.00123}
- meta_stacking: {'rebalance_4w': -0.00299, 'liquid_half': 0.00105, 'less_liquid_half': -0.00212}
- static_equal: {'rebalance_4w': -0.00252, 'liquid_half': 0.00143, 'less_liquid_half': -0.00145}
- best_single_ex_ante: {'rebalance_4w': -0.00324, 'liquid_half': -0.00115, 'less_liquid_half': -0.00281}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': -0.00031, 't_months': -0.12, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00251, 't_months': 0.8, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00045, 't_months': 0.16, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': 0.00131, 't_months': 0.29, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2105
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
| 50–55 % | 17221 | 0.4741 | 0.5147 | -0.0019 | -0.00361 | -0.05755 | -0.0019 | -0.0406 | calibrated |
| 55–60 % | 52 | 0.4231 | 0.5882 | -0.00997 | -0.0175 | -0.05391 | -0.00997 | -0.1652 | low_n |
| 60–65 % | 104 | 0.3942 | 0.6415 | -0.01353 | -0.01639 | -0.05267 | -0.01353 | -0.2473 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 15983 | 0.4751 | 0.5161 | -0.00027 | -0.00375 | -0.063 | -0.00027 | -0.041 | calibrated |
| 55–60 % | 168 | 0.4524 | 0.5947 | -0.00997 | -0.0072 | -0.06863 | -0.00997 | -0.1423 | low_n |
| 60–65 % | 659 | 0.5083 | 0.6206 | 0.01738 | 0.00255 | -0.08567 | 0.01738 | -0.1122 | overconfident |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 14645 | 0.4822 | 0.5136 | 0.00197 | -0.0033 | -0.07619 | 0.00197 | -0.0314 | calibrated |
| 55–60 % | 123 | 0.4309 | 0.5696 | -0.0031 | -0.01364 | -0.10659 | -0.0031 | -0.1387 | low_n |
| 60–65 % | 102 | 0.4216 | 0.6301 | -0.01048 | -0.01533 | -0.11213 | -0.01048 | -0.2086 | low_n |
| 65–70 % | 156 | 0.391 | 0.6503 | -0.02223 | -0.01237 | -0.12566 | -0.02223 | -0.2593 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0055, 'recent_ic': -0.0133, 'prior_ic': 0.0544, 'trend': 'stable', 'trend_t': -0.85, 'calibration_slope_recent': -0.00452, 'meta_weight': 0.3252, 'contribution': -0.00052}
- enet_xs20_v1: {'oos_ic': 0.0221, 'recent_ic': -0.0306, 'prior_ic': 0.0703, 'trend': 'stable', 'trend_t': -1.27, 'calibration_slope_recent': 8e-05, 'meta_weight': 0.0514, 'contribution': 0.00018}
- hgb_xs20_v1: {'oos_ic': 0.026, 'recent_ic': 0.0558, 'prior_ic': -0.0177, 'trend': 'stable', 'trend_t': 1.15, 'calibration_slope_recent': 0.01352, 'meta_weight': 0.2078, 'contribution': -0.0005}
- hgb_asym20_v1: {'oos_ic': 0.0101, 'recent_ic': 0.0007, 'prior_ic': -0.0263, 'trend': 'stable', 'trend_t': 0.58, 'calibration_slope_recent': -0.00756, 'meta_weight': 0.0994, 'contribution': -0.00092}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0109, 'recent_ic': 0.0676, 'prior_ic': -0.014, 'trend': 'improving', 'trend_t': 2.71, 'calibration_slope_recent': 0.01915, 'meta_weight': 0.0, 'contribution': 0.00018}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0234, 'recent_ic': -0.0098, 'prior_ic': 0.0082, 'trend': 'stable', 'trend_t': -0.31, 'calibration_slope_recent': -0.00915, 'meta_weight': 0.3162, 'contribution': -0.0008}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
