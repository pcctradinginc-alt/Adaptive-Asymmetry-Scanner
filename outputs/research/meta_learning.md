# Meta-Learning-Validierung – 2026-10-03T09:28:11+00:00

Version meta-v1 · Panel-Hash 27d80a9c11e2071a · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0075 | 0.127 | 0.146 | -0.1806 | 0.042 | 1.031 | 0.4862 | 0.06267 | -0.05749 | 1.09 | 0.00093 | 0.25029 | 0.01507 | 0.2233 | 0.1238 | 0.4565 | 11059 |
| best_single_ex_ante | -0.0254 | -0.273 | -0.298 | -0.2664 | -0.095 | 0.94 | 0.4716 | 0.06125 | -0.05817 | 1.053 | -0.00186 | 0.25054 | 0.01825 | 0.2049 | 0.1127 | 0.4893 | 11059 |
| trailing_ic_weighted | -0.0167 | -0.202 | -0.202 | -0.1853 | -0.09 | 0.955 | 0.4844 | 0.06056 | -0.0596 | 1.016 | -0.00139 | 0.24993 | 0.00278 | 0.2195 | 0.1177 | 0.4069 | 11059 |
| meta_regime_weights | 0.0126 | 0.184 | 0.237 | -0.1382 | 0.091 | 1.051 | 0.4859 | 0.06752 | -0.0607 | 1.112 | 0.00161 | 0.25035 | 0.01282 | 0.2326 | 0.1406 | 0.4454 | 11059 |
| meta_regime_weights__no_regime | -0.0291 | -0.293 | -0.299 | -0.2417 | -0.121 | 0.947 | 0.4715 | 0.0653 | -0.06151 | 1.062 | -0.00172 | 0.25056 | 0.01615 | 0.2172 | 0.1251 | 0.4477 | 11059 |
| meta_regime_weights__no_failure_memory | 0.0115 | 0.165 | 0.174 | -0.1142 | 0.101 | 1.058 | 0.4893 | 0.06577 | -0.05957 | 1.104 | 0.00176 | 0.24985 | 0.00797 | 0.234 | 0.1344 | 0.4136 | 11059 |
| meta_regime_weights__no_disagreement | 0.0168 | 0.248 | 0.313 | -0.143 | 0.118 | 1.064 | 0.4885 | 0.06664 | -0.05982 | 1.114 | 0.00195 | 0.25008 | 0.01098 | 0.235 | 0.1406 | 0.4395 | 11059 |
| meta_stacking | 0.018 | 0.237 | 0.282 | -0.1552 | 0.116 | 1.062 | 0.4982 | 0.06784 | -0.06343 | 1.07 | 0.00198 | 0.25015 | 0.00315 | 0.2422 | 0.1405 | 0.4368 | 11059 |
| meta_stacking__no_regime | -0.005 | -0.004 | -0.004 | -0.2576 | -0.02 | 1.002 | 0.478 | 0.07073 | -0.06466 | 1.094 | 5e-05 | 0.25035 | 0.00661 | 0.2414 | 0.141 | 0.3831 | 11059 |
| meta_stacking__no_disagreement | 0.0001 | 0.049 | 0.058 | -0.2256 | 0.0 | 1.019 | 0.4839 | 0.06884 | -0.06333 | 1.087 | 0.00063 | 0.25048 | 0.0048 | 0.2343 | 0.1414 | 0.3883 | 11059 |
| meta_stacking__no_failure_memory | -0.0133 | -0.105 | -0.117 | -0.221 | -0.06 | 0.99 | 0.4792 | 0.06695 | -0.06221 | 1.076 | -0.00032 | 0.25012 | 0.00605 | 0.2291 | 0.1335 | 0.5036 | 11059 |
| meta_stacking__no_sector | -0.0052 | -0.015 | -0.02 | -0.2596 | -0.02 | 1.004 | 0.4834 | 0.06825 | -0.06361 | 1.073 | 0.00013 | 0.24984 | 0.00735 | 0.234 | 0.1316 | 0.5076 | 11059 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.4, Δ Expectancy -0.00279, Δ MaxDD -0.0858, Δ PF -0.091, Δ Hit -0.0146, Δ Brier 0.00025, Δ ECE 0.00318, Δ Prec@K -0.0184 · Monats-Δ -0.002815 CI [-0.00673, 0.001329]
- **trailing_ic_weighted**: Δ Sharpe -0.329, Δ Expectancy -0.00232, Δ MaxDD -0.0047, Δ PF -0.076, Δ Hit -0.0018, Δ Brier -0.00036, Δ ECE -0.01229, Δ Prec@K -0.0038 · Monats-Δ -0.002144 CI [-0.005925, 0.001633]
- **meta_regime_weights**: Δ Sharpe 0.057, Δ Expectancy 0.00068, Δ MaxDD 0.0424, Δ PF 0.02, Δ Hit -0.0003, Δ Brier 6e-05, Δ ECE -0.00225, Δ Prec@K 0.0093 · Monats-Δ 0.000403 CI [-0.003184, 0.003905]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.42, Δ Expectancy -0.00265, Δ MaxDD -0.0611, Δ PF -0.084, Δ Hit -0.0147, Δ Brier 0.00027, Δ ECE 0.00108, Δ Prec@K -0.0061 · Monats-Δ -0.003096 CI [-0.007097, 0.000871]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.038, Δ Expectancy 0.00083, Δ MaxDD 0.0664, Δ PF 0.027, Δ Hit 0.0031, Δ Brier -0.00044, Δ ECE -0.0071, Δ Prec@K 0.0107 · Monats-Δ 0.000395 CI [-0.003305, 0.004324]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.121, Δ Expectancy 0.00102, Δ MaxDD 0.0376, Δ PF 0.033, Δ Hit 0.0023, Δ Brier -0.00021, Δ ECE -0.00409, Δ Prec@K 0.0117 · Monats-Δ 0.000698 CI [-0.003394, 0.004961]
- **meta_stacking**: Δ Sharpe 0.11, Δ Expectancy 0.00105, Δ MaxDD 0.0254, Δ PF 0.031, Δ Hit 0.012, Δ Brier -0.00014, Δ ECE -0.01192, Δ Prec@K 0.0189 · Monats-Δ 0.000889 CI [-0.004283, 0.006191]
- **meta_stacking__no_regime**: Δ Sharpe -0.131, Δ Expectancy -0.00088, Δ MaxDD -0.077, Δ PF -0.029, Δ Hit -0.0082, Δ Brier 6e-05, Δ ECE -0.00846, Δ Prec@K 0.0181 · Monats-Δ -0.000979 CI [-0.007755, 0.005492]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.078, Δ Expectancy -0.0003, Δ MaxDD -0.045, Δ PF -0.012, Δ Hit -0.0023, Δ Brier 0.00019, Δ ECE -0.01027, Δ Prec@K 0.011 · Monats-Δ -0.000544 CI [-0.00678, 0.005408]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.232, Δ Expectancy -0.00125, Δ MaxDD -0.0404, Δ PF -0.041, Δ Hit -0.007, Δ Brier -0.00017, Δ ECE -0.00902, Δ Prec@K 0.0058 · Monats-Δ -0.001736 CI [-0.00598, 0.002592]
- **meta_stacking__no_sector**: Δ Sharpe -0.142, Δ Expectancy -0.0008, Δ MaxDD -0.079, Δ PF -0.027, Δ Hit -0.0028, Δ Brier -0.00045, Δ ECE -0.00772, Δ Prec@K 0.0107 · Monats-Δ -0.001063 CI [-0.006159, 0.004562]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.057
- ❌ bootstrap_ci_positive: [-0.003184, 0.003905]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00347, 0.00413]
- ✅ no_regime_collapse: {'vix_lt_20': 0.0005, 'vix_ge_20': 0.00096, 'spy_uptrend': 0.00053, 'spy_downtrend': 0.00117}
- ✅ calibration_not_worse: {'brier': [0.25035, 0.25029], 'ece': [0.01282, 0.01507]}
- ❌ not_outlier_driven: -0.001186
- ✅ locked_delta_positive: 0.00478

## Locked-Holdout

- static_equal: Expectancy 0.00264, Sharpe 0.406, Trades 3048
- best_single_ex_ante: Expectancy -0.00593, Sharpe -0.992, Trades 3048
- trailing_ic_weighted: Expectancy 0.00035, Sharpe 0.006, Trades 3048
- meta_regime_weights: Expectancy 0.00742, Sharpe 1.045, Trades 3048
- meta_regime_weights__no_regime: Expectancy 0.00096, Sharpe 0.262, Trades 3048
- meta_regime_weights__no_failure_memory: Expectancy 0.00428, Sharpe 0.36, Trades 3048
- meta_regime_weights__no_disagreement: Expectancy -0.00026, Sharpe 0.014, Trades 3048
- meta_stacking: Expectancy 0.0119, Sharpe 2.053, Trades 3048
- meta_stacking__no_regime: Expectancy 0.01722, Sharpe 1.824, Trades 3048
- meta_stacking__no_disagreement: Expectancy 0.02138, Sharpe 2.376, Trades 3048
- meta_stacking__no_failure_memory: Expectancy 0.01334, Sharpe 1.342, Trades 3048
- meta_stacking__no_sector: Expectancy 0.00499, Sharpe 1.12, Trades 3048

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00172, 'sharpe': -0.293, 'brier': 0.25056, 'delta_vs_full_expectancy': -0.00333}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00176, 'sharpe': 0.165, 'brier': 0.24985, 'delta_vs_full_expectancy': 0.00015}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00195, 'sharpe': 0.248, 'brier': 0.25008, 'delta_vs_full_expectancy': 0.00034}
- meta_stacking__no_regime: {'expectancy': 5e-05, 'sharpe': -0.004, 'brier': 0.25035, 'delta_vs_full_expectancy': -0.00193}
- meta_stacking__no_disagreement: {'expectancy': 0.00063, 'sharpe': 0.049, 'brier': 0.25048, 'delta_vs_full_expectancy': -0.00135}
- meta_stacking__no_failure_memory: {'expectancy': -0.00032, 'sharpe': -0.105, 'brier': 0.25012, 'delta_vs_full_expectancy': -0.0023}
- meta_stacking__no_sector: {'expectancy': 0.00013, 'sharpe': -0.015, 'brier': 0.24984, 'delta_vs_full_expectancy': -0.00185}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00093, 'sharpe': 0.127}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': -0.00064, 'liquid_half': 0.0012, 'less_liquid_half': 0.00175}
- meta_stacking: {'rebalance_4w': 5e-05, 'liquid_half': 0.0024, 'less_liquid_half': -0.00056}
- static_equal: {'rebalance_4w': -0.00228, 'liquid_half': 0.00068, 'less_liquid_half': -0.00142}
- best_single_ex_ante: {'rebalance_4w': -0.0013, 'liquid_half': -0.00263, 'less_liquid_half': -0.00245}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': 0.00041, 't_months': 0.21, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00272, 't_months': 0.64, 'share_years_positive': 0.8, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': -0.00011, 't_months': -0.01, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00109, 't_months': -0.45, 'share_years_positive': 0.4, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2592
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: Consumer Defensive (IC 0.0484, t 2.12), Financial Services (IC 0.0645, t 2.54); versagt: Basic Materials (IC -0.0584, t -2.02)
- **hgb_asym20_v1** – funktioniert: Financial Services (IC 0.0498, t 2.02); versagt: –
- **hgb_xs20_momentum_v1** – funktioniert: –; versagt: Energy (IC -0.0823, t -2.81)
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: Basic Materials (IC -0.0565, t -2.03)

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 25156 | 0.4688 | 0.5147 | -0.00243 | -0.00478 | -0.06086 | -0.00243 | -0.0459 | calibrated |
| 55–60 % | 468 | 0.4338 | 0.5731 | -0.01124 | -0.01119 | -0.05664 | -0.01124 | -0.1394 | overconfident |
| 60–65 % | 61 | 0.4918 | 0.6364 | 0.03899 | -0.00568 | -0.08081 | 0.03899 | -0.1446 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 21094 | 0.4773 | 0.5254 | 0.00026 | -0.00352 | -0.06154 | 0.00026 | -0.0481 | calibrated |
| 55–60 % | 1609 | 0.4649 | 0.5549 | -0.00196 | -0.00626 | -0.05715 | -0.00196 | -0.0901 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 122 | 0.4836 | 0.75 | 0.01804 | -0.00603 | -0.09127 | 0.01804 | -0.2664 | low_n |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 5459 | 0.5027 | 0.5159 | 0.00057 | 0.00043 | -0.08282 | 0.00057 | -0.0133 | calibrated |
| 55–60 % | 3232 | 0.4882 | 0.5644 | 0.00979 | -0.00247 | -0.07406 | 0.00979 | -0.0761 | overconfident |
| 60–65 % | 111 | 0.6126 | 0.622 | 0.05419 | 0.02084 | -0.08467 | 0.05419 | -0.0094 | low_n |
| 65–70 % | 155 | 0.529 | 0.6667 | 0.00992 | 0.00631 | -0.0974 | 0.00992 | -0.1376 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0047, 'recent_ic': -0.1045, 'prior_ic': 0.0652, 'trend': 'deteriorating', 'trend_t': -2.44, 'calibration_slope_recent': -0.0365, 'meta_weight': 0.3471, 'contribution': 0.00068}
- enet_xs20_v1: {'oos_ic': 0.0219, 'recent_ic': -0.0958, 'prior_ic': 0.0976, 'trend': 'deteriorating', 'trend_t': -2.76, 'calibration_slope_recent': -0.02426, 'meta_weight': 0.1592, 'contribution': 9e-05}
- hgb_xs20_v1: {'oos_ic': 0.0212, 'recent_ic': -0.0723, 'prior_ic': -0.0145, 'trend': 'stable', 'trend_t': -0.81, 'calibration_slope_recent': -0.02561, 'meta_weight': 0.1717, 'contribution': 0.0014}
- hgb_asym20_v1: {'oos_ic': 0.0163, 'recent_ic': -0.0739, 'prior_ic': 0.0195, 'trend': 'deteriorating', 'trend_t': -2.62, 'calibration_slope_recent': -0.02487, 'meta_weight': 0.1471, 'contribution': -0.0002}
- hgb_xs20_momentum_v1: {'oos_ic': 0.011, 'recent_ic': 0.0733, 'prior_ic': -0.0213, 'trend': 'improving', 'trend_t': 3.53, 'calibration_slope_recent': 0.02259, 'meta_weight': 0.0, 'contribution': 0.00115}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0178, 'recent_ic': -0.0295, 'prior_ic': -0.008, 'trend': 'stable', 'trend_t': -0.4, 'calibration_slope_recent': -0.01323, 'meta_weight': 0.1749, 'contribution': 0.00059}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': {'prob': 0.55, 'agreement_sd': 9.0, 'lcb': -0.016747507965863866, 'n': 290, 'mean': -0.010802823832913692}, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
