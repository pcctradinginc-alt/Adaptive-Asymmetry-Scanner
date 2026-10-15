# Meta-Learning-Validierung – 2026-10-15T06:40:32+00:00

Version meta-v1 · Panel-Hash cf4c7603b0be4043 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | -0.0048 | -0.012 | -0.013 | -0.2062 | -0.024 | 0.998 | 0.4821 | 0.06231 | -0.05811 | 1.072 | -5e-05 | 0.25028 | 0.01555 | 0.2231 | 0.1228 | 0.4673 | 11006 |
| best_single_ex_ante | -0.0117 | -0.098 | -0.109 | -0.227 | -0.051 | 0.972 | 0.4772 | 0.06409 | -0.06018 | 1.065 | -0.00088 | 0.2507 | 0.0218 | 0.2209 | 0.1226 | 0.4877 | 11006 |
| trailing_ic_weighted | 0.0078 | 0.146 | 0.144 | -0.1351 | 0.058 | 1.021 | 0.4967 | 0.06133 | -0.0593 | 1.034 | 0.00062 | 0.25 | 0.00986 | 0.2257 | 0.1231 | 0.426 | 11006 |
| meta_regime_weights | 0.026 | 0.308 | 0.386 | -0.1056 | 0.247 | 1.09 | 0.4897 | 0.06966 | -0.06135 | 1.135 | 0.00281 | 0.25044 | 0.01656 | 0.243 | 0.1456 | 0.429 | 11006 |
| meta_regime_weights__no_regime | -0.0264 | -0.231 | -0.239 | -0.2356 | -0.112 | 0.957 | 0.4703 | 0.0669 | -0.06205 | 1.078 | -0.0014 | 0.25053 | 0.0167 | 0.2235 | 0.1287 | 0.4502 | 11006 |
| meta_regime_weights__no_failure_memory | 0.0271 | 0.323 | 0.359 | -0.0972 | 0.279 | 1.102 | 0.5002 | 0.06469 | -0.05873 | 1.101 | 0.003 | 0.24981 | 0.00744 | 0.236 | 0.1333 | 0.399 | 11006 |
| meta_regime_weights__no_disagreement | 0.0297 | 0.369 | 0.463 | -0.0969 | 0.307 | 1.098 | 0.4894 | 0.06881 | -0.06003 | 1.146 | 0.00302 | 0.25023 | 0.01413 | 0.2399 | 0.1432 | 0.4444 | 11006 |
| meta_stacking | -0.0142 | -0.12 | -0.145 | -0.221 | -0.064 | 0.987 | 0.4842 | 0.06876 | -0.06538 | 1.052 | -0.00043 | 0.24992 | 0.01338 | 0.2408 | 0.1385 | 0.469 | 11006 |
| meta_stacking__no_regime | 0.0103 | 0.157 | 0.197 | -0.21 | 0.049 | 1.041 | 0.4948 | 0.07098 | -0.06677 | 1.063 | 0.0014 | 0.24994 | 0.0102 | 0.2562 | 0.1472 | 0.4787 | 11006 |
| meta_stacking__no_disagreement | -0.0312 | -0.298 | -0.351 | -0.3031 | -0.103 | 0.938 | 0.4739 | 0.06601 | -0.06337 | 1.042 | -0.00205 | 0.25019 | 0.01794 | 0.2243 | 0.1273 | 0.4465 | 11006 |
| meta_stacking__no_failure_memory | -0.0099 | -0.085 | -0.102 | -0.1796 | -0.055 | 1.002 | 0.4865 | 0.06386 | -0.06037 | 1.058 | 6e-05 | 0.24994 | 0.01307 | 0.2232 | 0.1267 | 0.5947 | 11006 |
| meta_stacking__no_sector | -0.0142 | -0.12 | -0.145 | -0.221 | -0.064 | 0.987 | 0.4842 | 0.06876 | -0.06538 | 1.052 | -0.00043 | 0.24992 | 0.01338 | 0.2408 | 0.1385 | 0.469 | 11006 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.086, Δ Expectancy -0.00083, Δ MaxDD -0.0208, Δ PF -0.026, Δ Hit -0.0049, Δ Brier 0.00042, Δ ECE 0.00625, Δ Prec@K -0.0022 · Monats-Δ -0.0006 CI [-0.003807, 0.003015]
- **trailing_ic_weighted**: Δ Sharpe 0.158, Δ Expectancy 0.00067, Δ MaxDD 0.0711, Δ PF 0.023, Δ Hit 0.0146, Δ Brier -0.00028, Δ ECE -0.00569, Δ Prec@K 0.0026 · Monats-Δ 0.000942 CI [-0.002676, 0.004728]
- **meta_regime_weights**: Δ Sharpe 0.32, Δ Expectancy 0.00286, Δ MaxDD 0.1006, Δ PF 0.092, Δ Hit 0.0076, Δ Brier 0.00016, Δ ECE 0.00101, Δ Prec@K 0.0199 · Monats-Δ 0.002622 CI [-0.000886, 0.006148]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.219, Δ Expectancy -0.00135, Δ MaxDD -0.0294, Δ PF -0.041, Δ Hit -0.0118, Δ Brier 0.00025, Δ ECE 0.00115, Δ Prec@K 0.0004 · Monats-Δ -0.001761 CI [-0.005852, 0.002361]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.335, Δ Expectancy 0.00305, Δ MaxDD 0.109, Δ PF 0.104, Δ Hit 0.0181, Δ Brier -0.00047, Δ ECE -0.00811, Δ Prec@K 0.0129 · Monats-Δ 0.002706 CI [-0.000648, 0.006162]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.381, Δ Expectancy 0.00307, Δ MaxDD 0.1093, Δ PF 0.1, Δ Hit 0.0073, Δ Brier -5e-05, Δ ECE -0.00142, Δ Prec@K 0.0168 · Monats-Δ 0.002862 CI [-0.000286, 0.006025]
- **meta_stacking**: Δ Sharpe -0.108, Δ Expectancy -0.00038, Δ MaxDD -0.0148, Δ PF -0.011, Δ Hit 0.0021, Δ Brier -0.00036, Δ ECE -0.00217, Δ Prec@K 0.0177 · Monats-Δ -0.000791 CI [-0.00636, 0.004776]
- **meta_stacking__no_regime**: Δ Sharpe 0.169, Δ Expectancy 0.00145, Δ MaxDD -0.0038, Δ PF 0.043, Δ Hit 0.0127, Δ Brier -0.00034, Δ ECE -0.00535, Δ Prec@K 0.0331 · Monats-Δ 0.001274 CI [-0.004709, 0.007282]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.286, Δ Expectancy -0.002, Δ MaxDD -0.0969, Δ PF -0.06, Δ Hit -0.0082, Δ Brier -9e-05, Δ ECE 0.00239, Δ Prec@K 0.0012 · Monats-Δ -0.002205 CI [-0.007337, 0.003084]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.073, Δ Expectancy 0.00011, Δ MaxDD 0.0266, Δ PF 0.004, Δ Hit 0.0044, Δ Brier -0.00034, Δ ECE -0.00248, Δ Prec@K 0.0001 · Monats-Δ -0.000479 CI [-0.004731, 0.004349]
- **meta_stacking__no_sector**: Δ Sharpe -0.108, Δ Expectancy -0.00038, Δ MaxDD -0.0148, Δ PF -0.011, Δ Hit 0.0021, Δ Brier -0.00036, Δ ECE -0.00217, Δ Prec@K 0.0177 · Monats-Δ -0.000791 CI [-0.00636, 0.004776]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.32
- ❌ bootstrap_ci_positive: [-0.000886, 0.006148]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00124, 0.00634]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00233, 'vix_ge_20': 0.0037, 'spy_uptrend': 0.00259, 'spy_downtrend': 0.00371}
- ✅ calibration_not_worse: {'brier': [0.25044, 0.25028], 'ece': [0.01656, 0.01555]}
- ✅ not_outlier_driven: 0.001213
- ✅ locked_delta_positive: 0.005

## Locked-Holdout

- static_equal: Expectancy 0.00296, Sharpe 0.584, Trades 3147
- best_single_ex_ante: Expectancy -0.0006, Sharpe 0.179, Trades 3147
- trailing_ic_weighted: Expectancy 0.004, Sharpe 0.633, Trades 3147
- meta_regime_weights: Expectancy 0.00796, Sharpe 1.001, Trades 3147
- meta_regime_weights__no_regime: Expectancy 0.00884, Sharpe 1.237, Trades 3147
- meta_regime_weights__no_failure_memory: Expectancy 0.0068, Sharpe 0.723, Trades 3147
- meta_regime_weights__no_disagreement: Expectancy 2e-05, Sharpe 0.17, Trades 3147
- meta_stacking: Expectancy 0.02127, Sharpe 2.324, Trades 3147
- meta_stacking__no_regime: Expectancy 0.01437, Sharpe 1.533, Trades 3147
- meta_stacking__no_disagreement: Expectancy 0.01503, Sharpe 2.076, Trades 3147
- meta_stacking__no_failure_memory: Expectancy 0.0141, Sharpe 1.497, Trades 3147
- meta_stacking__no_sector: Expectancy 0.02127, Sharpe 2.324, Trades 3147

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.0014, 'sharpe': -0.231, 'brier': 0.25053, 'delta_vs_full_expectancy': -0.00421}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.003, 'sharpe': 0.323, 'brier': 0.24981, 'delta_vs_full_expectancy': 0.00019}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00302, 'sharpe': 0.369, 'brier': 0.25023, 'delta_vs_full_expectancy': 0.00021}
- meta_stacking__no_regime: {'expectancy': 0.0014, 'sharpe': 0.157, 'brier': 0.24994, 'delta_vs_full_expectancy': 0.00183}
- meta_stacking__no_disagreement: {'expectancy': -0.00205, 'sharpe': -0.298, 'brier': 0.25019, 'delta_vs_full_expectancy': -0.00162}
- meta_stacking__no_failure_memory: {'expectancy': 6e-05, 'sharpe': -0.085, 'brier': 0.24994, 'delta_vs_full_expectancy': 0.00049}
- meta_stacking__no_sector: {'expectancy': -0.00043, 'sharpe': -0.12, 'brier': 0.24992, 'delta_vs_full_expectancy': 0.0}
- without_dynamic_weighting (= static_equal): {'expectancy': -5e-05, 'sharpe': -0.012}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00048, 'liquid_half': 0.00199, 'less_liquid_half': 0.00109}
- meta_stacking: {'rebalance_4w': -0.00345, 'liquid_half': -0.00144, 'less_liquid_half': -0.00033}
- static_equal: {'rebalance_4w': -0.00292, 'liquid_half': 0.00038, 'less_liquid_half': -0.00231}
- best_single_ex_ante: {'rebalance_4w': -0.00417, 'liquid_half': -0.0004, 'less_liquid_half': -0.00362}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': -5e-05, 't_months': -0.05, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': 0.00239, 't_months': 0.58, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': -0.00051, 't_months': -0.16, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00125, 't_months': -0.49, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2269
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
| 50–55 % | 14659 | 0.4706 | 0.5152 | -0.00261 | -0.00415 | -0.05741 | -0.00261 | -0.0445 | calibrated |
| 55–60 % | 293 | 0.4744 | 0.5591 | 0.01117 | -0.00741 | -0.07982 | 0.01117 | -0.0847 | overconfident |
| 60–65 % | 104 | 0.4327 | 0.6232 | -0.0129 | -0.01378 | -0.05763 | -0.0129 | -0.1905 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 17086 | 0.4707 | 0.5172 | -0.00086 | -0.00442 | -0.06192 | -0.00086 | -0.0465 | calibrated |
| 55–60 % | 335 | 0.4896 | 0.5781 | 0.00413 | -0.00162 | -0.08592 | 0.00413 | -0.0886 | overconfident |
| 60–65 % | 62 | 0.5645 | 0.6429 | 0.02543 | 0.02018 | -0.07784 | 0.02543 | -0.0783 | low_n |
| 65–70 % | 126 | 0.5079 | 0.6667 | 0.02486 | 0.00439 | -0.0868 | 0.02486 | -0.1587 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 29446 | 0.4819 | 0.5141 | 0.00162 | -0.00273 | -0.0643 | 0.00162 | -0.0321 | calibrated |
| 55–60 % | 115 | 0.5304 | 0.5514 | 0.06515 | 0.00616 | -0.10439 | 0.06515 | -0.021 | low_n |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0059, 'recent_ic': -0.0262, 'prior_ic': 0.0582, 'trend': 'stable', 'trend_t': -1.03, 'calibration_slope_recent': -0.01119, 'meta_weight': 0.2779, 'contribution': -0.00072}
- enet_xs20_v1: {'oos_ic': 0.0231, 'recent_ic': -0.0321, 'prior_ic': 0.0858, 'trend': 'stable', 'trend_t': -1.47, 'calibration_slope_recent': -0.00382, 'meta_weight': 0.0015, 'contribution': 0.0}
- hgb_xs20_v1: {'oos_ic': 0.0225, 'recent_ic': 0.0001, 'prior_ic': -0.0239, 'trend': 'stable', 'trend_t': 0.32, 'calibration_slope_recent': -0.00544, 'meta_weight': 0.3207, 'contribution': 0.0003}
- hgb_asym20_v1: {'oos_ic': 0.0064, 'recent_ic': -0.0961, 'prior_ic': 0.0067, 'trend': 'deteriorating', 'trend_t': -2.26, 'calibration_slope_recent': -0.02984, 'meta_weight': 0.1316, 'contribution': -0.00133}
- hgb_xs20_momentum_v1: {'oos_ic': 0.011, 'recent_ic': 0.0978, 'prior_ic': -0.0202, 'trend': 'improving', 'trend_t': 4.52, 'calibration_slope_recent': 0.0282, 'meta_weight': 0.0, 'contribution': 0.00026}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0173, 'recent_ic': -0.0347, 'prior_ic': -0.0187, 'trend': 'stable', 'trend_t': -0.28, 'calibration_slope_recent': -0.01635, 'meta_weight': 0.2683, 'contribution': -0.00039}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': None, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
