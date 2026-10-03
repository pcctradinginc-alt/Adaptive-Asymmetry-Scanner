# Meta-Learning-Validierung – 2026-10-03T07:29:47+00:00

Version meta-v1 · Panel-Hash 87d212edf2f2dcd5 · Referenz: static_equal · Primär: meta_regime_weights · Leakage-Checks: OK
**Entscheidung (meta_regime_weights): NEED_MORE_DATA** – nicht erfüllt: bootstrap_ci_positive, halves_positive, not_outlier_driven
Aktives Ensemble für Research-Signale: static_equal

## Out-of-Sample (Meta-Testjahre vor Locked)

| Variante | CAGR | Sharpe | Sortino | MaxDD | Calmar | PF | Hit | Ø Gew. | Ø Verl. | Payoff | Expectancy | Brier | ECE | Prec@K | Recall stark | Turnover | Trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_equal | 0.0093 | 0.145 | 0.178 | -0.1788 | 0.052 | 1.036 | 0.4829 | 0.06266 | -0.05649 | 1.109 | 0.00105 | 0.25032 | 0.0153 | 0.2216 | 0.1223 | 0.4697 | 11059 |
| best_single_ex_ante | -0.0307 | -0.325 | -0.359 | -0.2823 | -0.109 | 0.925 | 0.4701 | 0.06123 | -0.05873 | 1.042 | -0.00234 | 0.25052 | 0.01944 | 0.2043 | 0.1135 | 0.4861 | 11059 |
| trailing_ic_weighted | -0.0171 | -0.205 | -0.201 | -0.1796 | -0.095 | 0.956 | 0.4845 | 0.06111 | -0.06006 | 1.017 | -0.00135 | 0.24986 | 0.0039 | 0.2208 | 0.1204 | 0.4095 | 11059 |
| meta_regime_weights | 0.0229 | 0.3 | 0.379 | -0.1519 | 0.151 | 1.081 | 0.4862 | 0.06813 | -0.05962 | 1.143 | 0.00249 | 0.25032 | 0.01342 | 0.2359 | 0.1417 | 0.4564 | 11059 |
| meta_regime_weights__no_regime | -0.0389 | -0.412 | -0.455 | -0.2728 | -0.143 | 0.92 | 0.4643 | 0.06387 | -0.06015 | 1.062 | -0.00257 | 0.25082 | 0.01843 | 0.2116 | 0.1202 | 0.4585 | 11059 |
| meta_regime_weights__no_failure_memory | 0.0117 | 0.168 | 0.177 | -0.1355 | 0.086 | 1.058 | 0.4924 | 0.06447 | -0.05911 | 1.091 | 0.00173 | 0.24989 | 0.01 | 0.2335 | 0.1308 | 0.4153 | 11059 |
| meta_regime_weights__no_disagreement | 0.0227 | 0.318 | 0.38 | -0.1357 | 0.167 | 1.078 | 0.4887 | 0.06709 | -0.05945 | 1.129 | 0.00238 | 0.25014 | 0.01207 | 0.2345 | 0.1387 | 0.4448 | 11059 |
| meta_stacking | 0.0308 | 0.38 | 0.47 | -0.1208 | 0.255 | 1.094 | 0.4963 | 0.07003 | -0.06311 | 1.11 | 0.00297 | 0.25029 | 0.00764 | 0.2445 | 0.1449 | 0.4175 | 11059 |
| meta_stacking__no_regime | 0.0248 | 0.307 | 0.343 | -0.1702 | 0.146 | 1.077 | 0.4961 | 0.07033 | -0.06427 | 1.094 | 0.0025 | 0.2505 | 0.00919 | 0.2495 | 0.1443 | 0.3935 | 11059 |
| meta_stacking__no_disagreement | -0.019 | -0.148 | -0.17 | -0.2805 | -0.068 | 0.973 | 0.4824 | 0.06713 | -0.0643 | 1.044 | -0.0009 | 0.2505 | 0.00686 | 0.2322 | 0.1345 | 0.3651 | 11059 |
| meta_stacking__no_failure_memory | -0.0157 | -0.128 | -0.145 | -0.2062 | -0.076 | 0.987 | 0.4785 | 0.0678 | -0.06305 | 1.075 | -0.00044 | 0.25002 | 0.00175 | 0.2273 | 0.1372 | 0.4732 | 11059 |
| meta_stacking__no_sector | -0.0162 | -0.144 | -0.173 | -0.2328 | -0.07 | 0.976 | 0.478 | 0.07042 | -0.0661 | 1.065 | -0.00085 | 0.24984 | 0.00841 | 0.2386 | 0.1392 | 0.4708 | 11059 |

## Deltas gegenüber Referenz (Bootstrap, Bonferroni)

- **best_single_ex_ante**: Δ Sharpe -0.47, Δ Expectancy -0.00339, Δ MaxDD -0.1035, Δ PF -0.111, Δ Hit -0.0128, Δ Brier 0.0002, Δ ECE 0.00414, Δ Prec@K -0.0173 · Monats-Δ -0.003404 CI [-0.007473, 0.000822]
- **trailing_ic_weighted**: Δ Sharpe -0.35, Δ Expectancy -0.0024, Δ MaxDD -0.0008, Δ PF -0.08, Δ Hit 0.0016, Δ Brier -0.00046, Δ ECE -0.0114, Δ Prec@K -0.0008 · Monats-Δ -0.002331 CI [-0.006206, 0.001517]
- **meta_regime_weights**: Δ Sharpe 0.155, Δ Expectancy 0.00144, Δ MaxDD 0.0269, Δ PF 0.045, Δ Hit 0.0033, Δ Brier 0.0, Δ ECE -0.00188, Δ Prec@K 0.0143 · Monats-Δ 0.001096 CI [-0.002575, 0.004773]
- **meta_regime_weights__no_regime**: Δ Sharpe -0.557, Δ Expectancy -0.00362, Δ MaxDD -0.094, Δ PF -0.116, Δ Hit -0.0186, Δ Brier 0.0005, Δ ECE 0.00313, Δ Prec@K -0.01 · Monats-Δ -0.004095 CI [-0.007652, -0.000626]
- **meta_regime_weights__no_failure_memory**: Δ Sharpe 0.023, Δ Expectancy 0.00068, Δ MaxDD 0.0433, Δ PF 0.022, Δ Hit 0.0095, Δ Brier -0.00043, Δ ECE -0.0053, Δ Prec@K 0.0119 · Monats-Δ 0.000236 CI [-0.003149, 0.0037]
- **meta_regime_weights__no_disagreement**: Δ Sharpe 0.173, Δ Expectancy 0.00133, Δ MaxDD 0.0431, Δ PF 0.042, Δ Hit 0.0058, Δ Brier -0.00018, Δ ECE -0.00323, Δ Prec@K 0.0129 · Monats-Δ 0.001029 CI [-0.00278, 0.004797]
- **meta_stacking**: Δ Sharpe 0.235, Δ Expectancy 0.00192, Δ MaxDD 0.058, Δ PF 0.058, Δ Hit 0.0134, Δ Brier -3e-05, Δ ECE -0.00766, Δ Prec@K 0.0229 · Monats-Δ 0.001756 CI [-0.004997, 0.008446]
- **meta_stacking__no_regime**: Δ Sharpe 0.162, Δ Expectancy 0.00145, Δ MaxDD 0.0086, Δ PF 0.041, Δ Hit 0.0132, Δ Brier 0.00018, Δ ECE -0.00611, Δ Prec@K 0.0279 · Monats-Δ 0.001295 CI [-0.00601, 0.008593]
- **meta_stacking__no_disagreement**: Δ Sharpe -0.293, Δ Expectancy -0.00195, Δ MaxDD -0.1017, Δ PF -0.063, Δ Hit -0.0005, Δ Brier 0.00018, Δ ECE -0.00844, Δ Prec@K 0.0106 · Monats-Δ -0.002309 CI [-0.009409, 0.004291]
- **meta_stacking__no_failure_memory**: Δ Sharpe -0.273, Δ Expectancy -0.00149, Δ MaxDD -0.0274, Δ PF -0.049, Δ Hit -0.0044, Δ Brier -0.0003, Δ ECE -0.01355, Δ Prec@K 0.0057 · Monats-Δ -0.002082 CI [-0.007298, 0.00317]
- **meta_stacking__no_sector**: Δ Sharpe -0.289, Δ Expectancy -0.0019, Δ MaxDD -0.054, Δ PF -0.06, Δ Hit -0.0049, Δ Brier -0.00048, Δ ECE -0.00689, Δ Prec@K 0.017 · Monats-Δ -0.002154 CI [-0.008002, 0.00404]

## Gate

- ✅ reproducible_leakage_checks: None
- ✅ enough_oos: [230, 53]
- ✅ delta_sharpe_positive: 0.155
- ❌ bootstrap_ci_positive: [-0.002575, 0.004773]
- ✅ years_positive: 0.8
- ❌ halves_positive: [-0.00356, 0.00558]
- ✅ no_regime_collapse: {'vix_lt_20': 0.00201, 'vix_ge_20': 0.00057, 'spy_uptrend': 0.00194, 'spy_downtrend': -0.00012}
- ✅ calibration_not_worse: {'brier': [0.25032, 0.25032], 'ece': [0.01342, 0.0153]}
- ❌ not_outlier_driven: -0.000397
- ✅ locked_delta_positive: 0.01788

## Locked-Holdout

- static_equal: Expectancy -0.00198, Sharpe -0.17, Trades 3048
- best_single_ex_ante: Expectancy -0.00624, Sharpe -1.013, Trades 3048
- trailing_ic_weighted: Expectancy -0.00478, Sharpe -0.542, Trades 3048
- meta_regime_weights: Expectancy 0.01589, Sharpe 2.104, Trades 3048
- meta_regime_weights__no_regime: Expectancy 0.00335, Sharpe 0.612, Trades 3048
- meta_regime_weights__no_failure_memory: Expectancy 0.00282, Sharpe 0.232, Trades 3048
- meta_regime_weights__no_disagreement: Expectancy 0.01022, Sharpe 1.201, Trades 3048
- meta_stacking: Expectancy 0.00987, Sharpe 1.52, Trades 3048
- meta_stacking__no_regime: Expectancy 0.01302, Sharpe 1.376, Trades 3048
- meta_stacking__no_disagreement: Expectancy 0.01096, Sharpe 1.698, Trades 3048
- meta_stacking__no_failure_memory: Expectancy 0.01686, Sharpe 2.049, Trades 3048
- meta_stacking__no_sector: Expectancy 0.01007, Sharpe 1.54, Trades 3048

## Ablationen

- meta_regime_weights__no_regime: {'expectancy': -0.00257, 'sharpe': -0.412, 'brier': 0.25082, 'delta_vs_full_expectancy': -0.00506}
- meta_regime_weights__no_failure_memory: {'expectancy': 0.00173, 'sharpe': 0.168, 'brier': 0.24989, 'delta_vs_full_expectancy': -0.00076}
- meta_regime_weights__no_disagreement: {'expectancy': 0.00238, 'sharpe': 0.318, 'brier': 0.25014, 'delta_vs_full_expectancy': -0.00011}
- meta_stacking__no_regime: {'expectancy': 0.0025, 'sharpe': 0.307, 'brier': 0.2505, 'delta_vs_full_expectancy': -0.00047}
- meta_stacking__no_disagreement: {'expectancy': -0.0009, 'sharpe': -0.148, 'brier': 0.2505, 'delta_vs_full_expectancy': -0.00387}
- meta_stacking__no_failure_memory: {'expectancy': -0.00044, 'sharpe': -0.128, 'brier': 0.25002, 'delta_vs_full_expectancy': -0.00341}
- meta_stacking__no_sector: {'expectancy': -0.00085, 'sharpe': -0.144, 'brier': 0.24984, 'delta_vs_full_expectancy': -0.00382}
- without_dynamic_weighting (= static_equal): {'expectancy': 0.00105, 'sharpe': 0.145}
- without_historical_analogies: n/a – Analogie-Engine ist nicht Teil der Basismodelle/Meta-Merkmale
- without_alternative_data: n/a – kein Basismodell nutzt alternative Daten (PIT-Historie zu kurz)

## Robustheit (Expectancy)

- meta_regime_weights: {'rebalance_4w': 0.00024, 'liquid_half': 0.00125, 'less_liquid_half': 0.00133}
- meta_stacking: {'rebalance_4w': -0.00194, 'liquid_half': 0.00316, 'less_liquid_half': 0.00143}
- static_equal: {'rebalance_4w': -0.00206, 'liquid_half': 0.00086, 'less_liquid_half': -0.00083}
- best_single_ex_ante: {'rebalance_4w': -0.00305, 'liquid_half': -0.00314, 'less_liquid_half': -0.00254}

## Disagreement (G)

- dis_rank_sd: {'low_minus_high_mean': -0.00083, 't_months': -0.33, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_bull_bear: {'low_minus_high_mean': -0.00217, 't_months': -0.76, 'share_years_positive': 0.4, 'empirically_supported': False}
- dis_rank_range: {'low_minus_high_mean': 0.00047, 't_months': 0.26, 'share_years_positive': 0.6, 'empirically_supported': False}
- dis_pred_sd: {'low_minus_high_mean': -0.00273, 't_months': -1.21, 'share_years_positive': 0.6, 'empirically_supported': False}
- use_in_score: False
- probability_disagreement: n/a – Basismodelle liefern Renditen/Ränge, keine Wahrscheinlichkeiten
- current_mean_rank_sd: 0.2592
- current_level: NORMAL

## Failure-Profile (F, nur gemessene Segmente mit |t| >= 2)

- **momentum_12_1** – funktioniert: –; versagt: –
- **enet_xs20_v1** – funktioniert: –; versagt: –
- **hgb_xs20_v1** – funktioniert: Financial Services (IC 0.072, t 3.05); versagt: Basic Materials (IC -0.064, t -2.22)
- **hgb_asym20_v1** – funktioniert: –; versagt: Utilities (IC -0.0459, t -2.29)
- **hgb_xs20_momentum_v1** – funktioniert: Technology (IC 0.0359, t 2.34); versagt: Energy (IC -0.082, t -2.9)
- **hgb_xs20_risk_regime_v1** – funktioniert: –; versagt: Basic Materials (IC -0.0598, t -2.19)

## Kalibrierungs-Buckets (P(Überrendite 20d > 0), Vorjahres-Isotonie)

**static_equal**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 28887 | 0.4611 | 0.5182 | -0.00321 | -0.00609 | -0.06041 | -0.00321 | -0.0571 | overconfident |
| 55–60 % | 617 | 0.4311 | 0.5643 | -0.00176 | -0.01274 | -0.07171 | -0.00176 | -0.1332 | overconfident |
| 60–65 % | 104 | 0.5096 | 0.6038 | -0.00233 | 0.00145 | -0.04962 | -0.00233 | -0.0942 | low_n |
| 65–70 % | 0 | None | None | None | None | None | None | None | empty |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_regime_weights**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 21344 | 0.4814 | 0.5203 | 0.00011 | -0.00295 | -0.05852 | 0.00011 | -0.0388 | calibrated |
| 55–60 % | 6397 | 0.4865 | 0.5635 | 0.00842 | -0.00277 | -0.06559 | 0.00842 | -0.077 | overconfident |
| 60–65 % | 0 | None | None | None | None | None | None | None | empty |
| 65–70 % | 61 | 0.5574 | 0.6818 | 0.04062 | 0.01704 | -0.08171 | 0.04062 | -0.1244 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |
**meta_stacking**
| Bucket | N | Trefferquote | Prognose | Ø Rendite | Median | Ø Drawdown | EV netto | Fehler | Flag |
|---|---|---|---|---|---|---|---|---|---|
| 50–55 % | 13864 | 0.4762 | 0.5173 | -0.00069 | -0.00403 | -0.07675 | -0.00069 | -0.0411 | calibrated |
| 55–60 % | 2138 | 0.5253 | 0.5538 | 0.01187 | 0.00494 | -0.07721 | 0.01187 | -0.0285 | calibrated |
| 60–65 % | 208 | 0.4856 | 0.6236 | -0.00541 | -0.00656 | -0.11328 | -0.00541 | -0.138 | overconfident |
| 65–70 % | 52 | 0.5 | 0.6875 | 0.02206 | 0.0025 | -0.10556 | 0.02206 | -0.1875 | low_n |
| 70–75 % | 0 | None | None | None | None | None | None | None | empty |
| 75–80 % | 0 | None | None | None | None | None | None | None | empty |
| 80–85 % | 0 | None | None | None | None | None | None | None | empty |
| 85–90 % | 0 | None | None | None | None | None | None | None | empty |
| >= 90 % | 0 | None | None | None | None | None | None | None | empty |

## Modell-Intelligenz

- momentum_12_1: {'oos_ic': 0.0047, 'recent_ic': -0.1045, 'prior_ic': 0.0652, 'trend': 'deteriorating', 'trend_t': -2.44, 'calibration_slope_recent': -0.0365, 'meta_weight': 0.4734, 'contribution': 0.0003}
- enet_xs20_v1: {'oos_ic': 0.0219, 'recent_ic': -0.0958, 'prior_ic': 0.0976, 'trend': 'deteriorating', 'trend_t': -2.76, 'calibration_slope_recent': -0.02426, 'meta_weight': 0.2247, 'contribution': 0.00038}
- hgb_xs20_v1: {'oos_ic': 0.0196, 'recent_ic': -0.0854, 'prior_ic': -0.0104, 'trend': 'stable', 'trend_t': -1.03, 'calibration_slope_recent': -0.02727, 'meta_weight': 0.1333, 'contribution': 0.00075}
- hgb_asym20_v1: {'oos_ic': 0.0041, 'recent_ic': -0.0008, 'prior_ic': -0.0252, 'trend': 'stable', 'trend_t': 0.54, 'calibration_slope_recent': -0.00563, 'meta_weight': 0.0, 'contribution': 0.0002}
- hgb_xs20_momentum_v1: {'oos_ic': 0.0121, 'recent_ic': 0.0719, 'prior_ic': -0.0216, 'trend': 'improving', 'trend_t': 3.55, 'calibration_slope_recent': 0.02192, 'meta_weight': 0.0, 'contribution': 0.00162}
- hgb_xs20_risk_regime_v1: {'oos_ic': 0.0169, 'recent_ic': -0.0584, 'prior_ic': -0.0045, 'trend': 'stable', 'trend_t': -1.0, 'calibration_slope_recent': -0.02305, 'meta_weight': 0.1686, 'contribution': 0.00032}

## High-Confidence-Regel (I)

- {'enabled': False, 'n_rules_tested': 6, 'best': {'prob': 0.55, 'agreement_sd': 9.0, 'lcb': -0.024689050721459667, 'n': 331, 'mean': -0.016905818853467598}, 'disabled_reason': 'keine Regel mit positiver unterer Schranke der Netto-Expectancy (Kalibrierjahre)'}
