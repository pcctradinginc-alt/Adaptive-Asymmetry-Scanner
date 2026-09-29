# Gesamtvalidierung Intelligenz-Komponenten – 2026-09-29T16:35:13+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: ['abstention'] · Gate nicht erfüllt: ['3_min_cohorts', '5_risk_adjusted', '6_no_regime_collapse']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0219 | 0.227 | 0.245 | 0.117 | -0.1875 | -0.08156 | 0.4742 | 1.068 | 0.00267 | 0.08871 | -0.07492 | 0.2657 | 0.24954 | 0.00309 | 0.69223 | 0.3456 | 11587 | 0.4861 (288) | 0.01223 | -0.00303 | 1.0 | 0.01883 |
| B | 0.0154 | 0.191 | 0.193 | 0.089 | -0.1723 | -0.06514 | 0.4723 | 1.056 | 0.00222 | 0.08848 | -0.075 | 0.2618 | 0.2497 | 0.01007 | 0.69255 | 0.3524 | 11587 | 0.45 (980) | 0.01119 | -0.00528 | 1.0 | 0.01169 |
| C | 0.0186 | 0.208 | 0.2 | 0.11 | -0.1696 | -0.07766 | 0.4741 | 1.072 | 0.0028 | 0.0883 | -0.07426 | 0.261 | 0.25025 | 0.01589 | 0.69536 | 0.3677 | 11587 | 0.4494 (988) | 0.01189 | -0.00736 | 1.0 | 0.01213 |
| D | 0.0238 | 0.255 | 0.25 | 0.151 | -0.1579 | -0.08014 | 0.4764 | 1.082 | 0.00308 | 0.08541 | -0.07183 | 0.2553 | 0.25013 | 0.01981 | 0.69342 | 0.3642 | 11587 | 0.4093 (728) | 0.01161 | -0.00675 | 1.0 | 0.01332 |
| E | 0.0236 | 0.252 | 0.248 | 0.15 | -0.1572 | -0.08045 | 0.4752 | 1.081 | 0.00307 | 0.08564 | -0.0717 | 0.2557 | 0.25012 | 0.01999 | 0.69341 | 0.3664 | 11587 | 0.4126 (744) | 0.01157 | -0.00686 | 1.0 | 0.0132 |
| F | 0.0157 | 0.231 | 0.269 | 0.119 | -0.1317 | -0.04695 | 0.4799 | 1.047 | 0.00157 | 0.07309 | -0.06441 | 0.2421 | 0.2495 | 0.00518 | 0.69253 | 0.6461 | 10505 | 0.4334 (616) | 0.00579 | -0.00347 | 1.0 | 0.00748 |
| A_blindspot | 0.0153 | 0.249 | 0.277 | 0.151 | -0.1015 | -0.04323 | 0.4782 | 1.021 | 0.0006 | 0.06065 | -0.05444 | 0.238 | 0.24914 | 0.01289 | 0.69144 | 0.5386 | 4504 | 0.4777 (157) | 0.00527 | -0.0017 | 1.0 | 0.00496 |
| A_abstention | 0.0835 | 0.677 | 0.785 | 0.651 | -0.1283 | -0.05717 | 0.5092 | 1.296 | 0.01066 | 0.09159 | -0.0733 | 0.2741 | 0.24842 | 0.01102 | 0.68994 | 0.3806 | 4619 | 0.6134 (269) | 0.03598 | -0.03035 | 0.4 | 0.03389 |
| G | 0.0835 | 0.677 | 0.785 | 0.651 | -0.1283 | -0.05717 | 0.5092 | 1.296 | 0.01066 | 0.09159 | -0.0733 | 0.2741 | 0.24842 | 0.01102 | 0.68994 | 0.3806 | 4619 | 0.6134 (269) | 0.03598 | -0.03035 | 0.4 | 0.03389 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ -0.000344 CI [-0.006263, 0.006265] · ohne Top-5 %-Monate -0.003186
- D: Monats-Δ 3.3e-05 CI [-0.005994, 0.006571] · ohne Top-5 %-Monate -0.002584
- E: Monats-Δ 1.7e-05 CI [-0.006024, 0.006489] · ohne Top-5 %-Monate -0.00263
- F: Monats-Δ -0.000963 CI [-0.007878, 0.005856] · ohne Top-5 %-Monate -0.004238
- A_blindspot: Monats-Δ -0.001061 CI [-0.009873, 0.007738] · ohne Top-5 %-Monate -0.004933
- A_abstention: Monats-Δ 0.004846 CI [-0.002571, 0.013141] · ohne Top-5 %-Monate 0.00064

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': True, 'decision_intelligence': 'KEEP'}

Abstinenz-Bestätigung (ungesehene Jahre [2019, 2020]): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.0302, 'inactive_expectancy': 0.00239, 'diff_t': 3.4, 'confirmed': True, 'years': [2019, 2020]}

Ablation (Δ Expectancy G − G ohne Komponente): {'abstention': 0.00799}

Gate: {'pass': False, 'criteria': {'1_no_leakage': True, '2_reproducible': True, '3_min_cohorts': False, '4_calibration_not_worse': True, '5_risk_adjusted': False, '6_no_regime_collapse': False, '7_hc_hit_rate_higher': True, '8_drawdown': True, '9_not_outlier_driven': True, '10_complexity_pays': True}, 'failed': ['3_min_cohorts', '5_risk_adjusted', '6_no_regime_collapse']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00583, 'hit_rate': 0.5186, 'n_cohorts': 43}, 'B': {'expectancy': 0.01198, 'hit_rate': 0.5363, 'n_cohorts': 43}, 'C': {'expectancy': 0.01539, 'hit_rate': 0.5498, 'n_cohorts': 43}, 'D': {'expectancy': 0.0148, 'hit_rate': 0.5577, 'n_cohorts': 43}, 'E': {'expectancy': 0.01464, 'hit_rate': 0.5572, 'n_cohorts': 43}, 'F': {'expectancy': 0.00765, 'hit_rate': 0.5172, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.00684, 'hit_rate': 0.5445, 'n_cohorts': 43}, 'A_abstention': {'expectancy': 0.00583, 'hit_rate': 0.5186, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.00721, 'hit_rate': 0.4169, 'n_cohorts': 13}, 'B': {'expectancy': -0.00434, 'hit_rate': 0.4323, 'n_cohorts': 13}, 'C': {'expectancy': -0.00818, 'hit_rate': 0.4431, 'n_cohorts': 13}, 'D': {'expectancy': -0.00514, 'hit_rate': 0.4523, 'n_cohorts': 13}, 'E': {'expectancy': -0.00459, 'hit_rate': 0.4538, 'n_cohorts': 13}, 'F': {'expectancy': -0.00538, 'hit_rate': 0.4467, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00532, 'hit_rate': 0.3861, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.00721, 'hit_rate': 0.4169, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.08646, 'hit_rate': 0.7132, 'n_cohorts': 8}, 'B': {'expectancy': 0.06717, 'hit_rate': 0.6495, 'n_cohorts': 8}, 'C': {'expectancy': 0.07393, 'hit_rate': 0.6593, 'n_cohorts': 8}, 'D': {'expectancy': 0.07043, 'hit_rate': 0.6373, 'n_cohorts': 8}, 'E': {'expectancy': 0.07022, 'hit_rate': 0.6348, 'n_cohorts': 8}, 'F': {'expectancy': 0.04362, 'hit_rate': 0.6522, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.03105, 'hit_rate': 0.6026, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.08646, 'hit_rate': 0.7132, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.1293, 'rate_shock': 0.148, 'crash': 0.2937, 'usd_shock': 0.164, 'oil_shock': 0.1488, 'inflation_shock': 0.0496, 'liquidity_shock': 0.1657, 'momentum_reversal': 0.5028, 'volatility_flip': 0.8171}}

Decision Intelligence: {'top_decile': {'cagr': 0.0223, 'sharpe': 0.229, 'sortino': 0.249, 'max_dd': -0.1875, 'calmar': 0.119, 'profit_factor': 1.068, 'hit_rate': 0.4742, 'avg_winner': 0.08871, 'avg_loser': -0.07492, 'payoff': 1.184, 'expectancy': 0.00267, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.3456, 'exposure': 1.0, 'n_trades': 11587, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0983, 'avg_mae': -0.0812, 'es5_monthly': -0.08156, 'sector_hhi': 0.1793}, 'diversified': {'cagr': 0.0248, 'sharpe': 0.252, 'sortino': 0.288, 'max_dd': -0.1857, 'calmar': 0.134, 'profit_factor': 1.072, 'hit_rate': 0.4746, 'avg_winner': 0.08864, 'avg_loser': -0.07469, 'payoff': 1.187, 'expectancy': 0.00282, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.3511, 'exposure': 1.0, 'n_trades': 11587, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0977, 'avg_mae': -0.0806, 'es5_monthly': -0.07663, 'sector_hhi': 0.1518}, 'verdict': 'KEEP'}

Anteil fragiler Positionen im Top-Dezil: 0.9406

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=94 (Segment 465), typischer Fehler -0.2329, Lift 2.02, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_002: n=98 (Segment 562), typischer Fehler -0.2082, Lift 1.74, Eigenschaften {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_003: n=177 (Segment 1032), typischer Fehler -0.2191, Lift 1.71, Eigenschaften {'sector': 'Communication Services', 'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_004: n=92 (Segment 547), typischer Fehler -0.2041, Lift 1.68, Eigenschaften {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=768 (Segment 4584), typischer Fehler -0.2048, Lift 1.68, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_006: n=205 (Segment 1243), typischer Fehler -0.2015, Lift 1.65, Eigenschaften {'liquidity': 'liquid', 'momentum': 'mid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=618 (Segment 3815), typischer Fehler -0.1993, Lift 1.62, Eigenschaften {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=1135 (Segment 7072), typischer Fehler -0.1991, Lift 1.6, Eigenschaften {'volatility': 'high_vol', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=56 (Segment 352), typischer Fehler -0.2098, Lift 1.59, Eigenschaften {'sector': 'Communication Services', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=860 (Segment 5432), typischer Fehler -0.1975, Lift 1.58, Eigenschaften {'liquidity': 'liquid', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=44 (Segment 278), typischer Fehler -0.1765, Lift 1.58, Eigenschaften {'sector': 'Technology', 'beta': 'low_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=190 (Segment 1213), typischer Fehler -0.2017, Lift 1.57, Eigenschaften {'liquidity': 'liquid', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=470 (Segment 3006), typischer Fehler -0.195, Lift 1.56, Eigenschaften {'sector': 'Technology', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=1098 (Segment 7034), typischer Fehler -0.1975, Lift 1.56, Eigenschaften {'liquidity': 'liquid', 'trend': 'uptrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=63 (Segment 406), typischer Fehler -0.1984, Lift 1.55, Eigenschaften {'sector': 'Healthcare', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen

