# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-03T12:28:25+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: – · Gate nicht erfüllt: ['keine Komponente mit KEEP – G ist identisch mit A']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0072 | 0.126 | 0.152 | 0.042 | -0.1712 | -0.04366 | 0.4861 | 1.032 | 0.00094 | 0.06238 | -0.05718 | 0.2225 | 0.25028 | 0.01452 | 0.69372 | 0.4631 | 11059 | 0.4269 (520) | 0.00935 | -0.00341 | 1.0 | -0.00245 |
| B | 0.0269 | 0.362 | 0.445 | 0.207 | -0.1298 | -0.03532 | 0.4918 | 1.088 | 0.00268 | 0.06758 | -0.06013 | 0.2379 | 0.25026 | 0.01281 | 0.69554 | 0.4584 | 11059 | 0.5 (230) | 0.01112 | -0.00172 | 1.0 | 0.01102 |
| C | 0.0363 | 0.502 | 0.586 | 0.274 | -0.1323 | -0.0349 | 0.4994 | 1.117 | 0.00345 | 0.06624 | -0.05918 | 0.2358 | 0.2504 | 0.01512 | 0.69502 | 0.4523 | 11059 | 0.4561 (2322) | 0.01223 | -0.00192 | 1.0 | 0.00233 |
| D | 0.046 | 0.66 | 0.749 | 0.576 | -0.0799 | -0.033 | 0.5038 | 1.142 | 0.00401 | 0.0641 | -0.05698 | 0.2323 | 0.25043 | 0.01484 | 0.69458 | 0.4332 | 11059 | 0.4608 (2576) | 0.01149 | -0.00116 | 1.0 | 0.0011 |
| E | 0.0475 | 0.683 | 0.786 | 0.592 | -0.0802 | -0.03279 | 0.5048 | 1.147 | 0.00414 | 0.06412 | -0.05701 | 0.2334 | 0.25036 | 0.01472 | 0.69392 | 0.4399 | 11059 | 0.4623 (2576) | 0.01163 | -0.00105 | 1.0 | 0.00113 |
| F | 0.0052 | 0.11 | 0.12 | 0.03 | -0.1709 | -0.03923 | 0.4951 | 1.018 | 0.00049 | 0.05731 | -0.05521 | 0.2187 | 0.25033 | 0.01081 | 0.69532 | 0.7266 | 10035 | 0.4892 (1018) | 0.00687 | -0.00335 | 1.0 | -0.00414 |
| A_blindspot | 0.0097 | 0.2 | 0.223 | 0.091 | -0.1063 | -0.03128 | 0.4789 | 1.021 | 0.00048 | 0.04922 | -0.0443 | 0.2203 | 0.2517 | 0.03375 | 0.69657 | 0.5475 | 4454 | 0.4417 (1449) | 0.00352 | -0.00244 | 0.817 | -0.00573 |
| A_abstention | 0.048 | 0.61 | 0.627 | 0.541 | -0.0888 | -0.04264 | 0.5272 | 1.257 | 0.00727 | 0.06742 | -0.05979 | 0.2397 | 0.24917 | 0.00878 | 0.69174 | 0.5112 | 4397 | 0.486 (393) | 0.02126 | -0.01566 | 0.4 | 0.0148 |
| G | 0.0072 | 0.126 | 0.152 | 0.042 | -0.1712 | -0.04366 | 0.4861 | 1.032 | 0.00094 | 0.06238 | -0.05718 | 0.2225 | 0.25028 | 0.01452 | 0.69372 | 0.4631 | 11059 | 0.4269 (520) | 0.00935 | -0.00341 | 1.0 | -0.00245 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ 0.002326 CI [-0.002838, 0.00736] · ohne Top-5 %-Monate -3e-05
- D: Monats-Δ 0.003075 CI [-0.002431, 0.008682] · ohne Top-5 %-Monate 0.000652
- E: Monats-Δ 0.003196 CI [-0.002309, 0.008887] · ohne Top-5 %-Monate 0.000677
- F: Monats-Δ -0.000273 CI [-0.003774, 0.003307] · ohne Top-5 %-Monate -0.001731
- A_blindspot: Monats-Δ 4.6e-05 CI [-0.004952, 0.004948] · ohne Top-5 %-Monate -0.001788
- A_abstention: Monats-Δ 0.003297 CI [-0.000785, 0.007845] · ohne Top-5 %-Monate 0.000853

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': False, 'decision_intelligence': 'REJECT'}

Abstinenz-Regel – historisch (Jahre [2019, 2020], Status CONTAMINATED, zählt nicht als Bestätigung): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.01575, 'inactive_expectancy': 0.00143, 'diff_t': 1.67, 'in_sample_rule_holds': True, 'confirmed': False, 'status': 'CONTAMINATED', 'years': [2019, 2020]}
Abstinenz-Regel – VORWÄRTS (bindend): {'forward_from': '2026-09-29', 'active_cohorts': 0, 'inactive_cohorts': 0, 'pending_cohorts': 2, 'active_expectancy': None, 'inactive_expectancy': None, 'diff_t': None, 'confirmed': False, 'status': 'ACCUMULATING'}

Ablation (Δ Expectancy G − G ohne Komponente): –

Gate: {'pass': False, 'criteria': {}, 'failed': ['keine Komponente mit KEEP – G ist identisch mit A']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00323, 'hit_rate': 0.529, 'n_cohorts': 43}, 'B': {'expectancy': 0.00543, 'hit_rate': 0.5271, 'n_cohorts': 43}, 'C': {'expectancy': 0.00921, 'hit_rate': 0.549, 'n_cohorts': 43}, 'D': {'expectancy': 0.00926, 'hit_rate': 0.5573, 'n_cohorts': 43}, 'E': {'expectancy': 0.00866, 'hit_rate': 0.5563, 'n_cohorts': 43}, 'F': {'expectancy': 0.00101, 'hit_rate': 0.5184, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.0013, 'hit_rate': 0.5066, 'n_cohorts': 9}, 'A_abstention': {'expectancy': 0.00323, 'hit_rate': 0.529, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.01112, 'hit_rate': 0.4167, 'n_cohorts': 13}, 'B': {'expectancy': -0.00358, 'hit_rate': 0.4679, 'n_cohorts': 13}, 'C': {'expectancy': -0.00407, 'hit_rate': 0.4712, 'n_cohorts': 13}, 'D': {'expectancy': -0.00382, 'hit_rate': 0.4728, 'n_cohorts': 13}, 'E': {'expectancy': -0.00355, 'hit_rate': 0.4776, 'n_cohorts': 13}, 'F': {'expectancy': -0.01009, 'hit_rate': 0.4256, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00955, 'hit_rate': 0.4211, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.01112, 'hit_rate': 0.4167, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.06734, 'hit_rate': 0.7296, 'n_cohorts': 8}, 'B': {'expectancy': 0.06566, 'hit_rate': 0.7347, 'n_cohorts': 8}, 'C': {'expectancy': 0.05334, 'hit_rate': 0.6913, 'n_cohorts': 8}, 'D': {'expectancy': 0.04911, 'hit_rate': 0.6556, 'n_cohorts': 8}, 'E': {'expectancy': 0.04864, 'hit_rate': 0.6607, 'n_cohorts': 8}, 'F': {'expectancy': 0.05017, 'hit_rate': 0.7042, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.02729, 'hit_rate': 0.6936, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.06734, 'hit_rate': 0.7296, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.2463, 'rate_shock': 0.2138, 'crash': 0.3595, 'usd_shock': 0.2455, 'oil_shock': 0.16, 'inflation_shock': 0.0526, 'liquidity_shock': 0.1947, 'momentum_reversal': 0.4665, 'volatility_flip': 0.7334}}

Decision Intelligence: {'top_decile': {'cagr': 0.0074, 'sharpe': 0.127, 'sortino': 0.155, 'max_dd': -0.1712, 'calmar': 0.043, 'profit_factor': 1.032, 'hit_rate': 0.4861, 'avg_winner': 0.06238, 'avg_loser': -0.05718, 'payoff': 1.091, 'expectancy': 0.00094, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4631, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0731, 'avg_mae': -0.0637, 'es5_monthly': -0.04366, 'sector_hhi': 0.1827}, 'diversified': {'cagr': 0.0003, 'sharpe': 0.042, 'sortino': 0.049, 'max_dd': -0.1759, 'calmar': 0.002, 'profit_factor': 1.012, 'hit_rate': 0.4839, 'avg_winner': 0.06151, 'avg_loser': -0.05703, 'payoff': 1.079, 'expectancy': 0.00034, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4674, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0724, 'avg_mae': -0.0635, 'es5_monthly': -0.04002, 'sector_hhi': 0.1591}, 'verdict': 'REJECT'}

Anteil fragiler Positionen im Top-Dezil: 0.9072

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=91 (Segment 332), typischer Fehler -0.1588, Lift 2.74, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_002: n=94 (Segment 355), typischer Fehler -0.1638, Lift 2.65, Eigenschaften {'sector': 'Energy', 'beta': 'high_beta'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_003: n=248 (Segment 1004), typischer Fehler -0.1857, Lift 2.47, Eigenschaften {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_004: n=112 (Segment 467), typischer Fehler -0.161, Lift 2.4, Eigenschaften {'sector': 'Energy', 'volatility': 'high_vol'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_005: n=36 (Segment 167), typischer Fehler -0.1697, Lift 2.16, Eigenschaften {'sector': 'Energy', 'liquidity': 'less_liquid'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_006: n=387 (Segment 1916), typischer Fehler -0.1783, Lift 2.02, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=138 (Segment 780), typischer Fehler -0.1535, Lift 1.77, Eigenschaften {'sector': 'Energy'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_008: n=515 (Segment 2911), typischer Fehler -0.1692, Lift 1.77, Eigenschaften {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=206 (Segment 1180), typischer Fehler -0.1671, Lift 1.75, Eigenschaften {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=244 (Segment 1418), typischer Fehler -0.1729, Lift 1.72, Eigenschaften {'sector': 'Consumer Cyclical', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=1197 (Segment 7145), typischer Fehler -0.1674, Lift 1.68, Eigenschaften {'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=132 (Segment 795), typischer Fehler -0.1831, Lift 1.66, Eigenschaften {'momentum': 'loser_12m', 'trend': 'downtrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=51 (Segment 312), typischer Fehler -0.1525, Lift 1.63, Eigenschaften {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=206 (Segment 1263), typischer Fehler -0.1693, Lift 1.63, Eigenschaften {'sector': 'Technology', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=370 (Segment 2287), typischer Fehler -0.1712, Lift 1.62, Eigenschaften {'momentum': 'loser_12m', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen

