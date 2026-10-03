# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-03T13:23:23+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: – · Gate nicht erfüllt: ['keine Komponente mit KEEP – G ist identisch mit A']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0147 | 0.214 | 0.25 | 0.092 | -0.16 | -0.04326 | 0.4877 | 1.053 | 0.00154 | 0.06225 | -0.05627 | 0.2233 | 0.25022 | 0.01439 | 0.69359 | 0.4704 | 11059 | 0.4444 (468) | 0.00874 | -0.00275 | 1.0 | 0.00589 |
| B | 0.0226 | 0.306 | 0.407 | 0.181 | -0.1252 | -0.03509 | 0.488 | 1.077 | 0.00238 | 0.06777 | -0.05996 | 0.2347 | 0.25029 | 0.01164 | 0.69565 | 0.4495 | 11059 | 0.4831 (178) | 0.00947 | -0.00222 | 1.0 | 0.01393 |
| C | 0.022 | 0.303 | 0.365 | 0.179 | -0.1226 | -0.04093 | 0.493 | 1.078 | 0.00235 | 0.06582 | -0.05935 | 0.2329 | 0.25036 | 0.0167 | 0.6951 | 0.4446 | 11059 | 0.4402 (1456) | 0.01326 | -0.0034 | 1.0 | 0.01213 |
| D | 0.0252 | 0.349 | 0.378 | 0.257 | -0.0979 | -0.04123 | 0.4954 | 1.084 | 0.00244 | 0.06375 | -0.05776 | 0.2276 | 0.25032 | 0.01405 | 0.69422 | 0.4283 | 11059 | 0.4572 (2524) | 0.01173 | -0.00296 | 1.0 | 0.01278 |
| E | 0.0249 | 0.344 | 0.375 | 0.243 | -0.1026 | -0.04052 | 0.4963 | 1.083 | 0.00243 | 0.06376 | -0.058 | 0.2276 | 0.25027 | 0.01576 | 0.69369 | 0.4323 | 11059 | 0.4536 (2524) | 0.01171 | -0.00295 | 1.0 | 0.01266 |
| F | -0.007 | -0.073 | -0.081 | -0.038 | -0.1824 | -0.03805 | 0.4848 | 0.978 | -0.0006 | 0.05612 | -0.05396 | 0.2071 | 0.25034 | 0.01093 | 0.69554 | 0.726 | 10038 | 0.4965 (864) | 0.00794 | -0.00356 | 1.0 | -0.00035 |
| A_blindspot | 0.0049 | 0.113 | 0.114 | 0.043 | -0.115 | -0.03742 | 0.4929 | 1.027 | 0.00062 | 0.04833 | -0.04576 | 0.2259 | 0.25062 | 0.02282 | 0.69444 | 0.5616 | 5019 | 0.4764 (275) | 0.00195 | -0.00193 | 0.817 | -0.00039 |
| A_abstention | 0.047 | 0.593 | 0.577 | 0.463 | -0.1015 | -0.04412 | 0.5258 | 1.275 | 0.0077 | 0.06788 | -0.05903 | 0.2413 | 0.24977 | 0.00927 | 0.69308 | 0.5152 | 4397 | 0.472 (375) | 0.02104 | -0.01782 | 0.4 | 0.02481 |
| G | 0.0147 | 0.214 | 0.25 | 0.092 | -0.16 | -0.04326 | 0.4877 | 1.053 | 0.00154 | 0.06225 | -0.05627 | 0.2233 | 0.25022 | 0.01439 | 0.69359 | 0.4704 | 11059 | 0.4444 (468) | 0.00874 | -0.00275 | 1.0 | 0.00589 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ 0.000581 CI [-0.003956, 0.005331] · ohne Top-5 %-Monate -0.00141
- D: Monats-Δ 0.000837 CI [-0.004351, 0.006146] · ohne Top-5 %-Monate -0.001422
- E: Monats-Δ 0.00081 CI [-0.004358, 0.006235] · ohne Top-5 %-Monate -0.001503
- F: Monats-Δ -0.001913 CI [-0.005152, 0.001599] · ohne Top-5 %-Monate -0.003312
- A_blindspot: Monats-Δ -0.000958 CI [-0.006281, 0.004082] · ohne Top-5 %-Monate -0.002726
- A_abstention: Monats-Δ 0.002611 CI [-0.001792, 0.007391] · ohne Top-5 %-Monate 0.000172

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': False, 'decision_intelligence': 'REJECT'}

Abstinenz-Regel – historisch (Jahre [2019, 2020], Status CONTAMINATED, zählt nicht als Bestätigung): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.01573, 'inactive_expectancy': 0.00178, 'diff_t': 1.66, 'in_sample_rule_holds': True, 'confirmed': False, 'status': 'CONTAMINATED', 'years': [2019, 2020]}
Abstinenz-Regel – VORWÄRTS (bindend): {'forward_from': '2026-09-29', 'active_cohorts': 0, 'inactive_cohorts': 0, 'pending_cohorts': 2, 'active_expectancy': None, 'inactive_expectancy': None, 'diff_t': None, 'confirmed': False, 'status': 'ACCUMULATING'}

Ablation (Δ Expectancy G − G ohne Komponente): –

Gate: {'pass': False, 'criteria': {}, 'failed': ['keine Komponente mit KEEP – G ist identisch mit A']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00371, 'hit_rate': 0.5275, 'n_cohorts': 43}, 'B': {'expectancy': 0.00291, 'hit_rate': 0.5173, 'n_cohorts': 43}, 'C': {'expectancy': 0.00719, 'hit_rate': 0.5392, 'n_cohorts': 43}, 'D': {'expectancy': 0.00669, 'hit_rate': 0.548, 'n_cohorts': 43}, 'E': {'expectancy': 0.0071, 'hit_rate': 0.5524, 'n_cohorts': 43}, 'F': {'expectancy': -0.00142, 'hit_rate': 0.5019, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': -0.00033, 'hit_rate': 0.5154, 'n_cohorts': 9}, 'A_abstention': {'expectancy': 0.00371, 'hit_rate': 0.5275, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.00955, 'hit_rate': 0.4247, 'n_cohorts': 13}, 'B': {'expectancy': -0.00133, 'hit_rate': 0.4696, 'n_cohorts': 13}, 'C': {'expectancy': -0.00489, 'hit_rate': 0.4615, 'n_cohorts': 13}, 'D': {'expectancy': -0.0051, 'hit_rate': 0.4631, 'n_cohorts': 13}, 'E': {'expectancy': -0.00541, 'hit_rate': 0.4631, 'n_cohorts': 13}, 'F': {'expectancy': -0.00933, 'hit_rate': 0.4371, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00749, 'hit_rate': 0.4425, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.00955, 'hit_rate': 0.4247, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.06711, 'hit_rate': 0.7296, 'n_cohorts': 8}, 'B': {'expectancy': 0.06731, 'hit_rate': 0.7245, 'n_cohorts': 8}, 'C': {'expectancy': 0.06286, 'hit_rate': 0.7194, 'n_cohorts': 8}, 'D': {'expectancy': 0.05395, 'hit_rate': 0.6633, 'n_cohorts': 8}, 'E': {'expectancy': 0.0535, 'hit_rate': 0.6658, 'n_cohorts': 8}, 'F': {'expectancy': 0.05294, 'hit_rate': 0.7099, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.02635, 'hit_rate': 0.6774, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.06711, 'hit_rate': 0.7296, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.2454, 'rate_shock': 0.192, 'crash': 0.3677, 'usd_shock': 0.2317, 'oil_shock': 0.1442, 'inflation_shock': 0.0509, 'liquidity_shock': 0.192, 'momentum_reversal': 0.4501, 'volatility_flip': 0.725}}

Decision Intelligence: {'top_decile': {'cagr': 0.015, 'sharpe': 0.216, 'sortino': 0.255, 'max_dd': -0.16, 'calmar': 0.094, 'profit_factor': 1.053, 'hit_rate': 0.4877, 'avg_winner': 0.06225, 'avg_loser': -0.05627, 'payoff': 1.106, 'expectancy': 0.00154, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4704, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0725, 'avg_mae': -0.0623, 'es5_monthly': -0.04326, 'sector_hhi': 0.1802}, 'diversified': {'cagr': 0.0029, 'sharpe': 0.075, 'sortino': 0.082, 'max_dd': -0.1815, 'calmar': 0.016, 'profit_factor': 1.019, 'hit_rate': 0.4847, 'avg_winner': 0.06128, 'avg_loser': -0.05655, 'payoff': 1.084, 'expectancy': 0.00056, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4734, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0718, 'avg_mae': -0.0625, 'es5_monthly': -0.04257, 'sector_hhi': 0.1585}, 'verdict': 'REJECT'}

Anteil fragiler Positionen im Top-Dezil: 0.9022

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=95 (Segment 331), typischer Fehler -0.1609, Lift 2.87, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_002: n=96 (Segment 340), typischer Fehler -0.1654, Lift 2.82, Eigenschaften {'sector': 'Energy', 'beta': 'high_beta'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_003: n=118 (Segment 467), typischer Fehler -0.1618, Lift 2.53, Eigenschaften {'sector': 'Energy', 'volatility': 'high_vol'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_004: n=224 (Segment 912), typischer Fehler -0.1845, Lift 2.46, Eigenschaften {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=38 (Segment 164), typischer Fehler -0.1688, Lift 2.32, Eigenschaften {'sector': 'Energy', 'liquidity': 'less_liquid'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_006: n=92 (Segment 411), typischer Fehler -0.1526, Lift 2.24, Eigenschaften {'sector': 'Energy', 'vix': 'vix_ge_20'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_007: n=353 (Segment 1800), typischer Fehler -0.1782, Lift 1.96, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=145 (Segment 783), typischer Fehler -0.1541, Lift 1.85, Eigenschaften {'sector': 'Energy'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_009: n=504 (Segment 2874), typischer Fehler -0.1696, Lift 1.75, Eigenschaften {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=245 (Segment 1408), typischer Fehler -0.1708, Lift 1.74, Eigenschaften {'sector': 'Consumer Cyclical', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=1185 (Segment 6968), typischer Fehler -0.1652, Lift 1.7, Eigenschaften {'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=130 (Segment 784), typischer Fehler -0.1861, Lift 1.66, Eigenschaften {'momentum': 'loser_12m', 'trend': 'downtrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=191 (Segment 1156), typischer Fehler -0.1724, Lift 1.65, Eigenschaften {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=362 (Segment 2201), typischer Fehler -0.1726, Lift 1.64, Eigenschaften {'momentum': 'loser_12m', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=109 (Segment 678), typischer Fehler -0.1821, Lift 1.61, Eigenschaften {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen

