# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-02T21:33:02+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: – · Gate nicht erfüllt: ['keine Komponente mit KEEP – G ist identisch mit A']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0033 | 0.08 | 0.088 | 0.018 | -0.1869 | -0.04651 | 0.4851 | 1.02 | 0.00058 | 0.06165 | -0.05697 | 0.2208 | 0.25029 | 0.01514 | 0.69373 | 0.4582 | 11059 | 0.4505 (364) | 0.00892 | -0.00381 | 1.0 | 0.00186 |
| B | 0.0054 | 0.105 | 0.122 | 0.036 | -0.1495 | -0.0417 | 0.4806 | 1.031 | 0.00097 | 0.06643 | -0.0596 | 0.2281 | 0.25055 | 0.01526 | 0.69562 | 0.4487 | 11059 | 0.5635 (252) | 0.00994 | -0.00379 | 1.0 | 0.00619 |
| C | 0.0235 | 0.321 | 0.398 | 0.167 | -0.1404 | -0.03664 | 0.4915 | 1.08 | 0.00239 | 0.06585 | -0.05894 | 0.2299 | 0.25046 | 0.01426 | 0.69542 | 0.4358 | 11059 | 0.4521 (2590) | 0.01229 | -0.00343 | 1.0 | 0.00541 |
| D | 0.0251 | 0.348 | 0.399 | 0.24 | -0.1046 | -0.03759 | 0.4934 | 1.08 | 0.00232 | 0.06363 | -0.05741 | 0.2246 | 0.25045 | 0.0167 | 0.69437 | 0.4194 | 11059 | 0.4453 (2524) | 0.01137 | -0.00329 | 1.0 | 0.00231 |
| E | 0.03 | 0.41 | 0.499 | 0.291 | -0.1032 | -0.03604 | 0.497 | 1.095 | 0.00273 | 0.06365 | -0.05745 | 0.2257 | 0.25043 | 0.01645 | 0.69402 | 0.4248 | 11059 | 0.4533 (2634) | 0.01156 | -0.00301 | 1.0 | 0.0021 |
| F | 0.0109 | 0.193 | 0.225 | 0.072 | -0.152 | -0.03931 | 0.4932 | 1.029 | 0.00081 | 0.05761 | -0.05448 | 0.2154 | 0.25032 | 0.01057 | 0.69518 | 0.7228 | 10044 | 0.5026 (784) | 0.0066 | -0.00302 | 1.0 | -0.00478 |
| A_blindspot | 0.0094 | 0.182 | 0.193 | 0.079 | -0.1185 | -0.03717 | 0.4934 | 1.04 | 0.00091 | 0.04859 | -0.04553 | 0.2263 | 0.25091 | 0.02543 | 0.69499 | 0.5524 | 4994 | 0.4468 (1428) | 0.00307 | -0.00191 | 0.817 | -0.00325 |
| A_abstention | 0.0436 | 0.544 | 0.48 | 0.425 | -0.1025 | -0.04518 | 0.5272 | 1.253 | 0.00706 | 0.06639 | -0.05908 | 0.2393 | 0.24934 | 0.00885 | 0.69208 | 0.4985 | 4397 | 0.4679 (312) | 0.02257 | -0.01091 | 0.4 | 0.02482 |
| G | 0.0033 | 0.08 | 0.088 | 0.018 | -0.1869 | -0.04651 | 0.4851 | 1.02 | 0.00058 | 0.06165 | -0.05697 | 0.2208 | 0.25029 | 0.01514 | 0.69373 | 0.4582 | 11059 | 0.4505 (364) | 0.00892 | -0.00381 | 1.0 | 0.00186 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ 0.001647 CI [-0.003612, 0.006593] · ohne Top-5 %-Monate -0.000704
- D: Monats-Δ 0.001756 CI [-0.003673, 0.007397] · ohne Top-5 %-Monate -0.000694
- E: Monats-Δ 0.002149 CI [-0.003366, 0.007797] · ohne Top-5 %-Monate -0.000339
- F: Monats-Δ 0.000525 CI [-0.003064, 0.004703] · ohne Top-5 %-Monate -0.001397
- A_blindspot: Monats-Δ 0.000363 CI [-0.004705, 0.005371] · ohne Top-5 %-Monate -0.001506
- A_abstention: Monats-Δ 0.003283 CI [-0.000941, 0.007972] · ohne Top-5 %-Monate 0.00088

Verdikte: {'counterfactual_filter': 'MODIFY', 'blind_spot_filter': 'MODIFY', 'abstention': False, 'decision_intelligence': 'MODIFY'}

Abstinenz-Regel – historisch (Jahre [2019, 2020], Status CONTAMINATED, zählt nicht als Bestätigung): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.01659, 'inactive_expectancy': 0.00357, 'diff_t': 1.54, 'in_sample_rule_holds': False, 'confirmed': False, 'status': 'CONTAMINATED', 'years': [2019, 2020]}
Abstinenz-Regel – VORWÄRTS (bindend): {'forward_from': '2026-09-29', 'active_cohorts': 0, 'inactive_cohorts': 0, 'pending_cohorts': 2, 'active_expectancy': None, 'inactive_expectancy': None, 'diff_t': None, 'confirmed': False, 'status': 'ACCUMULATING'}

Ablation (Δ Expectancy G − G ohne Komponente): –

Gate: {'pass': False, 'criteria': {}, 'failed': ['keine Komponente mit KEEP – G ist identisch mit A']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00235, 'hit_rate': 0.5197, 'n_cohorts': 43}, 'B': {'expectancy': 0.00507, 'hit_rate': 0.5275, 'n_cohorts': 43}, 'C': {'expectancy': 0.00775, 'hit_rate': 0.5475, 'n_cohorts': 43}, 'D': {'expectancy': 0.00775, 'hit_rate': 0.5602, 'n_cohorts': 43}, 'E': {'expectancy': 0.00853, 'hit_rate': 0.567, 'n_cohorts': 43}, 'F': {'expectancy': 0.00201, 'hit_rate': 0.5297, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': -0.0023, 'hit_rate': 0.5149, 'n_cohorts': 9}, 'A_abstention': {'expectancy': 0.00235, 'hit_rate': 0.5197, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.012, 'hit_rate': 0.4103, 'n_cohorts': 13}, 'B': {'expectancy': -0.00493, 'hit_rate': 0.4535, 'n_cohorts': 13}, 'C': {'expectancy': -0.00911, 'hit_rate': 0.4471, 'n_cohorts': 13}, 'D': {'expectancy': -0.0099, 'hit_rate': 0.4375, 'n_cohorts': 13}, 'E': {'expectancy': -0.00982, 'hit_rate': 0.4423, 'n_cohorts': 13}, 'F': {'expectancy': -0.00938, 'hit_rate': 0.4371, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00906, 'hit_rate': 0.4198, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.012, 'hit_rate': 0.4103, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.06905, 'hit_rate': 0.7577, 'n_cohorts': 8}, 'B': {'expectancy': 0.07047, 'hit_rate': 0.7321, 'n_cohorts': 8}, 'C': {'expectancy': 0.06677, 'hit_rate': 0.727, 'n_cohorts': 8}, 'D': {'expectancy': 0.05756, 'hit_rate': 0.6862, 'n_cohorts': 8}, 'E': {'expectancy': 0.05837, 'hit_rate': 0.6913, 'n_cohorts': 8}, 'F': {'expectancy': 0.05376, 'hit_rate': 0.7139, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.04105, 'hit_rate': 0.7303, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.06905, 'hit_rate': 0.7577, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.2662, 'rate_shock': 0.1859, 'crash': 0.3625, 'usd_shock': 0.2456, 'oil_shock': 0.1582, 'inflation_shock': 0.0599, 'liquidity_shock': 0.1909, 'momentum_reversal': 0.4706, 'volatility_flip': 0.7291}}

Decision Intelligence: {'top_decile': {'cagr': 0.0033, 'sharpe': 0.08, 'sortino': 0.09, 'max_dd': -0.1869, 'calmar': 0.018, 'profit_factor': 1.02, 'hit_rate': 0.4851, 'avg_winner': 0.06165, 'avg_loser': -0.05697, 'payoff': 1.082, 'expectancy': 0.00058, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4582, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0723, 'avg_mae': -0.063, 'es5_monthly': -0.04651, 'sector_hhi': 0.18}, 'diversified': {'cagr': -0.0012, 'sharpe': 0.026, 'sortino': 0.03, 'max_dd': -0.1849, 'calmar': -0.007, 'profit_factor': 1.007, 'hit_rate': 0.4846, 'avg_winner': 0.06115, 'avg_loser': -0.05707, 'payoff': 1.071, 'expectancy': 0.00022, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4613, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0719, 'avg_mae': -0.0629, 'es5_monthly': -0.04317, 'sector_hhi': 0.1578}, 'verdict': 'MODIFY'}

Anteil fragiler Positionen im Top-Dezil: 0.9025

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=97 (Segment 338), typischer Fehler -0.1555, Lift 2.87, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_002: n=103 (Segment 362), typischer Fehler -0.1595, Lift 2.85, Eigenschaften {'sector': 'Energy', 'beta': 'high_beta'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_003: n=120 (Segment 463), typischer Fehler -0.1597, Lift 2.59, Eigenschaften {'sector': 'Energy', 'volatility': 'high_vol'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_004: n=230 (Segment 921), typischer Fehler -0.1863, Lift 2.5, Eigenschaften {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=97 (Segment 412), typischer Fehler -0.1488, Lift 2.35, Eigenschaften {'sector': 'Energy', 'vix': 'vix_ge_20'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_006: n=370 (Segment 1851), typischer Fehler -0.1772, Lift 2.0, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=153 (Segment 782), typischer Fehler -0.1522, Lift 1.96, Eigenschaften {'sector': 'Energy'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_008: n=529 (Segment 2906), typischer Fehler -0.1692, Lift 1.82, Eigenschaften {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=211 (Segment 1166), typischer Fehler -0.1689, Lift 1.81, Eigenschaften {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=1197 (Segment 7009), typischer Fehler -0.1659, Lift 1.71, Eigenschaften {'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=234 (Segment 1387), typischer Fehler -0.1747, Lift 1.69, Eigenschaften {'sector': 'Consumer Cyclical', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=114 (Segment 682), typischer Fehler -0.1838, Lift 1.67, Eigenschaften {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=125 (Segment 749), typischer Fehler -0.1824, Lift 1.67, Eigenschaften {'momentum': 'loser_12m', 'trend': 'downtrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=43 (Segment 259), typischer Fehler -0.1572, Lift 1.66, Eigenschaften {'sector': 'Basic Materials', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=209 (Segment 1268), typischer Fehler -0.1697, Lift 1.65, Eigenschaften {'sector': 'Technology', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen

