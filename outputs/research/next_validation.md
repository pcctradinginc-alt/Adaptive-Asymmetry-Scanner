# Gesamtvalidierung Intelligenz-Komponenten – 2026-11-03T16:04:12+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: – · Gate nicht erfüllt: ['keine Komponente mit KEEP – G ist identisch mit A']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0153 | 0.216 | 0.25 | 0.088 | -0.1739 | -0.04702 | 0.4881 | 1.056 | 0.00166 | 0.064 | -0.05778 | 0.2301 | 0.25024 | 0.01576 | 0.69365 | 0.4602 | 10986 | 0.3846 (52) | 0.00869 | -0.00267 | 1.0 | 0.00285 |
| B | 0.0099 | 0.154 | 0.196 | 0.065 | -0.153 | -0.04116 | 0.4836 | 1.045 | 0.00143 | 0.06926 | -0.06209 | 0.2345 | 0.25028 | 0.01203 | 0.69558 | 0.4479 | 10986 | 0.5714 (126) | 0.00966 | -0.00318 | 1.0 | 0.00739 |
| C | 0.0213 | 0.297 | 0.395 | 0.17 | -0.1253 | -0.03676 | 0.4903 | 1.074 | 0.00232 | 0.06829 | -0.06113 | 0.2366 | 0.25031 | 0.01414 | 0.69478 | 0.4533 | 10986 | 0.4483 (1664) | 0.01058 | -0.00376 | 1.0 | 0.00519 |
| D | 0.0213 | 0.297 | 0.395 | 0.17 | -0.1253 | -0.03676 | 0.4903 | 1.074 | 0.00232 | 0.06829 | -0.06113 | 0.2366 | 0.25031 | 0.01414 | 0.69478 | 0.4533 | 10986 | 0.4483 (1664) | 0.01058 | -0.00376 | 1.0 | 0.00543 |
| E | 0.0213 | 0.297 | 0.395 | 0.17 | -0.1253 | -0.03676 | 0.4903 | 1.074 | 0.00232 | 0.06829 | -0.06113 | 0.2366 | 0.25031 | 0.01414 | 0.69478 | 0.4533 | 10986 | 0.4483 (1664) | 0.01058 | -0.00376 | 1.0 | 0.00547 |
| F | 0.0058 | 0.123 | 0.131 | 0.046 | -0.1257 | -0.03622 | 0.4884 | 1.018 | 0.00051 | 0.05827 | -0.05463 | 0.2187 | 0.2503 | 0.01228 | 0.69507 | 0.7211 | 9995 | 0.4816 (733) | 0.00369 | -0.00182 | 1.0 | -0.0033 |
| A_blindspot | -0.0042 | -0.05 | -0.044 | -0.034 | -0.1234 | -0.03815 | 0.4793 | 0.944 | -0.00134 | 0.0475 | -0.04632 | 0.2252 | 0.25283 | 0.0385 | 0.69977 | 0.6245 | 3002 | 0.4726 (948) | 0.00387 | -0.00394 | 0.735 | -0.00106 |
| A_abstention | 0.0511 | 0.64 | 0.618 | 0.567 | -0.0901 | -0.04044 | 0.5238 | 1.272 | 0.00777 | 0.06945 | -0.06008 | 0.2449 | 0.24909 | 0.00841 | 0.69186 | 0.5037 | 4370 | 0.462 (303) | 0.02188 | -0.02315 | 0.4 | 0.02549 |
| G | 0.0153 | 0.216 | 0.25 | 0.088 | -0.1739 | -0.04702 | 0.4881 | 1.056 | 0.00166 | 0.064 | -0.05778 | 0.2301 | 0.25024 | 0.01576 | 0.69365 | 0.4602 | 10986 | 0.3846 (52) | 0.00869 | -0.00267 | 1.0 | 0.00285 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ 0.000445 CI [-0.00444, 0.005591] · ohne Top-5 %-Monate -0.001942
- D: Monats-Δ 0.000445 CI [-0.00444, 0.005591] · ohne Top-5 %-Monate -0.001942
- E: Monats-Δ 0.000445 CI [-0.00444, 0.005591] · ohne Top-5 %-Monate -0.001942
- F: Monats-Δ -0.000938 CI [-0.00482, 0.002692] · ohne Top-5 %-Monate -0.002588
- A_blindspot: Monats-Δ -0.001806 CI [-0.007527, 0.003602] · ohne Top-5 %-Monate -0.00372
- A_abstention: Monats-Δ 0.002856 CI [-0.001327, 0.007681] · ohne Top-5 %-Monate 0.000253

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': False, 'decision_intelligence': 'KEEP'}

Abstinenz-Regel – historisch (Jahre [2019, 2020], Status CONTAMINATED, zählt nicht als Bestätigung): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.01658, 'inactive_expectancy': 0.00311, 'diff_t': 1.68, 'in_sample_rule_holds': True, 'confirmed': False, 'status': 'CONTAMINATED', 'years': [2019, 2020]}
Abstinenz-Regel – VORWÄRTS (bindend): {'forward_from': '2026-09-29', 'active_cohorts': 0, 'inactive_cohorts': 2, 'pending_cohorts': 6, 'active_expectancy': None, 'inactive_expectancy': -0.16058, 'diff_t': None, 'confirmed': False, 'status': 'ACCUMULATING'}

Ablation (Δ Expectancy G − G ohne Komponente): –

Gate: {'pass': False, 'criteria': {}, 'failed': ['keine Komponente mit KEEP – G ist identisch mit A']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00268, 'hit_rate': 0.516, 'n_cohorts': 43}, 'B': {'expectancy': 0.00268, 'hit_rate': 0.5096, 'n_cohorts': 43}, 'C': {'expectancy': 0.00744, 'hit_rate': 0.5343, 'n_cohorts': 43}, 'D': {'expectancy': 0.00744, 'hit_rate': 0.5343, 'n_cohorts': 43}, 'E': {'expectancy': 0.00744, 'hit_rate': 0.5343, 'n_cohorts': 43}, 'F': {'expectancy': 1e-05, 'hit_rate': 0.5105, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.0005, 'hit_rate': 0.5188, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.00268, 'hit_rate': 0.516, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.0094, 'hit_rate': 0.4183, 'n_cohorts': 13}, 'B': {'expectancy': -0.00496, 'hit_rate': 0.4663, 'n_cohorts': 13}, 'C': {'expectancy': -0.00869, 'hit_rate': 0.4519, 'n_cohorts': 13}, 'D': {'expectancy': -0.00869, 'hit_rate': 0.4519, 'n_cohorts': 13}, 'E': {'expectancy': -0.00869, 'hit_rate': 0.4519, 'n_cohorts': 13}, 'F': {'expectancy': -0.00717, 'hit_rate': 0.4537, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00776, 'hit_rate': 0.4393, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.0094, 'hit_rate': 0.4183, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.06978, 'hit_rate': 0.7398, 'n_cohorts': 8}, 'B': {'expectancy': 0.06577, 'hit_rate': 0.7092, 'n_cohorts': 8}, 'C': {'expectancy': 0.05877, 'hit_rate': 0.7041, 'n_cohorts': 8}, 'D': {'expectancy': 0.05877, 'hit_rate': 0.7041, 'n_cohorts': 8}, 'E': {'expectancy': 0.05877, 'hit_rate': 0.7041, 'n_cohorts': 8}, 'F': {'expectancy': 0.03854, 'hit_rate': 0.6317, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.0261, 'hit_rate': 0.6438, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.06978, 'hit_rate': 0.7398, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.231, 'rate_shock': 0.2073, 'crash': 0.3577, 'usd_shock': 0.2512, 'oil_shock': 0.1257, 'inflation_shock': 0.0548, 'liquidity_shock': 0.1847, 'momentum_reversal': 0.4629, 'volatility_flip': 0.7188}}

Decision Intelligence: {'top_decile': {'cagr': 0.0156, 'sharpe': 0.218, 'sortino': 0.255, 'max_dd': -0.1739, 'calmar': 0.09, 'profit_factor': 1.056, 'hit_rate': 0.4881, 'avg_winner': 0.064, 'avg_loser': -0.05778, 'payoff': 1.108, 'expectancy': 0.00166, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4602, 'exposure': 1.0, 'n_trades': 10986, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0749, 'avg_mae': -0.0639, 'es5_monthly': -0.04702, 'sector_hhi': 1.0}, 'diversified': {'cagr': 0.024, 'sharpe': 0.246, 'sortino': 0.269, 'max_dd': -0.1329, 'calmar': 0.181, 'profit_factor': 1.099, 'hit_rate': 0.4844, 'avg_winner': 0.0725, 'avg_loser': -0.06199, 'payoff': 1.17, 'expectancy': 0.00316, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.5169, 'exposure': 1.0, 'n_trades': 2816, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0816, 'avg_mae': -0.0688, 'es5_monthly': -0.08691, 'sector_hhi': 1.0}, 'verdict': 'KEEP'}

Anteil fragiler Positionen im Top-Dezil: 0.8978

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=277 (Segment 1086), typischer Fehler -0.1868, Lift 2.55, Eigenschaften {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_002: n=86 (Segment 414), typischer Fehler -0.16, Lift 2.08, Eigenschaften {'volatility': 'high_vol', 'near_high': 'near_52w_high'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_003: n=411 (Segment 2040), typischer Fehler -0.1802, Lift 2.01, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_004: n=546 (Segment 3076), typischer Fehler -0.1692, Lift 1.77, Eigenschaften {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=199 (Segment 1165), typischer Fehler -0.1631, Lift 1.71, Eigenschaften {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_006: n=140 (Segment 821), typischer Fehler -0.1807, Lift 1.7, Eigenschaften {'momentum': 'loser_12m', 'trend': 'downtrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=67 (Segment 408), typischer Fehler -0.1636, Lift 1.64, Eigenschaften {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=1253 (Segment 7632), typischer Fehler -0.1655, Lift 1.64, Eigenschaften {'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=346 (Segment 2176), typischer Fehler -0.1672, Lift 1.59, Eigenschaften {'momentum': 'loser_12m', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=379 (Segment 2482), typischer Fehler -0.1644, Lift 1.53, Eigenschaften {'momentum': 'winner_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=561 (Segment 3731), typischer Fehler -0.1638, Lift 1.5, Eigenschaften {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen

