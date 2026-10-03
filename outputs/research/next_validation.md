# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-03T07:35:37+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: – · Gate nicht erfüllt: ['keine Komponente mit KEEP – G ist identisch mit A']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0091 | 0.144 | 0.175 | 0.051 | -0.1788 | -0.04751 | 0.4829 | 1.036 | 0.00105 | 0.06266 | -0.05649 | 0.2216 | 0.25032 | 0.0153 | 0.6938 | 0.4697 | 11059 | 0.4424 (660) | 0.00846 | -0.00353 | 1.0 | -0.00198 |
| B | 0.0225 | 0.297 | 0.372 | 0.148 | -0.1519 | -0.03833 | 0.4862 | 1.081 | 0.00249 | 0.06813 | -0.05962 | 0.2359 | 0.25032 | 0.01342 | 0.69577 | 0.4564 | 11059 | 0.4772 (438) | 0.01254 | -0.00152 | 1.0 | 0.01589 |
| C | 0.0292 | 0.388 | 0.494 | 0.241 | -0.1214 | -0.03592 | 0.4928 | 1.097 | 0.00288 | 0.06616 | -0.0586 | 0.2324 | 0.25053 | 0.01739 | 0.6955 | 0.448 | 11059 | 0.436 (2236) | 0.01117 | -0.00311 | 1.0 | 0.00662 |
| D | 0.0313 | 0.421 | 0.52 | 0.324 | -0.0966 | -0.03836 | 0.4965 | 1.096 | 0.00278 | 0.06366 | -0.05726 | 0.227 | 0.25043 | 0.01702 | 0.69407 | 0.4314 | 11059 | 0.4569 (2760) | 0.01116 | -0.00302 | 1.0 | 0.00662 |
| E | 0.0316 | 0.426 | 0.508 | 0.327 | -0.0966 | -0.03761 | 0.4977 | 1.098 | 0.00281 | 0.0635 | -0.05732 | 0.2271 | 0.25042 | 0.01697 | 0.69404 | 0.4361 | 11059 | 0.4565 (2782) | 0.01081 | -0.00286 | 1.0 | 0.00656 |
| F | -0.004 | -0.025 | -0.027 | -0.02 | -0.199 | -0.04255 | 0.4868 | 0.99 | -0.00028 | 0.0568 | -0.05444 | 0.2102 | 0.2504 | 0.01121 | 0.69559 | 0.7356 | 10032 | 0.5067 (1038) | 0.00833 | -0.00334 | 1.0 | -0.00685 |
| A_blindspot | 0.0043 | 0.1 | 0.105 | 0.037 | -0.1154 | -0.03785 | 0.4905 | 1.023 | 0.00055 | 0.04973 | -0.04679 | 0.2219 | 0.25061 | 0.0229 | 0.6944 | 0.564 | 5350 | 0.4389 (1392) | 0.00303 | -0.00161 | 0.817 | -0.00377 |
| A_abstention | 0.0484 | 0.591 | 0.574 | 0.5 | -0.0968 | -0.04388 | 0.5256 | 1.273 | 0.00765 | 0.0679 | -0.05911 | 0.2415 | 0.24971 | 0.00879 | 0.69282 | 0.5027 | 4397 | 0.4545 (132) | 0.02211 | -0.01823 | 0.4 | 0.01387 |
| G | 0.0091 | 0.144 | 0.175 | 0.051 | -0.1788 | -0.04751 | 0.4829 | 1.036 | 0.00105 | 0.06266 | -0.05649 | 0.2216 | 0.25032 | 0.0153 | 0.6938 | 0.4697 | 11059 | 0.4424 (660) | 0.00846 | -0.00353 | 1.0 | -0.00198 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ 0.001591 CI [-0.003176, 0.006263] · ohne Top-5 %-Monate -0.000555
- D: Monats-Δ 0.001754 CI [-0.003528, 0.007033] · ohne Top-5 %-Monate -0.000618
- E: Monats-Δ 0.001772 CI [-0.003649, 0.007159] · ohne Top-5 %-Monate -0.000627
- F: Monats-Δ -0.001225 CI [-0.004908, 0.002662] · ohne Top-5 %-Monate -0.002838
- A_blindspot: Monats-Δ -0.000581 CI [-0.005617, 0.004256] · ohne Top-5 %-Monate -0.002281
- A_abstention: Monats-Δ 0.003162 CI [-0.001206, 0.008139] · ohne Top-5 %-Monate 0.000612

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': False, 'decision_intelligence': 'MODIFY'}

Abstinenz-Regel – historisch (Jahre [2019, 2020], Status CONTAMINATED, zählt nicht als Bestätigung): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.01612, 'inactive_expectancy': 0.00224, 'diff_t': 1.63, 'in_sample_rule_holds': False, 'confirmed': False, 'status': 'CONTAMINATED', 'years': [2019, 2020]}
Abstinenz-Regel – VORWÄRTS (bindend): {'forward_from': '2026-09-29', 'active_cohorts': 0, 'inactive_cohorts': 0, 'pending_cohorts': 2, 'active_expectancy': None, 'inactive_expectancy': None, 'diff_t': None, 'confirmed': False, 'status': 'ACCUMULATING'}

Ablation (Δ Expectancy G − G ohne Komponente): –

Gate: {'pass': False, 'criteria': {}, 'failed': ['keine Komponente mit KEEP – G ist identisch mit A']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00394, 'hit_rate': 0.528, 'n_cohorts': 43}, 'B': {'expectancy': 0.00439, 'hit_rate': 0.5222, 'n_cohorts': 43}, 'C': {'expectancy': 0.00842, 'hit_rate': 0.5461, 'n_cohorts': 43}, 'D': {'expectancy': 0.0087, 'hit_rate': 0.5588, 'n_cohorts': 43}, 'E': {'expectancy': 0.00808, 'hit_rate': 0.5578, 'n_cohorts': 43}, 'F': {'expectancy': -0.00082, 'hit_rate': 0.51, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.00176, 'hit_rate': 0.5217, 'n_cohorts': 9}, 'A_abstention': {'expectancy': 0.00394, 'hit_rate': 0.528, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.01069, 'hit_rate': 0.4151, 'n_cohorts': 13}, 'B': {'expectancy': -0.0035, 'hit_rate': 0.4663, 'n_cohorts': 13}, 'C': {'expectancy': -0.00462, 'hit_rate': 0.4631, 'n_cohorts': 13}, 'D': {'expectancy': -0.0054, 'hit_rate': 0.4615, 'n_cohorts': 13}, 'E': {'expectancy': -0.00508, 'hit_rate': 0.4647, 'n_cohorts': 13}, 'F': {'expectancy': -0.0102, 'hit_rate': 0.4213, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00738, 'hit_rate': 0.4298, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.01069, 'hit_rate': 0.4151, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.07139, 'hit_rate': 0.7372, 'n_cohorts': 8}, 'B': {'expectancy': 0.06896, 'hit_rate': 0.7296, 'n_cohorts': 8}, 'C': {'expectancy': 0.0625, 'hit_rate': 0.7321, 'n_cohorts': 8}, 'D': {'expectancy': 0.05837, 'hit_rate': 0.699, 'n_cohorts': 8}, 'E': {'expectancy': 0.05616, 'hit_rate': 0.6913, 'n_cohorts': 8}, 'F': {'expectancy': 0.05332, 'hit_rate': 0.693, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.02707, 'hit_rate': 0.6726, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.07139, 'hit_rate': 0.7372, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.2315, 'rate_shock': 0.2011, 'crash': 0.365, 'usd_shock': 0.232, 'oil_shock': 0.1599, 'inflation_shock': 0.054, 'liquidity_shock': 0.1976, 'momentum_reversal': 0.4642, 'volatility_flip': 0.7379}}

Decision Intelligence: {'top_decile': {'cagr': 0.0093, 'sharpe': 0.145, 'sortino': 0.178, 'max_dd': -0.1788, 'calmar': 0.052, 'profit_factor': 1.036, 'hit_rate': 0.4829, 'avg_winner': 0.06266, 'avg_loser': -0.05649, 'payoff': 1.109, 'expectancy': 0.00105, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4697, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0726, 'avg_mae': -0.063, 'es5_monthly': -0.04751, 'sector_hhi': 0.183}, 'diversified': {'cagr': 0.0026, 'sharpe': 0.072, 'sortino': 0.084, 'max_dd': -0.1744, 'calmar': 0.015, 'profit_factor': 1.016, 'hit_rate': 0.4816, 'avg_winner': 0.06163, 'avg_loser': -0.05634, 'payoff': 1.094, 'expectancy': 0.00048, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4742, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0719, 'avg_mae': -0.0628, 'es5_monthly': -0.04277, 'sector_hhi': 0.1588}, 'verdict': 'MODIFY'}

Anteil fragiler Positionen im Top-Dezil: 0.9114

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=94 (Segment 329), typischer Fehler -0.1603, Lift 2.86, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_002: n=96 (Segment 353), typischer Fehler -0.1638, Lift 2.72, Eigenschaften {'sector': 'Energy', 'beta': 'high_beta'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_003: n=117 (Segment 457), typischer Fehler -0.1632, Lift 2.56, Eigenschaften {'sector': 'Energy', 'volatility': 'high_vol'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_004: n=231 (Segment 903), typischer Fehler -0.1884, Lift 2.56, Eigenschaften {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=38 (Segment 165), typischer Fehler -0.1694, Lift 2.3, Eigenschaften {'sector': 'Energy', 'liquidity': 'less_liquid'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_006: n=66 (Segment 314), typischer Fehler -0.1607, Lift 2.1, Eigenschaften {'volatility': 'high_vol', 'near_high': 'near_52w_high'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=359 (Segment 1789), typischer Fehler -0.1809, Lift 2.01, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=139 (Segment 738), typischer Fehler -0.1563, Lift 1.88, Eigenschaften {'sector': 'Energy'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_009: n=508 (Segment 2868), typischer Fehler -0.171, Lift 1.77, Eigenschaften {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=247 (Segment 1430), typischer Fehler -0.1691, Lift 1.73, Eigenschaften {'sector': 'Consumer Cyclical', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=191 (Segment 1121), typischer Fehler -0.1684, Lift 1.7, Eigenschaften {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=119 (Segment 703), typischer Fehler -0.1806, Lift 1.69, Eigenschaften {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=1183 (Segment 6989), typischer Fehler -0.1668, Lift 1.69, Eigenschaften {'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=208 (Segment 1237), typischer Fehler -0.1718, Lift 1.68, Eigenschaften {'sector': 'Technology', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=353 (Segment 2180), typischer Fehler -0.1715, Lift 1.62, Eigenschaften {'momentum': 'loser_12m', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen

