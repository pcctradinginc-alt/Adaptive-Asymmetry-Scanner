# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-03T09:31:52+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: – · Gate nicht erfüllt: ['keine Komponente mit KEEP – G ist identisch mit A']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0074 | 0.126 | 0.144 | 0.041 | -0.1806 | -0.04911 | 0.4862 | 1.031 | 0.00093 | 0.06267 | -0.05749 | 0.2233 | 0.25029 | 0.01507 | 0.69373 | 0.4565 | 11059 | 0.4338 (468) | 0.00732 | -0.00373 | 1.0 | 0.00264 |
| B | 0.0123 | 0.182 | 0.233 | 0.089 | -0.1382 | -0.03971 | 0.4859 | 1.051 | 0.00161 | 0.06752 | -0.0607 | 0.2326 | 0.25035 | 0.01282 | 0.69578 | 0.4454 | 11059 | 0.4656 (1426) | 0.01169 | -0.00323 | 1.0 | 0.00742 |
| C | 0.0353 | 0.448 | 0.61 | 0.272 | -0.1299 | -0.03693 | 0.4963 | 1.114 | 0.00341 | 0.06728 | -0.05954 | 0.2361 | 0.2504 | 0.0139 | 0.695 | 0.4382 | 11059 | 0.4455 (2568) | 0.01595 | -0.00285 | 1.0 | 0.00559 |
| D | 0.0375 | 0.506 | 0.643 | 0.4 | -0.0938 | -0.03639 | 0.4999 | 1.118 | 0.00342 | 0.06485 | -0.05798 | 0.2316 | 0.25048 | 0.01606 | 0.69463 | 0.4224 | 11059 | 0.4533 (2738) | 0.01345 | -0.00261 | 1.0 | 0.0035 |
| E | 0.0413 | 0.554 | 0.722 | 0.447 | -0.0924 | -0.03492 | 0.5017 | 1.129 | 0.00371 | 0.0648 | -0.05778 | 0.2321 | 0.25045 | 0.01695 | 0.69409 | 0.4279 | 11059 | 0.454 (2738) | 0.01392 | -0.00223 | 1.0 | 0.00377 |
| F | -0.0011 | 0.019 | 0.021 | -0.006 | -0.1947 | -0.04214 | 0.4914 | 0.999 | -3e-05 | 0.05677 | -0.05493 | 0.216 | 0.25032 | 0.01132 | 0.695 | 0.7189 | 10044 | 0.5 (634) | 0.00524 | -0.004 | 1.0 | -0.00539 |
| A_blindspot | 0.0013 | 0.053 | 0.052 | 0.008 | -0.1571 | -0.04804 | 0.4914 | 1.014 | 0.00035 | 0.05052 | -0.04814 | 0.2268 | 0.25087 | 0.02416 | 0.69493 | 0.5535 | 5252 | 0.4426 (1437) | 0.00337 | -0.00221 | 0.817 | -0.00084 |
| A_abstention | 0.0542 | 0.669 | 0.675 | 0.659 | -0.0823 | -0.04254 | 0.5285 | 1.274 | 0.00771 | 0.06781 | -0.05966 | 0.2429 | 0.24951 | 0.00878 | 0.69269 | 0.4999 | 4397 | 0.4773 (375) | 0.02252 | -0.01599 | 0.4 | 0.02631 |
| G | 0.0074 | 0.126 | 0.144 | 0.041 | -0.1806 | -0.04911 | 0.4862 | 1.031 | 0.00093 | 0.06267 | -0.05749 | 0.2233 | 0.25029 | 0.01507 | 0.69373 | 0.4565 | 11059 | 0.4338 (468) | 0.00732 | -0.00373 | 1.0 | 0.00264 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ 0.002251 CI [-0.00334, 0.007859] · ohne Top-5 %-Monate -0.000504
- D: Monats-Δ 0.002395 CI [-0.003323, 0.008194] · ohne Top-5 %-Monate -0.000217
- E: Monats-Δ 0.002693 CI [-0.002933, 0.008497] · ohne Top-5 %-Monate 2e-06
- F: Monats-Δ -0.000821 CI [-0.00397, 0.00247] · ohne Top-5 %-Monate -0.002189
- A_blindspot: Monats-Δ -0.000626 CI [-0.005174, 0.003882] · ohne Top-5 %-Monate -0.002229
- A_abstention: Monats-Δ 0.003759 CI [-0.000775, 0.009081] · ohne Top-5 %-Monate 0.000997

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': False, 'decision_intelligence': 'REJECT'}

Abstinenz-Regel – historisch (Jahre [2019, 2020], Status CONTAMINATED, zählt nicht als Bestätigung): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.01634, 'inactive_expectancy': 0.00428, 'diff_t': 1.41, 'in_sample_rule_holds': False, 'confirmed': False, 'status': 'CONTAMINATED', 'years': [2019, 2020]}
Abstinenz-Regel – VORWÄRTS (bindend): {'forward_from': '2026-09-29', 'active_cohorts': 0, 'inactive_cohorts': 0, 'pending_cohorts': 2, 'active_expectancy': None, 'inactive_expectancy': None, 'diff_t': None, 'confirmed': False, 'status': 'ACCUMULATING'}

Ablation (Δ Expectancy G − G ohne Komponente): –

Gate: {'pass': False, 'criteria': {}, 'failed': ['keine Komponente mit KEEP – G ist identisch mit A']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00257, 'hit_rate': 0.5246, 'n_cohorts': 43}, 'B': {'expectancy': 0.0049, 'hit_rate': 0.5266, 'n_cohorts': 43}, 'C': {'expectancy': 0.00868, 'hit_rate': 0.5471, 'n_cohorts': 43}, 'D': {'expectancy': 0.00961, 'hit_rate': 0.5602, 'n_cohorts': 43}, 'E': {'expectancy': 0.0095, 'hit_rate': 0.5617, 'n_cohorts': 43}, 'F': {'expectancy': 0.00231, 'hit_rate': 0.5227, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.00169, 'hit_rate': 0.5321, 'n_cohorts': 9}, 'A_abstention': {'expectancy': 0.00257, 'hit_rate': 0.5246, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.01155, 'hit_rate': 0.4231, 'n_cohorts': 13}, 'B': {'expectancy': -0.00295, 'hit_rate': 0.4679, 'n_cohorts': 13}, 'C': {'expectancy': -0.00807, 'hit_rate': 0.4535, 'n_cohorts': 13}, 'D': {'expectancy': -0.01058, 'hit_rate': 0.4407, 'n_cohorts': 13}, 'E': {'expectancy': -0.01053, 'hit_rate': 0.4439, 'n_cohorts': 13}, 'F': {'expectancy': -0.00742, 'hit_rate': 0.4308, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00908, 'hit_rate': 0.4169, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.01155, 'hit_rate': 0.4231, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.07103, 'hit_rate': 0.75, 'n_cohorts': 8}, 'B': {'expectancy': 0.07404, 'hit_rate': 0.75, 'n_cohorts': 8}, 'C': {'expectancy': 0.07389, 'hit_rate': 0.7628, 'n_cohorts': 8}, 'D': {'expectancy': 0.06043, 'hit_rate': 0.7066, 'n_cohorts': 8}, 'E': {'expectancy': 0.06098, 'hit_rate': 0.7092, 'n_cohorts': 8}, 'F': {'expectancy': 0.05264, 'hit_rate': 0.7025, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.03605, 'hit_rate': 0.7371, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.07103, 'hit_rate': 0.75, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.2386, 'rate_shock': 0.1987, 'crash': 0.36, 'usd_shock': 0.2719, 'oil_shock': 0.1603, 'inflation_shock': 0.0528, 'liquidity_shock': 0.198, 'momentum_reversal': 0.4719, 'volatility_flip': 0.7321}}

Decision Intelligence: {'top_decile': {'cagr': 0.0075, 'sharpe': 0.127, 'sortino': 0.146, 'max_dd': -0.1806, 'calmar': 0.042, 'profit_factor': 1.031, 'hit_rate': 0.4862, 'avg_winner': 0.06267, 'avg_loser': -0.05749, 'payoff': 1.09, 'expectancy': 0.00093, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4565, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0733, 'avg_mae': -0.0635, 'es5_monthly': -0.04911, 'sector_hhi': 0.1826}, 'diversified': {'cagr': 0.0007, 'sharpe': 0.05, 'sortino': 0.056, 'max_dd': -0.1833, 'calmar': 0.004, 'profit_factor': 1.012, 'hit_rate': 0.4841, 'avg_winner': 0.06195, 'avg_loser': -0.05745, 'payoff': 1.078, 'expectancy': 0.00035, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.4603, 'exposure': 1.0, 'n_trades': 11059, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0728, 'avg_mae': -0.0635, 'es5_monthly': -0.04701, 'sector_hhi': 0.159}, 'verdict': 'REJECT'}

Anteil fragiler Positionen im Top-Dezil: 0.8996

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=92 (Segment 334), typischer Fehler -0.1556, Lift 2.75, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_002: n=97 (Segment 355), typischer Fehler -0.1606, Lift 2.73, Eigenschaften {'sector': 'Energy', 'beta': 'high_beta'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_003: n=112 (Segment 452), typischer Fehler -0.1596, Lift 2.48, Eigenschaften {'sector': 'Energy', 'volatility': 'high_vol'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_004: n=259 (Segment 1047), typischer Fehler -0.1842, Lift 2.47, Eigenschaften {'lottery': 'lottery_profile', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=395 (Segment 1988), typischer Fehler -0.1775, Lift 1.99, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_006: n=143 (Segment 747), typischer Fehler -0.1508, Lift 1.91, Eigenschaften {'sector': 'Energy'}, Abdeckung MEDIUM -> bekannt – Failure-Profil beobachten
- UNKNOWN_CLUSTER_007: n=541 (Segment 2982), typischer Fehler -0.1685, Lift 1.81, Eigenschaften {'recent_move': 'extreme_5d_move', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=202 (Segment 1157), typischer Fehler -0.1696, Lift 1.75, Eigenschaften {'momentum': 'loser_12m', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=119 (Segment 689), typischer Fehler -0.176, Lift 1.73, Eigenschaften {'sector': 'Consumer Cyclical', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=242 (Segment 1419), typischer Fehler -0.1698, Lift 1.71, Eigenschaften {'sector': 'Consumer Cyclical', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=1216 (Segment 7151), typischer Fehler -0.1652, Lift 1.7, Eigenschaften {'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=364 (Segment 2216), typischer Fehler -0.1716, Lift 1.64, Eigenschaften {'momentum': 'loser_12m', 'beta': 'high_beta'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=212 (Segment 1303), typischer Fehler -0.1707, Lift 1.63, Eigenschaften {'sector': 'Technology', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=127 (Segment 788), typischer Fehler -0.1856, Lift 1.61, Eigenschaften {'momentum': 'loser_12m', 'trend': 'downtrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=391 (Segment 2499), typischer Fehler -0.1618, Lift 1.56, Eigenschaften {'sector': 'Technology', 'momentum': 'winner_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen

