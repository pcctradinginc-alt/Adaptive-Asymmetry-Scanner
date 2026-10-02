# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-02T19:18:53+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: ['abstention'] · Gate nicht erfüllt: ['3_min_cohorts', '5_risk_adjusted', '6_no_regime_collapse']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0202 | 0.218 | 0.233 | 0.117 | -0.1724 | -0.08228 | 0.4735 | 1.063 | 0.0025 | 0.08846 | -0.07483 | 0.2642 | 0.24948 | 0.00567 | 0.69212 | 0.3624 | 11587 | 0.485 (334) | 0.01153 | -0.00319 | 1.0 | 0.01953 |
| B | -0.004 | 0.013 | 0.014 | -0.022 | -0.1819 | -0.06266 | 0.4721 | 1.013 | 0.0005 | 0.08242 | -0.07276 | 0.253 | 0.24985 | 0.0148 | 0.69286 | 0.416 | 11587 | 0.5682 (44) | 0.01028 | -0.00704 | 1.0 | 0.01805 |
| C | 0.0151 | 0.184 | 0.182 | 0.087 | -0.1732 | -0.07152 | 0.4765 | 1.062 | 0.00234 | 0.08458 | -0.07251 | 0.2573 | 0.25042 | 0.01394 | 0.6958 | 0.4171 | 11587 | 0.4341 (1092) | 0.01181 | -0.00697 | 1.0 | 0.0192 |
| D | 0.0146 | 0.185 | 0.18 | 0.093 | -0.1568 | -0.06482 | 0.4768 | 1.059 | 0.00213 | 0.08065 | -0.06943 | 0.2476 | 0.25024 | 0.01597 | 0.69364 | 0.4094 | 11587 | 0.4069 (1612) | 0.01127 | -0.00727 | 1.0 | 0.02028 |
| E | 0.0181 | 0.217 | 0.215 | 0.117 | -0.155 | -0.06403 | 0.4771 | 1.067 | 0.00242 | 0.08104 | -0.06931 | 0.2492 | 0.25026 | 0.01434 | 0.6937 | 0.4117 | 11587 | 0.409 (1560) | 0.01122 | -0.0069 | 1.0 | 0.02041 |
| F | 0.0202 | 0.277 | 0.331 | 0.151 | -0.134 | -0.04206 | 0.482 | 1.058 | 0.00192 | 0.07307 | -0.06429 | 0.2431 | 0.24927 | 0.00425 | 0.6917 | 0.6605 | 10508 | 0.4825 (143) | 0.00606 | -0.00324 | 1.0 | 0.005 |
| A_blindspot | 0.0124 | 0.202 | 0.269 | 0.099 | -0.1255 | -0.0388 | 0.478 | 1.018 | 0.00052 | 0.0607 | -0.05458 | 0.2372 | 0.24915 | 0.01451 | 0.69304 | 0.5585 | 4561 | 0.4676 (417) | 0.0056 | -0.00301 | 1.0 | 0.00632 |
| A_abstention | 0.0781 | 0.654 | 0.732 | 0.68 | -0.1149 | -0.05609 | 0.5101 | 1.29 | 0.01059 | 0.0924 | -0.07458 | 0.2786 | 0.24836 | 0.0085 | 0.68985 | 0.3911 | 4619 | 0.5609 (230) | 0.03164 | -0.02515 | 0.4 | 0.04333 |
| G | 0.0781 | 0.654 | 0.732 | 0.68 | -0.1149 | -0.05609 | 0.5101 | 1.29 | 0.01059 | 0.0924 | -0.07458 | 0.2786 | 0.24836 | 0.0085 | 0.68985 | 0.3911 | 4619 | 0.5609 (230) | 0.03164 | -0.02515 | 0.4 | 0.04333 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ -0.000532 CI [-0.006522, 0.005937] · ohne Top-5 %-Monate -0.003433
- D: Monats-Δ -0.000657 CI [-0.006576, 0.005297] · ohne Top-5 %-Monate -0.003137
- E: Monats-Δ -0.000375 CI [-0.006213, 0.005657] · ohne Top-5 %-Monate -0.002873
- F: Monats-Δ -0.000414 CI [-0.006461, 0.005893] · ohne Top-5 %-Monate -0.003882
- A_blindspot: Monats-Δ -0.001118 CI [-0.009836, 0.00755] · ohne Top-5 %-Monate -0.005559
- A_abstention: Monats-Δ 0.004559 CI [-0.002634, 0.012366] · ohne Top-5 %-Monate 0.000497

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': True, 'decision_intelligence': 'MODIFY'}

Abstinenz-Bestätigung (ungesehene Jahre [2019, 2020]): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.0307, 'inactive_expectancy': 0.0027, 'diff_t': 3.54, 'confirmed': True, 'years': [2019, 2020]}

Ablation (Δ Expectancy G − G ohne Komponente): {'abstention': 0.00809}

Gate: {'pass': False, 'criteria': {'1_no_leakage': True, '2_reproducible': True, '3_min_cohorts': False, '4_calibration_not_worse': True, '5_risk_adjusted': False, '6_no_regime_collapse': False, '7_hc_hit_rate_higher': True, '8_drawdown': True, '9_not_outlier_driven': True, '10_complexity_pays': True}, 'failed': ['3_min_cohorts', '5_risk_adjusted', '6_no_regime_collapse']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00831, 'hit_rate': 0.5344, 'n_cohorts': 43}, 'B': {'expectancy': 0.01324, 'hit_rate': 0.5465, 'n_cohorts': 43}, 'C': {'expectancy': 0.01559, 'hit_rate': 0.5512, 'n_cohorts': 43}, 'D': {'expectancy': 0.01501, 'hit_rate': 0.5605, 'n_cohorts': 43}, 'E': {'expectancy': 0.01518, 'hit_rate': 0.5609, 'n_cohorts': 43}, 'F': {'expectancy': 0.00524, 'hit_rate': 0.5113, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.00281, 'hit_rate': 0.5083, 'n_cohorts': 43}, 'A_abstention': {'expectancy': 0.00831, 'hit_rate': 0.5344, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.0064, 'hit_rate': 0.4123, 'n_cohorts': 13}, 'B': {'expectancy': -0.00891, 'hit_rate': 0.4277, 'n_cohorts': 13}, 'C': {'expectancy': -0.01181, 'hit_rate': 0.4215, 'n_cohorts': 13}, 'D': {'expectancy': -0.01063, 'hit_rate': 0.4231, 'n_cohorts': 13}, 'E': {'expectancy': -0.00972, 'hit_rate': 0.4215, 'n_cohorts': 13}, 'F': {'expectancy': -0.00707, 'hit_rate': 0.422, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.0095, 'hit_rate': 0.3955, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.0064, 'hit_rate': 0.4123, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.07974, 'hit_rate': 0.6961, 'n_cohorts': 8}, 'B': {'expectancy': 0.06591, 'hit_rate': 0.6422, 'n_cohorts': 8}, 'C': {'expectancy': 0.07521, 'hit_rate': 0.6593, 'n_cohorts': 8}, 'D': {'expectancy': 0.06724, 'hit_rate': 0.6324, 'n_cohorts': 8}, 'E': {'expectancy': 0.06775, 'hit_rate': 0.6348, 'n_cohorts': 8}, 'F': {'expectancy': 0.05294, 'hit_rate': 0.6902, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.03853, 'hit_rate': 0.6443, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.07974, 'hit_rate': 0.6961, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.1309, 'rate_shock': 0.1696, 'crash': 0.3005, 'usd_shock': 0.1651, 'oil_shock': 0.1512, 'inflation_shock': 0.0507, 'liquidity_shock': 0.1691, 'momentum_reversal': 0.5211, 'volatility_flip': 0.8033}}

Decision Intelligence: {'top_decile': {'cagr': 0.0206, 'sharpe': 0.22, 'sortino': 0.237, 'max_dd': -0.1724, 'calmar': 0.12, 'profit_factor': 1.063, 'hit_rate': 0.4735, 'avg_winner': 0.08846, 'avg_loser': -0.07483, 'payoff': 1.182, 'expectancy': 0.0025, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.3624, 'exposure': 1.0, 'n_trades': 11587, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0977, 'avg_mae': -0.0811, 'es5_monthly': -0.08228, 'sector_hhi': 0.1776}, 'diversified': {'cagr': 0.0194, 'sharpe': 0.212, 'sortino': 0.231, 'max_dd': -0.17, 'calmar': 0.114, 'profit_factor': 1.061, 'hit_rate': 0.4737, 'avg_winner': 0.08794, 'avg_loser': -0.07462, 'payoff': 1.178, 'expectancy': 0.00239, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.3629, 'exposure': 1.0, 'n_trades': 11587, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0971, 'avg_mae': -0.0806, 'es5_monthly': -0.07915, 'sector_hhi': 0.1524}, 'verdict': 'MODIFY'}

Anteil fragiler Positionen im Top-Dezil: 0.938

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=97 (Segment 470), typischer Fehler -0.227, Lift 2.06, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_002: n=174 (Segment 986), typischer Fehler -0.2165, Lift 1.76, Eigenschaften {'sector': 'Communication Services', 'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_003: n=96 (Segment 544), typischer Fehler -0.2014, Lift 1.76, Eigenschaften {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_004: n=751 (Segment 4470), typischer Fehler -0.2046, Lift 1.68, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=210 (Segment 1260), typischer Fehler -0.2021, Lift 1.67, Eigenschaften {'liquidity': 'liquid', 'momentum': 'mid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_006: n=89 (Segment 539), typischer Fehler -0.2017, Lift 1.65, Eigenschaften {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=618 (Segment 3798), typischer Fehler -0.1978, Lift 1.63, Eigenschaften {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=56 (Segment 345), typischer Fehler -0.2091, Lift 1.62, Eigenschaften {'sector': 'Communication Services', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=75 (Segment 465), typischer Fehler -0.2148, Lift 1.61, Eigenschaften {'sector': 'Financial Services', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=1111 (Segment 6992), typischer Fehler -0.1986, Lift 1.59, Eigenschaften {'volatility': 'high_vol', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=878 (Segment 5586), typischer Fehler -0.1988, Lift 1.57, Eigenschaften {'volatility': 'high_vol', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=827 (Segment 5333), typischer Fehler -0.1981, Lift 1.55, Eigenschaften {'liquidity': 'liquid', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=1068 (Segment 6968), typischer Fehler -0.1976, Lift 1.53, Eigenschaften {'liquidity': 'liquid', 'trend': 'uptrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=449 (Segment 2944), typischer Fehler -0.1949, Lift 1.52, Eigenschaften {'sector': 'Technology', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=206 (Segment 1352), typischer Fehler -0.1908, Lift 1.52, Eigenschaften {'sector': 'Energy'}, Abdeckung LOW -> dedizierten Research-Track anlegen

