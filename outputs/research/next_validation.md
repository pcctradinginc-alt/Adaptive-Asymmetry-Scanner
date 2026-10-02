# Gesamtvalidierung Intelligenz-Komponenten – 2026-10-02T15:12:19+00:00

**Entscheidung: KEEP_CHAMPION** · G-Komponenten: ['abstention'] · Gate nicht erfüllt: ['3_min_cohorts', '5_risk_adjusted', '6_no_regime_collapse']

| Variante | CAGR | Sharpe | Sortino | Calmar | MaxDD | ES5 Monat | Hit | PF | Expectancy | Ø Gew. | Ø Verl. | Prec@K | Brier | ECE | LogLoss | Turnover | Trades | HC-Hit (n) | Stabilität σJahr | min Regime | aktiv | Locked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0272 | 0.262 | 0.288 | 0.152 | -0.1794 | -0.08591 | 0.4759 | 1.08 | 0.00319 | 0.09037 | -0.07596 | 0.2671 | 0.24952 | 0.0062 | 0.69219 | 0.3529 | 11587 | 0.4635 (356) | 0.01274 | -0.00265 | 1.0 | 0.01947 |
| B | 0.0468 | 0.468 | 0.529 | 0.272 | -0.1722 | -0.04945 | 0.4836 | 1.126 | 0.00481 | 0.08895 | -0.07397 | 0.2695 | 0.24972 | 0.01321 | 0.6926 | 0.3694 | 11587 | 0.6818 (22) | 0.01505 | -0.00143 | 1.0 | 0.00944 |
| C | 0.0273 | 0.28 | 0.29 | 0.153 | -0.1782 | -0.06349 | 0.4786 | 1.089 | 0.00344 | 0.08794 | -0.07413 | 0.266 | 0.25031 | 0.01653 | 0.69514 | 0.3883 | 11587 | 0.443 (1535) | 0.01319 | -0.00638 | 1.0 | 0.02261 |
| D | 0.0251 | 0.272 | 0.28 | 0.149 | -0.1682 | -0.06639 | 0.4789 | 1.084 | 0.00312 | 0.08385 | -0.07107 | 0.255 | 0.25033 | 0.01351 | 0.69514 | 0.3882 | 11587 | 0.3944 (1240) | 0.0127 | -0.00589 | 1.0 | 0.02318 |
| E | 0.0251 | 0.272 | 0.28 | 0.149 | -0.1685 | -0.06659 | 0.4792 | 1.084 | 0.00313 | 0.08382 | -0.07113 | 0.2549 | 0.25036 | 0.0145 | 0.69519 | 0.3882 | 11587 | 0.3911 (1240) | 0.01276 | -0.00597 | 1.0 | 0.02325 |
| F | 0.0051 | 0.102 | 0.113 | 0.029 | -0.178 | -0.04486 | 0.4786 | 1.019 | 0.00063 | 0.07167 | -0.06457 | 0.2393 | 0.24933 | 0.00441 | 0.69184 | 0.649 | 10510 | 0.461 (141) | 0.00529 | -0.00479 | 1.0 | 0.00912 |
| A_blindspot | 0.0142 | 0.226 | 0.327 | 0.123 | -0.1155 | -0.03477 | 0.4782 | 1.03 | 0.00084 | 0.06117 | -0.05444 | 0.2379 | 0.24911 | 0.01433 | 0.69286 | 0.5599 | 4557 | 0.4664 (283) | 0.00548 | -0.00216 | 1.0 | 0.00559 |
| A_abstention | 0.089 | 0.706 | 0.812 | 0.719 | -0.1238 | -0.05644 | 0.5118 | 1.311 | 0.01146 | 0.09427 | -0.07536 | 0.281 | 0.24821 | 0.00695 | 0.68956 | 0.383 | 4619 | 0.6235 (340) | 0.0353 | -0.02791 | 0.4 | 0.04475 |
| G | 0.089 | 0.706 | 0.812 | 0.719 | -0.1238 | -0.05644 | 0.5118 | 1.311 | 0.01146 | 0.09427 | -0.07536 | 0.281 | 0.24821 | 0.00695 | 0.68956 | 0.383 | 4619 | 0.6235 (340) | 0.0353 | -0.02791 | 0.4 | 0.04475 |

## Komponenten gegen A (Bootstrap, Bonferroni über 6)

- C: Monats-Δ -0.000177 CI [-0.006335, 0.006498] · ohne Top-5 %-Monate -0.003032
- D: Monats-Δ -0.000419 CI [-0.006558, 0.006079] · ohne Top-5 %-Monate -0.003171
- E: Monats-Δ -0.000415 CI [-0.006707, 0.00626] · ohne Top-5 %-Monate -0.003186
- F: Monats-Δ -0.002329 CI [-0.009933, 0.004985] · ohne Top-5 %-Monate -0.006036
- A_blindspot: Monats-Δ -0.001628 CI [-0.01082, 0.007873] · ohne Top-5 %-Monate -0.006189
- A_abstention: Monats-Δ 0.004813 CI [-0.002613, 0.012935] · ohne Top-5 %-Monate 0.00073

Verdikte: {'counterfactual_filter': 'REJECT', 'blind_spot_filter': 'REJECT', 'abstention': True, 'decision_intelligence': 'MODIFY'}

Abstinenz-Bestätigung (ungesehene Jahre [2019, 2020]): {'active_cohorts': 52, 'inactive_cohorts': 53, 'active_expectancy': 0.03023, 'inactive_expectancy': 0.00336, 'diff_t': 3.27, 'confirmed': True, 'years': [2019, 2020]}

Ablation (Δ Expectancy G − G ohne Komponente): {'abstention': 0.00827}

Gate: {'pass': False, 'criteria': {'1_no_leakage': True, '2_reproducible': True, '3_min_cohorts': False, '4_calibration_not_worse': True, '5_risk_adjusted': False, '6_no_regime_collapse': False, '7_hc_hit_rate_higher': True, '8_drawdown': True, '9_not_outlier_driven': True, '10_complexity_pays': True}, 'failed': ['3_min_cohorts', '5_risk_adjusted', '6_no_regime_collapse']}

Stress: {'historical_windows': {'covid_crash_2020': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'rate_inflation_shock_2022': {'A': {'expectancy': 0.00816, 'hit_rate': 0.5316, 'n_cohorts': 43}, 'B': {'expectancy': 0.01419, 'hit_rate': 0.5447, 'n_cohorts': 43}, 'C': {'expectancy': 0.0154, 'hit_rate': 0.5526, 'n_cohorts': 43}, 'D': {'expectancy': 0.01552, 'hit_rate': 0.5628, 'n_cohorts': 43}, 'E': {'expectancy': 0.01569, 'hit_rate': 0.5628, 'n_cohorts': 43}, 'F': {'expectancy': 0.00594, 'hit_rate': 0.5136, 'n_cohorts': 43}, 'A_blindspot': {'expectancy': 0.00186, 'hit_rate': 0.5171, 'n_cohorts': 43}, 'A_abstention': {'expectancy': 0.00816, 'hit_rate': 0.5316, 'n_cohorts': 43}}, 'q4_selloff_2018': {'n_cohorts': 0, 'note': 'außerhalb des OOS-Zeitraums'}, 'regional_banks_2023': {'A': {'expectancy': -0.00661, 'hit_rate': 0.4323, 'n_cohorts': 13}, 'B': {'expectancy': -0.0073, 'hit_rate': 0.4385, 'n_cohorts': 13}, 'C': {'expectancy': -0.01114, 'hit_rate': 0.4308, 'n_cohorts': 13}, 'D': {'expectancy': -0.00887, 'hit_rate': 0.4338, 'n_cohorts': 13}, 'E': {'expectancy': -0.0088, 'hit_rate': 0.44, 'n_cohorts': 13}, 'F': {'expectancy': -0.00734, 'hit_rate': 0.4274, 'n_cohorts': 13}, 'A_blindspot': {'expectancy': -0.00679, 'hit_rate': 0.4194, 'n_cohorts': 13}, 'A_abstention': {'expectancy': -0.00661, 'hit_rate': 0.4323, 'n_cohorts': 13}}, 'tariff_shock_2025': {'A': {'expectancy': 0.08791, 'hit_rate': 0.7108, 'n_cohorts': 8}, 'B': {'expectancy': 0.06037, 'hit_rate': 0.6348, 'n_cohorts': 8}, 'C': {'expectancy': 0.07399, 'hit_rate': 0.6618, 'n_cohorts': 8}, 'D': {'expectancy': 0.06924, 'hit_rate': 0.6373, 'n_cohorts': 8}, 'E': {'expectancy': 0.0699, 'hit_rate': 0.6397, 'n_cohorts': 8}, 'F': {'expectancy': 0.04202, 'hit_rate': 0.6332, 'n_cohorts': 8}, 'A_blindspot': {'expectancy': 0.0407, 'hit_rate': 0.645, 'n_cohorts': 8}, 'A_abstention': {'expectancy': 0.08791, 'hit_rate': 0.7108, 'n_cohorts': 8}}}, 'scenario_turnover_top_decile': {'vol_spike': 0.127, 'rate_shock': 0.1538, 'crash': 0.3, 'usd_shock': 0.1594, 'oil_shock': 0.1468, 'inflation_shock': 0.0477, 'liquidity_shock': 0.1686, 'momentum_reversal': 0.5006, 'volatility_flip': 0.8055}}

Decision Intelligence: {'top_decile': {'cagr': 0.0277, 'sharpe': 0.264, 'sortino': 0.294, 'max_dd': -0.1794, 'calmar': 0.155, 'profit_factor': 1.08, 'hit_rate': 0.4759, 'avg_winner': 0.09037, 'avg_loser': -0.07596, 'payoff': 1.19, 'expectancy': 0.00319, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.3529, 'exposure': 1.0, 'n_trades': 11587, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0993, 'avg_mae': -0.0817, 'es5_monthly': -0.08591, 'sector_hhi': 0.1815}, 'diversified': {'cagr': 0.0236, 'sharpe': 0.242, 'sortino': 0.274, 'max_dd': -0.1687, 'calmar': 0.14, 'profit_factor': 1.069, 'hit_rate': 0.4751, 'avg_winner': 0.08908, 'avg_loser': -0.0754, 'payoff': 1.181, 'expectancy': 0.00274, 'precision_at_k': 0.0, 'recall_strong': 0.0, 'turnover': 0.3569, 'exposure': 1.0, 'n_trades': 11587, 'n_cohorts': 230, 'n_months': 53, 'avg_mfe': 0.0981, 'avg_mae': -0.081, 'es5_monthly': -0.07805, 'sector_hhi': 0.1526}, 'verdict': 'MODIFY'}

Anteil fragiler Positionen im Top-Dezil: 0.9344

Hinweise: {}

## Unknown-Unknown-Cluster

- UNKNOWN_CLUSTER_001: n=99 (Segment 464), typischer Fehler -0.2324, Lift 2.13, Eigenschaften {'sector': 'Energy', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_002: n=178 (Segment 1026), typischer Fehler -0.2208, Lift 1.73, Eigenschaften {'sector': 'Communication Services', 'volatility': 'high_vol'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_003: n=94 (Segment 554), typischer Fehler -0.2114, Lift 1.7, Eigenschaften {'sector': 'Communication Services', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_004: n=59 (Segment 354), typischer Fehler -0.2076, Lift 1.67, Eigenschaften {'sector': 'Communication Services', 'momentum': 'loser_12m'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_005: n=90 (Segment 541), typischer Fehler -0.208, Lift 1.66, Eigenschaften {'recent_move': 'extreme_5d_move', 'near_high': 'near_52w_high'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_006: n=77 (Segment 464), typischer Fehler -0.2181, Lift 1.66, Eigenschaften {'sector': 'Financial Services', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_007: n=757 (Segment 4572), typischer Fehler -0.206, Lift 1.66, Eigenschaften {'lottery': 'lottery_profile'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_008: n=212 (Segment 1306), typischer Fehler -0.2034, Lift 1.62, Eigenschaften {'liquidity': 'liquid', 'momentum': 'mid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_009: n=621 (Segment 3842), typischer Fehler -0.1998, Lift 1.62, Eigenschaften {'liquidity': 'liquid', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_010: n=1131 (Segment 7124), typischer Fehler -0.1993, Lift 1.59, Eigenschaften {'volatility': 'high_vol', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_011: n=212 (Segment 1345), typischer Fehler -0.1941, Lift 1.58, Eigenschaften {'sector': 'Energy'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_012: n=847 (Segment 5387), typischer Fehler -0.1986, Lift 1.57, Eigenschaften {'liquidity': 'liquid', 'vix': 'vix_lt_20'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_013: n=122 (Segment 780), typischer Fehler -0.2301, Lift 1.56, Eigenschaften {'sector': 'Communication Services', 'liquidity': 'liquid'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_014: n=870 (Segment 5621), typischer Fehler -0.2012, Lift 1.55, Eigenschaften {'volatility': 'high_vol', 'recent_move': 'extreme_5d_move'}, Abdeckung LOW -> dedizierten Research-Track anlegen
- UNKNOWN_CLUSTER_015: n=1086 (Segment 7021), typischer Fehler -0.1984, Lift 1.55, Eigenschaften {'liquidity': 'liquid', 'trend': 'uptrend'}, Abdeckung LOW -> dedizierten Research-Track anlegen

