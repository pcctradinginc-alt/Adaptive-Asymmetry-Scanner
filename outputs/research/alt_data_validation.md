# Alternative Data – inkrementelle Validierung (2026-10-03T07:22:37+00:00, Protokoll alt-v1)

## sec_deep_events: **REJECT** – kein inkrementeller Nutzen gegenüber der Baseline

Abdeckung (Dev-Jahre): 1.0 · ausgewählte Features: ['sec_late_filing_365d', 'sec_insider_net_value_90d', 'sec_exec_change_90d', 'sec_insider_cluster_30d', 'sec_filing_delay_z'] · Source Value Score 0.727 {'incremental_value': 0.508, 'coverage': 1.0, 'freshness': 1.0, 'stability': 0.429, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| sec_insider_buy_value_90d | 1.0 | 0.155 | dist_52w_high | 0.845 | 0.001 |
| sec_insider_buyers_90d | 1.0 | 0.153 | dist_52w_high | 0.847 | 0.0012 |
| sec_insider_net_value_90d | 1.0 | 0.349 | mom_12_1 | 0.651 | -0.0098 |
| sec_insider_cluster_30d | 1.0 | 0.081 | mom_3m | 0.919 | -0.0057 |
| sec_8k_count_30d_z | 0.955 | 0.146 | vol_ratio | 0.854 | -0.0012 |
| sec_8k_negative_90d | 0.965 | 0.028 | vol_60 | 0.972 | 0.0032 |
| sec_exec_change_90d | 0.965 | 0.075 | log_dollar_vol | 0.925 | 0.0067 |
| sec_filing_delay_z | 0.963 | 0.03 | vol_60 | 0.97 | 0.005 |
| sec_late_filing_365d | 0.965 | 0.074 | vol_60 | 0.926 | -0.0138 |

### Baseline enet_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.01842 | 0.01677 | -0.00165 |
| sharpe | 0.397 | 0.381 | -0.016 |
| expectancy | 0.00338 | 0.00311 | -0.00027 |
| max_dd | -0.1569 | -0.1421 | 0.0148 |
| precision_at_k | 0.2477 | 0.2471 | -0.0006 |
| hit_rate | 0.4925 | 0.4923 | -0.0002 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25026 | 0.25024 | -2e-05 |
| ece | 0.00836 | 0.00729 | -0.00107 |
| log_loss | 0.69521 | 0.69519 | -2e-05 |

Bootstrap: {'delta_monthly_mean': -0.000226, 'ci_monthly_mean': [-0.000972, 0.000508], 'ci_sharpe': [-0.137, 0.068], 'alpha_one_sided': 0.025, 'n_months': 77}
Δ je Jahr: {'2019': -0.00108, '2020': 0.00139, '2021': 0.0001, '2022': 0.0, '2023': 0.0, '2024': -0.00187, '2025': 0.0}
Regime: {'unknown': {'base': 0.00338, 'n_base': 15810, 'variant': 0.00311, 'n_variant': 15810, 'delta': -0.00027}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03182 | 0.02861 | -0.00321 |
| sharpe | 0.27 | 0.301 | 0.031 |
| expectancy | 0.00275 | 0.00304 | 0.00029 |
| max_dd | -0.1695 | -0.2153 | -0.0458 |
| precision_at_k | 0.2353 | 0.244 | 0.0087 |
| hit_rate | 0.5001 | 0.4999 | -0.0002 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.2505 | 0.25058 | 8e-05 |
| ece | 0.02016 | 0.01695 | -0.00321 |
| log_loss | 0.69417 | 0.69465 | 0.00048 |

Bootstrap: {'delta_monthly_mean': 0.000394, 'ci_monthly_mean': [-0.001499, 0.002301], 'ci_sharpe': [-0.194, 0.234], 'alpha_one_sided': 0.025, 'n_months': 77}
Δ je Jahr: {'2019': 0.00184, '2020': 0.00395, '2021': 0.00031, '2022': -0.00181, '2023': -0.00032, '2024': -0.00202, '2025': 0.00136}
Regime: {'unknown': {'base': 0.00275, 'n_base': 15810, 'variant': 0.00304, 'n_variant': 15810, 'delta': 0.00029}}
