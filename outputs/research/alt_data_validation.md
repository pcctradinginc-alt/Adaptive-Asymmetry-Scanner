# Alternative Data – inkrementelle Validierung (2026-10-03T13:01:29+00:00, Protokoll alt-v1)

## sec_deep_events: **MODIFY** – nur in einzelnen Sektoren positiv

Abdeckung (Dev-Jahre): 1.0 · ausgewählte Features: ['sec_late_filing_365d', 'sec_insider_net_value_90d', 'sec_exec_change_90d', 'sec_insider_cluster_30d', 'sec_filing_delay_z'] · Source Value Score 0.687 {'incremental_value': 0.444, 'coverage': 1.0, 'freshness': 1.0, 'stability': 0.286, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| sec_insider_buy_value_90d | 1.0 | 0.155 | dist_52w_high | 0.845 | 0.001 |
| sec_insider_buyers_90d | 1.0 | 0.153 | dist_52w_high | 0.847 | 0.0012 |
| sec_insider_net_value_90d | 1.0 | 0.349 | mom_12_1 | 0.651 | -0.0098 |
| sec_insider_cluster_30d | 1.0 | 0.081 | mom_3m | 0.919 | -0.0057 |
| sec_8k_count_30d_z | 0.955 | 0.146 | vol_ratio | 0.854 | -0.0012 |
| sec_8k_negative_90d | 0.965 | 0.028 | vol_60 | 0.972 | 0.0032 |
| sec_exec_change_90d | 0.965 | 0.075 | log_dollar_vol | 0.925 | 0.0067 |
| sec_filing_delay_z | 0.963 | 0.03 | vol_60 | 0.97 | 0.0049 |
| sec_late_filing_365d | 0.965 | 0.074 | vol_60 | 0.926 | -0.0138 |

### Baseline enet_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.01842 | 0.01677 | -0.00165 |
| sharpe | 0.398 | 0.381 | -0.017 |
| expectancy | 0.00338 | 0.0031 | -0.00028 |
| max_dd | -0.1569 | -0.1421 | 0.0148 |
| precision_at_k | 0.2477 | 0.2471 | -0.0006 |
| hit_rate | 0.4925 | 0.4923 | -0.0002 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25027 | 0.25024 | -3e-05 |
| ece | 0.00834 | 0.00718 | -0.00116 |
| log_loss | 0.69522 | 0.69518 | -4e-05 |

Bootstrap: {'delta_monthly_mean': -0.000232, 'ci_monthly_mean': [-0.00119, 0.000661], 'ci_sharpe': [-0.162, 0.081], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': -0.00108, '2020': 0.00138, '2021': 9e-05, '2022': 0.0, '2023': 0.0, '2024': -0.00188, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01307, 'n_base': 6414, 'variant': 0.01337, 'n_variant': 6414, 'delta': 0.0003}, 'vix_lt_20': {'base': -0.00323, 'n_base': 9396, 'variant': -0.0039, 'n_variant': 9396, 'delta': -0.00067}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03799 | 0.02733 | -0.01066 |
| sharpe | 0.434 | 0.301 | -0.133 |
| expectancy | 0.00404 | 0.00318 | -0.00086 |
| max_dd | -0.1432 | -0.2289 | -0.0857 |
| precision_at_k | 0.2326 | 0.2455 | 0.0129 |
| hit_rate | 0.5049 | 0.4987 | -0.0062 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25034 | 0.25071 | 0.00037 |
| ece | 0.01699 | 0.02086 | 0.00387 |
| log_loss | 0.69385 | 0.69497 | 0.00112 |

Bootstrap: {'delta_monthly_mean': -0.000884, 'ci_monthly_mean': [-0.003224, 0.001228], 'ci_sharpe': [-0.362, 0.087], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': 0.0018, '2020': -0.00044, '2021': -0.00195, '2022': -0.00128, '2023': -0.00514, '2024': -0.0013, '2025': 0.00635}
Regime: {'vix_ge_20': {'base': 0.00902, 'n_base': 6414, 'variant': 0.0089, 'n_variant': 6414, 'delta': -0.00012}, 'vix_lt_20': {'base': 0.00064, 'n_base': 9396, 'variant': -0.00072, 'n_variant': 9396, 'delta': -0.00136}}

## ted_procurement: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.0 · ausgewählte Features: – · Source Value Score 0.331 {'incremental_value': 0.0, 'coverage': 0.0, 'freshness': 1.0, 'stability': 0.0, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| ted_awards_90d | 0.0 | None | None | None | None |
| ted_any_award_365d | 0.0 | None | None | None | None |
| ted_award_value_365d | 0.0 | None | None | None | None |
| ted_awards_z | 0.0 | None | None | None | None |

## sec_xbrl_fundamentals: **MODIFY** – nur in einzelnen Sektoren positiv

Abdeckung (Dev-Jahre): 0.989 · ausgewählte Features: ['xbrl_rev_yoy', 'xbrl_asset_growth', 'xbrl_accruals', 'xbrl_share_change', 'xbrl_sue'] · Source Value Score 0.679 {'incremental_value': 0.409, 'coverage': 0.989, 'freshness': 1.0, 'stability': 0.357, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| xbrl_rev_yoy | 0.232 | 0.225 | mom_12_1 | 0.775 | 0.0397 |
| xbrl_sue | 0.912 | 0.227 | mom_12_1 | 0.773 | -0.0047 |
| xbrl_accruals | 0.628 | 0.143 | log_dollar_vol | 0.857 | -0.0178 |
| xbrl_asset_growth | 0.96 | 0.121 | mom_12_1 | 0.879 | 0.018 |
| xbrl_share_change | 0.866 | 0.127 | beta_126 | 0.873 | -0.0075 |

### Baseline enet_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.01842 | 0.00922 | -0.0092 |
| sharpe | 0.398 | 0.193 | -0.205 |
| expectancy | 0.00338 | 0.00185 | -0.00153 |
| max_dd | -0.1569 | -0.2109 | -0.054 |
| precision_at_k | 0.2477 | 0.2487 | 0.001 |
| hit_rate | 0.4925 | 0.4875 | -0.005 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25027 | 0.25016 | -0.00011 |
| ece | 0.00834 | 0.00397 | -0.00437 |
| log_loss | 0.69522 | 0.69413 | -0.00109 |

Bootstrap: {'delta_monthly_mean': -0.001572, 'ci_monthly_mean': [-0.004361, 0.001012], 'ci_sharpe': [-0.684, 0.102], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': 0.00011, '2020': 0.00114, '2021': -0.00505, '2022': -0.00528, '2023': 0.0, '2024': -0.001, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01307, 'n_base': 6414, 'variant': 0.01057, 'n_variant': 6414, 'delta': -0.0025}, 'vix_lt_20': {'base': -0.00323, 'n_base': 9396, 'variant': -0.00411, 'n_variant': 9396, 'delta': -0.00088}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03799 | 0.03507 | -0.00292 |
| sharpe | 0.434 | 0.407 | -0.027 |
| expectancy | 0.00404 | 0.00397 | -7e-05 |
| max_dd | -0.1432 | -0.183 | -0.0398 |
| precision_at_k | 0.2326 | 0.2428 | 0.0102 |
| hit_rate | 0.5049 | 0.4978 | -0.0071 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25034 | 0.25023 | -0.00011 |
| ece | 0.01699 | 0.01364 | -0.00335 |
| log_loss | 0.69385 | 0.69366 | -0.00019 |

Bootstrap: {'delta_monthly_mean': -0.000244, 'ci_monthly_mean': [-0.003071, 0.002643], 'ci_sharpe': [-0.417, 0.316], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': 0.00035, '2020': 0.00376, '2021': -0.00433, '2022': -5e-05, '2023': -0.00022, '2024': -0.00222, '2025': 0.00276}
Regime: {'vix_ge_20': {'base': 0.00902, 'n_base': 6414, 'variant': 0.00809, 'n_variant': 6414, 'delta': -0.00093}, 'vix_lt_20': {'base': 0.00064, 'n_base': 9396, 'variant': 0.00117, 'n_variant': 9396, 'delta': 0.00053}}
