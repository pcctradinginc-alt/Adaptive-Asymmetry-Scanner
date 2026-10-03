# Alternative Data – inkrementelle Validierung (2026-10-03T12:06:37+00:00, Protokoll alt-v1)

## sec_deep_events: **MODIFY** – nur in einzelnen Sektoren positiv

Abdeckung (Dev-Jahre): 1.0 · ausgewählte Features: ['sec_late_filing_365d', 'sec_insider_net_value_90d', 'sec_exec_change_90d', 'sec_insider_cluster_30d', 'sec_filing_delay_z'] · Source Value Score 0.715 {'incremental_value': 0.477, 'coverage': 1.0, 'freshness': 1.0, 'stability': 0.429, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

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
| sec_late_filing_365d | 0.965 | 0.074 | vol_60 | 0.926 | -0.0139 |

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
| ece | 0.00838 | 0.00715 | -0.00123 |
| log_loss | 0.69521 | 0.69519 | -2e-05 |

Bootstrap: {'delta_monthly_mean': -0.000226, 'ci_monthly_mean': [-0.001184, 0.000668], 'ci_sharpe': [-0.162, 0.081], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': -0.00108, '2020': 0.00139, '2021': 0.0001, '2022': 0.0, '2023': 0.0, '2024': -0.00187, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01306, 'n_base': 6414, 'variant': 0.01337, 'n_variant': 6414, 'delta': 0.00031}, 'vix_lt_20': {'base': -0.00323, 'n_base': 9396, 'variant': -0.0039, 'n_variant': 9396, 'delta': -0.00067}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03605 | 0.02249 | -0.01356 |
| sharpe | 0.355 | 0.309 | -0.046 |
| expectancy | 0.00343 | 0.00323 | -0.0002 |
| max_dd | -0.1635 | -0.2262 | -0.0627 |
| precision_at_k | 0.2399 | 0.2471 | 0.0072 |
| hit_rate | 0.5027 | 0.4979 | -0.0048 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25044 | 0.25064 | 0.0002 |
| ece | 0.01755 | 0.01364 | -0.00391 |
| log_loss | 0.69405 | 0.6948 | 0.00075 |

Bootstrap: {'delta_monthly_mean': -0.000229, 'ci_monthly_mean': [-0.002771, 0.00217], 'ci_sharpe': [-0.359, 0.219], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': 0.00109, '2020': 0.00333, '2021': 0.00012, '2022': -0.0038, '2023': -0.00182, '2024': -0.00107, '2025': 0.00162}
Regime: {'vix_ge_20': {'base': 0.00876, 'n_base': 6414, 'variant': 0.00827, 'n_variant': 6414, 'delta': -0.00049}, 'vix_lt_20': {'base': -0.00022, 'n_base': 9396, 'variant': -0.0002, 'n_variant': 9396, 'delta': 2e-05}}

## ted_procurement: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.0 · ausgewählte Features: – · Source Value Score 0.331 {'incremental_value': 0.0, 'coverage': 0.0, 'freshness': 1.0, 'stability': 0.0, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| ted_awards_90d | 0.0 | None | None | None | None |
| ted_any_award_365d | 0.0 | None | None | None | None |
| ted_award_value_365d | 0.0 | None | None | None | None |
| ted_awards_z | 0.0 | None | None | None | None |

## sec_xbrl_fundamentals: **MODIFY** – nur in einzelnen Sektoren positiv

Abdeckung (Dev-Jahre): 0.989 · ausgewählte Features: ['xbrl_rev_yoy', 'xbrl_asset_growth', 'xbrl_accruals', 'xbrl_share_change', 'xbrl_sue'] · Source Value Score 0.686 {'incremental_value': 0.428, 'coverage': 0.989, 'freshness': 1.0, 'stability': 0.357, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

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
| sharpe | 0.397 | 0.194 | -0.203 |
| expectancy | 0.00338 | 0.00185 | -0.00153 |
| max_dd | -0.1569 | -0.2105 | -0.0536 |
| precision_at_k | 0.2477 | 0.2487 | 0.001 |
| hit_rate | 0.4925 | 0.4875 | -0.005 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25026 | 0.25016 | -0.0001 |
| ece | 0.00838 | 0.00397 | -0.00441 |
| log_loss | 0.69521 | 0.69413 | -0.00108 |

Bootstrap: {'delta_monthly_mean': -0.001563, 'ci_monthly_mean': [-0.004353, 0.001013], 'ci_sharpe': [-0.684, 0.102], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': 0.00011, '2020': 0.00116, '2021': -0.00506, '2022': -0.00528, '2023': 0.0, '2024': -0.00095, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01306, 'n_base': 6414, 'variant': 0.01057, 'n_variant': 6414, 'delta': -0.00249}, 'vix_lt_20': {'base': -0.00323, 'n_base': 9396, 'variant': -0.0041, 'n_variant': 9396, 'delta': -0.00087}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03605 | 0.03219 | -0.00386 |
| sharpe | 0.355 | 0.378 | 0.023 |
| expectancy | 0.00343 | 0.00363 | 0.0002 |
| max_dd | -0.1635 | -0.1993 | -0.0358 |
| precision_at_k | 0.2399 | 0.2397 | -0.0002 |
| hit_rate | 0.5027 | 0.4991 | -0.0036 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25044 | 0.25018 | -0.00026 |
| ece | 0.01755 | 0.01344 | -0.00411 |
| log_loss | 0.69405 | 0.69356 | -0.00049 |

Bootstrap: {'delta_monthly_mean': 0.000125, 'ci_monthly_mean': [-0.002738, 0.002866], 'ci_sharpe': [-0.375, 0.33], 'alpha_one_sided': 0.008333333333333333, 'n_months': 77}
Δ je Jahr: {'2019': -0.00102, '2020': 0.00625, '2021': -0.00248, '2022': -0.00113, '2023': 0.00033, '2024': -0.00125, '2025': 0.00025}
Regime: {'vix_ge_20': {'base': 0.00876, 'n_base': 6414, 'variant': 0.00797, 'n_variant': 6414, 'delta': -0.00079}, 'vix_lt_20': {'base': -0.00022, 'n_base': 9396, 'variant': 0.00067, 'n_variant': 9396, 'delta': 0.00089}}
