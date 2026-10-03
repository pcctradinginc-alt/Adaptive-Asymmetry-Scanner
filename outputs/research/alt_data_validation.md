# Alternative Data – inkrementelle Validierung (2026-10-03T09:22:53+00:00, Protokoll alt-v1)

## sec_deep_events: **MODIFY** – nur in einzelnen Sektoren positiv

Abdeckung (Dev-Jahre): 1.0 · ausgewählte Features: ['sec_late_filing_365d', 'sec_insider_net_value_90d', 'sec_exec_change_90d', 'sec_insider_cluster_30d', 'sec_filing_delay_z'] · Source Value Score 0.72 {'incremental_value': 0.49, 'coverage': 1.0, 'freshness': 1.0, 'stability': 0.429, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

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
| sharpe | 0.396 | 0.381 | -0.015 |
| expectancy | 0.00337 | 0.0031 | -0.00027 |
| max_dd | -0.1569 | -0.1421 | 0.0148 |
| precision_at_k | 0.2476 | 0.2471 | -0.0005 |
| hit_rate | 0.4924 | 0.4922 | -0.0002 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25027 | 0.25024 | -3e-05 |
| ece | 0.00836 | 0.00715 | -0.00121 |
| log_loss | 0.69521 | 0.69519 | -2e-05 |

Bootstrap: {'delta_monthly_mean': -0.000224, 'ci_monthly_mean': [-0.001106, 0.0006], 'ci_sharpe': [-0.155, 0.076], 'alpha_one_sided': 0.0125, 'n_months': 77}
Δ je Jahr: {'2019': -0.00104, '2020': 0.00138, '2021': 0.0001, '2022': 0.0, '2023': 0.0, '2024': -0.00188, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01306, 'n_base': 6414, 'variant': 0.01336, 'n_variant': 6414, 'delta': 0.0003}, 'vix_lt_20': {'base': -0.00324, 'n_base': 9396, 'variant': -0.0039, 'n_variant': 9396, 'delta': -0.00066}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03376 | 0.02582 | -0.00794 |
| sharpe | 0.319 | 0.307 | -0.012 |
| expectancy | 0.0032 | 0.00317 | -3e-05 |
| max_dd | -0.1676 | -0.1994 | -0.0318 |
| precision_at_k | 0.2397 | 0.2457 | 0.006 |
| hit_rate | 0.5008 | 0.5008 | 0.0 |
| n_trades | 15810 | 15810 | 0 |
| brier | 0.25042 | 0.25074 | 0.00032 |
| ece | 0.01758 | 0.01987 | 0.00229 |
| log_loss | 0.69402 | 0.695 | 0.00098 |

Bootstrap: {'delta_monthly_mean': 3.4e-05, 'ci_monthly_mean': [-0.001957, 0.00191], 'ci_sharpe': [-0.247, 0.206], 'alpha_one_sided': 0.0125, 'n_months': 77}
Δ je Jahr: {'2019': 0.00114, '2020': 0.00184, '2021': 0.00065, '2022': -0.00132, '2023': 0.00019, '2024': -0.00202, '2025': -0.00062}
Regime: {'vix_ge_20': {'base': 0.00881, 'n_base': 6414, 'variant': 0.00857, 'n_variant': 6414, 'delta': -0.00024}, 'vix_lt_20': {'base': -0.00063, 'n_base': 9396, 'variant': -0.00052, 'n_variant': 9396, 'delta': 0.00011}}

## ted_procurement: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.0 · ausgewählte Features: – · Source Value Score 0.331 {'incremental_value': 0.0, 'coverage': 0.0, 'freshness': 1.0, 'stability': 0.0, 'mapping_quality': 0.808, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| ted_awards_90d | 0.0 | None | None | None | None |
| ted_any_award_365d | 0.0 | None | None | None | None |
| ted_award_value_365d | 0.0 | None | None | None | None |
| ted_awards_z | 0.0 | None | None | None | None |
