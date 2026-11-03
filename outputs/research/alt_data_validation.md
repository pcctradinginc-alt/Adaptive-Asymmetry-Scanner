# Alternative Data – inkrementelle Validierung (2026-11-03T15:47:36+00:00, Protokoll alt-v1)

## sec_deep_events: **REJECT** – kein inkrementeller Nutzen gegenüber der Baseline

Abdeckung (Dev-Jahre): 1.0 · ausgewählte Features: ['sec_late_filing_365d', 'sec_insider_net_value_90d', 'sec_exec_change_90d', 'sec_insider_cluster_30d', 'sec_filing_delay_z'] · Source Value Score 0.723 {'incremental_value': 0.506, 'coverage': 1.0, 'freshness': 0.967, 'stability': 0.429, 'mapping_quality': 0.806, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| sec_insider_buy_value_90d | 1.0 | 0.155 | dist_52w_high | 0.845 | 0.0009 |
| sec_insider_buyers_90d | 1.0 | 0.153 | dist_52w_high | 0.847 | 0.001 |
| sec_insider_net_value_90d | 1.0 | 0.349 | mom_12_1 | 0.651 | -0.01 |
| sec_insider_cluster_30d | 1.0 | 0.081 | mom_3m | 0.919 | -0.0057 |
| sec_8k_count_30d_z | 0.955 | 0.146 | vol_ratio | 0.854 | -0.0006 |
| sec_8k_negative_90d | 0.965 | 0.027 | vol_60 | 0.973 | 0.0033 |
| sec_exec_change_90d | 0.965 | 0.075 | log_dollar_vol | 0.925 | 0.0064 |
| sec_filing_delay_z | 0.963 | 0.032 | vol_60 | 0.968 | 0.0051 |
| sec_late_filing_365d | 0.965 | 0.074 | vol_60 | 0.926 | -0.0139 |

### Baseline enet_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.01805 | 0.01634 | -0.00171 |
| sharpe | 0.361 | 0.373 | 0.012 |
| expectancy | 0.0032 | 0.0031 | -0.0001 |
| max_dd | -0.1594 | -0.1406 | 0.0188 |
| precision_at_k | 0.2468 | 0.2467 | -0.0001 |
| hit_rate | 0.4921 | 0.4932 | 0.0011 |
| n_trades | 15729 | 15729 | 0 |
| brier | 0.25023 | 0.25022 | -1e-05 |
| ece | 0.00834 | 0.00631 | -0.00203 |
| log_loss | 0.69516 | 0.69515 | -1e-05 |

Bootstrap: {'delta_monthly_mean': -3.6e-05, 'ci_monthly_mean': [-0.00115, 0.000998], 'ci_sharpe': [-0.159, 0.119], 'alpha_one_sided': 0.0035714285714285718, 'n_months': 77}
Δ je Jahr: {'2019': -0.00037, '2020': 0.00096, '2021': 0.00021, '2022': 0.0, '2023': 0.0, '2024': -0.00103, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01344, 'n_base': 6387, 'variant': 0.01344, 'n_variant': 6387, 'delta': 0.0}, 'vix_lt_20': {'base': -0.00379, 'n_base': 9342, 'variant': -0.00397, 'n_variant': 9342, 'delta': -0.00018}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03094 | 0.02621 | -0.00473 |
| sharpe | 0.391 | 0.373 | -0.018 |
| expectancy | 0.00379 | 0.00402 | 0.00023 |
| max_dd | -0.1513 | -0.1752 | -0.0239 |
| precision_at_k | 0.2424 | 0.2544 | 0.012 |
| hit_rate | 0.5022 | 0.5026 | 0.0004 |
| n_trades | 15729 | 15729 | 0 |
| brier | 0.25035 | 0.25059 | 0.00024 |
| ece | 0.01121 | 0.01557 | 0.00436 |
| log_loss | 0.69478 | 0.69514 | 0.00036 |

Bootstrap: {'delta_monthly_mean': 0.000157, 'ci_monthly_mean': [-0.002519, 0.003127], 'ci_sharpe': [-0.288, 0.336], 'alpha_one_sided': 0.0035714285714285718, 'n_months': 77}
Δ je Jahr: {'2019': 0.00187, '2020': 0.00085, '2021': -0.00283, '2022': -0.00203, '2023': 0.00131, '2024': 0.00422, '2025': -0.00573}
Regime: {'vix_ge_20': {'base': 0.01059, 'n_base': 6387, 'variant': 0.0106, 'n_variant': 6387, 'delta': 1e-05}, 'vix_lt_20': {'base': -0.00085, 'n_base': 9342, 'variant': -0.00047, 'n_variant': 9342, 'delta': 0.00038}}

## ted_procurement: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.0 · ausgewählte Features: – · Source Value Score 0.328 {'incremental_value': 0.0, 'coverage': 0.0, 'freshness': 0.975, 'stability': 0.0, 'mapping_quality': 0.806, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| ted_awards_90d | 0.0 | None | None | None | None |
| ted_any_award_365d | 0.0 | None | None | None | None |
| ted_award_value_365d | 0.0 | None | None | None | None |
| ted_awards_z | 0.0 | None | None | None | None |

## sec_xbrl_fundamentals: **REJECT** – kein inkrementeller Nutzen gegenüber der Baseline

Abdeckung (Dev-Jahre): 0.989 · ausgewählte Features: ['xbrl_rev_yoy', 'xbrl_accruals', 'xbrl_asset_growth', 'xbrl_share_change', 'xbrl_sue'] · Source Value Score 0.711 {'incremental_value': 0.461, 'coverage': 0.989, 'freshness': 0.975, 'stability': 0.5, 'mapping_quality': 0.806, 'api_reliability': 1.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| xbrl_rev_yoy | 0.232 | 0.225 | mom_12_1 | 0.775 | 0.0397 |
| xbrl_sue | 0.912 | 0.227 | mom_12_1 | 0.773 | -0.0045 |
| xbrl_accruals | 0.627 | 0.147 | log_dollar_vol | 0.853 | -0.018 |
| xbrl_asset_growth | 0.96 | 0.123 | mom_12_1 | 0.877 | 0.0178 |
| xbrl_share_change | 0.866 | 0.126 | beta_126 | 0.874 | -0.007 |

### Baseline enet_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.01805 | 0.00887 | -0.00918 |
| sharpe | 0.361 | 0.204 | -0.157 |
| expectancy | 0.0032 | 0.00199 | -0.00121 |
| max_dd | -0.1594 | -0.2063 | -0.0469 |
| precision_at_k | 0.2468 | 0.2492 | 0.0024 |
| hit_rate | 0.4921 | 0.4876 | -0.0045 |
| n_trades | 15729 | 15729 | 0 |
| brier | 0.25023 | 0.25015 | -8e-05 |
| ece | 0.00834 | 0.00417 | -0.00417 |
| log_loss | 0.69516 | 0.69408 | -0.00108 |

Bootstrap: {'delta_monthly_mean': -0.001245, 'ci_monthly_mean': [-0.004501, 0.001657], 'ci_sharpe': [-0.676, 0.167], 'alpha_one_sided': 0.0035714285714285718, 'n_months': 77}
Δ je Jahr: {'2019': 0.00069, '2020': 0.00087, '2021': -0.00494, '2022': -0.00445, '2023': 0.0, '2024': -0.00016, '2025': 0.0}
Regime: {'vix_ge_20': {'base': 0.01344, 'n_base': 6387, 'variant': 0.01106, 'n_variant': 6387, 'delta': -0.00238}, 'vix_lt_20': {'base': -0.00379, 'n_base': 9342, 'variant': -0.00422, 'n_variant': 9342, 'delta': -0.00043}}

### Baseline hgb_xs20_v1

| Metrik | Baseline | + Quelle | Δ |
|---|---|---|---|
| ic | 0.03094 | 0.03144 | 0.0005 |
| sharpe | 0.391 | 0.434 | 0.043 |
| expectancy | 0.00379 | 0.00432 | 0.00053 |
| max_dd | -0.1513 | -0.1656 | -0.0143 |
| precision_at_k | 0.2424 | 0.2455 | 0.0031 |
| hit_rate | 0.5022 | 0.5008 | -0.0014 |
| n_trades | 15729 | 15729 | 0 |
| brier | 0.25035 | 0.25027 | -8e-05 |
| ece | 0.01121 | 0.01422 | 0.00301 |
| log_loss | 0.69478 | 0.69371 | -0.00107 |

Bootstrap: {'delta_monthly_mean': 0.000469, 'ci_monthly_mean': [-0.002449, 0.003331], 'ci_sharpe': [-0.334, 0.383], 'alpha_one_sided': 0.0035714285714285718, 'n_months': 77}
Δ je Jahr: {'2019': 0.00078, '2020': 0.00226, '2021': -0.00135, '2022': -0.0027, '2023': 0.00276, '2024': 0.00063, '2025': 0.00149}
Regime: {'vix_ge_20': {'base': 0.01059, 'n_base': 6387, 'variant': 0.00883, 'n_variant': 6387, 'delta': -0.00176}, 'vix_lt_20': {'base': -0.00085, 'n_base': 9342, 'variant': 0.00124, 'n_variant': 9342, 'delta': 0.00209}}

## commodity_price: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.06 · ausgewählte Features: – · Source Value Score 0.14 {'incremental_value': 0.0, 'coverage': 0.06, 'freshness': 0.0, 'stability': 0.0, 'mapping_quality': 0.806, 'api_reliability': 0.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| cmdx_oil__wti_ret_20d | 0.041 | None | None | None | None |
| cmdx_oil__wti_ret_60d | 0.041 | None | None | None | None |
| cmdx_oil__wti_z_60d | 0.041 | None | None | None | None |
| cmdx_natural_gas__henry_hub_ret_20d | 0.014 | None | None | None | None |
| cmdx_natural_gas__henry_hub_ret_60d | 0.014 | None | None | None | None |
| cmdx_copper__copper_ret_3m | 0.001 | None | None | None | None |
| cmdx_corn__corn_ret_3m | 0.002 | None | None | None | None |
| cmdx_wheat__wheat_ret_3m | 0.0 | None | None | None | None |
| cmdx_soybeans__soybeans_ret_3m | 0.001 | None | None | None | None |

## commodity_fundamental: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.053 · ausgewählte Features: – · Source Value Score 0.139 {'incremental_value': 0.0, 'coverage': 0.053, 'freshness': 0.0, 'stability': 0.0, 'mapping_quality': 0.806, 'api_reliability': 0.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| cmdx_oil__crude_stocks_vs_5y | 0.014 | None | None | None | None |
| cmdx_oil__crude_stocks_chg_z_52w | 0.041 | None | None | None | None |
| cmdx_oil__product_supplied_chg_4w | 0.041 | None | None | None | None |
| cmdx_oil__refinery_utilization_z_52w | 0.041 | None | None | None | None |
| cmdx_natural_gas__natgas_storage_vs_5y | 0.004 | None | None | None | None |
| cmdx_natural_gas__natgas_storage_chg_z_52w | 0.014 | None | None | None | None |
| cmdx_natural_gas__lng_exports_yoy | 0.012 | None | None | None | None |

## commodity_positioning: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.065 · ausgewählte Features: – · Source Value Score 0.14 {'incremental_value': 0.0, 'coverage': 0.065, 'freshness': 0.0, 'stability': 0.0, 'mapping_quality': 0.806, 'api_reliability': 0.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| cmdx_oil__cot_crude_oil_mm_pctile_1y | 0.041 | None | None | None | None |
| cmdx_natural_gas__cot_natural_gas_mm_pctile_1y | 0.014 | None | None | None | None |
| cmdx_gold__cot_gold_mm_pctile_1y | 0.002 | None | None | None | None |
| cmdx_silver__cot_silver_mm_pctile_1y | 0.0 | None | None | None | None |
| cmdx_copper__cot_copper_mm_pctile_1y | 0.002 | None | None | None | None |
| cmdx_corn__cot_corn_mm_pctile_1y | 0.005 | None | None | None | None |
| cmdx_wheat__cot_wheat_mm_pctile_1y | 0.0 | None | None | None | None |
| cmdx_soybeans__cot_soybeans_mm_pctile_1y | 0.002 | None | None | None | None |
| cmdx_oil__cot_crude_oil_mm_net_chg_4w | 0.041 | None | None | None | None |
| cmdx_natural_gas__cot_natural_gas_mm_net_chg_4w | 0.014 | None | None | None | None |
| cmdx_gold__cot_gold_mm_net_chg_4w | 0.002 | None | None | None | None |
| cmdx_silver__cot_silver_mm_net_chg_4w | 0.0 | None | None | None | None |
| cmdx_copper__cot_copper_mm_net_chg_4w | 0.002 | None | None | None | None |
| cmdx_corn__cot_corn_mm_net_chg_4w | 0.005 | None | None | None | None |
| cmdx_wheat__cot_wheat_mm_net_chg_4w | 0.0 | None | None | None | None |
| cmdx_soybeans__cot_soybeans_mm_net_chg_4w | 0.002 | None | None | None | None |

## commodity_divergence: **REJECT** – keine nicht-redundante Feature mit ausreichender Abdeckung

Abdeckung (Dev-Jahre): 0.054 · ausgewählte Features: – · Source Value Score 0.139 {'incremental_value': 0.0, 'coverage': 0.054, 'freshness': 0.0, 'stability': 0.0, 'mapping_quality': 0.806, 'api_reliability': 0.0, 'revision_risk': 1.0}

| Feature | Abdeckung | max |ρ| bestehend | ähnlichstes | Info-Score | IC (Auswahljahre) |
|---|---|---|---|---|---|
| cmdx_oil__div_oil_positioning_price | 0.041 | None | None | None | None |
| cmdx_oil__div_oil_price_inventory | 0.041 | None | None | None | None |
| cmdx_natural_gas__div_gas_storage_price | 0.004 | None | None | None | None |
| cmdx_copper__div_copper_positioning_price | 0.001 | None | None | None | None |
