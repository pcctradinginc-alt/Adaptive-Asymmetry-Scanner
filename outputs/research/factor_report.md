# Faktor-Report 2026-11-01

SHADOW: keine Produktionswirkung. Gewichte nur Forschung.

Ereignisse: 7805 ({'closed_trades': 117, 'ledger': 7688})

## Outcome `gross_1d` (n=6936, unabhängige Tage=20)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|
| ext.shipping_drybulk_z | 1690 | 20 | -0.1652 | [-0.5122, 0.2282] | -0.0854 | -0.0063 |  |
| ext.nfci_credit | 1649 | 19 | 0.1544 | [-0.2501, 0.5131] | 0.1737 | 0.0055 |  |
| ext.nfci | 1649 | 19 | 0.1488 | [-0.2556, 0.5088] | 0.1008 | 0.0055 |  |
| ext.de_truck_acceleration | 1690 | 20 | 0.1286 | [-0.2633, 0.4841] | 0.0783 | 0.0026 |  |
| ext.de_truck_z_1y | 1690 | 20 | 0.1286 | [-0.2633, 0.4841] | 0.0783 | 0.0026 |  |
| ext.freight_eu_z | 1690 | 20 | 0.1286 | [-0.2633, 0.4841] | 0.0783 | 0.0026 |  |
| ext.freight_global_z | 1690 | 20 | 0.1204 | [-0.2711, 0.4776] | 0.0584 | 0.0026 | decaying |
| ext.hdd_anomaly | 1690 | 20 | -0.1165 | [-0.4746, 0.2747] | -0.1247 | -0.0047 |  |
| ext.forecast_revision_cdd | 1690 | 20 | -0.1025 | [-0.4636, 0.2877] | 0.0331 | -0.0035 | decaying |
| ext.forecast_revision_hdd | 1690 | 20 | 0.1025 | [-0.2877, 0.4636] | -0.0331 | 0.0028 | decaying |
| ml_p_gt10_60 | 6630 | 19 | -0.1 | [-0.4712, 0.3012] | -0.0489 | -0.0043 | decaying |
| ml_q90_ret_60 | 6630 | 19 | -0.0995 | [-0.4708, 0.3017] | 0.0263 | -0.0037 | potential_leakage, decaying |
| ml_exp_ret_60 | 6630 | 19 | 0.0988 | [-0.3024, 0.4702] | 0.0848 | 0.0033 | potential_leakage |
| ml_q10_ret_60 | 6630 | 19 | 0.0946 | [-0.3063, 0.4669] | -0.0384 | 0.0032 | potential_leakage, decaying |
| ext.cdd_anomaly | 1690 | 20 | 0.0941 | [-0.2955, 0.4569] | 0.1323 | 0.0045 |  |
| dollar_volume | 5311 | 15 | -0.0902 | [-0.5119, 0.3665] | -0.0902 | -0.0057 |  |
| ml_hgb_xs20_risk_regime_v1 | 6627 | 19 | -0.0872 | [-0.461, 0.313] | 0.0454 | -0.0041 | decaying |
| ext.suez_z | 1649 | 19 | 0.0818 | [-0.3179, 0.4568] | 0.0279 | 0.0019 | decaying |
| ml_hgb_asym20_v1 | 6630 | 19 | -0.0818 | [-0.4568, 0.3178] | -0.0885 | -0.005 |  |
| ext.road_shipping_divergence_z | 1690 | 20 | 0.0768 | [-0.3113, 0.443] | 0.016 | 0.0039 | decaying |
| ml_hgb_xs20_v1 | 6630 | 19 | -0.0755 | [-0.4517, 0.3236] | 0.0204 | -0.0043 | decaying |
| ml_exp_dd_60 | 6630 | 19 | 0.0712 | [-0.3274, 0.4483] | -0.0404 | 0.003 | decaying |
| ml_enet_xs20_v1 | 6630 | 19 | -0.0709 | [-0.4481, 0.3277] | 0.0258 | -0.0032 | decaying |
| ext.freight_us_z | 1690 | 20 | 0.0703 | [-0.3172, 0.4377] | -0.0643 | 0.0026 | decaying |
| ext.us_freight_tsi_yoy | 1690 | 20 | -0.0703 | [-0.4377, 0.3172] | 0.0643 | 0.0002 | decaying |
| ext.us_freight_tsi_z | 1690 | 20 | 0.0703 | [-0.3172, 0.4377] | -0.0643 | 0.0026 | decaying |
| ext.shipping_global_z | 1690 | 20 | -0.0687 | [-0.4364, 0.3187] | -0.0185 | -0.0033 | decaying |
| scan_day_ret | 6783 | 19 | -0.0657 | [-0.4439, 0.3323] | -0.0812 | -0.0043 | potential_leakage |
| market_cap | 5311 | 15 | -0.063 | [-0.4914, 0.39] | -0.063 | -0.0045 |  |
| options_skew_25d | 1994 | 19 | 0.06 | [-0.3374, 0.4393] | 0.0142 | 0.0025 | decaying |
| ext.shipping_container_z | 1690 | 20 | 0.059 | [-0.3274, 0.4285] | 0.0446 | 0.0006 |  |
| ext.shipping_tanker_z | 1690 | 20 | 0.0581 | [-0.3282, 0.4278] | 0.0083 | 0.0037 | decaying |
| avg_volume | 5311 | 15 | -0.0562 | [-0.4862, 0.3958] | -0.0562 | -0.0025 |  |
| ext.usd_63d_chg | 1649 | 19 | 0.056 | [-0.341, 0.436] | -0.003 | 0.0026 | decaying |
| ext.shipping_negative_breadth | 1690 | 20 | -0.053 | [-0.4235, 0.3328] | -0.0093 | 0.003 | decaying |
| insider_count | 2086 | 20 | -0.0513 | [-0.4222, 0.3343] | -0.014 | -0.0004 | decaying |
| ml_disagreement_sd | 6630 | 19 | 0.0507 | [-0.3456, 0.4317] | 0.0563 | 0.0024 |  |
| ext.hard_data_vs_survey_divergence | 1690 | 20 | -0.0499 | [-0.421, 0.3355] | 0.0 | -0.0002 |  |
| ext.wti_63d_chg | 1649 | 19 | -0.0469 | [-0.4286, 0.349] | -0.0076 | -0.001 |  |
| ext.bab_el_mandeb_z | 1649 | 19 | 0.0455 | [-0.3502, 0.4275] | -0.02 | 0.0005 |  |
| ext.weather_disruption_index | 1649 | 19 | -0.0451 | [-0.4272, 0.3506] | -0.0709 | 0.0012 |  |
| ext.hormuz_z | 1649 | 19 | 0.0446 | [-0.3511, 0.4267] | -0.0213 | 0.0005 |  |
| ext.asia_export_z | 1690 | 20 | -0.0429 | [-0.4153, 0.3417] | -0.0437 | 0.0005 |  |
| insider_cluster | 2086 | 20 | -0.0408 | [-0.4135, 0.3436] | 0.011 | 0.0002 |  |
| ext.malacca_z | 1649 | 19 | 0.0407 | [-0.3545, 0.4235] | -0.0254 | 0.0026 |  |
| log_market_cap | 6783 | 19 | -0.0405 | [-0.4233, 0.3547] | -0.0362 | -0.0033 |  |
| ext.us_cpi_yoy | 1649 | 19 | 0.0352 | [-0.3592, 0.419] | -0.0018 | 0.0026 |  |
| ext.us_cpi_yoy_3m_ago | 1649 | 19 | -0.0352 | [-0.419, 0.3592] | 0.0018 | -0.0003 |  |
| fda_headline_count | 2086 | 20 | -0.035 | [-0.4086, 0.3487] | -0.0478 | 0.0008 |  |
| insider_buy_count | 2043 | 19 | -0.0324 | [-0.4167, 0.3617] | -0.007 | -0.0002 |  |
| ext.fed_assets_13w_chg | 1649 | 19 | -0.0269 | [-0.4121, 0.3665] | -0.0486 | -0.0023 |  |
| insider_sell_count | 2043 | 19 | -0.0246 | [-0.4102, 0.3685] | 0.0198 | 0.0003 |  |
| ml_momentum_12_1 | 6603 | 19 | -0.0244 | [-0.41, 0.3686] | -0.0005 | -0.0021 |  |
| has_near_earnings | 2086 | 20 | 0.023 | [-0.3592, 0.3986] | 0.0433 | 0.0022 |  |
| ext.de_power_price_z | 1649 | 19 | -0.0213 | [-0.4074, 0.3713] | -0.0506 | -0.0025 |  |
| ext.de_load_yoy | 1649 | 19 | 0.02 | [-0.3724, 0.4064] | 0.0751 | 0.0009 |  |
| sentiment_score | 6783 | 19 | 0.0194 | [-0.3729, 0.4059] | 0.0155 | 0.0007 |  |
| ml_hgb_xs20_momentum_v1 | 6627 | 19 | -0.0184 | [-0.4051, 0.3738] | 0.045 | -0.0015 |  |
| earnings_known | 2043 | 19 | -0.0169 | [-0.4038, 0.3751] | -0.0421 | 0.0006 |  |
| dealer_oi_ratio | 1933 | 19 | -0.0156 | [-0.4027, 0.3762] | -0.075 | 0.0009 |  |
| rel_volume | 6783 | 19 | -0.0123 | [-0.3999, 0.3791] | -0.0393 | -0.001 |  |
| short_pct_float | 6716 | 19 | 0.0117 | [-0.3796, 0.3994] | 0.0317 | 0.0017 |  |
| news_count | 6783 | 19 | 0.011 | [-0.3802, 0.3988] | 0.0216 | 0.0012 |  |
| ext.active_tropical_system | 1690 | 20 | -0.0098 | [-0.3875, 0.3706] | 0.0024 | 0.0034 |  |
| ext.de_power_price_7d | 1649 | 19 | -0.0089 | [-0.3971, 0.3819] | -0.0592 | -0.0011 |  |
| options_skew_ratio | 2086 | 20 | 0.0084 | [-0.3719, 0.3862] | 0.0298 | 0.0013 |  |
| sentiment_confidence | 6783 | 19 | 0.0076 | [-0.383, 0.396] | -0.0174 | 0.0013 |  |
| ext.panama_z | 1649 | 19 | 0.0062 | [-0.3842, 0.3948] | 0.0376 | 0.0021 |  |
| sentiment_drift | 6783 | 19 | 0.0034 | [-0.3867, 0.3924] | 0.007 | 0.0013 |  |
| short_mom | 6767 | 19 | -0.0031 | [-0.3922, 0.3869] | -0.0008 | 0.0002 |  |
| ext.freight_asia_z | 1690 | 20 | None | None | None | None | dead |
| ext.freight_breadth | 1690 | 20 | None | None | None | None | dead |
| ext.port_disruption_active_events | 1690 | 20 | None | None | None | None | dead |
| ext.port_disruption_curated_max_level | 1690 | 20 | None | None | None | None | dead |
| ext.port_disruption_curated_ports | 1690 | 20 | None | None | None | None | dead |
| ext.possible_weather_confounded_freight | 1649 | 19 | None | None | None | None | dead |
| ext.shipping_valid_port_count | 1690 | 20 | None | None | None | None | dead |
| insider_data_available | 2043 | 19 | None | None | None | None | dead |

Walk-Forward OOS: adaptiv {'n': 0} vs. gleichgewichtet {'n': 0}

Redundant: dollar_volume~log_market_cap (ρ=0.87); dollar_volume~market_cap (ρ=0.87); ext.asia_export_z~ext.hard_data_vs_survey_divergence (ρ=-0.8645); ext.asia_export_z~ext.shipping_container_z (ρ=-0.9519); ext.asia_export_z~ext.us_cpi_yoy (ρ=0.8528); ext.asia_export_z~ext.us_cpi_yoy_3m_ago (ρ=-0.8528); ext.bab_el_mandeb_z~ext.hormuz_z (ρ=0.9987); ext.bab_el_mandeb_z~ext.panama_z (ρ=-0.8744); ext.bab_el_mandeb_z~ext.shipping_negative_breadth (ρ=-0.9103); ext.bab_el_mandeb_z~ext.suez_z (ρ=0.8988); ext.de_power_price_7d~ext.de_power_price_z (ρ=0.9782); ext.de_truck_acceleration~ext.de_truck_z_1y (ρ=1.0); ext.de_truck_acceleration~ext.forecast_revision_cdd (ρ=-0.8534); ext.de_truck_acceleration~ext.forecast_revision_hdd (ρ=0.8534); ext.de_truck_acceleration~ext.freight_eu_z (ρ=1.0); ext.de_truck_acceleration~ext.freight_global_z (ρ=0.9664); ext.de_truck_acceleration~ext.usd_63d_chg (ρ=0.8029); ext.de_truck_acceleration~ext.wti_63d_chg (ρ=-0.8691); ext.de_truck_z_1y~ext.forecast_revision_cdd (ρ=-0.8534); ext.de_truck_z_1y~ext.forecast_revision_hdd (ρ=0.8534); ext.de_truck_z_1y~ext.freight_eu_z (ρ=1.0); ext.de_truck_z_1y~ext.freight_global_z (ρ=0.9664); ext.de_truck_z_1y~ext.usd_63d_chg (ρ=0.8029); ext.de_truck_z_1y~ext.wti_63d_chg (ρ=-0.8691); ext.forecast_revision_cdd~ext.forecast_revision_hdd (ρ=-1.0); ext.forecast_revision_cdd~ext.freight_eu_z (ρ=-0.8534); ext.forecast_revision_cdd~ext.freight_global_z (ρ=-0.8351); ext.forecast_revision_cdd~ext.usd_63d_chg (ρ=-0.8338); ext.forecast_revision_hdd~ext.freight_eu_z (ρ=0.8534); ext.forecast_revision_hdd~ext.freight_global_z (ρ=0.8351); ext.forecast_revision_hdd~ext.usd_63d_chg (ρ=0.8338); ext.freight_eu_z~ext.freight_global_z (ρ=0.9664); ext.freight_eu_z~ext.usd_63d_chg (ρ=0.8029); ext.freight_eu_z~ext.wti_63d_chg (ρ=-0.8691); ext.freight_global_z~ext.freight_us_z (ρ=0.8148); ext.freight_global_z~ext.malacca_z (ρ=0.8148); ext.freight_global_z~ext.us_freight_tsi_yoy (ρ=-0.8148); ext.freight_global_z~ext.us_freight_tsi_z (ρ=0.8148); ext.freight_global_z~ext.usd_63d_chg (ρ=0.8277); ext.freight_global_z~ext.wti_63d_chg (ρ=-0.8087); ext.freight_us_z~ext.us_freight_tsi_yoy (ρ=-1.0); ext.freight_us_z~ext.us_freight_tsi_z (ρ=1.0); ext.hard_data_vs_survey_divergence~ext.malacca_z (ρ=-0.8816); ext.hard_data_vs_survey_divergence~ext.shipping_container_z (ρ=0.8114); ext.hard_data_vs_survey_divergence~ext.us_cpi_yoy (ρ=-1.0); ext.hard_data_vs_survey_divergence~ext.us_cpi_yoy_3m_ago (ρ=1.0); ext.hard_data_vs_survey_divergence~ext.usd_63d_chg (ρ=-0.8326); ext.hard_data_vs_survey_divergence~ext.wti_63d_chg (ρ=0.8254); ext.hormuz_z~ext.panama_z (ρ=-0.8731); ext.hormuz_z~ext.shipping_negative_breadth (ρ=-0.9125); ext.hormuz_z~ext.suez_z (ρ=0.8987); ext.malacca_z~ext.us_cpi_yoy (ρ=0.8816); ext.malacca_z~ext.us_cpi_yoy_3m_ago (ρ=-0.8816); ext.malacca_z~ext.usd_63d_chg (ρ=0.9573); ext.malacca_z~ext.wti_63d_chg (ρ=-0.8615); ext.nfci~ext.nfci_credit (ρ=0.9918); ext.panama_z~ext.shipping_negative_breadth (ρ=0.9103); ext.road_shipping_divergence_z~ext.shipping_global_z (ρ=-0.9717); ext.shipping_container_z~ext.us_cpi_yoy (ρ=-0.8554); ext.shipping_container_z~ext.us_cpi_yoy_3m_ago (ρ=0.8554); ext.shipping_tanker_z~ext.suez_z (ρ=0.8122); ext.us_cpi_yoy~ext.us_cpi_yoy_3m_ago (ρ=-1.0); ext.us_cpi_yoy~ext.usd_63d_chg (ρ=0.8326); ext.us_cpi_yoy~ext.wti_63d_chg (ρ=-0.8254); ext.us_cpi_yoy_3m_ago~ext.usd_63d_chg (ρ=-0.8326); ext.us_cpi_yoy_3m_ago~ext.wti_63d_chg (ρ=0.8254); ext.us_freight_tsi_yoy~ext.us_freight_tsi_z (ρ=-1.0); ext.usd_63d_chg~ext.wti_63d_chg (ρ=-0.8374); insider_buy_count~insider_count (ρ=1.0); log_market_cap~market_cap (ρ=1.0)

## Outcome `gross_20d` (n=3156, unabhängige Tage=9)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|
| ml_enet_xs20_v1 | 2963 | 8 | -0.4189 | [-0.8281, 0.2815] | -0.4148 | -0.0875 |  |
| ml_q90_ret_60 | 2963 | 8 | -0.3469 | [-0.7996, 0.3573] | -0.3496 | -0.0755 | potential_leakage |
| ml_exp_dd_60 | 2963 | 8 | 0.3224 | [-0.3811, 0.7894] | 0.3209 | 0.0686 |  |
| ml_hgb_xs20_risk_regime_v1 | 2960 | 8 | -0.2952 | [-0.7779, 0.4065] | -0.2902 | -0.0698 |  |
| ml_hgb_xs20_v1 | 2963 | 8 | -0.2797 | [-0.7711, 0.4205] | -0.3134 | -0.0646 |  |
| ml_q10_ret_60 | 2963 | 8 | 0.2607 | [-0.4373, 0.7626] | 0.2623 | 0.063 | potential_leakage |
| ml_momentum_12_1 | 2951 | 8 | -0.2347 | [-0.7508, 0.4594] | -0.2536 | -0.05 |  |
| dollar_volume | 1532 | 4 | -0.2277 | [-0.9542, 0.8882] | -0.2277 | -0.0632 |  |
| options_skew_25d | 899 | 8 | 0.2277 | [-0.4651, 0.7476] | 0.2313 | 0.0658 |  |
| ml_p_gt10_60 | 2963 | 8 | -0.2066 | [-0.7376, 0.4823] | -0.1392 | -0.0371 |  |
| avg_volume | 1532 | 4 | -0.1821 | [-0.9497, 0.8978] | -0.1821 | -0.0428 |  |
| insider_sell_count | 936 | 8 | -0.1744 | [-0.722, 0.5076] | -0.1678 | -0.0296 |  |
| log_market_cap | 3003 | 8 | -0.1676 | [-0.7187, 0.5128] | -0.173 | -0.0442 |  |
| ml_hgb_xs20_momentum_v1 | 2960 | 8 | -0.1623 | [-0.716, 0.5168] | -0.1678 | -0.0385 |  |
| market_cap | 1532 | 4 | -0.1593 | [-0.9474, 0.9023] | -0.1593 | -0.0443 |  |
| news_count | 3003 | 8 | -0.1467 | [-0.7081, 0.5284] | -0.1464 | -0.0267 |  |
| ml_exp_ret_60 | 2963 | 8 | 0.1244 | [-0.5446, 0.6966] | 0.122 | 0.0278 | potential_leakage |
| rel_volume | 3003 | 8 | -0.0958 | [-0.6814, 0.5646] | -0.0705 | -0.0221 |  |
| ext.hard_data_vs_survey_divergence | 943 | 9 | -0.0819 | [-0.6373, 0.5295] | -0.1359 | 0.0007 |  |
| ml_hgb_asym20_v1 | 2963 | 8 | 0.0789 | [-0.5761, 0.6722] | 0.0529 | 0.0229 |  |
| insider_count | 979 | 9 | -0.0786 | [-0.6353, 0.5319] | -0.0788 | -0.0026 |  |
| ext.shipping_container_z | 943 | 9 | 0.0726 | [-0.5362, 0.6317] | 0.0837 | 0.0273 |  |
| ext.shipping_drybulk_z | 943 | 9 | -0.0726 | [-0.6317, 0.5362] | -0.0837 | -0.0131 |  |
| dealer_oi_ratio | 905 | 9 | -0.0716 | [-0.6311, 0.5369] | -0.0935 | -0.0254 |  |
| ext.cdd_anomaly | 943 | 9 | 0.0705 | [-0.5378, 0.6304] | 0.0708 | 0.0223 |  |
| insider_cluster | 979 | 9 | -0.0702 | [-0.6303, 0.5379] | -0.1193 | -0.0009 |  |
| ext.hdd_anomaly | 943 | 9 | -0.0638 | [-0.6264, 0.5425] | -0.1019 | -0.0129 |  |
| scan_day_ret | 3003 | 8 | -0.0552 | [-0.6589, 0.5918] | -0.084 | -0.0136 | potential_leakage |
| ext.nfci_credit | 902 | 8 | 0.054 | [-0.5926, 0.6582] | 0.0737 | 0.008 |  |
| fda_headline_count | 979 | 9 | 0.0496 | [-0.5525, 0.6177] | 0.0551 | 0.0111 |  |
| insider_buy_count | 936 | 8 | -0.0468 | [-0.6541, 0.5972] | -0.0299 | 0.0033 |  |
| ext.nfci | 902 | 8 | 0.0409 | [-0.601, 0.6507] | 0.0244 | 0.008 |  |
| has_near_earnings | 979 | 9 | 0.0391 | [-0.5597, 0.6111] | 0.0558 | 0.0158 |  |
| ml_disagreement_sd | 2963 | 8 | -0.0385 | [-0.6494, 0.6026] | -0.0168 | -0.0086 |  |
| sentiment_score | 3003 | 8 | 0.0383 | [-0.6027, 0.6492] | 0.0171 | 0.0073 |  |
| ext.shipping_negative_breadth | 943 | 9 | -0.0322 | [-0.6067, 0.5645] | -0.0045 | 0.0035 |  |
| sentiment_confidence | 3003 | 8 | 0.0295 | [-0.6082, 0.6441] | 0.0332 | 0.0044 |  |
| ext.freight_global_z | 943 | 9 | 0.0258 | [-0.5688, 0.6027] | -0.0146 | 0.0164 |  |
| ext.de_truck_acceleration | 943 | 9 | 0.0249 | [-0.5694, 0.6021] | 0.004 | 0.0164 |  |
| ext.de_truck_z_1y | 943 | 9 | 0.0249 | [-0.5694, 0.6021] | 0.004 | 0.0164 |  |
| ext.freight_eu_z | 943 | 9 | 0.0249 | [-0.5694, 0.6021] | 0.004 | 0.0164 |  |
| short_pct_float | 2969 | 8 | 0.0228 | [-0.6125, 0.6402] | 0.0226 | 0.0104 |  |
| short_mom | 2994 | 8 | 0.0208 | [-0.6137, 0.639] | 0.0282 | 0.0052 |  |
| ext.freight_us_z | 943 | 9 | 0.0204 | [-0.5725, 0.5992] | -0.038 | 0.0164 |  |
| ext.us_freight_tsi_yoy | 943 | 9 | -0.0204 | [-0.5992, 0.5725] | 0.038 | 0.0106 |  |
| ext.us_freight_tsi_z | 943 | 9 | 0.0204 | [-0.5725, 0.5992] | -0.038 | 0.0164 |  |
| sentiment_drift | 3003 | 8 | 0.0195 | [-0.6145, 0.6382] | 0.0148 | 0.0069 |  |
| ext.fed_assets_13w_chg | 902 | 8 | -0.0178 | [-0.6372, 0.6156] | -0.0393 | 0.001 |  |
| ext.shipping_tanker_z | 943 | 9 | 0.0162 | [-0.5753, 0.5965] | 0.0324 | 0.0019 |  |
| earnings_known | 936 | 8 | 0.0153 | [-0.6171, 0.6357] | 0.0138 | 0.0053 |  |
| options_skew_ratio | 979 | 9 | 0.015 | [-0.5761, 0.5957] | 0.0441 | 0.0044 |  |
| ext.asia_export_z | 943 | 9 | -0.0136 | [-0.5948, 0.577] | 0.0366 | -0.0066 |  |
| ext.shipping_global_z | 943 | 9 | -0.0136 | [-0.5948, 0.577] | 0.0366 | -0.0066 |  |
| ext.road_shipping_divergence_z | 943 | 9 | 0.0087 | [-0.5803, 0.5917] | -0.0444 | 0.012 |  |
| ext.forecast_revision_cdd | 943 | 9 | -0.0063 | [-0.5901, 0.5819] | 0.004 | 0.0059 |  |
| ext.forecast_revision_hdd | 943 | 9 | 0.0063 | [-0.5819, 0.5901] | -0.004 | 0.0016 |  |
| ext.de_power_price_7d | 902 | 8 | -0.0049 | [-0.6295, 0.6236] | -0.0308 | -0.0088 |  |
| ext.de_power_price_z | 902 | 8 | -0.0049 | [-0.6295, 0.6236] | -0.0308 | -0.0088 |  |
| ext.wti_63d_chg | 902 | 8 | 0.0047 | [-0.6237, 0.6294] | -0.004 | -0.011 |  |
| ext.de_load_yoy | 902 | 8 | 0.0018 | [-0.6254, 0.6276] | 0.0334 | -0.0069 |  |
| ext.weather_disruption_index | 902 | 8 | -0.0009 | [-0.6271, 0.626] | -0.0079 | 0.004 |  |
| ext.bab_el_mandeb_z | 902 | 8 | 0.0003 | [-0.6264, 0.6267] | -0.0249 | 0.008 |  |
| ext.hormuz_z | 902 | 8 | 0.0003 | [-0.6264, 0.6267] | -0.0249 | 0.008 |  |
| ext.malacca_z | 902 | 8 | 0.0003 | [-0.6264, 0.6267] | -0.0249 | 0.008 |  |
| ext.panama_z | 902 | 8 | -0.0003 | [-0.6267, 0.6264] | 0.0249 | 0.0147 |  |
| ext.suez_z | 902 | 8 | 0.0003 | [-0.6264, 0.6267] | -0.0249 | 0.008 |  |
| ext.usd_63d_chg | 902 | 8 | 0.0003 | [-0.6264, 0.6267] | -0.0249 | 0.008 |  |
| ext.active_tropical_system | 943 | 9 | None | None | None | None | dead |
| ext.freight_asia_z | 943 | 9 | None | None | None | None | dead |
| ext.freight_breadth | 943 | 9 | None | None | None | None | dead |
| ext.port_disruption_active_events | 943 | 9 | None | None | None | None | dead |
| ext.port_disruption_curated_max_level | 943 | 9 | None | None | None | None | dead |
| ext.port_disruption_curated_ports | 943 | 9 | None | None | None | None | dead |
| ext.possible_weather_confounded_freight | 902 | 8 | None | None | None | None | dead |
| ext.shipping_valid_port_count | 943 | 9 | None | None | None | None | dead |
| ext.us_cpi_yoy | 902 | 8 | None | None | None | None | dead |
| ext.us_cpi_yoy_3m_ago | 902 | 8 | None | None | None | None | dead |
| insider_data_available | 936 | 8 | None | None | None | None | dead |

Walk-Forward OOS: adaptiv {'n': 0} vs. gleichgewichtet {'n': 0}

Redundant: dollar_volume~log_market_cap (ρ=0.8848); dollar_volume~market_cap (ρ=0.8848); ext.asia_export_z~ext.nfci (ρ=-0.8019); ext.asia_export_z~ext.nfci_credit (ρ=-0.8498); ext.asia_export_z~ext.road_shipping_divergence_z (ρ=-0.9482); ext.asia_export_z~ext.shipping_global_z (ρ=1.0); ext.bab_el_mandeb_z~ext.de_load_yoy (ρ=0.8729); ext.bab_el_mandeb_z~ext.freight_global_z (ρ=0.8157); ext.bab_el_mandeb_z~ext.hormuz_z (ρ=1.0); ext.bab_el_mandeb_z~ext.malacca_z (ρ=1.0); ext.bab_el_mandeb_z~ext.nfci (ρ=0.8208); ext.bab_el_mandeb_z~ext.nfci_credit (ρ=0.8005); ext.bab_el_mandeb_z~ext.panama_z (ρ=-1.0); ext.bab_el_mandeb_z~ext.shipping_negative_breadth (ρ=-0.9583); ext.bab_el_mandeb_z~ext.shipping_tanker_z (ρ=0.9583); ext.bab_el_mandeb_z~ext.suez_z (ρ=1.0); ext.bab_el_mandeb_z~ext.usd_63d_chg (ρ=1.0); ext.de_load_yoy~ext.freight_us_z (ρ=0.8416); ext.de_load_yoy~ext.hormuz_z (ρ=0.8729); ext.de_load_yoy~ext.malacca_z (ρ=0.8729); ext.de_load_yoy~ext.panama_z (ρ=-0.8729); ext.de_load_yoy~ext.shipping_tanker_z (ρ=0.9109); ext.de_load_yoy~ext.suez_z (ρ=0.8729); ext.de_load_yoy~ext.us_freight_tsi_yoy (ρ=-0.8416); ext.de_load_yoy~ext.us_freight_tsi_z (ρ=0.8416); ext.de_load_yoy~ext.usd_63d_chg (ρ=0.8729); ext.de_power_price_7d~ext.de_power_price_z (ρ=1.0); ext.de_power_price_7d~ext.wti_63d_chg (ρ=0.8298); ext.de_power_price_z~ext.wti_63d_chg (ρ=0.8298); ext.de_truck_acceleration~ext.de_truck_z_1y (ρ=1.0); ext.de_truck_acceleration~ext.freight_eu_z (ρ=1.0); ext.de_truck_acceleration~ext.freight_global_z (ρ=0.8157); ext.de_truck_acceleration~ext.wti_63d_chg (ρ=-0.8443); ext.de_truck_z_1y~ext.freight_eu_z (ρ=1.0); ext.de_truck_z_1y~ext.freight_global_z (ρ=0.8157); ext.de_truck_z_1y~ext.wti_63d_chg (ρ=-0.8443); ext.forecast_revision_cdd~ext.forecast_revision_hdd (ρ=-1.0); ext.freight_eu_z~ext.freight_global_z (ρ=0.8157); ext.freight_eu_z~ext.wti_63d_chg (ρ=-0.8443); ext.freight_global_z~ext.freight_us_z (ρ=0.902); ext.freight_global_z~ext.hormuz_z (ρ=0.8157); ext.freight_global_z~ext.malacca_z (ρ=0.8157); ext.freight_global_z~ext.nfci (ρ=0.8525); ext.freight_global_z~ext.nfci_credit (ρ=0.8149); ext.freight_global_z~ext.panama_z (ρ=-0.8157); ext.freight_global_z~ext.shipping_negative_breadth (ρ=-0.8554); ext.freight_global_z~ext.suez_z (ρ=0.8157); ext.freight_global_z~ext.us_freight_tsi_yoy (ρ=-0.902); ext.freight_global_z~ext.us_freight_tsi_z (ρ=0.902); ext.freight_global_z~ext.usd_63d_chg (ρ=0.8157); ext.freight_us_z~ext.us_freight_tsi_yoy (ρ=-1.0); ext.freight_us_z~ext.us_freight_tsi_z (ρ=1.0); ext.hard_data_vs_survey_divergence~insider_cluster (ρ=0.9742); ext.hard_data_vs_survey_divergence~insider_count (ρ=0.869); ext.hormuz_z~ext.malacca_z (ρ=1.0); ext.hormuz_z~ext.nfci (ρ=0.8208); ext.hormuz_z~ext.nfci_credit (ρ=0.8005); ext.hormuz_z~ext.panama_z (ρ=-1.0); ext.hormuz_z~ext.shipping_negative_breadth (ρ=-0.9583); ext.hormuz_z~ext.shipping_tanker_z (ρ=0.9583); ext.hormuz_z~ext.suez_z (ρ=1.0); ext.hormuz_z~ext.usd_63d_chg (ρ=1.0); ext.malacca_z~ext.nfci (ρ=0.8208); ext.malacca_z~ext.nfci_credit (ρ=0.8005); ext.malacca_z~ext.panama_z (ρ=-1.0); ext.malacca_z~ext.shipping_negative_breadth (ρ=-0.9583); ext.malacca_z~ext.shipping_tanker_z (ρ=0.9583); ext.malacca_z~ext.suez_z (ρ=1.0); ext.malacca_z~ext.usd_63d_chg (ρ=1.0); ext.nfci~ext.nfci_credit (ρ=0.9454); ext.nfci~ext.panama_z (ρ=-0.8208); ext.nfci~ext.road_shipping_divergence_z (ρ=0.8095); ext.nfci~ext.shipping_container_z (ρ=0.8019); ext.nfci~ext.shipping_drybulk_z (ρ=-0.8019); ext.nfci~ext.shipping_global_z (ρ=-0.8019); ext.nfci~ext.shipping_negative_breadth (ρ=-0.9065); ext.nfci~ext.suez_z (ρ=0.8208); ext.nfci~ext.usd_63d_chg (ρ=0.8208); ext.nfci_credit~ext.panama_z (ρ=-0.8005); ext.nfci_credit~ext.road_shipping_divergence_z (ρ=0.8294); ext.nfci_credit~ext.shipping_container_z (ρ=0.8498); ext.nfci_credit~ext.shipping_drybulk_z (ρ=-0.8498); ext.nfci_credit~ext.shipping_global_z (ρ=-0.8498); ext.nfci_credit~ext.shipping_negative_breadth (ρ=-0.9073); ext.nfci_credit~ext.suez_z (ρ=0.8005); ext.nfci_credit~ext.usd_63d_chg (ρ=0.8005); ext.panama_z~ext.shipping_negative_breadth (ρ=0.9583); ext.panama_z~ext.shipping_tanker_z (ρ=-0.9583); ext.panama_z~ext.suez_z (ρ=-1.0); ext.panama_z~ext.usd_63d_chg (ρ=-1.0); ext.road_shipping_divergence_z~ext.shipping_global_z (ρ=-0.9482); ext.shipping_container_z~ext.shipping_drybulk_z (ρ=-1.0); ext.shipping_negative_breadth~ext.shipping_tanker_z (ρ=-0.8394); ext.shipping_negative_breadth~ext.suez_z (ρ=-0.9583); ext.shipping_negative_breadth~ext.usd_63d_chg (ρ=-0.9583); ext.shipping_tanker_z~ext.suez_z (ρ=0.9583); ext.shipping_tanker_z~ext.usd_63d_chg (ρ=0.9583); ext.suez_z~ext.usd_63d_chg (ρ=1.0); ext.us_freight_tsi_yoy~ext.us_freight_tsi_z (ρ=-1.0); insider_buy_count~insider_count (ρ=1.0); insider_cluster~insider_count (ρ=0.8412); log_market_cap~market_cap (ρ=1.0)

## Outcome `gross_5d` (n=5890, unabhängige Tage=17)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|
| ml_p_gt10_60 | 5618 | 16 | -0.1247 | [-0.5238, 0.3193] | -0.0434 | -0.0097 | decaying |
| ml_enet_xs20_v1 | 5618 | 16 | -0.1019 | [-0.5068, 0.3399] | 0.0323 | -0.0062 | decaying |
| dollar_volume | 4266 | 12 | -0.0877 | [-0.5623, 0.4304] | -0.0877 | -0.0094 |  |
| ml_q90_ret_60 | 5618 | 16 | -0.0696 | [-0.4823, 0.3684] | 0.066 | -0.0045 | potential_leakage, decaying |
| ext.cdd_anomaly | 1583 | 17 | 0.0684 | [-0.3549, 0.4685] | 0.0487 | 0.0058 |  |
| avg_volume | 4266 | 12 | -0.0682 | [-0.5488, 0.4463] | -0.0682 | -0.007 |  |
| ext.shipping_drybulk_z | 1583 | 17 | -0.0637 | [-0.4648, 0.3591] | -0.0474 | -0.0004 |  |
| ext.panama_z | 1542 | 16 | 0.063 | [-0.3741, 0.4772] | 0.0846 | 0.0107 |  |
| market_cap | 4266 | 12 | -0.0567 | [-0.5407, 0.4555] | -0.0567 | -0.0062 |  |
| ml_exp_dd_60 | 5618 | 16 | 0.0556 | [-0.3804, 0.4714] | -0.0583 | 0.004 | decaying |
| insider_count | 1874 | 17 | -0.0544 | [-0.4575, 0.3672] | -0.0543 | -0.0043 |  |
| ml_q10_ret_60 | 5618 | 16 | 0.0523 | [-0.3833, 0.4688] | -0.0559 | 0.0039 | potential_leakage, decaying |
| insider_buy_count | 1831 | 16 | -0.0506 | [-0.4675, 0.3847] | -0.0482 | -0.0034 |  |
| ext.bab_el_mandeb_z | 1542 | 16 | -0.0502 | [-0.4672, 0.385] | -0.0996 | -0.0065 |  |
| ext.hormuz_z | 1542 | 16 | -0.0502 | [-0.4672, 0.385] | -0.0996 | -0.0065 |  |
| ext.de_load_yoy | 1542 | 16 | -0.0389 | [-0.4583, 0.3946] | -0.0309 | -0.0065 |  |
| options_skew_25d | 1784 | 16 | 0.0381 | [-0.3953, 0.4577] | -0.019 | 0.0043 |  |
| sentiment_score | 5737 | 16 | 0.0375 | [-0.3958, 0.4572] | 0.0383 | 0.0029 |  |
| ml_hgb_xs20_risk_regime_v1 | 5615 | 16 | -0.0364 | [-0.4563, 0.3968] | 0.1067 | -0.0035 |  |
| ext.shipping_negative_breadth | 1583 | 17 | 0.0351 | [-0.3838, 0.442] | 0.0702 | 0.0083 |  |
| scan_day_ret | 5737 | 16 | -0.0347 | [-0.4549, 0.3982] | -0.0013 | -0.0036 | potential_leakage |
| log_market_cap | 5737 | 16 | -0.0342 | [-0.4546, 0.3986] | -0.0046 | -0.0049 |  |
| ext.hdd_anomaly | 1583 | 17 | -0.0332 | [-0.4405, 0.3854] | -0.0429 | -0.001 |  |
| sentiment_drift | 5737 | 16 | 0.0312 | [-0.4012, 0.4522] | 0.0227 | 0.0032 |  |
| insider_cluster | 1874 | 17 | -0.0308 | [-0.4386, 0.3874] | -0.0382 | -0.0026 |  |
| news_count | 5737 | 16 | -0.0296 | [-0.4509, 0.4025] | 0.0045 | -0.0024 |  |
| insider_sell_count | 1831 | 16 | -0.0264 | [-0.4484, 0.4052] | 0.0076 | -0.0038 |  |
| has_near_earnings | 1874 | 17 | -0.0263 | [-0.4349, 0.3913] | 0.0256 | -0.0018 |  |
| ext.suez_z | 1542 | 16 | -0.0256 | [-0.4477, 0.4058] | -0.0678 | -0.0054 |  |
| ext.malacca_z | 1542 | 16 | -0.0233 | [-0.4459, 0.4078] | -0.0627 | -0.0011 |  |
| ext.hard_data_vs_survey_divergence | 1583 | 17 | -0.0231 | [-0.4323, 0.394] | -0.0192 | -0.003 |  |
| ext.active_tropical_system | 1583 | 17 | 0.0209 | [-0.3959, 0.4305] | 0.0325 | 0.0035 |  |
| ext.wti_63d_chg | 1542 | 16 | 0.0209 | [-0.4097, 0.444] | 0.0341 | 0.0017 |  |
| dealer_oi_ratio | 1743 | 17 | 0.0202 | [-0.3965, 0.4299] | 0.0376 | 0.0001 |  |
| ext.nfci | 1542 | 16 | 0.0202 | [-0.4103, 0.4434] | -0.0256 | -0.0016 |  |
| ml_hgb_xs20_momentum_v1 | 5615 | 16 | -0.0177 | [-0.4414, 0.4125] | 0.0394 | -0.0017 |  |
| ext.de_power_price_7d | 1542 | 16 | 0.0174 | [-0.4126, 0.4412] | -0.0193 | -0.0006 |  |
| ext.shipping_container_z | 1583 | 17 | 0.0173 | [-0.3989, 0.4276] | 0.0369 | -0.0007 |  |
| rel_volume | 5737 | 16 | -0.0173 | [-0.441, 0.4128] | -0.0124 | -0.0009 |  |
| sentiment_confidence | 5737 | 16 | 0.0168 | [-0.4131, 0.4407] | 0.0112 | 0.0029 |  |
| ext.usd_63d_chg | 1542 | 16 | -0.0165 | [-0.4404, 0.4134] | -0.0531 | -0.0011 |  |
| ext.nfci_credit | 1542 | 16 | 0.0161 | [-0.4138, 0.4401] | 0.0125 | -0.0016 |  |
| ext.weather_disruption_index | 1542 | 16 | 0.0139 | [-0.4156, 0.4383] | 0.0061 | 0.0007 |  |
| fda_headline_count | 1874 | 17 | -0.0139 | [-0.4248, 0.4018] | -0.027 | -0.0018 |  |
| ext.de_power_price_z | 1542 | 16 | 0.0132 | [-0.4161, 0.4378] | -0.0144 | -0.0006 |  |
| ext.forecast_revision_cdd | 1583 | 17 | -0.0131 | [-0.4242, 0.4024] | 0.0095 | 0.0012 |  |
| ext.forecast_revision_hdd | 1583 | 17 | 0.0131 | [-0.4024, 0.4242] | -0.0095 | -0.002 |  |
| ml_hgb_asym20_v1 | 5618 | 16 | 0.0131 | [-0.4162, 0.4377] | 0.0127 | 0.0006 |  |
| earnings_known | 1831 | 16 | 0.0129 | [-0.4164, 0.4375] | 0.0063 | -0.0021 |  |
| ml_exp_ret_60 | 5618 | 16 | -0.0115 | [-0.4364, 0.4176] | 0.0338 | -0.001 | potential_leakage |
| ext.de_truck_acceleration | 1583 | 17 | 0.0106 | [-0.4045, 0.4221] | -0.0313 | -0.0008 |  |
| ext.de_truck_z_1y | 1583 | 17 | 0.0106 | [-0.4045, 0.4221] | -0.0313 | -0.0008 |  |
| ext.freight_eu_z | 1583 | 17 | 0.0106 | [-0.4045, 0.4221] | -0.0313 | -0.0008 |  |
| options_skew_ratio | 1874 | 17 | 0.01 | [-0.4051, 0.4216] | 0.0254 | 0.0011 |  |
| ext.freight_us_z | 1583 | 17 | -0.0084 | [-0.4203, 0.4064] | -0.1168 | -0.0008 |  |
| ext.us_freight_tsi_yoy | 1583 | 17 | 0.0084 | [-0.4064, 0.4203] | 0.1168 | 0.0038 |  |
| ext.us_freight_tsi_z | 1583 | 17 | -0.0084 | [-0.4203, 0.4064] | -0.1168 | -0.0008 |  |
| ext.road_shipping_divergence_z | 1583 | 17 | -0.0078 | [-0.4198, 0.4069] | -0.0446 | -0.0009 |  |
| ext.us_cpi_yoy | 1542 | 16 | 0.0076 | [-0.4208, 0.4332] | -0.0118 | -0.0011 |  |
| ext.us_cpi_yoy_3m_ago | 1542 | 16 | -0.0076 | [-0.4332, 0.4208] | 0.0118 | -0.0032 |  |
| short_pct_float | 5680 | 16 | -0.0072 | [-0.4329, 0.4211] | 0.002 | 0.0013 |  |
| ext.freight_global_z | 1583 | 17 | 0.0047 | [-0.4094, 0.4173] | -0.0517 | -0.0008 |  |
| ml_momentum_12_1 | 5596 | 16 | 0.0031 | [-0.4245, 0.4295] | 0.1025 | 0.0002 |  |
| ext.shipping_global_z | 1583 | 17 | 0.0024 | [-0.4114, 0.4153] | 0.0361 | 0.0036 |  |
| ml_disagreement_sd | 5618 | 16 | 0.0021 | [-0.4253, 0.4287] | 0.0081 | -0.0013 |  |
| ml_hgb_xs20_v1 | 5618 | 16 | -0.0021 | [-0.4288, 0.4253] | 0.1138 | -0.0008 |  |
| ext.fed_assets_13w_chg | 1542 | 16 | 0.0019 | [-0.4255, 0.4285] | -0.0179 | 0.0009 |  |
| short_mom | 5723 | 16 | -0.0015 | [-0.4282, 0.4258] | 0.0111 | -0.0003 |  |
| ext.shipping_tanker_z | 1583 | 17 | -0.0009 | [-0.4141, 0.4126] | -0.0007 | -0.0016 |  |
| ext.asia_export_z | 1583 | 17 | 0.0003 | [-0.4131, 0.4136] | 0.0145 | 0.0035 |  |
| ext.freight_asia_z | 1583 | 17 | None | None | None | None | dead |
| ext.freight_breadth | 1583 | 17 | None | None | None | None | dead |
| ext.port_disruption_active_events | 1583 | 17 | None | None | None | None | dead |
| ext.port_disruption_curated_max_level | 1583 | 17 | None | None | None | None | dead |
| ext.port_disruption_curated_ports | 1583 | 17 | None | None | None | None | dead |
| ext.possible_weather_confounded_freight | 1542 | 16 | None | None | None | None | dead |
| ext.shipping_valid_port_count | 1583 | 17 | None | None | None | None | dead |
| insider_data_available | 1831 | 16 | None | None | None | None | dead |

Walk-Forward OOS: adaptiv {'n': 0} vs. gleichgewichtet {'n': 0}

Redundant: dollar_volume~log_market_cap (ρ=0.871); dollar_volume~market_cap (ρ=0.871); ext.asia_export_z~ext.hard_data_vs_survey_divergence (ρ=-0.8523); ext.asia_export_z~ext.shipping_container_z (ρ=-0.9457); ext.asia_export_z~ext.us_cpi_yoy (ρ=0.8382); ext.asia_export_z~ext.us_cpi_yoy_3m_ago (ρ=-0.8382); ext.bab_el_mandeb_z~ext.hormuz_z (ρ=1.0); ext.bab_el_mandeb_z~ext.panama_z (ρ=-0.8459); ext.bab_el_mandeb_z~ext.shipping_negative_breadth (ρ=-0.9068); ext.bab_el_mandeb_z~ext.suez_z (ρ=0.924); ext.de_power_price_7d~ext.de_power_price_z (ρ=0.9734); ext.de_truck_acceleration~ext.de_truck_z_1y (ρ=1.0); ext.de_truck_acceleration~ext.forecast_revision_cdd (ρ=-0.8609); ext.de_truck_acceleration~ext.forecast_revision_hdd (ρ=0.8609); ext.de_truck_acceleration~ext.freight_eu_z (ρ=1.0); ext.de_truck_acceleration~ext.freight_global_z (ρ=0.9603); ext.de_truck_acceleration~ext.usd_63d_chg (ρ=0.8063); ext.de_truck_acceleration~ext.wti_63d_chg (ρ=-0.8815); ext.de_truck_z_1y~ext.forecast_revision_cdd (ρ=-0.8609); ext.de_truck_z_1y~ext.forecast_revision_hdd (ρ=0.8609); ext.de_truck_z_1y~ext.freight_eu_z (ρ=1.0); ext.de_truck_z_1y~ext.freight_global_z (ρ=0.9603); ext.de_truck_z_1y~ext.usd_63d_chg (ρ=0.8063); ext.de_truck_z_1y~ext.wti_63d_chg (ρ=-0.8815); ext.forecast_revision_cdd~ext.forecast_revision_hdd (ρ=-1.0); ext.forecast_revision_cdd~ext.freight_eu_z (ρ=-0.8609); ext.forecast_revision_cdd~ext.freight_global_z (ρ=-0.8393); ext.forecast_revision_cdd~ext.malacca_z (ρ=-0.8013); ext.forecast_revision_cdd~ext.usd_63d_chg (ρ=-0.8499); ext.forecast_revision_hdd~ext.freight_eu_z (ρ=0.8609); ext.forecast_revision_hdd~ext.freight_global_z (ρ=0.8393); ext.forecast_revision_hdd~ext.malacca_z (ρ=0.8013); ext.forecast_revision_hdd~ext.usd_63d_chg (ρ=0.8499); ext.freight_eu_z~ext.freight_global_z (ρ=0.9603); ext.freight_eu_z~ext.usd_63d_chg (ρ=0.8063); ext.freight_eu_z~ext.wti_63d_chg (ρ=-0.8815); ext.freight_global_z~ext.freight_us_z (ρ=0.8208); ext.freight_global_z~ext.malacca_z (ρ=0.8207); ext.freight_global_z~ext.us_freight_tsi_yoy (ρ=-0.8208); ext.freight_global_z~ext.us_freight_tsi_z (ρ=0.8208); ext.freight_global_z~ext.usd_63d_chg (ρ=0.8365); ext.freight_global_z~ext.wti_63d_chg (ρ=-0.809); ext.freight_us_z~ext.us_freight_tsi_yoy (ρ=-1.0); ext.freight_us_z~ext.us_freight_tsi_z (ρ=1.0); ext.hard_data_vs_survey_divergence~ext.malacca_z (ρ=-0.8664); ext.hard_data_vs_survey_divergence~ext.us_cpi_yoy (ρ=-1.0); ext.hard_data_vs_survey_divergence~ext.us_cpi_yoy_3m_ago (ρ=1.0); ext.hard_data_vs_survey_divergence~ext.usd_63d_chg (ρ=-0.8249); ext.hard_data_vs_survey_divergence~ext.wti_63d_chg (ρ=0.803); ext.hormuz_z~ext.panama_z (ρ=-0.8459); ext.hormuz_z~ext.shipping_negative_breadth (ρ=-0.9068); ext.hormuz_z~ext.suez_z (ρ=0.924); ext.malacca_z~ext.us_cpi_yoy (ρ=0.8664); ext.malacca_z~ext.us_cpi_yoy_3m_ago (ρ=-0.8664); ext.malacca_z~ext.usd_63d_chg (ρ=0.9774); ext.malacca_z~ext.wti_63d_chg (ρ=-0.8358); ext.nfci~ext.nfci_credit (ρ=0.99); ext.nfci~ext.road_shipping_divergence_z (ρ=0.8216); ext.nfci_credit~ext.road_shipping_divergence_z (ρ=0.8241); ext.panama_z~ext.shipping_negative_breadth (ρ=0.9068); ext.road_shipping_divergence_z~ext.shipping_global_z (ρ=-0.9686); ext.shipping_container_z~ext.us_cpi_yoy (ρ=-0.8382); ext.shipping_container_z~ext.us_cpi_yoy_3m_ago (ρ=0.8382); ext.shipping_tanker_z~ext.suez_z (ρ=0.8613); ext.us_cpi_yoy~ext.us_cpi_yoy_3m_ago (ρ=-1.0); ext.us_cpi_yoy~ext.usd_63d_chg (ρ=0.8249); ext.us_cpi_yoy~ext.wti_63d_chg (ρ=-0.803); ext.us_cpi_yoy_3m_ago~ext.usd_63d_chg (ρ=-0.8249); ext.us_cpi_yoy_3m_ago~ext.wti_63d_chg (ρ=0.803); ext.us_freight_tsi_yoy~ext.us_freight_tsi_z (ρ=-1.0); ext.usd_63d_chg~ext.wti_63d_chg (ρ=-0.8417); insider_buy_count~insider_count (ρ=1.0); log_market_cap~market_cap (ρ=1.0)

## Outcome `net` (n=117, unabhängige Tage=40)

| Feature | n | Tage | Rank-IC | 90%-KI | EWMA-IC | Terzil-Spread | Tags |
|---|---|---|---|---|---|---|---|
| price_move_48h | 61 | 29 | 0.222 | [-0.0965, 0.4993] | -0.0457 | 0.6024 | decaying |
| sigma_30d | 117 | 40 | 0.1182 | [-0.1506, 0.3706] | 0.1242 | 0.856 |  |
| iv_rank | 59 | 26 | 0.1035 | [-0.2347, 0.4193] | 0.0353 | 0.2647 | decaying |
| quick_mc_hit_rate | 109 | 38 | 0.0914 | [-0.1842, 0.3538] | 0.0589 | 0.6406 |  |
| surprise | 117 | 40 | -0.0789 | [-0.3359, 0.1891] | -0.0139 | -0.151 | decaying |
| mismatch | 117 | 40 | -0.0747 | [-0.3321, 0.1932] | -0.2217 | 0.215 |  |
| z_score | 117 | 40 | 0.0426 | [-0.224, 0.3032] | 0.1629 | -0.2416 |  |
| impact | 117 | 40 | -0.0329 | [-0.2943, 0.2332] | -0.002 | 0.1332 |  |
| eps_drift | 117 | 40 | 0.0077 | [-0.2568, 0.2712] | 0.0222 | -0.5426 |  |

Walk-Forward OOS: adaptiv {'n': 0} vs. gleichgewichtet {'n': 0}

Redundant: mismatch~price_move_48h (ρ=-0.804); mismatch~z_score (ρ=-0.9313); price_move_48h~z_score (ρ=0.8998)

## LLM-Richtung vs. naive Preis-Baseline (gepaart)

| h | n | Tage | LLM-Mittel | Baseline-Mittel | Diff/Tag | t | Urteil |
|---|---|---|---|---|---|---|---|
| h5 | 922 | 15 | 0.0029 | 0.0009 | 0.0028 | 1.0343 | insufficient_data |
| h20 | 441 | 7 | -0.0076 | 0.0028 | -0.0109 | -1.2722 | insufficient_data |
| h45 | 0 | 0 | None | None | None | None | insufficient_data |

