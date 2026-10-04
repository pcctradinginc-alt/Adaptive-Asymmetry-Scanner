# Feature Source Dependencies (generiert: python -m modules.source_health deps)

Fällt eine Quelle aus (SCHEMA_CHANGED/BROKEN, STALE, UNVALIDATED), sind ihre Features **unavailable**
(NaN, Verfügbarkeit 0 – nie alte Werte, nie 0), Data Quality/Confidence sinken, und produktive
Signale, die zwingend davon abhängen, werden blockiert oder laufen über einen explizit getesteten Fallback.

| Feature | Quelle(n) | Fallback | Entscheidungspfade der Quelle | Modelle/Hypothesen der Quelle |
|---|---|---|---|---|
| beta_126 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| cmdx_copper__copper_ret_3m | fred_commodities | – | – | – |
| cmdx_copper__cot_copper_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_copper__cot_copper_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_copper__div_copper_positioning_price | cftc_cot, fred_commodities | – | – | – |
| cmdx_corn__corn_ret_3m | fred_commodities | – | – | – |
| cmdx_corn__cot_corn_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_corn__cot_corn_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_gold__cot_gold_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_gold__cot_gold_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_natural_gas__cot_natural_gas_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_natural_gas__cot_natural_gas_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_natural_gas__div_gas_storage_price | eia_natural_gas, fred_commodities | – | – | – |
| cmdx_natural_gas__henry_hub_ret_20d | fred_commodities | – | – | – |
| cmdx_natural_gas__henry_hub_ret_60d | fred_commodities | – | – | – |
| cmdx_natural_gas__lng_exports_yoy | eia_natural_gas | – | – | – |
| cmdx_natural_gas__natgas_storage_chg_z_52w | eia_natural_gas | – | – | – |
| cmdx_natural_gas__natgas_storage_vs_5y | eia_natural_gas | – | – | – |
| cmdx_oil__cot_crude_oil_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_oil__cot_crude_oil_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_oil__crude_stocks_chg_z_52w | eia_petroleum_weekly | – | – | – |
| cmdx_oil__crude_stocks_vs_5y | eia_petroleum_weekly | – | – | – |
| cmdx_oil__div_oil_positioning_price | cftc_cot, fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| cmdx_oil__div_oil_price_inventory | eia_petroleum_weekly, fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| cmdx_oil__product_supplied_chg_4w | eia_petroleum_weekly | – | – | – |
| cmdx_oil__refinery_utilization_z_52w | eia_petroleum_weekly | – | – | – |
| cmdx_oil__wti_ret_20d | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| cmdx_oil__wti_ret_60d | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| cmdx_oil__wti_z_60d | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| cmdx_silver__cot_silver_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_silver__cot_silver_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_soybeans__cot_soybeans_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_soybeans__cot_soybeans_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_soybeans__soybeans_ret_3m | fred_commodities | – | – | – |
| cmdx_wheat__cot_wheat_mm_net_chg_4w | cftc_cot | – | – | – |
| cmdx_wheat__cot_wheat_mm_pctile_1y | cftc_cot | – | – | – |
| cmdx_wheat__wheat_ret_3m | fred_commodities | – | – | – |
| cpi_yoy | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| curve_10y_3m | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| dist_52w_high | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| fed_assets_13w_chg | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| finbert_sentiment | news_finnhub | – | candidate_discovery | – |
| implied_vol | options_chain, options_chain_yfinance | options_chain_yfinance | options_design, roi_precheck | – |
| log_dollar_vol | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| max_ret_21 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| mom_12_1 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| mom_3m | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| news_flow | news_finnhub | – | candidate_discovery | – |
| option_quotes | options_chain, options_chain_yfinance | options_chain_yfinance | options_design, roi_precheck | – |
| price_history | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| realized_vol | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| relvol_5_60 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| ret_5d | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| rev_1m | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| rs_63 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| sec_8k_count_30d_z | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_8k_negative_90d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_exec_change_90d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_filing_delay_z | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_insider_buy_value_90d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_insider_buyers_90d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_insider_cluster_30d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_insider_net_value_90d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| sec_late_filing_365d | sec_form345, sec_submissions | – | – | ALT-SEC-001, ALT-SEC-002, ALT-SEC-003, ALT-SEC-004 |
| spy_mom_63 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| spy_trend_200 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| ted_any_award_365d | ted_awards | – | – | ALT-TED-001, ALT-TED-002 |
| ted_award_value_365d | ted_awards | – | – | ALT-TED-001, ALT-TED-002 |
| ted_awards_90d | ted_awards | – | – | ALT-TED-001, ALT-TED-002 |
| ted_awards_z | ted_awards | – | – | ALT-TED-001, ALT-TED-002 |
| tnx | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| usd_63d_chg | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| vix | vix_fred, vix_level | vix_fred | risk_gates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| vix_chg_21 | vix_level | vix_fred | risk_gates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| vol_20 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| vol_60 | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| vol_ratio | market_prices | – | intraday_delta, mismatch_score, monte_carlo, scanner_candidates | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1, momentum_12_1 |
| wti_63d_chg | fred_regime_macro | – | – | enet_xs20_v1, hgb_asym20_v1, hgb_xs20_v1 |
| xbrl_accruals | sec_companyfacts | – | – | ALT-XBRL-001, ALT-XBRL-002, ALT-XBRL-003, ALT-XBRL-004 |
| xbrl_asset_growth | sec_companyfacts | – | – | ALT-XBRL-001, ALT-XBRL-002, ALT-XBRL-003, ALT-XBRL-004 |
| xbrl_rev_yoy | sec_companyfacts | – | – | ALT-XBRL-001, ALT-XBRL-002, ALT-XBRL-003, ALT-XBRL-004 |
| xbrl_share_change | sec_companyfacts | – | – | ALT-XBRL-001, ALT-XBRL-002, ALT-XBRL-003, ALT-XBRL-004 |
| xbrl_sue | sec_companyfacts | – | – | ALT-XBRL-001, ALT-XBRL-002, ALT-XBRL-003, ALT-XBRL-004 |
