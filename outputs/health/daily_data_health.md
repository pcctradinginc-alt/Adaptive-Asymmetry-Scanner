# Daily Data Health Report – 2026-10-13

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.872

Status: HEALTHY 28 · DEGRADED 2 · STALE 5 · BROKEN 1 · UNVALIDATED 8

## Neue Änderungen

- degradation: fred_market_expectations UNVALIDATED → BROKEN (Ingest FAIL: FRED ALFRED-Abruf fehlgeschlagen (DGS2): 400 für https://api.stlouisfed.org/fred/series/observations (nicht retrybar): {"error_code":400,"error_mes)

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| fred_market_expectations | BROKEN | IMPORTANT | 3 | – | – | Ingest FAIL: FRED ALFRED-Abruf fehlgeschlagen (DGS2): 400 für https://api.stlouisfed.org/fred/series/observations (nicht retrybar): {"error_ |
| gleif_lei | STALE | NON_CRITICAL | 0 | – | – | letzte erfolgreiche Aktualisierung 10.81 T > 9 T |
| sec_companyfacts | STALE | NON_CRITICAL | 0 | 2026-09-28 | xbrl_accruals, xbrl_asset_growth, xbrl_rev_yoy, xbrl_share_change | letzte erfolgreiche Aktualisierung 10.35 T > 9 T |
| sec_form345 | STALE | NON_CRITICAL | 0 | 2026-09-30 | sec_8k_count_30d_z, sec_8k_negative_90d, sec_exec_change_90d, sec_filing_delay_z | letzte erfolgreiche Aktualisierung 10.35 T > 9 T |
| sec_submissions | STALE | NON_CRITICAL | 0 | 2026-10-03 | sec_8k_count_30d_z, sec_8k_negative_90d, sec_exec_change_90d, sec_filing_delay_z | jüngste Beobachtung 10.77 T alt > 10 T; letzte erfolgreiche Aktualisierung 10.35 T > 9 T |
| ted_awards | STALE | NON_CRITICAL | 0 | 2026-10-02 | ted_any_award_365d, ted_award_value_365d, ted_awards_90d, ted_awards_z | letzte erfolgreiche Aktualisierung 10.34 T > 9 T |
| fred_commodities | DEGRADED | NON_CRITICAL | 0 | 2026-10-06 | cmdx_copper__copper_ret_3m, cmdx_copper__div_copper_positioning_price, cmdx_corn__corn_ret_3m, cmdx_natural_gas__div_gas_storage_price | Ingest WARN: 7384 ALFRED-Vintage-Beobachtungen (Brent/HenryHub/Kupfer/Weizen/Mais/Soja) | DQ: OUT_OF_RANGE; Data-Quality-Befund schwerwiegen |
| noaa_storm_events | DEGRADED | NON_CRITICAL | 0 | 2025-01-01 | – | Data-Quality-Befund schwerwiegend; Duplikate None / Konflikte 932 |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: ['sec_8k_count_30d_z', 'sec_8k_negative_90d', 'sec_exec_change_90d', 'sec_filing_delay_z', 'sec_insider_buy_value_90d', 'sec_insider_buyers_90d', 'sec_insider_cluster_30d', 'sec_insider_net_value_90d', 'sec_late_filing_365d', 'ted_any_award_365d', 'ted_award_value_365d', 'ted_awards_90d', 'ted_awards_z']
- Cache (stale) genutzt: ['xbrl_accruals', 'xbrl_asset_growth', 'xbrl_rev_yoy', 'xbrl_share_change', 'xbrl_sue']

Wiederkehrende Instabilität (30 T): ['sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._