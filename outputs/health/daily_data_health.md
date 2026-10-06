# Daily Data Health Report – 2026-10-06

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.992

Status: HEALTHY 32 · DEGRADED 3 · STALE 0 · BROKEN 0 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| eia_natural_gas | DEGRADED | NON_CRITICAL | 0 | 2026-09-29 | cmdx_natural_gas__div_gas_storage_price, cmdx_natural_gas__lng_exports_yoy, cmdx_natural_gas__natgas_storage_chg_z_52w, cmdx_natural_gas__natgas_storage_vs_5y | Ingest WARN: übersprungen: alle Serien auf dem Stand der letzten Veröffentlichung; Data-Quality-Befund schwerwiegend |
| fred_commodities | DEGRADED | NON_CRITICAL | 0 | 2026-09-29 | cmdx_copper__copper_ret_3m, cmdx_copper__div_copper_positioning_price, cmdx_corn__corn_ret_3m, cmdx_natural_gas__div_gas_storage_price | Ingest WARN: 7374 ALFRED-Vintage-Beobachtungen (Brent/HenryHub/Kupfer/Weizen/Mais/Soja) | DQ: OUT_OF_RANGE; Data-Quality-Befund schwerwiegen |
| noaa_storm_events | DEGRADED | NON_CRITICAL | 0 | 2025-01-01 | – | Data-Quality-Befund schwerwiegend; Duplikate None / Konflikte 932 |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['bts_open_data_tsi', 'sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._