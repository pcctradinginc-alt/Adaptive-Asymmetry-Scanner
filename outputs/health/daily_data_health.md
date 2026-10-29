# Daily Data Health Report – 2026-10-29

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.96

Status: HEALTHY 44 · DEGRADED 2 · STALE 1 · BROKEN 0 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 2 | 2026-10-23 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'finra_short_interest', 'fred_market_expectations', 'imf_portwatch_ports', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._