# Daily Data Health Report – 2026-10-30

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.912

Status: HEALTHY 42 · DEGRADED 3 · STALE 1 · BROKEN 1 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_ports | BROKEN | IMPORTANT | 4 | 2026-10-23 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 2 | 2026-10-25 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| imf_portwatch_disruptions | DEGRADED | NON_CRITICAL | 2 | 2026-08-14 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'estat_jp_truck', 'imf_portwatch_ports', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._