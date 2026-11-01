# Daily Data Health Report – 2026-11-01

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.9

Status: HEALTHY 42 · DEGRADED 2 · STALE 1 · BROKEN 2 · UNVALIDATED 8

## Neue Änderungen

- safe_mode_off: * True → False

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_disruptions | BROKEN | NON_CRITICAL | 4 | 2026-08-14 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| imf_portwatch_ports | BROKEN | IMPORTANT | 6 | 2026-10-23 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| estat_jp_truck | STALE | NON_CRITICAL | 1 | 2026-03-01 | – | Ingest FAIL: e-Stat getStatsList fehlgeschlagen: HTTPSConnectionPool(host='api.e-stat.go.jp', port=443): Read timed out. (read timeout=30);  |
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 0 | 2026-10-25 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['estat_jp_truck', 'imf_portwatch_chokepoints', 'imf_portwatch_ports', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._