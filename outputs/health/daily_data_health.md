# Daily Data Health Report – 2026-11-02

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.916

Status: HEALTHY 43 · DEGRADED 1 · STALE 1 · BROKEN 2 · UNVALIDATED 8

## Neue Änderungen

- recovery: imf_portwatch_chokepoints DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_disruptions | BROKEN | NON_CRITICAL | 6 | 2026-08-14 | – | Ingest FAIL: [Isolation basis=OK outFields=OK] ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters |
| imf_portwatch_ports | BROKEN | IMPORTANT | 8 | 2026-10-23 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; letzte erfolgreiche Aktualisierung 5.37 T > 3 T; 8 Fehlschläge in Folge |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
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