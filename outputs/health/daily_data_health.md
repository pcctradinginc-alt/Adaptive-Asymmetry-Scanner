# Daily Data Health Report – 2026-11-17

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.869 – 2 wichtige Quellen ausgefallen aus 1 unabhängigen Ausfall-Domänen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports'] (Domänen: ['imf_portwatch_arcgis']; Safe-Mode-Schwelle unverändert)

Status: HEALTHY 44 · DEGRADED 4 · STALE 0 · BROKEN 2 · UNVALIDATED 13

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 7 | 2026-11-08 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| imf_portwatch_ports | BROKEN | IMPORTANT | 7 | 2026-11-06 | – | Ingest FAIL: [Isolation basis=OK where=OK outFields=OK orderByFields=FAIL(400) paging=FAIL(400)] ArcGIS-Query-Fehler: {'code': 400, 'message |
| imf_portwatch_disruptions | DEGRADED | NON_CRITICAL | 0 | 2026-08-14 | – | RECOVERING 1/2: kein neuer erfolgreicher Abruf seit der letzten Prüfung, Bestätigung ausstehend |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |
| nws_alerts | DEGRADED | IMPORTANT | 2 | – | – | Ingest FAIL: SourceTimeout('Zeitbudget 300s überschritten') |
| nws_forecast | DEGRADED | IMPORTANT | 0 | – | – | Ingest WARN: 12 von 45 Locations fehlgeschlagen |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'usaspending_contracts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._