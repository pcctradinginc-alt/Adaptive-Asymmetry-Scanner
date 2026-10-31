# Daily Data Health Report – 2026-10-31

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.876 – 2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports']

Status: HEALTHY 42 · DEGRADED 1 · STALE 1 · BROKEN 3 · UNVALIDATED 8

## Neue Änderungen

- degradation: imf_portwatch_chokepoints DEGRADED → BROKEN (Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; 3 Fehlschläge in Folge)
- degradation: imf_portwatch_disruptions DEGRADED → BROKEN (Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform query. Please check)
- safe_mode_on: * False → True (2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports'])

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 3 | 2026-10-25 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; 3 Fehlschläge in Folge |
| imf_portwatch_disruptions | BROKEN | NON_CRITICAL | 3 | 2026-08-14 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| imf_portwatch_ports | BROKEN | IMPORTANT | 5 | 2026-10-23 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Cannot perform query. Invalid query parameters.', 'details': ['Unable to perform |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'estat_jp_truck', 'imf_portwatch_chokepoints', 'imf_portwatch_ports', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._