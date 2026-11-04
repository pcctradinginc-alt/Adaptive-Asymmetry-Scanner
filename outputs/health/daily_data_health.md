# Daily Data Health Report – 2026-11-04

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.896 – 2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports']

Status: HEALTHY 42 · DEGRADED 2 · STALE 1 · BROKEN 2 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 4 | 2026-10-25 | – | Ingest FAIL: [Isolation basis=FAIL(400) where=FAIL(400) outFields=FAIL(400) orderByFields=FAIL(400) paging=FAIL(400)] ArcGIS-Query-Fehler: { |
| imf_portwatch_ports | BROKEN | IMPORTANT | 12 | 2026-10-23 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; letzte erfolgreiche Aktualisierung 7.29 T > 3 T; 12 Fehlschläge in Folge |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |
| usaspending_contracts | DEGRADED | NON_CRITICAL | 2 | 2026-11-03 | – | Ingest FAIL: USAspending: 502 für https://api.usaspending.gov/api/v2/search/spending_by_award/; Data-Quality-Befund schwerwiegend |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['estat_jp_truck', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._