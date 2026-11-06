# Daily Data Health Report – 2026-11-06

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.888 – 2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports']

Status: HEALTHY 49 · DEGRADED 2 · STALE 1 · BROKEN 2 · UNVALIDATED 9

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 8 | 2026-10-25 | – | Ingest FAIL: [Isolation basis=FAIL(429) where=FAIL(429) outFields=FAIL(429) orderByFields=FAIL(429) paging=FAIL(429)] ArcGIS-Query-Fehler: { |
| imf_portwatch_ports | BROKEN | IMPORTANT | 16 | 2026-10-23 | – | Ingest FAIL: [Isolation basis=FAIL(429) where=FAIL(429) outFields=FAIL(429) orderByFields=FAIL(429) paging=FAIL(429)] ArcGIS-Query-Fehler: { |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_disruptions | DEGRADED | NON_CRITICAL | 2 | 2026-08-14 | – | Ingest FAIL: [Isolation basis=FAIL(429) outFields=FAIL(429) paging=FAIL(429)] ArcGIS-Query-Fehler: {'code': 429, 'message': 'Unable to perfo |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['estat_jp_truck', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._