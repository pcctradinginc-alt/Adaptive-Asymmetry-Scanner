# Daily Data Health Report – 2026-11-05

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.896 – 2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports']

Status: HEALTHY 50 · DEGRADED 1 · STALE 1 · BROKEN 2 · UNVALIDATED 9

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 6 | 2026-10-25 | – | Ingest FAIL: [UPSTREAM_UNAVAILABLE] [Isolation basis=FAIL(400) where=SKIPPED(UPSTREAM_DOWN) outFields=SKIPPED(UPSTREAM_DOWN) orderByFields=S |
| imf_portwatch_ports | BROKEN | IMPORTANT | 14 | 2026-10-23 | – | Ingest FAIL: [UPSTREAM_UNAVAILABLE] [Isolation basis=FAIL(400) where=SKIPPED(UPSTREAM_DOWN) outFields=SKIPPED(UPSTREAM_DOWN) orderByFields=S |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
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