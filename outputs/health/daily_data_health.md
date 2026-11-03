# Daily Data Health Report – 2026-11-03

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.888 – 2 unabhängige wichtige Quellen ausgefallen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports']

Status: HEALTHY 42 · DEGRADED 2 · STALE 1 · BROKEN 2 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 2 | 2026-10-25 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend |
| imf_portwatch_ports | BROKEN | IMPORTANT | 10 | 2026-10-23 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; letzte erfolgreiche Aktualisierung 6.28 T > 3 T; 10 Fehlschläge in Folge |
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_disruptions | DEGRADED | NON_CRITICAL | 0 | 2026-08-14 | – | RECOVERING 1/2: kein neuer erfolgreicher Abruf seit der letzten Prüfung, Bestätigung ausstehend |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['estat_jp_truck', 'imf_portwatch_chokepoints', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._