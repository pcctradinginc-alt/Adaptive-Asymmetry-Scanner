# Daily Data Health Report – 2026-11-21

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.959

Status: HEALTHY 47 · DEGRADED 1 · STALE 1 · BROKEN 1 · UNVALIDATED 13

## Neue Änderungen

- degradation: nasdaq100_members HEALTHY → BROKEN (Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend)

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| nasdaq100_members | BROKEN | NON_CRITICAL | 1 | 2026-11-20 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend |
| imf_portwatch_ports | STALE | IMPORTANT | 0 | 2026-11-13 | – | Ingest WARN: 8918 observations (kuratierte Häfen + GLOBAL/Gruppen-Aggregate aus 608874 Rohbeobachtungen); 4 unresolved ports: ['Houston', 'L |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'entsoe_power', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'usaspending_contracts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._