# Daily Data Health Report – 2026-11-08

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.936

Status: HEALTHY 51 · DEGRADED 1 · STALE 1 · BROKEN 1 · UNVALIDATED 9

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| destatis_truck_toll | BROKEN | IMPORTANT | 4 | 2026-09-01 | – | Ingest FAIL: GENESIS meldet Status 7000: Für die Teiltabelle mit der Metadatenkombination:
Indizes aus der Lkw-Mauterhebung (42191)
Lkw-Maut |
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