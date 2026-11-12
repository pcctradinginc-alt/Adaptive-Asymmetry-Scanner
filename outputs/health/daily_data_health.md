# Daily Data Health Report – 2026-11-12

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.967

Status: HEALTHY 47 · DEGRADED 3 · STALE 0 · BROKEN 0 · UNVALIDATED 13

## Neue Änderungen

- recovery: finra_short_interest DEGRADED → HEALTHY
- degradation: nhc_storms HEALTHY → DEGRADED (Ingest WARN: DQ: SCALE_SHIFT; Data-Quality-Befund schwerwiegend)

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | DEGRADED | NON_CRITICAL | 0 | 2026-04-01 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |
| nhc_storms | DEGRADED | IMPORTANT | 0 | – | – | Ingest WARN: DQ: SCALE_SHIFT; Data-Quality-Befund schwerwiegend |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._