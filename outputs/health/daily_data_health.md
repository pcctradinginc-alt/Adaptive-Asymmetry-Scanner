# Daily Data Health Report – 2026-11-09

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.952

Status: HEALTHY 49 · DEGRADED 3 · STALE 2 · BROKEN 0 · UNVALIDATED 9

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| finra_short_interest | STALE | NON_CRITICAL | 0 | 2026-10-15 | – | veraltet (Frequenz-Grenze der Registry) |
| destatis_truck_toll | DEGRADED | IMPORTANT | 0 | 2026-10-01 | – | RECOVERING 1/2: kein neuer erfolgreicher Abruf seit der letzten Prüfung, Bestätigung ausstehend |
| entsoe_power | DEGRADED | NON_CRITICAL | 2 | 2026-11-09 | – | Ingest FAIL: SourceTimeout('Zeitbudget 420s überschritten') |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._