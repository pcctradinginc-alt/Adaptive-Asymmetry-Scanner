# Daily Data Health Report – 2026-11-18

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.943

Status: HEALTHY 47 · DEGRADED 2 · STALE 1 · BROKEN 0 · UNVALIDATED 13

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_ports | STALE | IMPORTANT | 0 | 2026-11-13 | – | Ingest WARN: 10829 observations (kuratierte Häfen + GLOBAL/Gruppen-Aggregate aus 739347 Rohbeobachtungen); 4 unresolved ports: ['Houston', ' |
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 0 | 2026-11-15 | – | RECOVERING 1/2: kein neuer erfolgreicher Abruf seit der letzten Prüfung, Bestätigung ausstehend |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'usaspending_contracts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._