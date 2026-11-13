# Daily Data Health Report – 2026-11-13

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.992

Status: HEALTHY 49 · DEGRADED 1 · STALE 0 · BROKEN 0 · UNVALIDATED 13

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'noaa_storm_events', 'sec_13f_holdings']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._