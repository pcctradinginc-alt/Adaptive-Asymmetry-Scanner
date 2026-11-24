# Daily Data Health Report – 2026-11-24

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.976

Status: HEALTHY 47 · DEGRADED 2 · STALE 1 · BROKEN 0 · UNVALIDATED 13

## Neue Änderungen

- degradation: finra_short_interest HEALTHY → STALE (veraltet (Frequenz-Grenze der Registry))

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| finra_short_interest | STALE | NON_CRITICAL | 0 | 2026-10-30 | – | veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 2 | 2026-11-13 | – | Ingest FAIL: UPSTREAM_UNAVAILABLE service=https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/ |
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