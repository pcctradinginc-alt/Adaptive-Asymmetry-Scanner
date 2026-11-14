# Daily Data Health Report – 2026-11-14

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.951

Status: HEALTHY 46 · DEGRADED 4 · STALE 0 · BROKEN 0 · UNVALIDATED 13

## Neue Änderungen

- degradation: imf_portwatch_ports HEALTHY → DEGRADED (Ingest FAIL: [UPSTREAM_RATE_LIMITED] ArcGIS-Query-Fehler trotz Wartezeit (höchstens 3 Versuche, Wartebudget 130s je Host): {'code': 429, 'message': 'Unable to p)
- degradation: imf_portwatch_chokepoints HEALTHY → DEGRADED (Ingest FAIL: UPSTREAM_RATE_LIMITED host=services9.arcgis.com: Kontingent in diesem Lauf bereits erschöpft (Unable to perform query. Too many requests.) – kein w)
- degradation: imf_portwatch_disruptions HEALTHY → DEGRADED (Ingest FAIL: UPSTREAM_RATE_LIMITED host=services9.arcgis.com: Kontingent in diesem Lauf bereits erschöpft (Unable to perform query. Too many requests.) – kein w)

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 1 | 2026-11-08 | – | Ingest FAIL: UPSTREAM_RATE_LIMITED host=services9.arcgis.com: Kontingent in diesem Lauf bereits erschöpft (Unable to perform query. Too many |
| imf_portwatch_disruptions | DEGRADED | NON_CRITICAL | 1 | 2026-08-14 | – | Ingest FAIL: UPSTREAM_RATE_LIMITED host=services9.arcgis.com: Kontingent in diesem Lauf bereits erschöpft (Unable to perform query. Too many |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 1 | 2026-11-06 | – | Ingest FAIL: [UPSTREAM_RATE_LIMITED] ArcGIS-Query-Fehler trotz Wartezeit (höchstens 3 Versuche, Wartebudget 130s je Host): {'code': 429, 'me |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'noaa_storm_events', 'sec_13f_holdings', 'usaspending_contracts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._