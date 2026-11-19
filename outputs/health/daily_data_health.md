# Daily Data Health Report – 2026-11-19

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.943

Status: HEALTHY 46 · DEGRADED 3 · STALE 1 · BROKEN 0 · UNVALIDATED 13

## Neue Änderungen

- degradation: entsoe_power HEALTHY → DEGRADED (Ingest WARN: 398 Tageswerte (5 Preiszonen, Last ['DE']) | Fehler: DE_LU: 400 für https://web-api.tp.entsoe.eu/api (nicht retrybar): <?xml version="1.0" encoding)
- recovery: imf_portwatch_chokepoints DEGRADED → HEALTHY
- degradation: estat_jp_truck HEALTHY → DEGRADED (Ingest FAIL: e-Stat getStatsList fehlgeschlagen: HTTPSConnectionPool(host='api.e-stat.go.jp', port=443): Read timed out. (read timeout=30); Data-Quality-Befund )

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_ports | STALE | IMPORTANT | 0 | 2026-11-13 | – | Ingest WARN: 10192 observations (kuratierte Häfen + GLOBAL/Gruppen-Aggregate aus 695856 Rohbeobachtungen); 4 unresolved ports: ['Houston', ' |
| entsoe_power | DEGRADED | NON_CRITICAL | 0 | 2026-11-18 | – | Ingest WARN: 398 Tageswerte (5 Preiszonen, Last ['DE']) | Fehler: DE_LU: 400 für https://web-api.tp.entsoe.eu/api (nicht retrybar): <?xml ve |
| estat_jp_truck | DEGRADED | NON_CRITICAL | 1 | 2026-04-01 | – | Ingest FAIL: e-Stat getStatsList fehlgeschlagen: HTTPSConnectionPool(host='api.e-stat.go.jp', port=443): Read timed out. (read timeout=30);  |
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