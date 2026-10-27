# Daily Data Health Report – 2026-10-27

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.952

Status: HEALTHY 44 · DEGRADED 2 · STALE 1 · BROKEN 0 · UNVALIDATED 8

## Neue Änderungen

- recovery: finra_short_interest DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 2 | 2026-10-18 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Invalid URL', 'details': ['Invalid URL']} (outFields=capacity,capacity_cargo,cap |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 2 | 2026-10-16 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Invalid URL', 'details': ['Invalid URL']} (outFields=ISO3,country,date,export,ex |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'finra_short_interest', 'fred_market_expectations', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._