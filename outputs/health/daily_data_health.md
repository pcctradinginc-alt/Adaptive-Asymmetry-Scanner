# Daily Data Health Report – 2026-10-27

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.952

Status: HEALTHY 43 · DEGRADED 3 · STALE 1 · BROKEN 0 · UNVALIDATED 8

## Neue Änderungen

- degradation: imf_portwatch_ports HEALTHY → DEGRADED (Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Invalid URL', 'details': ['Invalid URL']} (outFields=ISO3,country,date,export,export_cargo,export_co)
- degradation: imf_portwatch_chokepoints HEALTHY → DEGRADED (Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Invalid URL', 'details': ['Invalid URL']} (outFields=capacity,capacity_cargo,capacity_container,capa)
- degradation: estat_jp_truck HEALTHY → STALE (veraltet (Frequenz-Grenze der Registry))

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| finra_short_interest | DEGRADED | NON_CRITICAL | 0 | 2026-10-15 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 1 | 2026-10-18 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Invalid URL', 'details': ['Invalid URL']} (outFields=capacity,capacity_cargo,cap |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 1 | 2026-10-16 | – | Ingest FAIL: ArcGIS-Query-Fehler: {'code': 400, 'message': 'Invalid URL', 'details': ['Invalid URL']} (outFields=ISO3,country,date,export,ex |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'finra_short_interest', 'fred_market_expectations', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._