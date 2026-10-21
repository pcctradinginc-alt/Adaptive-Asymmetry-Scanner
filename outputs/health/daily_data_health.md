# Daily Data Health Report – 2026-10-21

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.992

Status: HEALTHY 45 · DEGRADED 1 · STALE 0 · BROKEN 1 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| vix_fred | BROKEN | CRITICAL | 2 | – | vix, risk_gates | Schema: kein JSON: Expecting value: line 1 column 1 (char 0) |
| estat_jp_truck | DEGRADED | NON_CRITICAL | 2 | 2026-03-01 | – | Ingest FAIL: e-Stat getStatsList fehlgeschlagen: HTTPSConnectionPool(host='api.e-stat.go.jp', port=443): Read timed out. (read timeout=30);  |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: ['vix']
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['bea_release_calendar', 'bls_release_calendar', 'eia_natural_gas', 'fomc_calendar', 'fred_market_expectations', 'noaa_storm_events', 'sec_13f_holdings', 'sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._