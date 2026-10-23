# Daily Data Health Report – 2026-10-23

**Safe Mode (Daten): aus** – gewichtete Data Quality 1.0

Status: HEALTHY 47 · DEGRADED 0 · STALE 0 · BROKEN 0 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| – | alle HEALTHY | | | | | |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['bea_release_calendar', 'bls_release_calendar', 'eia_natural_gas', 'fomc_calendar', 'fred_market_expectations', 'noaa_storm_events', 'sec_13f_holdings', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._