# Daily Data Health Report – 2026-10-22

**Safe Mode (Daten): aus** – gewichtete Data Quality 1.0

Status: HEALTHY 46 · DEGRADED 1 · STALE 0 · BROKEN 0 · UNVALIDATED 8

## Neue Änderungen

- recovery: estat_jp_truck DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| vix_fred | DEGRADED | CRITICAL | 0 | 2026-10-21 | vix, risk_gates | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['bea_release_calendar', 'bls_release_calendar', 'eia_natural_gas', 'fomc_calendar', 'fred_market_expectations', 'noaa_storm_events', 'sec_13f_holdings', 'sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._