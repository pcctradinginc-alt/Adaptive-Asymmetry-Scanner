# Daily Data Health Report – 2026-10-28

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.984

Status: HEALTHY 46 · DEGRADED 0 · STALE 1 · BROKEN 0 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'finra_short_interest', 'fred_market_expectations', 'noaa_storm_events', 'sec_companyfacts', 'vix_fred']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._