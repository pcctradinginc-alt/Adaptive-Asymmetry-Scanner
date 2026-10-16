# Daily Data Health Report – 2026-10-16

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.992

Status: HEALTHY 42 · DEGRADED 1 · STALE 1 · BROKEN 3 · UNVALIDATED 8

## Neue Änderungen

- recovery: fred_market_expectations DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| bea_release_calendar | BROKEN | NON_CRITICAL | 2 | 2027-11-04 | – | Ingest WARN: 66 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 66 Beobachtungen mit available_at > retrieved_at; Data-Quality- |
| bls_release_calendar | BROKEN | NON_CRITICAL | 2 | 2026-12-30 | – | Ingest WARN: 33 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 29 Beobachtungen mit available_at > retrieved_at; Data-Quality- |
| fomc_calendar | BROKEN | NON_CRITICAL | 2 | 2027-10-27 | – | Ingest WARN: 9 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 9 Beobachtungen mit available_at > retrieved_at; Data-Quality-Be |
| sec_13f_holdings | STALE | NON_CRITICAL | 0 | 2026-02-28 | – | veraltet (Frequenz-Grenze der Registry) |
| noaa_storm_events | DEGRADED | NON_CRITICAL | 0 | 2025-01-01 | – | Data-Quality-Befund schwerwiegend; Duplikate None / Konflikte 932 |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['fred_market_expectations', 'sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._