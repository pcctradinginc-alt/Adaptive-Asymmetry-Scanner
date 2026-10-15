# Daily Data Health Report – 2026-10-15

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.976

Status: HEALTHY 41 · DEGRADED 2 · STALE 1 · BROKEN 3 · UNVALIDATED 8

## Neue Änderungen

- recovery: sec_submissions DEGRADED → HEALTHY
- recovery: sec_form345 DEGRADED → HEALTHY
- recovery: sec_companyfacts DEGRADED → HEALTHY
- recovery: gleif_lei DEGRADED → HEALTHY
- recovery: ted_awards DEGRADED → HEALTHY
- recovery: fred_commodities DEGRADED → HEALTHY
- new_failure: fomc_calendar None → BROKEN (Ingest WARN: 9 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 9 Beobachtungen mit available_at > retrieved_at; Data-Quality-Befund schwerwiegend)
- new_failure: bls_release_calendar None → BROKEN (Ingest WARN: 33 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 29 Beobachtungen mit available_at > retrieved_at; Data-Quality-Befund schwerwiegend)
- new_failure: bea_release_calendar None → BROKEN (Ingest WARN: 66 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 66 Beobachtungen mit available_at > retrieved_at; Data-Quality-Befund schwerwiegend)
- new_failure: sec_13f_holdings None → STALE (veraltet (Frequenz-Grenze der Registry))

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| bea_release_calendar | BROKEN | NON_CRITICAL | 1 | 2027-11-04 | – | Ingest WARN: 66 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 66 Beobachtungen mit available_at > retrieved_at; Data-Quality- |
| bls_release_calendar | BROKEN | NON_CRITICAL | 1 | 2026-12-30 | – | Ingest WARN: 33 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 29 Beobachtungen mit available_at > retrieved_at; Data-Quality- |
| fomc_calendar | BROKEN | NON_CRITICAL | 1 | 2027-10-27 | – | Ingest WARN: 9 Termine im Fenster | DQ: FUTURE_OBSERVATION; PIT-Verletzung: 9 Beobachtungen mit available_at > retrieved_at; Data-Quality-Be |
| sec_13f_holdings | STALE | NON_CRITICAL | 0 | 2026-05-31 | – | veraltet (Frequenz-Grenze der Registry) |
| fred_market_expectations | DEGRADED | IMPORTANT | 0 | 2026-10-14 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| noaa_storm_events | DEGRADED | NON_CRITICAL | 0 | 2025-01-01 | – | Data-Quality-Befund schwerwiegend; Duplikate None / Konflikte 932 |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._