# Daily Data Health Report – 2026-10-03

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.919

Status: HEALTHY 26 · DEGRADED 2 · STALE 0 · BROKEN 2 · UNVALIDATED 9

## Neue Änderungen

- new_failure: sec_companyfacts None → BROKEN (PIT-Verletzung: 1 Zeitstempel in der Zukunft / available > retrieved; Ausreißer-Anteil 3.0%)
- new_failure: bts_open_data_tsi None → BROKEN (Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; veraltet (Frequenz-Grenze der Registry))

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| bts_open_data_tsi | BROKEN | IMPORTANT | 1 | 2024-04-01 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; veraltet (Frequenz-Grenze der Registry) |
| sec_companyfacts | BROKEN | NON_CRITICAL | 1 | 2026-09-28 | xbrl_accruals, xbrl_asset_growth, xbrl_rev_yoy, xbrl_share_change | PIT-Verletzung: 1 Zeitstempel in der Zukunft / available > retrieved; Ausreißer-Anteil 3.0% |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 0 | 2026-09-25 | – | Ingest WARN: 10682 observations (kuratierte Häfen + GLOBAL/Gruppen-Aggregate aus 607110 Rohbeobachtungen); 0 unresolved ports: [] | volume g |
| vix_fred | DEGRADED | CRITICAL | 1 | – | vix, risk_gates | Timeout: FetchError: HTTPSConnectionPool(host='fred.stlouisfed.org', port=443): Read timed out. (read timeout=20); Antwortzeit 66.83 s > 15  |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: ['xbrl_accruals', 'xbrl_asset_growth', 'xbrl_rev_yoy', 'xbrl_share_change', 'xbrl_sue']

Wiederkehrende Instabilität (30 T): keine

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._