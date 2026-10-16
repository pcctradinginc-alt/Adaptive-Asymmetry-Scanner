# Öffentliche Research-Quellen (2026-10-15)

Alle Quellen hier sind **RESEARCH_SHADOW**: Sie werden nur archiviert bzw. als Research-Features berechnet. Sie wirken
nicht auf Champion, Risk Gates, Kill-Switch, EA-Shadow oder Promotion. Es gibt keine kostenpflichtigen Dienste.
Zugangsdaten stehen nie im Code.

| ID | Quelle | Kennzahlen | PIT (`available_at`) | Zugang |
|---|---|---|---|---|
| A4 `nasdaq100_members` | nasdaq.com, offizielle Nasdaq-100-Liste | Mitgliederzahl, Symbole (attrs) je Tag | Abrufzeit | ohne Key |
| B1 `alt_sec_earnings` (Feature) | SEC 8-K Item 2.02 aus dem bestehenden Submissions-Archiv | Tage seit Release, Sitzung (BMO/RTH/AMC), Releases in 365 T | `acceptanceDateTime` (exakt) | ohne Key |
| B3 `finra_short_volume` | FINRA Reg SHO Daily (CNMS) | Leerverkaufsanteil am Volumen je Ticker/Tag | Abrufzeit | ohne Key |
| B3 `finra_short_interest` | FINRA Query API, Consolidated Short Interest | Short-Interest-Stücke, Days-to-Cover | Abrufzeit | ohne Key (live geprüft 2026-10-15); optional `FINRA_API_CLIENT_ID`/`FINRA_API_CLIENT_SECRET`, falls FINRA künftig Zugangsdaten verlangt (dann `AUTH_MISSING`) |
| B4 `fred_liquidity` | FRED/ALFRED: RRPONTSYD, WTREGEN, T10Y2Y | ON-RRP, TGA, Kurve 10J−2J | ALFRED `realtime_start` | vorhandenes `FRED_API_KEY` |
| B5 `wikimedia_pageviews` | Wikimedia REST (nur menschliche Nutzer) | Seitenaufrufe je Ticker/Tag | Abrufzeit | ohne Key |
| B7 `fomc_calendar`, `bls_release_calendar`, `bea_release_calendar` | Federal Reserve (HTML), BLS (iCalendar), BEA (release_dates.json, sonst Plan-Seite) | geplante Termine (UTC), FOMC mit SEP markiert | Abrufzeit | ohne Key |
| B8 `clinicaltrials_sponsors` | ClinicalTrials.gov API v2 | aktive Studien, aktive Phase 3, Primärabschluss in 90 T, Updates in 7 T | Abrufzeit | ohne Key |
| B9 `usaspending_contracts` | USAspending.gov API v2 | Bundesaufträge mit Aktivität in 7 T (USD, Anzahl) | Abrufzeit | ohne Key |
| B10 `sec_13f_holdings` | SEC Form-13F-Datensätze (DERA) | institutionelle Halter, gehaltene Stücke je Quartal | Tag nach dem letzten einbezogenen Filing | ohne Key |

## Regeln

- Fehlendes bleibt fehlend: Es gibt keine Nullen als Ersatz und keine geschätzten Werte.
- Schemaänderungen setzen den Status `SCHEMA_CHANGED` und lösen den bestehenden Alarm aus. Sie werden nie still
  umgedeutet.
- Namenszuordnung (B8–B10) erfolgt über `normalize_name` der Entity-Map: exakter Name oder GLEIF-Tochter (B9),
  Sponsor-Präfix (B8). Mehrdeutige Namen werden verworfen. Die Abdeckung steht in der Meldung jedes Laufs.
- 13F: Es zählen nur fristgerechte 13F-HR (höchstens 45 Tage nach Quartalsende). Amendments, Optionen und
  Nicht-Aktien sind ausgeschlossen. Pro Lauf wird ein Datensatz verarbeitet; ein Marker im Archiv verhindert eine
  doppelte Verarbeitung. Der Preflight archiviert nichts und zählt daher nie als verarbeitet.
- Die Nasdaq-100-Liste von nasdaq.com wird täglich archiviert (PIT-Mitgliederhistorie).
- Produktionsquelle Nasdaq-100 (`universe._nasdaq100_live`, Freigabe 2026-10-16): Wikipedia, sonst die offizielle
  nasdaq.com-Liste, sonst `_NASDAQ100_STATIC`. Anlass: Die Wikipedia-Seite enthält seit spätestens 15.10. keine
  Komponententabelle mehr, die Produktion lief auf der veralteten statischen Liste (92 Titel, u. a. EA ohne Kurse;
  26 aktuelle Mitglieder fehlten). `get_universe`, `_clean` und die statischen Listen sind unverändert
  (UNIVERSE_V1-Code-Hash gleich); das gescannte Universum folgt wieder dem aktuellen Index.
- Täglicher Abgleich (`production_mismatch_count`): offizielle Liste gegen die Liste, die die Produktion tatsächlich
  nutzt (Herkunft in `attrs.production_source`: `wikipedia`, `nasdaq_official` oder `static_fallback`, danach
  `_clean`). Abweichungen meldet das Ops-Monitoring (`UNIVERSE_SOURCE_MISMATCH:nasdaq100`).
- **Kalenderquellen** (`fomc_calendar`, `bls_release_calendar`, `bea_release_calendar`) haben eine einheitliche
  PIT-Semantik:
  - `observation_time` ist der geplante Termin (UTC; FOMC 14:00 New York, BLS laut TZID, BEA ohne Offset = New York).
  - `available_at` = `retrieved_at` ist der erste Abruf. Der tatsächliche Ankündigungszeitpunkt ist nicht belegt und
    wird nie früher angesetzt. Vor dem ersten Abruf ist ein Termin unbekannt und für Backtests gesperrt.
  - Ganztägige BLS-Einträge erhalten 08:30 New York und `attrs.time_assumed`.
  - `allow_future_observations: true` mit `max_future_days: 410`: Angekündigte Termine sind kein
    FUTURE_OBSERVATION-Befund. Weiter entfernte Zeitpunkte bleiben einer.
  - Echte PIT-Verletzungen (`available_at > retrieved_at`) prüft der Orchestrator. Source Health meldet sie als
    „PIT-Verletzung“.
- **Terminänderungen:**
  - Bei geändertem Plan (oder beim ersten Mal) speichert jeder Abruf eine Momentaufnahme (`schedule_snapshot`). Sie
    enthält alle Termine im Fenster sowie `added` und `withdrawn`, also verschobene oder abgesagte Termine.
  - `calendars.schedule_as_of(source_id, t)` liefert den Plan, wie er zum Zeitpunkt t bekannt war, und `None` vor der
    ersten Momentaufnahme.
  - Terminzeilen allein reichen dafür nicht: Ein verschobener Termin bleibt dort neben dem neuen stehen.
  - Unveränderte Wiederholungen schreiben nichts (idempotent).
  - Die Zeilen vom 15./16.10. haben noch keine Momentaufnahme. Für `t` vor der ersten Momentaufnahme gibt es keinen
    Plan.
- **13F:**
  - Der gemeldete Datenstand ist der jüngste verarbeitete Datensatz. Der Backfill älterer Sätze setzt ihn nicht
    zurück (Live 16.10.: STALE durch Rücksprung auf 28.02.).
  - `max_staleness_days: 120` bleibt. Die SEC veröffentlicht die 3-Monats-Sätze nach Ende der Monate Feb/Mai/Aug/Nov.
    Laut DERA-Bibliothek war der Satz bis 30.11.2025 am 03.12.2025 online. Die 120 T decken ≈ 92 T Fensterabstand
    plus ≈ 28 T Veröffentlichungsverzug ab.
  - `available_at` ist der Tag nach dem letzten einbezogenen Filing (öffentlich auf EDGAR). Das DERA-Zip erscheint
    später. Strategien, die nur das Zip nutzen, müssen `retrieved_at` verwenden.
- **noaa_storm_events:**
  - Der Ereignistyp fehlte in der Identität: 68 Identitäten für 1005 Werte. Folgen: 932 Konflikte, 776 falsche
    Revisionen und 161 verworfene Werte. Es gab nur einen echten Abruf am 04.10. und keine inkonsistenten
    Quelldaten.
  - Ab `storm-events-2` gilt `series_id = <Jahr>:<Ereignistyp>`.
  - Statische Quellen werden bei geändertem Parser sofort neu abgerufen, ohne auf `min_refetch_days` zu warten.
  - Die Altzeilen (`parser_version` "1") bleiben im Archiv. Ihre Identität unterscheidet sich, deshalb sind sie
    getrennt erkennbar. Sie haben keine Verbraucher.
- Die Live-Formate werden mit dem Workflow „External Data Preflight“ geprüft. Der Eingang `family` steht dafür auf
  `public_research,real_economy`.

## Archivwachstum (A5)

Kalte, abgeschlossene Partitionen, die seit `compress_cold_after_days` (21) Tagen unberührt sind, werden verlustfrei
nach `<key>.jsonl.gz` komprimiert. Alle Leser lesen `.jsonl` und `.jsonl.gz` transparent.
