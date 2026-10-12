# Betriebsreife vor dem Merge von PR #111 (Stand 2026-10-11)

Ziel ist ausschließlich ein zuverlässig laufender Scanner, der dauerhaft Daten sammelt. Es gibt keine neuen Research-Module und keinen Architekturumbau.

Folgende Teile bleiben unverändert:
- Champion-Entscheidungslogik
- Risk Gates
- Kill-Switch (`active: false`)
- EA-Shadow
- Promotion-Verträge
- `max_automatic_influence: ABSTENTION_ONLY`

Es gibt keine Orders und keine automatische Modellfreigabe.

## 1. Gemessene Ausgangslage

Grundlage sind die Actions-API und die Läufe vom 17.09. bis 10.10.2026.

| Workflow | Cron (UTC) | tatsächlicher Start (UTC) | Verzögerung |
|---|---|---|---|
| External Data | 06:17 | ca. 13:25–13:40 | ca. 7 h |
| Source Health | 12:41 | ca. 17:26–19:03 | 5–6 h |
| Scanner | 13:30 | ca. 17:06–21:08 | 4–7,5 h |
| Feedback | 15:30 / 19:30 | ca. 20:00–20:45 / 23:00–00:53 | 4,5–5,5 h |

Folgerungen:
- **Kein fester UTC-Zeitpunkt trifft die US-Sitzung verlässlich.**
  - Die Sitzung liegt im Sommer bei 13:30–20:00 UTC und im Winter bei 14:30–21:00 UTC.
  - Der Vorschlag „14:35 UTC ganzjährig“ würde real etwa 19:35–21:35 UTC feuern, im Sommer oft nach Handelsschluss.
  - Würde er pünktlich ausgeführt, läge er im Winter bei 09:35 ET, also 5 Minuten nach der Eröffnung mit den breitesten Spreads.
- **Der Scanner-Guard war DST-blind.** `not_before_utc 13:30` bedeutet 09:30 ET nur im Sommer, im Winter 08:30 ET (Pre-Market).
- **Der Guard prüfte nur Wochentage.** An NYSE-Feiertagen lief der Scanner mit Quotes eines geschlossenen Marktes.
- **`option_quotes.session_of` kannte keine Feiertage und keine verkürzten Sitzungen.**

## 2. Behobene Probleme

| # | Problem | Behebung | Test |
|---|---|---|---|
| 1 | RTH-Marks nur im Scanner-Job; bei spätem Start (z. B. 10-05, 21:08 UTC) keine Marks | Neuer Workflow `options_capture.yml`, unabhängig vom Scanner (Details unter der Tabelle) | `test_ops_readiness.py` (Fenster, Abdeckung, Idempotenz) |
| 2 | Marks nur, wenn der rechnende Lauf selbst in der Sitzung liegt | `candidate_ledger` nutzt außerhalb der Sitzung archivierte, verwertbare RTH-Quotes. Gewählt wird der früheste NY-Handelstag ab Fälligkeit; bei Spreads müssen beide Legs vom selben Tag sein. Feld `*_mark_source_{h}d = archive_rth`; „late“ zählt ab dem Mark-Tag. Nie ein Modellpreis. | ebd. |
| 3 | DST-blinder Scanner-Guard | `not_before_ny: "09:30"` (New-York-Zeit) und zweiter Cron 14:30 UTC (Winter). Der jeweils andere Trigger endet nach etwa 1 min mit `SKIP_*`. | `test_scanner_guard_uses_new_york_time_and_calendar` |
| 4 | Scanner an Börsenfeiertagen | `trading_days_only: true` führt zu `SKIP_MARKET_CLOSED`. Der Watchdog erwartet an Feiertagen keinen Lauf (`NOT_SCHEDULED`, keine Recovery). | `test_watchdog_expects_no_scanner_on_market_holiday` |
| 5 | Kalender ohne Sonderschließungen | `config/market_calendar.yaml` (u. a. 2025-01-09); `market_snapshot` ist die einzige Kalenderquelle. Geprüft gegen die offiziellen NYSE-Listen 2025–2027. | `test_session_of_follows_nyse_calendar_across_dst` |
| 6 | `session_of` ohne Feiertage und verkürzte Sitzungen | Nutzt den Kalender: Feiertag ganztägig `CLOSED`, an verkürzten Tagen ab 13:00 ET `POST` | ebd. |
| 7 | Marks und Quotes gingen bei einem Scanner-Fehler verloren (Fehlerpfad committete nur `outputs/monitoring/`) | Commit direkt nach dem RTH-Schritt (vor der rund 50-minütigen Pipeline). Der Fehlerpfad sichert zusätzlich `outputs/options_archive/`. | `test_scanner_persists_*` |
| 8 | Ein Feedback-Fehler verwarf die Exit-Quotes | Der Fehlerpfad committet `outputs/options_archive/` (append-only Beobachtungen, nie halbfertige Zustände) | `test_feedback_persists_*` |
| 9 | Ein gescheiterter Push (Konflikt, Netz) verlor die Daten mit dem kurzlebigen Runner | `actions/upload-artifact` bei Fehler (14 Tage), Wiederherstellung siehe §5 | ebd. |
| 10 | Monitor-Flag „Handelstag“ basierte nur auf dem Wochentag | NYSE-Kalender, NY-Datum | – |
| 11 | Settlement-Marks verfallener Kontrakte trugen außerhalb der Sitzung `mark_source = archive_rth` (falsche Herkunft) | Herkunft wird nur gesetzt, wenn tatsächlich eine Archiv-Quote zählt | `test_settlement_mark_traegt_keine_archiv_herkunft_*` |
| 12 | Tradier-Datumsfeld `0` wurde als 1970 gelesen (Quote fälschlich STALE und für Marks unbrauchbar) | `0`/leer gilt als „kein Datum“, das nächste Feld zählt | `test_zero_date_field_is_not_an_exchange_time` |
| 13 | Eröffnungsphase nur im Capture gemieden; Live-Marks und Archiv-Quotes um 09:35 ET zählten | Ein gemeinsames Mark-Fenster `option_quotes.mark_window` (Eröffnung + 30 min bis Schluss − 10 min) für Capture, Live-Marks und verwertbare Archiv-Quotes | `test_mark_quotes_*` |
| 14 | Mark-Tag war der Abruftag, auch wenn die Börsenzeit vom Vortag stammte | Mark-Tag = NY-Tag der Börsenzeit (`quote_day`) | `test_mark_day_is_the_quotes_exchange_day_not_the_retrieval_day` |
| 15 | Kalender-Config: Pfad hing am Arbeitsverzeichnis, und ein unquotiertes `13:00` verwarf die ganze Datei (auch Sonderschließungen) | Absoluter Pfad; Einträge werden einzeln geprüft, `13:00` (YAML-Minuten) wird gelesen | `test_calendar_override_*` |
| 16 | `ci_push.sh` beendete sich ohne gestagte Änderung, ohne einen lokal committeten, ungepushten Commit zu pushen | Ausstehende lokale Commits werden trotzdem gepusht; `origin/main` wird nach dem Push nachgeführt | `test_ci_push_pushes_pending_local_commit_*` (Bare-Repo, ohne Netz) |
| 17 | Defekte `history.json` stoppte den Capture still (Job grün) | Capture läuft für Kandidaten-Marks weiter (Warnung); ein echter Fehler macht den Job jetzt rot (Exit 1) | `test_capture_survives_broken_history_*` |

Details zu #1, `options_capture.yml`:
- 6 gestaffelte Crons; einer davon trifft das Fenster bei jeder GitHub-Verzögerung von 0–7 h, in Sommer und Winter und an verkürzten Tagen.
- Das Fenster ergibt sich aus dem Kalender: frühestens 30 min nach Eröffnung, spätestens 10 min vor Schluss.
- Idempotent je Symbol und NY-Handelstag.
- Schreibt nur `outputs/options_archive/` (Union-Merge) und hat eine eigene Concurrency-Gruppe.

## 3. Verträglichkeit mit den Entry-/Exit-Regeln

- **Entry (Champion):**
  - Unverändert erfolgt der Entry zum Scanner-Lauf.
  - Der Guard verhindert jetzt nur in beiden Jahreszeiten einen Start vor 09:30 ET, so wie bisher im Sommer.
- **Exit (Champion):**
  - Unverändert entscheidet `feedback.py` zu seinen Laufzeiten.
  - Die Marks und die Capture-Quotes beeinflussen keine Exit-Entscheidung; `history.json` wird von ihnen nie geschrieben.
- **Kandidaten-Marks (5/20/45/120 Kalendertage):**
  - Der Mark entsteht am ersten Handelstag ab Fälligkeit mit einer RTH-Quote zwischen 10:00 ET und 10 min vor Schluss.
  - Die Tageszeit schwankt dabei (je nach Cron-Verzögerung) innerhalb dieses Fensters.
  - Bei Horizonten in Kalendertagen ist das unkritisch; Abweichungen werden über `*_mark_date_{h}d` und `late` sichtbar.

## 4. Datenabdeckung: Ursachen (reale Daten, 2026-10-11)

### 120 geschlossene Champion-Trades

| Befund | Anzahl | Ursache | vermeidbar? |
|---|---|---|---|
| UNUSABLE `EXIT_PRICE_UNKNOWN` | 74 | `outcome_method` nicht erfasst (Schließungen 2026-05-26 bis 2026-09-23) | **historisch nicht rekonstruierbar** (kein archivierter Quote, Preistyp unbekannt) |
| UNUSABLE `EXIT_PRICE_UNKNOWN`, rekonstruiert | 40 | wie oben, nachträglich rekonstruiert | **historisch nicht rekonstruierbar** |
| DATA_BAD `ENTRY_DEBIT_INVALID` | 3 | Entry-Debit fehlt (Altbestand) | **historisch nicht rekonstruierbar** |
| OK (`spread_quote`) | 3 | alle Schließungen seit Beginn der Methodenerfassung (29.09.) | – |

Ab Beginn der Methodenerfassung gab es keine vermeidbare Lücke: 3 von 3 Trades sind OK. Neu erfasst werden ab sofort Entry-Quotes mit Zeitstempel und RTH-Exit-Quotes (Capture).

### Fällige 5-Tage-Marks

| Mark | Fällig | Erledigt | Offen: Ursache |
|---|---|---|---|
| `real_strat_ret_5d` | 153 | 26 | 120: Single-Leg-Strategien (80 LONG_CALL, 40 LONG_PUT) wurden nie gemarkt (Fehler, in PR #111 behoben). Sie werden nachgeholt, aber als `LATE_MARK` geflaggt: Ein pünktlicher Mark ist **nicht rekonstruierbar**, weil zur Fälligkeit keine Quote archiviert wurde.<br>7: erst am Sonntag, 2026-10-11, fällig (regulär, Mark am Montag). |
| `real_opt_ret_5d` | 153 | 112 | 40: am Sonntag fällig (regulär).<br>1: seit 2026-10-05 überfällig, weil der Scanner an dem Tag erst nach Schluss lief. **Künftig vermeidbar** (Capture). |

## 5. Persistenz zwischen Actions-Läufen

Der Runner ist kurzlebig; dauerhaft ist nur, was nach `main` gepusht wird (`scripts/ci_push.sh`):
- 5 Push-Versuche mit Rebase.
- `*.jsonl` per Union-Merge.
- Abgeleitete `outputs/state|health/*.json` in der Version dieses Laufs.
- Jeder andere Konflikt führt zu einem roten Lauf; es wird nie still eine Seite verworfen.

| Daten | Pfad | Schreiber (Concurrency-Gruppe) | Sicherung |
|---|---|---|---|
| Options-Quotes | `outputs/options_archive/quotes/YYYY-MM-DD.jsonl` | Scanner, Feedback (`history-write`), Capture (`options-capture`) | Erfolg: Commit; Fehler: Commit des Archivs plus Artefakt; append-only, Union-Merge |
| Kandidaten-Ledger (Evidenzbasis, Marks) | `outputs/candidate_ledger/*.jsonl` | nur Scanner und Feedback (eine Gruppe, also seriell); Capture schreibt nie | Marks direkt nach dem RTH-Schritt committet; Fehler: Artefakt |
| Evidence-Ledger und Scorecard | abgeleitet aus Ledger, `history.json` und Archiv | – | jederzeit neu berechenbar; Momentaufnahme in `outputs/monitoring/ops_status.json` |
| Modellartefakte | `outputs/models/meta_label/` | nur Feedback | Commit `outputs/`; Fehler: Artefakt |
| Monitoring-Historie | `outputs/monitoring/ops_history.jsonl`, `alert_state.json` | nur Scanner-Job | Erfolg und Fehler: Commit |

Weitere Punkte:
- Keiner der Pfade ist per `.gitignore` ausgeschlossen (Test `test_persistent_data_is_not_gitignored`).
- **Wiederherstellung nach gescheitertem Push:**
  1. Run in Actions öffnen und Artefakt `scanner-data-<run>` (bzw. `feedback-data-*` / `options-capture-*`) herunterladen.
  2. Die Dateien in einen lokalen Checkout von `main` kopieren.
  3. Bei `*.jsonl` nur fehlende Zeilen anhängen (keine Duplikate).
  4. Per PR einspielen.
- **Restrisiko Concurrency:** GitHub hält je Gruppe nur einen wartenden Lauf. Ein dritter Trigger verdrängt den älteren wartenden Lauf.
  - Betroffen ist `history-write` (Scanner mit zwei Crons plus Source-Health-Trigger, dazu Feedback).
  - Ein verdrängter Scanner-Trigger ist harmlos, weil er ohnehin `SKIP` ergäbe.
  - Ein verdrängter Feedback-Lauf wird vom zweiten täglichen Feedback-Lauf aufgefangen.
  - Capture hat eine eigene Gruppe und verdrängt dort nichts.

## 6. Entscheidungen zu Champion-Zeitpunkten (2026-10-11 vom Nutzer getroffen, umgesetzt als timing-v2)

Die drei früher offenen Punkte sind entschieden und versioniert umgesetzt (`config/timing_policy.yaml`, `docs/TIMING_POLICY.md`). Der Altbestand bleibt timing-v1 und wird getrennt ausgewertet.

1. **Feedback nach Börsenschluss:** Feedback darf nach Schluss laufen und Exit-Entscheidungen treffen. Das wirtschaftliche Outcome entsteht aber erst zu einer zulässigen Optionsquote ab dem Exit-Entscheid, bei Nachbörse also in der nächsten Sitzung. Bis dahin liegt der Trade in `exit_pending_trades`; ohne zulässige Quote endet er als `closed_trades_unusable` mit Grund. Basiswert- oder Modellpreise sind nie Ersatz.
2. **Scanner-Start nach Schluss:** Scan, Research und Signale laufen weiter. Der Entry wird frühestens in der nächsten zulässigen Optionssitzung ausgeführt. Der Entscheidungszeitpunkt bleibt gespeichert (`decision_time`), und kein Fill liegt vor der Entscheidung.
3. **Feiertage:** Feedback verarbeitet vorhandene Daten und erstellt Berichte. Es macht keine kursbasierte Exit-Prüfung und erzeugt keine neuen Markt-Outcomes aus Vortagesquotes.

Weiterhin offen:
- **Sonderschließungen pflegen:** Neue NYSE-Ankündigungen in `config/market_calendar.yaml` per PR eintragen.
- **Secrets:** sind laut Nutzer gesetzt.

## 7. Unabhängige Prüfung

Ein separater Test-Agent mit eigenem Testentwurf hat `tests/test_ops_readiness_independent.py` erstellt, mit über 380 Tests:
- **Kalender:** Feiertage und verkürzte Sitzungen 2022–2027 gegen die offiziellen NYSE-Listen; alle vier Zeitumstellungen 2026/27.
- **Capture-Abdeckung:** jeder Handelstag 2025–2027 bei jeder Verzögerung von 0–7 h.
- **`ci_push.sh`:** End-to-end gegen ein lokales Bare-Repo mit parallelen Schreibern.
- **Testgüte:** 36 gezielte Mutationen, alle gefangen.

Seine drei Fehlerbefunde und die berechtigten Risiken sind behoben (#11–#17).

Bewusst verbleibende Restrisiken:
- **Verfallene Kontrakte:** Alle fälligen Horizonte werden aus dem Settlement bewertet (Mark-Tag = Lauftag, als late geflaggt). Das ist bestehende Spezifikation.
- **Illiquide Optionen:** Eine Börsenzeit älter als 26 h (STALE) macht die Quote nie verwertbar. Sie wird je Handelstag höchstens sechsmal erneut angefordert.
- **Cron-Abdeckung an verkürzten Sommer-Sitzungen:** Sie endet bei 7 h 10 min Verzögerung; nächster Fall ist 2028-07-03, 2026/27 gibt es keinen. Gemessenes Maximum der Verzögerung: 7 h 38 min, einmalig, beim Scanner.
- **Call/Put beim Settlement:** Die Kandidaten-Settlement-Logik bestimmt Call/Put über `direction`. Die aktuellen Daten sind konsistent: 188 BULLISH/C, 85 BEARISH/P.
- **Erste echte Funktionsprobe erst nach dem Merge:** `options_capture.yml` ist erst nach dem Merge per `workflow_dispatch` auslösbar.

## 8. Merge-Checkliste

- [ ] CI `pytest` auf dem aktuellen Head von PR #111 grün (push und pull_request).
- [ ] Entscheidungsregeln unverändert (Review des Diffs):
  - Exit-Regeln in `feedback.py` (Stop, Target, Haltedauer) und Gate-Reihenfolge in `pipeline.py` sind nicht angefasst.
  - Die Änderungen in beiden Dateien betreffen nur Outcome-Erfassung, Bewertung und Quote-Archivierung.
  - `config/promotion_policy.yaml` ist unverändert.
  - Kill-Switch neu mit `active: false`.
  - Adapter/Controller: nur Kill-Switch, Meta-Label-Feld, `ALLOW_SIZE_DOWN`-Abbildung; alle Verträge bleiben bei Einfluss `NONE`.
  - EA: nur die Hash-Prüfung der Expressions.
- [ ] Secrets vorhanden: `TRADIER_API_KEY`, `FRED_API_KEY`, `GMAIL_SENDER` / `GMAIL_APP_PW` / `NOTIFY_EMAIL`.
- [x] Entscheidungen aus §6 getroffen (2026-10-11) und als timing-v2 umgesetzt; Altbestand bleibt timing-v1.
- [ ] Timing-Diff geprüft: `feedback.py` ändert nur, wann und zu welchem Preis ein Exit als ausgeführt gilt, nicht den Exit-Entscheid selbst (`docs/TIMING_POLICY.md`).
- [ ] Nach dem Merge, erster Handelstag:
  - `options_capture.yml`: mindestens ein Lauf mit `IN_WINDOW`, die übrigen mit `SKIP`;
  - Commit `chore: options capture` auf `main`;
  - Scanner-Lauf: Guard `RUN`, Schritt „Options-Marks (RTH)“, Commit `chore: options marks (RTH)`;
  - `outputs/monitoring/ops_status.md` erzeugt und Mail angekommen (nur bei neuen, geänderten oder behobenen Warnungen);
  - im Monitor-Bericht `exit_active.mark_today` > 0 und die Zahl offener 5-Tage-Marks sinkend.
  - im Monitor-Abschnitt „Timing-Policy“: neue Trades `timing-v2`, Altbestand `timing-v1`; keine Warnung `TIMING_UNUSABLE` ohne erklärbaren Capture-Ausfall.
- [ ] Danach: Daten sammeln, keine grundlegenden Modelländerungen.
