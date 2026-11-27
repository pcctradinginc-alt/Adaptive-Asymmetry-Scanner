# Prüfung des externen Audits vom 26.11.2026

Geprüft gegen `main` 8b617861 und die eingecheckten Daten (Tagesberichte, Kandidaten-Ledger, Option-Learner-Status,
Scanner-Logs der Läufe 23.11. und 25.11.). Grundsatz: nur belegte Fehler beheben. Nichts ändert die
vorregistrierte Champion-Entscheidungslogik (`champion-v3`, Hash unverändert), keine Schwelle wird nach
beobachteten Ergebnissen verschoben, keine neue Datenquelle, kein neues KI-Modul.

## Bewertung der Befunde

| ID | Audit-Befund | Prüfung | Entscheidung |
|---|---|---|---|
| AUD-001 | SEC-TTM-EPS summiert die vier jüngsten Einträge | **Bestätigt.** companyfacts liefert je 10-Q Quartal *und* kumulierten Wert mit demselben Periodenende; die alte Summe ergibt im Testbeispiel 15,4 statt 8,7. Praktische Wirkung bisher gering: in 404 von 427 Trades war `eps_drift` 0, weil EDGAR meist keinen Wert lieferte. | **Behoben** (`ttm_eps_from_units`: Geschäftsjahr bzw. GJ + YTD − Vorjahres-YTD bzw. 4 Einzelquartale, Revisionen nach `filed`, sonst `None`). |
| AUD-002 | Data Health ≠ wirtschaftliche Gesundheit | **Richtig**, und am 25.11. konkret eingetreten (siehe NEU-1): Data Health „proceed“, aber der Markt-Provider fiel aus. Getrennte Kennzahlen existieren bereits (Data Health, Champion-Protokoll, Learner-Verdikt). | Über NEU-1 adressiert; kein neuer Score. |
| AUD-003 | `main` ungeschützt | **Bestätigt** (Issue #202). Zusätzlich: `main` hat seit 23.11. 18:24 UTC eine neue Wurzel – Bot-Commit d26b9cdc ohne Eltern mit 17.095 Dateien; der Code entspricht dem Stand davor, die Git-Historie auf `main` ist abgeschnitten. | Nur durch den Owner lösbar (Rulesets, siehe #202). |
| AUD-004 | Safe Mode schützt nur Research-Einfluss | Bestätigt, **bewusste** Architektur; Champion ist Paper-Trading und reine Messung. | Keine Änderung. |
| AUD-005 | PIT-Reader prüft Revisionen nicht | **Teilweise.** Jede Revision ist eine eigene Beobachtung mit eigenem `available_at`; alle Konnektoren setzen `vintage_time <= available_at`. Lücke nur bei inkonsistenten Zeitstempeln. | **Gehärtet** (`vintage_time <= t`) + Negativtest. |
| AUD-006 | Sentimentfehler als 0.0 | Code bestätigt, **Daten: 0 Fälle** (10.075 von 10.075 Ledger-Zeilen `sentiment_status=ok`); Status steht im Ledger. | Keine Änderung (Learner-Feature-Änderung würde die vorab festgehaltenen Prognosen entwerten). |
| AUD-007 | Ledger-Schreibfehler wird ignoriert | **Bestätigt**; bisher kein verlorener Tag (36 Tage geprüft). | **Behoben:** `flush()` liefert Status, Tagesbericht `stats.candidate_ledger_flush` mit `research_evaluable`. |
| AUD-008 | Sektorfilter nimmt BULLISH an | Bestätigt, aber konsistent mit der Produktion (Bearish abgeschaltet); Ledger und Zensus enthalten alle Kandidaten vor dem Filter. | Keine Änderung (Champion eingefroren). Bei Wiederzulassung von Bearish richtungsneutral machen. |
| AUD-009 | 30 % Friktion erlaubt | Zahl stimmt; das Audit selbst rät von willkürlicher Änderung ab. Teil des Champion-Hashes. | Keine Änderung; Ausführungskosten werden im Schatten gemessen. |
| AUD-010 | `news_novelty` 0 % Abdeckung | Bestätigt; Varianten mit 80–93 % werden bereits getestet. | Keine Änderung (Feature-Set ändern = neue Experimentversion). |
| AUD-011 | Census- und Directional-Kohorte vermischt | **Bestätigt** (5d: 507 Directional / 980 Census; Walk-Forward 91 von 92 Entscheidungen aus dem Census). | Draft-PR #206 (Zahlen geprüft) übernommen; der Kohorten-Challenger bleibt eine separate, zu registrierende Entscheidung. |
| AUD-012/014/015 | MC-Kalibrierung, 8 Trades, Learner unter Referenz | Zahlen bestätigt (8 verlässliche Trades, Trefferquote 12,5 %, PF 0,18; Learner −3,87 % vs. Referenz −1,39 % vs. nicht handeln 0 %, 6 Tage, 2 Perioden). **Korrektur:** Die 8 Trades sind die Paper-Historie vor `champion-v3`; v3 hat noch 0 verlässliche Trades. | Keine Codeänderung; das Champion-Protokoll prüft vorab festgelegt bei 30/60/90 Trades. |
| AUD-013 | Watchdog „PENDING“ trotz Erfolg | Bestätigt als Zeitpunkteffekt: der Monitor wird im laufenden Scanner-Job erzeugt. | Keine Änderung. |

## Zusätzliche Befunde (nicht im Audit)

**NEU-1 – Provider-Ausfall wurde als Marktbefund verbucht (25.11.2026).** yfinance lieferte für 473 von 518 Tickern
HTTP 401 („Invalid Crumb“). Die eingefrorene V1-Logik zählt ein info-Dict ohne `marketCap` als „Market Cap < 2 Mrd.“:
462 S&P-500/Nasdaq-100-Werte galten als zu klein, der Tag endete mit „Keine Kandidaten nach Hard-Filter“, der
Tagesbericht existierte, der Guard hielt den Tag für versorgt – kein Ledger-Eintrag, kein Recovery-Lauf.
*Behoben:* `hard_filter_outage()` (≥ 50 % ohne Daten oder Kontrollabfrage AAPL/MSFT/NVDA ohne Börsenwert und
≥ 50 % no_data + „zu klein“) → `stats.data_failure`, kein LLM/Trade; `artifact.unless_field: stats.data_failure`
macht den Tag für Guard und Watchdog zu „nicht versorgt“ (höchstens eine Recovery je Tag, wie bisher). Die
V1-Filterlogik (`_evaluate_ticker`) bleibt unverändert.

**NEU-2 – Der unabhängige LLM-Zensus hat nie eine Kohorte ausgewertet.** Seit dem 11.11. wurden 9 Kohorten
eingefroren (`outputs/research/llm_census_inputs/`), aber `outputs/research/llm_census_offline/` existiert nicht:
Der Workflow übergab `--day "$(date -u)"`, der 22:30-UTC-Cron startet real 4–7 h später am nächsten UTC-Tag –
für diesen Tag gibt es nie eine Eingabe. Zusätzlich verwarf `DeepAnalysis.run` jede Red-Team-VETO-Analyse; der
Zensus hätte nur die Nicht-VETO-Teilmenge gemessen (10.11.: beide gezogenen Kandidaten VETO → „no_analysis“, obwohl
bezahlt), und der Bericht las den Offline-Zensus gar nicht.
*Behoben:* Ohne `--day` werden zeitnah (≤ 36 h) eingefrorene Kohorten ohne Ergebnis ausgewertet, ältere einmalig
als `SKIPPED stale_cohort` vermerkt (keine nachträgliche LLM-Analyse mit späteren Kursen); `keep_vetoed` nur für
den Zensus (Produktion unverändert), `census_red_team_verdict` und `by_red_team` im Bericht; der Bericht verbindet
die Offline-Ergebnisse mit den Forward-Renditen des Ledgers. Kosten: k = 5 Analysen je Handelstag, Budget-Guard
und Monatsdeckel (10 USD) unverändert.

**Nachtrag 27.11.:** Die Recovery für NEU-1 hätte so nicht gegriffen – der Watchdog zählte den Scanner-Direkt-Dispatch
aus Source Health als Recovery und hielt einen veralteten Snapshot für „versorgt“. Behoben in
`docs/SCIENTIFIC_AUDIT_2026-11-27.md` (DA-01/DA-02).

## Nicht umgesetzt, Owner-Entscheidung

* Branch-Schutz/Rulesets für `main` (Issue #202) und Klärung der abgeschnittenen Historie vom 23.11.
* Prospektiver Kohorten-Challenger (Census vs. Directional) als neue registrierte Hypothese mit Ledger-Alpha.
