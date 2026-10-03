# Monday Intelligence Report

## Zweck
Wöchentlicher Bericht (Montag) über Systemzustand, echte Paper-/Forward-Performance,
Modell- und Meta-Learning-Status, Kandidaten und Risiken. Er liest nur vorhandene
Ergebnisdateien und erfindet nichts: fehlt eine Eingabe, zeigt der Abschnitt
„keine Daten“ bzw. „n/a“.

**Research-/Paper-Signale, keine Orderausführung, keine Anlageberatung.**

## Struktur (seit 2026-10-03)
Neun Hauptabschnitte (sieben gemäß Spezifikation + FORWARD EVIDENCE + UNIVERSE V1 / V2), danach alle bisherigen
Detailabschnitte als **Anhang A1–A19** (Inhalt unverändert, nur neu nummeriert).

| # | Hauptabschnitt | Inhalt / Quelle |
|---|---|---|
| 1 | SYSTEM STATUS | Health, Data Quality, Safe Mode, Drift (kanonischer SystemState), Datenquellen-Zähler, Champion-Version, Meta-Modell, Zahl aktiver Hypothesen/Challenger/promoteter Hypothesen, Warnungen |
| 2 | WHAT THE SYSTEM LEARNED | letzte 7 Tage: Promotion-Übergänge (bestätigt/verworfen/demotet), verworfene Fabrik-Hypothesen, Blind-Spot-Cluster, Alpha Decay, gemessene Änderungen seit dem Vorbericht |
| 3 | HYPOTHESIS SCOREBOARD | gruppiert RESEARCH IDEA / CHALLENGER / FORWARD VALIDATED / PROMOTED / REJECTED. Forward N, unabhängige Tage, Zeitraum, Effekt, CI, Expectancy, Calibration, Produktionswirkung nur aus prospektiver Forward-Evidenz (PromotionController). Historische Ergebnisse (Fabrik, Hypothesen-DB) erreichen höchstens RESEARCH IDEA |
| 4 | RESEARCH INTELLIGENCE | neue Hypothesen, unorthodoxe Cross-Domain-Ideen, Data Gaps mit kostenlosen Quellen, wichtigste Forschungsfragen (Director-Priorität), Bereiche mit nachgewiesenem Informationswert (nur getestete Richtungen) |
| 5 | CURRENT MARKET / WORLD MODEL | World-Model-Dimensionen (Zustand, Score, Unsicherheit, Vorwoche) und Veränderungen |
| 6 | TOP TRADE CANDIDATES | **nur** Vorschläge der Produktionspipeline aus den Tagesreports der letzten 7 Tage; Research-/HC-Kandidaten nie. Je Kandidat alle verfügbaren Felder, fehlende als „n/a“ mit Grund. Ohne High-Confidence-Kandidat: `NO HIGH-CONFIDENCE TRADE THIS WEEK.` |
| 7 | PERFORMANCE | A) echte Forward-/Paper-Performance (Expectancy, Win Rate, PF, Sharpe/Sortino je Trade, MaxDD, N), Champion vs. Adaptive (prospektiv), Calibration (vorhergesagt vs. realisiert); B) Walk-Forward OOS der Research-Modelle; C) Backtest Meta-Learning – strikt getrennt |
| 8 | FORWARD EVIDENCE | A) Champion-Verträge (v1): N, unabhängige Tage, Spanne, E getroffen vs. Baseline, Δ Expectancy, Abstand zur Promotion; B) Final-MC-Verträge (v2, eigene Population): zusätzlich unabhängige Ereignis-Cluster, Survivors je Monat; C) ROI-Teil-Gates (`fail_gates`, Mindest-n 30). Unzureichende Evidenz wird ausdrücklich als `NEED_MORE_DATA` ausgewiesen. Champion- und Final-MC-Evidenz werden nie zusammengerechnet |
| 9 | UNIVERSE V1 / V2 | V1-Definition (eingefroren), V2-Snapshot (optionierbar/research/tradeable je Market-Cap-Bucket), V2-Shadow-Signale je Bucket (Netto-Expectancy, Spread, Slippage, Kosten), Segment-Verträge (N, Cluster, Tage, Spanne, Netto-Exp., Precision@3, Brier, Abstand zur Promotion, Status, Stufe) |

**High-Confidence (Berichts-Label, kein Trade-Gate):** Das kalibrierte MC-Band des Kandidaten muss auf
echten Paper-Trades belegt sein (n ≥ 20, Expectancy > 0, Profit Factor ≥ 1,2), und es darf kein Safe Mode
gelten. Die „kalibrierte Wahrscheinlichkeit“ ist die realisierte Win Rate dieses Bands (ab n ≥ 10),
nicht die Modellzahl. Erwartete 60d-Rendite/Drawdown stammen aus ML-Research-Modellen und sind als
„nicht produktiv validiert“ markiert.

## Anhang: Detailabschnitte und Datenquellen
| # | Abschnitt | Quelle |
|---|-----------|--------|
| A1 | System Status (Champion, Meta-Modell, Regime, Pipelines, Freshness, Drift, Kalibrierung, Safe Mode) | `ml_research.json`, `meta_learning.json`, `meta_state.json`, `external_data/health/source_health.json` |
| A2 | Live / Forward Performance | `outputs/history.json` (`closed_trades`, `active_trades`), `candidate_ledger/*.jsonl` |
| A3 | Confidence Performance | `meta_learning.calibration_buckets[primary_meta]` |
| A4 | Current Model Intelligence | `meta_learning.model_intelligence` (Pfeil nur aus `trend`, `trend_t` angezeigt) |
| A5 | Meta-Learning Status | `meta_learning.approaches/deltas/decision`, `ml_research.json` (Backtest), `ml_forward.json` (Forward-Shadow) |
| A6 | High-Confidence Candidates | `hc_candidates.json` |
| A7 | Recent Signal Review (letzte 4 Wochen) | `history.json`, Verlierer-Ursache aus `trade_memory.jsonl` (Match Ticker + entry_date) |
| A8 | What the system learned | Diff gegen `outputs/reports/weekly_state.json` |
| A9 | Research Pipeline | `hypothesis_db.json`, `failure_analysis.json` |
| A10 | Risk / Health Warnings | regelbasiert aus den obigen Daten |

Alle Eingaben liegen unter `outputs/` bzw. `outputs/research/` (siehe `--root`).

## Konventionen
- **Forward vs. Backtest:** Abschnitt 2 ist ausschließlich echte Paper-Performance
  aus `history.json`. Alles aus `ml_research`/`meta_learning` ist Backtest bzw.
  Walk-Forward und so beschriftet. `ml_forward.json` ist Forward-*Shadow* (keine Trades).
- **Zuverlässig vs. alle:** Tabelle A ohne Trades mit `outcome_method_reconstructed = delta_approx`,
  Tabelle B mit allen (Näherung).
- **Fenster:** nach `close_date` relativ zum Berichtsdatum: seit Beginn, 12 Monate (365 d),
  3 Monate (91 d), 4 Wochen (28 d). *Signals* = Trades (inkl. offen) mit `entry_date` im Fenster.
- **Outcome** ist die Rendite je Trade als Anteil (0.10 = +10 %). Payoff = Ø Gewinner / |Ø Verlierer|,
  Profit Factor = Σ Gewinne / |Σ Verluste|, Expectancy = Ø Outcome. Gewinner: Outcome > 0.
- **Max-Drawdown-Konvention:** kumulierte Outcome-Kurve in Reihenfolge `close_date`,
  additiv je Trade (jeder Trade = eine Einheit), Start 0; MaxDD = tiefster Abstand zum laufenden
  Maximum (inkl. Start), Einheit „Positionen“ (−1.0 = eine volle Einheit).
- **LOW SAMPLE:** n < 20 (`LOW_SAMPLE_N`).
- **Warnschwellen** sind Modulkonstanten in `reports/weekly.py` (`LOW_SAMPLE_N`, `DEGRADATION_MIN_N`, …).
  PERFORMANCE DEGRADATION: 4-Wochen-Expectancy < 0 und < langfristig, mind. 5 Trades.
- **Health-Aggregation:** `PASS`; `WARN`/`SCHEMA_CHANGED`/`STALE` → WARNING; `FAIL` → FAIL; `DEFERRED` wird separat gezählt.

## Lern-Snapshot (Abschnitt 8)
`outputs/reports/weekly_state.json` enthält Hypothesen-Status, Modell-Verdikte, Meta-Gewichte,
Kalibrierungs-Coverage, Champion, Safe Mode und Meta-Verdikt. Berichtet werden nur gemessene Diffs.
Ohne Snapshot: „Erster Bericht – keine Vergleichsbasis.“ Der Snapshot wird nur nach erfolgreichem
Versand (`--send`, Status `sent`) oder mit `--save-state` fortgeschrieben; `--dry-run` ändert keinen Zustand.

## Konfiguration (nur Umgebungsvariablen)
| Variable | Bedeutung |
|----------|-----------|
| `MAIL_PROVIDER` | `gmail_smtp` (Default) oder `smtp` |
| `GMAIL_SENDER`, `GMAIL_APP_PW` | Gmail (SMTP_SSL smtp.gmail.com:465) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS` | generischer SMTP (`SMTP_STARTTLS=true` → STARTTLS, sonst SSL) |
| `MAIL_FROM` | Absender (Default: `GMAIL_SENDER` bzw. `SMTP_USER`) |
| `MAIL_TO` | Empfänger, komma-getrennt (Fallback: `NOTIFY_EMAIL`) |

Neue Provider: Klasse mit `configured()` und `send(msg)` in `modules/mailer.py` (`PROVIDERS`) registrieren.
`send_mail(...)` liefert `sent | dry_run | not_configured | failed`, wirft nie, wiederholt begrenzt
(`max_retries=3`, linearer Backoff) und loggt Empfänger nur maskiert (`a***@domain`), nie Passwörter.

## CLI
```
python -m reports.weekly --dry-run [--date YYYY-MM-DD] [--root DIR] [--out-dir DIR]
python -m reports.weekly --send
python -m reports.weekly --dry-run --save-state    # Snapshot ohne Versand aktualisieren
```
Ausgabe: `outputs/reports/weekly_YYYY-MM-DD.{html,txt,md}` (Default). Betreff:
`Adaptive Asymmetry Scanner – Monday Intelligence Report – YYYY-MM-DD`.
Exit-Code 1 nur, wenn der Versand endgültig scheitert; „not_configured“ ist eine Warnung.

## Workflow
`.github/workflows/weekly_report.yml`: Montag 05:47 UTC sowie manuell (`send` true/false).
Secrets `GMAIL_SENDER`, `GMAIL_APP_PW`, `NOTIFY_EMAIL`; anschließend werden `outputs/reports/`
committet und gepusht (Concurrency-Gruppe `research-write` wie ml_research).

## Sicherheit
Keine Credentials oder Adressen im Repo, in Tests oder Logs – ausschließlich Env/Secrets.
Tests mocken SMTP, es werden keine Mails gesendet und keine Netzwerkzugriffe ausgeführt.
Der Bericht führt keine Orders aus.
