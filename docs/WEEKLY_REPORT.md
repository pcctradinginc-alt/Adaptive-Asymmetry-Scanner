# Weekly Intelligence Report

## Zweck
Wöchentlicher Bericht (Montag) über Systemzustand, echte Paper-/Forward-Performance,
Modell- und Meta-Learning-Status, Kandidaten und Risiken. Er liest nur vorhandene
Ergebnisdateien und erfindet nichts: fehlt eine Eingabe, zeigt der Abschnitt
„keine Daten“ bzw. „n/a“.

**Research-/Paper-Signale, keine Orderausführung, keine Anlageberatung.**

## Abschnitte und Datenquellen
| # | Abschnitt | Quelle |
|---|-----------|--------|
| 1 | System Status (Champion, Meta-Modell, Regime, Pipelines, Freshness, Drift, Kalibrierung, Safe Mode) | `ml_research.json`, `meta_learning.json`, `meta_state.json`, `external_data/health/source_health.json` |
| 2 | Live / Forward Performance | `outputs/history.json` (`closed_trades`, `active_trades`), `candidate_ledger/*.jsonl` |
| 3 | Confidence Performance | `meta_learning.calibration_buckets[primary_meta]` |
| 4 | Current Model Intelligence | `meta_learning.model_intelligence` (Pfeil nur aus `trend`, `trend_t` angezeigt) |
| 5 | Meta-Learning Status | `meta_learning.approaches/deltas/decision`, `ml_research.json` (Backtest), `ml_forward.json` (Forward-Shadow) |
| 6 | High-Confidence Candidates | `hc_candidates.json` |
| 7 | Recent Signal Review (letzte 4 Wochen) | `history.json`, Verlierer-Ursache aus `trade_memory.jsonl` (Match Ticker + entry_date) |
| 8 | What the system learned | Diff gegen `outputs/reports/weekly_state.json` |
| 9 | Research Pipeline | `hypothesis_db.json`, `failure_analysis.json` |
| 10 | Risk / Health Warnings | regelbasiert aus den obigen Daten |

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
`Adaptive Asymmetry Scanner – Weekly Intelligence Report – YYYY-MM-DD`.
Exit-Code 1 nur, wenn der Versand endgültig scheitert; „not_configured“ ist eine Warnung.

## Workflow
`.github/workflows/weekly_report.yml`: Montag 05:47 UTC sowie manuell (`send` true/false).
Secrets `GMAIL_SENDER`, `GMAIL_APP_PW`, `NOTIFY_EMAIL`; anschließend werden `outputs/reports/`
committet und gepusht (Concurrency-Gruppe `research-write` wie ml_research).

## Sicherheit
Keine Credentials oder Adressen im Repo, in Tests oder Logs – ausschließlich Env/Secrets.
Tests mocken SMTP, es werden keine Mails gesendet und keine Netzwerkzugriffe ausgeführt.
Der Bericht führt keine Orders aus.
