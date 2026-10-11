# Monitoring und Forward-Alpha-Controlling (Stand 2026-10-11)

Das Monitoring ist **read-only**. Es liest nur vorhandene Ledger, Archive und Zustandsdateien und schreibt ausschließlich nach `outputs/monitoring/`. Es verändert keine Trades, Modelle, Verträge, Konfigurationen oder Champion-Entscheidungen. EA-Shadow-Isolation, Promotion-Verträge und Risk Gates bleiben unberührt.

| Baustein | Aufgabe |
|---|---|
| `modules/ops_monitor.py` → `collect()` | Kennzahlen (Schema `ops-status-v1`) aus echten Ledgern |
| `modules/ops_report.py` → `run()` | Bericht, Trends, Warnungen mit Dedupe, Mail über `modules/mailer.send_mail` |
| `.github/workflows/scanner.yml`, Schritt „Ops-Monitoring (read-only)“ | automatische Ausführung nach jedem regulären Scanner-Lauf |

## 1. Wo der Bericht entsteht

Der Scanner-Job ruft nach dem Feedback-Loop `python -m modules.ops_report --mail` auf. Der Schritt läuft mit `if: always()`, damit auch ein Scanner-Fehler berichtet wird, und mit `continue-on-error`, damit er den Job nie blockiert. Er erzeugt:

| Datei | Inhalt |
|---|---|
| `outputs/monitoring/ops_status.json` | vollständiger Status, Trendzeile, Deltas, Warnungen (maschinenlesbar) |
| `outputs/monitoring/ops_status.md` / `.html` | menschenlesbarer Bericht (HTML = Mail-Inhalt) |
| `outputs/monitoring/ops_history.jsonl` | eine Kennzahlenzeile je Kalendertag (append-only; die erste Zeile des Tages ist kanonisch) |
| `outputs/monitoring/alert_state.json` | Dedupe-Zustand der Warnungen |

Committet wird der Bericht mit „Commit daily report“ (`outputs/`). Nach einem Scanner-Fehler gibt es einen eigenen Commit nur für `outputs/monitoring/`.

Ein schmaler Commit **vor** dem Tagescommit wäre falsch: `scripts/ci_push.sh` stasht nicht committete Dateien, die Tagesausgaben gingen dabei verloren.

## 2. Manueller Aufruf

```bash
python -m modules.ops_report                       # Bericht nach outputs/monitoring/ (keine Mail)
python -m modules.ops_report --out /tmp/ops        # in ein anderes Verzeichnis (eigene Historie/Zustand)
python -m modules.ops_report --dry-run-mail        # Mail nur vorbereiten; der Warnungszustand bleibt unverändert
python -m modules.ops_report --mail                # Mail bei NEW/CHANGED/RESOLVED (braucht Mail-Secrets)
python -c "import json; from modules import ops_monitor; print(json.dumps(ops_monitor.collect(), default=str)[:2000])"
```

Ein Beispielbericht aus **echten** Repo-Daten liegt unter `docs/examples/ops_status_2026-10-11.md` bzw. `.json`.

## 3. Automatische Ausführung aktivieren

1. Den PR mit diesem Branch nach `main` mergen. Der Scanner-Workflow läuft nur auf `main` (Checkout `ref: main`).
2. Mail-Versand: Die Secrets `GMAIL_SENDER`, `GMAIL_APP_PW` und `NOTIFY_EMAIL` nutzt der Scanner bereits; der Monitoring-Schritt bekommt dieselben. Ohne Secrets meldet `send_mail` `not_configured`, und der Bericht entsteht trotzdem.
3. Sonst ist nichts nötig: kein neuer Workflow, keine neue Abhängigkeit, kein Bezahldienst.

## 4. Was gemessen wird (Nenner und Ausschlüsse)

| Abschnitt | Kennzahl | Nenner / Definition |
|---|---|---|
| Lauf | Champion-Kandidaten, entworfene Kontrakte, Trades | Tagesreport: `stats.final_mc`, `stats.roi_ok + stats.roi_rejects.n`, `stats.trades` |
| Lauf | Schritt-Outcomes des aktuellen Jobs | `OPS_STEP_OUTCOMES` (RTH-Marks, Scanner, Feedback) |
| Quote-Abdeckung Entry | archivierte Entry-Quote (OK/SUSPECT) | Champion-Trades mit Entry im Zeitraum **und bekanntem Kontrakt** |
| Quote-Abdeckung Exit (aktiv) | Mark heute | aktive Champion-Trades mit Kontrakt |
| Quote-Abdeckung Exit (geschlossen) | beobachtet (archiviert oder nicht) bzw. Settlement | im Zeitraum geschlossene Champion-Trades |
| Vollständig bewertbar | Bewertung `OK` (trade-eval-v1) | im Zeitraum geschlossene Champion-Trades |
| Kandidaten-Marks | fällig / markiert / offen / verspätet | Real-Option- und Strategie-Kontrakte mit Entry |
| Lebenszyklus | Zustände und Probleme je Trade | siehe Abschnitt 5 |
| Forward Champion | n, unabhängige Tage, Σ/Ø netto, Trefferquote, Verluste, Drawdown, CI | nur Bewertung `OK`, netto = Brutto (Bid/Ask gekreuzt) − Kommission 0,65 $/Kontrakt/Seite; Modell- und Proxy-Preise separat (`not_evaluable`) |
| Forward Challenger | vermiedene Verluste, entgangene Gewinne, Δ gegenüber Champion | dieselbe eingefrorene Population: Decision-Ledger mit `decision_outcomes` |

Ausschlüsse werden immer ausgewiesen, nie still verworfen:
- `NO_CONTRACT`;
- Altbestand vor dem Archivstart (`ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE`);
- Kandidaten außerhalb des Snapshot-Budgets (`EXCLUDED_BUDGET_RANDOM_EXCLUSION`, deterministische Zufallsauswahl).

Fehlende Kennzahlen erscheinen als `N/A`, nie als 0.

## 5. Lebenszyklus

**Champion-Trade:** `SIGNAL → CONTRACT_SELECTED → ENTRY_OBSERVED → EXIT_PENDING → OUTCOME_FINAL → LEARNING_ELIGIBLE → LEARNED`.

Probleme je Trade:
- `NO_CONTRACT`;
- `ENTRY_NOT_CAPTURED(_BEFORE_ARCHIVE)`, `ENTRY_NOT_IN_ARCHIVE`, `ENTRY_UNUSABLE`;
- `EXIT_OVERDUE` (aktiv länger als `close_after_days` + 2);
- `OUTCOME_UNUSABLE` / `OUTCOME_DATA_BAD`;
- `LEARNED_NOT_ELIGIBLE`: Training trotz unzulässiger Daten;
- `ELIGIBLE_NOT_LEARNED`;
- Altbestand: `LEGACY_LEARNED_UNTRACKED`, `LEGACY_NO_STORED_EVALUATION`.

Exactly-once wird über die Consume-once-Schlüssel `<trade_id>:close:<Ziel>` geprüft. Ein Verstoß liegt nur vor, wenn ein nicht lernbarer Trade gelernt wurde oder ein lernbarer Trade doppelt vorkommt. Altduplikate aus dem April (nicht lernbar) werden gesondert gezählt.

**Kandidaten:** `SIGNAL`, `CONTRACT_SELECTED`, `ENTRY_PENDING`/`ENTRY_OBSERVED`, `EXIT_PENDING`, `EXIT_OVERDUE` (5-Tage-Mark mehr als 2 Tage überfällig), `OUTCOME_FINAL`, `LATE_MARK`.

## 6. Warnungen (Dedupe: NEW → ONGOING → CHANGED → RESOLVED)

| ID | Stufe | Auslöser |
|---|---|---|
| `SCANNER_FAILED:current_job` | CRITICAL | ein Schritt des laufenden Scanner-Jobs mit `failure` |
| `SCANNER_FAILED:*` / `ARCHIVE_FAILED:*` | CRITICAL/WARNING | Watchdog: MISSED, RECOVERY_EXHAUSTED oder `failure` |
| `QUOTE_COVERAGE_LOW:entry_7d` | WARNING | Entry-Abdeckung unter 80 % (ohne Altbestand), Nenner ≥ 3 |
| `QUOTE_COVERAGE_LOW:exit_active` | WARNING | Mark-Abdeckung aktiver Trades unter 80 % an Handelstagen |
| `QUOTE_COVERAGE_LOW:complete_30d` | INFO | vollständig bewertbar unter 50 %, Nenner ≥ 5 |
| `OUTCOMES_OVERDUE:*` | WARNING | überfällige Exits oder Marks (offen länger als 3 Tage) |
| `LEDGER_INTEGRITY` | CRITICAL | Promotion-Hash, Kette, Registry, Duplikate, Quote- oder External-Archiv-Hash |
| `LEARNING_EXACTLY_ONCE` | CRITICAL | Exactly-once verletzt oder Training trotz unzulässiger Daten |
| `TRAINING_INADMISSIBLE:*` | CRITICAL | Manifest: Label nach Trainingszeit bzw. unter Readiness; aktives Artefakt beschädigt |
| `PROMOTION_UNJUSTIFIED` | CRITICAL | Einfluss ungleich NONE ohne Integrität oder ohne aktiven Produktionszustand |
| `DATA_DEGRADATION:*` | WARNING | +2 Quellen FAIL/STALE/DEGRADED gegenüber dem Vorlauf; Quote-Archiv unter 50 % des 14-Tage-Medians |
| `MONITOR_SECTION_ERROR:*` / `MONITOR_COLLECT_FAILED` | WARNING | ein Abschnitt oder die Sammlung selbst ist fehlerhaft |

Gemailt wird nur bei **NEW, CHANGED oder RESOLVED**. Auch eine kritische Warnung wird ohne Statusänderung nicht täglich wiederholt; sie bleibt im Bericht (ONGOING) und im Job-Log (`::warning::`) sichtbar.

## 7. Was eine spätere Modell-Promotion blockiert (konkret)

Jeder Vertrag braucht ausschließlich Forward-Evidenz der **Einsatzpopulation**. Die folgenden Untergrenzen sind gepinnt in `config/promotion_policy.yaml` und nur strenger änderbar.

Mindestdaten:
- mindestens 50 Beobachtungen (Champion-Verträge v1: 60; FINAL_MC v2: 150);
- mindestens 20 unabhängige Signaltage (v2: 30);
- mindestens 90 Kalendertage Spanne;
- mindestens 2 beobachtete Regime;
- kein Sektor über 60 %;
- Mindestzahl an Regel-Treffern und Treffer-Tagen (v1: 15/8; v2: 30/10);
- v2 zusätzlich Ereignis-Cluster (80/15).

Promotion-Gate:
- Δ Expectancy und die CI-Untergrenze (Bonferroni über Familie × 12 Looks, Bootstrap über Signaltage) über den Schwellen;
- der Effekt bleibt nach Top-1-, Top-3- und Top-5-%-Trim positiv;
- mindestens 2 von 3 Zeitfenstern positiv;
- Kalibrierung und Drawdown nicht schlechter.

Automatisch höchstens `ABSTENTION_ONLY`. Sizing, Rerank und Gewicht gibt es nur per menschlicher Freigabe; Champion-Wechsel und neue Strategien nur per PR (`never_automatic`).

Sperren zusätzlich:
- Safe Mode: aktuell aktiv, Drift MODERATE pausiert Promotionen;
- Kill-Switch;
- Integritätsfehler;
- Population ungleich Champion: nur Empfehlung.

Meta-Label (PROM-META-001):
- Readiness 200 Labels und 20 Signaltage der Zielpopulation (Late-Stage/Champion), sonst `NEED_MORE_DATA`;
- danach volle Forward-Evidenz des Vertrags.

Stand 2026-10-11: alle 14 Verträge `PROSPECTIVE_CHALLENGER`, Einfluss `NONE`. Die Blockierungsgründe je Vertrag stehen im Bericht (z. B. „n 1/60; unabhängige Tage 1/20; Kalenderspanne 0/90 T …“).

## 8. Status und offene Datenlücken

| Frage | Antwort |
|---|---|
| Läuft der Scheduler? | Der Scanner läuft täglich, verifiziert über die Actions-API (letzte Läufe 06.–10.10.). Der Monitoring-Schritt ist **auf dem Branch vorbereitet** und läuft erst nach dem Merge nach `main`. |
| E-Mail getestet? | Nur technisch: die Versandentscheidung mit gemocktem `send_mail` und der Dry-Run. In dieser Umgebung gibt es keine Mail-Secrets; ein echter Versand wurde **nicht** getestet. |
| Daten, die fehlen | Siehe Liste unten. |

Fehlende Daten:
- Entry-Quotes aller bisherigen Champion-Trades. Das Archiv existiert erst seit dem 2026-10-10, der Altbestand ist nicht rekonstruierbar.
- Nur 3 von 120 geschlossenen Trades sind vollständig bewertbar.
- 41 Option- bzw. 127 Strategie-Marks (5 Tage) sind offen, weil RTH-Marks fehlten. Behoben durch den RTH-Schritt und den Single-Leg-Fix; die Marks werden nachgeholt und gegebenenfalls als verspätet geflaggt.
- Das Decision-Ledger hat erst 3 Entscheidungen und 1 Outcome, die Challenger-Evidenz ist praktisch leer.
- Meta-Label: noch keine 20-Tage-Labels (ab 2026-10-19).
- Zeitumstellung: Ab 2026-11-01 ist 13:30 UTC = 08:30 ET (Pre-Market). Startet der Scanner zur Cron-Zeit, überspringt
  der Schritt `Options-Marks (RTH)` mangels regulärer Session; dann fehlen Exit-Marks des Tages (Warnung im Bericht).
  Über Startzeit/Cron entscheidet ein Mensch. `session_of` kennt zudem keine US-Feiertage.
