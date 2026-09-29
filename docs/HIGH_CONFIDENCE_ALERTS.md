# High-Confidence Trade Alerts

**Keine Orderausführung.** `modules/hc_scanner.py` erzeugt ausschließlich
Research-Signale, als Datei und optional per E-Mail. Es gibt keinen
Broker-Anschluss; ein Test (`tests/test_hc_scanner.py`) stellt das sicher.

## Ablauf
Das Modul läuft in `.github/workflows/ml_research.yml`: wöchentlich am Samstag
nach den neuen Modellprognosen und monatlich nach der Meta-Validierung.

```
python -m modules.hc_scanner --dry-run   # nichts senden, keinen Zustand ändern
python -m modules.hc_scanner --send      # Alerts senden, Zustand und Gedächtnis fortschreiben
```

## Bedingungen (alle müssen erfüllt sein)
**Global.** Wenn eine dieser Bedingungen fehlt, gibt es keine Alerts:
1. Eine OOS-validierte Regel ist aktiv (`hc_thresholds.json`, erzeugt von
   `meta_learning.calibrate_hc_rule`).
   - Die Schwellen für Wahrscheinlichkeit und Modell-Übereinstimmung werden nur
     auf den **Kalibrierjahren** gewählt, nach der unteren 95-%-Schranke der
     Netto-Expectancy.
   - Validiert wird auf den späteren Jahren inklusive Locked-Holdout.
   - Die Regel ist nur aktiv, wenn die untere Schranke > 0 ist **und** die
     Validierung positiv ausfällt.
2. Die Wahrscheinlichkeiten des aktiven Ensembles sind kalibriert (ECE ≤ 0,05).
3. Die Unsicherheitsintervalle sind kalibriert: Die konformal korrigierte
   Abdeckung liegt innerhalb von ±5 Pp. um 80 %.
4. Es gibt keinen Feature-Drift: Die Regime-Merkmale liegen im 1–99-%-Band des
   Trainings.
5. Die Prognosen sind höchstens 8 Tage alt.

**Je Titel:**
- kalibrierte P(20d-Überrendite > 0) ≥ Regel-Schwelle
- Modell-Uneinigkeit ≤ Regel-Schwelle, nur wenn der Disagreement-Test (G) sie
  empirisch belegt hat
- Liquidität in der oberen Hälfte des Universums
- Datenqualität HIGH, erwartete 60d-Rendite > 0, Regime-Konfidenz nicht LOW
- mindestens 25 historische Analogien mit einem Gewinneranteil von mindestens
  50 %, Asymmetrie (Analog-MFE / |MAE|) ≥ 1
- ein bekanntes Earnings-Datum, das nicht innerhalb von 10 Tagen liegt; ist das
  Datum unbekannt, gibt es keinen Alert

Die Plausibilitätsfilter (Analogien, Asymmetrie) sind konservativ. Sie können
Alerts nur verhindern, nie erzeugen.

**VERY HIGH:** Die Wahrscheinlichkeit liegt mindestens 5 Pp. über der
Regel-Schwelle, und der Gewinneranteil der Analogien beträgt mindestens 60 %.

Eine Mindestanzahl gibt es nicht. „No high-confidence candidates“ ist der
Normalfall, solange keine Regel die OOS-Prüfung besteht.

## Alert-Inhalt
- **Identifikation:** Ticker, Firmenname, Signaldatum, Signal-ID, Modell- und
  Meta-Version, Confidence.
- **Wahrscheinlichkeit:** kalibrierte Wahrscheinlichkeit mit Definition.
- **Renditen:** erwartete 20d-Rendite (marktbereinigt, OOS-Mittel
  vergleichbarer Ränge) und 60d-Rendite (Median-Quantil). Die 120d-Rendite ist
  **nicht modelliert** und wird deshalb nicht erfunden.
- **Risiko:** Downside (unteres 80-%-Band), erwarteter MAE, MFE der Analogien,
  Asymmetrie.
- **Einordnung:** Modell-Übereinstimmung (Ränge, Streuung, Bull-Anteil),
  Regime-Kompatibilität (Regime, Konfidenz, Top-Modell und seine gemessenen
  Schwachstellen), Datenqualität, Analogien (Anzahl, Gewinneranteil,
  Median-Rendite).
- **Bull-, Bear- und Risikofaktoren:** Sie stammen aus gemessenen Daten, also
  aus kontrafaktischen Treibern, Failure-Profilen und Analogien. Es gibt keine
  LLM-Erzählung.
- **Invalidierungsbedingungen:** Bruch des unteren Bands, Kippen des
  wichtigsten Treibers, Rangverlust, Regimewechsel.
- **Unsicherheitshinweis:** Kalibrierte Querschnitts-Wahrscheinlichkeiten
  liegen realistisch nur wenige Prozentpunkte über 50 %.

## Deduplizierung
Der Zustand liegt in `outputs/research/alerts_state.json` (je Ticker:
signal_id, first_alert, last_alert, last_seen, Verlauf,
confidence_change, prediction_change). Zusätzlich gibt es das append-only Log
`alerts_log.jsonl`.

Ein Kandidat wird erneut gemeldet, wenn
- er neu ist oder nach mehr als 21 Tagen Pause neu entsteht (dann mit neuer
  signal_id),
- die Wahrscheinlichkeit um mindestens 5 Pp. steigt,
- die erwartete 20d-Rendite um mindestens 2 Pp. steigt,
- das Regime wechselt (VIX-Schwelle 20 oder SPY-Trend),
- die Confidence von HIGH auf VERY HIGH steigt.

Sonst wird nur der Verlauf fortgeschrieben, ohne Mail.

## Betreffzeilen
`[HIGH CONFIDENCE] TICKER – Adaptive Asymmetry Signal` bzw.
`[VERY HIGH CONFIDENCE] TICKER – Adaptive Asymmetry Signal`.

## Gedächtnis
Jeder `--send`-Lauf schreibt die Top-50 des aktiven Ensembles und alle
Kandidaten append-only nach `outputs/research/prediction_memory/`. Realisierte
Werte werden später als eigenes Ereignis angehängt: Rendite, Drawdown,
MFE/MAE, win/loss und Fehlerkategorie.

## Konfiguration
Der Versand läuft über `modules/mailer.py`, siehe `docs/WEEKLY_REPORT.md`.
Konfiguriert wird er nur über Env-Variablen/Secrets (`GMAIL_SENDER`,
`GMAIL_APP_PW`, `NOTIFY_EMAIL` bzw. `MAIL_*`/`SMTP_*`). Ohne Konfiguration
wird nicht gesendet, und der Status lautet `not_configured`.
