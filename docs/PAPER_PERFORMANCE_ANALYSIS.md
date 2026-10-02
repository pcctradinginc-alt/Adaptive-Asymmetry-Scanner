# Paper-Performance des täglichen Scanners: Ursachenanalyse (Audit P1-9)

**Quelle:** echte Paper-Trades aus `outputs/history.json`, kein Backtest.
Ausgewertet sind nur zuverlässige Outcomes (ohne `delta_approx`): **n = 79**
von 119 geschlossenen Trades.

**Reproduzierbar mit** `python scripts/paper_performance_analysis.py`
→ `outputs/research/paper_performance_analysis.{json,md}`.
**Test:** `tests/test_paper_performance_analysis.py`.

**Gesamt:**

| Kennzahl | Wert |
|---|---|
| Trefferquote | 35,4 % |
| Ø Ergebnis je Trade | +4,2 % |
| Median | −31,3 % |
| Profit-Faktor | 1,12 |

## Befunde (gemessen, n beachten)

### 1. Die MC-Trefferquote ist keine Gewinnwahrscheinlichkeit

| MC-Trefferquote (Bucket) | n | vorhergesagt | realisiert | PF |
|---|---|---|---|---|
| 0,55–0,65 | 8 | 0,60 | 0,25 | 0,37 |
| 0,65–0,75 | 27 | 0,71 | 0,22 | 0,42 |
| ≥ 0,75 | 29 | 0,86 | 0,41 | 1,22 |

Die Simulation setzt einen positiven Drift aus den LLM-Scores Impact und
Surprise an (`mirofish_simulation.MAX_SIGNAL_ALPHA`). Sie misst außerdem
„Kurs berührt das Ziel“ und nicht „Option schließt im Gewinn“.
**Konsequenz (umgesetzt):** Die tägliche Mail kennzeichnet die MC-Trefferquote
ausdrücklich als nicht kalibriert (`email_reporter.MC_CALIBRATION_NOTE`).

### 2. Die LLM-Scores ordnen die Trades nicht monoton

| Score | Wert | n | Trefferquote | PF |
|---|---|---|---|---|
| Impact | 4 | 12 | 50 % | 2,49 |
| Impact | 5 | 31 | 26 % | 0,78 |
| Impact | 6 | 30 | 37 % | 1,04 |
| Impact | 7 | 3 | 0 % | 0 |
| Surprise | 2–3 | 15 | – | ≥ 2,7 |
| Surprise | 4 | 24 | 21 % | 0,45 |

Höhere Scores bedeuten keine besseren Trades. Das deckt sich mit dem
Engine-Monitor: alle Feature-Korrelationen ≤ 0, Lern-Loop eingefroren.

### 3. Der Erfolg hängt an einem Zeitfenster

| Einstiegsmonat | n | PF |
|---|---|---|
| April 2026 | 46 | 1,64 |
| Mai 2026 | 25 | 0,56 |
| Juli–Sept. 2026 | 8 | überwiegend Verluste |

Das Gesamtergebnis stammt also fast vollständig aus einem Monat. Es ist kein
stabiler Edge.

### 4. Strategie

| Strategie | n | PF |
|---|---|---|
| Long Call | 36 | 1,67 |
| Bull Call Spread | 38 | 0,79 |

Spreads haben im Paper-Betrieb Wert vernichtet. Gründe sind wahrscheinlich die
gedeckelte Upside bei gleicher Trefferquote und doppelte Spread-Kosten.

### 5. Exits und Katalysatoren

| Gruppe | n | Ergebnis |
|---|---|---|
| Stop-Loss | 20 | 0 % Treffer, Ø −64 % |
| Take-Profit | 6 | 100 % Treffer, Ø +100 % |
| Earnings-Katalysator | 6 | PF 0,44 |

## Bewertung

Mit n = 79 und einem einzigen guten Monat ist **kein** belastbarer Edge des
täglichen Scanners nachgewiesen. Die Wahrscheinlichkeitsangaben der Mail (MC)
sind deutlich überkonfident.

**Empfehlung, Entscheidung des Menschen:**
1. Den Scanner weiter nur als Paper- und Research-Signal betreiben.
2. Spreads nur nach eigener Vorwärts-Evidenz.
3. Die MC-Trefferquote nicht als Wahrscheinlichkeit verwenden.
4. Die Abstinenz-Hypothese (VIX ≥ 20 oder SPY < SMA200) vorwärts auch für die
   tägliche Mail beobachten: Die schwachen Monate liegen überwiegend in ruhigen
   Aufwärtsphasen. Das ist zu prüfen, nicht zu unterstellen.
