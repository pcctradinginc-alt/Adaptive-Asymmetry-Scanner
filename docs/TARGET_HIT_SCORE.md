# Target-Hit Score (unkalibriert) – Bezeichnung und Verwendung (2026-10-09)

`simulation.hit_rate` / `mc_hit_rate` ist der Anteil der Monte-Carlo-Pfade, in denen das **Underlying** das simulierte Kursziel erreicht. Die Kennzahl misst **nicht**, ob die Option oder der Spread nach Bid/Ask, Slippage, Zeitwertverlust, IV-Effekt, Gebühren und Exit-Ausführung profitabel ist. Historisch lag der Score im Mittel bei rund 75 %, profitabel waren rund 31 % der Trades. Dieser Vergleich ist **explorativ**, weil er alle Outcome-Klassen enthält, auch die unsicheren.

**Bezeichnung in allen Reports:** `Target-Hit Score (uncalibrated)` – Ranking signal only. Die Erklärung steht in `modules/score_labels.TARGET_HIT_EXPLANATION`. Der Status lautet `uncalibrated`, solange `LIVE_FORWARD_CALIBRATION` nicht `CALIBRATED` ist (mindestens 30 prospektive RELIABLE-Outcomes, siehe `modules/learning_health.py`).

## Geänderte Labels

Nur Reporting; die gespeicherten Zahlen sind identisch.

| Ort | vorher | jetzt |
|---|---|---|
| Trade-Mail | „MC Hit-Rate: 70 % (Modell, P Kurs > Ziel) / Kalibriert: …“, Block „Wahrscheinlichkeiten“ | „Target-Hit Score: 70 % (uncalibrated) – Ranking signal only / Band-Win-Rate (Paper-Trades, deskriptiv)“, Block „Ranking-Kennzahlen“, Erklärung |
| Tagesbericht | „**Hit-Rate:** 70.2 %“ | „**Target-Hit Score (uncalibrated):** 70.2 % … Ranking signal only, keine Gewinnwahrscheinlichkeit“ + Erklärung |
| Montagsbericht | „MC-Band-Wahrscheinlichkeit“, „vorhergesagt“, „kalibriertes MC-Band“, „Modell sagte“ | „Target-Hit Score (uncalibrated)“, „Band-Win-Rate (deskriptiv)“, „Ø Target-Hit Score“, „Target-Hit-Score-Band“ + Erklärung |

## Verwendung im Code (unverändert, nur dokumentiert)

| Ort | Verwendung | Bemerkung |
|---|---|---|
| `pipeline.py` Pre-MC / Quick MC / Final MC | Gates `hit_rate >= Schwelle` | Gate auf einem Ranking-Score, nicht als Wahrscheinlichkeit bezeichnet |
| `modules/options_designer.py` `_compute_roi` | `roi_delta = move × delta × leverage × min(0.95, score)` | **Probability-of-Profit-Semantik in der Logik:** Der Score wird als Wahrscheinlichkeitsgewicht eines „Erwartungswerts“ genutzt. Kommentar korrigiert, Logik unverändert → **Owner-Entscheidung** |
| `options_designer.py:552` | `qmc.get("hit_rate", 0.65)` | stiller Default 0,65, falls Quick-MC fehlt → Owner-Prüfung (nicht geändert) |
| `modules/deep_analysis.py` Prompt | „Hit-Rate: x %“ im LLM-Prompt | Produktionsinput; nicht geändert, weil sich sonst das LLM-Verhalten ändert |
| `modules/abstention_intelligence.py` | `p_model_wrong = 1 − Band-Win-Rate` | nutzt die gemessene Band-Win-Rate, nicht den Score als Wahrscheinlichkeit |

Eine echte Kalibrierung „Target-Hit Score → beobachtete Options-Gewinnquote“ entsteht erst, wenn genügend prospektive RELIABLE-Daten vorliegen. Bis dahin gilt `UNCALIBRATED` bzw. `NEED_MORE_DATA`.
