# Failure-Taxonomie (Trade-Gedächtnis)

Implementierung: `modules/trade_memory.py`. Die Regeln und Schwellen wurden **vor** der
Auswertung festgelegt und **nicht** auf Ergebnisse optimiert. Klassifiziert werden nur
Verlusttrades (`outcome < 0`). Jede Regel vergibt ein Label (Multi-Label); die primäre
Ursache ist das erste zutreffende Label in fester Priorität. Alles deterministisch, ohne LLM.

## Skalen (an `outputs/history.json` geprüft)

| Feld | Skala |
|---|---|
| `simulation.sigma`, `features.sigma_30d` | **tägliche** Vola als Anteil (Median 0,026 bzw. 0,025; Bereich 0,01–0,07) |
| `entry_quote.spread_pct`, `option.spread_ratio` | Anteil (0,05 = 5 %); Bereich 0,003–0,10 |
| `outcome`, `peak_return` | Options-Rendite als Anteil (-1 = Totalverlust) |
| `deep_analysis.data_confidence` | Text: `low` / `medium` (kein `high` in den Daten) |
| `deep_analysis.time_to_materialization` | Text: „4-8 Wochen“, „2-3 Monate“, „6 Monate“ |
| `simulation.current_price` | Underlying bei Entry; `close_price` = Underlying bei Exit |

## Regeln und Priorität

| # | Code | Regel | Begründung der Schwelle |
|---|---|---|---|
| 1 | `unreliable_outcome` | `outcome_method_reconstructed == "delta_approx"` -> keine weitere Klassifikation | Outcome nur geschätzt; jede Ursachenaussage wäre Scheingenauigkeit |
| 2 | `exit_gave_back` | `peak_return >= +0,30` und Endergebnis < 0 | +30 % ist deutlich mehr als Spread-/Bewertungsrauschen; Exit-Regel hätte Gewinn sichern können |
| 3 | `timing_too_early` | schwache Gegenbewegung (Schwelle von `signal_wrong` ≤ Underlying-Rendite < 0) **und** Untergrenze von `time_to_materialization` (Woche = 7, Monat = 30 Tage) > Haltedauer | Das Signal war innerhalb des erwarteten Zeitfensters noch nicht falsifiziert. Nur bei vorhandenem Feld. Ein „Katalysator nicht eingetreten“-Feld existiert nicht und wird nicht geraten |
| 4 | `signal_wrong` | signierte Underlying-Rendite < `-0,5 · sigma_daily · sqrt(Handelstage)`; Handelstage = Kalendertage · 5/7 (min. 1); ohne Sigma: < -2 % | 0,5σ des Haltedauer-Bereichs trennt Gegenbewegung von Rauschen; -2 % als Ersatz entspricht etwa einer typischen Tagesvola |
| 5 | `structure_decay` | signierte Underlying-Rendite >= 0, Option trotzdem im Verlust | Richtung stimmte -> Theta/IV-Crush/Spread/Strike-Wahl (Struktur) |
| 6 | `weak_adverse_move` | signierte Rendite in [Schwelle, 0) | Lücke zwischen 4 und 5: leicht falsche Richtung innerhalb des Rauschens; weder klares Signalversagen noch Struktur |
| 7 | `underlying_unknown` | `close_price` oder Entry-Preis fehlt (v. a. Shadow-Trades) | ehrlich „unbekannt“ statt geratener Ursache |
| 8 | `entry_cost_high` | `entry_quote.spread_pct > 0,10` (Fallback `option.spread_ratio`) | 10 % Bid/Ask-Spread ist ein hoher Round-Trip-Kostenblock; entspricht dem Rand der Pipeline-Filter |
| 9 | `low_data_confidence` | `data_confidence == "low"` oder eines der Kernfeatures (impact, surprise, mismatch, z_score, sigma_30d) fehlt | Entscheidung auf dünner Datenbasis |
| 10 | `regime_changed` | Kern-Regime (`fin_conditions`, `credit`, `trend`, `vix`) bei Entry und Exit unterscheiden sich in mindestens einem gemeinsamen Schlüssel (Neben-Labels wie Öl/Dollar ändern sich fast immer und wären unspezifisch) | reines Kontext-Label |
| 11 | `other` | nichts trifft zu | Restklasse |

Regel 8-10 sind Kontext-Labels: sie werden immer als Label vergeben, sind aber nur dann
primäre Ursache, wenn keine Regel 2-7 zutrifft. Begründung der Reihenfolge: zuerst die
Ursachen, die eine konkrete Gegenmaßnahme haben (Exit, Timing, Signal, Struktur), dann
Kontext.

## Bekannte Einschränkungen

* Regime kommen aus `factor_monitor` (VIX-Tagesreports, Makro-Archiv). Ein Label wird nur
  mit höchstens 3 Tagen Rückblick (PIT) übernommen. Über eine Haltedauer von ~45 Tagen
  ändert sich mindestens einer von 8 Schlüsseln fast immer; `regime_changed` ist daher
  unspezifisch und dient nur als Kontext.
* Underlying-Rendite und Handelstage sind Näherungen (Kalendertage · 5/7).
* `peak_return` ist nur für wenige Trades vorhanden; `exit_gave_back` ist deshalb selten.
* Sehr kleine Stichproben: Anteile beschreiben, sie beweisen nichts. Unzuverlässige
  Outcomes werden in der Aggregation getrennt ausgewiesen.
* Doppelte Kombination Ticker/Entry-Datum/Strategie ergibt dieselbe `case_id`; nur der erste Fall zählt.
* `similar_cases` mit `query_date` nutzt nur Fälle, deren Entry **und** Close vor dem Datum
  liegen (das Outcome war dann bekannt).
