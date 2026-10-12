# Timing-Policy: korrekte Handelszeiten und PIT-sichere Entry-/Exit-Logik (Stand 2026-10-11)

Diese Policy regelt nur, **wann und zu welchem Preis** ein Entry oder Exit wirtschaftlich als ausgeführt gilt. Sie gilt für echte Options-Outcomes: Champion-Trades, Counterfactual-Trades und Kandidaten-Optionen.

Unverändert bleiben:
- die Entscheidungen des Champions (Booking, Take-Profit/Stop-Loss/Haltedauer, `entry_debit`);
- Risk Gates, Kill-Switch und EA-Shadow;
- Promotion-Verträge.

Es gibt keine Orders und keine automatische Modellfreigabe.

Konfiguration: `config/timing_policy.yaml`. Code: `modules/timing_policy.py`. Kalender: `modules/market_snapshot.py` mit `config/market_calendar.yaml`.

## 1. Versionen

| Policy | Gilt für | Regel |
|---|---|---|
| `timing-v1` | Altbestand: alle Datensätze **ohne** Feld `timing_policy` | Entry zur Chain-Quote zum Scanner-Lauf, Exit zur Quote zum Feedback-Lauf, in jeder Session; Outcome sofort. Wird nie umgedeutet, umgerechnet oder überschrieben. |
| `timing-v2` | jede neue Entscheidung ab dem Merge (`timing_policy: timing-v2`) | Entry und Exit nur zu einer zulässigen Optionsquote, nie vor der Entscheidung; sonst PENDING, danach UNUSABLE mit Grund |

Auswertung immer getrennt:
- Scorecard-Zellen `…|timing=timing-v1` und `…|timing=timing-v2`;
- Monitor-Abschnitt „Timing-Policy“;
- Evidenz-Meta `timing_policy`.

Fehlt die Policy-Datei oder ist sie defekt, gilt für neue Entscheidungen die strengere `timing-v2`, nie still der Altbestand.

## 2. Zulässige Optionsquote (`option_quotes.usable_rth_quote`)

Eine Quote ist zulässig, wenn alle Bedingungen gelten:
- **Abrufzeit im Fenster:** Abruf (`retrieved_at`) im Fenster [Eröffnung + 30 min, Schluss − 10 min] eines NYSE-Handelstags.
- **Börsenzeit in RTH:** `quote_time` liegt in der regulären Sitzung.
- **Qualität:** nicht UNUSABLE und nicht STALE (Börsenzeit älter als 26 h).
- **Preise:** Bid > 0 und Ask > 0.

**Kalender:**
- NYSE-Feiertage (inklusive Good Friday und Juneteenth) und verkürzte Sitzungen (3. Juli, Tag nach Thanksgiving, Heiligabend; Schluss 13:00 ET).
- Sonderschließungen kommen aus der Config, z. B. 2025-01-09.
- Die Zeitumstellung wird über `America/New_York` gerechnet. Die europäische Umstellung hat keinen Einfluss; getestet für die Wochen, in denen USA und EU unterschiedlich umgestellt sind.

**Instrumentenspezifische Handelszeiten:**
- Aktienoptionen handeln bis 16:00 ET.
- Diese Wurzeln handeln bis 16:15 ET, an verkürzten Tagen bis 13:15 ET: SPY, QQQ, IWM, SPX, SPXW, XSP, NDX, NDXP, RUT, RUTW, VIX, VIXW. Die Liste ist in `config/market_calendar.yaml` hinterlegt und bewusst konservativ: Eine fehlende Wurzel lässt eine zulässige Quote nur ungenutzt, eine falsche würde unzulässige Quotes als zulässig werten.

## 3. Entry

- **Entscheidungszeitpunkt:** `decision_time` bei Champion- und Counterfactual-Trades, `signal_timestamp` bei Kandidaten. Er wird unverändert gespeichert.
- **Liegt die Entscheidung im Fenster:** Die Chain-Quote der Entscheidung zählt, wenn sie höchstens 15 min alt ist. Sonst zählt die nächste zulässige Quote ab der Entscheidung.
- **Liegt sie außerhalb des Fensters** (nachbörslich, vor Eröffnung, Feiertag, verspäteter Lauf): Ausgeführt wird frühestens in der **nächsten zulässigen Sitzung**, mit Abruf nie vor der Entscheidung.
- **Ausführungszeit:** `max(decision_time, Abrufzeit)`. Ein verspäteter Lauf wird nie so behandelt, als wäre eine frühere Ausführung möglich gewesen.
- **Preis:** Kauf zum Ask; beim Spread Long-Ask − Short-Bid, beide Legs aus derselben Erfassungsrunde (≤ 15 min Abstand).
- **Keine zulässige Quote in der frühesten Sitzung:** PENDING. Nach Fensterende plus 12 h Karenz (für verspätet committete Quotes) wird der Entry UNUSABLE (`NO_ADMISSIBLE_QUOTE_IN_SESSION`).

## 4. Exit

- **Exit-Entscheid:** Er bleibt Sache des Champions (`close_time`). Feedback darf nach Börsenschluss laufen.
- **Ausführung:**
  - Liegt der Exit-Entscheid im Fenster, zählt die zulässige Exit-Quote desselben Laufs (≤ 5 min vor dem Entscheid).
  - Sonst zählt die erste zulässige Quote der nächsten Sitzung.
- **Preis:** Verkauf zum Bid; beim Spread max(Long-Bid − Short-Ask, 0).
- **Verfall:** Liegt die früheste Ausführung nach dem Verfall oder ist am Verfallstag nach Fensterende keine Quote mehr möglich, gilt das **offizielle Settlement**: innerer Wert aus dem unbereinigten Schluss am Verfallstag. Fehlt der Schluss, bleibt der Exit PENDING und wird nach 5 Tagen UNUSABLE.
- **Settlement-Zeitpunkt:** Der Schlusskurs wird frühestens 1 h nach dem offiziellen Schluss abgefragt, nie ein noch laufender Intraday-Kurs (`SETTLEMENT_CLOSE_DELAY`).
- **Kein Ersatzpreis:** Basiswert- (auch nachbörslich) und Modellpreise sind nie Ersatz.

## 5. Lebenszyklus eines timing-v2-Champion-Trades

1. **Booking:** wie bisher, zusätzlich `decision_time` und `timing_policy`. Der Entry-Status (`trade.timing.entry`) wird je Lauf informativ aktualisiert. Ein einmal festgestellter Entry-Fill wird eingefroren: Eine später committete, früher abgerufene Quote ändert ihn nie nachträglich.
2. **Exit-Entscheid:** wie bisher. `outcome_at_decision` (Return zum Entscheid) bleibt zum Vergleich sichtbar.
3. **Weiterer Weg je nach Fill:**
   - Zulässiger Fill sofort möglich: `closed_trades` mit dem zulässigen Outcome.
   - Sonst: `exit_pending_trades`.
   - Ohne zulässigen Preis: `closed_trades_unusable` (Outcome `None`, Grund in `trade.timing.reasons`).
4. **Auflösung:** Sie geschieht in jedem Feedback-Lauf. Sie ist idempotent; Wiederholungsläufe erzeugen nichts doppelt.
5. **Bei Ausführung:**
   - Outcome-Record mit den tatsächlichen Fill-Quotes (`trade-eval-v1` ergibt dann `REALIZED_RTH_REPRODUCIBLE`). Bei Settlement nennt er keine Exit-Quote.
   - Das Label gilt in der Evidenz erst ab dem Tag nach der Ausführung als bekannt (`exit_fill_time`), nicht ab dem Exit-Entscheid (Point-in-Time).
   - Lernen genau einmal (consume-once).
   - Decision-Outcomes für Challenger erst dann.
6. **Counterfactual-Trades:** Sie folgen derselben Policy (fairer Vergleich), werden aber nicht gelernt.

## 6. Geschlossene Börsentage

An NYSE-geschlossenen Tagen (NY-Datum, Kalender):
- keine kursbasierte Exit-Prüfung;
- keine Counterfactual-, Trailing- oder Exit-Simulationen;
- keine neuen Markt-Outcomes aus übernommenen Vortagesquotes.

Ausstehende timing-v2-Exits werden weiter aus vorhandenen, zulässigen Archiv-Quotes aufgelöst. Ledger-Verarbeitung und Berichte laufen weiter.

Das gilt für alle Trades. Damit erzeugt auch der Altbestand an Feiertagen kein Outcome mehr; das ist eine bewusste, versionierte Änderung mit Regressionstest.

## 7. Bewahrung historischer Evidenz

- Altbestand ohne Policy-Feld bleibt `timing-v1`. Kein Datensatz, keine Rohquote und kein ursprüngliches Outcome wird überschrieben.
- Neue Felder werden nur ergänzt: `decision_time`, `timing_policy`, `timing`, `close_time`, `outcome_at_decision`.
- **Sichtbare Regressionstests:**
  - `test_feedback_v1_closes_immediately_v2_defers_to_admissible_quote`: gleicher Exit-Entscheid; v1 schließt sofort zum Entscheidungs-Return, v2 erst zur zulässigen Quote der nächsten Sitzung (Abweichung 0,60 → 0,30 im Testfall).
  - `test_flush_timing_v2_never_uses_unarchived_or_inadmissible_chain_quote`: derselbe Kandidat bekommt unter v1 einen Sofort-Entry, unter v2 bleibt er offen.
- **Monitor:** Er zeigt die Abweichung „Ausführung vs. Entscheid“ je timing-v2-Trade gemittelt.

## 8. Validierung (`tests/test_timing_policy.py`, `tests/test_timing_independent.py`)

Der unabhängige Test (Sonnet) umfasst 285 Tests mit zufälligen Szenarien 2025–2027 und Mutationsprüfung. Er fand 4 Fehler, alle behoben:
- Settlement vor dem offiziellen Schluss;
- falsche Entry-Quote im Outcome-Record;
- Exit-Quote bei Settlement im Outcome-Record;
- Label-Verfügbarkeit ab Entscheid statt ab Ausführung.

| Anforderung | Tests |
|---|---|
| Sommer-/Winterzeit, unterschiedliche Umstellungswochen USA/EU | `test_window_follows_new_york_not_europe`, Session-Grenzen beidseits jeder Umstellung |
| US-Feiertage, Sonderschließung, Early Close | `test_fill_day_skips_closed_days`, `test_instrument_specific_close`, Kalender 2022–2027 |
| Verzögerte Scanner- und Feedback-Läufe | Entry nach Schluss, Freitagabend nach Montag, Exit-Entscheid nach Schluss |
| Fehlende und verspätete Quotes | PENDING, dann UNUSABLE nach Karenz; Point-in-Time (Quote erst ab Abruf sichtbar) |
| Abgelaufene Optionen | Settlement Call/Put/Spread/wertlos; fehlender Schluss PENDING, dann UNUSABLE |
| Wiederholte Scheduler-Läufe | dritter Feedback-Lauf: kein doppeltes Outcome, kein doppeltes Lernen |
| Keine rückwirkenden Zeitpunkte | Property-Test über zufällige Entscheidungen und Quotes |
| Keine synthetischen Preise | `test_never_ok_without_option_quote`; unzulässige Quotes (Eröffnung, Schluss, Bid 0, STALE, gekreuzt) füllen nie |

## 9. Grenzen

- **Abhängigkeit vom Capture:** Wird in der frühesten Sitzung keine Quote erfasst (Capture-Workflow und Scanner-Schritt fallen beide aus), endet der Trade UNUSABLE statt mit einem späteren Preis. Das ist bewusst streng: Eine spätere Ausführung wäre eine andere Entscheidung.
- **Toleranzen:** 15 min für die Entry-Chain-Quote und 5 min für die Exit-Quote desselben Laufs. Beide sind dokumentiert und konfigurierbar.
- **Simulationen außerhalb der Policy:** Schatten-, Trailing- und Exit-Simulationen sind Research-Simulationen. Sie sind keine echten Options-Outcomes und keine Evidenz dieser Policy. An geschlossenen Tagen pausieren Trailing- und Exit-Simulationen; Schatten-Trades werten weiter nur historische Schlusskurse fälliger Horizonte aus.
