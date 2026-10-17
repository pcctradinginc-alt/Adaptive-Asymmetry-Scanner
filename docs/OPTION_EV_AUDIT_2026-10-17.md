# Quant-Audit: Options-Erwartungswert und Profitabilität (17.10.2026)

## Gegenstand

**Frage:** Erkennt der Scanner Optionsstrategien mit positivem erwartetem Netto-Ertrag, oder sagt er nur die Aktienrichtung voraus?

**Rahmen:** Das ist ein Audit, kein Redesign. Es gibt keine neuen Modelle, keine Parameteroptimierung und keine Änderung produktiver Handelsentscheidungen.
- Champion, Risk Gates, Kill-Switch, EA-Shadow und Promotion-Verträge sind unverändert.
- Fehler im Entscheidungsweg stehen unten nur als Vorschlag (§7).

**Methode:**
- Code gelesen und Formeln nachgerechnet.
- Drei unabhängige Prüfungen: MC/Volatilität, Pricing/ROI/Sizing, Entry/Exit/Outcome.
- Alle sechs Champion-Trades mit beobachtetem Options-Outcome rekonstruiert, aus `outputs/history.json` und `outputs/options_archive`.
- Bestehende Evidenz-Scorecard (`modules/evidence_ledger.scorecard`) und Shadow-Ledger ausgewertet.
- Synthetische Reproduktionen und Regressionstests in `tests/test_option_ev_audit.py`.

## 1. Entscheidungsweg

| Schritt | Code | Was wirklich berechnet wird |
|---|---|---|
| **Aktienprognose** | `deep_analysis` (LLM) | `impact`, `surprise`, Richtung. Daraus Signal-Alpha `sqrt(impact/10 · surprise/10) · 0,001`/Tag |
| **Volatilität** | `mirofish_simulation._get_hist_params` | σ und μ aus 6 Monaten Tagesschlüssen. μ ist auf ±0,5 %/Tag begrenzt, sonst 0 |
| **Pre-, Quick- und Final-MC** | `run_for_dte` | Wahrscheinlichkeit, dass ein Kursziel `max(8 %, 0,5·σ·√T)` innerhalb von T Schritten *berührt* wird. Drift = `hist_mu + Signal-Alpha`, mit Decay |
| **Optionsauswahl** | `options_designer` (Tier ≥ 150 T) | Long Call bzw. Bull Call Spread. Debit = Long-Ask − Short-Bid |
| **ROI-Gate, Stufe 1** | `_compute_roi` | `Zielbewegung · Delta(·0,8 bei Spread) · Hebel · p − 2·spread_pct(Long-Bein) − Vega-Verlust − Kommission` |
| **Edge-Gate** | `options_designer` | 45-T-Kursziel gegen Breakeven bei Verfall |
| **ROI-Gate, Stufe 2 (MC-P&L)** | `simulate_option_pnl` | GBM mit konstanter Drift, Bewertung des **Long-Calls** nach DTE/2. `roi_net` wird durch `mean_net` ersetzt, Gate: `mean_net ≥ min_roi` und `median_net ≥ 0` |
| **Sizing** | `position_sizing` | Kelly mit `b = roi_net/0,8` |
| **Entry** | `pipeline`, `timing_policy` | Long-Ask bzw. Long-Ask − Short-Bid |
| **Exit** | `feedback.check_exit_rules` | Gemessen am ausführbaren Wert (Long-Bid − Short-Ask), siehe Exit-Regeln unten |
| **Outcome** | `feedback.compute_outcome`, `timing_policy` | Exit/Entry − 1, netto abzüglich Kommission. Bei timing-v2 Fill an der nächsten zulässigen RTH-Quote |

**Exit-Regeln** (`feedback.check_exit_rules`):
- Long Call: Take-Profit +50 %, Stop −45 %.
- Bull Call Spread: Take-Profit bei 70 % des Maximalgewinns, Stop −50 %.
- Maximale Haltedauer 45 T.

## 2. Ergebnis-Tabelle

| Prüfbereich | Ergebnis | Evidenz | Priorität |
|---|---|---|---|
| **Aktienprognose** | Kein belegter Richtungs-Edge. Die MC-Drift kommt zu 86 % aus dem 6-Monats-Trend (`hist_mu`), nicht aus dem Signal. | Kandidaten 5 T: Aktie Ø −0,07 % (n=478). Shadow-Ledger 45 T: Ø −2,9 % (n=59, 18 Signaltage). Champion-Basiswert über die Haltedauer: Ø +0,3 % (n=6). `hist_mu`-Standardfehler 0,22 %/Tag > Schätzwert (t≈0,6). | P1 |
| **Monte-Carlo-Logik** | „Trefferquote“ ist eine Berühr-, keine Gewinnwahrscheinlichkeit. Ohne Drift liegt sie strukturell bei etwa 0,5. Kalender-DTE wird als Handelsschritte simuliert. Die P&L-MC hat eine Drift ohne Decay. Es gibt keine Kalibrierung. | Vorhersage Ø 0,64 gegen realisiert 0,36 (45 T, n=59, 18 Tage). Bei den 6 Trades ohne Drift 0,45–0,54 statt 0,61–0,75. Streuung um √(365/252)=1,20 überschätzt. Kumulierte Drift in der P&L-MC 1,7–2,9× höher als in `run_for_dte`. | **P0** (Einheit, Decay), P1 (Kalibrierung) |
| **Volatilität** | Pfade laufen mit realisierter Vola, nie mit IV. IV dient nur als Startwert für Bewertung und OU-Prozess. Die IV-Crush-Annahme wirkt im finalen Gate nicht. | IV/RV₆ₘ im Median 1,00 (n=58). 0/28 Ticker haben ≥ 30 IV-Tage, also läuft immer die OU-Heuristik mit θ = Entry-IV (Ø IV-Änderung −0,1 %). `sigma_30d or DEFAULT_SIGMA` ohne measured-Prüfung. | P1 |
| **Options-Pricing** | Black-Scholes ist für Einzel-Calls konsistent. Die MC-P&L bewertet Bull Call Spreads aber als nackten Long Call: Einstieg Long-Ask, kein Short-Bein, kein Cap. | BS-Rekonstruktion trifft die beobachteten Exit-Mids auf 2–9 pp. Gate-`roi_net` liegt über dem theoretischen Maximalertrag des Spreads: BBY 171 % > 138 %, ADM 56 % > 53 %, KDP 96 % > 82 %. Reproduziert in `test_mc_pnl_gate_spread_mean_bounded_by_max_payoff` (xfail): 118 % > 53 %. | **P0** |
| **Erwarteter Netto-Payoff** | Es wird kein Erwartungswert berechnet. Stufe 1 ist p · Gewinn ohne Verlustzweig und ohne Theta. Stufe 2 ist der Long-Call-Mittelwert unter extrapoliertem Trend. Kosten zählen nur das Long-Bein. | Risikoneutral (ohne Edge) gilt EV = −Round-Trip. Für die Spreads sind das −15 % bis −38 % des Debits, für MO −6 %. Ein korrekter Spread-MC ohne Trend-Drift liefert −13 % bis −15 % (BBY/ADM/KDP). | **P0**/P1 |
| **ROI-Gate** | Das Gate misst die angenommene Drift, keinen Edge. Der Spread-Cap fehlt. Das Netto-Delta ist 0,8 × Long-Delta statt Long − Short (3,9× zu hoch). | Gate-ROI der 6 Trades: +51 % bis +171 % (Ø +87 %). Realisiert: −74 % bis +51 % (Ø −30 %). `_compute_roi` liefert 123 % bei einem Maximum von 53 % (`test_compute_roi_spread_bounded_by_max_payoff`, xfail). | **P0** |
| **Sizing** | Kelly mit `b = roi_net/0,8` zählt p doppelt. Der Maximalverlust ist mit 80 % statt 100 % angesetzt. | Synthetisch: Kelly ≤ 0 für `roi_net` ≤ 0,5 bei p ≤ 0,55. Positiv wird es nur mit aufgeblähtem `roi_net`. | P1 |
| **Exit-Logik** | Der Stopp entscheidet ohne Uhrzeit- und Quote-Qualitätsprüfung. Der Spread-Stopp −50 % sitzt 12–38 pp über dem Sofortverlust beim Einstieg. Der Time-Exit greift nie, weil die 45-T-Grenze vorher erreicht wird. | KDP 15.10.: Stopp um 22:33 UTC auf einer Quote von 19:59:50 UTC (Combo 0,20 zu 2,70, ausführbar −87,9 %, Fair Value −12 %, Basiswert +1,1 %). Am nächsten RTH-Fill −18,2 %. BBY −73,8 % bei Basiswert +1,5 % (Fair Value nach BS ca. −15 %): Quote-Artefakt sehr wahrscheinlich, aber ohne Archiv nicht beweisbar. | **P0** (Quote), P1 |
| **Forward-Outcomes** | Die Outcome-Rechnung ist korrekt. Kein Lookahead und kein PIT-Verstoß gefunden. | AMZN, KDP und MO exakt aus dem Quote-Archiv nachgerechnet, Kommission korrekt. Nur KDP ist voll reproduzierbar (Entry und Exit archiviert, RTH). Bei BBY, ADM und SNPS ist der Exit nicht archiviert (Archiv beginnt 12.10.). | Info, P2 |
| **Baseline-Vergleich** | Der Champion ist schlechter als Aktie und kein Trade. Der Verlust stammt überwiegend aus Ausführungskosten und Exit, nicht aus der Aktienprognose. | Pro Trade Ø: Champion −30,4 % brutto bzw. −30,9 % netto, Aktie +0,3 %, kein Trade 0. Kandidaten 5 T: Option −12,6 % (n=272), Strategie −17,6 % (n=222), Aktie −0,07 % (n=478). Mid-Kontrafakt AMZN/KDP/MO: +20,5 % gegen realisiert +0,4 %. | P1 |

## 3. Champion-Trades einzeln

Angaben in % des Debits. Für die Zerlegung wurde die IV beider Beine aus den Entry-Mids zurückgerechnet (Black-Scholes, r = 4 %, ohne Dividende).
- **Basiswert:** Wertänderung bei unveränderter Restlaufzeit.
- **Zeitwert:** Laufzeiteffekt.
- **IV/Rest:** beobachteter Exit-Mid minus Modellwert.
- **Exit-Kreuzung:** Exit-Mid minus Fill.

Ohne archivierte Exit-Quote sind IV, Exit-Kreuzung und Quote-Artefakt nicht trennbar und stehen gemeinsam als „unerklärt“.

| Trade | Struktur, Haltedauer | Gate-ROI | Max. | MC-p | Basiswert | Entry-Kreuzung | Basiswert-Effekt | Zeitwert | IV/Rest | Exit-Kreuzung bzw. unerklärt | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AMZN | BCS 260/285, 45 T | +75 % | +92 % | 0,72 | −1,8 % | −18,8 | −5,2 | −3,0 | −1,7 | −2,9 | **−31,5** |
| KDP | BCS 33/36, 0–1 T | +96 % | +82 % | 0,74 | +1,1 % | −12,1 | +4,6 | 0,0 | +9,0 | −19,7 | **−18,2** |
| MO | LC 65, 22 T | +51 % | – | 0,61 | +8,2 % | −3,1 | +63,0 | −3,6 | −2,9 | −2,4 | **+51,0** |
| BBY | BCS 90/100, 27 T | +171 % | +138 % | 0,75 | +1,5 % | −19,0 | +5,0 | −1,0 | – | −58,7 (unerklärt) | **−73,8** |
| ADM | BCS 77,5/85, 9 T | +56 % | +53 % | 0,68 | −5,5 % | −14,3 | −15,3 | −0,1 | – | −29,5 (unerklärt) | **−59,2** |
| SNPS | BCS 500/550, 2 T | +74 % | +98 % | 0,69 | −1,7 % | −19,0 | −4,2 | 0,0 | – | −27,5 (unerklärt) | **−50,8** |
| **Ø** | | **+87 %** | | **0,70** | **+0,3 %** | **−14,4** | **+8,0** | **−1,3** | | **−22,7** | **−30,4** |

**Ursachen je Trade:**
- **AMZN:** Prognose leicht falsch (Ziel +8 % verfehlt). Den Verlust bestimmt aber die Einstiegskreuzung mit 19 von 32 pp.
- **KDP:** Prognose richtig, der Spread-Mid stieg um 15,5 %. Der Verlust ist reine Ausführung (12 + 20 pp). Ausgelöst wurde er durch einen Stopp auf einer nachbörslichen Quote.
- **MO:** Prognose richtig (Ziel erreicht). Die Option hat die Bewegung getragen, Kosten 5,5 pp.
- **BBY:** Prognose leicht richtig. Der Verlust kommt aus Einstiegskosten und einem nicht erklärbaren Exit-Abschlag von 59 pp. Sehr wahrscheinlich eine weite Schlussquote, unbelegbar.
- **ADM:** Prognose falsch (−5,5 % in 9 T), dazu Kosten.
- **SNPS:** Schwache Gegenbewegung. Den Verlust bestimmen Kosten und Stopp nach 2 T.

**Zusammenfassung:**
- Im Mittel trug die Aktienprognose +8 pp bei, getrieben von MO.
- Kosten und Exit kosteten etwa −37 pp.
- Das MC-Ziel erreichte nur 1 von 6 Trades. Erwartet waren 0,70 · 6 ≈ 4,2, allerdings zensiert durch frühe Stopps.
- Mit n = 6 ist keine dieser Aussagen statistisch belastbar. Sie ist aber mit den großen Stichproben konsistent: Kandidaten-Optionen 5 T −12,6 %, davon −11,9 pp sofortige Bid/Ask-Kreuzung und −0,1 pp Markt und Theta.

**Datenstand:**
- MO und AMZN: Entry nicht archiviert (`ENTRY_QUOTE_NOT_ARCHIVED`), Exit exakt.
- KDP: Entry und Exit archiviert (`REALIZED_RTH_REPRODUCIBLE`).
- BBY, ADM, SNPS: Exit nur implizit aus dem Outcome. Es gibt keine nachträglich ersetzten Daten.

## 4. Baselines

| Baseline | Ergebnis | Bemerkung |
|---|---|---|
| Champion (6 Trades) | Ø −30,4 % brutto, −30,9 % netto | `CHAMPION_TRADE`-Zelle der Scorecard |
| Aktie, gleiche Signale, gleiche Haltedauer | Ø +0,3 % | aus `simulation.current_price` → `close_price`, Kosten vernachlässigbar |
| Kein Trade | 0 | |
| Einfache vordefinierte Option (z. B. ATM-Call, fixe DTE) | **nicht fair rechenbar** | Für andere Kontrakte gibt es keine historischen Quotes (Quote-Archiv ab 12.10., Chain-Summaries ab 15.10.). Eine Rekonstruktion aus heutigen Daten wäre Lookahead. Deshalb bewusst nicht berechnet. |
| Gleicher Kontrakt, Ausführung zum Mid (AMZN, KDP, MO) | Ø +20,5 % gegen realisiert +0,4 % | Kontrafakt ohne Ausführungskosten, nicht optimiert |
| Scorecard Kandidaten, 5 T | Aktie −0,07 % (n=478); Option −12,6 % (n=272); Strategie −17,6 % (n=222); Late-Stage-Strategie −24,0 % (n=64) | `evidence_ledger.scorecard`, Bonferroni, netto |

Keine Baseline wurde an die sechs Outcomes angepasst.

## 5. Befunde

### Nachgewiesen (Code, Formel, reproduzierbar)

**P0-1: MC-P&L-Gate bewertet Bull Call Spreads als Long Call**
- Fundstellen: `mirofish_simulation.py` 508 und 536–537, `options_designer.py` 755–790.
- Einstieg ist `option["ask"]` (Long-Ask), `net_debit` und `spread_leg` werden nie gelesen.
- Das Ergebnis ersetzt `roi_net` und entscheidet das Gate.
- Belege: 3 von 5 realen Gate-ROIs liegen über dem Maximalertrag. Test `test_mc_pnl_gate_spread_mean_bounded_by_max_payoff` (strict xfail).

**P0-2: `_compute_roi` ohne Spread-Cap und mit falschem Netto-Delta** (`options_designer.py` 956–957, 974–979)
- Der Spread-Cap fehlt.
- Das Netto-Delta ist 0,8 × Long-Delta statt Long − Short.
- Beleg: Test `test_compute_roi_spread_bounded_by_max_payoff` (strict xfail).

**P0-3: Drift der P&L-MC ohne Decay**
- `_compute_mismatch_drift` (Z. 366–377) behauptet „identische Formel wie `run_for_dte`“.
- `run_for_dte` lässt die Drift mit `exp(−decay·d)` abklingen, `simulate_option_pnl` nicht.
- Folge: Bei 140 T beträgt die kumulierte Drift 17,9 % statt 10,4 %.

**P0-4: Kalenderzeit als Handelsschritte**
- `run_for_dte` simuliert `days_to_expiry` Kalendertage als Handelsschritte mit Tages-σ aus Handelstagen.
- Das gilt für den Final-MC-Shadow (120/140 Kalender-DTE) und den ttm-Modus.
- Wirkung: Die Streuung ist um 20 % überschätzt, die Trefferquote z. B. 0,648 statt 0,558 (σ = 2 %, 45 T).
- Bei `legacy_45` ist nur die Einheit des Labels „45 Tage“ unklar.

**P0-5: Stopp auf nicht ausführbarer Quote**
- Fundstellen: `feedback.check_exit_rules` (Z. 550) und der Aufruf in Z. 1456.
- Es gibt keine Prüfung von Uhrzeit, Quote-Alter oder Spread-Breite vor der Stopp-Entscheidung.
- Beleg KDP: Stopp auf einer Combo-Quote 0,20/2,70, 2,5 h nach Handelsschluss. Das Archiv belegt es.

**Behoben (§6): Shadow-Exit-Monitoring ohne Zeitstempel**
- `feedback._spread_execution_monitor`: `STALE` konnte nie feuern.
- Der Shadow-Pfad hat KDP deshalb als ausführbar (VERY_WIDE) bewertet.

**Behoben (§6): Post-mortem misst Spread-Kosten am Long-Bein**
- Fundstelle: `trade_memory.py`.
- `entry_cost_high` feuerte bei keinem einzigen Spread. Median der Spread-Kosten: 5,4 % statt real 15,1 %.

**P1, nachgewiesen:**
- Stufe-1-ROI ohne Verlustzweig und ohne Theta.
- Kosten nur über das Long-Bein (Median 9,8 % Modell gegen 16,6 % real, 38 von 41 Spreads).
- Doppeltes p im Kelly-Sizing.
- Quote-Filter lässt `ask = bid = 0` durch (`options_designer.py` 1087–1091 und 1199–1204), wodurch ein Tier unnötig verworfen wird.
- Horizonte passen nicht zusammen:

  | Größe | Horizont |
  |---|---|
  | MC-Ziel | 45 T |
  | Option | 150–200 DTE |
  | P&L-MC | DTE/2 = 77–100 T |
  | maximale Haltedauer | 45 T |
  | Time-Exit | nie aktiv |

- p kommt aus der 30-T-Quick-MC, das Kursziel aus dem 45-T-Final-MC.
- Der Final-MC lehnt seit 12.06. nie ab (stimmt an jedem Lauftag mit dem Quick-Ergebnis überein).
- `hist_mu` außerhalb ±0,5 %/Tag wird auf 0 gesetzt statt gekappt.

**P2:**
- n_paths-Labels sind falsch (Quick 3000 statt 5000, Final 10000 statt 5000). Nicht behoben: Das Quick-Label geht in den LLM-Prompt ein.
- `impact`/`surprise` = 0 wird per `or 5` zu 5 (latent, weil das Gate vorher greift).
- Vorab-ROI-Check prüft eine 21–45-T-Chain, gehandelt werden ≥ 150 T.
- Spread-Demotion rechnet weiter mit 4 Fills.
- `trade_id` enthält den Short-Strike nicht.
- Bei den alten Put-Trades ist `close_reason` invertiert.
- Bear Put Spread: Take-Profit +50 % statt 70 % des Maximalgewinns.
- Kommission wird auch bei Settlement doppelt berechnet.

### Nur Hypothese (nicht beweisbar mit vorhandenen Daten)
- **BBY, ADM, SNPS:** Exit-Abschläge (−28 bis −59 pp unerklärt) stammen aus weiten Schlussquoten. Es gibt keine archivierten Exit-Quotes, die Zeitreihe in der git-History passt aber zum Muster.
- **Volatilitätsprämie:** Ob Optionen gegenüber realisierter Vola systematisch teuer oder billig sind, ist unklar. IV/RV im Median 1,00 bei n = 58, das reicht nicht.
- **`hist_mu`:** Hat der 6-Monats-Trend Vorhersagekraft? Der t-Wert liegt bei ≈ 0,6, gegen 45 T gibt es keine Validierung.
- **VIX-Faktor:** Senkt die Hürde dort, wo die ROI-Formel ohnehin optimistisch ist.
- **Total-Loss-Zensur:** timing-v2 macht einen Exit mit Bid 0 zu UNUSABLE statt −100 %. Die Häufigkeit ist unbekannt.

## 6. Umgesetzte Korrekturen (ohne Einfluss auf Handelsentscheidungen)

### 1. Shadow-Exit-Qualität: STALE wirkt jetzt
- `option_quotes.quote_time_of()` extrahiert die Börsenzeit einer Roh-Quote. Das ist dieselbe Logik wie im Archiv, `_normalize` nutzt sie unverändert.
- `feedback._tradier_option_quote` gibt `quote_time` mit, wenn ein Zeitstempel vorhanden ist. Maßgeblich ist die jüngere Seite von `bid_date`/`ask_date`, damit ein illiquides Bein mit unverändertem Bid während RTH nicht fälschlich STALE wird. Bid/Ask und Produktionsrechnung bleiben unverändert.
- `_spread_execution_monitor` übergibt `now` und den ältesten Leg-Zeitstempel an `spread_execution.monitor`.
- Wirkung KDP: Statt `VERY_WIDE` mit −87,9 % „ausführbar“ heißt es jetzt `STALE` mit `NO_RELIABLE_QUOTE`.
- Der Produktions-Stopp ist unverändert (Test `test_production_stop_rule_unchanged`).
- Dadurch misst der Shadow-Pfad ab jetzt, wie viele Produktions-Stopps auf nicht ausführbaren Quotes beruhen. Das ist die Evidenz für Vorschlag V5.

### 2. Post-mortem: Einstiegskosten der ganzen Kombination
- `trade_memory` nutzt bei Spreads `((L_ask − S_bid) − (L_bid − S_ask)) / (L_ask − S_bid)`, auf derselben Skala wie bisher.
- Long Calls und Spreads ohne Short-Bein sind unverändert.
- Auf dem Bestand ändern sich bei 116 Fällen 5 Label-Sätze: BBY, ADM, SNPS, KDP und AMZN erhalten `entry_cost_high`. Keine Primärursache ändert sich.
- Doku: `docs/research/FAILURE_TAXONOMY.md`.

## 7. Vorschläge (nicht umgesetzt, Freigabe erforderlich)

Jeder Punkt ändert Handelsentscheidungen oder Champion-Logik.

| # | Vorschlag | Behebt | Wirkung |
|---|---|---|---|
| V1 | MC-P&L für Spreads mit beiden Beinen auf denselben Pfaden. Einstieg = Debit, Exit zum Combo-Bid oder Mid minus halbe Combo-Spanne, Kosten aus der Combo-Spanne. | P0-1 | Spreads bestehen das Gate nur noch mit echtem Edge. Auf BBY, ADM und KDP läge der Spread-Mean ohne Trend-Drift bei −13 % bis −15 %. |
| V2 | `_compute_roi`: Payoff-Cap, Netto-Delta = Long − Short, Verlustzweig (1−p) und Theta, oder die Stufe 1 für Spreads ganz durch V1 ersetzen. | P0-2, P1 | konsistentes Vorfilter-Gate |
| V3 | Gleicher Drift-Decay in `simulate_option_pnl`. `hist_mu` als Drift erst nach Validierung (sonst 0). | P0-3, P1 | Das Gate misst nicht mehr den Trend. |
| V4 | `run_for_dte`: Kalender-DTE in Handelstage umrechnen (×252/365) und festlegen, ob „45 T“ Handels- oder Kalendertage meint. | P0-4 | korrekte Streuung. Betrifft auch den Final-MC-Shadow, also Versionierung des v2-Vertrags nötig. |
| V5 | Stopp nur auf einer RTH-zulässigen Quote (`usable_rth_quote`, Quote-Qualität nicht STALE oder VERY_WIDE). Sonst Entscheidung auf den nächsten RTH-Mark verschieben. Spread-Stopp auf Fair Value bzw. bereinigt um den Sofortverlust. | P0-5, P1 | keine Schein-Stopps (KDP). Datenbasis dafür liefert ab jetzt der Shadow-Pfad. |
| V6 | Kelly mit Gewinn/Verlust aus der korrekten P&L-Verteilung, Maximalverlust 100 %. | P1 | |
| V7 | Quote-Filter: `bid ≤ 0` oder `ask ≤ 0` nie als liquide werten. | P1 | |
| V8 | Ein konsistenter Horizont für MC-Ziel, P&L-Haltedauer, maximale Haltedauer und Time-Exit. | P1 | |

**Empfohlene Reihenfolge:**
1. Zuerst V5.
2. Dann V1 und V2.
3. Dann V3 und V4, jeweils zuerst als SHADOW-Berechnung neben dem Produktionswert, damit die Forward-Daten den korrigierten Wert gegen den alten vergleichen können.

## 8. Antworten

**1. Gibt es einen methodisch belastbaren positiven Netto-Erwartungswert?**

Nein. Der Scanner berechnet keinen Erwartungswert des gehandelten Instruments:
- Stufe 1 ist p · Gewinn ohne Verlustzweig.
- Stufe 2 ist bei Spreads der Mittelwert eines Long-Calls. Zu 73–90 % treibt ihn die Fortschreibung des 6-Monats-Trends.
- Kosten werden nur über das Long-Bein gezählt.

Ohne nachgewiesenen Richtungs-Edge ist der Netto-EV eines zum Ask gekauften und zum Bid verkauften Spreads gleich minus dem Round-Trip, hier −15 % bis −38 %.

Die realisierten Zahlen passen dazu:

| Population | Ø netto | n |
|---|---|---|
| Champion-Trades | −30,9 % | 6 |
| Kandidaten-Strategien, 5 T | −17,6 % | 222 |
| Kandidaten-Optionen, 5 T | −12,6 % | 272 |
| Kandidaten-Aktie, 5 T | −0,07 % | 478 |

**2. Was genau fehlt?**
- **Bewertung des tatsächlich gehandelten Instruments:** beide Beine, Cap, auf dem echten Haltehorizont, mit Exit zu ausführbaren Preisen.
- **Validierter Edge:** Eine out-of-sample geprüfte Drift bzw. Gewinnwahrscheinlichkeit statt Trend-Extrapolation. Ohne Validierung Drift 0 plus separat belegter Signal-Effekt.
- **Kalibrierung:** Berühr- gegen Terminalwahrscheinlichkeit, Brier/Reliability gegen realisierte Ziele.
- **Kostenmodell der Kombination.**
- **Exit-Regeln** passend zu Kosten und Quote-Qualität.
- **Ein einheitlicher Horizont.**

**3. Welche Fehler sind nachgewiesen, welche nur vermutet?**
- **Nachgewiesen:** P0-1 bis P0-5, die beiden behobenen Fehler und die P1-Punkte aus §5. Alle sind mit Code-Zeilen und Reproduktion belegt; P0-1 und P0-2 zusätzlich als strict-xfail-Tests.
- **Vermutet:** Quote-Artefakte bei BBY, ADM und SNPS, Volatilitätsprämie, Vorhersagekraft von `hist_mu`, VIX-Faktor, Total-Loss-Zensur.
- **Kein Lookahead, kein PIT-Verstoß** gefunden. Die Outcome-Rechnung ist korrekt.

**4. Was muss vor weiterer Datensammlung behoben werden?**
- **Im Messpfad nichts mehr.** Die Shadow-Exit-Qualität und die Post-mortem-Kosten sind korrigiert. Outcomes, Quote-Archiv, timing-v2 und Preisbasis (#121) messen korrekt.
- **Vor weiteren Champion-Entscheidungen empfohlen, mit Freigabe:**
  - **V5:** Schein-Stopps erzeugen falsche Outcomes und falsche Lern-Labels.
  - **V1 und V2:** Solange das Gate nach einem unmöglichen ROI auswählt, messen neue Champion-Outcomes eine fehlerhafte Auswahlregel.
  - **Minimal:** V1 bis V4 als SHADOW mitschreiben, damit die Forward-Daten den Vergleich erlauben.

**5. Was erst ab ≥ 30 verlässlichen Outcomes und 20 unabhängigen Handelstagen?**
- Kalibrierung der MC-Trefferquote (Reliability, Brier) und Wahl der Schwellen.
- Ob das Signal-Alpha über `hist_mu` und eine reine Trend-Baseline hinaus etwas erklärt.
- Option gegen Aktie für dieselben Signale auf 20 und 45 T, netto.
- Wirkung der Stopp-Regel: Shadow-Verdikte EXECUTION_DRIVEN, STOP_CONFIRMED und NO_RELIABLE_QUOTE.
- Long Call gegen Spread nach Kosten.
- IV gegen RV (Volatilitätsprämie).
- `min_roi` bzw. Hurdle-Höhe.

Vorher werden keine Parameter angepasst.

## 9. Prüfung

**Neue Tests:** `tests/test_option_ev_audit.py`, 10 bestanden und 2 strict-xfail. Synthetisch und auf echten Daten (KDP und MO aus `history.json`, KDP-Quotes aus dem Archiv):
- Zeitstempel aus dem Tradier-Payload, Preise unverändert.
- Combo-Preis der Produktion unverändert.
- KDP-Nachbörsen-Quote wird STALE.
- Frische Quote wird nicht STALE, das älteste Bein entscheidet.
- yfinance ohne Zeitstempel verhält sich wie bisher.
- Produktions-Stopp unverändert.
- Spread-Kosten im Post-mortem (synthetisch und real), Long Call und Spread ohne Bein unverändert.

**strict-xfail:** P0-1 und P0-2 als Invarianten „Gate-ROI ≤ Maximalertrag“. Wird ein Fix freigegeben, meldet pytest XPASS und der Marker muss entfernt werden.

**Weitere Tests:** Alle 28 Testmodule, die `feedback`, `trade_memory` oder `option_quotes` nutzen, sind grün: 1423 bestanden, 2 xfail.

**Lint:** `ruff` sauber.

**Reproduktionen:** synthetische Skripte der drei Prüfungen, im Bericht zusammengefasst. Die Zahlen in §2 bis §4 sind aus Repo-Daten vom 17.10. nachgerechnet.
