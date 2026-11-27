# Scientific Deep Repository Audit – 27.11.2026

Geprüfter Stand: `main` **60872106** (27.11.2026, 13:56 UTC; inklusive PR #224). Grundlage: Code, Konfiguration,
eingecheckte Daten (Kandidaten-Ledger Okt./Nov. ≈ 10.700 Zeilen, `history.json`, Option-Learner-Status 5d/20d,
Tagesberichte, Kosten-Ledger), GitHub-Actions-Läufe und -Logs (Scanner 23.11./25.11., Watchdog/Source Health 24.11.),
Issues/PRs (#202, #206). Unabhängig nachgerechnet: MC-Trefferquote (analytisch + Simulation), Filterwirkung auf
Forward-Renditen, Teststärke des Champion-Protokolls, Recovery-Plan des 24.11.

**Nicht zugänglich / nicht geprüft:** SEC EDGAR und Yahoo live (Proxy-Sperre in dieser Umgebung), echte Order-Fills
(Paper-Trading), GitHub-Actions-Minutenverbrauch, feldgenaue Lineage aller 63 überwachten Quellen, Mutationstests.
Fehlende Evidenz wird unten nie als Fehlerfreiheit gewertet.

---

## A. Executive Summary

**Nachweislich funktioniert**
* Messinfrastruktur: point-in-time Kandidaten-Ledger (Einstieg zum Quote-Mid beim Signal, Rendite ab Folgetag –
  0 von 2.511 Zeilen verletzen das), archivgeprüfte Options-Labels (Ask-Einstieg, Bid-Ausstieg), korrektes Embargo im
  Walk-Forward (`option_learner.walk_forward`: Training nur mit `label_day` = Exit-Mark < Testtag), ML-Merkmale aus dem
  jeweils letzten Wochenlauf.
* Wissenschaftliche Governance: vorregistrierter Champion mit Konfigurations-Hash, globales Alpha-Ledger
  (0,014 von 0,05 zugeteilt), eingefrorene Prüfberichte, Challenger im Schatten, ehrliche MC-Beschriftung
  („Target-Hit-Score, unkalibriert, keine Gewinnwahrscheinlichkeit“).
* Kosten: LLM-Ausgaben seit 28.10. 0,03–0,10 USD/Tag.

**Funktioniert nicht bzw. ist widerlegt**
* **Kein Handelsvorteil nachweisbar, die Daten sprechen dagegen.** 8 verlässliche Paper-Trades: 1 Gewinn,
  Ø −29 %, Profit-Faktor 0,18. Options-Forschungsstichprobe (1.479 Labels, 23 Tage, 7 Perioden): zum Mid-Preis
  Ø +0,2 % (KI −4,3 … +4,8 %), nach Ask/Bid **−15,9 %** – der gesamte Verlust sind Ausführungskosten (−16,2 Pkt.).
* **Das Relevante ist nicht vorhersagbar.** Die Richtung (Delta) erklärt ≈ 100 % der Renditevarianz, ihre
  Vorhersagbarkeit ist null (IC 0,04, t = 0,5). Gut vorhersagbar sind nur Theta/Gamma (IC 0,88/0,41) – mechanisch
  und ohne Gewicht. LLM-BULLISH-Signale Okt.: −0,14 % (5d) gegenüber der Tageskohorte (n = 674, 17 Tage).
* **Das Hauptexperiment ist so nicht entscheidbar.** Champion-v3 seit 12.11.: 0 Trades in 11 Handelstagen;
  historisch 0,19 Vorschläge/Tag. Bei der ersten Prüfung (30 Trades) wäre erst ein Mittel von **+25 % je Trade**
  nachweisbar (80 % Power, α je Prüfung 0,0029), bei der letzten (90) +14 %.
* **Stille Systemfehler kosteten Handelstage.** 5 von 40 NYSE-Tagen (12,5 %) ohne verwertbaren Scan; die Ursache
  vom 24.11. war ein Zählfehler im Watchdog (DA-01), der auch die in #224 eingeführte Ausfall-Recovery wirkungslos
  gemacht hätte.

**Wissenschaftlich unzureichend validiert:** MC-Gates (Zirkelschluss, DA-05), alle vorgelagerten Filter (DA-07),
Option-Learner (unter Referenz und unter „nicht handeln“), LLM-Urteil (Zensus läuft erst seit #224), alternative
Daten (7/7 REJECT).

**Größte Risiken:** (1) ein Experiment, das nie zu einer Entscheidung kommt, wird als „noch zu wenig Daten“
gelesen statt als Designfehler; (2) stille Ausfälle; (3) ungeschützter `main` (Historie am 23.11. abgeschnitten).

**Größtes ungenutztes Potenzial:** Entscheidungen an zufälligen Stichproben mit bekanntem π bewerten (40
Options-Snapshots/Tag + LLM-Zensus), statt auf seltene Champion-Trades zu warten; Kosten als erstes Kriterium.

---

## B. Systemarchitektur und Datenfluss (tatsächlich beobachtet)

```
Universum S&P500+Nasdaq100 (518)
 └─ Stufe 0 VIX-Gate (fest 35) ─ Stufe 1 Hard-Filter (yfinance .info: Cap, Volumen, RV) ── 143–422/Tag
     ├─ Ledger: ALLE Überlebenden + Forward-Renditen 1/5/20d  ─────────────► Forschung (Filterwirkung, Learner)
     ├─ LLM-Zensus: k=5 zufällig, π protokolliert (offline, ab #224 aktiv) ─► Forschung (LLM-Wert)
     └─ 1c Sektor-Momentum (BULLISH-Annahme) ─ Economics-first: 8 Quote-Proben, ≤12 Haiku ── 3–8/Tag
         └─ Haiku ─ Sonnet (≤3) + Red Team ─ Mismatch ─ Quick-MC ─ Intraday ─ Final-MC ─ ROI-Gate ── 0–2/Tag
             └─ Paper-Trade (1 Kontrakt, Ask) ─ Feedback 2×/Tag (SL/TP/Zeit, Bid) ─ history.json ─► Champion-Protokoll
 Parallel: 40 zufällige Options-Snapshots/Tag ─ Quote-Archiv ─ Option-Learner (5d/20d, Walk-Forward, Forward-Pläne)
 Parallel: 63 überwachte Quellen, davon in Produktionsentscheidungen ≈ 6 (yfinance, Tradier, Finnhub/NewsAPI, FRED-VIX,
           FinBERT, Anthropic); alternative Daten nur Forschung (7/7 REJECT)
```

**Abweichungen zwischen den vier Ebenen**

| Ebene | Beabsichtigt / dokumentiert | Implementiert / beobachtet |
|---|---|---|
| Champion | „Vorregistriertes Experiment, Auswertung bei 30/60/90 Trades“ | 0 Trades seit Start; Prüfungen bei aktueller Rate unerreichbar (DA-04) |
| MC | „Wahrscheinlichkeit, dass die These aufgeht“ (Ursprung) | Berührwahrscheinlichkeit unter angenommener Drift; ohne Drift ≈ 50–57 % für jede Aktie ab σ ≥ 2 %/Tag (DA-05) |
| Recovery | „Watchdog startet verpasste Läufe höchstens einmal je Tag“ | Budget durch Source-Health-Ketten-Dispatch verbraucht (DA-01) |
| Data Health | „Pflichtdaten geprüft“ | geprüft wurden History/Options, nicht der vom Hard-Filter gelesene `.info`-Endpunkt (DA-03) |
| LLM-Zensus | „unverzerrte Messung des LLM-Werts“ | bis 26.11. nie gelaufen (Datumsfehler), VETO-Analysen verworfen (in #224 behoben) |
| Lernen | „autonomes Lernen“ | Produktion lernt bewusst nicht (eingefroren); Forschung lernt, ohne Mehrwert gegenüber Baselines |

**Informationsverlust (quantifiziert):** Von ≈ 250 Hard-Filter-Überlebenden/Tag sehen ≈ 2–3 % ein LLM, ≈ 0 % werden
gehandelt. Für die Forschung bleibt die Information erhalten (Ledger 100 %, Options-Snapshots ≈ 40/Tag, Zensus 5/Tag).
Kritische Systemgrenzen: GitHub-Cron-Verzug 4–7 h, UTC-Tagesgrenze vs. NYSE-Tag, Yahoo-Crumb-Fehler.

---

## C. Bewertungsmatrix (0–100, Unsicherheit in Klammern)

| Bereich | Punkte | Evidenz |
|---|---:|---|
| Grundlegende Systemlogik | 30 (mittel) | Grundhypothese (News-Unterreaktion → Long-Optionen) empirisch nicht gestützt; MC-Zirkelschluss; Champion-Design statistisch nicht entscheidbar. Pluspunkt: saubere Trennung Produktion/Forschung. |
| Datenqualität | 55 (mittel) | Quotes archiviert, PIT-Ledger; aber Provider-Ausfall wurde als Marktbefund verbucht (bis #224), 12,5 % Handelstage ohne Scan, 13 unvalidierte Quellen. |
| Datenintegrität | 65 (mittel) | Append-only Ledger, Hash-Pins, PIT-Reader (gehärtet); `main`-Historie am 23.11. abgeschnitten, kein Branch-Schutz. |
| Datenverarbeitung | 55 (mittel) | EPS-Fehler (behoben); „fehlend → 0“-Muster (`eps_drift or 0.0`, `validate_strict`, V1-Marktkapitalisierung); Wirkung gemessen klein. |
| Datenfluss | 50 (mittel) | Zensus lief nie, Offline-Ergebnisse wurden nicht gelesen (behoben); ≈ 57 von 63 Quellen ohne Produktionsverbraucher (Schätzung aus den Produktionspfaden, nicht feldgenau). |
| Informationsausnutzung | 40 (mittel) | 2–3 % der Kandidaten erreichen das LLM; vorhandene Kosten-Evidenz (Spread-Klassen) noch nicht in Entscheidungen. |
| Mathematische Korrektheit | 60 (mittel) | Outcome-Formeln korrekt (Ask/Bid, Kommission); MC: σ je Handelstag auf Kalendertage, Berührziel ≠ Options-Gewinn; z-Score ohne √2 (dokumentiert, Schatten). |
| Statistische Validität | 65 (mittel) | Methoden gut (Tagescluster, Embargo, Bonferroni-Ledger, eingefrorene Looks); Ergebnisse durchweg nicht signifikant, Champion unterpowert. |
| Schutz vor Datenleckagen | 75 (mittel) | Embargo, Forward-Renditen, ML-Merkmale geprüft und korrekt; alternative Daten und ML-Training nicht vollständig auditiert. |
| Entscheidungsqualität | 15 (hoch) | 1/8 Gewinne, Ø −29 %; Vorschläge −12,5 % ggü. Kohorte (n = 3); kein positiver Erwartungswert belegt. |
| Prognosequalität | 15 (mittel) | Richtungs-IC 0,04; LLM-BULLISH −0,14 %; Learner −3,9 %/Tag vs. Referenz −1,4 % vs. nicht handeln 0 %. |
| Lernfähigkeit | 30 (mittel) | Geschlossener Kreislauf in der Forschung vorhanden, aber kein Mehrwert ggü. einfacher Baseline nachgewiesen. |
| Robustheit | 45 (mittel) | Drei stille Fehler mit Tagesverlust (Outage, Zensus, Recovery) erst durch Audits gefunden. |
| Reproduzierbarkeit | 60 (niedrig) | Seeds, Hashes, eingefrorene Berichte, Temperatur-Override mit Fallback; LLM nicht deterministisch garantiert, Historie gekürzt. |
| Operative Stabilität | 55 (mittel) | CI grün; 5/40 Handelstage verloren; Cron-Verzug strukturell. |
| Monitoring | 50 (mittel) | Umfangreich, aber Data Health meldete am 25.11. „proceed“, während der Tag verloren ging; prüfte den falschen Endpunkt. |
| Testqualität | 60 (mittel) | 4.444 Tests; keiner erkannte EPS-Summe, Zensus-Datum, Recovery-Zählung, Outage-Fehlklassifikation – Tests prüfen lokale Absicht, selten Systeminvarianten. |
| Kosteneffizienz | nicht bewertbar | LLM billig; Actions-Minuten nicht einsehbar; hoher Code-/Workflow-Umfang ohne nachgewiesenen Nutzen. |

---

## D. Fehlerregister

Legende Status: **N** nachgewiesen · **V** begründeter Verdacht · **H** unvalidierte Hypothese · **X** nicht prüfbar.

| ID | Prio | Status | Befund (Ort) | Nachweis | Auswirkung | Lösung / Stand |
|---|---|---|---|---|---|---|
| DA-01 | P0 | N | Recovery-Budget zählt jeden `workflow_dispatch` des UTC-Tags, auch den Scanner-Direkt-Dispatch aus Source Health (`workflow_health.plan_recovery`, `source_health.yml`) | 24.11.: Dispatch 00:18 UTC (verspäteter Vortagslauf) → `RECOVERY_EXHAUSTED`; Nachlauf mit Originaldaten | Handelstag verloren; #224-Outage-Recovery hätte nie gegriffen | **Umgesetzt:** Ketten-Dispatch mit Lauftitel `(nach Source Health)` markiert und nicht gezählt; Test + Replay |
| DA-02 | P1 | N | Watchdog: Source Health „versorgt“ nach Datum; Scanner-Guard verlangt ≤ 12 h | Snapshot 00:17 UTC, Watchdog 17:51 UTC: „OK“, Guard: `UPSTREAM_NOT_READY` | kein Source-Health-Recovery, Scanner startet nie | **Umgesetzt:** `artifact.max_age_hours: 12` = Guard-Frische |
| DA-03 | P1 | N | Data Health prüft nicht den Endpunkt des Hard-Filters (`config/source_health.yaml`) | Probes nur `history`/`options`; 25.11. `.info` 401 bei Data Health „proceed“ | Gesundheitsanzeige misst Erreichbarkeit, nicht Verwendbarkeit | **Umgesetzt:** Probe `yfinance_info` (marketCap, averageVolume > 0), IMPORTANT, nicht blockierend |
| DA-04 | P0 | N | Champion-Protokoll bei beobachteter Rate nicht entscheidbar (`config/champion_protocol.yaml`) | 0 Trades/11 Tage; MDE +25 %/+17 %/+14 % bei 30/60/90 (SD 0,38 aus Vor-Start-Outcomes) | „NEED_MORE_DATA“ auf Dauer; Produktionslogik wird nie validiert | **Umgesetzt (Messung):** `feasibility` im Status + Wochenbericht. **Owner:** Neuentwurf (siehe F) |
| DA-05 | P1 | N | MC-„Trefferquote“ = Berührwahrscheinlichkeit unter angenommener Drift (`mirofish_simulation.run_for_dte`) | analytisch 2(1−Φ(0,5)) = 61,7 %; Simulation: ohne Drift 50–57 % ab σ ≥ 2 %/Tag, α = 0,1 %/Tag hebt um 7–30 Pkt. | Gate ≈ Volatilitätsfilter + LLM-Annahme; Scheingenauigkeit | Beschriftung bereits ehrlich; Challenger `no_mc_gates` misst den Gate-Wert. Keine Änderung am eingefrorenen Champion |
| DA-06 | P1 | N | Options-Ökonomie: Verlust = Ausführungskosten; Richtung unvorhersagbar | Zerlegung 1.479 Labels: Kosten −16,2 (KI −18,2…−14,1), Mid +0,2; Varianzanteil Delta 1,08, IC 0,04 | kein positiver Erwartungswert ohne Kostensenkung *und* Richtungsvorteil | Messung vorhanden; Limit-Szenario −8,7 % statt −15,9 % (nur modelliert) |
| DA-07 | P1 | V | Frühe Filter verwerfen überdurchschnittliche Kandidaten, späte die schwächeren | 20d-Überrendite ggü. Tageskohorte: Pre-MC-σ +5,8 %, MC-Schwelle +2,8 %, Haiku-Nein +0,4 %; Red-Team-Veto −4,5 %, ROI −3,3 %, Earnings −7,5 %; Vorschläge −12,5 % (n = 3) | mögliche systematische Falsch-Negative | Explorativ (eine Marktphase, überlappende Fenster); Bestätigung nur prospektiv über registrierte Challenger |
| DA-08 | P1 | N | 12,5 % der NYSE-Tage ohne verwertbaren Scan | 05.10., 12.10., 13.10., 24.11. ohne Bericht; 25.11. Provider-Ausfall | Lücken im Champion und in den Forschungsdaten | DA-01/02 + #224; Abnahme: Anteil versorgter Tage ≥ 97 % über 20 Tage |
| DA-09 | P2 | N | Champion-Hash pinnt nur Konfiguration, nicht Entscheidungscode (Exit-Regeln, MC, VIX 35 hart codiert) | `DECISION_SECTIONS` in `champion_protocol.py`; z. B. `feedback.check_exit_rules` | „eingefrorener“ Champion kann sich per Code ändern | Empfehlung: Code-Drift-Diagnose (nicht brechend). Nicht umgesetzt: hohes Fehlalarmrisiko bei jedem PR |
| DA-10 | P2 | N | `main` ungeschützt; Historie am 23.11. durch Bot-Wurzelcommit d26b9cdc abgeschnitten | `GET /branches/main` protected=false (#202); `git log` | Nachvollziehbarkeit, Manipulationsrisiko | **Owner** (Rulesets) |
| DA-11 | P3 | N | `.DS_Store` committet | `git ls-files` | Hygiene | **Umgesetzt** (entfernt, `.gitignore`) |
| DA-12 | P2 | N | „fehlend → 0“: `mismatch_scorer` `eps_drift or 0.0`, `validate_strict` füllt 0.0, V1 zählt fehlenden Börsenwert als „zu klein“ | Code; Daten: eps_drift 404/427 = 0, Sentiment 0 Fehlerfälle | Verzerrung bei Provider-Fehlern | V1 eingefroren; Ausfall wird seit #224 erkannt. Weitere Änderung würde Champion/Experimente ändern |
| DA-13 | P2 | N | Alternative Daten ohne Mehrwert | `alt_data_validation`/`source_scoreboard`: 7/7 REJECT; ≈ 57 von 63 Quellen ohne Produktionsverbraucher (geschätzt) | Komplexität, Laufzeit, Wartung | Empfehlung: Ingestion ohne registrierten Verbraucher stilllegen (Owner) |
| DA-14 | P2 | N | Learner überwiegend auf Research-Census trainiert (91/92 Walk-Forward-Entscheidungen) | `feature_diagnostics.json` | nicht auf Produktion übertragbar | dokumentiert (#206 übernommen); Kohorten-Challenger separat |
| DA-15 | P2 | H | Stop-Loss bei 2×-täglicher Prüfung ist kein wirksamer Verlustbegrenzer | realisiert Ø −80 % bei Schwelle −45/−50 % (config-Kommentar, n = 30); exit_sim n = 10 | Verluste über Schwelle, ggf. Realisierung temporärer Drawdowns | zu wenig Daten; als Challenger registrierbar |
| DA-16 | X | X | SEC-EPS-Korrektheit live, echte Fills, Actions-Kosten | Netz/Zugang | – | in CI bzw. durch Owner prüfbar |

**Korrektur des eigenen Berichts vom 26.11.:** Dort stand, ein Datenausfall-Tag werde „höchstens einmal je Tag“
nachgestartet. Wegen DA-01 hätte das nicht funktioniert; erst mit dieser Änderung ist die Aussage richtig.

---

## E. Priorisierung

* **P0:** DA-01 (behoben), DA-04 (gemessen; Neuentwurf nötig).
* **P1:** DA-02, DA-03 (behoben), DA-05, DA-06, DA-07, DA-08.
* **P2:** DA-09, DA-10, DA-12, DA-13, DA-14.
* **P3:** DA-11 (behoben), DA-15.

---

## F. Maßnahmenplan (nach Abhängigkeiten)

1. **Messbarkeit sichern (erledigt in diesem PR):** DA-01/02/03/11, Machbarkeitsdiagnose.
   *Abnahme:* in den nächsten 20 Handelstagen ≥ 97 % versorgte Tage; bei Provider-Ausfall erscheint `data_failure` und
   ein Recovery-Lauf; `champion_protocol/status.json` enthält `feasibility`.
2. **Branch-Schutz (Owner, #202)** – Voraussetzung, damit jede weitere Evidenz manipulationssicher ist.
   *Abnahme:* Ruleset aktiv, Datenschreiber pfadbeschränkt, ein PR-Merge und ein Archiv-Push erfolgreich.
3. **Champion entscheidbar machen (Owner-Entscheidung, neue Protokollversion):** Die Produktionsentscheidung als
   *Regel* auf einer randomisierten Stichprobe bewerten statt auf seltenen Live-Trades: LLM-Zensus aus der
   Options-Snapshot-Stichprobe ziehen, sodass jede Zensuszeile ein archivgeprüftes Options-Label hat (5/Tag →
   ≈ 100 Beobachtungen in 20 Tagen), Champion-Gates darauf kontrafaktisch anwenden.
   *Abnahme:* vorregistrierter Test mit Power ≥ 80 % für einen ökonomisch relevanten Effekt (z. B. +5 % je Trade)
   innerhalb von 6 Monaten.
4. **Kosten zuerst:** Einstiegsspanne als vorregistrierter Challenger (Spread-Klassen 0–2 % −8 %, ≥ 10 % −34 %,
   deskriptiv). *Abnahme:* gepaarter Forward-Vergleich, Alpha aus dem Ledger.
5. **MC als Gate prüfen:** Ergebnis des Challengers `no_mc_gates` abwarten; ohne Mehrwert in der nächsten
   Champion-Version entfernen (kein Lockern nach Augenschein, nur nach vorregistriertem Test).
6. **Vereinfachen:** Datenquellen ohne registrierten Verbraucher stilllegen (DA-13), PPO-Artefakte archivieren.
   *Abnahme:* gleiche Entscheidungen, weniger Workflows/Laufzeit.
7. **Langfristig überwachen:** Anteil versorgter Handelstage, Machbarkeitsurteil, Zensus-IC (20/45 Tage), Kosten je
   Entscheidung.

---

## G. Ungenutztes Potenzial

| Potenzial | Art | Evidenz |
|---|---|---|
| ≈ 10.700 Ledger-Zeilen mit Forward-Renditen je Filterstufe | empirisch vorhanden | Filterwirkung (DA-07) bisher nur explorativ (`alpha_discovery`, monatlich) |
| 1.479 zerlegte Options-Labels (5d) | empirisch | Kosten- und Strukturwahl lernbar; Richtung nicht |
| LLM-Zensus (seit #224 aktiv) | theoretisch → bald empirisch | erste Kohorte ab nächstem Handelstag; IC frühestens nach ~4–6 Wochen |
| Red-Team-Veto und Earnings-Gate als Negativfilter | empirisch (explorativ) | −4,5 % bzw. −7,5 % (20d) der Verworfenen |
| Kontrollgruppe „nicht handeln“ | vorhanden | in Learner-Berichten bereits Benchmark; für den Champion fehlt sie (DA-04) |
| Limit-Ausführung | nur modelliert | −8,7 % statt −15,9 %, ohne echte Fills nicht belegbar |

---

## 17. Adversariale Gegenprüfung (Kurzfassung)

* *Wie kann das System hervorragend aussehen und trotzdem falsch sein?* 4.444 grüne Tests, Data Health „proceed“, grüne
  Workflows – und trotzdem ging der Handelstag 25.11. verloren; die Tests prüften lokale Absicht, nicht die
  Invariante „jeder NYSE-Tag wird versorgt oder sichtbar als Ausfall markiert“.
* *Einzelne Annahme, die alles entwertet:* dass Long-Optionen auf LLM-bewertete News nach Spread einen positiven
  Erwartungswert haben. Alle verfügbaren Daten (Paper, Forschung, LLM-Kohorte) sprechen dagegen; widerlegt ist sie
  wegen kleiner, abhängiger Stichproben aber nicht.
* *Gegenprüfung meiner Vorschläge:* Die Filterbefunde (DA-07) stammen aus einer Risk-off-Phase; der Pre-MC-σ-Filter
  könnte in einer Risk-on-Phase gerade nützen – deshalb nur prospektive Challenger, keine Gate-Änderung. Die
  Machbarkeitsdiagnose nutzt die SD von 5 Vor-Start-Trades; die MDE kann um ±30 % daneben liegen, die Größenordnung
  (zweistellige Prozent je Trade) bleibt.

---

## 20. Zielarchitektur – was ich heute anders bauen würde

* **Beibehalten:** PIT-Ledger aller Kandidaten, Quote-Archiv, Ask/Bid-Outcomes, Vorregistrierung + Alpha-Ledger,
  Ausfall-Erkennung, eingefrorene Prüfberichte.
* **Entfernen/stilllegen:** Datenquellen ohne registrierten Verbraucher; MC als Entscheidungs-Gate (nur noch als
  Bewertungswerkzeug für Optionspreise); PPO-Reste.
* **Vereinfachen:** Trichter = liquides Universum → Kostenfilter (Einstiegsspanne) → randomisierte + priorisierte
  LLM-Stichprobe mit bekanntem π → Strukturwahl nach erwartetem Netto-P&L → Paper-Trade mit Limit-Fill-Erfassung.
* **Korrigieren:** Zielvariable überall = ausführbarer Options-P&L zum Entscheidungszeitpunkt (nicht
  Underlying-Berührung); fehlend bleibt fehlend; Systeminvarianten (Tagesversorgung) als Tests.
* **Fundamental anders:** Validierung an randomisierten Stichproben statt an seltenen Champion-Trades; Power-Analyse
  vor jeder Registrierung (die Infrastruktur dafür existiert bereits im Forschungsteil); eine einzige Kennzahl für
  Entscheidungsqualität (Netto-P&L je Entscheidung gegenüber „nicht handeln“).

Ziel ist nicht mehr Code, sondern ein System, das in Monaten statt Jahren klären kann, ob es einen Vorteil hat –
und das bis dahin ehrlich „nein, nicht belegt“ sagt.
