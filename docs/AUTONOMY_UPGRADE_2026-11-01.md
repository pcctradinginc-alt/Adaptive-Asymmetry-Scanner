# Autonomie-Upgrade 2026-11-01: Hypothesen, Options-Lernen und Alpha-Discovery

Research und Shadow-Betrieb ausschließlich. Keine Orders, keine Strategiefreigabe, keine Änderung an
Champion, Handelslogik, Risk-/ROI-Gates, Position Sizing, Budgets, Prüfplänen oder geschützten Protokollen
(`config/research_protocol.yaml`, `config/factory_protocol.yaml` unverändert, Hashes gepinnt). Kein LLM,
keine neuen Datenquellen, keine zusätzlichen API-Aufrufe.

## 1. Bestandsaufnahme (A1, B1, C1, D)

| Kette | Vorhanden und ausgeführt | Befund |
|---|---|---|
| Hypothesis Factory (monatlich) | Cross-Domain, Divergenz (5 feste), Drift, offene Fragen, Quellen-Bedingungen; Memory-Ähnlichkeit, Familien-Deckel, Budget 6 | Ideen nur aus Vorlagen oder vorberechneten Zellen; **keine gemessenen Anomalien**; Dossier unvollständig (keine Forschungsfrage, keine Kosten) |
| Research Lab | Leakage, Duplikat (Rang-Korrelation), Walk-Forward, BH über alle, Locked (kontaminiert, nur ablehnend) | **Defekt:** Text-Ähnlichkeitssperre auf Titel/Aussage verwarf am 2026-10-03 vier von sechs Fabrik-Tests ungeprüft (`blocked_similar_to_rejected`, Jaccard 0,90–0,92 nur wegen Vorlagentext) |
| Explorer (genetisch) | blend / interaction / regime, BH, nur prospektiv bestätigbar | unverändert |
| Lab-Discovery | Divergenzen, Trend-/VIX-Vorzeichen über Aktienmerkmale | unverändert; Anomalie-Scan schließt diesen Suchraum aus |
| Causal Research / Knowledge Graph / World Model | vorab erklärte Paare, Kanten mit Evidenz, PIT-Zustände | World-Model-Zustände jetzt für die Regime-Fehleranalyse des Options-Lernmodells genutzt |
| Fabrik → Optionsforschung | Challenger → Forward-Ledger (nur monatlich) | **Lücke:** Prospective Challenger nicht mit Alpha→Options verbunden; Forward-Ledger nur einmal im Monat |
| Options-Lernmodell | Labels nur VERIFIED, Embargo, Journal, Instanzen, Referenz, Prüfplan | 0 Instanzen der aktuellen Version: **kein Defekt** – Version entstand am Samstag 2026-10-31, erster Scanner-Lauf 2026-11-02 (isolierte Simulation: Instanz `519261c95a8a6da7` wird registriert, 359 Labels bis 2026-10-28, Referenz eingefroren, Prüfplan angelegt). 20 T: 0 geprüfte Labels, weil der erste 20-T-Mark eines archivierten Einstiegs am 2026-11-03 fällig wird |
| Workflows | Stufen-Bilanz mit CRITICAL/DEGRADED | Options-Lernmodell, Alpha→Options-Link nur DEGRADED (ml_research) bzw. gar nicht bilanziert (Scanner): Ausfall erschien als grüner Lauf |

## 2. Änderungen

### A – Hypothesengenerierung (`modules/hypothesis_factory.py`)
* `scan_anomalies`: fester, vorab erklärter Suchraum (212 Ausdrücke der Lab-Sprache) aus fünf Beziehungstypen –
  Divergenz Unternehmensinformation vs. Preis, verzögerte Reaktion eines Sektors auf Wirtschaftsindikatoren,
  Widerspruch Fundamentaldaten/Nachrichten/Kurs, unterschiedliche Reaktion vergleichbarer Unternehmen (Sektor),
  veränderte Faktorbeziehung je Regime. Gemessen **nur** im Discovery-Fenster des geschützten Protokolls,
  BH (q 0,10) über alle gescannten Ausdrücke inkl. nicht auswertbarer (p = 1), höchstens 4 Ideen; danach
  Memory-Sperre, Familien-Deckel und unverändertes Budget. Bekannte Signale (getestet, Fabrik, Lab-Discovery)
  werden über den AST-Schlüssel ausgeschlossen. Aktie vs. Optionsbewertung: `DATA_GAP` (keine PIT-Historie).
* `dossier`: für jede Idee Forschungsfrage, Mechanismus, Daten und Verfügbarkeit, Population, Richtung, Lag,
  Horizont, Vergleichsmodell, Metrik, Falsifikation, Datenqualität/-lücken, Aufwand/Kosten, Neuheit.
* Priorität (nur Reihenfolge): Protokoll-Formel × Plausibilität (Mechanismus offen 0,75) × Testbarkeit
  (Abdeckung) × unabhängige Evidenz (prospektiv widerlegte ähnliche Ideen 0,5); Komponenten sichtbar.
* `forward`-Befehl: wöchentliches Forward-Ledger der Prospective Challenger.

### Research Lab (`modules/research_lab.py`) – Defektkorrektur
Maschinelle Fabrik-Spezifikationen (`spec_hash` + `family`) sind von der Text-Ähnlichkeitssperre ausgenommen;
für sie gelten die Spezifikations-Ähnlichkeit der Fabrik und die statistische Duplikat-Prüfung. Historische
Einträge bleiben unverändert (die vier 2026-10-03 gesperrten Hypothesen bleiben `REJECTED` laut Historie und
werden nicht automatisch neu getestet – Entscheidung für einen Menschen).

### B – Options-Lernen (`modules/option_learner.py`, `modules/research_director.py`)
* `label_pipeline`: geprüft, unabhängige Signaltage, noch nicht fällig, technisch fehlend (mit Grund),
  historische Lücke vor Archivstart. Fehlende Labels werden nie als Rendite gezählt.
* `learning_status` aus bestehenden Verdikten: `TECHNICAL_DEFECT` > `NO_DATA` > `NOT_VALIDATED` >
  `FORWARD_NEGATIVE` > `FORWARD_POSITIVE` > `INCONCLUSIVE` > `NEED_MORE_DATA`.
* Bewertung je Auswahl: ADVERSE-Sensitivität (Einstieg um 25 % der Spanne ungünstiger; Exit-Spread nicht im
  Lern-Datensatz – benannt), Segmente (Optionsform, Stichprobe, Filter, Sektor, Regime aus dem World Model,
  PIT). Schwache Segmente werden zu Lücken und zu Vorschlägen `segment_review` – immer mit Freigabe, nie
  automatisch (neue Merkmals-/Auswahldefinition braucht eine eigene versionierte Spezifikation).
* Leistungszeugnis: zwei neue Zeilen (Labels mit Ursachen, Lernstatus inkl. netto/ADVERSE/Segmente).
* Label- und Kostendefinition, Experimentversion, Prüfpläne und Fehlerbudget unverändert.

### C – Discovery (`hypothesis_factory.battery`, `option_link`)
* Ablation zusätzlich nach Sektor für Signale ohne Sektor-Bezug (nur strenger) und `explained_by` je bekannter
  Faktorgruppe (Momentum, Volatilität, Größe/Liquidität, Sektor; Regime: Lab-Regime-Verdikte).
* Ergebnisse tragen `evidence_class = EXPLORATORY_HISTORICAL`; Locked-Holdout als kontaminiert benannt;
  unabhängige Bestätigung nur prospektiv.
* `factory_challenger` als Quelle des Alpha→Options-Links: nur frische Forward-Ledger-Zeilen nach der
  Registrierung und ab `forward_start`, keine Nachträge, deterministische Hash-Auswahl aus dem Top-Dezil
  (keine Auswahl nach späterer Rendite), unverändertes Regelwerk (`rule_version`), keine Orders.

### D – Workflows
* `ml_research.yml`: wöchentlicher Schritt `hypothesis_factory forward` vor dem Alpha→Options-Link.
  Stufen-Bilanz: `factory_forward`, `alpha_option_link`, `option_learner` jetzt CRITICAL.
* `scanner.yml`: Stufen-Bilanz zusätzlich `option_learner` (Prognosen werden dort eingefroren) als CRITICAL.
* Keine neuen Workflows, keine Zeitplan-Änderung, Recovery/Safe-Mode unverändert.

## 3. Grenzen

* Ein positiver Alpha-Nachweis existiert nicht. Synthetische Tests (`tests/test_autonomy_upgrade.py`) belegen
  nur technische Funktionsfähigkeit.
* Anomalien werden auf 2015–2018 gemessen und auf späteren, bereits vielfach genutzten Jahren geprüft: das ist
  explorativ. Bestätigung nur prospektiv.
* Optionsbewertung vs. Aktienbewegung bleibt `DATA_GAP`, bis eine PIT-Historie impliziter Volatilität existiert.
* Regime im Options-Lernmodell: wöchentliche World-Model-Zustände (grob); Segmente sind beschreibend.
