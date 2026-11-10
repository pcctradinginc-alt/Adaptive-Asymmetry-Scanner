# Aktionsplan 2026-11-09 – Umsetzung

Grundlage: die Systemanalyse vom 2026-11-09 (Prüfpunkte 1–9, priorisierte Aktionsliste mit 8 Punkten).
Leitprinzip: konsequent vereinfachen, damit Ergebnisse reproduzierbar sind. Es gibt keine realen Orders. Bereits
registrierte Hypothesen und historische Daten bleiben unverändert. Keine Schwelle wurde wegen beobachteter Ergebnisse
gelockert.

Alle Aussagen zu Alpha sind **nicht belegt**, solange das Champion-Protokoll (Punkt 1) nicht bestätigt hat.

---

## 1. Champion als vorregistriertes Experiment

* `config/champion_protocol.yaml` und `modules/champion_protocol.py`:
  * eine Primärkennzahl: `outcome_net` verlässlicher geschlossener Champion-Trades (Entry Ask / Exit Bid,
    abzüglich Kommission), ab `start_date` 2026-11-11;
  * einseitiger t-Test auf Tagesmitteln (ein Entry-Tag = eine Beobachtung);
  * Prüfungen bei 30/60/90 Trades, Ledger-Alpha gleich auf die Prüfungen verteilt;
  * Prüfberichte create-only eingefroren (`outputs/research/champion_protocol/look_k.json`).
* `decision_config_sha256` pinnt die Abschnitte `gates`, `filters`, `risk`, `options`, `pipeline`, `models`,
  `costs` und `learning` aus `config.yaml`, dazu `config/economics_first.yaml` und die Prompt-Versionen von Deep
  Analysis und Prescreening. Jede Änderung führt zu `PROTOCOL_BROKEN` (Test: `tests/test_champion_protocol.py`).
* Start bewusst 2026-11-11: das ist der erste Handelstag, der sicher unter der gepinnten Konfiguration läuft
  (Merge am 10.11.).
* Tote Config-Schlüssel entfernt:
  * `pipeline.confidence_gate`, `n_simulation_paths`, `simulation_days`;
  * die Legacy-Schlüssel `min_impact_threshold` und `max_mismatch`;
  * `options.target_move_pct`, `min_roi_after_spread`;
  * `learning.learning_rate`, `min_bin_count`;
  * `rl.*` außer der Provenienz `model_path`.
* README auf den tatsächlichen Ablauf gekürzt.
* Der Status steht im Wochenbericht (Zeile „Champion-Protokoll“).

## 2. QuasiML/RL/PPO raus, Einheitsgröße statt Kelly

* Stufe 9 ist nur noch die finale Auswahl.
  * Vorher sortierte QuasiML nach `final_score`. Hinter `rank_proposals` (Sortierung nach Trade-Score) entschied
    das nur noch Gleichstände; jetzt gilt bei Gleichstand die Final-MC-Reihenfolge.
  * Das RL-Veto war aus.
* Entfernt:
  * die Module `quasi_ml`, `rl_agent`, `rl_environment`, `rl_robust_shadow`, `rl_promotion` samt Tests;
  * das RL-Training und die Pearson-Gewichte in `feedback.py`;
  * der Schritt `rl_promotion` im Wochen-Workflow;
  * `stable-baselines3` und `gymnasium`.
* `ppo_robust_shadow` bleibt in `challengers.yaml` aktiv, bis sein registriertes Ende erreicht ist.
  * Ein vorzeitiges Austragen würde `n_aktiv` der Altfamilie senken und damit das Alpha der übrigen
    Alt-Challenger lockern.
  * Ausgewertet werden nur die bis 2026-11-09 erfassten Zeilen.
* `portfolio.sizing_mode: unit`: 1 Kontrakt bzw. 1 Struktur je Vorschlag.
  * Kelly erscheint nur informativ unter `kelly_info`.
  * Eine Überschreitung des Positions-Caps wird nur geflaggt.
  * Begründung: Kelly rechnet mit unkalibrierten MC-Wahrscheinlichkeiten (Live-Kalibrierung n = 0).
  * Zurück auf Kelly erst nach einer Kalibrierung auf mindestens 30 verlässlichen Trades.

## 3. MC-Gates, ROI-Vorcheck, Wortlaut

* Der ROI-Vorcheck (Stufe 3) ist gestrichen: er war ein Duplikat des echten ROI-/Edge-Gates in Stufe 10 auf
  geschätzten Prämien.
* Challenger `no_mc_gates`: Champion ohne die drei MC-Gates (Quick MC, Intraday, Final MC) gegen den Champion,
  vorregistriert. Start 2026-11-10, `real_strat_ret_45d`, min_n 30, 12 Cluster, Alpha aus dem Ledger.
* Die MC-Kennzahl heißt jetzt **Target-Hit-Score (MC, unkalibriert)**, in Mail, Tagesreport und
  Wochenbericht. Die Feldnamen (`hit_rate`) bleiben, damit historische Daten lesbar bleiben.

## 4. Gerichteter Mismatch ab Nachrichtenzeitpunkt

* `data_ingestion` speichert je Ticker das Zeitfenster der verwendeten Schlagzeilen (Finnhub, NewsAPI, yfinance;
  kein Zeitstempel bedeutet `None`, nie geschätzt).
  * Die Zuordnung geschieht nach `_evaluate_ticker`, weil dessen Code Teil der eingefrorenen UNIVERSE_V1-Definition
    ist.
* `mismatch_scorer` schreibt für alle Analysen im Schatten:
  * `mismatch_raw`;
  * `mismatch_dir_48h` (Bewegung in Thesenrichtung);
  * `mismatch_dir_news`: Anker ist der letzte NY-Schlusskurs vor der ersten Schlagzeile.
* Challenger `mismatch_directional_news` (vorregistriert, Ledger-Alpha). Die Produktion bleibt unverändert.

## 5. Ausführungskosten zuerst messen (Schatten)

* `option_learner`: Kosten nach Spread-Klasse (< 2 / 2–5 / 5–10 / ≥ 10 %), ein Limit-Szenario (k = 0,5 des
  Halbspreads) und Haltedauer gepaart (20 Tage gegen 5 Tage auf denselben Kontrakten).
* Erster deskriptiver Befund (echte Daten, 5 Tage, n = 492, 4 unabhängige Perioden; **kein Test, keine Schwelle
  geändert**):

  | Klasse / Szenario | Kosten | Ausführbare Rendite | n |
  |---|---|---|---|
  | Spread ≥ 10 % | −39 % | −32 % | 94 |
  | Spread < 5 % | −4 bis −8 % | – | – |
  | Gesamt | −14,4 % | −9,3 % | 492 |
  | Gesamt, Limit-Szenario | – | −2,8 % (KI enthält 0) | 492 |
  | Haltedauer 20 T − 5 T (Paare) | −2,4 % | +12,6 % | 130 |

* Ein Spread-Filter oder eine Limit-Logik wird erst als Challenger registriert, wenn der Befund prospektiv hält.

## 6. LLM-Zensus

* `modules/llm_census.py`: k = 2 gleichverteilte Ziehungen je Lauf aus den Überlebenden der Hard-Filter, ohne
  Gates.
* π = k/N wird protokolliert. Ergebnisse stehen nur in `census_*`-Feldern, nie in `direction`, haben also keine
  Trade-Wirkung.
* Nicht analysierte Ziehungen werden mit Grund notiert. Ohne API-Schlüssel gibt es keinen Lauf; es gibt ein
  Zeitbudget mit Reserve.
* Wöchentlicher Bericht: IC und richtungsbereinigte Rendite, nur auf analysierten Zeilen.

## 7. Research konsolidieren

* **Ein Fehlerbudget.** Neue Challenger, Promotion-Verträge und das Champion-Protokoll erhalten Alpha
  ausschließlich aus dem globalen Ledger.
  * `modules/ledger_registration.py` ist der einzige Schreiber im ML-Research-Workflow.
  * Zuteilung genau einmal, idempotent, fail closed.
  * Eine nachträglich geänderte Vorregistrierung ergibt `SPEC_CHANGED`, und der Workflow wird rot.
* **Altfamilien.** Vor dem 2026-11-09 registrierte Tests laufen exakt nach ihrer registrierten Regel weiter:
  Challenger mit 0,10/(n_aktiv · Looks), Verträge mit 0,05/(Familie · 12).
  * Sie sind außerhalb des globalen Budgets ausgewiesen.
  * Systemweite Schranke: 0,05 + Altfamilien bis zu ihrem Ende, offen dokumentiert.
* **Research-Scope** (`config/research_scope.yaml`):
  * Sieben parallele Generatoren sind eingefroren: factory_plan, alpha_explore, director, feature_discovery_v3,
    feature_director_v1, causal, next_intel. Sie erzeugen keine neuen Hypothesen; registrierte werden weiter
    ausgewertet.
  * Eingefrorener Code ist per SHA-256 gepinnt (26 Dateien).

## 8. Reproduzierbarkeit, Safe Mode, externe Daten

* **LLM-Temperatur 0 – umgesetzt per API-Version-Override (Owner-Entscheidung 2026-11-10), nicht live verifiziert.**
  * Hintergrund: `requirements.txt` (`anthropic>=0.28.0`) installiert in CI anthropic 2.x. Dieses SDK sendet sonst die
    API-Version `2026-10-21`, die `temperature`, `top_p` und `top_k` auf jedem Modell mit HTTP 400 ablehnt.
  * Prescreening und Deep Analysis senden je Aufruf bzw. Message Batch `anthropic-version: 2023-06-01` mit
    `temperature=0` (`modules/cost_telemetry.py`: `tracked_create`/`run_batch` mit `sampling=True`, Konstante
    `SAMPLING_TAG = "temperature=0@anthropic-version=2023-06-01"`).
  * Fallback: Lehnt die API den Override ab (4xx außer 429) und gelingt derselbe Aufruf ohne Override, läuft der Rest
    des Prozesses mit API-Default weiter.
    * Protokoll: `sampling_fallback` im Kosten-Ledger und `llm_audit.sampling = "api-default"` je Ergebnis.
    * Scheitert auch die Gegenprobe, bleibt das Sampling an und der Fehler wird durchgereicht.
  * Reproduzierbarkeit zusätzlich: Analyse-Cache für identische Eingaben (speichert nur unter der Prompt-Version, unter
    der eine Antwort entstand) und `llm_audit` mit Modell, Sampling, Prompt-Version und SHA-256 von Prompt und
    Rohantwort.
  * Prompt-Versionen enthalten das Sampling-Regime. `model_routing` zählt A/B-, Bearish-Vorfilter- und
    Vorlagen-Paare nur im aktuell wirksamen Regime. Folgen mit echten Daten:
    * Modell-A/B (103 Altpaare) und kompakte Vorlage starten neu;
    * der Bearish-Vorfilter (1036 Altpaare) ist inaktiv, bis er unter `temperature=0` neu validiert ist (mind. 40
      bestandene Kandidaten an 10 Tagen); bis dahin werden alle Kandidaten analysiert.
  * Das Champion-Protokoll wurde vor seinem Start (2026-11-11, noch keine Ledger-Zuteilung) neu gepinnt.
  * Kein Live-Test möglich (keine bezahlten Calls). Ob der Override angenommen wird, zeigt der erste echte
    Scanner-Lauf (Kosten-Ledger `kind=sampling_fallback` bzw. `llm_audit.sampling`).
    Das hält das ganze Repo zurück.
* **Safe-Mode-Semantik.** `system_state.safe_mode_scope = research_influence_only`, am Code belegt:
  * keine positiven Boosts oder Gewichte des Production Intelligence Adapters; validierte Abstinenz bleibt erlaubt;
  * Promotionen pausiert;
  * Meta-Learning nicht aktiv;
  * Gates, Filter und Optionsauswahl des Champions laufen unverändert weiter.
  * Berichte heißen jetzt „Research-Safe-Mode“.
* **Externe Aufnahme nur mit registriertem Nutzer.**
  * `config/external_consumers.yaml` ordnet jeder aktiven Quelle ihren Nutzer zu: ein Modul oder Protokoll, das die
    Quelle oder ihre Familie nachweislich liest.
  * Ohne Nutzer ist nur eine begründete PIT-Archiv-Pflicht zulässig, und nur ohne nachholbare Historie.
  * `tests/test_external_consumers.py` macht neue Quellen ohne Nutzer rot.
  * Auf `DEFERRED` gesetzt (alle mit nachholbarer Historie, Archiv bleibt): `clinicaltrials_sponsors`,
    `usaspending_contracts`, `sec_13f_holdings`, `noaa_storm_events`.

---

## Offene Punkte / Grenzen

* **A/B-Paare beim Modell-Routing.** Sie tragen jetzt die Prompt-Version und werden je Sampling-Regime gezählt;
  Paare verschiedener Regime werden nicht gemischt.
* **Live-Bestätigung des Overrides.** Steht aus. Der erste echte Scanner-Lauf zeigt, ob `temperature=0` mit
  `2023-06-01` angenommen wird oder der Fallback greift (Kosten-Ledger `kind=sampling_fallback`, `llm_audit.sampling`).
* **Bearish-Vorfilter.** Solange er aktiv ist, wird nur eine Stichprobe von 2 BEARISH-Kandidaten je Lauf
  analysiert. Seine laufende Verlustschätzung ist dadurch nach unten verzerrt. Das ist ein bestehendes Design; es
  wird hier nur dokumentiert.
* **Challenger-Daten.** Challenger mit Start 2026-11-10 haben noch keine reifen Outcomes (Horizont 45 Tage).
* **Champion-Protokoll.** Die Ledger-Zuteilung erfolgt beim ersten ML-Research-Lauf nach dem Merge. Bis dahin
  lautet der Status `AWAITING_ALLOCATION`.
