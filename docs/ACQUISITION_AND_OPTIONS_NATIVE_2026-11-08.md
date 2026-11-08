# Erfassungs-Bandit und options-native Ziele (08.11.2026)

Umsetzung der Punkte 3 und 4 aus der Analyse vom 08.11. Punkte 1 und 2 stehen in `docs/STATISTICAL_POWER_2026-11-08.md`.

Nur Research:
- keine Orders;
- das produktive Richtungssignal (LLM-Stichprobe) ist unverändert;
- das Basis-Journal des Lernmodells ist unverändert;
- keine Schwelle wurde gelockert.

## 3. Erfassungs-Bandit (`option_learner.acquisition_*`, `candidate_ledger._research_selection`)

### Problem: teilweise Rückmeldung bei der Erfassung

Das Snapshot-Budget (`ledger.max_option_snapshots`) reicht je Lauf für **37–39 von 142–421** Kandidaten ohne Richtung, also 9–27 %. Nur erfasste Kandidaten haben je ein Options-Ergebnis. Bisher wählte eine deterministische Hash-Stichprobe sie gleichverteilt aus. Welche Ergebnisse beobachtbar werden, ist damit eine Entscheidung, die bisher nicht gelernt wurde.

### Verfahren

**Modell:**
- Thompson-Sampling auf einem bayesschen Ridge.
- Merkmale: nur `LEDGER_FEATURES`, also nichts, was erst der Snapshot liefert.
- Label: das geprüfte Netto-Label der Einzeloption, je Kandidat einmal.
- Nur Labels mit Mark-Tag *vor* dem Stichtag fließen ein.
- Posterior: N(coef, σ²(XᵀX + αI)⁻¹).

**Einschlusswahrscheinlichkeit:**

π_i = ε · B/N + (1 − ε) · P(i unter den Top-B einer Posterior-Ziehung)

- ε = 0,5; 400 Ziehungen.
- Deterministisch über das Datum, nie über Ergebnisse.
- Σ π = B, jedes π ≤ 1.
- Jedes Gewicht ist durch 1/π ≤ 2N/B beschränkt.

**Ziehung:**
- Systematische Stichprobe nach Madow in Hash-Reihenfolge.
- P(gezogen) = π exakt, Umfang exakt B, nie über Budget.

**Protokoll:** Je Pool-Zeile werden gespeichert:
- `option_sample_policy`;
- `option_sample_propensity`;
- `option_sample_model`;
- `snapshot_eligible_n`.

Damit ist die Population jeder Policy über den *ganzen* Pool erwartungstreu schätzbar, nicht nur über die erfassten Kandidaten.

**Fail safe:** In folgenden Fällen gilt die bisherige Hash-Stichprobe mit π = B/N:
- kein Modell;
- Modell älter als 7 Tage;
- Modell mit Stichtag in der Zukunft;
- Fehler beim Rechnen.

**Abschalter:** `config.yaml` → `ledger.acquisition_bandit: false`.

**Training:** täglich im Scanner (`recommend --horizon 5`). Der Lauf am Folgetag nutzt das Modell.

### Bewertung (`status_5d.json` → `acquisition`)

**Schätzer:**
- Horvitz-Thompson je Tag: V_d = (1/N_d) Σ a_i · y_i/π_i.
- Zusätzlich selbstnormiert (SNIPS).
- Block-Bootstrap über unabhängige Haltefenster (`_paired`).

**Policies:**
- alle handeln;
- nur positive Prognose des Erfassungsmodells (Walk-Forward);
- nie handeln = 0.

**Alt-Zeilen:** Gleichverteilte Alt-Zeilen gehen mit π = Budget/Pool ein. Damit ist auch der Altbestand gewichtbar.

**Fehlende Labels** gezogener Kandidaten werden als Coverage-Lücke berichtet und nie ersetzt.

### Echter Stand (08.11.)

**Modell:**
- 427 geprüfte Labels, 22 Merkmale, σ ≈ 0,23.
- Kein Koeffizient mit |t| > 2,1. Die stärksten: `ml_q90_ret_60` mit t = −2,08 und `options_skew_ratio` mit t = −1,88.
- Folge: Das Modell **exploriert überwiegend**. Es behauptet keinen Vorhersagewert.

**Letzter echter Pool** (166 Kandidaten, Budget 38):
- π zwischen 0,11 und 0,52;
- maximales Gewicht 8,7 (Schranke 2N/B = 8,7).

**Horvitz-Thompson:** noch leer. Die 230 Research-Zeilen seit 30.10. sind noch nicht fällig (5 Handelstage + 10 Tage Frist).

### Wichtig vor dem Merge: Population der laufenden Pläne

Der Bandit ändert ab dem Merge, *welche* Kandidaten ohne Richtung erfasst werden. Die eingefrorenen Forward-Pläne des Lernmodells (FWD-5d/20d-4a0f886d1979, registriert 02.11., letzte Prüfung 02.03.2027) werten alle erfassten Kandidaten aus.

**Was gleich bleibt:**
- Die Auswahl hängt nur von Merkmalen vor der Erfassung und von früheren Labels ab, nie vom Ergebnis des Kandidaten selbst.
- Die Testgröße bleibt daher für die tatsächlich erfasste Population erhalten.

**Was sich ändert:**
- die Population, also die Schätzgröße: statt einer gleichverteilten Stichprobe eine bandit-gewichtete Erfassung, angereichert um Kandidaten mit höherer Prognose;
- Folge: Die Ergebnisse der laufenden Pläne beziehen sich ab dem Merge auf die bandit-gewichtete Erfassung, nicht mehr auf eine gleichverteilte Stichprobe.

Wer die bisherige Population bis zum Planende behalten will, setzt `ledger.acquisition_bandit: false`. Die π werden dann weiter (gleichverteilt) protokolliert.

## 4. Options-native Ziele (`option_learner.decompose`, `native_*`, `options_native_report`)

### Problem

Das Lernmodell sagt die Netto-Options-Rendite als *eine* Zahl voraus. Darin sind fünf Anteile vermischt:
- Richtung;
- Konvexität;
- Zeitwert;
- IV-Änderung;
- Kosten.

Auch die Wahl der Struktur (Einzeloption oder Spread) beruht auf demselben vermischten Ziel. So bleibt unsichtbar, *woher* eine Rendite kommt und was davon vorhersagbar ist.

### Exakte Zerlegung (Mid zu Mid, Black-Scholes)

**Eingaben:**
- σ0 je Leg: implizit aus dem Entry-Mid (`bs_pricing.implied_vol`, neu).
- S1 aus `ret_{h}d_at_mark`, nur wenn dessen Datum gleich dem Options-Mark-Tag ist. Sonst `NO_UNDERLYING_AT_MARK`. Für BEARISH wird die Richtungsbereinigung des Ledgers zurückgerechnet.

**Komponenten:**

| Komponente | Formel | Bedeutung |
|---|---|---|
| delta | Δ0 (S1 − S0)/V0 | Richtung |
| gamma | [V(S1, σ0, T1) − V(S0, σ0, T1) − Δ0 (S1 − S0)]/V0 | Konvexität (realisierte gegen implizite Bewegung) |
| theta | [V(S0, σ0, T1) − V0]/V0 | Zeitwert |
| vega | [V1 − V(S1, σ0, T1)]/V0 | IV-Änderung (Rest) |
| cost | ausführbar − Mid | Spread, Kommission |

- **Exaktheit:** delta + gamma + theta + vega = Mid-Rendite, exakt. Die Abweichung wird je Label als `check` gespeichert.
- **RV − IV:** gamma + theta zusammen sind der Varianz-Anteil, also realisierte gegen eingepreiste Bewegung.
- **dvol** ist die parallele IV-Verschiebung, die V1 trifft. Spreads sind in σ nicht monoton. Deshalb gilt die betragskleinste Nullstelle eines festen Rasters, verfeinert per Bisektion.

### Lernen und Strukturwahl

**Strukturunabhängige Ziele** (bestehendes `fit()`):
- Underlying-Bewegung: nur Signal-Merkmale, ein Label je Signal;
- IV-Verschiebung.

**Strukturwahl:** Beide Prognosen werden auf jede erfasste Struktur umgerechnet. Daraus folgt ein ausführbarer Erwartungswert:
- Einstieg zum Ask (netto);
- Ausstieg Mid − halber Spread;
- Kommission 0,65 je Leg und Seite.

Gehandelt (`TRADE`) wird die Struktur mit dem höchsten Erwartungswert, nur wenn er > 0 ist; sonst `ABSTAIN`.

**Prospektiv:** Täglich im Scanner (`recommend`, beide Horizonte) landen Schatten-Entscheidungen in einem eigenen Journal `outputs/research/option_learner/options_native.jsonl`:
- append-only, einmal je Signal und Horizont;
- Trainings-Stichtag = Entscheidungstag, nur Labels davor;
- **nie** im Basis-Journal.

### Bericht (`status_{h}d.json` → `options_native`)

- Komponenten mit Block-Bootstrap;
- Varianzanteile;
- Walk-Forward-IC je Komponente: was ist vorhersagbar?
- Walk-Forward der EV-Wahl: gegen „nie handeln“ und gegen „immer Einzeloption“;
- prospektive Bewertung des eigenen Journals mit offenen und fehlenden Ergebnissen.

Der Walk-Forward läuft auf gesehenen Daten und ist **nie eine Bestätigung**. Eine bestätigende Prüfung der EV-Wahl müsste als eigene Hypothese über das globale Fehlerbudget registriert werden (Klasse Modellvariante). Das geschieht hier bewusst nicht automatisch.

### Echter Stand (08.11.)

- 495 (5 T) bzw. 130 (20 T) geprüfte Labels, **alle `NO_UNDERLYING_AT_MARK`**.
- Grund: Das Underlying zum Options-Mark-Tag gibt es erst seit #168 (07.11.). Der Ledger wurde zuletzt am 06.11. aktualisiert.
- Der nächste Scanner-Lauf füllt es für alle Zeilen mit Options-Mark nach: 699 bei 5 T, 392 bei 20 T (`candidate_ledger._fill_at_mark`, nur exakter Schluss am Mark-Tag).
- Bis dahin gibt es kein Modell und keine Schatten-Entscheidungen.

### Grenzen

- Black-Scholes europäisch, konstanter Zins 4 %, ohne Dividenden.
- S0 ist der Einstiegskurs des Signals. Die Options-Quotes können später erfasst sein. Dieser Versatz landet im Rest (vega).

## Tests

**`tests/test_acquisition_bandit.py` (7):**
- Σ π = Budget, Schranken, Determinismus;
- Thompson bevorzugt hohe Prognosen und exploriert weiter;
- systematische Stichprobe mit P(gezogen) = π (empirisch);
- Horvitz-Thompson erwartungstreu, ungewichtet verzerrt;
- Fallback, Abschalter, Fehler → Hash-Stichprobe, nie über Budget;
- prospektives Modell, nur frisch;
- Bericht mit Alt-Zeilen.

**`tests/test_options_native.py` (8):**
- implizite Volatilität und Delta;
- exakte Zerlegung trifft eine bekannte IV-Verschiebung;
- BEARISH;
- nur derselbe Mark-Tag;
- Spread je Leg;
- EV-Wahl nur bei positivem Erwartungswert;
- prospektives eigenes Journal, idempotent, nie im Basis-Journal;
- Bericht beschreibend, fällige Schatten-Entscheidungen bewertet.
