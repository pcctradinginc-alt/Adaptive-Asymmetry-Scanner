# Nüchterne Bewertung 2026-11-06: Alpha, Fehler, Datenqualität, Auswertung, Lernen

Stand `main` 6145fc3d, Daten bis 2026-11-06. Nur Research und Paper-Trading. Es wurden keine Gates, Schwellen, Budgets,
Modelle oder historischen Daten geändert. Alle Zahlen sind aus den Repository-Daten nachgerechnet (Methode in
Abschnitt 7).

## Kurzfazit

1. **Es gibt kein belegtes Alpha, weder bekannt noch unbekannt.**
   - Die 7 verifizierten Trades haben einen Profitfaktor von 0,18.
   - Im Kandidaten-Census sagt keine Stufe und kein Score des Systems die Querschnitts-Überrendite robust positiv
     voraus.
   - Die Selbstprüfungen des Systems kommen zum selben Ergebnis: `factor_report`, `ml_research` (alle 6 Modelle
     „rejected_so_far“), `alt_data_validation` (SEC und TED REJECT) und `autonomous_alpha` (0 von 120 Experimenten
     prospektiv bestätigt).
2. **Die Statistik trägt noch keine Aussage.**
   - 20-Tage-Labels gibt es für 11 Signaltage, die sich stark überlappen; das sind etwa 1–2 unabhängige Perioden.
   - Ein Rang-IC von 0,05 lässt sich mit 80 % Power erst nach etwa 35 unabhängigen 5-Tage-Perioden nachweisen, also
     nach rund 6 Monaten. Ein IC von 0,03 braucht etwa 1,4 Jahre.
3. **Bei Optionen ist die Reibung der größte steuerbare Posten.**
   - Auf 381 echten Census-Optionen (20 T) beträgt die Mid-zu-Mid-Rendite Ø +9,0 %, die ausführbare Rendite (Ask
     kaufen, Bid verkaufen) Ø −3,3 %.
   - Der Spread kostet also rund 12 Prozentpunkte je Runde. Ein Signal müsste das erst einmal übertreffen.
4. **Die aktuelle Dürre (15 Tage) hat zwei Ursachen. Ein zu scharfer Mismatch-Filter ist keine davon.**
   - 19.–27.10.: Das ROI-Gate lehnt alle Kontrakte ab, bei steigendem VIX.
   - Seit 28.10.: Das economics_first-Budget lässt nur 1–3 LLM-Analysen pro Tag zu (vorher 28–96). Bei der
     bisherigen Quote von 1,1 % Trades je Analyse sind für diese Tage 0,2 Trades zu erwarten.
   - Der Engine-Monitor meldete „10 Tage, dominanter Killer mismatch_ok“. Beides war ein Artefakt; behoben, siehe F1.
5. **Zwei Datendefekte sind behoben (F1, F2), vier weitere belegt.** Die vier offenen Punkte brauchen eine
   Owner-Entscheidung, weil sie Labeldefinition, Experimentversion oder Strategiewahl ändern (F3–F6).
   - Der folgenreichste ist F6: Der „IV-Rank“, nach dem die Struktur gewählt wird, misst realisierte statt implizite
     Vola.

## 1 Datenbasis und Aussagekraft

| Bestand | Umfang | Unabhängige Information |
|---|---|---|
| `history.json` geschlossene Trades | 117, davon 7 RELIABLE | 104 aus einem 45-Tage-Fenster (April/Mai), RELIABLE in 2 Fenstern |
| Kandidaten-Ledger (Census) | 8.995 Zeilen, 27 Signaltage (28.09.–06.11.), 464 Ticker | `ret_5d` für 22 Tage, `ret_20d` für 11 Tage (bis 16.10.) |
| Echte Census-Optionen | 885 Kontrakte; 688 mit 5-T-Label, 381 mit 20-T-Label | 11 Tage (20 T); vor dem 28.10. nur LLM-analysierte Kandidaten |
| ML-Panel (`ml_research`) | 294.631 Zeilen, 600 Ticker, 2014–2026 | einziger Bestand mit statistischer Power; nur Preis- und Makromerkmale |

Labels zählen in **Kalendertagen**: `ret_20d` umfasst etwa 14 Handelstage.

Der Ledger deckt **ein einziges Marktregime** ab, Ende September bis Anfang November 2026: Markt +2,6 bis +5,4 %
in 20 Tagen, danach VIX-Anstieg auf etwa 25. Was dort auffällt, ist ein Regimebefund. Als Alpha gilt es nicht.

## 2 Ergebnisse (unabhängig nachgerechnet)

### 2.1 Merkmale im Census

Gemessen ist die Rang-Korrelation (IC) je Signaltag gegen die Rendite minus dem Tagesmedian aller Kandidaten,
anschließend gemittelt über die Tage. „pos“ ist der Anteil der Tage mit positivem IC. Insgesamt sind es 89 Tests;
die Bonferroni-Schranke liegt bei p < 0,00056.

| Merkmal | IC 5 T (Tage) | pos | IC 20 T (11 Tage) | pos | Lesart |
|---|---|---|---|---|---|
| `ml_enet_xs20_v1` (System-Score) | −0,184 (21) | 0,29 | −0,432 | 0,00 | negativ, Vola-Exposure (ρ 0,68 mit Vola-Proxy) |
| `ml_hgb_xs20_v1` (System-Score) | +0,032 (21) | 0,52 | −0,369 | 0,00 | negativ im 20-T-Fenster |
| `ml_p_gt10_60` | −0,184 (18) | 0,22 | −0,287 (8 Tage) | 0,00 | belohnt Volatilität (ρ 0,69 mit Vola-Proxy) |
| Vola-Proxy (q90 − q10) | — | — | sektorneutral −11 pp Top/Bottom | 0,00 | Niedrig-Vola schlug Hoch-Vola |
| `log_market_cap` | −0,085 (21) | 0,33 | −0,214 | 0,00 | Small Caps vorn |
| `sentiment_score` (FinBERT) | +0,024 (21) | 0,76 | +0,023 | 0,64 | ≈ 0 |
| `rel_volume` | −0,048 (21) | 0,33 | −0,110 | 0,09 | ≈ 0 bis leicht negativ; `news_count` ähnlich (−0,04 / −0,14) |
| `options_skew_25d` (Put-IV/Call-IV) | +0,099 (18) | 0,67 | +0,307 | 1,00 | stärkster positiver Wert; nach Vola, Größe und Sektor +2,6 pp (0,91), siehe 5.5 |

Alle 20-T-Werte stammen aus einem überlappenden Fenster. Dass sie an „0 von 11 Tagen“ negativ bzw. positiv sind, ist
**eine** Beobachtung, keine elf. Das System stellt die Werte mit ehrlichen Konfidenzintervallen dar (`factor_report`:
`ml_enet_xs20_v1` −0,42, 90-%-KI [−0,83; +0,28]). Geprüft habe ich auch, ob der dort gepoolte IC
(statt je Tag) verzerrt: Er weicht nur bei `ml_exp_ret_60` merklich ab, sonst nicht.

### 2.2 Trichter: Wählt eine Stufe besser als die abgelehnten Kandidaten?

Gemessen ist die sektorneutrale Überrendite, gemittelt je Datum.

| Kohorte | 5 T | 20 T | n (20 T) |
|---|---|---|---|
| verworfen vom LLM-Prescreen (`prescreen_no`) | −0,07 % | −0,33 % | 2.448 |
| Deep Analysis erreicht | +0,07 % | −0,91 % | 464 |
| Options-Design erreicht | +0,42 % | −2,51 % | 151 |
| vorgeschlagen | −0,64 % | −3,61 % | 7 |
| verworfen `pre_mc_sigma_too_low` (zu wenig Vola) | +0,34 % | +2,26 % | 164 |
| verworfen `earnings_gate` | +0,08 % | +2,18 % | 117 |

Keine Stufe zeigt Selektionsgüte. Die tieferen Stufen liegen im 20-T-Fenster eher hinten. Das passt zu ihrer
Volatilitätsneigung (2.4), ist aber nicht signifikant.

### 2.3 LLM-Stufe

- **Richtung:** BULLISH gegen BEARISH, je Tag gepaart: 5 T +0,62 pp (t = 0,98, 19 Tage), 20 T −1,44 pp (t = −1,14).
  Das deckt sich mit `factor_report` („LLM-Richtung vs. naive Baseline“: `insufficient_data`).
- **Impact / Surprise:** IC 20 T −0,08 bzw. −0,09. **Mismatch:** +0,05 bis +0,08 (n. s.).
- **Fazit:** Im Census ist kein Informationsgehalt der LLM-Urteile nachweisbar. Ein Nachweis des Gegenteils ist das
  ebenfalls nicht, dafür ist die Power zu klein.

### 2.4 Pre-MC- und MC-Gate filtern konstruktionsbedingt nach Volatilität

- Der Pre-MC-Lauf setzt `impact = surprise = 5` fest (`pipeline.py`, Stufe 3b). Das Ziel ist
  `max(8 %, 0,5·σ·√T)` (`mirofish_simulation._compute_dynamic_target`). Die Trefferquote hängt damit fast nur von σ und
  der historischen Drift ab.
  - Bei niedrigem σ dominiert der 8-%-Boden, die Trefferquote fällt, der Titel wird verworfen
    (`pre_mc_sigma_too_low`).
  - Bei hohem σ ist die Quote nahezu konstant hoch.
- Das Gate selektiert also Hoch-Vola-Titel, und für deren Optionen zahlt man die höhere implizite Vola.
- Im Census lag genau diese Gruppe hinten: `sigma_30d` IC 20 T −0,50 unter den analysierten BULLISH-Kandidaten
  (220 Zeilen, 11 Tage).
- **Hypothese, nicht bewiesen:** Das ist eine strukturelle Neigung zu Titeln mit schwächerer Erwartung. Die Literatur
  stützt sie (Niedrig-Vola- bzw. MAX-Effekt; Varianzrisikoprämie bei Optionskäufen). Sie ist prospektiv prüfbar
  (5.2).

### 2.5 Optionen: Reibung gegen Signal

Datenbasis: echte Census-Kontrakte, Einstieg zum Ask, Ausstieg zum Bid.

| Struktur | n (20 T) | ausführbar | Mid-zu-Mid | Underlying |
|---|---|---|---|---|
| LONG_CALL | 216 | 0,0 % | +14,4 % | +0,5 % |
| LONG_PUT | 118 | −9,1 % | −0,2 % | +0,9 % |
| BULL_CALL_SPREAD | 35 | −3,3 % | +8,6 % | −1,0 % |
| alle | 381 | −3,3 % (Median −9,5 %) | +9,0 % | +0,5 % |

- Der relative Spread bei Einstieg liegt im Median bei 7,8 % (Mittel 10,5 %). Die Differenz zwischen Mid und
  ausführbar beträgt 11–12 pp, bei 5 T wie bei 20 T.
- Das positive Mid-Ergebnis der Calls bei kaum bewegtem Underlying passt zum VIX-Anstieg (Vega-Gewinn langer
  Laufzeiten). Auch das ist ein Regime, kein Signal.
- Die IC-Werte von `spread_rel` und Open Interest gegen die ausführbare Rendite sind mechanisch. Darin steckt keine
  Prognose.

### 2.6 Autonomes Lernen

**Was gut funktioniert:**
- Gelernt wird nur auf VERIFIED-Labels.
- Walk-Forward und eingefrorene Referenz.
- Sequenzielle Forward-Freigabe; die erste Prüfung des 20-T-Plans ist nach dem 12.03.2027 fällig.
- Die Experimentversion ist an Merkmale und Labels gebunden.
- RL ist eingefroren, und die PPO-Policy ist ehrlich als kollabiert gemeldet.

**Der Engpass:** Es fehlt an Daten, nicht an Methode.
- Das 20-T-Modell hat seit dem 06.11. eine erste Instanz, gelernt aus 130 Labels und einem einzigen Regime.
- Die Research-Watchlist zeigt deshalb Prognosen wie „SNDK Long Call +159,9 %“. Das ist Extrapolation (F5).
- Ein Modell lernt hier zwangsläufig das Regime, also das Momentum der Halbleiter im Oktober, kein übertragbares
  Alpha.

## 3 Fehler

### Behoben in diesem Branch

**F1 – Dürre-Diagnose (`modules/engine_monitor.py`)**
- Der Streak wurde über höchstens 10 Daily-Reports gezählt. Gemeldet waren „10 Tage“, tatsächlich waren es 13
  (Stand 05.11.).
- Der „dominante Killer“ bezog sich nur auf diese 10 Tage: `mismatch_ok 6×`. Über den ganzen Streak ist es
  `roi_ok 8×`.
- Bei einem Einbruch des Durchsatzes ist der Killer-Befund außerdem nicht aussagekräftig.
- Neu sind:
  - Zählung über bis zu 120 Reports.
  - Erwartete Trades bei der Quote der 30 Tage vor dem Streak (Trades je LLM-Analyse) und P(0) nach Poisson.
  - Ausweis „Durchsatz seit … unter 25 % der Basis (seitdem erwartet x Trades)“.
- Es ist reine Beschreibung; kein Gate wurde geändert.
- Live am 06.11.: 15 Tage, erwartet 3,7 Trades, P(0) = 2 %. Seit dem 28.10. erwartet 0,2 Trades.
- Tests: `tests/test_engine_monitor.py`, drei neue Tests.

**F2 – 1-Tages-Label aus dem Schluss des Einstiegstags (`modules/candidate_ledger.py`)**
- „Letzter Schluss ≤ Einstieg + 1 Kalendertag“ ergibt bei Freitags- und Vorfeiertags-Signalen den Schluss desselben
  Tages.
- Betroffen sind 1.776 von 8.381 `ret_1d`-Labels (21 %). Ihre Streuung beträgt 0,67 % statt 3,2 %; sie messen nur
  den Rest der Sitzung, ohne Nacht.
- Neue Labels nehmen den ersten Schluss nach dem Einstiegstag, sobald dessen Sitzung abgeschlossen ist, und speichern
  dessen Datum (`ret_1d_close_date`). MFE/MAE nutzen denselben Zeitraum.
- Historische Zeilen bleiben unverändert. `one_day_label_is_same_session()` erkennt sie, und `factor_monitor`
  schließt sie aus `gross_1d` aus.
- Horizonte ab 5 Tagen bleiben unverändert.
- Tests: `tests/test_label_horizon_integrity.py`, 10 Tests (inkl. „zu kurze Historie: nie ein späterer Ersatz“).

### Belegt, offen (Owner-Entscheidung)

**F3 – `news_novelty` wird als dict gespeichert**
- Das Options-Lernmodell liest nur Zahlen. Das Merkmal ist deshalb in 0 % der Labels befüllt.
- Das System hat das am 31.10. selbst als P1 gemeldet (IMP-4a0d2d31); umgesetzt ist es noch nicht.
- Dieselbe Ursache: Der Novelty-Term in `economics_first.priority()` liest `features.novelty_max`, das es in
  `c["features"]` nie gibt (`pipeline._note_news_novelty` schreibt nur ins Ledger). Er ist also immer 0, und die
  „vorregistrierte“ Priorität enthält stillschweigend nur Volumen, News-Zahl und Dollarvolumen.
- Jede Korrektur ergibt eine neue Experimentversion, weil sich die Merkmale ändern. Ich habe sie deshalb nicht
  umgesetzt.
- Empfehlung: Merkmal als `news_novelty_max` (Zahl) neu führen oder bewusst streichen. Den Term in der Priorität
  entweder entfernen oder anschließen; der Census-IC von `novelty_mean` ist ≈ 0 bis leicht negativ.

**F4 – Options-Mark und Underlying-Label liegen am Wochenende auseinander**
- Options-Marks nehmen den ersten Handelstag am oder nach dem Fälligkeitstag, `ret_{h}d` den letzten davor.
- Bei 233 von 699 5-T-Options-Labels (33 %) liegen dadurch 3–4 Kalendertage zwischen Options- und Aktienfenster
  derselben Zeile; bei 20 T sind es 18 von 392.
- Betroffen ist jede Zerlegung „Option gegen Underlying“, etwa Alpha-to-Options oder Delta-Attribution.
- Empfehlung: einen Underlying-Schluss zum Mark-Datum der Option ergänzen, z. B. `ret_{h}d_at_mark`. Eine
  Umdefinition von `ret_{h}d` wäre eine Protokolländerung.

**F5 – Research-Watchlist zeigt Punktprognosen bis +160 %**
- Die Prognosen stammen aus 130 Labels eines Regimes.
- Sie sind korrekt als „UNVALIDIERT“ gekennzeichnet, laden aber zur Fehllesung ein.
- Empfehlung: Ohne Freigabe nur Rang und Intervall zeigen, keine Prozentzahl.

**F6 – Der „IV-Rank“ enthält kein IV-Niveau**
- `options_designer.compute_iv_rank_components` setzt sich so zusammen:
  - 80 % `rv_score`: Perzentil der aktuellen realisierten 21-Tage-Vola in ihrer 1-Jahres-Verteilung.
  - 20 % `term_score`: Steigung der IV-Laufzeitkurve.
- Das IV-Niveau des Kontrakts selbst und das Verhältnis IV zu RV gehen nicht ein.
- Nach diesem Wert wählt die Produktion Long Call oder Spread (`effective_gate`, z. B. 52). Das gilt auch für die
  Replikation im Census (`replicated_iv_rank`).
- Folge: Ist die realisierte Vola niedrig, die implizite aber hoch (typisch vor Ereignissen), gelten Optionen als
  „billig“ und werden gekauft, obwohl die Prämie gerade dann am teuersten ist.
- In der Doku ist das nirgends als Näherung ausgewiesen.
- Eine Korrektur ändert die Strategiewahl, also Produktionslogik. Sie ist nur nach Forward-Vergleich und mit
  Owner-Freigabe zulässig. Voraussetzung ist Punkt 5.2: RV-Niveau und Kontrakt-IV je Zeile erfassen, dann IV/RV
  berechnen. Ein echter IV-Rang entsteht aus dem eigenen Quote-Archiv, sobald etwa ein Jahr Historie vorliegt.

## 4 Geprüfte und verworfene eigene Hypothesen

| Hypothese | Ergebnis |
|---|---|
| Die Budget-Priorität (relatives Volumen, News) bevorzugt bereits gelaufene Titel, die am Mismatch-Filter scheitern | **Widerlegt.** Vor dem 28.10. lag die Mismatch-Quote in jedem Prioritäts-Quartil bei 44–49 %. Titel mit großer Tagesbewegung bestehen sogar häufiger (52 % gegen 40 %), weil der Mismatch die vorzeichenbehaftete Bewegung nutzt. |
| Unterreaktion ohne LLM: Die Tagesbewegung setzt sich fort, stärker bei wenig Aufmerksamkeit | **Kein Effekt.** IC der Tagesrendite gegen die Folge-Überrendite: 5 T −0,03/−0,01, 20 T +0,01/+0,09 (niedriges/hohes relatives Volumen). |
| Der gepoolte IC in `factor_monitor` verzerrt gegenüber dem IC je Tag | **Nicht materiell.** Die Merkmalsvarianz zwischen den Tagen ist meist < 5 %; das Vorzeichen bleibt überall gleich. |

## 5 Verbesserungen nach erwartetem Erkenntnisgewinn je Aufwand

1. **Ausführung vor Signal.**
   - 12 pp Reibung je Runde sind mehr, als jedes bisher gemessene Signal liefern könnte.
   - Die Kontraktvariante `real_option_alt` (liquide Laufzeit 45–90 T, engster Spread) läuft seit 02.11.; ihre ersten
     5-T-Labels sind ab 07.11. fällig. Sie direkt gegen `real_option` auswerten.
   - Für Produktion und Lernlabel getrennt messen, was ein Limit zum Mid oder etwas darüber kosten würde (Fill-Quote).
     Das gehört zu `spread_execution`.
2. **Realisierte Volatilität je Census-Zeile erfassen.**
   - Gemeint sind RV20/RV60 zum Signalzeitpunkt, daraus IV−RV je Kontrakt.
   - Die Lücke zwischen impliziter und realisierter Vola ist der am besten belegte Prädiktor für Optionsrenditen
     (Varianzrisikoprämie).
   - Heute liegt nur das RV-Perzentil `rv_score` vor (siehe F6), und `sigma_30d` nur für 338 Zeilen.
   - Der Aufwand ist gering: Die Kurshistorie wird für die Outcomes ohnehin geladen.
   - Das ist eine neue Experimentversion des Lernmodells, also eine bewusste Entscheidung.
3. **Durchsatz ist eine Owner-Entscheidung.**
   - Bei 8 Quote-Probes und 3 Sonnet-Slots erzeugt die Produktions-Engine etwa 0,4 Trades pro Monat (1,1 % von etwa
     1,75 Analysen am Tag).
   - Damit entsteht praktisch keine Forward-Evidenz über die Produktionskette. Die eigentliche Lernmaschine ist der
     Census mit 40 Optionen am Tag.
   - Entweder das Budget anheben oder die Produktionskette bewusst als Demonstrator führen und alle Bewertung auf den
     Census stützen.
4. **Historisch vortesten, was historisch testbar ist.**
   - Nur das ML-Panel 2014–2026 hat Power. Neue Aktien-Merkmale sollten zuerst dort laufen, wie es `alt_data` für SEC
     und TED bereits tut.
   - Für Optionsmerkmale gibt es keine Historie im Repo. Sie lassen sich nur prospektiv prüfen; umso wichtiger ist
     Punkt 2.
5. **Vorregistrieren statt nachjustieren.**
   - `options_skew_25d` ist der einzige Rest-Kandidat: nach Vola, Größe und Sektor +2,6 pp Top/Bottom in 20 T,
     positiv an 10 von 11 Tagen.
   - Das Vorzeichen widerspricht der Literatur (steiler Put-Skew → schwächere Rendite) und deutet auf das Regime.
   - Als Hypothese mit Richtung, Horizont und Abbruchkriterium registrieren. Nicht handeln.
6. **Volatilitätsneigung prüfen.**
   - Shadow-Vergleich: Census-Optionen der wegen `pre_mc_sigma_too_low` verworfenen Titel gegen die durchgelassenen.
     Beide sind im Census vorhanden; die Optionsdaten für die verworfenen fehlen bisher, weil vor dem 28.10. nur
     LLM-analysierte Kandidaten Kontrakte erhielten.
   - Erst nach Forward-Evidenz über das Gate entscheiden.

## 6 Bewusst nicht geändert

- Das Budget von economics_first, das ROI-Gate, Mismatch-, MC- und VIX-Regeln, das Sizing, der Safe Mode und die
  Recovery-Regeln.
- Merkmale des Lernmodells (F3), weil sie die Experimentversion ändern, und die IV-Rank-Formel (F6), weil sie
  die Strategiewahl ändert.
- Die Labeldefinition ab 5 Tagen (F4).
- Historische Ledger-Zeilen und `history.json`.

## 7 Methode (reproduzierbar)

- **Quelle:** `outputs/candidate_ledger/*.jsonl`, je (Datum, Ticker) die erste Zeile.
- **BEARISH-Zeilen:** `ret_{h}d` ist dort richtungsbereinigt; für Querschnittsvergleiche wurde das Vorzeichen auf die
  Rohrendite zurückgedreht.
- **Überrendite:** Rendite minus Tagesmedian aller Kandidaten. Sektorneutral heißt: minus Median je (Tag, Sektor).
- **IC:** Spearman je Tag ab 30 Beobachtungen, Mittel über die Tage, t nach Newey-West (Lag 1/4/14). Wegen der
  Überlappung sind die 20-T-t-Werte nicht interpretierbar.
- **Top/Bottom-Spread:** Regression je Tag auf den Perzentil-Rang (−0,5 bis +0,5) mit Sektor-Dummies, optional
  kontrolliert für Vola-Proxy (`ml_q90_ret_60 − ml_q10_ret_60`) und `log_market_cap`.
- **Optionen:** `real_opt_ret_{h}d` (ausführbar) und `real_opt_ret_mid_{h}d` (Mid) aus dem Ledger. Spread relativ =
  (Ask − Bid) / Ask bei Einstieg.
- **Power:** Streuung des Tages-IC (5 T) im Median 0,106. Benötigte unabhängige Perioden
  ≈ ((1,96 + 0,84)·0,106 / IC)².

## 8 Audit-Matrix (Auftrag „Kritisches Systemaudit“, 2026-11-06)

| ID | Prio | Befund | Beleg | Status | Alpha-Relevanz | Änderung | Abnahmetest |
|---|---|---|---|---|---|---|---|
| A1 | P0 | Dürre-Streak auf 10 Reports gedeckelt; Killer ohne Durchsatzbezug | live 13 statt „10“ Tage, `roi_ok` 8× statt `mismatch_ok` 6× | FIXED (#163) | Fehldiagnose lenkt Forschung auf das falsche Gate | `engine_monitor` Durchsatz/Poisson | `test_engine_monitor` (3 neu) |
| A2 | P0 | `ret_1d` aus dem Schluss des Einstiegstags (Fr/Vorfeiertag) | 1.776/8.381, Streuung 0,7 % statt 3,2 % | FIXED (#163, neue Zeilen; alte gefiltert) | verfälscht 1-T-IC | `_label_close`, `one_day_label_is_same_session` | `test_label_horizon_integrity` (10) |
| A3 | P0 | Cooldown nach Wanduhr statt Laufdatum | Test seit 06.11. auf `main` rot | FIXED (#163; Cherry-Pick in #164/#165) | Reproduzierbarkeit des Buchungspfads | `book_proposals` | bestehender Test |
| A4 | P0 | Long-Exit-Marks aus dem Archiv verlangen Bid > 0; der Live-Pfad akzeptiert Bid 0 | `usable_rth_quote`/`rth_quote_history` vs. `_fill_real_option_marks` live | VERIFIED_DEFECT (latent), OPEN | strukturelle Zensierung wertloser Optionen; gemessen 0 von 1.053 fälligen Marks betroffen (einziger Abbruch CTVA-Kontraktanpassung), 5 von 3.989 Archiv-Quotes mit Bid 0; Abdeckungsgate ≥ 80 % begrenzt den Bias | Vorschlag: Bid 0 bei Ask > 0 als Exit zum Bid zulassen (wie Short-Legs, #152); Protokolländerung → Freigabe | – |
| A5 | P1 | `news_novelty` als Objekt, im Lernmodell 0 % Abdeckung | IMP-4a0d2d31 | FIXED als Variante (#164, Freigabe), Basis unverändert | Merkmal nur scheinbar in der Spezifikation | Allowlist `news_novelty_field` | `test_option_learner_novelty_variant` (19) |
| A6 | P1 | Ungeprüfte Hypothesen per Text-Jaccard gesperrt und als REJECTED gezählt | 33 Fälle, verschiedene Signale | FIXED (#165, CODEOWNERS) | falsches negatives Forschungswissen, verdrängt neue Ideen | Duplikate nach Signalwerten; `NOT_TESTED` | `test_research_lab` (+2) |
| A7 | P1 | „IV-Rank“ ohne IV-Niveau (80 % RV-Perzentil) | `compute_iv_rank_components` | VERIFIED_DEFECT, OPEN (Strategiewahl → Freigabe) | Strukturwahl ignoriert die Varianzrisikoprämie | Vorschlag: RV-Niveau und Kontrakt-IV je Zeile erfassen, IV/RV als Variante | – |
| A8 | P1 | Wochenend-Versatz Option-Mark vs. Underlying-Label | 233/699 (5 T) | OPEN (Labeldefinition) | verzerrt Zerlegung Option gegen Underlying | Vorschlag: `ret_{h}d_at_mark` | – |
| A9 | P1 | Census-Optionen vor 28.10. nur für LLM-analysierte Kandidaten | `option_sample`, Stufen | DATA_GAP | Selektionsbias in 20-T-Labels bis Mitte Okt. | seit 30.10. Hash-Zufallsstichprobe mit protokollierter Auswahlwahrscheinlichkeit (`snapshot_eligible_n`, `snapshot_budget`) – fair, kein Handlungsbedarf | – |
| A10 | P1 | Lernmodell-Prognosen außerhalb der Stichprobe | `predictions.jsonl` × Ledger | NEED_MORE_DATA | 40 reife 5-T-Prognosen eines einzigen Tages (30.10.): Rang-IC 0,51; TRADE n=7 Ø +15,6 % gegen ABSTAIN Ø −6,2 % – eine unabhängige Beobachtung, keine Evidenz | – | – |
| A11 | P1 | Budget von economics_first: 1–3 LLM-Analysen pro Tag | Daily-Reports ab 28.10. | OPEN (Owner) | Produktionskette erzeugt ≈ 0,4 Trades pro Monat → keine Forward-Evidenz | Entscheidung: Budget oder Census als Bewertungsbasis | – |
| A12 | P1 | ML-Scores im Census negativ, LLM-Urteile ohne Informationsgehalt | Abschnitt 2 | NEED_MORE_DATA / REJECTED_SO_FAR (`ml_research`) | kein Alpha belegt | – | – |
| A13 | P2 | Ablehnungsgründe nur als Freitext, keine Klassen (widerlegt / zu wenig Power / Kosten / Defekt) | 126 REJECTED-Einträge, 0 mit Klassenfeld; häufigste Gründe „t < 2“ (85), „bei Stresskosten nicht positiv“ (73), „instabil über Hälften“ (68) | OPEN | „zu wenig Power“ und „widerlegt“ sind im Gedächtnis gleich gewichtet | Vorschlag: Klassifizierung beim Schreiben (CODEOWNERS) | – |
