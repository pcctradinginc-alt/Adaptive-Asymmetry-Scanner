# Learning Engine: geschlossener Lernkreislauf (Stand 2026-10-10)

Ziel ist risikoadjustiertes Netto-Alpha nach Kosten. Gelernt wird zuerst **Abstinenz**, also welche Champion-Signale sich nicht lohnen. Richtung wird nicht gelernt. Produktionswirkung entsteht nur über registrierte Verträge, den PromotionController und den Adapter.

Status-Begriffe in diesem Dokument:
- **implementiert**: Der Code existiert.
- **technisch getestet**: Die Mechanik ist mit synthetischen Daten geprüft.
- **statistisch validiert**: Forward-Evidenz hat das Promotion-Gate bestanden.
- **wirtschaftlich nachgewiesen**: Realisierte Netto-P&L nach Kosten liegt vor.

## 1. Kreislauf

| Schritt | Modul | Status |
|---|---|---|
| Entscheidung mit eingefrorenen Features | `candidate_ledger` (jeder Kandidat nach Prescreening), Decision-Ledger des Adapters | implementiert, getestet |
| Options-Quotes dauerhaft (nur beobachtet, nie modelliert) | `option_quotes` + Hooks in `market_snapshot`, `pipeline`, `feedback` | implementiert, getestet; Archiv füllt sich ab Merge |
| Outcome mit Preistyp und Qualität | `feedback.py` (`outcome_price_type`, `expiry_intrinsic`), `trade_outcomes.jsonl` (brutto + netto nach Kommission) | implementiert, getestet |
| Einheitliche Evidenz-Sicht | `evidence_ledger` (`evidence-v1`), `labeled(as_of)` = Leakage-Schutz | implementiert, getestet |
| Validierung | `evidence_core`: Purged/Embargo-Walk-Forward, Gruppen-Bootstrap, Bonferroni | implementiert, getestet |
| Training | `meta_label`: L2-Logit-Baseline gegen HGB-Challenger, feste Hyperparameter, Base-Rate- und Impact×Surprise-Vergleich | implementiert, getestet |
| Version, Manifest, Rollback | `outputs/models/meta_label/` (Registry, sha256, Manifest mit Trainings-IDs und `new_labels_since_previous`) | implementiert, getestet |
| Shadow-Prognosen | `predictions/*.jsonl`: nur für Entscheidungen nach dem Training, je (Prognose, Version) genau einmal | implementiert, getestet |
| Produktionswirkung | Adapter-Merkmal `meta_p_take` → Vertrag **PROM-META-001** (Abstinenz bei p < 0,40) → Controller | implementiert, getestet; **nicht validiert** |
| Workflow | `feedback.yml`: `python -m modules.meta_label train` nach `feedback.py`, vor dem Controller | implementiert |

## 2. Regeln des Lernzyklus

- **Kein Lookahead.** Training nutzt nur Labels mit `label_available_at <= as_of`. `predict_one` liefert nur dann eine Prognose, wenn das Modell vor der Entscheidung trainiert wurde; sonst None, nie geraten. Fehlende Merkmale bleiben NaN; Spalten ohne Werte werden pro Fit verworfen.
- **Readiness.** Trainiert wird erst ab 200 Labels über mindestens 20 Signaltage; sonst `NEED_MORE_DATA`.
- **Neutraining.** Nur bei mindestens 25 neuen Labels seit der letzten Version und höchstens wöchentlich; sonst `UNCHANGED` mit Grund.
- **Champion gegen Challenger.**
  - Die Logit-Baseline ist standardmäßig aktiv. Ihr Signal ist zunächst nur Shadow.
  - HGB wird nur aktiv, wenn alle drei Bedingungen gelten:
    - Out-of-sample im Purged/Embargo-Walk-Forward hat es eine kleinere Brier-Zahl als die Base Rate.
    - Die einseitige Untergrenze des Netto-Abstinenzwerts ist > 0. Gemessen wird nach 20 bp Kosten, per Bootstrap über Signaltage, mit α nach Bonferroni über beide Modelle.
    - Die Brier-Zahl ist kleiner als die der Logit-Baseline.
  - Ein schlechterer Challenger wird nie aktiv.
- **Rollback** (`python -m modules.meta_label rollback --reason …`):
  - Der Zeiger springt auf die Vorversion und bleibt dort (`hold`).
  - Neue Versionen werden weiter trainiert und protokolliert, aber nicht aktiv (`TRAINED_HELD`).
  - Erst `release` aktiviert die neueste Version.
- **Integrität.**
  - Weicht der Artefakt-Hash ab, gibt es kein Modell; `meta_p_take` ist dann None und der Vertrag ist nicht auswertbar.
  - Artefakte werden nur aus dem eigenen Registry-Pfad geladen.
- **Aufbewahrung.**
  - Modell-Artefakte bleiben für die letzten 6 Versionen und die aktive Version erhalten.
  - Ältere Versionen behalten ihr Manifest (Hash, Cutoff, Evaluation). Die Liste der Trainings-IDs fällt weg, ist aber aus Ledger und `as_of` rekonstruierbar.
- **Wirkung nur per Vertrag.**
  - `meta_label` selbst ändert nichts an der Produktion.
  - PROM-META-001 gehört zur Bonferroni-Familie `abstention`. Die Familie wächst damit von 5 auf 6 Mitglieder, und α_eff wird für alle Mitglieder strenger.

## 3. Promotion-Stufen (Shadow → Abstinenz → Sizing → Rerank → Gewicht)

| Stufe | Wie erreicht | Wirkung |
|---|---|---|
| NONE (Shadow) | Registrierung | keine; Auswertung und Logging |
| ABSTENTION_ONLY | automatisch nach Forward-Gate (Policy-Obergrenze) | Champion-Trade blockieren |
| SIZE_DOWN_ONLY | Klasse `sizing`, nur menschliche Freigabe | Position × Faktor in [0,25; 1) |
| RERANK_ONLY / SCORE_LIMITED / WEIGHT_10/25 | nur menschliche Freigabe, je Look eine Stufe | wie bisher, harte Caps |

Die Demotion erfolgt automatisch nach vorab fixierten Kriterien, jeweils eine Stufe pro Schritt.

Der globale **Kill-Switch** (`config/production_kill_switch.yaml`) schaltet sofort auf den reinen Champion zurück.

## 4. Alpha-Fokus: drei Hypothesenfamilien

| Familie | Verträge | Warum im Fokus | Wirkung höchstens |
|---|---|---|---|
| `abstention` (Champion) | PROM-ABST-001..005 v1, **PROM-META-001** | Schlechte Trades vermeiden ist der risikoärmste Weg zu Netto-Alpha. Das Meta-Label bündelt die Einzelsignale in einem gelernten Modell. | ABSTENTION_ONLY automatisch |
| `abstention@FINAL_MC_SURVIVOR` | PROM-ABST-001..005 v2 | gleiche Fragen auf der größeren Final-MC-Population, also schnellere Evidenz | nur Empfehlung; Champion-Wirkung nur über einen neuen Vertrag (PR) |
| `research_only@EA_NEWS_CANDIDATE` | EA001–EA007 | Expectation Gap bei Nachrichten, eine unabhängige Alpha-Quelle | nur Research |

Ohne neue Entwicklung laufen weiter im Shadow: `universe_segment@V2_CANDIDATE` (4 Verträge) und alle Research-Module.

## 5. Datenlage am 2026-10-10 (gemessen, nichts aufgefüllt)

| Population | Datensätze | mit Outcome | Qualität |
|---|---|---|---|
| CANDIDATE (je Horizont 5/20/45 Tage) | 1434 | 144 (nur 5-Tage) | PROXY (Underlying-Schluss, 20 bp Kosten) |
| CHAMPION_TRADE | 120 | 120 | 3 RELIABLE (`spread_quote`), 40 RECONSTRUCTED, 77 UNKNOWN (Altbestand ohne Outcome-Methode) |

Folgerungen:
- Der Candidate-Ledger beginnt am 2026-09-28 und hat bisher 8 Signaltage.
- Die ersten 20-Tage-Labels sind ab dem **2026-10-19** bekannt.
- Die Readiness von 20 Signaltagen ist frühestens **Mitte November 2026** erreicht. Bis dahin meldet `meta_label train` `NEED_MORE_DATA`, und `meta_p_take` ist None.
- Damit ist PROM-META-001 vorerst nicht auswertbar und hat nie Wirkung.
- Danach braucht der Vertrag volle Forward-Evidenz: Mindest-N, Signaltage, Regime, Cluster und Looks im 28-Tage-Takt. Eine Promotion vor 2027 ist nicht zu erwarten.

Options-Quote-Archiv und `trade_outcomes.jsonl` sind leer, bis die Hooks auf `main` laufen. Realisierte Options-P&L mit beobachteten Quotes gibt es bisher für **3 Trades**.

## 6. Ehrliche Einordnung

- **Statistisch validiert:** nichts. Kein Vertrag hat das Forward-Gate bestanden.
- **Wirtschaftlich nachgewiesen:** nichts. Es gibt 3 RELIABLE Champion-Trades.
- Die synthetischen Tests (`tests/test_learning_engine.py`, `tests/test_promotion_stages.py`) belegen die Mechanik: Leakage-Schutz, Lernzyklus, Challenger-Ablehnung, Rollback und Kill-Switch. Sie sind **kein** Alpha-Nachweis.
- Bekannte Grenzen:
  - Candidate-Labels sind Underlying-Proxies, keine Options-P&L.
  - Survivorship ist nur prospektiv gelöst.
  - Das Meta-Label-Signal ändert sich mit jeder Modellversion. Der Vertrag prüft die Prozedur (SPEC-Hash), nicht ein festes Artefakt; das Decision-Ledger protokolliert dafür `meta_label.version`.
