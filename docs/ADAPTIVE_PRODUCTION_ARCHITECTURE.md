# Adaptive Production – Architektur und Integrationsaudit

Stand 2026-10-03. Ziel: Research-Intelligence erhält **nach echter prospektiver Evidenz** einen
kleinen, reversiblen Produktionseinfluss – ohne Backtest-Overfitting. Champion bleibt unverändert.

## 1. Integrationsaudit (Ist-Zustand vor dieser Änderung)

### Produktion (täglich, `.github/workflows/scanner.yml` → `pipeline.py`)

```
PRODUCTION DATA (Universum, News, Kurse, Optionsketten)
 → Stufe 0 Risk Gates (VIX) → 1 Hard-Filter → 1b FinBERT → 1c Sektor-Momentum → 2 Prescreen (LLM)
 → 2b Alpha Sources + Data Validation → 3 ROI-Pre-Check → 3b Pre-MC → 4 Deep Analysis (LLM)
 → 4a/4b Bearish-/Impact-Gates → 5 Mismatch → 6 Quick MC → 7 Intraday-Delta → 8 Final MC
 → 9 RL-Scoring → 10 Options Design + ROI-Gate → rank_proposals (trade_score)
 → Score-Gate (trade_score_min) → 10b Korrelations-Check
 → [NEU 10c Production Intelligence Adapter]
 → Schatten-Trades → Sizing → history.json active_trades (TRADE PROPOSAL) → Report/Mail
```

Outcomes: `feedback.py` (täglich) schließt Trades (`closed_trades.outcome`) und bewertet
Schatten-Trades (`shadow_trades.outcome`). Jeder Kandidat steht im Candidate Ledger
(`outputs/candidate_ledger/*.jsonl`, Underlying-Outcomes).

### Research / Shadow

| Komponente | Datei / Output | Rolle |
|---|---|---|
| Research Director + Active Learning | `research_director.py` → `research_candidates.json`, `active_learning.json` | Hypothesen aus Befunden |
| Hypothesis Database / Research Lab / Discovery | `research_lab.py` → `hypothesis_db.json` | historische Prüfung (BH, Locked) |
| Alpha Discovery | `alpha_discovery.py` → `alpha_discovery.json` | Ledger-basierte Merkmalssuche |
| Challenger Registry | `challengers.yaml`, `challenger.py`, `challenger_registrar.py` | Gate-Challenger (Ledger), nur Empfehlung |
| ML-Research / Meta-Learning | `ml_research.py`, `meta_learning.py` | Querschnittsmodelle, Meta-Gewichte |
| World Model / Causal / KG | `world_model.py`, `causal_research.py`, `knowledge_graph.py` | Zustand, Treiber |
| Counterfactual / Blind Spots / Next Intelligence | `counterfactual.py`, `blind_spots.py`, `next_intelligence.py` | Fragilität, Unknown-Unknowns, Abstinenz-Bestätigung |
| Meta-Cognition + Safe Mode + Alpha Decay | `meta_cognition.py` → `safe_mode.json`, `machine_state.json` | Systemzustand |
| High-Confidence Scanner | `hc_scanner.py` → `hc_candidates.json` | Research-Alerts (keine Orders) |
| Prediction-/Trade-Memory | `prediction_memory.py`, `trade_memory.py` | Gedächtnis, Failure Analyzer |
| Alternative Data | `modules/alt_data`, `config/alt_hypotheses.yaml` | SEC-Features, Verträge |

### Realer Datenfluss und fehlende Verbindung

```
RESEARCH → HYPOTHESIS (hypothesis_db, alt contracts, challengers.yaml)
 → EXPERIMENT (research_lab, ml_research walk-forward)
 → CHALLENGER (challenger.py / Forward-Ledger)
 → FORWARD OUTCOME (ml_forward, alt_forward_ledger, candidate ledger outcomes)
 → VALIDATION (next_validation, meta_learning gate, challenger verdict)
 → ???   ← nichts liest diese Ergebnisse entscheidungswirksam
```

Einzige Berührung vor dieser Änderung: `pipeline._ml_shadow_ranks` schreibt ML-Ränge als
**Beobachtung** in den Candidate Ledger. `challenger.py` liefert nur `promote_recommended`
(menschlicher PR auf `config.yaml`). **Fehlende Verbindung:** eine verifizierte, gekappte,
auditierbare Stelle zwischen Champion-Entscheidung (nach Score-/Korrelations-Gate) und
Trade-Eintragung, plus ein Controller, der Einfluss ausschließlich aus Forward-Evidenz vergibt.

## 2. Zielarchitektur (umgesetzt)

```
OBSERVE → FIND ERROR/OPPORTUNITY (bestehende Research-Module)
 → FORM HYPOTHESIS → FREEZE SPECIFICATION (hypothesis_contract: Vertrag, spec_hash, Registry-Kette)
 → HISTORICAL RESEARCH (max. HISTORICALLY_VALIDATED)
 → PROSPECTIVE CHALLENGER (Adapter wertet Regel je Champion-Trade aus, friert sie im Decision-Ledger ein)
 → FORWARD EVIDENCE (feedback.py Outcomes → resolve_outcomes)
 → PROMOTION CONTROLLER (Gate, Multiple Testing, Looks)
 → LIMITED PRODUCTION INFLUENCE (Adapter, harte Caps, Default ABSTENTION_ONLY)
 → REAL OUTCOMES → MONITOR (Demotion-Kriterien) → PROMOTE / RETAIN / DEMOTE / ROLLBACK ↺
```

| Neu | Aufgabe |
|---|---|
| `modules/hypothesis_contract.py` | einheitlicher Vertrag, Validierung, Hash, Registry (append-only, Hash-Kette), Similarity, H0/H_alt, sicherer Signal-Ausdruck |
| `modules/promotion_controller.py` | Zustandsmaschine, Transition-Log, Forward-Evidence, Multiple Testing, Entscheidungen, Demotion/Rollback, Champion-vs-Adaptive-Evaluation, `promotion_state.json`, Benachrichtigung |
| `modules/production_intelligence_adapter.py` | **einzige** Wirkungsstelle in `pipeline.py`; Verifikation, Caps, Safe Mode, Decision-/Counterfactual-Ledger |
| `config/promotion_policy.yaml` | geschützte Policy (Hash gepinnt), `max_automatic_influence: ABSTENTION_ONLY` |
| `config/promotion_hypotheses.yaml` | 5 eingefrorene Abstinenz-Verträge (Safe Mode, Modelluneinigkeit, Blind-Spot-Sektor, ML-Drawdown, VIX>30) |
| `config/promotion_approvals.yaml` | menschliche Freigaben/Rollbacks (CODEOWNERS) |

Workflows: `scanner.yml` (Adapter in der Pipeline, Ledger-Commit über `outputs/`),
`feedback.yml` (Outcomes, dann `python -m modules.promotion_controller --notify`).

## 3. Default nach Implementierung

* Production Champion: **unverändert**.
* Research Intelligence: **aktiv** (jede Regel wird je Champion-Trade ausgewertet und protokolliert).
* Automatischer Produktionseinfluss: **höchstens ABSTENTION_ONLY**, und derzeit **NONE**, weil noch
  keine Hypothese Forward-Evidenz hat (Forward ab 2026-10-05).
* Kein Score-Boost, kein neuer Trade durch Intelligence, kein Champion-Wechsel.
