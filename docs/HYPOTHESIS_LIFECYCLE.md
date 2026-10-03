# Hypothesen-Lebenszyklus

## Vertrag (`modules/hypothesis_contract.py`, `config/promotion_hypotheses.yaml`)

Pflichtfelder: `hypothesis_id, version, created_at, registered_at, created_by, source_type,
research_question, economic_rationale, signal_definition, direction, universe, sector_scope,
regime_scope, features, thresholds, outcome_horizon, primary_metric, secondary_metrics, baseline,
minimum_sample_size, minimum_independent_dates, minimum_calendar_span, promotion_criteria,
demotion_criteria, maximum_initial_influence, production_class, forward_start` plus die
falsifizierbare Aussage `population, exposure, failure_condition`. `spec_hash` = SHA-256 der
kanonischen Spezifikation.

`source_type`: blind_spot, alpha_discovery, regime_failure, model_drift, feature_drift,
active_learning, causal_research, alternative_data, historical_analogue, meta_learning_failure,
unknown_unknown, manual (Quellen A–J des Auftrags).

`production_class`: research_only < abstention < rerank < score < weight – die höchste Wirkungsart,
für die der Vertrag überhaupt gedacht ist. `maximum_initial_influence` <= Klasse und <= WEIGHT_10.

## Regeln

* **Unveränderlich:** Registry `outputs/intelligence/contract_registry.jsonl` (append-only,
  Hash-Kette, gespeicherte Spezifikation muss zum Hash passen). Gleiche `hypothesis_id@version` mit
  anderem Hash → `INVALID_MODIFIED` (nie bewertet, nie angewendet). Neue Schwelle/Richtung/Merkmal →
  neue ID oder Version.
* **Mindest-Evidenz** darf die Policy-Untergrenzen (50 Beobachtungen, 20 unabhängige Tage, 90 Tage
  Spanne) nicht unterschreiten.
* **Similarity-Check** gegen alle registrierten Verträge (auch abgelehnte/abgelaufene): Jaccard über
  Signalmerkmale, Operator, Richtung, Klasse, Scope; identisches Signal+Richtung ≥ 0,9. Ab 0,80 →
  `DUPLICATE_SIMILAR`, außer `distinct_from: {ids, justification}`.
* **H0/H_alt** werden aus der Spezifikation erzeugt und in der Registry gespeichert. Gewinnt H_alt
  signifikant → REJECT; kein Vorzeichenwechsel, sondern neue Hypothese mit neuer Registrierung.
* **Signal** = sicherer Ausdruck (Namen, Zahlen, + − × ÷, Vergleiche, and/or, abs/min/max). Fehlende
  Merkmale → Regel nicht auswertbar (nie 0).

## Zustände (`modules/promotion_controller.py`)

```
IDEA → HISTORICAL_RESEARCH → HISTORICALLY_VALIDATED → PROSPECTIVE_CHALLENGER → FORWARD_VALIDATED
     → GUARDED_PRODUCTION (Abstinenz) → LIMITED_PRODUCTION (Rerank/Score/10 %/25 %) → FULL_PRODUCTION (nur Mensch)
Seitwege: DEMOTED, REJECTED, EXPIRED (terminal)
```

* Historische Evidenz kann höchstens `HISTORICALLY_VALIDATED` erreichen (Übergang
  HISTORICALLY_VALIDATED → FORWARD_VALIDATED ist verboten).
* Jeder Übergang wird in `outputs/intelligence/promotion_transitions.jsonl` mit `previous_state,
  new_state, timestamp, reason, evidence_snapshot, metrics, code_version, data_version` und
  Hash-Kette protokolliert; nicht erlaubte Übergänge werfen `TransitionError`.
* Neue Verträge werden bei Registrierung direkt als `PROSPECTIVE_CHALLENGER` eingefroren (für die
  Champion-Trade-Population gibt es keine belastbare Historie).

## Inventar bestehender Research-Hypothesen

`hypothesis_contract.research_inventory()` zählt die Research-Lab-/Alt-Data-Hypothesen
(`hypothesis_db.json`) als `research_only` für Multiple Testing (Stand: 24). Sie betreffen das
ML-Querschnittspanel, nicht die Champion-Trades, und erhalten daher keinen direkten Einfluss; ein
Transfer erfordert einen eigenen Vertrag mit Champion-Population.
