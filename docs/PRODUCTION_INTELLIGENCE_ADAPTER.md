# ProductionIntelligenceAdapter

`modules/production_intelligence_adapter.py` – **die einzige Schnittstelle**, über die Research
die Produktion beeinflussen darf. Aufruf genau einmal in `pipeline.py` (Stufe 10c, nach Score- und
Korrelations-Gate, vor Schatten-Trades/Sizing/Trade-Eintragung). Test:
`test_pipeline_uses_only_adapter_for_research_influence`.

## Eingabe

Champion-akzeptierte Trades (Signal, `trade_score`, Wahrscheinlichkeit = Final-MC-Hit-Rate, Sektor),
VIX → Regime (`vix_low/normal/high/stress`), Research-Kontext read-only: `safe_mode.json`,
Blind-Spot-Sektoren (`next_validation.json`), ML-Karten (Modelluneinigkeit, erwartetes Drawdown).

## Ablauf

1. `load_verified_state`: `promotion_state.json` (state_hash), Vertrags-Hash = Registry-Hash =
   State-Hash, Registry- und Transition-Kette intakt, State = letzter Übergang der Kette. Jeder Fehler →
   Hypothese ignoriert / Einfluss NONE (fail-safe Champion).
2. Je Trade und Hypothese: in Scope? auswertbar? feuert? → im Decision-Ledger eingefroren.
3. Wirkung nur gemäß freigegebener Stufe:

| Stufe | Wirkung | harte Grenze (Code) |
|---|---|---|
| NONE (Shadow) | keine; nur „was Intelligence täte“ | – |
| ABSTENTION_ONLY | Champion-Trade blockieren → Counterfactual-Trade (gleiche Exit-Regeln, kein Lernen) | nie neuer Trade |
| RERANK_ONLY | Reihenfolge der akzeptierten Trades | fehlt ein Signal → Champion-Reihenfolge |
| SCORE_LIMITED | ± Score-Punkte | `HARD_CAPS["score_points"] = 3` |
| WEIGHT_10 / WEIGHT_25 | P_final = (1−w)·P_champ + w·P_int | w ≤ 0,10 / 0,25; Summe ≤ 0,25 |

Die Policy kann diese Grenzen nur senken (`min`). Angefragte Gewichte (z. B. Meta-Modell will 100 %)
werden gekappt.

4. **Safe Mode** (aktiv laut `safe_mode.json`): kein positiver Score-Boost, kein Ensemble-Gewicht,
   kein Rerank; nur bereits freigegebene defensive Abstinenz greift. Intelligence kann Safe Mode nicht
   überschreiben.
5. Score-Einfluss, der unter `trade_score_min` drückt, wirkt als Abstinenz (nie umgekehrt).

## Ausgabe je Trade (Decision-Ledger `outputs/intelligence/decision_ledger/YYYY-MM.jsonl`)

`original/champion_decision, champion_score, champion_probability, champion_rank, intelligence_decision,
intelligence_adjustment {score, score_raw, probability}, final_production_decision, production_score,
influence_level, abstention_reason, active_hypotheses, hypothesis_versions, promotion_levels,
intelligence {je Hypothese: spec_hash, level, state, in_scope, evaluable, fired, signal_value, applied},
safe_mode_state, meta_model_version, world_model_version, data_snapshot (vix, env_hash), code_commit,
evidence_snapshot, intelligence_rank`; später Outcome in `decision_outcomes.jsonl`.

Damit bleibt jederzeit auswertbar: Was hätte der Champion allein getan, was hätte Intelligence getan,
was wurde umgesetzt.
