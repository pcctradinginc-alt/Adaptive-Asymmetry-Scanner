# Adaptive Production – Validierung

## Evaluation (nur prospektive Forward-Daten)

`promotion_controller.evaluate_arms` auf denselben aufgelösten Champion-Trades:
`CHAMPION_ONLY` vs. `ADAPTIVE_ACTUAL` (tatsächlich umgesetzt) vs. `CHAMPION_PLUS_ABSTENTION_SHADOW`
(hypothetisch alle Abstinenz-Regeln) plus Ranking-Vergleich (Champion- vs. Intelligence-Rang:
Precision@1/@3, Mittel Top1/Top3). Metriken: Win Rate, Expectancy, Profit Factor, Sharpe, Sortino,
Max Drawdown, MFE/MAE, Brier/ECE, Trade Count, Missed Winners, Avoided Losers. `data_kind =
prospective_forward_only` – keine Vermischung mit Backtest oder Walk-Forward. Champion-only wird auch
nach einer Aktivierung parallel weiter aufgezeichnet (champion_decision in jeder Ledger-Zeile).

## Tests (`tests/test_promotion.py`, 33 Tests)

| Bereich | Tests |
|---|---|
| Unveränderlicher Vertrag, neue Version, Validierung | `test_immutable_contract_and_new_version`, `test_contract_validation_rules`, `test_repo_contracts_valid_and_complete` |
| Hash-/Registry-Manipulation | `test_registry_tamper_detected_fail_closed`, `test_state_file_manipulation_ignored`, `test_contract_changed_after_promotion_rolls_back` |
| Zustandsmaschine, Log, FULL nur Mensch | `test_state_machine_transitions_logged_and_guarded` |
| Future-only / keine historische Kontamination | `test_future_only_and_hash_bound_observations` |
| Mindest-N, Tage, Spanne; n=4 mit +300 % | `test_need_more_data_even_with_spectacular_returns` |
| Ausreißer erklärt alles | `test_single_outlier_explains_effect_no_promotion` |
| Ein Sektor / ein Regime | `test_single_sector_dominance_needs_scoped_hypothesis`, `test_single_regime_insufficient` |
| Multiple Testing, Looks, Ablauf | `test_multiple_testing_alpha_and_look_spending`, `test_looks_exhausted_expires` |
| H_alt → REJECT ohne Vorzeichenwechsel | `test_h_alt_wins_reject_no_sign_flip` |
| Promotion → Abstinenz → Alpha-Zerfall → Demotion | `test_acceptance_promote_abstention_then_decay_demotes` |
| Policy-Obergrenze, menschlicher Rollback | `test_policy_max_auto_none_keeps_shadow`, `test_human_rollback` |
| Caps (Score +8→+3, Meta 100 %→10/25 %), Policy hebt nichts an | `test_score_adjustment_hard_cap_and_safe_mode`, `test_meta_model_requests_100pct_weight_capped`, `test_policy_cannot_raise_hard_caps` |
| Safe-Mode-Vorrang | `test_safe_mode_precedence_abstention_only_validated` (+ Score/Weight/Rerank-Tests) |
| Regime-/Sektor-Scoping, fehlende Merkmale | `test_regime_and_sector_scoping`, `test_missing_feature_never_fires` |
| Rerank-only, nie neue Trades | `test_rerank_only_reorders_never_adds`, `test_adapter_never_creates_trades_property` |
| Production-/Counterfactual-Ledger | `test_default_no_state_champion_unchanged_but_logged`, `test_counterfactual_outcome_of_blocked_trade_resolved`, `test_feedback_keeps_open_abstention_shadows` |
| Research Director/Agents ändern keine Produktionsconfig; einziger Codepfad | `test_research_components_cannot_write_production_config`, `test_pipeline_uses_only_adapter_for_research_influence` |
| Policy-Pin + CODEOWNERS | `test_policy_pinned_and_codeowned` |

## Aktueller Befund (2026-10-03)

Fünf Abstinenz-Verträge registriert und als PROSPECTIVE_CHALLENGER eingefroren; Forward ab 2026-10-05;
0 Forward-Beobachtungen → alle `KEEP_SHADOW / NEED_MORE_DATA`, Einfluss NONE. Ein empirischer
inkrementeller Nutzen gegenüber Champion-only ist **noch nicht** messbar. Realistische Dauer bis zum
ersten Gate: mindestens 90 Kalendertage und 60 aufgelöste Champion-Trades mit ≥ 15 Regel-Treffern.
