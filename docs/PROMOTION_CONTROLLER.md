# PromotionController

`python -m modules.promotion_controller [--notify]` – täglich in `feedback.yml` nach `feedback.py`.

## Ablauf

1. `resolve_outcomes`: Decision-Ledger ↔ `history.json` (umgesetzte Trades → `closed_trades`,
   abstinierte → `counterfactual_closed` (gleicher Lebenszyklus wie echte Trades: Exit-Regeln TP/SL/Time, Outcome-Methode)), append-only `decision_outcomes.jsonl`.
2. Verträge registrieren/prüfen (Registry-Kette, Hash).
3. Je Vertrag: **Forward-Evidenz** nur aus Ledger-Zeilen mit Entscheidungszeit ≥ `forward_start` und
   > `registered_at`, gleichem `spec_hash`, im Scope, auswertbar, mit Outcome. Die Regel-Auswertung
   stammt aus dem Entscheidungszeitpunkt (eingefroren), nie aus Nachberechnung.
4. Kennzahlen: n, unabhängige Tage, Kalenderspanne, Treffer/Treffer-Tage, Win Rate, Mittel/Median,
   Expectancy, Profit Factor, MFE/MAE (falls vorhanden), Max Drawdown, Brier, ECE, Precision,
   False-Positive-Rate, Δ Expectancy ggü. Champion-only, Block-Bootstrap-CI über Signaltage,
   Ausreißer-Trims (Top1/Top3/Top5 %), drei Zeitfenster, Regime, Sektor-Konzentration,
   Abstinenz-Bilanz (`n_blocked, blocked_trade_win_rate, blocked_trade_expectancy, blocked_trade_mean_return,
   avoided_losses, missed_winners, net_value_of_abstention`).
5. **Multiple Testing:** α_eff = 0,05 / (Familiengröße × geplante Looks). Familie = production_class,
   Familiengröße = alle je registrierten Verträge (auch abgelehnte). Looks höchstens alle 28 Tage
   (`promotion_looks.jsonl`); Promotion-Entscheidungen nur an Looks; Looks erschöpft → EXPIRED.
   `number_of_hypotheses_tested` = Promotion-Verträge + Research-Inventar.
6. Entscheidung (`decide`), Übergang protokollieren, `promotion_state.json` schreiben (`state_hash`).

## Entscheidungen

| Entscheidung | Wann |
|---|---|
| ROLLBACK | Integritätsfehler bei aktivem Einfluss, oder menschlicher Rollback (`promotion_approvals.yaml`) |
| REJECT | H_alt signifikant (CI-Obergrenze < 0); Looks erschöpft (→ EXPIRED) |
| KEEP_SHADOW | NEED_MORE_DATA (n, Tage, Spanne, Treffer, < 2 Regime, Sektor > 60 % ohne Sektor-Scope), Kriterien nicht erfüllt, kein Look fällig |
| ALLOW_ABSTENTION | Gate bestanden, Klasse abstention, Policy erlaubt (Default-Obergrenze) |
| ALLOW_RERANK / ALLOW_10_/25_PERCENT_WEIGHT | nur wenn `max_automatic_influence` es zulässt (Default: nein → Empfehlung) |
| RECOMMEND_FULL_PROMOTION | nach WEIGHT_25 mit weiterer Evidenz – Mensch + PR |
| DEMOTE | vorab festgelegte Demotion-Kriterien im rollierenden Fenster seit Promotion |

Promotion-Gate (alle Bedingungen): Δ Expectancy > `delta_expectancy_min`; CI-Untergrenze (α_eff) >
`ci_lower_min`; Effekt bleibt nach Top1/Top3/Top5 %-Trim > 0; ≥ 2 von 3 Zeitfenstern positiv; Brier der
Policy nicht schlechter (+0,005); Drawdown nicht deutlich schlechter.

Nie automatisch (Policy `never_automatic`): Champion-Wechsel, neue Assetklasse/Short-/Optionsstrategie,
höheres Risiko/Positionsgröße, neue externe Datenquelle als Gate, > 25 % Modelleinfluss, Risk-Gate-Änderungen.

## Benachrichtigung

Neue Promotion-Kandidaten → `INTELLIGENCE PROMOTION CANDIDATE` (Mail über `modules.mailer`, einmal je
Hypothese×Stufe, `promotion_notified.json`) und Weekly Report Abschnitt 18.

## Populationen (`eligible_stage`, seit 2026-10-03)

Eine neue Population ist eine neue wissenschaftliche Fragestellung und damit ein neuer Vertrag
(neue `version`), nie eine Erweiterung eines bestehenden.

| | v1 (`PROM-ABST-00x@v1`) | v2 (`PROM-ABST-00x@v2`) |
|---|---|---|
| Population | Champion-Trades (kein Feld = `CHAMPION_TRADE`, Spezifikation unverändert) | `FINAL_MC_SURVIVOR` – jeder Survivor von Stufe 8, unabhängig von späteren Gates |
| Ledger | `outputs/intelligence/decision_ledger/` (Adapter) | `outputs/intelligence/final_mc_ledger/` (`modules/final_mc_ledger.py`) |
| Outcome | Optionsrendite, nur Quote-basiert | Richtungsrendite des Basiswerts 45 T (20/60 T beschreibend), MFE/MAE |
| Vergleich | durchgelassen vs. alle Champion-Trades | triggered vs. non-triggered **innerhalb** der Survivors |
| Bonferroni-Familie | `abstention` | `abstention@FINAL_MC_SURVIVOR` (v1-Alpha unverändert) |
| Unabhängigkeit | unabhängige Signaltage | zusätzlich Ereignis-Cluster (gleicher Ticker ≤ 10 T = ein Ereignis); Bootstrap über Tage UND Cluster, beide Untergrenzen > 0 |
| Höchster automatischer Zustand | GUARDED_PRODUCTION (Abstinenz) | FORWARD_VALIDATED, Einfluss NONE (`cap_population`) – Übertragung auf Champion-Trades nur per neuem Champion-Vertrag (PR) |

Pflichtfelder für Populations-Verträge (`hypothesis_contract.validate_stage`): `eligible_stage`,
`population_definition`, `horizon_days`, `baseline`, `baseline_population` (= `eligible_stage`),
`minimum_independent_event_clusters`, `promotion_criteria.min_fired_event_clusters`, gespeicherter
`spec_hash` (muss zur Spezifikation passen). Der Adapter wertet nur Champion-Verträge auf
Champion-Trades aus. Test: `tests/test_final_mc_ledger.py` (End-to-End inkl. forward_start-Grenze
und v1/v2-Trennung).

## UNIVERSE_V2-Segmente (`production_class: universe_segment`)

Population `V2_CANDIDATE`, `universe_version: V2`, Evidenz nur aus `outputs/universe/v2_ledger` auf
`net_realizable_return`. Leiter: NONE → RERANK_ONLY → WEIGHT_10 → TRADE_RECOMMENDATION_ENABLED; automatisch
höchstens FORWARD_VALIDATED, jede weitere Stufe nur per `approved_level` in `config/promotion_approvals.yaml`
(`apply_approval`: eine Stufe je Look, nur bei aktuell erfüllter Evidenz). Demotion eine Stufe je Verstoß
(`universe_v2_ledger.segment_demotion_reasons`). Details: `docs/UNIVERSE.md`.
