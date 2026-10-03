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
