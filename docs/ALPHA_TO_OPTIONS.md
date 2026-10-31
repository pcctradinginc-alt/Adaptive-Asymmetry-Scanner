# Alpha-to-Options Research System

Research and paper evaluation only. No orders, no LLM calls, no production
influence (champion, gates, position sizes, risk limits unchanged). No new
orchestrator and no parallel data store: the link uses the existing journals,
the existing quote archive and the existing workflows.

## Phase A – Linking hypotheses to option decisions and outcomes

### Status before Phase A (verified)

* `modules/autonomous_alpha/forward_options.py` hard-coded every real trade to
  `alpha_id: None, alpha_attribution: UNATTRIBUTED`. That was correct, because
  `trade-outcome-v1` records have no ex-ante hypothesis binding. The 4 real trade outcomes are scanner **paper** trades valued from archived
  bid/ask quotes. They carry no broker fills, so they can never be `EXECUTED` evidence (see below). Since Phase A
  the result is **checked per trade**
  (`option_link.ex_ante_attribution`). A trade counts for a hypothesis only if
  its record carries an `alpha_signal_id` that was frozen before the entry.
  The 4 real trades on main carry none, so they stay `UNATTRIBUTED`
  (`NO_EX_ANTE_SIGNAL_ID`). The trading path is deliberately unchanged.
* Prospective alpha hypotheses with real data: 6 registered ML models
  (`config/model_registry.yaml`) with weekly predictions in
  `outputs/research/ml_predictions/`. There are 8 signal dates (48 rows),
  2026-09-28 to 2026-10-30. Frozen AUTO-* factor hypotheses: currently 0.
* No contemporaneous option decision exists for those 48 predictions. They are
  **not linked retroactively**. The coverage report lists them as `PRE_LINK`
  (`NOT_LINKABLE_RETROACTIVELY`).

### Data flow

| Step | Who / when | Store (append-only, content-hashed) |
|---|---|---|
| 1. Signals | `ml_research.yml` (weekly), `python -m modules.autonomous_alpha option-link` | `outputs/research/autonomous_alpha/option_link/signals/` |
| 2. Decision (TRADE/ABSTAIN) | `options_capture.yml` → `scripts/rth_marks.py --capture-only` → `option_link.decide` | `outputs/options_archive/alpha_link/decisions/` + quote records in `outputs/options_archive/quotes/` |
| 3. Exit quotes | the same capture job (`option_link.due_capture_targets`) | `outputs/options_archive/quotes/` |
| 4. Outcome | weekly (`option_link.settle`), archived quotes only | `outputs/research/autonomous_alpha/option_link/outcomes/` |
| 5. Coverage | weekly (`option_link.coverage`) | `outputs/research/autonomous_alpha/option_link.json` |
| Corrections | human only (`option-link-correct`) | `outputs/research/autonomous_alpha/option_link/corrections/` |

Decisions sit next to the quote archive because the capture job may only
commit `outputs/options_archive/`. That rule is protected by
`tests/test_ops_readiness*` and stays unchanged.

### What is frozen before the outcome

Each **signal** records:

* Hypothesis: ID, version (`spec_hash`), hash, registration time, model ID and
  experiment version (`spec_hash:train_cutoff:code_sha`).
* Source: the hash of the immutable part of the source prediction.
* Timing: signal time (`created_at`) and the PIT data date.
* Position: ticker, direction and horizon (20 trading days).
* Context: features and market regime, taken only from rows of the prediction
  date (no labels).
* Comparison group: `REFERENCE` is the frozen benchmark `momentum_12_1`,
  `TREATMENT` covers the challengers and AUTO-*, and `CONTROL` is a
  deterministic random sample from the same universe on the same date.
* Ex-ante expectation: the locked-holdout mean, otherwise `UNKNOWN`.

Each **decision** (one per structure) records:

* Rule: version and hash.
* Prices observed: the underlying quote, expiry and DTE, and per leg the OCC
  symbol, bid/ask, spread, OI, IV, quote time and archive `record_id`.
* Economics: net debit at ask/bid, capital required, max loss and breakeven.
* Predicted edge: marked `HEURISTIC_UNCALIBRATED`; it does not gate the
  decision.
* Result: `TRADE` or `ABSTAIN` with reasons, the exit target and deadline, and
  the evidence class at entry.

Each **outcome** records:

* The exit quotes with `record_id`, or `EXIT_MISSING`.
* The evidence class: `EXECUTED`, `QUOTE_VERIFIED_PAPER` or `UNVERIFIABLE`.

### Contract rule `alpha-option-rule-v1` (config/alpha_option_link.yaml)

* **Structures:** only `LONG_CALL` and `BULL_CALL_SPREAD`. Bearish tails are
  documented as `NO_ALLOWED_STRUCTURE`.
* **Contract selection:**
  * Expiry: the first expiry with DTE 45–90.
  * Long leg: the call nearest the underlying mid (tie → lower strike).
  * Short leg: the production window [1.05, 1.20]× long strike, target 1.10×.
    It is identical to `options_designer.pick_spread_leg_strike` (tested).
  * Size: 1 contract.
* **ABSTAIN gates:**
  * leg not usable for RTH;
  * bid < 0.05;
  * spread > 25 %;
  * OI < 50;
  * IV unknown or > 150 %;
  * capital > $5,000;
  * debit ≤ 0;
  * no expiry or no short leg in the window.
* **Timing:**
  * The decision is the first usable RTH capture after registration.
  * If no decision is made within 3 trading days, the result is
    `DECISION_WINDOW_MISSED`.
  * The first observation is binding; there is no re-quoting for a better
    price.
  * Exit is the first leg-paired usable RTH quote on or after the 20th trading
    day, at most 5 trading days later. Otherwise the result is `EXIT_MISSING`
    (`UNVERIFIABLE`, never a result).
* **Costs:** $0.65 per contract and side; the adverse scenario adds 25 % of the
  spread (applied in Phase B).
* **Rule changes:** any change under the same `rule_version` stops the
  pipeline (`EvidenceIntegrityError`). Use a new version instead.
* **Budgets:**
  * ≤ 30 tradable signals per date (5 of them control);
  * ≤ 15 underlyings and 240 s per capture run;
  * ≤ 40 exit symbols per capture;
  * existing Tradier source only, no new paid APIs.

### Integrity

* **Hashing and duplicates:**
  * Every line is a `{sha256, payload}` pair; a changed line fails closed.
  * Identical duplicates (git union merge) are tolerated.
  * Conflicting duplicates of the same logical ID: the earliest wins. The
    conflict is reported (`CONFLICT_DETECTED`) and never resolved by outcome.
* **Idempotent IDs:** signal, decision and outcome IDs are deterministic.
  Repeated runs write nothing new.
* **No future information:** the order is enforced as prediction →
  registration → decision → exit. Predictions older than 4 days, from the
  future, from before the hypothesis registration or with a changed spec hash
  are never linked.
* **Corrections:**
  * Versioned and human-initiated (`EXCLUDE_DATA_ERROR`,
    `DOWNGRADE_EVIDENCE`).
  * The original lines stay byte-identical.
  * An evidence class is never upgraded.
* **EXECUTED evidence:** requires a documented broker execution
  (`broker_execution` with broker, plus order ID, fill price and fill time for
  entry and exit) **and** an ex-ante `alpha_signal_id`.
  * Paper bookings and archived bid/ask valuations stay `QUOTE_VERIFIED_PAPER`
    at best.
  * The classification is derived only; it is never written back, never
    upgraded, and never changes a historical record.
* **Coverage:** `option_link.json` is reported per hypothesis and structure.
  It includes TRADE/ABSTAIN with reasons, pending and overdue decisions,
  exit status, evidence classes, distinct signal dates, PRE_LINK counts and
  the check of executed trades. The status is `NEED_MORE_DATA` until there are
  ≥ 20 quote-verified outcomes on ≥ 6 signal dates.

Phase A makes **no economic statement** (`economic_result:
NOT_EVALUATED_IN_PHASE_A`).

## Phase B – Economic evaluation per hypothesis (`option_economics.py`)

Inputs are only the Phase A journals and the quote archive. Nothing is
re-quoted, re-selected or optimized on outcomes; the contract rule stays the
frozen `alpha-option-rule-v1`.

* **Execution model.**
  * Entry: long @ ask, short @ bid. Exit: long @ bid, short @ ask.
  * Fees: $0.65 per contract and side.
  * **ADVERSE** (primary scenario): every fill moves 25 % of the leg's
    bid/ask spread against us, with bids floored at 0.
  * **BASE**: plain bid/ask.
  * Hand-computed examples are in `tests/test_alpha_option_economics.py`.
* **Evidence classes** are kept separate and never mixed into one metric:
  * `EXECUTED`: real trades with an ex-ante `alpha_signal_id` (currently 0).
  * `QUOTE_VERIFIED_PAPER`: every entry and exit quote is re-verified in the
    archive (record ID, contract, price, RTH usability, day, multiplier).
  * `UNVERIFIABLE`: counted with reasons, never in a metric.
* **Metrics** (verified paper positions):
  * n, winrate, average win and loss, net EV in return and USD;
  * profit factor, capital-weighted return;
  * max loss, structural max loss;
  * drawdown on the cumulative P&L series and peak concurrent capital;
  * independent periods (non-overlapping 28-day windows, same blocking as
    `option_learner._periods`);
  * concentration (top ticker and top date share of |P&L|);
  * results by regime;
  * quote coverage and failure reasons.
* **Comparisons** use the same population and budget: per signal date and
  equal budget per signal, with ABSTAIN = cash = 0, on dates whose decisions
  are all final.
  * **Base strategy:** `CONTROL`, the identical rule on random names of the
    same universe and date.
  * **No trade:** 0.
  * **Frozen reference:** `REFERENCE`, the benchmark model.
  * **Missing results** never count as positive. The primary view excludes
    them and reports coverage; the conservative view books them as a total
    loss.
  * Confidence intervals come from a block bootstrap over non-overlapping
    windows with t-quantiles. They are descriptive, not a test; the test
    happens in Phase C.
* **Research decision** (never an order), one of:
  * `NEED_MORE_DATA`: fewer than 20 verified outcomes, fewer than 6
    independent periods, or quote coverage below 80 %;
  * `NO_ECONOMIC_EDGE_AFTER_COSTS`;
  * `NOT_BETTER_THAN_BASE_STRATEGY`;
  * `ADVANTAGE_NOT_ESTABLISHED`;
  * `CANDIDATE_FOR_PROSPECTIVE_VALIDATION`.

  Per hypothesis, `proposals` names the structure for further prospective
  research. Only candidates are eligible; the one with the highest lower CI
  bound versus the base strategy wins, and a tie goes to the spread. The
  selection used exploratory data, so Phase C validates it on new data only.
