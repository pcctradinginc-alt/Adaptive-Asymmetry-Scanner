# Options learning: evidence-first expansion (2026-11-10)

## Boundaries
* **Production unchanged:** `config.yaml`, `config/economics_first.yaml`,
  `config/champion_protocol.yaml`, production prompts, ROI, MC, risk,
  trading and positions unchanged. Frozen Champion SHA must pass CI.
* Every new output is research-only, with **no broker orders** and no
  automatic promotion. No new backtest may be called out-of-sample if it
  is evaluated on data used to select its rule.
* The locked holdout remains contaminated; these additions do not cure it.
* More candidates/observations are not evidence of positive net alpha.

## Historical ENTRY_NOT_ARCHIVED labels
`python -m modules.option_label_forensics --as-of YYYY-MM-DD` creates
`outputs/research/option_learner/entry_provenance_audit.json`.
It checks the original OCC symbol, entry date, entry record ID, matching
usable RTH quotes and recorded ask/bid for every matured option expression.
It reports the first available archive day and loss reasons. It cannot
recover quotes which were never recorded. **No historical candidate label
is rewritten and zero synthetic quotes are produced.** The historical 308
missing entries are *possible research gaps*, not automatically 308 restored
labels. Manual exact records with verifiable timestamp/source would be
necessary for any later recovery procedure, in a separately reviewed PR.

## LLM census isolated from scanner
`config/llm_census.yaml` now freezes **5** randomly selected hard-filter
survivors on the signal day before sector/Haiku/MC/Sonnet filtering.
`outputs/research/llm_census_inputs/YYYY-MM-DD.json` is create-only,
checksummed and committed by the existing scanner job. No Sonnet calls
remain in scanner finalization when execution=external.

On successful scanner completion, the new `llm_census_research.yml`
workflow reads that day's frozen input. It rejects stale cohorts, missing
API keys, unknown/pressured cost budgets, insufficient monthly API
headroom and incomplete frozen input checksums. It runs at most five deep
analyses on a separate budget and writes observations to
`outputs/research/llm_census_runs/YYYY-MM-DD.json`. A second attempt
cannot overwrite the first observations. Their LLM costs are attributed
to the **research** scope (new opt-in `telemetry_workflow`, default
DeepAnalysis production behavior unchanged).

The research analyzer intentionally withholds unavailable frozen macro and
48h price-move information rather than querying later data. It can only
run on the original signal UTC date; a delayed job is SKIPPED, never
backfilled with later market prices. The weekly census report joins
the sidecar with underlying outcomes for **descriptive** estimates only.
This is not executable option P&L and not a validated signal.

Once actual daily cost, coverage and latency are stable, a new
research-only PR may increase k to 10 then 20; the code currently
has hard limit MAX_K=10 and monthly API budget remains $10 soft.

## LLM-free, options-based fixed research baseline
`modules/llm_free_challenger.py` reads *only* the research-census
option rows, where production LLM direction is null, and ranks them with
predeclared point-in-time relative volume, sentiment, price movement
and observed contract friction. At most 6 ranked + 4 random audit
long-call contracts per day, no LLM calls, no trades. It records the
selection **after** the scanner. This is a fixed comparator, NOT an
autonomous trained alpha model.

Paper outcomes only exist if a usable RTH **entry ASK after prediction
and on a later trading session**, plus a valid RTH **exit BID after the
5-/20-calendar-day horizon**, can be reproduced from the quote archive.
Missing quotes remain `NO_ENTRY_ARCHIVED`/`NO_EXIT_ARCHIVED` and do
not get zero P&L. The baseline does not assume its scan-day quote was
fillable after the later decision; such a claim would be lookahead.
The baseline currently makes no extra Tradier calls. Consequently,
its next-session entry coverage may remain very limited until a
separately quota-verified quote-capture extension is approved.

Scanner job: `python -m modules.llm_free_challenger propose --date TODAY`.
Weekly job: `python -m modules.llm_free_challenger evaluate --date TODAY`.
Output: `outputs/research/llm_free_challenger/`. Present policy result
is always `NEED_FORWARD_EVIDENCE`. Once enough independent days and
verified net results exist, a future experiment can compare baseline vs
existing option_learner vs LLM-additive selection under matched universe,
entry schedule, fees and risk limits.

## Option-snapshot capacity (40 -> 80)
`config/research_capture_expansion.yaml` records a **conditional**
research-only 80-snapshot ceiling. It defaults to disabled:
the Tradier provider plan is configured as `unknown`; 80 cannot be
justified as free or safely runnable without actual quota/call and
runtime evidence. The 40 baseline remains operational.

Activation requires, together:
1. Explicit `enabled: true` AND
   `provider_quota_verified: true` only after a documented real
   provider quota and runtime evidence;
2. explicit runner environment
   `RESEARCH_EXTRA_OPTION_SNAPSHOTS_APPROVED=1`;
3. valid unchanged 40-snapshot production baseline and a healthy
   cost budget with >= $3 projected monthly headroom.

The only quantity affected after approval is the number of
**research-census option snapshots**, not production gates.
Be aware every research contract can incur multiple chain calls and
later capture targets. If the provider quota cannot be verified,
leave disabled; do not invent vendor costs.

## Before declaring a learning improvement
Require a stable, paired, forward-only evaluation with genuine archived
ask/bid exits and sufficient independent entry days. Include never-trade
and all-trade baselines, holdout reuse checks, transaction costs, regime
strata, calibration and selection propensity weights. **No automatic
parameter changes or merge to the Champion are authorized.**

## Rollback
Disable external census in `config/llm_census.yaml` (research only),
remove the independent workflow, or disable the LLM-free baseline step.
Research snapshot expansion is disabled unless specifically approved.
The existing scanner policy remains unchanged.
