# LLM-free candidate census (research-only)

Run `python -m modules.research_candidate_census --date 2026-10-29` to
audit **all existing candidate-ledger rows** for that date without LLM calls.
Produces a deterministic 40-ticker audit sample, counts budget deferrals,
and distinguishes underlying outcomes from actually observed option outcomes.

**Important:** This does not acquire new option chains or create shadow marks.
A sample is a proposed research queue, not a filled trade or observed option
quote. It does not change the existing production cap of 8, any gates,
position sizing, risk rules, scanner schedules, or API quotas.

Next phase requires verifying provider call budgets, capture timestamps,
option contract/leg identity and legally available historical marks, then
implementing a separately bounded research-only capture workflow with tests.
Do not equate a larger ledger with more real-option training labels.

## Real-option research sample (since 2026-10-30)

The candidate ledger now acquires real option labels for candidates without
an LLM direction (`modules/candidate_ledger.flush`):

- Budget: the existing `ledger.max_option_snapshots` (40). Directional
  candidates (LLM direction) are served first; the unused rest goes to a
  deterministic hash sample (`sha1(date:research:signal_id)`) of candidates
  without a direction. Never more Tradier calls than before
  economics_first (2026-10-28), no LLM call.
- Contract: `research_direction = BULLISH`, the only direction production
  trades (long call or the replicated call strategy). `direction` stays
  `null`, so scorecards, evidence ledger and meta-label (rows with a
  direction only) are unchanged.
- Rows carry `option_sample: research_census | directional`; unselected
  candidates get no snapshot fields.
- Marks, timing-v2 entries and the quote archive are the existing ones, so
  `research_evidence_discovery` can verify these labels against archived
  quotes like any other.
- Active only with `TRADIER_API_KEY`; switch off with
  `ledger.research_option_sample: false` in `config.yaml` (key deliberately
  absent there, so the champion rule hash stays unchanged).

Why: since economics_first only 1–3 candidates per day reach the LLM; real
option labels fell from about 40 to 1–3 per day.
