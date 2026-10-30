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
