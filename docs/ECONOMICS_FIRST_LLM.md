# Economics-first LLM acquisition (2026-10-28)

## Scope
The change is confined to the production scanner's acquisition of Haiku and
Sonnet analyses. It does **not** modify ROI/Monte Carlo formulas, risk limits,
RL, position sizing, learning/promotion, shadow evaluation or feedback jobs.
It does **not** relax a single trading gate.

## Execution
After existing ingestion/FinBERT and sector checks, a bounded quote probe
screens candidates **before Haiku**. It uses the actual production long-DTE
option tier (150–365 days), not the old 21–45d preliminary ROI check. Checks:
earnings exclusion, availability of IV rank and quote, entry debit validity,
and executable entry-to-immediate-liquidation bid/ask friction. The last check
rejects only at >30% on quoted contracts. Unavailable or inconsistent quotes
are `unknown`, never labeled no-alpha. No uncalibrated price forecast enters
this early exclusion.

This acquisition policy selects up to 24 quote probes including 2 deterministic
audit probes, up to 12 Haiku candidates and up to 3 Sonnet candidates,
including 1 deterministic audit slot. The numbers are budget limits, **not
evidence-validated alpha thresholds**. All calculations are deterministic and
do not call an LLM. Sonnet and its downstream unchanged gates then evaluate
the limited subset. Prescreening still applies to those candidates.

If the pipeline no longer has a candidate after either acquisition gate, it
still runs the usual `send_email()`/snapshot/ledger finalization, rather than
silently ending.

## Evidence preservation
All upstream sector-pass candidates were already written to Candidate Ledger
with point-in-time underlying data, before acquisition gates. Every economics
acquisition decision now receives a ledger note, and excluded candidates
receive explicit reasons, e.g. `economics_first_quote_probe_budget` or
`economics_first_haiku_budget`. The existing candidate-ledger update job
continues to track their forward underlying returns, including 1/5/20/45/60/120
days. Existing shadow trades, downstream options outcomes, feedback and all
saved reports are untouched. The new stages do **not** invent an entry
direction, real option contract, or later real-option outcome for a candidate
that was not given a real, time-stamped quote.

Important limitation: this design retains **underlying forward outcomes for
every previously tracked candidate**, but full counterfactual *option* P&L
for an unprobed/unscreened candidate cannot be reconstructed from future
quotes. Full no-LLM vs LLM option comparison requires prospective collection
of sufficient contemporaneous quote snapshots within the independent API-call
budget. Do not claim all options outcomes were preserved for those candidates.

## Release / audit plan
1. PR first: the live `main` branch remains untouched until approved.
2. Run `pytest -q tests/test_economics_first.py` and the full CI suite.
3. Prefer initial rollout with `mode: shadow` to compare funnel and sampling.
   In shadow mode, all original candidates still go to Haiku/Sonnet.
4. After confirming quote coverage, chain latency and false-negative samples,
   switch to `active`. All budget-deferred stocks must remain in the forward
   cohort and count separately from *economically rejected* stocks.
5. Measure per-day Haiku/Sonnet calls, cost, upper bound on missed candidate
   opportunities and sampling uncertainty. Never interpret a reduction in
   proposed trades as evidence of improved alpha.

Rollback: `config/economics_first.yaml: mode: off` retains original LLM
candidate flow. `mode: shadow` runs acquisition decisions without restricting
either LLM stage. The unchanged old ROI Pre-Check still runs after Haiku; its
short-horizon inconsistency remains explicitly a separate subsequent audit.
