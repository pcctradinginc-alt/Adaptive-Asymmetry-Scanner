# PR #180 — Research Data Utilization & Candidate Coverage Audit

## Goal and scope

This is an **observability-only** change. It does not turn unused sources
into indicators, make trading decisions, modify a frozen research/alpha
protocol, increase any market-data or option-quote API budget, fetch new
historical option marks, or train a second learner.

The established Scientific Hypothesis Factory, Research Director and
alpha-option forward/quote-validation chain are unchanged.

## 1. Previously orphaned candidate census

Existing \`modules/research_candidate_census.py\` is the basis for a new
**weekly-only** research-workflow step. The only new mode,
\`python -m modules.research_candidate_census --latest-completed\`,
selects the most recent ledger day **strictly earlier than the UTC run
date**, never a partial same-day sample, and writes a daily JSON snapshot
to \`outputs/research/candidate_census/YYYY-MM-DD.json\`.

It counts **all existing candidate-ledger rows**, including economics-first
and quote-budget rejects, and reports rejection reasons, ticker counts,
underlying marks and recorded real-strategy marks. It deliberately does
**not** mark an outcome as quote-verified. Its marker
\`LEDGER_REPORTED_NOT_QUOTE_VERIFIED\` and existing
\`research_quote_reconciliation\` distinguish recorded marks from
actual RTH archived entry/exit bid/ask proof.

A stable, hash-based ticker list is a **40-name audit sample**, NOT an
order, quote request, or backfilled outcome.

If no complete ledger day exists, invalid JSON is found, or inputs are
missing, the stage fails explicitly and appears in the existing research
stage-balance report as degraded; never silently outputs zero successes.

## 2. Source-utilization and cost inventory

A new **read-only**, weekly \`modules.research_source_utilization\` step
reads exactly the already published
\`outputs/external_data/health/source_health.json\` and
\`outputs/external_data/health/storage_telemetry.json\` (measured per-source
telemetry). It writes \`outputs/research/source_utilization.json\`.

The initial scope is the eight source IDs flagged in the 08 November
code-linkage audit, not every collected dataset:

| Source | Human review priority | Alpha status |
| --- | --- | --- |
| FRED/ALFRED liquidity | P1 | NOT_EVALUATED |
| FOMC calendar | P1 | NOT_EVALUATED |
| BLS calendar | P1 | NOT_EVALUATED |
| BEA calendar | P1 | NOT_EVALUATED |
| ClinicalTrials sponsors | P2 | NOT_EVALUATED |
| USAspending awards | P2 | NOT_EVALUATED |
| SEC 13F | P3 | NOT_EVALUATED |
| NOAA Storm Events | P3 | NOT_EVALUATED |

For each, we expose latest ingestion health, measured historical storage
costs and backfill bytes, and the currently **manual** source-reader
classification. The report does **not** claim to automatically detect
a new consumer from ingestion PASS. Missing cost metrics stay null, not
zero. No source is silently disabled.

The source list is an audit baseline. Updating the mapping when research
consumers change requires code review.

## 3. Separate future registration — not part of this PR

Potential new research protocol: **FRED vintage changes + time to scheduled
BEA/BLS/FOMC release**. Before any beta alpha-scan integration:

1. Freeze a new versioned registry of allowed economic, falsifiable signals,
   lag/horizon, sector/regime mapping and explicit timestamps (PIT).
2. Assess revised release dates and ALFRED vintages as-of. An observation
   date is **not** automatically a valid publication timestamp; no backfill
   presumed tradable.
3. Freeze a baseline, ablation, placebo, walk-forward vs locked-holdout
   restrictions, FDR/BH error-budget registration and independent OOS
   periods **before** testing the new expressions.
4. Evaluate **incremental** stock advantage first; only then compare net
   options economics using archived, independently verified RTH bids/asks,
   commissions and adverse slippage.
5. Record rejection as well as confirmation to Research Memory, with
   independent human approval for any production change.

This is a **draft research brief**, not a new preregistered hypothesis and
not part of the live alpha-testing multiplicity budget.

## Tests, operational guarantees

- Synthetic tests: date cutoff; all-candidate counts, missing/real option
  marks flagged as not independently verified, corrupt/missing data fail
  closed, source costs & missing cost, and weekly workflow integration.
- Workflow saves new reports through its existing \`outputs/research/\`
  commit step; missed steps appear in the stage-balance result.
- No new paid source or credential, LLM, model, market API, quote request,
  real order, code promotion, or workflow schedule.
- Existing research protocols and trade/risk gates unchanged.

**Success criterion:** audit execution and truthful coverage reporting.
It **does not** establish better alpha selection or any verified positive
options returns. Those require real prospective evidence.
