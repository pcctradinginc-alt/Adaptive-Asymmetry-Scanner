# Throughput shadow – more measured opportunities, NOT more production trades

## Problem and boundary
On 2026-11-10, 175 sector survivors led to only 8 quote probes, 2 Deep
Analyses and 0 proposed trades. Budget-deferred candidates are *not* rejected
for economic reasons. The Champion was frozen on 2026-11-09, starting
2026-11-11: changing `config/economics_first.yaml` invalidates that
pre-registered protocol. This change deliberately DOES NOT modify
`config.yaml`, `config/economics_first.yaml`, the Champion hash, any
production filter, order path, option selection, ROI, MC, risk or sizing.

## Research-only implementation
`config/throughput_shadow.yaml` controls a separate, explicitly bounded
research pass AFTER the Champion decisions and the existing LLM census.
It uses the same already collected sector survivor inputs, deep-copied
before Champion acquisition. No result ever flows into the production
candidate, shortlist, analyses, proposals, score, trade, direction or status.

* Up to **24 total quote probes** with 6 reproducible pseudo-random audit
  slots; previously executed Champion probes are reused when the verdict
  was actually observed. Thus there are never 24 *extra* unique probes
  simply due to this research pass. New probes are separately counted.
* At most **12 Haiku shortlist slots overall**. Candidates already sent
  to Champion Haiku are excluded from research Haiku and Sonnet calls,
  avoiding double payment. Extra candidates receive at most one
  independent Haiku batch.
* At most **6 research Sonnet calls** (2 audit slots), restricted to
  newly prescreened YES candidates. The shadow intentionally does NOT
  apply the production Pre-MC gate before Sonnet, to acquire evidence
  about potential false negatives. This means shadow Sonnet survivors
  are **not** executable/eligible Champion trade candidates.
* The pre-existing economics-first earnings and quote viability check
  applies unchanged. Bad/unavailable quotes remain distinguished.
  The existing deterministic priority is retained; this PR does
  NOT claim to implement or validate a liquidity-optimized ranking.
* Stop if no Anthropic key, monthly LLM budget is unknown or pressured,
  projected monthly spend leaves less than $3 headroom, or the run cannot
  finish while reserving at least 120 seconds for finalization.
  Even if YAML is changed, hard caps are 24/12/6 and 240 seconds.
  Error states are not silently treated as failed investment signals.
* All research observations are nested in
  `candidate_ledger.features.throughput_shadow`. Each row identifies
  quote status and reason, Haiku outcome, Sonnet outcome and (if observed)
  research-only impact, surprise and direction. There is no top-level
  `direction`, rejection, or proposed trade. `stats.throughput_shadow`
  records new vs cached probes, actual LLM throughput and why a run
  was skipped. Existing underlying forward outcomes can later be paired,
  but **unquoted securities cannot be assigned executable option P&L**.

## Important caveats and evaluation
This is a *prospective acquisition/measurement improvement*, not an
alpha improvement. The 6 random quote slots are selected without replacement
from the population outside the deterministic priority top set. They do
not make the whole sample unconditionally representative. Haiku/Deep
Analysis selections remain conditional and potentially biased.
Interpret observed 5d/20d underlying returns separately from
time-stamped, independently verified option ask-entry/bid-exit outcomes.

The 24-probe challenger cannot by itself prove 20–30 **reliably priceable**
opportunities a day. Record the actual quote pass/unknown/reject counts;
the restricted 12-slot Haiku stage may still be the binding constraint.
A subsequent candidate for promotion requires a SEPARATE, registered
experiment with positive forward net options evidence, a true matched
Champion/benchmark comparison, adequate independent entry days and
appropriate familywise alpha spending. Do NOT optimize thresholds on
observed holdout outcomes or modify the frozen Champion implicitly.

## Release/test/rollback
`pytest -q tests/test_throughput_shadow.py tests/test_economics_first.py
tests/test_champion_protocol.py`, then full CI. A PR is required.
Rollback: set `enabled: false` in the shadow config (or revert the
integration): Champion remains identical. At the first live scan verify
`stats.throughput_shadow.status` and monthly spend before any budget
increase. No automated brokerage action is provided here.
