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

* **Adaptive 12/16/24 total quote probes** depending on free API-budget
  headroom; `max_quote_probes: 24` is a ceiling, never a required spend.
  Previously observed Champion probes are explicitly chosen first and reused.
  Remaining quote slots alternate between the unchanged legacy priority and
  `cheap_rank_v1` (relative volume, dollar volume, novelty, news count),
  with up to 6 deterministic SHA256 random audit picks. Each arm's selection
  and provenance are logged. The rankers are NOT validated alpha scores. New probes are separately counted.
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
  The original deterministic priority is preserved as the **control arm**;
  the new scoring coefficients are unvalidated and do not establish alpha.
* Stop if no Anthropic key, monthly LLM budget is unknown or pressured,
  projected monthly spend leaves less than $3 headroom, or the run cannot
  finish while reserving at least 120 seconds for finalization. Check the
  budget again before research Haiku and Sonnet, and do not start a new
  LLM stage when the reserve is gone. Hard caps remain 24/12/6 and
  240 seconds. This is a conservative **research spending guard**, not
  a guarantee that the independent production costs can never exceed $10.
  Error states are not silently treated as failed investment signals.
* All research observations are nested in
  `candidate_ledger.features.throughput_shadow`. Each row identifies
  `sample_origin` (cached/legacy/cheap/audit/deferred), quote status
  and reason, Haiku outcome, Sonnet outcome and (if observed)
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

The expanded challenger cannot by itself prove 20–30 **reliably priceable**
opportunities a day. Record the actual quote pass/unknown/reject counts;
the restricted 12-slot Haiku stage may still be the binding constraint.
A subsequent candidate for promotion requires a SEPARATE, registered
experiment with positive forward net options evidence, a true matched
Champion/benchmark comparison, adequate independent entry days and
appropriate familywise alpha spending. Do NOT optimize thresholds on
observed holdout outcomes or modify the frozen Champion implicitly.

## Research comparison (pre-registered before interpreting outcomes)

Report quote-pass rate, executable friction, Haiku YES rate, Sonnet gate pass,
independent signal days, and quote-verified *net* option outcomes separately
by `sample_origin`. Preserve the prior Champion cohort and the 6 random
quote audits. Never infer missed option returns from underlying returns.
Document overlap, unequal selection probability, quote failures, and
right-censoring: distinct arm selections are not a randomized treatment effect.

Evaluate ranking-vs-legacy on future held-out days, not the days used to
choose the scores; at least 20 independent days for initial diagnostics.
No auto-promotion or changed production gates. Options-entry/exit quotes
must exist prospectively before any option P&L attribution.

## Release/test/rollback
`pytest -q tests/test_throughput_shadow.py tests/test_economics_first.py
tests/test_champion_protocol.py tests/test_shadow_acquisition.py`, then full CI. A PR is required.
Rollback: set `enabled: false` in the shadow config (or revert the
integration): Champion remains identical. At the first live scan verify
`stats.throughput_shadow.status` and monthly spend before any budget
increase. No automated brokerage action is provided here.
