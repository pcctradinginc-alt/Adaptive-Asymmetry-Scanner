# P0/P1 reliability changes — review contract (2026-11-04)

Scope: surgical remediation for five previously identified P0/P1 defects. No live orders, no relaxed safety gates, no fake market data, no synthetic labels, no changes to alpha promotion policy.

## P0 — Commodity evaluator
- Root cause: `python -m modules.commodity_intelligence evaluate` reached the CLI entrypoint before the later `evaluate()` function was defined; GitHub job logged `NameError: name 'evaluate' is not defined`.
- Fix: execute CLI only after all function definitions. This does **not** imply the commodity factors add economic value.

## P0 — IMF PortWatch
- Observed on 2026-11-04: ports metadata missing `fields`; chokepoints query returns ArcGIS HTTP/JSON `400 Invalid URL`. The public ArcGIS publisher may be updating or withdrawing layers.
- Fix: when and ONLY when endpoint metadata is absent or a query says `Invalid URL`, re-resolve verified publisher endpoints once. Retry only if the discovered service URL/layer actually changed. Preserve original `SCHEMA_CHANGED`/`FAIL` when no valid replacement is found. Never accept guessed fields or stale cache as fresh observations.
- **External acceptance still required:** a later live `External Data Ingestion` followed by `Source Health` must show both `imf_portwatch_ports` and `imf_portwatch_chokepoints` passing, two independent recovery checks per configured policy, fresh archived observations, and global data safe mode correctly deactivating. Code cannot guarantee remote availability.

## P1 — Options RTH quote coverage
- Existing six scheduled captures remain; no out-of-window quote backfill is permitted.
- `scripts/rth_marks.py` now emits explicit `capture_status`: `SKIPPED_OUTSIDE_WINDOW`, `NO_TARGETS`, `ALREADY_CAPTURED`, `PARTIAL_MISSING_QUOTES`, `COMPLETE`, or `ALPHA_LINK_ERROR`.
- Last *in-window* capture status persists at `outputs/options_archive/capture_status/YYYY-MM-DD.json`. Delayed cron runs after the window cannot overwrite evidence from the preceding in-window run.
- Missing quotes (for example the observed CTVA put) are NOT filled with stale or theoretical prices. A `PARTIAL_MISSING_QUOTES` status is not a successful quote archive. Inspect actual daily coverage and provider responses before claiming this P1 is operationally closed.

## P1 — Alpha-to-Options attribution / report freshness
- Previously registered RTH paper decisions existed in `outputs/options_archive/alpha_link/decisions/2026-11.jsonl` (46 entries, 29 `TRADE` and 17 `ABSTAIN` at inspection), while `outputs/research/autonomous_alpha/option_link.json` showed 0, because it preceded the capture.
- After `option_link.decide()`, capture records observed RTH quotes, runs `option_link.settle()` for already matured paper exits using only archived RTH quotes, then refreshes `option_link.coverage()`. It commits paper-outcome journals and the report alongside quote/decision journals.
- Paper decisions remain paper. Validated end-of-horizon outcomes, an immutable ex-ante signal link, and economic performance must still be established prospectively; no P&L claims from currently pending exits.

## P1 — GitHub Actions truthfulness
- `modules.ops_workflows` now adds a `recent_outcomes` breakdown for core workflows, including the actual CI stage-balance lines from ML/feedback logs because GitHub marks `continue-on-error` stages green:
  - `SKIPPED_SCAN` for green guard-only runs with skipped scan job
  - `DEGRADED_STEP_FAILURE` for any failed step hidden inside an otherwise successful workflow/job
  - `TECHNICALLY_SUCCESSFUL` for actual successful steps (but NOT a trade/quote/alpha guarantee)
  - `UNVERIFIED` when job details cannot be read
- `business_evidence` always says `NOT_EVALUATED`; query actual archive/status reports separately.

## Regression commands

```bash
python -m pytest -q tests/test_p0_p1_reliability_20261104.py tests/test_maritime_source.py tests/test_source_stabilization_2026_11.py tests/test_rth_marks.py tests/test_ops_workflows.py tests/test_autonomy_upgrade.py
python -m pytest -q -p no:cacheprovider
```

## Release / Claude Code checklist
1. Review changes for preservation of PIT/as-of, idempotence, data provenance, no live execution, no auto-promotion.
2. Run regression + full CI on the proposed head SHA. Do not call green Actions proof of business outcome.
3. Verify fresh external PortWatch responses (metadata + one representative read-only query), owner/item IDs, unchanged schema invariants, and two recovery passes; record results.
4. Check that a real NYSE in-window capture commits `capture_status`, alpha-linked decisions/outcomes, and updated `option_link.json` counts; verify missing quotes remain marked missing, never inferred. Confirm only mature exits are settled.
5. Match immutable signal IDs in signals/decisions/outcomes, distinct signal dates and 20-trading-day maturity. Do not attribute paper marks to broker executions.
6. Inspect latest `recent_outcomes` for a skipped scanner and (if applicable) a degraded ML step. All required stages must execute before release.
7. Clearly report any unresolved external, time-dependent, or statistical validation as OPEN — do not auto-merge additional changes or lift Safe Mode.
