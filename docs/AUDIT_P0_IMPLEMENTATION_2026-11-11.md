# Forensic Quant Audit: P0 correction (2026-11-11)

Status: **partial safety/measurement patch, NOT an alpha confirmation**.

## Implemented
- Fresh candidate observations carry `decision_frozen_at` after analysis and contract selection, plus `entry_anchor_policy: decision-frozen-v1`. Their archived entry quotes must be observed at/after the frozen decision time. Existing historical `timing-v2` records retain the original recorded signal-timestamp convention and are not silently rewritten.
- Strict provenance for newly frozen entries requires exact archived quote record IDs and explicitly rejects a pre-decision entry. Legacy price-matched quotes remain exploratory and **are not equivalent to verified executable entries**.
- Default unit sizing now enforces `portfolio.max_position_pct`: if the debit of one option exceeds it, zero contracts are recommended and the proposal is dropped. The pipeline separately counts risk-cap rejections.
- The per-session drift/volatility MC hit simulation now uses approximately `round(DTE*252/365)` trading steps instead of treating every calendar day as a session. Deterministic candidate-local and contract-local random seeds make identical MC inputs reproducible. This is a timebase correction, **not proof that the signal drift is correct**.
- The champion test no longer declares a zero p-value when the day means have zero sample variance. New champion cohort `champion-v3-2026-11-11` starts 2026-11-12 and needs a fresh global alpha-ledger allocation. Previous cohort reports and outcomes are retained.

## Not implemented / cannot be claimed
- A licensed survivorship-free PIT US equity universe, complete adjusted OCC contract history and vendor-vintage snapshots.
- A complete independent entry-cohort statistical validation and non-overlapping market-regime test; these require a further preregistered protocol and observations.
- Broker execution, slippage/market-impact curves, capacity-adjusted performance, or evidence of post-cost alpha.
- Wholesale deletion of old registered research engines and champion gates: existing pre-registered controls should be evaluated with the already frozen `no_mc_gates` and other matched challengers, not silently reconfigured on seen outcomes.
- GitHub branch protection / mandatory owner reviews: must be enabled under repository rules by an authorized owner.
- A newly validated autonomous learner or automatically promoted model.

## Test requirements
```bash
pytest -q tests/test_audit_p0_integrity.py tests/test_candidate_ledger.py tests/test_champion_protocol.py tests/test_timing_policy.py tests/test_research_evidence_discovery.py tests/test_mirofish_behaviour.py
pytest -q
```
Merge only if required CI checks are green. Follow up before any positive alpha claim with a new, complete, prospectively recorded trade-entry cohort and externally reproducible market data.
