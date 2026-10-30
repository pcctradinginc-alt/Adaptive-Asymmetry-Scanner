# Evidence-first discovery: research-only first increment

This is **not** completion of the requested P0–P4 program. The repository
already has candidate-ledger, option capture, ROI shadow, ops monitoring,
ML and hypothesis infrastructure. This increment adds an offline, no-LLM
real-option-outcome discovery and holdout check.

Run `python -m modules.research_evidence_discovery --as-of YYYY-MM-DD`.
The command reads existing candidate-ledger files and writes only an
independent research JSON report. It does not trigger on the live scanner.

Integrity:
- Missing option outcomes, Black–Scholes proxies and underlying returns
  are never treated as realized option profit.
- A 20-day label must be at least 21 calendar days old. This is a
  conservative maturity gate, **not** proof of mark provenance. The
  upstream outcome provenance must be audited separately.
- Candidate feature thresholds and the best rule are selected on the
  earlier training dates. The later holdout is evaluated once.
- Reports remain descriptive. They are not significance tests or
  authorization to promote models, send trade alerts, or change gates.
- No API calls, LLM calls, quote budget changes, or production influence.

## Still required before claiming P0–P4 complete

P0: prove exactly-once quote→outcome→training lifecycle in actual
production artifacts, repair gaps and publish denominators.

P1: schedule full-universe daily feature capture, with consistent
point-in-time timestamps and explicit missingness.

P2: verify provider quotas, then implement bounded stratified real
option-contract capture and subsequent 5/20/45-day marks, with
independent quote provenance and transaction costs.

P3: expand beyond this single-feature rule screen to pre-registered
interactions/regime analysis, correct multiple testing and selection bias.

P4: repeated purged walk-forward plus untouched live forward validation
against a simple baseline, with confidence intervals, risk metrics and
explicit human-approved promotion.

Do not interpret this module as evidence of profitable trades.
