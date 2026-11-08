# Autonomous Alpha – E2E-Betriebsprüfung (GitHub-only, research only)

## Goal and non-goals

Verify the operational path **existing prospective alpha signals → immutable
option decisions → genuinely archived RTH bid/ask quotes → quote-verified
net-of-cost outcomes → GitHub-only $100,000 paper account → frozen forward
validation → canonical Research Memory → next Research Director cycle**.

This is a **reconciliation and observability adapter**, not a replacement for
option_link, option_economics, option_validation, option_learner, research_memory
or research_director. There is no second training dataset, no duplicated
P&L label, no broker paper account, no real orders, no production release,
no new paid feed, no API credentials. Tradier remains **read-only market
data** through the preexisting GitHub Actions Secret.

## Operation

The existing US-RTH options capture continues to run once per cron. It
persists the latest daily status and now a separate content-addressed
append-only attempt receipt within the **same options archive**:
`outputs/options_archive/capture_status/runs/YYYY-MM-DD/SHA256.json`.
A later successful capture cannot erase the diagnostic evidence of an
earlier failed in-window attempt. A skipped/out-of-window cron creates none.
The existing single archive Git push and quote collection process are
unchanged.

The weekly existing `.github/workflows/ml_research.yml` calls, in order:
1. Existing `option-link` + economics + existing prospective validation.
2. Existing `paper-account --paper-mode weekly` settlement/report (never
   new entries from aged signals).
3. Existing option learner, Research Improvement, notification, data-quality.
4. **`python -m modules.autonomous_alpha e2e-audit`**: synchronizes the
   **existing** Research Memory with the already finished reports, then
   checks source IDs/valid archived quotes, account receipt integrity,
   capture attempts, frozen qualification and matching Memory states.
5. Existing `paper-report` e-mail includes E2E section H and up to three
   highest-priority issues. The same Research-output Git commit includes
   E2E report, Memory updates and immutable dated proof receipts.

The E2E CLI writes the latest report to
`outputs/research/autonomous_alpha/e2e_audit.json` plus write-once
content-addressed receipt in `e2e_audit_receipts/`. Its nonzero exit code
on a confirmed **TECHNICAL_DEFECT** is included in the existing critical
stage-balance step, which runs after the research Git push. Consequently
artifacts survive an error and the workflow is visibly red.

## Evidence semantics

| State | Operational interpretation | Does it prove alpha? |
|---|---|---|
| NOT_OBSERVED / NEED_MORE_DATA | No RTH capture or account activation, missing quotes, unexpired/missing exits, insufficient forward windows | No |
| DEGRADED | Old capture failure was followed by success, stale recent capture telemetry, report or nonfinal memory is delayed | No |
| TECHNICAL_DEFECT | Invalid journal/receipt hash, broken signal→decision→exit reference, missing confirmed Research Memory receipt, a source-invalid close, unbacked forward CONFIRMED | No – workflow red |
| OPERATIONAL_NOT_ALPHA_PROOF | The source-to-memory path passes integrity checks with currently available evidence | Not unless a separate registered forward verdict is independently qualified |
| confirmed + memorized | Existing validated plan was qualified for human review and matched by Memory, supported by **post-registration, quote-verified settled treated positions** and frozen look receipt, with at least 30 observations and 8 nonoverlapping periods | Scoped research evidence only, never production approval |

The audit **reuses** `option_economics.trades` for verified spreads,
commissions and adverse scenario. It does not infer executable prices from
model values, recreate historical quotes, or use paper-account P&L as a
second independent training observation.

The account is only activated by an actual `PA-INIT` event during a
future valid capture; preexisting experimental entries are not backfilled.
A paper `OPEN` is not an outcome. A verified source `EXIT_OBSERVED` is
not yet a `CLOSE` until the existing ledger reconciles it.
Unexpected differences are surfaced without silent mutation.

## Release checks and observations

- Regression suite: `tests/test_alpha_e2e_operations.py` covers no-data
  bootstrap, quote/capture failures, attempt tampering/recovery, hash
  tampering, memory idempotence, FALSIFIED feedback, fake CONFIRMED without
  genuine quote evidence, and workflow ordering.
- Weekly e-mail: `tests/test_paper_report.py` includes E2E report assertions.
- Mandatory GitHub CI must pass **all** tests before merge. This PR
  must remain in draft until CI and human review complete.
- **Scientific validation needs time**: first in-window Git capture,
  genuine prospective signals, archived exit quotes after the frozen
  20-trading-day horizon and independent non-overlapping test periods.
  No automation can honestly claim that has already occurred.

Known limitation: the latest daily capture status and GitHub Actions
schedule are independent; a delayed/holiday session may show a stale
capture warning, never automatically a confirmed quote-integrity failure.
This is an operational proof for the committed artifacts, not independent
verification of the data provider's real-time market entitlements.
