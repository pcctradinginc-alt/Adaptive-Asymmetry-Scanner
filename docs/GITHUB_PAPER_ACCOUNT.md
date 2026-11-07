# GitHub-only 100,000 USD Paper Account (draft PR, no live orders)

## Scope and existing architecture

This is an **accounting view**, not a new alpha hypothesis engine or a broker.

Existing prospective flow remains the source of truth:

1. Existing frozen signal and ex-ante hypothesis/model binding in
   option_link.signals, including hypothesis/version, model, regime and comparator group.
2. Existing decisions in option_link.decisions and real RTH quote archive in
   option_quotes, with a choice to trade or abstain *before* outcomes.
3. Account selects **at most one** structure per eligible TREATMENT signal.
   LONG_CALL is preferred ex ante; BULL_CALL_SPREAD is used only when the
   long call is ineligible or exceeds account-level capital/risk checks.
   Research alternatives, frozen ML baselines and random controls **never**
   double-book to the principal account.
4. Existing option_link.outcomes are reverified against exact quote records
   with option_economics._verify, then booked via the existing
   option_economics.position_economics for BASE and ADVERSE P&L.
5. Research outcomes continue to flow into existing
   option_validation, research_memory and research_director by their
   existing route. Account P&L is **not** an additional independent label or
   proof of improved model performance.

The Git journal has separate unique, hashed receipt files under
outputs/options_archive/paper_account/events/: INIT, OPEN, SKIP, CLOSE.
Receipts are write-once (exclusive creation) and deterministic; replay fails
closed on duplicate signals, overdrawn cash, excess account risk, wrong policy
hash, invalid receipts and impossible close P&L. Git history provides an
additional tamper audit. This does not confer external custodial/ledger audit
certification.

## Operation

Policy: config/paper_account.yaml, starting cash **$100,000**. The account is
DISABLED by default and has no approved quote source until a provider with
account-free market data, suitable quote timestamps, archive provenance and
permitted automated use has passed explicit human review. Its exposure
limits apply only to this account, not to production position sizing or
production RiskGates. Changing a used policy hash requires a separate reviewed
migration.

The options-capture job first commits all observed quotes and decisions.
Then it runs:

    python -m modules.autonomous_alpha paper-account --paper-mode capture

It commits only the account receipt path. Therefore, account failure **cannot
discard already committed quotes**. The first run creates an activation
receipt; it never backfills prior Research decisions. Later runs only book
decisions that occurred after activation and no more than 15 minutes ago.
Settlement uses existing validated outcomes. Neither path contains order
transmission or a broker dependency.

The weekly research job, after its existing alpha-option evaluation, runs:

    python -m modules.autonomous_alpha paper-account --paper-mode weekly

This never opens new trades. It generates
outputs/research/paper_account/account.json with cash, risk, base/adverse
profit, realized return, closed win rate, profit factor,
closed-book drawdown, observed market NAV or explicit INCOMPLETE_MISSING_QUOTES,
and separately marked NAV at cost. Only archived RTH quotes can mark the
account. Missing exits or valuations remain unresolved; they are **not**
converted into verified profits, losses or reconstructed theoretical prices.

After the existing option learner and research improvement steps, the same
weekly job sends a plaintext German digest via existing Gmail secrets:

    python -m modules.autonomous_alpha paper-report

The report includes the account, research strategy comparison, alpha and
option-learner status, data gaps and at most three priorities. Historical
SENT receipts dedupe a repeated weekly invocation. SMTP and Git commit
cannot be an exactly-once distributed transaction: if SMTP succeeds and
persistence fails, a retry may resend the same week (fixed Message-ID and
receipt enable investigation). No credentials are stored in Git.

## Known limitations and release gates

- Hard blocker: the existing Alpha→Options live decision path uses Tradier,
  whose official API requires an active brokerage account and account token.
  This is unacceptable as a *necessary* dependency for a strictly broker-free
  account. The current GitHub-only account therefore cannot automatically
  start until an approved nonbroker option-quote decision/capture feed is added.
  Changing provider configuration alone is insufficient if the underlying
  option_link.decide path still uses Tradier.
- Cboe explicitly prohibits automated downloading of its free delayed quote
  table; do not scrape it. Yahoo/yfinance option quote freshness and provenance
  are not established to the standard needed for verified paper fills.
- Draft PR: not merged or activated; **no live paper fills** and no
  economic alpha evidence from this feature.
- The account starts only at its first post-approval capture, and intentionally
  sacrifices that capture's trade entries to enforce a no-backfill
  boundary. The next fresh capture may open an eligible signal.
- No intra-day P&L is claimed when leg quotes are missing or not sufficiently
  synchronous. Observed market NAV can therefore be null.
- Missing exit quotes remain open/unverified. A separate, approved policy is
  required for handling settlement/expiry using genuinely sourced official
  settlement data; a fabricated mark or zero loss is forbidden.
- Closed-book drawdown is not true mark-to-market peak-to-trough drawdown.
- This account is a TREATMENT-only portfolio, not the benchmark experiment.
  All Alpha→Options treatments, controls and frozen references remain in
  the *existing* research evaluation to avoid survivor and selection bias.
- The 5- and 20-day Option Learner has a different candidate population
  and labeling protocol from the 20-trading-day alpha-option link. Account
  outcomes may not be naively appended as extra training labels; an independent
  registered label-source experiment is needed.
- No new auto-promotion, production tuning, broker access, or paid API.
- Statistical claims require adequate independent forward windows and frozen
  multiple-testing budgets; positive portfolio P&L alone proves neither alpha
  nor an improved learning algorithm.

## Release checklist

1. New tests and full pre-existing tests must pass on the final head.
2. Run repeated synthetic capture/settlement and replay checks; confirm
   prevention of double booking and prevention of real orders.
3. Review the two workflow edits and read-only data dependencies.
4. Confirm sender secrets, quote freshness, recovery procedure and coverage.
5. Obtain manual CODEOWNER approval and merge explicitly. **No auto merge.**
