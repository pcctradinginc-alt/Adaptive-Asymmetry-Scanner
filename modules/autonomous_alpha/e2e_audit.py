"""Broker-free operational audit of the EXISTING alpha-to-options learning loop.

This module is not a predictor, portfolio manager, strategy tester, or second
training ledger. It reconciles source-of-truth journals, verified bid/ask
economics, the GitHub virtual account, frozen forward verdicts and the existing
Research Memory. It never fabricates missing market quotes or profitable labels.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from modules import research_memory as memory
from . import option_link as link, option_economics as economics, paper_account as account
from .registry import _atomic_json, canonical, digest

REPORT = Path("outputs/research/autonomous_alpha/e2e_audit.json")
RECEIPTS = Path("outputs/research/autonomous_alpha/e2e_audit_receipts")
VALIDATION = Path("outputs/research/autonomous_alpha/option_validation.json")
CAPTURE = Path("outputs/options_archive/capture_status")
ACCOUNT_REPORT = account.REPORT
MEMORY = memory.MEMORY
NO_AUTO_ACTION = "Research only; no new API, broker connection, real order, or production influence."


def _json(root: Path, relative: Path) -> dict:
    p = root / relative
    if not p.exists():
        return {}
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected object: " + str(relative))
    return value


def _capture(root: Path, now: datetime, issues: list[dict]) -> dict:
    paths = sorted((root / CAPTURE).glob("????-??-??.json"))
    if not paths:
        issues.append(_issue("CAPTURE_NOT_OBSERVED", "NEED_MORE_DATA", "capture",
                             "No in-window archived capture-status receipt; activation not proven",
                             "Verify the existing options_capture.yml workflow at the next US trading session"))
        return {"status": "NOT_OBSERVED", "latest_session": None}
    rec = _json(root, paths[-1].relative_to(root))
    last = paths[-1].stem
    if rec.get("mode") != "capture" or rec.get("window") != "IN_WINDOW":
        issues.append(_issue("CAPTURE_RECEIPT_INVALID", "TECHNICAL_DEFECT", "capture",
                             "Stored capture receipt is not an in-window capture",
                             "Inspect scripts/rth_marks.py status persistence and Git archive"))
    observed = rec.get("observed_at")
    try:
        when = datetime.fromisoformat(observed)
        if when.tzinfo is None or when > now:
            raise ValueError("future/naive capture timestamp")
    except (TypeError, ValueError):
        issues.append(_issue("CAPTURE_TIMESTAMP_INVALID", "TECHNICAL_DEFECT", "capture",
                             "Missing, naive or future capture timestamp",
                             "Inspect capture provenance and clock"))
    if (rec.get("alpha_link") or {}).get("status") == "ERROR":
        issues.append(_issue("ALPHA_DECISION_CAPTURE_FAILED", "TECHNICAL_DEFECT", "alpha_option_link",
                             str((rec.get("alpha_link") or {}).get("error") or "unknown")[:200],
                             "Fix the existing read-only quote decision capture; do not reconstruct fills"))
    if (rec.get("paper_account") or {}).get("status") == "ERROR":
        issues.append(_issue("PAPER_BOOKING_CAPTURE_FAILED", "TECHNICAL_DEFECT", "paper_account",
                             str((rec.get("paper_account") or {}).get("error") or "unknown")[:200],
                             "Fix virtual ledger sync while retaining the archived market quotes"))
    if rec.get("capture_status") == "PARTIAL_MISSING_QUOTES":
        issues.append(_issue("MISSING_CAPTURE_QUOTES", "NEED_MORE_DATA", "capture",
                             "Some requested contracts had no usable observed RTH quotes",
                             "Monitor the existing Tradier market-data access; never backfill missing quotes"))
    # A delayed GitHub cron or market holiday is not proof of a broken program.
    # Five *weekday* gaps are an investigation trigger, not an automatic failure.
    days = 0
    try:
        d = datetime.fromisoformat(last).date()
        cur = now.date()
        while d < cur and days < 6:
            from datetime import timedelta
            d += timedelta(days=1)
            if d.weekday() < 5:
                days += 1
    except ValueError:
        days = 6
    if days >= 5:
        issues.append(_issue("CAPTURE_MONITOR_STALE", "DEGRADED", "capture",
                             "No recent in-window receipt in at least five weekdays; holidays not inferred",
                             "Inspect schedule delays, Actions runs and read-only Tradier quote entitlements"))
    # Inspect immutable run receipts as well: daily latest-state JSON may have
    # overwritten an earlier failed capture attempt from the same session.
    import hashlib
    failed_attempts = 0
    attempts = sorted((root / CAPTURE / "runs").glob("????-??-??/*.json"))[-100:]
    for path in attempts:
        wrapped = _json(root, path.relative_to(root))
        payload = wrapped.get("payload")
        if not isinstance(payload, dict):
            issues.append(_issue("CAPTURE_ATTEMPT_INTEGRITY", "TECHNICAL_DEFECT", "capture",
                                 "Non-object immutable capture attempt",
                                 "Investigate original Git capture receipt history"))
            break
        checksum = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                  ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
        if wrapped.get("sha256") != checksum or path.stem != checksum:
            issues.append(_issue("CAPTURE_ATTEMPT_INTEGRITY", "TECHNICAL_DEFECT", "capture",
                                 "Capture-attempt content checksum does not match",
                                 "Investigate the append-only Git archive before trusting results"))
            break
        observed_at = _age(payload.get("observed_at"), now)
        if observed_at is not None and 0 <= observed_at <= 168 and (
                (payload.get("alpha_link") or {}).get("status") == "ERROR"
                or (payload.get("paper_account") or {}).get("status") == "ERROR"):
            failed_attempts += 1
    if failed_attempts and (rec.get("alpha_link") or {}).get("status") != "ERROR" and (
            rec.get("paper_account") or {}).get("status") != "ERROR":
        issues.append(_issue("RECOVERED_CAPTURE_ERRORS", "DEGRADED", "capture",
                             str(failed_attempts) + " earlier failed attempts in 7 days; latest receipt recovered",
                             "Check historic capture logs and ensure remediation is durable"))
    return {"status": rec.get("capture_status") or "UNKNOWN",
            "recent_failed_attempts": failed_attempts,
            "latest_session": last, "observed_at": observed,
            "paper_status": (rec.get("paper_account") or {}).get("status"),
            "alpha_link_status": (rec.get("alpha_link") or {}).get("status"),
            "missing_contracts": len(rec.get("missing") or [])}


def _issue(code: str, severity: str, stage: str, fact: str, next_action: str) -> dict:
    return {"id": code, "severity": severity, "stage": stage,
            "observed_fact": fact, "next_action": next_action}


def _age(iso, now: datetime) -> float | None:
    try:
        stamp = datetime.fromisoformat(str(iso))
        if stamp.tzinfo is None:
            return None
        return (now - stamp).total_seconds() / 3600
    except (TypeError, ValueError):
        return None



def _entry_clock_skips(root: Path, skipped: dict, decisions: dict) -> tuple[int, int]:
    """Identify false paper SKIPs caused *only* by a lost subsecond timestamp.

    No decision, quote or historical signed event is modified. Rechecking a
    copy with the exact archived retrieval time is diagnostic, never a fill.
    """
    from modules import option_quotes as oq

    by_signal = {}
    for d in decisions.values():
        by_signal.setdefault(d["signal_id"], []).append(d)
    legacy, current = set(), set()
    for sid, event in skipped.items():
        rejected = set(event.get("reason") or [])
        for d in by_signal.get(sid, []):
            if (d.get("decision") != "TRADE"
                    or d.get("evidence_class_at_entry") != "QUOTE_VERIFIED_PAPER"
                    or d.get("structure", "") + ":ENTRY_QUOTE_NOT_VERIFIED" not in rejected):
                continue
            t = link._ts(d.get("decided_at"))
            legs = d.get("legs") or []
            if t is None or not legs or any(not leg.get("quote_record_id") for leg in legs):
                continue
            from datetime import timedelta
            day = d["decision_day"]
            start = (datetime.fromisoformat(day) - timedelta(days=1)).date().isoformat()
            end = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
            quotes = oq.by_ids([leg["quote_record_id"] for leg in legs], start, end,
                               root=root / oq.ROOT)
            if len(quotes) != len(legs):
                continue
            retrieved = [link._ts(quotes[leg["quote_record_id"]].get("retrieved_at"))
                         for leg in legs]
            if any(x is None for x in retrieved):
                continue
            latest = max(retrieved)
            # A same-second timestamp inversion, with every other quote check
            # passing, is specifically the historic second-truncation bug.
            if not (latest > t and latest.replace(microsecond=0) == t.replace(microsecond=0)):
                continue
            if account._entry_archived(root, {**d, "decided_at": latest.isoformat()}):
                key = "current" if "." in d["decided_at"].split("+")[0] else "legacy"
                (current if key == "current" else legacy).add(sid)
    return len(legacy), len(current)


def _existing_memory(root: Path) -> dict:
    return memory.latest(memory.load(root / MEMORY))


def build(root: Path = Path("."), *, now: datetime | None = None) -> dict:
    """Read-only deterministic evidence audit; no replayed API or model learning."""
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("E2E timestamp must be timezone-aware")
    issues: list[dict] = []
    capture = _capture(root, now, issues)
    cfg = account.policy(root)
    data = link.read_all(root)
    link._check_rule(link.config(root), data["decisions"])
    raw = data["integrity"]
    conflicts = sum(int(v.get("conflicting_duplicates") or 0) for v in raw.values())
    if conflicts:
        issues.append(_issue("ALPHA_JOURNAL_CONFLICT", "TECHNICAL_DEFECT", "option_link",
                             str(conflicts) + " conflicting journal record IDs",
                             "Investigate append-only alpha journals, retain all forensic originals"))
    sig = {s["signal_id"]: s for s in data["signals"]}
    decisions = {d["decision_id"]: d for d in data["decisions"]}
    outcomes = {o["decision_id"]: o for o in data["outcomes"]}
    dangling_decisions = sorted(did for did, d in decisions.items() if d["signal_id"] not in sig)
    dangling_outcomes = sorted(oid for oid, o in outcomes.items() if oid not in decisions)
    if dangling_decisions or dangling_outcomes:
        issues.append(_issue("BROKEN_ALPHA_LINEAGE", "TECHNICAL_DEFECT", "option_link",
                             "Unknown signals/decisions for decision IDs " + str(dangling_decisions[:5])
                             + " and outcome IDs " + str(dangling_outcomes[:5]),
                             "Inspect immutable signal/decision/outcome IDs; do not edit old receipts"))
    # Reuse option_economics._verify via the EXISTING trade evaluator; no
    # independent price calculation and no second training/settlement label.
    rows, counts = economics.trades(root, data, link.config(root))
    verified = {r["decision_id"]: r for r in rows
                if r["status"] == "SETTLED" and r["evidence_class"] == "QUOTE_VERIFIED_PAPER"}
    unverifiable = Counter(r["status"] for r in rows if r["status"] in
                           ("UNVERIFIABLE", "EXIT_MISSING", "EXCLUDED_BY_CORRECTION"))
    # Reconstruct account ONLY from its own signed event receipts; a cached
    # account.json is not proof of activation or a completed booking.
    state = account._replay(account.events(root), cfg)
    if cfg["enabled"] and not state["activated"]:
        issues.append(_issue("PAPER_ACCOUNT_NOT_ACTIVATED", "NEED_MORE_DATA", "paper_account",
                             "No genuine PA-INIT event has been committed",
                             "Check the next regular in-window capture; do not backfill old signals"))
    booked = state["positions"] if state["activated"] else {}
    legacy_skips, current_skips = _entry_clock_skips(root, state["skips"], decisions)
    if legacy_skips:
        issues.append(_issue("PAPER_LEGACY_QUOTE_CLOCK_TRUNCATION", "DEGRADED", "paper_account",
                             str(legacy_skips) + " historical SKIP signals had authentic same-second quotes "
                             "but whole-second decisions; old signed SKIPs remain immutable",
                             "Retain old receipts; use precise timestamps only for future paper entries"))
    if current_skips:
        issues.append(_issue("PAPER_QUOTE_CLOCK_ORDER_DEFECT", "TECHNICAL_DEFECT", "paper_account",
                             str(current_skips) + " new SKIP signals have a verified archived quote "
                             "rejected solely because quote retrieval follows the stored decision time",
                             "Fix the decision/quote timestamp order; never retroactively book trades"))
    for did, p in sorted(booked.items()):
        if did not in decisions:
            issues.append(_issue("BOOKING_SOURCE_MISSING", "TECHNICAL_DEFECT", "paper_account",
                                 "Booked decision is absent from the immutable alpha decision journal",
                                 "Inspect account event and source hashes before attempting correction"))
        elif p["status"] == "CLOSED" and did not in verified:
            issues.append(_issue("BOOKED_CLOSE_NOT_VERIFIABLE", "TECHNICAL_DEFECT", "paper_account",
                                 "A closed account trade has no verified source outcome: " + did,
                                 "Investigate original archived bid/ask quotes and correction journal"))
        elif p["status"] == "OPEN" and did in verified:
            issues.append(_issue("SETTLED_OUTCOME_NOT_BOOKED", "TECHNICAL_DEFECT", "paper_account",
                                 "Verified alpha exit not yet closed in account: " + did,
                                 "Re-run the existing weekly paper-account reconciliation, without new orders"))
    valuation = _json(root, ACCOUNT_REPORT)
    if state["activated"] and not valuation:
        issues.append(_issue("ACCOUNT_WEEKLY_REPORT_NOT_OBSERVED", "DEGRADED", "paper_account",
                             "Verified account events exist but no weekly valuation report is committed",
                             "Inspect existing paper-account weekly step"))
    if valuation:
        # The public account.json is only a summary, never the source of truth.
        # Reconcile cash and position counts with the immutable signed events.
        expected_open = sum(p["status"] == "OPEN" for p in booked.values())
        expected_closed = len(booked) - expected_open
        consistent = ((valuation.get("status") == "ACTIVE") == state["activated"])
        if state["activated"]:
            consistent = (consistent and valuation.get("status") == "ACTIVE"
                          and valuation.get("cash_usd") is not None
                          and abs(float(valuation["cash_usd"]) - float(state["cash"])) <= 0.01
                          and valuation.get("open_trades") == expected_open
                          and valuation.get("closed_trades") == expected_closed
                          and valuation.get("broker_orders") == 0)
        if not consistent:
            issues.append(_issue("ACCOUNT_SUMMARY_DIFFERS_FROM_SIGNED_LEDGER",
                                 "TECHNICAL_DEFECT", "paper_account",
                                 "Cached cash/position report conflicts with Git-only signed booking receipts",
                                 "Regenerate the existing account report from verified events only"))
    validation = _json(root, VALIDATION)
    if not validation:
        issues.append(_issue("VALIDATION_NOT_OBSERVED", "NEED_MORE_DATA", "option_validation",
                             "No option-validation result has been published",
                             "Run existing alpha-option research step; not a separate learning algorithm"))
    latest = _existing_memory(root)
    from . import option_validation as validator
    validation_cfg = validator.config(root)
    plan_proofs = []
    verified_mem = 0
    for plan in validation.get("plans") or []:
        pid = str(plan["plan_id"])
        status = str(plan.get("status") or "")
        cls = plan.get("classification")
        expected = ("FORWARD_VALIDATED" if status == "CONFIRMED"
                    and cls == "QUALIFIED_FOR_HUMAN_REVIEW" else
                    memory.ALPHA_PLAN_STATUS.get(status, "PENDING_FORWARD"))
        proof_pop = [r for r in verified.values()
                     if r["hypothesis_id"] == plan.get("hypothesis_id")
                     and r["structure"] == plan.get("structure")
                     and r["comparison_group"] == "TREATMENT"
                     and str(r["pit_data_date"]) >= str(plan.get("start_date") or "9999-12-31")]
        independent = economics.independent_periods(r["pit_data_date"] for r in proof_pop)
        frozen_id = str((plan.get("binding") or {}).get("report_id") or "")
        # The existing freeze_look_report creates an immutable checksum-verified
        # file. A report_id string by itself is NOT a frozen forward proof.
        look = {}
        if frozen_id and frozen_id.replace("-", "").replace("_", "").isalnum():
            look = _json(root, validator.OUT / validator.LOOK_DIR / (frozen_id + ".json"))
        if look:
            from modules import option_learner as learner
            frozen_sha = look.get("report_sha256")
            look_ok = (look.get("report_id") == frozen_id
                       and look.get("plan_id") == pid
                       and look.get("verdict") == status
                       and isinstance(frozen_sha, str)
                       and frozen_sha == learner._sha({k: v for k, v in look.items()
                                                      if k != "report_sha256"}))
        else:
            look_ok = False
        provenance_ok = (len(proof_pop) >= int(validation_cfg["min_verified_outcomes"])
                         and independent >= int(validation_cfg["min_independent_periods"])
                         and look_ok)
        if expected == "FORWARD_VALIDATED" and not provenance_ok:
            issues.append(_issue("FORWARD_CONFIRMATION_PROVENANCE_MISSING", "TECHNICAL_DEFECT",
                                 "option_validation",
                                 "Plan " + pid + " has " + str(len(proof_pop))
                                 + " quote-verified post-registration treated exits, "
                                 + str(independent) + " independent periods or no frozen look receipt",
                                 "Investigate existing validation look receipt and archived quotes; "
                                 "do not treat cached CONFIRMED as alpha proof"))
        entry = latest.get("alpha_option_validation:" + pid)
        observed = entry.get("status") if entry else None
        if observed == expected:
            verified_mem += 1
        else:
            issues.append(_issue("FORWARD_VERDICT_NOT_IN_RESEARCH_MEMORY",
                                 "TECHNICAL_DEFECT" if expected in ("FORWARD_VALIDATED", "REJECTED") else "DEGRADED",
                                 "research_memory",
                                 "Validation plan " + pid + " says " + expected
                                 + "; current Research Memory says " + str(observed),
                                 "Synchronize the existing Research Memory after alpha-option validation"))
        plan_proofs.append({"plan_id": pid, "hypothesis_id": plan.get("hypothesis_id"),
                            "structure": plan.get("structure"), "forward_verdict": status,
                            "classification": cls, "expected_memory_status": expected,
                            "observed_memory_status": observed, "memory_confirmed": observed == expected,
                            "verified_post_registration_outcomes": len(proof_pop),
                            "independent_quote_periods": independent,
                            "frozen_look_receipt_verified": look_ok,
                            "confirmation_proven": expected == "FORWARD_VALIDATED"
                                                  and observed == expected and provenance_ok})
    if verified and not _json(root, economics.REPORT):
        issues.append(_issue("ECONOMICS_OUTPUT_MISSING", "TECHNICAL_DEFECT", "option_economics",
                             "Source quotes yield settled trades but the economics report is absent",
                             "Repair the existing option-link economics step; do not create a replacement model"))
    if not verified:
        issues.append(_issue("NO_VERIFIED_ALPHA_OPTION_EXITS", "NEED_MORE_DATA", "option_link",
                             "No quote-verified settled alpha-option trades",
                             "Collect genuine future option entry/exit quotes for the frozen 20-day horizon"))
    n_confirmed = sum(x["confirmation_proven"] for x in plan_proofs)
    pending = sum(p["status"] == "OPEN" for p in booked.values())
    tech = sum(i["severity"] == "TECHNICAL_DEFECT" for i in issues)
    return {
        "schema": "autonomous-alpha-e2e-operations-v1", "generated_at": now.isoformat(),
        "status": "TECHNICAL_DEFECT" if tech else
                  "NEED_MORE_DATA" if any(x["severity"] == "NEED_MORE_DATA" for x in issues)
                  else "DEGRADED" if any(x["severity"] == "DEGRADED" for x in issues)
                  else "OPERATIONAL_NOT_ALPHA_PROOF",
        "decision_type": "RESEARCH_ONLY", "production_influence": "NONE",
        "real_orders": False, "broker_connection": "NONE", "no_second_learner": True,
        "capture": capture,
        "source_chain": {"signals": len(sig), "decisions": len(decisions),
                         "outcomes": len(outcomes), "conflicting_records": conflicts,
                         "dangling_decisions": len(dangling_decisions),
                         "dangling_outcomes": len(dangling_outcomes),
                         "verified_settled_economics": len(verified),
                         "pending_economics": counts.get("PENDING", 0),
                         "unverifiable_economics": dict(sorted(unverifiable.items()))},
        "paper_account": {"activation_verified_from_events": state["activated"],
                          "open_bookings": pending,
                          "closed_bookings": len(booked) - pending,
                          "snapshot_status": valuation.get("status") or "NOT_REPORTED",
                          "signed_events_are_source_of_truth": True,
                          "new_live_broker_orders": 0},
        "existing_learning": {"engine": "option_validation -> research_memory -> research_director",
                              "validation_plans": len(plan_proofs),
                              "verdicts_reconciled_to_memory": verified_mem,
                              "forward_confirmed_and_memorized": n_confirmed,
                              "frozen_plan_proofs": plan_proofs[:100],
                              "options_pnl_only_from_existing_economics": True,
                              "paper_account_not_a_training_label": True,
                              "alpha_improvement_proven": n_confirmed > 0,
                              "reason_if_unproven": None if n_confirmed else
                                  "NO_INDEPENDENT_CONFIRMED_POST_COST_FORWARD_ALPHA"},
        "issues": sorted(issues, key=lambda x: (x["severity"] != "TECHNICAL_DEFECT",
                                               x["severity"] != "DEGRADED", x["id"])),
        "n_technical_defects": tech,
        "next_action": (next((x["next_action"] for x in issues if x["severity"] == "TECHNICAL_DEFECT"),
                        None) or next((x["next_action"] for x in issues), NO_AUTO_ACTION)),
        "safety": NO_AUTO_ACTION,
    }


def run(root: Path = Path("."), *, now: datetime | None = None,
        sync_memory: bool = True) -> dict:
    """Use the existing Memory adapter once, then audit the entire data path.

    Never append a second outcome label. Persist observations for Git review
    and flag technical defects via the existing weekly stage-balance check.
    """
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    try:
        # P0: validate ALL source journals and their provenance BEFORE allowing
        # existing Research Memory to ingest any new research interpretation.
        preflight = build(root, now=now)
        untrusted = [i for i in preflight["issues"]
                     if i["severity"] == "TECHNICAL_DEFECT"
                     and i["id"] != "FORWARD_VERDICT_NOT_IN_RESEARCH_MEMORY"]
        if sync_memory and not untrusted:
            sources = {k: root / rel for k, rel in memory.SOURCES.items()}
            memory.sync(path=root / MEMORY, sources=sources, now=now.isoformat())
            result = build(root, now=now)
        else:
            # Never turn an unverified confirmation into a canonical memory win.
            result = preflight
    except Exception as exc:  # noqa: BLE001 – fail CLOSED; persist the error for owner mail/CI
        problem = _issue("E2E_PRECHECK_FAILED", "TECHNICAL_DEFECT", "integrity",
                         type(exc).__name__ + ": " + str(exc)[:240],
                         "Inspect the existing source journals/receipts; do not learn or trade from this run")
        result = {
            "schema": "autonomous-alpha-e2e-operations-v1", "generated_at": now.isoformat(),
            "status": "TECHNICAL_DEFECT", "decision_type": "RESEARCH_ONLY",
            "production_influence": "NONE", "real_orders": False,
            "broker_connection": "NONE", "no_second_learner": True,
            "source_chain": {}, "paper_account": {},
            "existing_learning": {"validation_plans": 0, "verdicts_reconciled_to_memory": 0,
                                  "forward_confirmed_and_memorized": 0,
                                  "alpha_improvement_proven": False},
            "issues": [problem], "n_technical_defects": 1,
            "next_action": problem["next_action"], "safety": NO_AUTO_ACTION,
        }
    _atomic_json(root / REPORT, result)
    day = result["generated_at"][:10]
    receipt_payload = {"schema": "alpha-e2e-observation-receipt-v1",
                       "as_of": day, "report_digest": digest(result),
                       "status": result["status"], "technical_defects": result["n_technical_defects"]}
    dest = root / RECEIPTS / (day + "-" + receipt_payload["report_digest"][:16] + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    value = canonical({"sha256": digest(receipt_payload), "payload": receipt_payload}) + "\n"
    try:
        with dest.open("x", encoding="utf-8") as f:
            f.write(value)
    except FileExistsError:
        if dest.read_text(encoding="utf-8") != value:
            raise ValueError("conflicting immutable E2E audit receipt")
    return result
