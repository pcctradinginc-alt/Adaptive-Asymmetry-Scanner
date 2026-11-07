"""Broker-free virtual account attached to the existing alpha-option journals.

Only this account's immutable booking receipts are new. Signals, decisions,
archived quotes, settlement and research labels remain the single existing
sources of truth (option_link / option_economics / option_validation).
This module never connects to a broker, places an order, or changes a gate.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import option_link as al
from . import option_economics as oe
from .registry import _atomic_json, canonical, digest

CONFIG = Path("config/paper_account.yaml")
JOURNAL = Path("outputs/options_archive/paper_account/events")
REPORT = Path("outputs/research/paper_account/account.json")
VERSION = "github-paper-account-v1"
SCHEMA = "github-paper-booking-v1"
MULTIPLIER = 100


def policy(root: Path = Path(".")) -> dict:
    cfg = yaml.safe_load((Path(root) / CONFIG).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != VERSION:
        raise ValueError("invalid paper-account policy")
    if cfg.get("broker_connection") != "NONE" or cfg.get("real_orders") is not False:
        raise ValueError("paper account must be broker-free")
    if cfg.get("production_influence") != "NONE":
        raise ValueError("paper account must never affect production")
    if cfg.get("comparison_group") != "TREATMENT":
        raise ValueError("paper account may only trade treatment signals")
    sources = cfg.get("approved_quote_sources")
    if not isinstance(sources, list) or any(not isinstance(v, str) for v in sources):
        raise ValueError("approved account-free quote sources must be an explicit list")
    if any("tradier" in source.lower() or not source.strip() for source in sources):
        raise ValueError("broker-required Tradier data cannot activate the paper account")
    if cfg.get("strategy_preference") != list(al.STRUCTURES):
        raise ValueError("paper strategy selection must remain ex-ante")
    if not 1000 <= float(cfg.get("starting_cash_usd", 0)) <= 1_000_000:
        raise ValueError("invalid paper-account starting cash")
    for key in ("max_position_risk_fraction", "max_total_open_risk_fraction",
                "max_ticker_open_risk_fraction"):
        if not 0 < float(cfg.get(key, 0)) <= 0.25:
            raise ValueError("invalid account-only risk limit: " + key)
    if not 30 <= int(cfg.get("max_decision_age_seconds", 0)) <= 900:
        raise ValueError("paper booking must be contemporaneous with its quote")
    return cfg


def _utc(value) -> datetime:
    ts = al._ts(value)
    if ts is None:
        raise ValueError("invalid paper-account timestamp")
    return ts.astimezone(timezone.utc)


def _amount(value) -> float:
    if isinstance(value, bool):
        raise ValueError("invalid monetary amount")
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric monetary amount") from exc
    if not math.isfinite(n):
        raise ValueError("non-finite monetary amount")
    return round(n, 4)


def _event_id(kind: str, key: str) -> str:
    return "PA-" + kind + "-" + digest({"type": kind, "key": key})[:24]


def _event_path(root: Path, event_id: str) -> Path:
    if not event_id.startswith("PA-") or not all(x.isalnum() or x == "-" for x in event_id):
        raise ValueError("invalid event id")
    return Path(root) / JOURNAL / (event_id + ".json")


def _save(root: Path, payload: dict) -> bool:
    if payload.get("schema") != SCHEMA:
        raise ValueError("foreign account event")
    dest = _event_path(root, payload["event_id"])
    wrapped = canonical({"sha256": digest(payload), "payload": payload}) + "\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("x", encoding="utf-8") as stream:
            stream.write(wrapped)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if dest.read_text(encoding="utf-8") != wrapped:
            raise ValueError("conflicting immutable paper event: " + payload["event_id"])
        return False
    return True


def events(root: Path = Path(".")) -> list[dict]:
    records = []
    for path in sorted((Path(root) / JOURNAL).glob("*.json")):
        wrapped = json.loads(path.read_text(encoding="utf-8"))
        p = wrapped["payload"]
        if (wrapped.get("sha256") != digest(p) or p.get("schema") != SCHEMA
                or _event_path(root, p["event_id"]) != path):
            raise ValueError("paper journal integrity violation: " + str(path))
        records.append(p)
    return sorted(records, key=lambda r: (r["occurred_at"], r["event_id"]))


def _replay(records: list[dict], cfg: dict) -> dict:
    if not records:
        return {"activated": False, "positions": {}, "skips": {}, "closed": [],
                "cash": 0.0, "initial": 0.0, "activation": None}
    init = [e for e in records if e["type"] == "INIT"]
    if len(init) != 1 or init[0]["event_id"] != "PA-INIT":
        raise ValueError("paper account needs exactly one immutable activation")
    start = init[0]
    if start["policy_hash"] != digest(cfg):
        raise ValueError("paper policy changed after activation: reviewed migration required")
    if _amount(start["starting_cash_usd"]) != _amount(cfg["starting_cash_usd"]):
        raise ValueError("starting cash mismatch")
    cash = _amount(start["starting_cash_usd"])
    positions, skips, closed = {}, {}, []
    seen_signals = set()
    for event in records:
        kind = event["type"]
        if kind == "INIT":
            continue
        if _utc(event["occurred_at"]) <= _utc(start["occurred_at"]):
            raise ValueError("paper booking predates activation")
        if kind in ("OPEN", "SKIP"):
            sid = event["signal_id"]
            if sid in seen_signals:
                raise ValueError("double account decision for signal " + sid)
            seen_signals.add(sid)
            if kind == "SKIP":
                skips[sid] = event
                continue
            did = event["decision_id"]
            if did in positions:
                raise ValueError("duplicate virtual position")
            cost, risk = _amount(event["entry_cost_usd"]), _amount(event["risk_usd"])
            if cost <= 0 or risk < cost or cost > cash + 0.0001:
                raise ValueError("cash or max-loss violation at paper entry")
            equity = cash + sum(_amount(p["entry_cost_usd"]) for p in positions.values()
                                if p["status"] == "OPEN")
            exposure = sum(_amount(p["risk_usd"]) for p in positions.values() if p["status"] == "OPEN")
            ticker = event["ticker"]
            ticker_risk = sum(_amount(p["risk_usd"]) for p in positions.values()
                              if p["status"] == "OPEN" and p["ticker"] == ticker)
            if risk > equity * float(cfg["max_position_risk_fraction"]) + 0.0001:
                raise ValueError("paper position risk limit breached")
            if exposure + risk > equity * float(cfg["max_total_open_risk_fraction"]) + 0.0001:
                raise ValueError("paper gross risk limit breached")
            if ticker_risk + risk > equity * float(cfg["max_ticker_open_risk_fraction"]) + 0.0001:
                raise ValueError("paper ticker concentration limit breached")
            positions[did] = {**event, "status": "OPEN"}
            cash = _amount(cash - cost)
        elif kind == "CLOSE":
            did = event["decision_id"]
            p = positions.get(did)
            if p is None or p["status"] != "OPEN":
                raise ValueError("orphan or duplicate paper close")
            proceeds = _amount(event["proceeds_usd"])
            pnl = _amount(event["pnl_usd"])
            if abs(proceeds - _amount(p["entry_cost_usd"]) - pnl) > 0.0002:
                raise ValueError("paper P&L is not balanced")
            cash = _amount(cash + proceeds)
            p["status"], p["close_event"] = "CLOSED", event
            closed.append({**p, "pnl_usd": pnl, "adverse_pnl_usd": event["adverse_pnl_usd"]})
        else:
            raise ValueError("unknown paper journal event type")
        if cash < -0.0001:
            raise ValueError("negative cash / virtual leverage forbidden")
    return {"activated": True, "positions": positions, "skips": skips, "closed": closed,
            "cash": cash, "initial": _amount(start["starting_cash_usd"]),
            "activation": start["occurred_at"]}


def _entry_archived(root: Path, d: dict) -> bool:
    """Never book an entry without each exact, usable, archived quote."""
    from modules import option_quotes as oq
    approved = set(policy(root)["approved_quote_sources"])
    if not approved:
        return False  # no broker-free data source has been approved yet
    legs = d.get("legs") or []
    if len(legs) not in (1, 2) or not all(x.get("quote_record_id") for x in legs):
        return False
    day = d["decision_day"]
    start = (datetime.fromisoformat(day) - timedelta(days=1)).date().isoformat()
    end = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    archive = oq.by_ids([x["quote_record_id"] for x in legs], start, end,
                        root=Path(root) / oq.ROOT)
    for leg in legs:
        q = archive.get(leg["quote_record_id"])
        if (q is None or q.get("source") not in approved
                or q.get("occ_symbol") != leg.get("occ_symbol")
                or oq.quote_day(q) != day or not oq.usable_rth_quote(q)):
            return False
        if _utc(q["retrieved_at"]) > _utc(d["decided_at"]):
            return False
        if _amount(q.get("bid")) != _amount(leg.get("bid")):
            return False
        if _amount(q.get("ask")) != _amount(leg.get("ask")):
            return False
        if oq._num(q.get("multiplier")) not in (None, 100.0):
            return False
    return True


def _within_limits(d: dict, state: dict, cfg: dict) -> str | None:
    cost, risk = _amount(d["capital_usd"]), _amount(d["max_loss_usd"])
    if cost <= 0 or risk < cost:
        return "INVALID_DEBIT"
    if cost > state["cash"]:
        return "INSUFFICIENT_CASH"
    positions = [p for p in state["positions"].values() if p["status"] == "OPEN"]
    equity = state["cash"] + sum(_amount(p["entry_cost_usd"]) for p in positions)
    if risk > equity * float(cfg["max_position_risk_fraction"]):
        return "MAX_POSITION_RISK"
    if risk + sum(_amount(p["risk_usd"]) for p in positions) > equity * float(cfg["max_total_open_risk_fraction"]):
        return "MAX_TOTAL_RISK"
    if risk + sum(_amount(p["risk_usd"]) for p in positions if p["ticker"] == d["ticker"]) \
            > equity * float(cfg["max_ticker_open_risk_fraction"]):
        return "MAX_TICKER_RISK"
    return None


def _open(root: Path, now: datetime, data: dict, cfg: dict, current: dict) -> int:
    signals = {s["signal_id"]: s for s in data["signals"]}
    by_signal = defaultdict(list)
    for d in data["decisions"]:
        s = signals.get(d["signal_id"])
        if not s or d.get("excluded_by_correction") or s.get("excluded_by_correction"):
            continue
        if s.get("comparison_group") != cfg["comparison_group"] or not s.get("tradable"):
            continue
        t = _utc(d["decided_at"])
        lag = (now - t).total_seconds()
        if (t <= _utc(current["activation"]) or lag < 0
                or lag > int(cfg["max_decision_age_seconds"])):
            continue
        if _utc(s["registered_at"]) >= t or d.get("rule_hash") != al.rule_hash(al.config(root)):
            continue
        by_signal[d["signal_id"]].append(d)
    added = 0
    for sid, ds in sorted(by_signal.items(), key=lambda x: (min(d["decided_at"] for d in x[1]), x[0])):
        if sid in current["skips"] or any(p["signal_id"] == sid for p in current["positions"].values()):
            continue
        s = signals[sid]
        chosen, reasons = None, []
        for structure in cfg["strategy_preference"]:
            d = next((x for x in ds if x["structure"] == structure), None)
            if d is None or d["decision"] != "TRADE":
                reasons.append(structure + ":NO_TRADE_DECISION")
                continue
            if (d.get("evidence_class_at_entry") != "QUOTE_VERIFIED_PAPER"
                    or not _entry_archived(root, d)):
                reasons.append(structure + ":ENTRY_QUOTE_NOT_VERIFIED")
                continue
            reason = _within_limits(d, current, cfg)
            if reason:
                reasons.append(structure + ":" + reason)
                continue
            chosen = d
            break
        if chosen:
            d = chosen
            event = {"schema": SCHEMA, "type": "OPEN", "event_id": _event_id("OPEN", d["decision_id"]),
                     "occurred_at": now.isoformat(timespec="seconds"),
                     "decision_id": d["decision_id"], "signal_id": sid,
                     "hypothesis_id": d["hypothesis_id"], "hypothesis_version": d["hypothesis_version"],
                     "model_id": s.get("model_id"), "experiment_version": s.get("experiment_version"),
                     "comparison_group": s["comparison_group"], "ticker": d["ticker"],
                     "structure": d["structure"], "decision_day": d["decision_day"],
                     "entry_cost_usd": _amount(d["capital_usd"]), "risk_usd": _amount(d["max_loss_usd"]),
                     "legs": [{"occ_symbol": leg["occ_symbol"], "side": leg["side"],
                               "quote_record_id": leg["quote_record_id"]} for leg in d["legs"]],
                     "exit_fee_per_contract_usd": _amount(al.config(root)["rule"]["commission_usd_per_contract_per_side"]),
                     "broker_order": False, "source": "ALPHA_OPTION_LINK_EX_ANTE"}
        else:
            event = {"schema": SCHEMA, "type": "SKIP", "event_id": _event_id("SKIP", sid),
                     "occurred_at": now.isoformat(timespec="seconds"), "signal_id": sid,
                     "hypothesis_id": s["hypothesis_id"], "hypothesis_version": s["hypothesis_version"],
                     "model_id": s.get("model_id"), "reason": sorted(set(reasons)),
                     "broker_order": False}
        if _save(root, event):
            added += 1
        current = _replay(events(root), cfg)
    return added


def _close(root: Path, data: dict, cfg: dict, current: dict) -> int:
    decisions = {d["decision_id"]: d for d in data["decisions"]}
    outcomes = {o["decision_id"]: o for o in data["outcomes"]}
    written = 0
    for did, p in sorted(current["positions"].items()):
        d, o = decisions.get(did), outcomes.get(did)
        if d is None or d.get("excluded_by_correction"):
            raise ValueError("previously booked decision is missing or corrected")
        if p["status"] == "CLOSED":
            # Existing booked outcomes cannot silently lose verification.
            if o is None or o.get("outcome_id") != p["close_event"]["outcome_id"] \
                    or o.get("excluded_by_correction") or o.get("status") != "EXIT_OBSERVED":
                raise ValueError("previously booked exit was corrected / lost")
            continue
        if o is None or o.get("status") != "EXIT_OBSERVED":
            continue  # EXIT_MISSING never becomes a fictional settlement
        if o.get("excluded_by_correction") or o.get("evidence_class") != "QUOTE_VERIFIED_PAPER":
            continue
        archive = oe._archive(root, [d], [o])
        ids = [leg["quote_record_id"] for leg in d["legs"] + o["exit_legs"]]
        approved = set(cfg["approved_quote_sources"])
        if not approved or any(archive.get(rid, {}).get("source") not in approved for rid in ids):
            continue  # broker-required or unapproved quotes cannot close the account
        if oe._verify(d, o, archive):
            continue
        rule = al.config(root)["rule"]
        base = oe.position_economics(d["legs"], o["exit_legs"], rule, "BASE")
        adverse = oe.position_economics(d["legs"], o["exit_legs"], rule, "ADVERSE")
        if abs(_amount(base["capital_usd"]) - _amount(p["entry_cost_usd"])) > 0.0002:
            raise ValueError("paper entry debit conflicts with quote-verified economics")
        evt = {"schema": SCHEMA, "type": "CLOSE", "event_id": _event_id("CLOSE", did),
               "occurred_at": o["recorded_at"], "decision_id": did,
               "signal_id": p["signal_id"], "outcome_id": o["outcome_id"],
               "exit_day": o["exit_day"], "exit_quote_record_ids": [
                   leg["quote_record_id"] for leg in o["exit_legs"]],
               "proceeds_usd": _amount(base["exit_value_usd"]),
               "pnl_usd": _amount(base["pnl_usd"]),
               "adverse_pnl_usd": _amount(adverse["pnl_usd"]),
               "evidence_class": "QUOTE_VERIFIED_PAPER", "broker_order": False}
        if _save(root, evt):
            written += 1
        current = _replay(events(root), cfg)
    return written


def _mark(root: Path, position: dict, as_of: datetime) -> dict:
    """Observed same-day, near-synchronous leg prices only; no model fallback."""
    from modules import option_quotes as oq
    legs = position["legs"]
    symbols = [x["occ_symbol"] for x in legs]
    history = oq.rth_quote_history(
        symbols, as_of, root=Path(root) / oq.ROOT, lookback_days=4,
        short_symbols=[x["occ_symbol"] for x in legs if x["side"] == "short"],
        long_exit=True)
    available = [{day for day, _ in history.get(sym, [])} for sym in symbols]
    common = set.intersection(*available) if available else set()
    for day in sorted(common, reverse=True):
        records = [next((r for d, r in reversed(history[sym])
                         if d == day and r.get("source") in set(policy(root)["approved_quote_sources"])), None)
                   for sym in symbols]
        if any(r is None for r in records):
            continue
        times = [_utc(r["retrieved_at"]) for r in records]
        if (max(times) - min(times)).total_seconds() > 900:
            continue
        if max(times) < _utc(position["occurred_at"]):
            continue
        value = sum((_amount(r["bid"]) if leg["side"] == "long" else -_amount(r["ask"])) * MULTIPLIER
                    for leg, r in zip(legs, records))
        value -= len(legs) * _amount(position["exit_fee_per_contract_usd"])
        return {"status": "OBSERVED", "day": day, "value_usd": _amount(value),
                "quote_record_ids": [r["record_id"] for r in records]}
    return {"status": "NO_VERIFIED_MARK", "day": None, "value_usd": None}


def report(root: Path, cfg: dict, current: dict, now: datetime) -> dict:
    if not current["activated"]:
        return {"schema": "github-paper-account-report-v1", "status": "NOT_ACTIVATED",
                "starting_cash_usd": float(cfg["starting_cash_usd"]), "broker_orders": 0}
    opened = [p for p in current["positions"].values() if p["status"] == "OPEN"]
    closed = current["closed"]
    realized = _amount(sum(p["pnl_usd"] for p in closed))
    cost_basis = _amount(sum(p["entry_cost_usd"] for p in opened))
    marks = {p["decision_id"]: _mark(root, p, now) for p in opened}
    marks_complete = all(m["status"] == "OBSERVED" for m in marks.values())
    market_value = _amount(sum(m["value_usd"] for m in marks.values())) if marks_complete else None
    pnl_values = [p["pnl_usd"] for p in closed]
    wins, losses = [p for p in pnl_values if p > 0], [p for p in pnl_values if p < 0]
    profit_factor = round(sum(wins) / -sum(losses), 4) if losses else (None if not wins else "NO_LOSSES")
    equity = current["initial"]
    peak = equity
    drawdown = 0.0
    for trade in sorted(closed, key=lambda p: p["close_event"]["occurred_at"]):
        equity += trade["pnl_usd"]
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    by_structure = defaultdict(list)
    for trade in closed:
        by_structure[trade["structure"]].append(trade["pnl_usd"])
    nav_cost = _amount(current["cash"] + cost_basis)
    nav_market = _amount(current["cash"] + market_value) if market_value is not None else None
    return {
        "schema": "github-paper-account-report-v1", "status": "ACTIVE",
        "generated_at": now.isoformat(timespec="seconds"),
        "activated_at": current["activation"], "starting_cash_usd": current["initial"],
        "cash_usd": current["cash"],
        "open_risk_usd": _amount(sum(p["risk_usd"] for p in opened)),
        "open_cost_basis_usd": cost_basis, "open_market_value_usd": market_value,
        "nav_at_cost_usd": nav_cost, "nav_market_usd": nav_market,
        "nav_market_status": "VERIFIED_MARKS" if marks_complete else "INCOMPLETE_MISSING_QUOTES",
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": _amount(market_value - cost_basis) if market_value is not None else None,
        "return_on_starting_capital_realized": round(realized / current["initial"], 6),
        "return_on_starting_capital_at_cost": round((nav_cost / current["initial"]) - 1, 6),
        "open_trades": len(opened), "closed_trades": len(closed), "account_skips": len(current["skips"]),
        "winrate_closed": round(len(wins) / len(closed), 6) if closed else None,
        "profit_factor_closed": profit_factor, "max_drawdown_closed_book": round(drawdown, 6),
        "net_pnl_base_usd": realized,
        "net_pnl_adverse_usd": _amount(sum(p["adverse_pnl_usd"] for p in closed)),
        "trades_by_structure": {k: {"count": len(v), "net_pnl_usd": _amount(sum(v))}
                                for k, v in sorted(by_structure.items())},
        "marks": marks,
        "research_source": "EXISTING_ALPHA_OPTION_LINK_AND_OPTION_VALIDATION",
        "verified_outcome_count": len(closed),
        "statistical_alpha_status": "NOT_ESTABLISHED_BY_ACCOUNT_PERFORMANCE",
        "broker_orders": 0, "broker_connection": "NONE",
        "production_influence": "NONE",
    }


def sync(root: Path = Path("."), *, now: datetime | None = None,
         capture: bool = False, persist_report: bool = True) -> dict:
    """Capture: open only from fresh registered decisions, settle any due exits.
    Weekly: settle and report without booking older decisions as new trades.
    Both workflows reconstruct every balance from immutable Git-tracked events.
    """
    root = Path(root)
    cfg = policy(root)
    now = _utc(now or datetime.now(timezone.utc))
    current = _replay(events(root), cfg)
    if not cfg["enabled"]:
        return {"status": "DISABLED", "broker_orders": 0}
    if not current["activated"]:
        if not capture:
            return report(root, cfg, current, now)
        _save(root, {"schema": SCHEMA, "type": "INIT", "event_id": "PA-INIT",
                     "occurred_at": now.isoformat(timespec="seconds"),
                     "policy_hash": digest(cfg), "starting_cash_usd": _amount(cfg["starting_cash_usd"]),
                     "broker_order": False})
        # Deliberately do NOT backfill decisions from before this first live capture.
        out = report(root, cfg, _replay(events(root), cfg), now)
        out["new_events"] = 1
        out["note"] = "ACTIVATED_NO_HISTORICAL_BACKFILL"
        return out
    data = al.read_all(root)
    if any(v.get("conflicting_duplicates") for v in data.get("integrity", {}).values()):
        raise ValueError("alpha-option journal conflict; paper trading halted")
    al._check_rule(al.config(root), data["decisions"])
    opened = _open(root, now, data, cfg, current) if capture else 0
    current = _replay(events(root), cfg)
    closed = _close(root, data, cfg, current)
    out = report(root, cfg, _replay(events(root), cfg), now)
    out["new_events"] = opened + closed
    out["new_open_or_skip"] = opened
    out["new_close"] = closed
    if persist_report:
        _atomic_json(root / REPORT, out)
    return out
