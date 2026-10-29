"""Research-only ROI learning: bounded rejected-contract shadow cohorts.

Use the options that OptionsDesigner already evaluated and rejected.
Never lower production thresholds, create an order, call an LLM, invent a
quote, or convert stock returns into option P&L. Entry and marks require
independent, archived RTH bid/ask quotes *after* registration.

Schedule: scanner registers at most 5 distinct rejects per NY day;
options-capture fetches the already-registered legs at entry and 5/20/45
calendar-day marks; feedback appends strictly quote-verifiable outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from modules import option_quotes as oq

CONFIG = Path("config/roi_learning_shadow.yaml")
ROOT = Path("outputs/research/roi_learning")
COHORTS = ROOT / "cohorts"
MARKS = ROOT / "marks"
SUMMARY = ROOT / "summary.json"
MAX_DAY_HARD = 5
MAX_CAPTURE_HARD = 30
RESEARCH_THRESHOLDS = (0.0, 0.05, 0.07)
HORIZONS = (5, 20, 45)


class ResearchIntegrityError(ValueError):
    pass


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(x):
    return hashlib.sha256(_canonical(x).encode("utf-8")).hexdigest()


def _config(root: Path = Path(".")) -> dict:
    p = Path(root) / CONFIG
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("version") != "roi-learning-v1":
        raise ResearchIntegrityError("invalid ROI research configuration")
    if obj.get("production_influence") != "NONE" or obj.get("allow_llm") is not False:
        raise ResearchIntegrityError("production/LLM access forbidden")
    if obj.get("roi_research_thresholds") != list(RESEARCH_THRESHOLDS) or obj.get("horizons_calendar_days") != list(HORIZONS):
        raise ResearchIntegrityError("non-preregistered research thresholds or horizons")
    if not 1 <= int(obj.get("max_new_per_day", 0)) <= MAX_DAY_HARD:
        raise ResearchIntegrityError("daily research budget exceeds hard cap")
    if not 1 <= int(obj.get("max_quote_symbols_per_capture", 0)) <= MAX_CAPTURE_HARD:
        raise ResearchIntegrityError("capture exceeds research hard cap")
    if not 1 <= int(obj.get("max_entry_lag_days", 0)) <= 5:
        raise ResearchIntegrityError("entry freshness cannot be relaxed")
    if not 1 <= int(obj.get("max_exit_lag_days", 0)) <= 8:
        raise ResearchIntegrityError("mark freshness cannot be relaxed")
    if not 0 <= float(obj.get("commission_usd_per_leg_per_side", -1)) <= 3:
        raise ResearchIntegrityError("invalid research fee parameter")
    return obj


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _journal_read(directory: Path, *, max_rows=10000) -> list[dict]:
    seen_ids, out = {}, []
    if not directory.exists():
        return []
    for path in sorted(directory.glob("????-??.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                if len(out) >= max_rows:
                    raise ResearchIntegrityError("research journal record budget exceeded")
                try:
                    event = json.loads(line)
                    rec = event["record"]
                    event_id = event["event_id"]
                    if not isinstance(rec, dict) or event_id != _hash(rec):
                        raise ValueError("content hash mismatch")
                    logical = rec["cohort_id"] if directory.name == "cohorts" else rec["cohort_id"] + ":" + str(rec["horizon_days"])
                    if logical in seen_ids:
                        if seen_ids[logical] != event_id:
                            raise ValueError("conflicting frozen record")
                        continue  # Git union merge can duplicate identical rows.
                    seen_ids[logical] = event_id
                    out.append(rec)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ResearchIntegrityError(f"corrupt journal {path}:{lineno}: {exc}") from exc
    return out


def _journal_append(directory: Path, rows: list[dict], *, known: list[dict]) -> int:
    seen = {x["cohort_id"] if directory.name == "cohorts" else x["cohort_id"] + ":" + str(x["horizon_days"]) for x in known}
    to_write = []
    for r in rows:
        key = r["cohort_id"] if directory.name == "cohorts" else r["cohort_id"] + ":" + str(r["horizon_days"])
        if key in seen:
            continue
        seen.add(key)
        to_write.append(r)
    for r in to_write:
        d = r["signal_date"] if directory.name == "cohorts" else r["marked_at"][:10]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (d[:7] + ".jsonl")).open("a", encoding="utf-8") as fh:
            fh.write(_canonical({"event_id": _hash(r), "record": r}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return len(to_write)


def _symbols(row: dict) -> tuple[list[dict], str | None]:
    opt = row.get("option") or {}
    ticker = str(row.get("ticker") or "").strip().upper()
    right = "put" if "PUT" in str(row.get("strategy") or "").upper() else "call"
    is_spread = "SPREAD" in str(row.get("strategy") or "").upper()
    if not ticker or not isinstance(opt, dict) or not opt:
        return [], "no_option_contract"
    if is_spread and not isinstance(opt.get("spread_leg"), dict):
        return [], "spread_leg_missing"
    legs = [("long", opt)]
    if is_spread:
        legs.append(("short", opt["spread_leg"]))
    output = []
    for side, leg in legs:
        expiry = leg.get("expiry") or opt.get("expiry")
        sym = leg.get("option_symbol") or leg.get("symbol")
        if not sym:
            try:
                sym = oq.occ_symbol(ticker, expiry, right, float(leg.get("strike")))
            except (TypeError, ValueError):
                return [], "no_occ_identity"
        sym = str(sym).strip().upper().replace(" ", "")
        if not oq.option_root(sym):
            return [], "invalid_occ_identity"
        output.append({"side": side, "occ_symbol": sym})
    if len(set(x["occ_symbol"] for x in output)) != len(output):
        return [], "duplicate_legs"
    return output, None


def _rank(rejects: list[dict], existing: set[str], signal_day: str, max_new: int) -> list[dict]:
    # Stratification by actual failing gate; avoid selecting 5 near-identical
    # "barely failed" cases and misrepresenting the rejection population.
    groups = defaultdict(list)
    for row in rejects:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in existing:
            continue
        legs, err = _symbols(row)
        if err:
            continue
        failed = sorted({v for v in (row.get("fail_gates") or {}).values() if v})
        group = failed[0] if failed else "unknown"
        key = _hash({"date": signal_day, "ticker": ticker, "strategy": row.get("strategy"),
                     "legs": legs})
        groups[group].append((key, row, legs))
    for g in groups.values():
        g.sort(key=lambda x: x[0])
    keys = sorted(groups)
    out, chosen_tickers = [], set()
    while len(out) < max_new and keys:
        for k in list(keys):
            if not groups[k]:
                keys.remove(k)
                continue
            _, row, legs = groups[k].pop(0)
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker in chosen_tickers:
                continue  # same ticker + multiple strategy candidates is 1 daily cohort
            chosen_tickers.add(ticker)
            out.append((row, legs))
            if len(out) >= max_new:
                break
    return out


def register(rejects: list[dict], signal_day: str, *,
             now: datetime | None = None, root: Path = Path(".")) -> dict:
    """Add research shadow contracts, never modify OptionsDesigner or trades."""
    root = Path(root)
    cfg = _config(root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ResearchIntegrityError("decision timestamp must be timezone-aware")
    if date.fromisoformat(signal_day) != now.date():
        # Scanner date usually NY date; allow NY day across UTC midnight only
        from modules import market_snapshot as ms
        if date.fromisoformat(signal_day) != now.astimezone(ms.NY_TZ).date():
            raise ResearchIntegrityError("cannot backdate research registration")
    old = _journal_read(root / COHORTS)
    day_old = [r for r in old if r["signal_date"] == signal_day]
    slots = max(0, int(cfg["max_new_per_day"]) - len(day_old))
    result = {"mode": "research_shadow_only", "input_rejects": len(rejects),
              "already_registered_today": len(day_old), "registered": 0,
              "production_impact": False, "llm_calls": 0}
    if not cfg.get("enabled") or slots <= 0:
        return result
    seen_tickers = {r["ticker"] for r in day_old}
    chosen = _rank(rejects, seen_tickers, signal_day, slots)
    rows = []
    for source, legs in chosen:
        ticker = str(source["ticker"]).strip().upper()
        roi = _num(source.get("roi_net"))
        hurdle = _num(source.get("roi_hurdle"))
        failures = dict(source.get("fail_gates") or {})
        cohort_id = _hash({"date": signal_day, "ticker": ticker, "strategy": source.get("strategy"),
                           "legs": legs})[:24]
        rows.append({
            "schema": "roi-learning-cohort-v1",
            "cohort_id": cohort_id, "signal_date": signal_day,
            "decision_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "ticker": ticker, "strategy": source.get("strategy"),
            "legs": legs,
            "original_roi_net": roi, "original_production_hurdle": hurdle,
            "roi_only_ge": {f"{x:.2f}": roi >= x if roi is not None else None for x in RESEARCH_THRESHOLDS},
            "other_production_gates": failures, "selection": "deterministic_stratified_gate_failures",
            "evidence": "REJECTED_ONLY_NOT_TRADABLE",
            "production_eligible": False,
            "commission_usd_per_leg_per_side": float(cfg["commission_usd_per_leg_per_side"]),
        })
    result["registered"] = _journal_append(root / COHORTS, rows, known=old)
    result["cohort_ids"] = [r["cohort_id"] for r in rows]
    return result


def _time(value):
    return oq._parse_ts(value)


def _quote_pairs(legs: list[dict], quotes_by_symbol: dict[str, list[dict]],
                 *, after: datetime, before: datetime):
    """Earliest actual executable bid/ask pair at/beyond decision/mark time."""
    left = []
    for leg in legs:
        vals = [q for q in quotes_by_symbol.get(leg["occ_symbol"], [])
                if oq.usable_rth_quote(q) and (ts := _time(q.get("retrieved_at")))
                and after <= ts <= before]
        vals.sort(key=lambda q: str(q.get("retrieved_at")))
        if not vals:
            return None
        left.append(vals)
    if len(left) == 1:
        q = left[0][0]
        return [q], _time(q["retrieved_at"])
    for a in left[0]:
        ta = _time(a["retrieved_at"])
        for b in left[1]:
            tb = _time(b["retrieved_at"])
            if abs((tb-ta).total_seconds()) <= 900 and oq.quote_day(a) == oq.quote_day(b):
                return [a, b], max(ta,tb)
    return None


def _debit(entry: list[dict]) -> float | None:
    value = _num(entry[0].get("ask"))
    if len(entry) > 1 and value is not None:
        value -= _num(entry[1].get("bid")) or 0
    return value if value is not None and value > 0 else None


def _credit(exit: list[dict]) -> float | None:
    value = _num(exit[0].get("bid"))
    if len(exit) > 1 and value is not None:
        ask = _num(exit[1].get("ask"))
        value = value - ask if ask is not None else None
    return value


def _quotes_since(root: Path, cohorts: list[dict], *, today: date) -> dict[str, list[dict]]:
    if not cohorts:
        return {}
    start = min(r["signal_date"] for r in cohorts)
    rows = oq.read(start, today.isoformat(), root=root / "outputs/options_archive/quotes")
    want = {leg["occ_symbol"] for r in cohorts for leg in r["legs"]}
    result = defaultdict(list)
    for q in rows:
        sym = q.get("occ_symbol")
        if sym in want and (ts := _time(q.get("retrieved_at"))) and ts.date() <= today:
            result[sym].append(q)
    for q in result.values():
        q.sort(key=lambda x: str(x.get("retrieved_at")))
    return result


def due_capture_symbols(*, today: date | None = None, root: Path = Path(".")) -> list[str]:
    """For options-capture only: bounded extra symbols, no expensive LLM/market scan."""
    root = Path(root)
    cfg = _config(root)
    if not cfg.get("enabled"):
        return []
    today = today or datetime.now(timezone.utc).date()
    cohorts = _journal_read(root / COHORTS)
    marks = _journal_read(root / MARKS)
    done = {(x["cohort_id"], x["horizon_days"]) for x in marks}
    wanted = []
    for c in sorted(cohorts, key=lambda x: (x["signal_date"], x["cohort_id"])):
        d = date.fromisoformat(c["signal_date"])
        elapsed = (today-d).days
        horizons = (0, *HORIZONS)
        for h in horizons:
            max_lag = int(cfg["max_entry_lag_days"] if h == 0 else cfg["max_exit_lag_days"])
            if (h == 0 or (c["cohort_id"], h) not in done) and h <= elapsed <= h + max_lag:
                wanted.extend(x["occ_symbol"] for x in c["legs"])
                break
    return list(dict.fromkeys(wanted))[:int(cfg["max_quote_symbols_per_capture"])]





def _candidate_underlying_by_gate(root: Path, today: date, lookback_days: int = 100) -> dict:
    """Use existing PIT stock outcomes. Never present them as option P&L."""
    cutoff = today - timedelta(days=lookback_days)
    seen, buckets = set(), {}
    unreadable = rows_seen = 0
    for path in sorted((root / "outputs/candidate_ledger").glob("????-??.jsonl")):
        try:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        day = date.fromisoformat(row["date"])
                        if not cutoff <= day <= today:
                            continue
                        identity = row.get("signal_id") or (
                            row["date"], row.get("ticker"), row.get("event_id")
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        rows_seen += 1
                        reason = ("proposed" if row.get("status") == "proposed"
                                  else str(row.get("reject_reason") or "unlabeled"))
                        bucket = buckets.setdefault(reason, {
                            "registered": 0, "direction_known": 0,
                            "returns": {5: [], 20: [], 45: []},
                        })
                        bucket["registered"] += 1
                        bucket["direction_known"] += int(
                            row.get("direction") in ("BULLISH", "BEARISH")
                        )
                        outcomes = row.get("outcomes") or {}
                        if (outcomes.get("price_basis") or {}).get("status") == "CHANGED":
                            continue
                        for h in (5, 20, 45):
                            ret = outcomes.get(f"ret_{h}d")
                            if (isinstance(ret, (int, float)) and
                                    not isinstance(ret, bool) and math.isfinite(ret)):
                                bucket["returns"][h].append((row["date"], float(ret)))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        unreadable += 1
        except OSError:
            unreadable += 1
    result = {}
    for reason, bucket in sorted(buckets.items()):
        horizons = {}
        for h, values in bucket["returns"].items():
            days = len({day for day, _ in values})
            enough = len(values) >= 20 and days >= 10
            horizons[str(h)] = {
                "observed": len(values), "independent_days": days,
                "mean_stock_return": (
                    round(sum(v for _, v in values) / len(values), 6)
                    if enough else None
                ),
                "status": "DESCRIPTIVE_ONLY" if enough else "NEED_MORE_DATA",
            }
        result[reason] = {
            "registered": bucket["registered"],
            "direction_known": bucket["direction_known"],
            "underlying": horizons,
        }
    return {
        "lookback_days": lookback_days, "signals_read": rows_seen,
        "unreadable_rows": unreadable, "by_reject_reason": result,
        "stocks_are_not_option_returns": True,
        "direction_mixed_or_unknown": True,
        "interpretation": (
            "Direction unknown for early deferrals, so their stock returns are "
            "not comparable to long-call option outcomes. No causal inference."
        ),
    }


def _economics_acquisition_report(root: Path, today: date, limit: int = 10) -> dict:
    """Read-only funnel: acquisition deferrals are NOT economic rejections."""
    rows = []
    daily_dir = root / "outputs/daily_reports"
    for path in sorted(daily_dir.glob("????-??-??.json"), reverse=True):
        try:
            day = date.fromisoformat(path.stem)
            if day > today:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            stats = data.get("stats") or {}
            ef = stats.get("economics_first") or {}
            if ef.get("mode") not in ("active", "shadow"):
                continue
            rejects = data.get("rejects") or {}

            def count(reason: str) -> int:
                return int((rejects.get(reason) or {}).get("count", 0))

            rows.append({
                "date": day.isoformat(), "mode": ef["mode"],
                "sector_survivors": int(stats.get("sector_ok") or 0),
                "quote_probes": int(ef.get("probed") or 0),
                "budget_not_probed": count("economics_first_quote_probe_budget"),
                "earnings_rejected": count("economics_first_earnings_gate"),
                "haiku_inputs": int(ef.get("haiku") or 0),
                "haiku_audit_inputs": int(ef.get("audit_haiku") or 0),
                "sonnet_analyzed": int(stats.get("analyzed") or 0),
                "mismatch_rejected": count("mismatch_below_min"),
                "quick_mc_survivors": int(stats.get("quick_mc") or 0),
                "roi_approved": int(stats.get("roi_ok") or 0),
                "trades": int(stats.get("trades") or 0),
            })
            if len(rows) == limit:
                break
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    keys = ("sector_survivors", "quote_probes", "budget_not_probed",
            "earnings_rejected", "haiku_inputs", "haiku_audit_inputs",
            "sonnet_analyzed", "mismatch_rejected", "quick_mc_survivors",
            "roi_approved", "trades")
    totals = {key: sum(row[key] for row in rows) for key in keys}
    return {
        "status": "OBSERVED_DESCRIPTIVE_ONLY" if rows else "NO_ECONOMICS_FIRST_DATA",
        "days": rows, "totals": totals,
        "budget_not_probed_is_unprofitable": False,
        "options_outcomes_for_unprobed_candidates": "UNKNOWN",
        "interpretation": "Quote-probe budget deferrals are unpriced, not economic rejects.",
    }


def _observed_option_economics(cohorts: list[dict], marks: list[dict],
                               today: date, cfg: dict) -> dict:
    """Observed RTH marks only; overlapping reject gates and ROI buckets remain distinct."""
    by_mark = {(m["cohort_id"], int(m["horizon_days"])): m for m in marks}
    entry_lag = int(cfg["max_entry_lag_days"])
    exit_lag = int(cfg["max_exit_lag_days"])

    def metric(rows: list[dict], horizon: int) -> dict:
        """Fully matured denominator includes missing quotes; avoids survivor-only coverage."""
        matured = [
            c for c in rows
            if (today - date.fromisoformat(c["signal_date"])).days >= horizon + entry_lag + exit_lag
        ]
        observed = [
            (c, by_mark[(c["cohort_id"], horizon)])
            for c in rows if (c["cohort_id"], horizon) in by_mark
        ]
        mature_observed = sum(
            (c["cohort_id"], horizon) in by_mark for c in matured
        )
        usable = [
            (c, float(m["net_return"])) for c, m in observed
            if isinstance(m.get("net_return"), (int, float))
            and math.isfinite(float(m["net_return"]))
            and m.get("price_evidence") == "OBSERVED_RTH_EXECUTABLE_BID_ASK"
        ]
        independent_days = len({c["signal_date"] for c, _ in usable})
        sufficient = len(usable) >= 20 and independent_days >= 10
        return {
            "registered": len(rows), "marked": len(observed),
            "fully_matured": len(matured), "fully_matured_marked": mature_observed,
            "matured_quote_coverage": (
                round(mature_observed / len(matured), 4) if matured else None
            ),
            "observed_valid_quote_outcomes": len(usable),
            "independent_signal_days": independent_days,
            "mean_net": (
                round(sum(v for _, v in usable) / len(usable), 6)
                if sufficient else None
            ),
            "median_net": (
                round(__import__("statistics").median(v for _, v in usable), 6)
                if sufficient else None
            ),
            "win_rate": (
                round(sum(v > 0 for _, v in usable) / len(usable), 6)
                if sufficient else None
            ),
            "status": "DESCRIPTIVE_ONLY" if sufficient else "NEED_MORE_MATURED_QUOTES",
        }

    horizons = {str(h): metric(cohorts, h) for h in HORIZONS}
    gates = sorted({
        str(reason)
        for c in cohorts for reason in (c.get("other_production_gates") or {}).values()
        if reason
    })
    by_rejection_gate = {
        reason: {
            str(h): metric([
                c for c in cohorts
                if reason in {
                    str(g) for g in (c.get("other_production_gates") or {}).values()
                    if g
                }
            ], h)
            for h in HORIZONS
        }
        for reason in gates
    }
    return {
        "horizons": horizons, "rejection_gates": by_rejection_gate,
        "gate_groups_overlap": True,
        "horizon_unit": "calendar_days",
        "maturity_rule": "signal_date + horizon + max_entry_lag + max_exit_lag",
        "outcomes_only_observed_quotes": True,
        "fixed_horizon_marks_not_real_stop_take_profit": True,
        "minimum_descriptive_evidence": {"outcomes": 20, "independent_signal_days": 10},
        "statistical_promotion_allowed": False,
    }


def _gate_economics_markdown(report: dict) -> str:
    """Compact diagnostics, not a trading decision or a promotion recommendation."""
    acquisition = report["economics_first_acquisition"]
    economics = report["observed_option_economics"]
    lines = [
        f"# Gate- und Optionsökonomie – {report['as_of']}",
        "",
        "**READ-ONLY SHADOW – keine Handels- oder Gate-Freigabe.**",
        "",
        "## Economics-first: letzte verfügbare Läufe",
        "",
        "| Datum | Sektor | Quote-Probes | Budget-untersucht: nein | Earnings-Reject | Haiku | Sonnet | Quick-MC | ROI OK | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in acquisition["days"]:
        values = [row[k] for k in (
            "date", "sector_survivors", "quote_probes", "budget_not_probed",
            "earnings_rejected", "haiku_inputs", "sonnet_analyzed",
            "quick_mc_survivors", "roi_approved", "trades"
        )]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += [
        "",
        "Budget-bedingt nicht untersuchte Kandidaten sind **keine** nachgewiesenen",
        "ökonomischen Rejects. Für unbepreiste Aktien existieren keine realen Options-Outcomes.",
        "",
        "## ROI-Reject-Shadow: echte RTH-Bid-/Ask-Marks",
        "",
        "| Horizont (Kalendertage) | Registriert | Markiert | Voll ausgereift | Davon markiert | Quote-Abdeckung | Ø Netto* | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for horizon, values in economics["horizons"].items():
        pct = ("n/v" if values["matured_quote_coverage"] is None
               else f'{values["matured_quote_coverage"]:.1%}')
        mean = ("n/v" if values["mean_net"] is None else f'{values["mean_net"]:.1%}')
        lines.append(
            f'| {horizon} | {values["registered"]} | {values["marked"]} | '
            f'{values["fully_matured"]} | {values["fully_matured_marked"]} | '
            f'{pct} | {mean} | {values["status"]} |'
        )
    lines += [
        "",
        "*Netto-Mittelwerte nur ab 20 verwertbaren Marks an mindestens 10 unabhängigen",
        "Signaltagen. Rein deskriptiv, kein Alpha-Nachweis.",
        "Voll ausgereift: Signalzeitpunkt plus Horizont, maximal zulässiger",
        "Entry-Lag und maximal zulässiger Exit-Lag.",
        "",
        "## Reject-Gründe (Gruppen können sich überlappen)",
        "",
        "| Gate | Registrierte Kontrakte | 5d markiert | 20d markiert | 45d markiert |",
        "|---|---:|---:|---:|---:|",
    ]
    for gate, metrics in economics["rejection_gates"].items():
        lines.append(
            f'| {gate} | {metrics["5"]["registered"]} | '
            f'{metrics["5"]["marked"]} | {metrics["20"]["marked"]} | '
            f'{metrics["45"]["marked"]} |'
        )
    lines += [
        "",
        "Fest definierte Marks sind nicht mit Take-Profit-/Stop-Loss-Exits",
        "gleichzusetzen. Unbeobachtete Quotes bleiben UNKNOWN.",
        "Es erfolgt keine Änderung an Champion, Gates, Sizing, PPO oder Promotion.",
        "",
    ]
    stock = report["candidate_underlying_by_gate"]
    lines += [
        "",
        "## Kandidaten-Forward-Renditen (Aktien, nicht Optionen)",
        "",
        "| Ablehnungsgrund | Signale | Richtung bekannt | Aktienmark 5d | Ø 5d* | Aktienmark 20d | Ø 20d* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(stock["by_reject_reason"].items(),
                    key=lambda item: (-item[1]["registered"], item[0]))
    for reason, bucket in ranked[:15]:
        m5, m20 = bucket["underlying"]["5"], bucket["underlying"]["20"]
        avg5 = "n/v" if m5["mean_stock_return"] is None else f'{m5["mean_stock_return"]:.1%}'
        avg20 = "n/v" if m20["mean_stock_return"] is None else f'{m20["mean_stock_return"]:.1%}'
        lines.append(
            f'| {reason} | {bucket["registered"]} | {bucket["direction_known"]} | '
            f'{m5["observed"]} | {avg5} | {m20["observed"]} | {avg20} |'
        )
    lines += [
        "",
        "*Mittelwerte erst ab 20 Marks an 10 unabhängigen Signaltagen; nicht kausal.",
        "Richtungsunbekannte Budget-Signale sind keine Long-Call-Prognosen.",
        "Aktienrenditen dürfen nicht als verpasste Optionsgewinne gelten.",
    ]
    return "\n".join(lines)

def evaluate(*, today: date | None = None, root: Path = Path(".")) -> dict:
    """Mature observed marks, no forward return or training input is fabricated."""
    root = Path(root)
    cfg = _config(root)
    today = today or datetime.now(timezone.utc).date()
    cohorts = _journal_read(root / COHORTS)
    old = _journal_read(root / MARKS)
    done = {(x["cohort_id"], x["horizon_days"]) for x in old}
    active = [c for c in cohorts if 0 <= (today - date.fromisoformat(c["signal_date"])).days <= max(HORIZONS) + int(cfg["max_exit_lag_days"])]
    by_symbol = _quotes_since(root, active, today=today)
    new, unknown = [], Counter()
    for c in active:
        decision = _time(c["decision_at"])
        if decision is None or decision.date() > today:
            raise ResearchIntegrityError("frozen decision timestamp invalid")
        entry = _quote_pairs(c["legs"], by_symbol,
                             after=decision,
                             before=min(datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc),
                                        decision + timedelta(days=int(cfg["max_entry_lag_days"]))))
        if not entry:
            unknown["entry_quote_missing"] += 1
            continue
        entry_legs, entry_at = entry
        debit = _debit(entry_legs)
        if debit is None:
            unknown["entry_not_positive_debit"] += 1
            continue
        for horizon in HORIZONS:
            if (c["cohort_id"], horizon) in done or (today - entry_at.date()).days < horizon:
                continue
            mark_start = datetime.combine(entry_at.date() + timedelta(days=horizon),
                                          datetime.min.time(), tzinfo=timezone.utc)
            end = min(datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc),
                      mark_start + timedelta(days=int(cfg["max_exit_lag_days"])))
            mark = _quote_pairs(c["legs"], by_symbol, after=mark_start, before=end)
            if not mark:
                unknown[f"mark_{horizon}_quote_missing"] += 1
                continue
            exit_legs, mark_at = mark
            credit = _credit(exit_legs)
            if credit is None:
                unknown[f"mark_{horizon}_nonexecutable"] += 1
                continue
            fees = (2 * len(c["legs"]) * float(c["commission_usd_per_leg_per_side"])) / 100.0
            net_return = (credit - debit - fees) / debit
            new.append({
                "schema": "roi-learning-mark-v1",
                "cohort_id": c["cohort_id"], "horizon_days": horizon,
                "marked_at": mark_at.isoformat(timespec="seconds"),
                "entry_at": entry_at.isoformat(timespec="seconds"),
                "entry_quote_ids": [q["record_id"] for q in entry_legs],
                "mark_quote_ids": [q["record_id"] for q in exit_legs],
                "entry_debit": round(debit, 6),
                "mark_liquidation_credit": round(credit, 6),
                "round_trip_commissions_per_share": round(fees, 6),
                "net_return": round(net_return, 6),
                "price_evidence": "OBSERVED_RTH_EXECUTABLE_BID_ASK",
                "strategy_alpha_attributable": False,
                "production_eligible": False,
            })
    added = _journal_append(root / MARKS, new, known=old)
    all_marks = [*old, *new]
    total_by_horizon = Counter(m["horizon_days"] for m in all_marks)
    # Research thresholds are labels on the ORIGINAL rejected model's
    # expected ROI, not permission to trade, and are not backtest successes.
    buckets = {f"{t:.2f}": {"eligible_counterfactual_cohorts": 0, "realized_mark_45": 0}
               for t in RESEARCH_THRESHOLDS}
    ids45 = {m["cohort_id"] for m in all_marks if m["horizon_days"] == 45}
    mark45 = {m["cohort_id"]: m for m in all_marks if m["horizon_days"] == 45}
    for c in cohorts:
        roi = c.get("original_roi_net")
        for t in RESEARCH_THRESHOLDS:
            if roi is not None and roi >= t:
                bucket = buckets[f"{t:.2f}"]
                bucket["eligible_counterfactual_cohorts"] += 1
                bucket["realized_mark_45"] += int(c["cohort_id"] in ids45)
    # A descriptive prospective calibration comparison – NOT a causal
    # backtest and never a permission to alter the production ROI hurdle.
    from statistics import median
    for threshold in RESEARCH_THRESHOLDS:
        bucket = buckets[f"{threshold:.2f}"]
        observed = [(mark45[c["cohort_id"]]["net_return"], c["signal_date"])
                    for c in cohorts if c["cohort_id"] in mark45
                    and c.get("original_roi_net") is not None
                    and c["original_roi_net"] >= threshold]
        dates = len({d for _, d in observed})
        bucket["n_independent_signal_dates"] = dates
        bucket["net_mean_45d"] = (round(sum(x for x,_ in observed)/len(observed), 6)
                                    if len(observed) >= 20 and dates >= 10 else None)
        bucket["net_median_45d"] = (round(median(x for x,_ in observed), 6)
                                      if len(observed) >= 20 and dates >= 10 else None)
        bucket["win_rate_45d"] = (round(sum(x > 0 for x,_ in observed)/len(observed), 6)
                                    if len(observed) >= 20 and dates >= 10 else None)
        bucket["calibration_status"] = ("DESCRIPTIVE_ONLY" if bucket["net_mean_45d"] is not None
                                       else "NEED_MORE_MATURED_QUOTES")
    report = {
        "schema": "roi-learning-summary-v1", "as_of": today.isoformat(),
        "n_rejected_candidates_registered": len(cohorts),
        "n_new_marks": added, "n_all_marks": len(all_marks),
        "marks_by_horizon": dict(sorted(total_by_horizon.items())),
        "missing_quotes_by_reason": dict(sorted(unknown.items())),
        "research_roi_thresholds": buckets,
        "actual_production_roi_gate_changed": False,
        "research_only": True, "orders_created": 0, "llm_calls": 0,
        "verified_trades_do_not_imply_factor_alpha": True,
    }
    # Read-only research diagnostics: existing journals and daily reports only.
    report["economics_first_acquisition"] = _economics_acquisition_report(root, today)
    report["candidate_underlying_by_gate"] = _candidate_underlying_by_gate(root, today)
    report["observed_option_economics"] = _observed_option_economics(cohorts, all_marks, today, cfg)
    markdown = _gate_economics_markdown(report)
    md_output = root / ROOT / "gate_economics.md"
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_temp = md_output.with_suffix(".md.tmp")
    md_temp.write_text(markdown, encoding="utf-8")
    os.replace(md_temp, md_output)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temp = root / (str(SUMMARY) + ".tmp")
    output = root / SUMMARY
    output.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate", "status", "due"])
    opts = parser.parse_args(argv)
    if opts.command == "evaluate":
        obj = evaluate()
    elif opts.command == "due":
        obj = {"symbols": due_capture_symbols()}
    else:
        obj = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {"status": "NOT_STARTED"}
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
