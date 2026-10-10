"""modules/spread_execution.py – Spread-Pricing (fair vs. executable vs. fill), Quote Quality,
Immediate Liquidation Loss und SPREAD_EXECUTION_LIQUIDITY_GATE (SHADOW_ONLY). Audit 2026-10-09 (SNPS).

ROOT CAUSE (dokumentierter Codepfad vor diesem Modul):
  Entry   options_designer._find_best_option: net_debit = long.ask − short.bid (Leg-by-Leg NBBO,
          = combo_ask); pipeline.build_trade_record: entry_debit = net_debit, entry_quote nur LONG-Leg.
  ROI     options_designer._compute_roi: Kostenbasis = net_debit (executable), Friktion aber
          spread_pct × 2 mit spread_pct = (ask − bid)/ask des LONG-Legs allein – ignoriert die
          Short-Leg-Spanne und die Hebelwirkung des Netto-Debits (Spannen addieren sich, der Debit ist
          eine Differenz). SNPS: modelliert 2 × 8,7 % = 17,4 %, tatsächlich 38,1 % Immediate Loss.
  Filter  cfg.risk.max_bid_ask_ratio nur je Long-Leg; kein Combo-Level-Check.
  Stop    feedback.check_exit_rules: Spread-SL −50 % auf compute_outcome = executable value
          (long.bid − short.ask, feedback.get_current_spread_price) – der Entry-Friktionsverlust
          steht ab der ersten Beobachtung im PnL.
  Mid / Fair Value / Combo-Bid/Ask / Quote-Frische existierten für Spreads nirgends.
=> Inkonsistenz: ROI unterschätzt die Execution-Kosten, Entry und Stop rechnen executable.

Dieses Modul ändert KEINE Produktionsentscheidung. Es berechnet und protokolliert (SHADOW_ONLY).
"""
from __future__ import annotations

import hashlib
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modules.atomic_io import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

CONFIG = Path("config/spread_execution.yaml")
LEDGER_DIR = Path("outputs/intelligence/spread_execution")
GATE_NAME = "SPREAD_EXECUTION_LIQUIDITY_GATE"


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG).read_text(encoding="utf-8"))


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ── Pricing ──────────────────────────────────────────────────────────────────
def combo_quote(long_q: dict | None, short_q: dict | None) -> dict:
    """Debit-Spread (long – short): combo_bid = long.bid − short.ask, combo_ask = long.ask − short.bid."""
    if not long_q or not short_q:
        return {"available": False}
    lb, la, sb, sa = (_f(long_q.get("bid")), _f(long_q.get("ask")), _f(short_q.get("bid")), _f(short_q.get("ask")))
    if None in (lb, la, sb, sa):
        return {"available": False}
    cb, ca = round(lb - sa, 4), round(la - sb, 4)
    out = {"available": True, "long_bid": lb, "long_ask": la, "short_bid": sb, "short_ask": sa,
           "combo_bid": cb, "combo_ask": ca}
    if ca > 0 and ca >= cb:
        mid = round((cb + ca) / 2, 4)
        out.update(combo_mid=mid, fair_value=mid,
                   relative_combo_spread=round((ca - cb) / mid, 4) if mid > 0 else None)
    return out


def quote_quality(long_q: dict | None, short_q: dict | None, combo: dict, cfg: dict,
                  now: datetime | None = None, quote_ts: str | None = None) -> str:
    """HEALTHY | WIDE | VERY_WIDE | STALE | INVALID | UNAVAILABLE – nie eine scheinpräzise PnL ohne Status."""
    if not long_q or not short_q or not combo.get("available"):
        return "UNAVAILABLE"
    qc = cfg.get("quote_quality") or {}
    lb, la, sb, sa = combo["long_bid"], combo["long_ask"], combo["short_bid"], combo["short_ask"]
    if lb <= 0 or sb < 0 or la <= lb or sa <= sb or sa <= 0 or combo["combo_ask"] <= 0 \
            or combo["combo_ask"] < combo["combo_bid"]:
        return "INVALID"                                                     # crossed/locked/leer
    if quote_ts and now:
        try:
            age = (now - datetime.fromisoformat(str(quote_ts).replace("Z", "+00:00"))).total_seconds() / 60
            if age > float(qc.get("max_quote_age_minutes", 30)):
                return "STALE"
        except ValueError:
            return "STALE"
    min_oi = float(qc.get("min_open_interest_leg", 0))
    ois = [_f((q or {}).get("open_interest")) for q in (long_q, short_q)]
    if any(o is not None and o < min_oi for o in ois):
        return "VERY_WIDE"                                                   # Leg-Liquidität unzureichend
    rel = combo.get("relative_combo_spread")
    if rel is None:
        return "INVALID"
    if rel >= float(qc.get("very_wide_rel_combo_spread", 0.40)):
        return "VERY_WIDE"
    if rel >= float(qc.get("wide_rel_combo_spread", 0.15)):
        return "WIDE"
    return "HEALTHY"


def immediate_liquidation_loss(entry_price: float | None, exit_value: float | None) -> float | None:
    """(Entry − sofort ausführbarer Exit-Wert) / Entry; SNPS: (25.20 − 15.60) / 25.20 = 38,1 %."""
    e, x = _f(entry_price), _f(exit_value)
    if e is None or x is None or e <= 0:
        return None
    return round((e - x) / e, 4)


def bucket(ill: float | None, cfg: dict) -> str:
    if ill is None:
        return "UNAVAILABLE"
    edges = list(cfg.get("buckets") or [0.05, 0.10, 0.15, 0.20, 0.30])
    lo = 0.0
    for e in edges:
        if ill < e:
            return f"<{e * 100:.0f}%" if lo == 0 else f"{lo * 100:.0f}-{e * 100:.0f}%"
        lo = e
    return f">={edges[-1] * 100:.0f}%"


# ── Entry-Bewertung + Shadow-Gate ────────────────────────────────────────────
def assess_entry(option: dict, roi: dict | None = None, cfg: dict | None = None, actual_fill: float | None = None,
                 now: datetime | None = None, commission_per_contract: float = 0.65) -> dict:
    """Alle Entry-Kennzahlen eines Debit-Spreads aus dem designten Kontrakt (keine neuen API-Calls)."""
    cfg = cfg or load_cfg()
    sl = (option or {}).get("spread_leg") or {}
    long_q = {"bid": option.get("bid"), "ask": option.get("ask"), "open_interest": option.get("open_interest")}
    short_q = {"bid": sl.get("bid"), "ask": sl.get("ask"), "open_interest": sl.get("open_interest")} if sl else None
    combo = combo_quote(long_q, short_q)
    qq = quote_quality(long_q, short_q, combo, cfg, now, option.get("quote_ts"))
    assumed = combo.get("combo_ask") if combo.get("available") else _f(option.get("net_debit"))
    entry = _f(actual_fill) if actual_fill is not None else assumed
    ill = immediate_liquidation_loss(entry, combo.get("combo_bid")) if combo.get("available") else None
    commission_pct = (commission_per_contract * 2 * 2) / (entry * 100) if entry and entry > 0 else None
    gross = _f((roi or {}).get("roi_gross"))
    vega = _f((roi or {}).get("vega_loss")) or 0.0
    mid = combo.get("combo_mid")
    entry_fric = round((entry - mid) / entry, 4) if entry and mid is not None else None
    exit_fric = round((mid - combo["combo_bid"]) / entry, 4) if entry and mid is not None else None
    roundtrip = round(entry_fric + exit_fric + (commission_pct or 0), 4) if entry_fric is not None else None
    net_exec = round(gross - roundtrip - vega, 4) if gross is not None and roundtrip is not None else None
    verdict, reasons = "PASS", []
    if qq in (cfg.get("shadow_reject_quote_quality") or []):
        verdict, reasons = "WOULD_REJECT", [f"QUOTE_QUALITY={qq}"]
    elif ill is not None and ill >= float(cfg.get("shadow_reject_ill", 0.30)):
        verdict, reasons = "WOULD_REJECT", [f"immediate_liquidation_loss {ill:.1%} >= {cfg['shadow_reject_ill']:.0%}"]
    elif (ill is not None and ill >= float(cfg.get("shadow_flag_ill", 0.15))) or qq in ("WIDE", "VERY_WIDE"):
        verdict = "FLAG"
        reasons = [r for r in (f"immediate_liquidation_loss {ill:.1%}" if ill is not None else None,
                               f"QUOTE_QUALITY={qq}" if qq in ("WIDE", "VERY_WIDE") else None) if r]
    return {
        "gate": GATE_NAME, "gate_mode": cfg.get("mode", "SHADOW_ONLY"), "config_version": cfg.get("version"),
        "legs": {"long": {"strike": option.get("strike"), "bid": combo.get("long_bid"), "ask": combo.get("long_ask"),
                          "open_interest": option.get("open_interest")},
                 "short": {"strike": sl.get("strike"), "bid": combo.get("short_bid"), "ask": combo.get("short_ask"),
                           "open_interest": sl.get("open_interest")}},
        "expiry": option.get("expiry"), "dte": option.get("dte"),
        "combo_bid": combo.get("combo_bid"), "combo_ask": combo.get("combo_ask"), "combo_mid": mid,
        "fair_value": combo.get("fair_value"), "relative_combo_spread": combo.get("relative_combo_spread"),
        "assumed_entry": assumed, "fill_assumption_method": cfg.get("fill_assumption", "NATURAL_LEG_BY_LEG"),
        "actual_entry_fill": _f(actual_fill), "immediately_executable_exit_value": combo.get("combo_bid"),
        "immediate_liquidation_loss_pct": ill, "ill_bucket": bucket(ill, cfg), "quote_quality": qq,
        "gross_expected_roi": gross, "expected_entry_friction": entry_fric, "expected_exit_friction": exit_fric,
        "commission_pct": round(commission_pct, 4) if commission_pct is not None else None,
        "estimated_slippage": entry_fric, "expected_roundtrip_cost": roundtrip,
        "net_expected_roi": net_exec, "production_roi_net": _f((roi or {}).get("roi_net")),
        "production_friction_model": (round(2 * _f(roi.get("spread_pct")), 4)
                                      if roi and _f(roi.get("spread_pct")) is not None else None),
        "shadow_verdict": verdict, "shadow_reasons": reasons,
        "production_effect": "NONE (SHADOW_ONLY)",
    }


# ── Monitoring / Stop-Shadow ─────────────────────────────────────────────────
def monitor(trade: dict, long_q: dict | None, short_q: dict | None, underlying_price: float | None,
            cfg: dict | None = None, now: datetime | None = None, quote_ts: str | None = None) -> dict:
    """Fair-Value-PnL (Mid), Executable-PnL (combo_bid) und Underlying-Bewegung – keine Zahl ohne Status."""
    cfg = cfg or load_cfg()
    entry = _f(trade.get("entry_debit")) or _f((trade.get("option") or {}).get("net_debit"))
    combo = combo_quote(long_q, short_q)
    qq = quote_quality(long_q, short_q, combo, cfg, now, quote_ts)
    ref = _f((trade.get("simulation") or {}).get("current_price"))
    und = round(underlying_price / ref - 1, 4) if ref and underlying_price else None
    usable = qq in ("HEALTHY", "WIDE", "VERY_WIDE") and entry
    return {"ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "quote_quality": qq,
            "combo_bid": combo.get("combo_bid"), "combo_ask": combo.get("combo_ask"), "combo_mid": combo.get("combo_mid"),
            "fair_value": combo.get("fair_value") if usable else None,
            "executable_exit_value": combo.get("combo_bid") if usable else None,
            "fair_value_pnl_pct": round(combo["combo_mid"] / entry - 1, 4) if usable and combo.get("combo_mid") is not None else None,
            "executable_pnl_pct": round(combo["combo_bid"] / entry - 1, 4) if usable else None,
            "underlying_return_pct": und}


def stop_shadow(observations: list[dict], production_stop: float = -0.50, cfg: dict | None = None) -> dict:
    """Robuste Stop-Bewertung (nur SHADOW; Produktions-Stop unverändert)."""
    cfg = cfg or load_cfg()
    sc = cfg.get("stop_shadow") or {}
    k = int(sc.get("confirm_observations", 2))
    fv_stop = float(sc.get("fair_value_stop", production_stop))
    small = float(sc.get("small_underlying_move", 0.05))
    last = observations[-1] if observations else {}
    prod_hit = last.get("executable_pnl_pct") is not None and last["executable_pnl_pct"] <= production_stop
    if last.get("quote_quality") in ("STALE", "INVALID", "UNAVAILABLE"):
        return {"production_stop_hit": False, "shadow_stop": "NO_RELIABLE_QUOTE",
                "reason": f"QUOTE_QUALITY={last.get('quote_quality')} – keine belastbare Stop-Auslösung"}
    recent = [o for o in observations[-k:] if o.get("quote_quality") in ("HEALTHY", "WIDE", "VERY_WIDE")]
    confirmed = len(recent) == k and all((o.get("executable_pnl_pct") or 0) <= production_stop for o in recent)
    fv = last.get("fair_value_pnl_pct")
    und = last.get("underlying_return_pct")
    if prod_hit and fv is not None and fv > fv_stop and und is not None and abs(und) < small:
        verdict = "EXECUTION_DRIVEN"          # Stop durch Bid/Ask-Struktur, nicht durch ökonomischen Verlust
    elif prod_hit and confirmed and (fv is None or fv <= fv_stop):
        verdict = "STOP_CONFIRMED"            # echter Einbruch: Produktionsschutz bleibt voll wirksam
    elif prod_hit:
        verdict = "UNCONFIRMED"
    else:
        verdict = "NO_STOP"
    return {"production_stop_hit": prod_hit, "shadow_stop": verdict, "confirmed_observations": len(recent),
            "fair_value_pnl_pct": fv, "executable_pnl_pct": last.get("executable_pnl_pct"),
            "underlying_return_pct": und}


# ── Ledger ───────────────────────────────────────────────────────────────────
def record(kind: str, ticker: str, date: str, payload: dict, ledger_dir: Path | None = None) -> bool:
    """Append-only, idempotent je (kind, ticker, date, Payload-Hash)."""
    ledger_dir = Path(ledger_dir or LEDGER_DIR)
    row = {"kind": kind, "ticker": ticker, "date": str(date)[:10], **payload}
    rid = hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = ledger_dir / f"{str(date)[:7]}.jsonl"
    if path.exists() and any(r.get("record_id") == rid for r in read_jsonl(path)):
        return False
    append_jsonl(path, [{"record_id": rid, **row}], ensure_ascii=False, default=str)
    return True


# ── Shadow-Analyse ───────────────────────────────────────────────────────────
def analyse(trades: list[dict], cfg: dict | None = None, production_stop: float = -0.50) -> dict:
    """Zusammenhang immediate_liquidation_loss <-> Outcome je Bucket. n < min_n -> NEED_MORE_DATA.
    Historische Outcomes sind überwiegend NON_RELIABLE -> explorativ gekennzeichnet. Keine Schwellenoptimierung."""
    from modules.outcomes import is_reliable_outcome
    cfg = cfg or load_cfg()
    min_n = int(cfg.get("min_n_per_bucket", 30))
    rows = []
    for t in trades:
        if "SPREAD" not in str(t.get("strategy") or "") or not (t.get("option") or {}).get("spread_leg"):
            continue
        a = assess_entry(t["option"], None, cfg)
        if a["immediate_liquidation_loss_pct"] is None:
            continue
        rows.append({"bucket": a["ill_bucket"], "ill": a["immediate_liquidation_loss_pct"],
                     "quote_quality": a["quote_quality"], "outcome": t.get("outcome"),
                     "reliable": bool(t.get("outcome") is not None and is_reliable_outcome(t)),
                     "stop": t.get("close_reason") == "stop_loss", "dte": (t.get("option") or {}).get("dte"),
                     "underlying": _f(t.get("close_price")) / _f((t.get("simulation") or {}).get("current_price")) - 1
                     if _f(t.get("close_price")) and _f((t.get("simulation") or {}).get("current_price")) else None,
                     "mfe": t.get("mfe"), "mae": t.get("mae") or t.get("peak_return")})
    edges = list(cfg.get("buckets") or [0.05, 0.10, 0.15, 0.20, 0.30])
    order = [f"<{edges[0] * 100:.0f}%"] + [f"{a * 100:.0f}-{b * 100:.0f}%" for a, b in zip(edges, edges[1:])] \
        + [f">={edges[-1] * 100:.0f}%"]
    out = {}
    for b in order:
        g = [r for r in rows if r["bucket"] == b]
        done = [r for r in g if r["outcome"] is not None]
        stops = [r for r in done if r["stop"]]
        false_stops = [r for r in stops if r["underlying"] is not None and abs(r["underlying"]) < 0.05]
        out[b] = {"n_candidates": len(g), "n_with_outcome": len(done), "n_reliable": sum(r["reliable"] for r in done),
                  "status": "NEED_MORE_DATA" if len(done) < min_n else "OK",
                  "mean_outcome_exploratory": round(statistics.mean(r["outcome"] for r in done), 4) if done else None,
                  "stop_frequency": round(len(stops) / len(done), 3) if done else None,
                  "false_stop_frequency": round(len(false_stops) / len(stops), 3) if stops else None,
                  "median_dte": statistics.median([r["dte"] for r in g if r["dte"]]) if any(r["dte"] for r in g) else None}
    return {"n_spreads": len(rows), "buckets": out, "label": "EXPLORATORY (überwiegend NON_RELIABLE Outcomes)",
            "note": "Keine automatische Schwellenoptimierung; n < min_n je Bucket -> NEED_MORE_DATA."}
