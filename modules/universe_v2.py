"""modules/universe_v2.py – UNIVERSE_V2: vollständiges US-Optionsuniversum (SHADOW/RESEARCH).

UNIVERSE_V1 (Produktion, unverändert): S&P 500 + Nasdaq-100 mit den Hard-Filtern aus
modules/data_ingestion.py – eingefroren in outputs/universe/universe_v1_frozen.json.

UNIVERSE_V2 trennt strikt:
  RESEARCH_UNIVERSE_V2   alle regulär gehandelten US-Aktien mit gelisteten US-Equity-Optionen,
                         die die großzügigen research_gates erfüllen (keine Cap-Untergrenze)
  TRADEABLE_UNIVERSE_V2  nur Titel, deren Aktie UND Optionen die strengen production_gates
                         erfüllen (eine vorhandene Option Chain ist nie automatisch tradeable)

Je Titel: Market Cap + Bucket (ULTRA_MICRO…MEGA, fehlend = UNKNOWN, nie 0), liquidity_bucket,
options_liquidity_bucket, Spread-/Slippage-/Kostenschätzung, execution_quality, Risikoflags.
Snapshots sind point-in-time (as_of je Lauf, append-only), Delistings werden aus Snapshot-Diffs
protokolliert (Survivorship). Netzwerkzugriffe sind injizierbar (Tests ohne Netz).

Nichts hier verändert eine Produktionsentscheidung.
    python -m modules.universe_v2 [--freeze-v1]
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

CONFIG = Path("config/universe_v2.yaml")
OUT = Path("outputs/universe")
V1_FROZEN = OUT / "universe_v1_frozen.json"
SNAP_DIR = OUT / "v2_snapshots"
DELISTINGS = OUT / "v2_delisting_events.jsonl"
CAP_BUCKETS = ("ULTRA_MICRO", "MICRO", "SMALL", "MID", "LARGE", "MEGA")
UNIVERSE_V1, UNIVERSE_V2 = "V1", "V2"


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG).read_text(encoding="utf-8"))


def universe_version_of(obj: dict) -> str:
    """Verträge/Zeilen ohne Feld stammen aus der Zeit vor V2 -> V1 (Spezifikationen bleiben unverändert)."""
    return obj.get("universe_version") or UNIVERSE_V1


# ── Buckets ─────────────────────────────────────────────────────────────────
def tier(value, thresholds: dict) -> str | None:
    """Größte Stufe, deren untere Grenze <= value. None/NaN -> None (unbekannt, nie 0)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    best = None
    for name, lo in sorted(thresholds.items(), key=lambda kv: kv[1]):
        if value >= lo:
            best = name
    return best


def market_cap_bucket(mc, cfg: dict | None = None) -> str:
    cfg = cfg or load_cfg()
    if mc is None or not isinstance(mc, (int, float)) or mc <= 0 or math.isnan(mc):
        return "UNKNOWN"
    return tier(float(mc), cfg["market_cap_buckets"]) or "UNKNOWN"


# ── Discovery ───────────────────────────────────────────────────────────────
def valid_equity_ticker(t: str, cfg: dict) -> bool:
    if not t or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", t):
        return False
    return not any(re.search(p, t) for p in cfg["discovery"]["excluded_ticker_patterns"])


def parse_sec_listings(js: dict, cfg: dict) -> tuple[list[dict], dict]:
    """SEC company_tickers_exchange.json -> reguläre US-Börsen + gültiges Equity-Mapping."""
    fields = js.get("fields") or ["cik", "name", "ticker", "exchange"]
    allowed = set(cfg["discovery"]["allowed_exchanges"])
    out, skipped = {}, {"exchange": 0, "ticker_pattern": 0, "duplicate": 0}
    for row in js.get("data") or []:
        r = dict(zip(fields, row))
        t = str(r.get("ticker") or "").upper().strip()
        ex = r.get("exchange") or ""
        if ex not in allowed:
            skipped["exchange"] += 1
            continue
        if not valid_equity_ticker(t, cfg):
            skipped["ticker_pattern"] += 1
            continue
        if t in out:
            skipped["duplicate"] += 1
            continue
        out[t] = {"ticker": t, "cik": r.get("cik"), "name": r.get("name"), "exchange": ex}
    return sorted(out.values(), key=lambda x: x["ticker"]), skipped


def listing_status(last_trade: str | None, today: date, cfg: dict) -> str:
    """ACTIVE / STALE (delisted oder suspendiert: kein Handel seit max_days_since_last_trade) / UNKNOWN."""
    if not last_trade:
        return "UNKNOWN"
    d = date.fromisoformat(str(last_trade)[:10])
    lim = int(cfg["discovery"]["max_days_since_last_trade"])
    bdays = sum(1 for k in range(1, (today - d).days + 1) if (d + timedelta(days=k)).weekday() < 5)
    return "ACTIVE" if bdays <= lim else "STALE"


# ── Metriken ────────────────────────────────────────────────────────────────
def underlying_metrics(q: dict) -> dict:
    """q: {price, bid, ask, avg_volume, volume, last_trade, closes[]} -> Kennzahlen (fehlend = None)."""
    price, bid, ask = q.get("price"), q.get("bid"), q.get("ask")
    avg = q.get("avg_volume")
    spread = ((ask - bid) / ((ask + bid) / 2)) if bid and ask and ask >= bid > 0 else None
    closes = [c for c in q.get("closes") or [] if c]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    return {"price": price, "avg_volume": avg,
            "dollar_volume": price * avg if price and avg else None,
            "stock_spread_rel": round(spread, 5) if spread is not None else None,
            "rel_volume": (q.get("volume") / avg) if q.get("volume") and avg else None,
            "max_abs_gap_20d": round(max(abs(r) for r in rets[-20:]), 4) if rets else None,
            "last_trade": q.get("last_trade")}


def option_metrics(chain: list[dict], price: float | None, today: date, gates: dict) -> dict:
    """chain: Optionszeilen {expiry, strike, option_type, bid, ask, open_interest, volume}.
    Kennzahlen nur aus Verfällen im DTE-Fenster und Strikes ±20 % (fehlend = None)."""
    exps = sorted({r["expiry"] for r in chain})
    out = {"expiries_listed": len(exps), "expiries_in_window": 0, "strikes_near_money": 0,
           "open_interest_near": None, "option_volume_near": None, "median_spread_rel": None,
           "median_spread_abs": None, "best_call": None}
    if not chain or not price:
        return out
    lo, hi = int(gates["min_dte"]), int(gates["max_dte"])
    inwin = [e for e in exps if lo <= (date.fromisoformat(e) - today).days <= hi]
    out["expiries_in_window"] = len(inwin)
    near = [r for r in chain if r["expiry"] in inwin and abs(r["strike"] / price - 1) <= 0.20]
    calls = [r for r in near if r.get("option_type", "call") == "call"]
    out["strikes_near_money"] = len({r["strike"] for r in calls})
    if not calls:
        return out
    out["open_interest_near"] = int(sum(r.get("open_interest") or 0 for r in calls))
    out["option_volume_near"] = int(sum(r.get("volume") or 0 for r in calls))
    sp = [((r["ask"] - r["bid"]) / ((r["ask"] + r["bid"]) / 2), r["ask"] - r["bid"]) for r in calls
          if r.get("bid") and r.get("ask") and r["ask"] >= r["bid"] > 0]
    if sp:
        out["median_spread_rel"] = round(statistics.median(x for x, _ in sp), 4)
        out["median_spread_abs"] = round(statistics.median(y for _, y in sp), 4)
    atm = sorted((r for r in calls if r.get("bid") and r.get("ask")),
                 key=lambda r: (abs(r["strike"] / price - 1), -(r.get("open_interest") or 0)))
    out["best_call"] = atm[0] if atm else None
    return out


def execution_cost(bid, ask, open_interest, cfg: dict) -> dict:
    """Geschätzte Roundtrip-Kosten relativ zur Mid-Prämie (Label ESTIMATED):
    Spread (je Seite halber Spread), Slippage (Anteil des Spreads, bei dünnem OI erhöht), Kommission."""
    if not bid or not ask or ask < bid or bid <= 0:
        return {"available": False, "estimated_spread_cost": None, "estimated_slippage": None,
                "estimated_commission": None, "estimated_execution_cost": None}
    ec = cfg["execution_costs"]
    mid = (bid + ask) / 2
    spread_rel = (ask - bid) / mid
    slip_mult = float(ec["slippage_low_oi_multiplier"]) if (open_interest or 0) < \
        int(cfg["production_gates"]["min_open_interest"]) else 1.0
    slippage = float(ec["slippage_spread_fraction"]) * spread_rel * 2 * slip_mult
    commission = 2 * float(ec["commission_per_contract"]) * int(ec["contracts_assumed"]) / (mid * 100)
    return {"available": True, "estimated_spread_cost": round(spread_rel, 5), "estimated_slippage": round(slippage, 5),
            "estimated_commission": round(commission, 5),
            "estimated_execution_cost": round(spread_rel + slippage + commission, 5), "label": "ESTIMATED"}


def execution_quality(cost: float | None, cfg: dict) -> str:
    if cost is None:
        return "UNKNOWN"
    q = cfg["execution_quality"]
    if cost <= q["GOOD"]:
        return "GOOD"
    if cost <= q["FAIR"]:
        return "FAIR"
    if cost <= q["POOR"]:
        return "POOR"
    return "UNTRADEABLE"


def gate_check(um: dict, om: dict, cost: dict, gates: dict) -> list[str]:
    """Fehlende Werte bestehen ein Gate nie (unbekannt != erfüllt)."""
    fails = []

    def need(cond, msg):
        if not cond:
            fails.append(msg)
    v = um.get("price")
    need(v is not None and v >= gates["min_price"], f"Kurs {v} < {gates['min_price']}")
    v = um.get("avg_volume")
    need(v is not None and v >= gates["min_avg_volume"], f"Ø-Volumen {v} < {gates['min_avg_volume']}")
    v = um.get("dollar_volume")
    need(v is not None and v >= gates["min_dollar_volume"], f"Dollar-Volumen {v} < {gates['min_dollar_volume']}")
    v = um.get("stock_spread_rel")
    need(v is not None and v <= gates["max_stock_spread_rel"], f"Aktien-Spread {v} > {gates['max_stock_spread_rel']}")
    need(om.get("expiries_in_window", 0) >= gates["min_expiries"],
         f"geeignete Verfälle {om.get('expiries_in_window', 0)} < {gates['min_expiries']}")
    v = om.get("open_interest_near")
    need(v is not None and v >= gates["min_open_interest"], f"Open Interest {v} < {gates['min_open_interest']}")
    v = om.get("option_volume_near")
    need(v is not None and v >= gates["min_option_volume"], f"Optionsvolumen {v} < {gates['min_option_volume']}")
    v = om.get("median_spread_rel")
    need(v is not None and v <= gates["max_option_spread_rel"], f"Options-Spread rel {v} > {gates['max_option_spread_rel']}")
    v = om.get("median_spread_abs")
    need(v is not None and v <= gates["max_option_spread_abs"], f"Options-Spread abs {v} > {gates['max_option_spread_abs']}")
    need(om.get("strikes_near_money", 0) >= gates["min_strikes_near_money"],
         f"Strike-Dichte {om.get('strikes_near_money', 0)} < {gates['min_strikes_near_money']}")
    v = cost.get("estimated_execution_cost")
    need(v is not None and v <= gates["max_estimated_execution_cost"],
         f"Ausführungskosten {v} > {gates['max_estimated_execution_cost']}")
    return fails


def risk_flags(um: dict, bucket: str, cfg: dict, corporate_action: bool = False) -> list[str]:
    rf = cfg["risk_flags"]
    flags = []
    if um.get("price") is not None and um["price"] < rf["penny_price"]:
        flags.append("PENNY_STOCK")
    if um.get("max_abs_gap_20d") is not None and um["max_abs_gap_20d"] > rf["extreme_gap_abs"]:
        flags.append("EXTREME_GAP")
    if (um.get("rel_volume") or 0) > rf["manipulation_rel_volume"] and bucket in ("ULTRA_MICRO", "MICRO", "UNKNOWN"):
        flags.append("MANIPULATION_RISK")
    if corporate_action:
        flags.append("CORPORATE_ACTION")
    return flags


def assess(ticker: str, quote: dict, chain: list[dict], market_cap, today: date, cfg: dict | None = None,
           market_cap_source: str | None = None, corporate_action: bool = False) -> dict:
    """Vollständige V2-Bewertung eines Titels (research vs. production strikt getrennt)."""
    cfg = cfg or load_cfg()
    um = underlying_metrics(quote)
    status = listing_status(um.get("last_trade"), today, cfg)
    om_r = option_metrics(chain, um.get("price"), today, cfg["research_gates"])
    om_p = option_metrics(chain, um.get("price"), today, cfg["production_gates"])
    bc = om_p.get("best_call") or om_r.get("best_call") or {}
    cost = execution_cost(bc.get("bid"), bc.get("ask"), bc.get("open_interest"), cfg)
    bucket = market_cap_bucket(market_cap, cfg)
    optionable = om_r["expiries_listed"] >= int(cfg["discovery"]["min_expiries_listed"])
    r_fail = ([] if status == "ACTIVE" else [f"Status {status}"]) + ([] if optionable else ["keine gelisteten Optionen"]) \
        + gate_check(um, om_r, cost, cfg["research_gates"])
    p_fail = ([] if status == "ACTIVE" else [f"Status {status}"]) + ([] if optionable else ["keine gelisteten Optionen"]) \
        + gate_check(um, om_p, cost, cfg["production_gates"])
    flags = risk_flags(um, bucket, cfg, corporate_action)
    if "MANIPULATION_RISK" in flags or "PENNY_STOCK" in flags:
        p_fail.append("Risikoflag " + ",".join(f for f in flags if f in ("MANIPULATION_RISK", "PENNY_STOCK")))
    return {"ticker": ticker, "universe_version": UNIVERSE_V2, "as_of": today.isoformat(), "status": status,
            "optionable": optionable, "market_cap": market_cap if bucket != "UNKNOWN" else None,
            "market_cap_source": market_cap_source, "market_cap_bucket": bucket,
            "underlying_liquidity": um, "liquidity_bucket": tier(um.get("dollar_volume"), cfg["liquidity_buckets"]) or "UNKNOWN",
            "options_liquidity": {k: v for k, v in om_r.items() if k != "best_call"},
            "options_liquidity_bucket": tier(om_r.get("open_interest_near"), cfg["options_liquidity_buckets"]) or "NONE",
            "spread_metrics": {"stock_spread_rel": um.get("stock_spread_rel"), "option_spread_rel": om_r.get("median_spread_rel"),
                               "option_spread_abs": om_r.get("median_spread_abs")},
            "estimated_slippage": cost.get("estimated_slippage"), "execution_cost": cost.get("estimated_execution_cost"),
            "execution_quality": execution_quality(cost.get("estimated_execution_cost"), cfg),
            "reference_option": {k: bc.get(k) for k in ("symbol", "expiry", "strike", "bid", "ask", "open_interest")} if bc else None,
            "risk_flags": flags, "research_ok": not r_fail, "research_fail": r_fail,
            "tradeable_ok": not p_fail, "tradeable_fail": p_fail}


# ── Netto-Outcome (theoretisch vs. realistisch handelbar) ───────────────────
def net_outcome(*, raw_underlying_return: float | None, entry_bid, entry_ask, exit_bid=None, exit_ask=None,
                exit_mid=None, open_interest=None, cfg: dict | None = None, delisted: bool = False) -> dict:
    """option_theoretical_return = Mid→Mid. net_realizable_return = Kauf zum Ask + Slippage,
    Verkauf zum Bid − Slippage, minus Kommission. Ohne Exit-Quote: Exit-Spread = relativer
    Einstiegs-Spread (dokumentierte Annahme). Delisted: konservativer Totalverlust (gezählt)."""
    cfg = cfg or load_cfg()
    ec = cfg["execution_costs"]
    out = {"raw_underlying_return": raw_underlying_return, "option_theoretical_return": None,
           "estimated_spread_cost": None, "estimated_slippage": None, "estimated_execution_cost": None,
           "net_realizable_return": None, "liquid_option_available": False, "outcome_status": "UNRESOLVED"}
    if not entry_bid or not entry_ask or entry_ask < entry_bid or entry_bid <= 0:
        out["outcome_status"] = "NO_LIQUID_OPTION"
        return out
    out["liquid_option_available"] = True
    if delisted:
        v = float(cfg["outcomes"]["delisted_option_outcome"])
        out.update(option_theoretical_return=v, net_realizable_return=v, outcome_status="DELISTED_WORST_CASE")
        return out
    mid_in = (entry_bid + entry_ask) / 2
    rel_in = (entry_ask - entry_bid) / mid_in
    if exit_bid and exit_ask and exit_ask >= exit_bid >= 0:
        mid_out, rel_out, status = (exit_bid + exit_ask) / 2, ((exit_ask - exit_bid) / ((exit_bid + exit_ask) / 2)
                                                               if exit_bid + exit_ask > 0 else 0.0), "QUOTED_EXIT"
    elif exit_mid is not None:
        mid_out, rel_out, status = float(exit_mid), rel_in, "ESTIMATED_EXIT_SPREAD"
    else:
        return out
    slip_mult = float(ec["slippage_low_oi_multiplier"]) if (open_interest or 0) < \
        int(cfg["production_gates"]["min_open_interest"]) else 1.0
    slip = float(ec["slippage_spread_fraction"]) * slip_mult
    buy = mid_in * (1 + rel_in / 2 + slip * rel_in)
    sell = max(0.0, mid_out * (1 - rel_out / 2 - slip * rel_out))
    comm = 2 * float(ec["commission_per_contract"]) * int(ec["contracts_assumed"]) / 100
    theo = mid_out / mid_in - 1
    net = (sell - buy - comm) / buy
    out.update(option_theoretical_return=round(theo, 6), net_realizable_return=round(max(-1.0, net), 6),
               estimated_spread_cost=round((rel_in + rel_out) / 2, 6), estimated_slippage=round(slip * (rel_in + rel_out), 6),
               estimated_execution_cost=round(theo - net, 6), outcome_status=status)
    return out


# ── Snapshot (point-in-time) + Survivorship ─────────────────────────────────
def summarize(records: list[dict]) -> dict:
    def cnt(sel, key):
        c: dict[str, int] = {}
        for r in records:
            if sel(r):
                c[r[key]] = c.get(r[key], 0) + 1
        return dict(sorted(c.items()))
    return {"n_listed_checked": sum(1 for r in records if r.get("status") != "UNCHECKED"),
            "n_unchecked": sum(1 for r in records if r.get("status") == "UNCHECKED"),
            "n_optionable": sum(1 for r in records if r["optionable"]),
            "n_research": sum(1 for r in records if r["research_ok"]),
            "n_tradeable": sum(1 for r in records if r["tradeable_ok"]),
            "research_by_bucket": cnt(lambda r: r["research_ok"], "market_cap_bucket"),
            "tradeable_by_bucket": cnt(lambda r: r["tradeable_ok"], "market_cap_bucket"),
            "optionable_by_bucket": cnt(lambda r: r["optionable"], "market_cap_bucket"),
            "execution_quality": cnt(lambda r: r["research_ok"], "execution_quality"),
            "risk_flags": {f: sum(1 for r in records if f in r["risk_flags"])
                           for f in ("PENNY_STOCK", "EXTREME_GAP", "MANIPULATION_RISK", "CORPORATE_ACTION")}}


def save_snapshot(records: list[dict], as_of: date, meta: dict | None = None, snap_dir: Path | None = None,
                  delistings: Path | None = None) -> dict:
    """Append-only: je Lauf eine Datei. Titel, die im Vor-Snapshot RESEARCH waren und jetzt fehlen oder
    STALE sind, werden als Delisting-/Suspendierungsereignis protokolliert (nie still entfernt)."""
    snap_dir, delistings = snap_dir or SNAP_DIR, delistings or DELISTINGS
    prev = latest_snapshot(snap_dir)
    snap = {"universe_version": UNIVERSE_V2, "as_of": as_of.isoformat(), "config_version": load_cfg()["version"],
            "meta": meta or {}, "summary": summarize(records), "records": records}
    snap["snapshot_hash"] = hashlib.sha256(json.dumps(snap, sort_keys=True, default=str).encode()).hexdigest()
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"{as_of.isoformat()}.json.gz"
    if path.exists():
        log.warning(f"universe_v2: Snapshot {path} existiert – nicht überschrieben (append-only)")
        return _read_snap(path)
    # Kompakt: nicht optionierbare Titel nur mit Status/Grund (Größe im Repo), optionierbare vollständig
    snap["records"] = [r if r.get("optionable") else
                       {k: r.get(k) for k in ("ticker", "as_of", "status", "optionable", "research_ok", "tradeable_ok",
                                              "market_cap_bucket", "exchange")} | {"research_fail": (r.get("research_fail") or [])[:2]}
                       for r in records]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(snap, default=str))
    if prev:
        now_by = {r["ticker"]: r for r in records}
        events = []
        for r in prev.get("records") or []:
            if not r.get("research_ok"):
                continue
            cur = now_by.get(r["ticker"])
            if cur is None or cur["status"] == "STALE":
                events.append({"ticker": r["ticker"], "detected_at": as_of.isoformat(),
                               "previous_as_of": prev["as_of"], "event": "MISSING" if cur is None else "STALE",
                               "market_cap_bucket": r.get("market_cap_bucket")})
        if events:
            delistings.parent.mkdir(parents=True, exist_ok=True)
            with open(delistings, "a", encoding="utf-8") as fh:
                for e in events:
                    fh.write(json.dumps(e) + "\n")
    return snap


def _read_snap(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.loads(fh.read())


def latest_snapshot(snap_dir: Path | None = None) -> dict | None:
    d = snap_dir or SNAP_DIR
    files = sorted(d.glob("*.json.gz")) if d.exists() else []
    return _read_snap(files[-1]) if files else None


def membership(snapshot: dict | None, kind: str = "research") -> dict[str, dict]:
    key = "research_ok" if kind == "research" else "tradeable_ok"
    return {r["ticker"]: r for r in (snapshot or {}).get("records") or [] if r.get(key)}


# ── UNIVERSE_V1 einfrieren ──────────────────────────────────────────────────
def v1_definition() -> dict:
    from modules import data_ingestion as di, universe as uv
    import inspect
    src = "".join(inspect.getsource(f) for f in (uv.get_universe, uv._clean, di.DataIngestion._evaluate_ticker))
    try:
        from modules.config import cfg as _cfg
        universe = _cfg.filters.universe
    except Exception:  # noqa: BLE001 – Definition bleibt ohne Config ableitbar
        universe = "sp500_nasdaq100"
    return {"universe_version": UNIVERSE_V1, "universe": universe,
            "hard_filters": {"min_market_cap_usd": di.MIN_MARKET_CAP_USD, "min_avg_volume": di.MIN_AVG_VOLUME,
                             "min_dollar_volume_usd": di.MIN_DOLLAR_VOLUME_USD, "rv_base_threshold": di.RV_BASE_THRESHOLD},
            "static_fallback_hash": hashlib.sha256(json.dumps([sorted(uv._SP500_STATIC), sorted(uv._NASDAQ100_STATIC),
                                                               sorted(uv._DELISTED)]).encode()).hexdigest(),
            "code_hash": hashlib.sha256(src.encode()).hexdigest()}


def freeze_v1(path: Path | None = None, now: datetime | None = None) -> dict:
    """Einmalig; existiert die Datei, wird nur verglichen (nie überschrieben)."""
    path = path or V1_FROZEN
    d = v1_definition()
    d["definition_hash"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    if path.exists():
        return json.loads(path.read_text())
    d["frozen_at"] = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    d["assignment"] = ("Alle bestehenden Forward-Verträge, Hypothesen, Challenger, Outcomes und Promotion-Evidenz "
                       "ohne universe_version-Feld gehören zu V1.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    return d


def v1_unchanged(path: Path | None = None) -> bool:
    frozen = json.loads((path or V1_FROZEN).read_text())
    d = v1_definition()
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == frozen["definition_hash"]


# ── Netzwerk (nur CI) ───────────────────────────────────────────────────────
SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def fetch_sec_listings() -> dict:
    import requests
    from modules.entity_resolution.sources import SEC_HEADERS     # eine SEC-Fair-Access-Konvention im Repo
    r = requests.get(SEC_URL, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def _tradier(path: str, params: dict) -> dict | None:
    import os
    import requests
    key = os.environ.get("TRADIER_API_KEY")
    if not key:
        return None
    r = requests.get(f"https://api.tradier.com/v1/{path}", params=params, timeout=20,
                     headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    if r.status_code == 429:
        time.sleep(5)
        return None
    r.raise_for_status()
    return r.json()


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """Tradier-Quotes in Blöcken (Bid/Ask/Volumen/Ø-Volumen/letzter Handel)."""
    out = {}
    for i in range(0, len(tickers), 100):
        js = _tradier("markets/quotes", {"symbols": ",".join(tickers[i:i + 100])}) or {}
        qs = (js.get("quotes") or {}).get("quote") or []
        for q in [qs] if isinstance(qs, dict) else qs:
            if q.get("type") not in (None, "stock"):
                continue
            td = q.get("trade_date")
            out[q["symbol"]] = {"price": q.get("last") or q.get("prevclose"), "bid": q.get("bid"), "ask": q.get("ask"),
                                "avg_volume": q.get("average_volume"), "volume": q.get("volume"),
                                "last_trade": datetime.fromtimestamp(td / 1000, timezone.utc).date().isoformat()
                                if td else None}
    return out


def select_expiries(exps: list[str], today: date, cfg: dict, n: int = 3) -> list[str]:
    """Max. n Verfälle im Research-Fenster, Produktionsfenster ZUERST. Vorher nur die frühesten
    Research-Verfälle (Weeklies < 30 T) -> Produktionsfenster sah nie einen Verfall, tradeable
    war systematisch 0 (Live 2026-10-04, z. B. AAPL)."""
    def dte(e):
        return (date.fromisoformat(e) - today).days
    r, p = cfg["research_gates"], cfg["production_gates"]
    research = sorted(e for e in exps if int(r["min_dte"]) <= dte(e) <= int(r["max_dte"]))
    prod = [e for e in research if int(p["min_dte"]) <= dte(e) <= int(p["max_dte"])][:n]
    return sorted(prod + [e for e in research if e not in prod][:n - len(prod)])


def fetch_chain(ticker: str, today: date, cfg: dict) -> tuple[list[dict], str]:
    """Option Chain für max. 3 Verfälle (Produktionsfenster bevorzugt, sonst Research-Fenster):
    Tradier, sonst yfinance."""
    lo, hi = int(cfg["research_gates"]["min_dte"]), int(cfg["research_gates"]["max_dte"])
    try:
        js = _tradier("markets/options/expirations", {"symbol": ticker})
        if js is not None:
            exps = ((js.get("expirations") or {}) or {}).get("date") or []
            exps = [exps] if isinstance(exps, str) else exps
            rows = []
            for e in select_expiries(exps, today, cfg):
                cj = _tradier("markets/options/chains", {"symbol": ticker, "expiration": e}) or {}
                opts = (cj.get("options") or {}).get("option") or []
                for o in [opts] if isinstance(opts, dict) else opts:
                    rows.append({"symbol": o.get("symbol"), "expiry": e, "strike": float(o["strike"]),
                                 "option_type": o.get("option_type"), "bid": o.get("bid"), "ask": o.get("ask"),
                                 "open_interest": o.get("open_interest"), "volume": o.get("volume")})
            # Verfälle außerhalb des Fensters zählen für "optionierbar", aber ohne Kettenzeilen
            rows += [{"expiry": e, "strike": 0.0, "option_type": "none"} for e in exps
                     if not (lo <= (date.fromisoformat(e) - today).days <= hi)][:1]
            return rows, "tradier"
    except Exception as e:  # noqa: BLE001 – Fallback auf getesteten yfinance-Pfad
        log.debug(f"universe_v2: Tradier {ticker} ({e})")
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        exps = list(t.options or [])
        rows = []
        for e in select_expiries(exps, today, cfg):
            ch = t.option_chain(e)
            for _, o in ch.calls.iterrows():
                rows.append({"symbol": o.get("contractSymbol"), "expiry": e, "strike": float(o["strike"]),
                             "option_type": "call", "bid": float(o.get("bid") or 0) or None,
                             "ask": float(o.get("ask") or 0) or None,
                             "open_interest": int(o.get("openInterest") or 0), "volume": int(o.get("volume") or 0)})
        rows += [{"expiry": e, "strike": 0.0, "option_type": "none"} for e in exps][:1] if exps and not rows else []
        return rows, "yfinance"
    except Exception as e:  # noqa: BLE001
        log.debug(f"universe_v2: yfinance-Optionen {ticker} ({e})")
        return [], "none"


def fetch_market_cap(ticker: str) -> tuple[float | None, str]:
    try:
        import yfinance as yf
        mc = yf.Ticker(ticker).fast_info.market_cap
        return (float(mc), "yfinance_fast_info") if mc else (None, "missing")
    except Exception:  # noqa: BLE001 – fehlend bleibt None (UNKNOWN-Bucket)
        return None, "missing"


def unchecked_record(ticker: str, listing: dict) -> dict:
    return {"ticker": ticker, "universe_version": UNIVERSE_V2, "as_of": "", "status": "UNCHECKED",
            "optionable": False, "market_cap": None, "market_cap_source": None, "market_cap_bucket": "UNKNOWN",
            "risk_flags": [], "research_ok": False, "research_fail": ["UNCHECKED (Discovery-Budget)"],
            "tradeable_ok": False, "tradeable_fail": ["UNCHECKED (Discovery-Budget)"],
            "exchange": listing.get("exchange"), "cik": listing.get("cik"), "option_source": None}


def discover(today: date | None = None, cfg: dict | None = None, *, listings_js: dict | None = None,
             quotes_fn=fetch_quotes, chain_fn=fetch_chain, cap_fn=fetch_market_cap,
             budget_s: float | None = None, prev: dict | None = None) -> tuple[list[dict], dict]:
    """Rollierende Discovery: Titel mit der ältesten Prüfung zuerst; nicht geprüfte Titel behalten
    ihre letzte (datierte) Bewertung aus dem Vor-Snapshot (as_of bleibt deren Prüfdatum)."""
    today = today or datetime.now(timezone.utc).date()
    cfg = cfg or load_cfg()
    budget_s = budget_s if budget_s is not None else float(cfg["discovery"]["request_budget_s"])
    listings, skipped = parse_sec_listings(listings_js if listings_js is not None else fetch_sec_listings(), cfg)
    prev_by = {r["ticker"]: r for r in (prev or {}).get("records") or []}
    order = sorted(listings, key=lambda x: prev_by.get(x["ticker"], {}).get("as_of", ""))
    quotes = quotes_fn([x["ticker"] for x in order])
    t0, records, checked, unchecked = time.monotonic(), [], 0, 0
    for x in order:
        t = x["ticker"]
        if time.monotonic() - t0 > budget_s:
            if t in prev_by:
                records.append(prev_by[t])
            else:
                # Erstlauf/neue Listings über Budget: explizit UNCHECKED (nie geschätzt, nie RESEARCH/
                # TRADEABLE); as_of leer -> nächster Lauf prüft sie zuerst. Ohne dies lief der Erstlauf
                # ohne Budget in den Job-Timeout und kein Snapshot entstand (Live 2026-10-04).
                records.append(unchecked_record(t, x))
                unchecked += 1
            continue
        q = quotes.get(t) or {}
        chain, src = chain_fn(t, today, cfg) if q else ([], "no_quote")
        mc, mc_src = cap_fn(t) if q else (None, "no_quote")
        rec = assess(t, q, chain, mc, today, cfg, mc_src)
        rec.update(exchange=x["exchange"], cik=x["cik"], option_source=src)
        records.append(rec)
        checked += 1
    meta = {"listings": len(listings), "skipped": skipped, "checked_this_run": checked,
            "unchecked_new": unchecked, "carried_forward": len(records) - checked - unchecked}
    return records, meta


def main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-v1", action="store_true")
    a = ap.parse_args(argv)
    v1 = freeze_v1()
    print(f"UNIVERSE_V1 eingefroren: {v1['definition_hash'][:12]} (unverändert: {v1_unchanged()})")
    if a.freeze_v1:
        return 0
    today = datetime.now(timezone.utc).date()
    records, meta = discover(today, prev=latest_snapshot())
    snap = save_snapshot(records, today, meta)
    print(json.dumps(snap["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
