"""modules/chain_summary.py – Kennzahlen aus ohnehin abgerufenen Tradier-Ketten (B2, RESEARCH/SHADOW, 2026-10-15).

Keine zusätzlichen API-Aufrufe: market_snapshot._fetch_chain und options_designer._tradier_chain reichen jede
geladene Kette (Ticker + Verfall) hierher weiter. Pro Kette wird EINE kompakte Zeile angehängt
(outputs/options_archive/chain_summaries/<UTC-Datum>.jsonl, append-only, Union-Merge):

  call_oi / put_oi, call_volume / put_volume, pc_oi_ratio, pc_volume_ratio,
  atm_iv (Mittel der mid_iv von Call und Put mit |Delta| am nächsten an 0,5), atm_strike, n_contracts

Daraus (features): OI-/Volumen-Änderung gegenüber der letzten früheren Zusammenfassung derselben Kette und die
IV-Laufzeitstruktur (atm_iv der fernsten minus der nächsten Laufzeit desselben Tickers am selben Tag).
Fehlende Werte bleiben None (nie 0). Wirft nie in die aufrufende Pipeline; keine Entscheidungswirkung.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path("outputs/options_archive/chain_summaries")
SCHEMA = "chain-summary-v1"
_LOCK = threading.Lock()
_SEEN: set[tuple] = set()                 # (Tag, Ticker, Verfall) je Prozess nur einmal schreiben


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None           # NaN -> None


def summarize(ticker: str, expiration: str, options: list[dict], now: datetime | None = None) -> dict | None:
    now = now or datetime.now(timezone.utc)
    opts = [o for o in options or [] if isinstance(o, dict) and o.get("option_type") in ("call", "put")]
    if not opts:
        return None

    def agg(kind: str, field: str) -> float | None:
        vals = [_num(o.get(field)) for o in opts if o["option_type"] == kind]
        vals = [v for v in vals if v is not None]
        return float(sum(vals)) if vals else None

    def atm(kind: str) -> dict | None:
        best = None
        for o in opts:
            if o["option_type"] != kind:
                continue
            g = o.get("greeks") or {}
            d, iv = _num(g.get("delta")), _num(g.get("mid_iv"))
            if d is None or iv is None or iv <= 0.01:
                continue
            k = abs(abs(d) - 0.5)
            if best is None or k < best[0]:
                best = (k, iv, _num(o.get("strike")))
        return {"iv": best[1], "strike": best[2]} if best else None

    c, p = atm("call"), atm("put")
    ivs = [x["iv"] for x in (c, p) if x]
    coi, poi, cv, pv = agg("call", "open_interest"), agg("put", "open_interest"), agg("call", "volume"), agg("put", "volume")
    try:
        dte = (date.fromisoformat(expiration) - now.date()).days
    except ValueError:
        dte = None
    return {"schema": SCHEMA, "ticker": ticker, "expiration": expiration, "dte": dte,
            "retrieved_at": now.isoformat(timespec="seconds"), "n_contracts": len(opts),
            "call_oi": coi, "put_oi": poi, "call_volume": cv, "put_volume": pv,
            "pc_oi_ratio": round(poi / coi, 4) if poi is not None and coi else None,
            "pc_volume_ratio": round(pv / cv, 4) if pv is not None and cv else None,
            "atm_iv": round(sum(ivs) / len(ivs), 6) if ivs else None,
            "atm_strike": (c or p or {}).get("strike")}


def record(ticker: str, expiration: str, options: list[dict], source: str, now: datetime | None = None,
           root: Path | None = None) -> bool:
    """Hängt die Zusammenfassung an (einmal je Tag/Ticker/Verfall und Prozess). Nie werfend."""
    if root is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return False                       # Tests mit gemockten Ketten schreiben nie ins Repo-Archiv
    try:
        s = summarize(ticker, expiration, options, now)
        if s is None:
            return False
        day = s["retrieved_at"][:10]
        key = (str(root), day, ticker, expiration)
        with _LOCK:
            if key in _SEEN:
                return False
            _SEEN.add(key)
            base = Path(root or ROOT)
            base.mkdir(parents=True, exist_ok=True)
            with open(base / f"{day}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**s, "source": source}, sort_keys=True) + "\n")
        return True
    except Exception as e:  # noqa: BLE001 – Research-Nebenprodukt darf die Pipeline nie stören
        log.debug(f"chain_summary.record [{ticker} {expiration}] ignoriert: {e}")
        return False


def load(root: Path | None = None) -> list[dict]:
    out = []
    for f in sorted(Path(root or ROOT).glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def features(rows: list[dict]) -> list[dict]:
    """Je (Tag, Ticker): Laufzeitstruktur + je Kette die Änderung zur letzten früheren Zusammenfassung."""
    by_chain: dict[tuple, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r.get("retrieved_at") or ""):
        by_chain.setdefault((r["ticker"], r["expiration"]), []).append(r)
    latest_per_day: dict[tuple, dict] = {}
    for (tk, exp), rs in by_chain.items():
        for r in rs:
            latest_per_day[(r["retrieved_at"][:10], tk, exp)] = r          # letzter Stand des Tages
    out: dict[tuple, dict] = {}
    for (day, tk, exp), r in sorted(latest_per_day.items()):
        prev = next((p for p in reversed(by_chain[(tk, exp)]) if p["retrieved_at"][:10] < day), None)
        f = out.setdefault((day, tk), {"date": day, "ticker": tk, "expirations": [], "term_slope_iv": None,
                                       "oi_change_1d": {}, "volume": {}})
        f["expirations"].append({"expiration": exp, "dte": r.get("dte"), "atm_iv": r.get("atm_iv"),
                                 "pc_oi_ratio": r.get("pc_oi_ratio")})
        if prev is not None:
            tot = [r.get("call_oi"), r.get("put_oi"), prev.get("call_oi"), prev.get("put_oi")]
            if None not in tot:
                f["oi_change_1d"][exp] = {"since": prev["retrieved_at"][:10],
                                          "call": tot[0] - tot[2], "put": tot[1] - tot[3]}
        f["volume"][exp] = {"call": r.get("call_volume"), "put": r.get("put_volume")}
    for f in out.values():
        ivs = sorted((e for e in f["expirations"] if e["atm_iv"] is not None and e["dte"] is not None),
                     key=lambda e: e["dte"])
        if len(ivs) >= 2 and ivs[-1]["dte"] > ivs[0]["dte"]:
            f["term_slope_iv"] = round(ivs[-1]["atm_iv"] - ivs[0]["atm_iv"], 6)
            f["term_dte"] = [ivs[0]["dte"], ivs[-1]["dte"]]
    return list(out.values())
