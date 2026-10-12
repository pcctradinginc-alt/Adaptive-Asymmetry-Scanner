"""modules/timing_policy.py – versionierte Entry-/Exit-Timing-Policy für echte Options-Outcomes (2026-10-11).

Konfiguration und Begründung: config/timing_policy.yaml. Kurz:
  timing-v1  Altbestand (Datensätze ohne Feld `timing_policy`): Entry/Exit zu jeder Quote zum Laufzeitpunkt.
             Wird nie umgedeutet, umgerechnet oder überschrieben.
  timing-v2  Entry und Exit nur zu einer ZULÄSSIGEN Optionsquote (option_quotes.usable_rth_quote: Mark-Fenster
             einer regulären Sitzung des Kontrakts, Bid/Ask > 0, nicht STALE), deren Abruf NICHT vor dem
             Entscheidungszeitpunkt liegt (Chain-Quote der Entscheidung höchstens decision_quote_max_age_minutes
             alt). Entscheidung außerhalb des Fensters -> frühestens die nächste zulässige Sitzung. Verfallene
             Kontrakte nur über das offizielle Settlement. Ohne zulässige Quote: PENDING, nach der Sitzung UNUSABLE.

Die Champion-Entscheidungen (Booking, TP/SL/Haltedauer) bleiben unberührt; dieses Modul bestimmt nur, wann und zu
welchem Preis ein Entry/Exit wirtschaftlich als ausgeführt gilt. Reine Funktionen über dem Quote-Archiv (Point-in-
Time: nur Records mit retrieved_at <= now), wirft nie.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "timing_policy.yaml"
V1, V2 = "timing-v1", "timing-v2"
EVAL_VERSION = "timing-eval-v1"
SETTLEMENT_CLOSE_DELAY = timedelta(hours=1)       # Schlusskurs frühestens 1 h nach dem offiziellen Schluss abrufen
_DEFAULTS = {"decision_quote_max_age_minutes": 15, "decision_quote_slack_minutes": 5, "fill_session_grace_hours": 12,
             "spread_leg_pairing_minutes": 15, "settlement_retry_days": 5}


@lru_cache(maxsize=2)
def _load(path: str | None = None) -> dict:
    p = Path(path) if path else CONFIG
    try:
        import yaml
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None
        return doc if isinstance(doc, dict) else {}
    except Exception as e:  # noqa: BLE001 – defekte Policy-Datei: strengste Policy (v2) mit Standardwerten
        log.warning(f"timing_policy: {p} nicht lesbar ({type(e).__name__}: {e}) – timing-v2 mit Standardwerten")
        return {}


def current() -> str:
    """Policy für NEUE Entscheidungen. Fehlende/defekte/unbekannte Angabe -> timing-v2 (nie still der Altbestand)."""
    cur = _load().get("current")
    return cur if cur in (V1, V2) and cur in (_load().get("policies") or {V2: None}) else V2


def params() -> dict:
    pol = ((_load().get("policies") or {}).get(V2) or {})
    out = dict(_DEFAULTS)
    for k in _DEFAULTS:
        for src in (pol, pol.get("entry") or {}, pol.get("exit") or {}):
            if isinstance(src, dict) and isinstance(src.get(k), (int, float)) and not isinstance(src.get(k), bool):
                out[k] = src[k]
    return out


def policy_of(rec: dict | None) -> str:
    """Policy eines Datensatzes; ohne Feld = Altbestand timing-v1."""
    return (rec or {}).get("timing_policy") or V1


def _ts(v) -> datetime | None:
    from modules import option_quotes as oq
    return oq._parse_ts(v)


def _iso(t: datetime | None) -> str | None:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds") if t else None


def _right(strategy: str) -> str:
    s = str(strategy or "").upper()
    return "put" if "PUT" in s or "BEAR" in s else "call"


def trade_legs(trade: dict) -> list[dict]:
    """Legs eines Champion-/Counterfactual-Trades: [{symbol, side, strike, expiry}] (Long zuerst, Spread-Short danach).
    Fehlendes Symbol wird aus Ticker/Verfall/Strike rekonstruiert; nicht rekonstruierbar -> symbol None."""
    from modules import option_quotes as oq
    opt = trade.get("option") or {}
    right = _right(trade.get("strategy"))
    out = []
    for side, leg in (("long", opt), ("short", opt.get("spread_leg"))):
        if not isinstance(leg, dict) or not leg:
            continue
        expiry = leg.get("expiry") or leg.get("expiration") or opt.get("expiry") or opt.get("expiration")
        sym = leg.get("option_symbol") or leg.get("symbol")
        sym = str(sym).strip().upper().replace(" ", "") if sym else None        # wie im Quote-Archiv normalisiert
        if not sym:
            try:
                sym = oq.occ_symbol(trade.get("ticker"), expiry, right, float(leg.get("strike")))
            except (TypeError, ValueError):
                sym = None
        out.append({"symbol": sym, "side": side, "strike": leg.get("strike"), "expiry": expiry})
    return out


def _expiry(legs: list[dict]) -> date | None:
    ds = []
    for g in legs:
        try:
            ds.append(date.fromisoformat(str(g.get("expiry"))[:10]))
        except (TypeError, ValueError):
            pass
    return min(ds) if ds else None


# ── Sitzungen ───────────────────────────────────────────────────────────────
def _window(d: date, symbol=None) -> tuple[datetime, datetime] | None:
    """Mark-/Fill-Fenster [Eröffnung + 30 min, Schluss − 10 min] am NY-Tag d (instrumentenspezifisch)."""
    from modules import market_snapshot as ms
    from modules import option_quotes as oq
    b = ms.regular_session_utc(d, oq.option_root(symbol))
    if b is None:
        return None
    return b[0] + oq.MARK_MIN_AFTER_OPEN, b[1] - oq.MARK_MIN_BEFORE_CLOSE


def fill_day(t: datetime, symbol=None) -> date | None:
    """Frühester NY-Handelstag, an dem ein ab `t` gültiger Fill möglich ist: heute, falls das Fenster noch nicht
    vorbei ist, sonst die nächste zulässige Sitzung (Wochenenden, Feiertage, Sonderschließungen übersprungen)."""
    from modules import market_snapshot as ms
    d = t.astimezone(ms.NY_TZ).date()
    for _ in range(20):
        w = _window(d, symbol)
        if w is not None and t <= w[1]:
            return d
        d += timedelta(days=1)
    return None


def in_window(t: datetime, symbol=None) -> bool:
    from modules import option_quotes as oq
    return oq.mark_window(t, symbol) == "IN_WINDOW"


# ── Fills aus dem Quote-Archiv ──────────────────────────────────────────────
def find_fill(symbols: list[str], after: datetime, now: datetime, *, archive=None) -> dict:
    """Erster zulässiger Fill aller `symbols` (Spread: Legs aus derselben Erfassungsrunde) mit Abruf >= `after`,
    ausschließlich in der frühesten möglichen Sitzung (fill_day). -> {status FILLED|PENDING|UNUSABLE, day, time,
    records, reason}. Nur Quotes mit retrieved_at <= now (Point-in-Time); nie ein Ersatzpreis."""
    from modules import option_quotes as oq
    p = params()
    day = fill_day(after, symbols[0] if symbols else None)
    if day is None:
        return {"status": "UNUSABLE", "reason": "NO_SESSION_FOUND", "day": None}
    out = {"day": day.isoformat()}
    if now >= after:
        hist = oq.rth_quote_history(symbols, now, root=archive,
                                    lookback_days=max(1, (now.date() - after.date()).days + 2))
        cands = {}
        for s in symbols:
            cands[s] = [r for qd, r in hist.get(s) or []
                        if qd == day.isoformat() and oq.ny_date(r.get("retrieved_at")) == day.isoformat()
                        and (_ts(r.get("retrieved_at")) or after) >= after]
        if symbols and all(cands[s] for s in symbols):
            if len(symbols) == 1:
                rec = cands[symbols[0]][0]
                return {**out, "status": "FILLED", "time": _iso(_ts(rec["retrieved_at"])), "records": [rec]}
            pair = timedelta(minutes=p["spread_leg_pairing_minutes"])
            for r0 in cands[symbols[0]]:
                t0 = _ts(r0["retrieved_at"])
                rest = []
                for s in symbols[1:]:
                    m = next((r for r in cands[s] if abs(_ts(r["retrieved_at"]) - t0) <= pair), None)
                    if m is None:
                        break
                    rest.append(m)
                if len(rest) == len(symbols) - 1:
                    recs = [r0, *rest]
                    return {**out, "status": "FILLED", "records": recs,
                            "time": _iso(max(_ts(r["retrieved_at"]) for r in recs))}
    w = _window(day, symbols[0] if symbols else None)
    if w is not None and now > w[1] + timedelta(hours=p["fill_session_grace_hours"]):
        return {**out, "status": "UNUSABLE", "reason": "NO_ADMISSIBLE_QUOTE_IN_SESSION"}
    return {**out, "status": "PENDING",
            "reason": "AWAITING_SESSION" if w is None or now < w[0] else "AWAITING_ADMISSIBLE_QUOTE"}


def _num(v) -> float | None:
    from modules import option_quotes as oq
    return oq._num(v)


def entry_price(records: list[dict]) -> float | None:
    """Kauf zum Ask; Spread: Long-Ask − Short-Bid (Records in Leg-Reihenfolge Long, Short)."""
    a = _num(records[0].get("ask"))
    if a is None:
        return None
    if len(records) == 1:
        return round(a, 4)
    b = _num(records[1].get("bid"))
    return None if b is None else round(a - b, 4)


def exit_price(records: list[dict]) -> float | None:
    """Verkauf zum Bid; Spread: Long-Bid − Short-Ask, nie negativ."""
    b = _num(records[0].get("bid"))
    if b is None:
        return None
    if len(records) == 1:
        return round(b, 4)
    a = _num(records[1].get("ask"))
    return None if a is None else round(max(b - a, 0.0), 4)


def settlement_value(trade: dict, legs: list[dict], expiry: date, close: float) -> float | None:
    """Innerer Wert am Verfall aus dem offiziellen (unbereinigten) Schluss: Long-Option bzw. Debit-Spread
    (Breite gekappt). Keine Quote, kein Modellpreis – das Settlement selbst."""
    put = _right(trade.get("strategy")) == "put"

    def iv(k):
        k = float(k)
        return max(k - close, 0.0) if put else max(close - k, 0.0)
    try:
        long_k = legs[0]["strike"]
        if len(legs) == 1:
            return round(iv(long_k), 4)
        return round(max(iv(long_k) - iv(legs[1]["strike"]), 0.0), 4)
    except (TypeError, ValueError, KeyError, IndexError):
        return None


# ── Bewertung ───────────────────────────────────────────────────────────────
def evaluate_entry(trade: dict, now: datetime, *, archive=None) -> tuple[dict, list[dict]]:
    """Entry nach timing-v2 -> ({status FILLED|PENDING|UNUSABLE, ...}, Fill-Records)."""
    dt = _ts(trade.get("decision_time"))
    if dt is None:
        return {"status": "UNUSABLE", "reason": "DECISION_TIME_MISSING"}, []
    frozen = _frozen_entry(trade, archive)
    if frozen is not None:                       # einmal festgestellter Fill bleibt (reproduzierbar, nie nachträglich)
        return frozen
    legs = trade_legs(trade)
    syms = [g["symbol"] for g in legs]
    if not syms or any(s is None for s in syms):
        return {"status": "UNUSABLE", "reason": "CONTRACT_UNKNOWN"}, []
    p = params()
    # Chain-Quote der Entscheidung zählt nur, wenn die Entscheidung selbst im Fenster liegt und die Quote frisch ist
    after = dt - timedelta(minutes=p["decision_quote_max_age_minutes"]) if in_window(dt, syms[0]) else dt
    f = find_fill(syms, after, now, archive=archive)
    blk = {"status": f["status"], "decision_time": _iso(dt), "day": f.get("day"), "reason": f.get("reason")}
    if f["status"] != "FILLED":
        return blk, []
    exp = _expiry(legs)
    if exp is not None and date.fromisoformat(f["day"]) > exp:
        return {**blk, "status": "UNUSABLE", "reason": "ENTRY_AFTER_EXPIRY"}, []
    price = entry_price(f["records"])
    if price is None or price <= 0:
        return {**blk, "status": "UNUSABLE", "reason": "ENTRY_PRICE_INVALID"}, []
    t_fill = max(dt, _ts(f["time"]))                          # nie vor der Entscheidung ausgeführt
    return {**blk, "time": _iso(t_fill), "price": price,
            "record_ids": [r.get("record_id") for r in f["records"]],
            "quote_retrieved_at": [r.get("retrieved_at") for r in f["records"]]}, f["records"]


def _frozen_entry(trade: dict, archive=None) -> tuple[dict, list[dict]] | None:
    """Bereits festgestellter Entry-Fill (trade['timing']['entry'] FILLED mit Record-IDs) samt seinen Records aus dem
    Archiv; ein später committeter, früher abgerufener Record ändert einen festgestellten Fill nie nachträglich."""
    from modules import option_quotes as oq
    e = (trade.get("timing") or {}).get("entry") or {}
    ids = [i for i in e.get("record_ids") or [] if i]
    if e.get("status") != "FILLED" or not ids or not e.get("day"):
        return None
    d = date.fromisoformat(e["day"])
    found = oq.by_ids(ids, (d - timedelta(days=1)).isoformat(), (d + timedelta(days=1)).isoformat(), root=archive)
    recs = [found.get(i) for i in ids]
    return (dict(e), recs) if all(recs) else None


def evaluate_exit(trade: dict, now: datetime, *, archive=None, close_fn=None) -> tuple[dict, list[dict]]:
    """Exit nach timing-v2 ab dem Exit-Entscheid (trade['close_time']) -> ({status FILLED|SETTLED|PENDING|UNUSABLE,
    ...}, Fill-Records). Verfall vor einer möglichen Ausführung -> offizielles Settlement."""
    xt = _ts(trade.get("close_time"))
    if xt is None:
        return {"status": "UNUSABLE", "reason": "EXIT_DECISION_TIME_MISSING"}, []
    legs = trade_legs(trade)
    syms = [g["symbol"] for g in legs]
    if not syms or any(s is None for s in syms):
        return {"status": "UNUSABLE", "reason": "CONTRACT_UNKNOWN"}, []
    p = params()
    exp = _expiry(legs)
    blk = {"decision_time": _iso(xt)}
    day = fill_day(xt, syms[0])
    if exp is not None and (day is None or day > exp):
        return _settle(trade, legs, exp, now, blk, close_fn), []
    after = xt - timedelta(minutes=p["decision_quote_slack_minutes"]) if in_window(xt, syms[0]) else xt
    f = find_fill(syms, after, now, archive=archive)
    blk.update(status=f["status"], day=f.get("day"), reason=f.get("reason"))
    if f["status"] != "FILLED":
        w = _window(exp, syms[0]) if exp is not None else None
        if exp is not None and day == exp and w is not None and now > w[1]:
            return _settle(trade, legs, exp, now, blk, close_fn), []      # am Verfallstag nicht mehr ausführbar
        return blk, []
    price = exit_price(f["records"])
    if price is None:
        return {**blk, "status": "UNUSABLE", "reason": "EXIT_PRICE_INVALID"}, []
    return {**blk, "time": _iso(max(xt, _ts(f["time"]))), "price": price,
            "record_ids": [r.get("record_id") for r in f["records"]],
            "quote_retrieved_at": [r.get("retrieved_at") for r in f["records"]]}, f["records"]


def _settle(trade: dict, legs: list[dict], exp: date, now: datetime, blk: dict, close_fn) -> dict:
    from modules import market_snapshot as ms
    blk = {**blk, "day": exp.isoformat(), "settlement": True}
    # Erst nach dem OFFIZIELLEN Schluss (+ Karenz für den konsolidierten Tagesbalken), nie ein Intraday-Kurs
    b = ms.regular_session_utc(exp, None)
    if b is not None and now < b[1] + SETTLEMENT_CLOSE_DELAY:
        return {**blk, "status": "PENDING", "reason": "AWAITING_EXPIRY_CLOSE"}
    close = (close_fn or ms.unadjusted_close_on)(trade.get("ticker"), exp)
    if close is None:
        late = now.date() > exp + timedelta(days=int(params()["settlement_retry_days"]))
        return {**blk, "status": "UNUSABLE" if late else "PENDING",
                "reason": "SETTLEMENT_CLOSE_MISSING" if late else "AWAITING_SETTLEMENT_CLOSE"}
    val = settlement_value(trade, legs, exp, float(close))
    if val is None:
        return {**blk, "status": "UNUSABLE", "reason": "SETTLEMENT_UNCOMPUTABLE"}
    b = ms.regular_session_utc(exp, None)
    return {**blk, "status": "SETTLED", "price": val, "underlying_close": float(close), "time": _iso(b[1]) if b else None}


def evaluate(trade: dict, now: datetime, *, archive=None, close_fn=None) -> tuple[dict, list[dict], list[dict]]:
    """Wirtschaftliche Bewertung eines timing-v2-Trades -> (Block, Entry-Records, Exit-Records).
    Block: {policy, version, evaluated_at, status OK|PENDING|UNUSABLE, reasons, entry, exit, outcome, method,
    price_type}. Ohne Exit-Entscheid wird nur der Entry bewertet (status PENDING bis zum Exit)."""
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    blk = {"policy": V2, "version": EVAL_VERSION, "evaluated_at": _iso(now), "outcome": None, "method": None,
           "price_type": None, "exit": None}
    try:
        entry, e_recs = evaluate_entry(trade, now, archive=archive)
        blk["entry"] = entry
        if not trade.get("close_time"):
            blk.update(status="UNUSABLE" if entry["status"] == "UNUSABLE" else "PENDING",
                       reasons=[f"ENTRY_{entry['reason']}" if entry.get("reason") else "OPEN"])
            return blk, e_recs, []
        ex, x_recs = evaluate_exit(trade, now, archive=archive, close_fn=close_fn)
        blk["exit"] = ex
        reasons = []
        if entry["status"] == "UNUSABLE":
            reasons.append(f"ENTRY_{entry['reason']}")
        elif entry["status"] == "FILLED" and _ts(entry.get("time")) > _ts(ex.get("decision_time")):
            reasons.append("EXIT_BEFORE_ENTRY")                    # nie ausgeführt: Exit-Entscheid vor dem Entry-Fill
        if ex["status"] == "UNUSABLE":
            reasons.append(f"EXIT_{ex['reason']}")
        if reasons:
            blk.update(status="UNUSABLE", reasons=reasons)
            return blk, e_recs, x_recs
        if entry["status"] == "PENDING" or ex["status"] == "PENDING":
            pending = [f"{k}_{b['reason']}" for k, b in (("ENTRY", entry), ("EXIT", ex)) if b["status"] == "PENDING"]
            blk.update(status="PENDING", reasons=pending)
            return blk, e_recs, x_recs
        spread = len(trade_legs(trade)) > 1
        settled = ex["status"] == "SETTLED"
        blk.update(status="OK", reasons=[], outcome=round(ex["price"] / entry["price"] - 1.0, 6),
                   method="expiry_intrinsic" if settled else ("spread_quote" if spread else "option_quote"),
                   price_type="SETTLEMENT_INTRINSIC" if settled else "OBSERVED_QUOTE")
        return blk, e_recs, x_recs
    except Exception as e:  # noqa: BLE001 – Bewertung bricht den Lauf nie; Fehler sind sichtbar UNUSABLE
        log.warning(f"timing_policy.evaluate [{trade.get('ticker')}]: {type(e).__name__}: {e}")
        blk.update(status="UNUSABLE", reasons=[f"EVALUATION_ERROR:{type(e).__name__}"])
        return blk, [], []


def candidate_entry_admissible(signal_ts, quote_ts, symbol=None) -> bool:
    """Kandidaten (timing-v2): sofortiger Entry zur Chain-Quote nur, wenn der Signalzeitpunkt im Fenster liegt und die
    Quote höchstens decision_quote_max_age_minutes vor dem Signal notiert ist (sonst nächste zulässige Quote)."""
    st, qt = _ts(signal_ts), _ts(quote_ts)
    if st is None or qt is None or not in_window(st, symbol):
        return False
    return timedelta(0) <= st - qt <= timedelta(minutes=params()["decision_quote_max_age_minutes"]) \
        or timedelta(0) <= qt - st <= timedelta(minutes=1)
