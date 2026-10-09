"""
feedback.py – Adaptive Lern-Loop v5.0

Änderungen v5.0:
    - Tradier Live-API als primäre Datenquelle für Optionspreise (P&L-Tracking)
      Endpoint: /v1/markets/options/chains (Optionspreis via Strike-Filter)
      Endpoint: /v1/markets/quotes        (Aktienkurs Real-Time)
    - get_current_price():        Tradier Primary → yfinance Fallback
    - get_current_option_price(): realisierbarer Preis (Verkauf=Bid), Tradier → yfinance
    - compute_outcome():          strategy-Parameter für saubere Call/Put-Erkennung
    - TRADIER_API_KEY via os.environ (bereits als GitHub Secret hinterlegt)
    - Warum wichtig: RL-Agent trainiert auf Outcomes — falsche Preise (yfinance
      ~15min delayed) führen zu fehlerhaften Lern-Signalen für den PPO-Agenten.

Änderungen v4.0:
    - Nach Trade-Close: PPO-Agent wird auf neuem closed_trade nachtrainiert
    - RL-Training: Inkrementelles Update (Continual Learning)
    - Bestehende Fixes M-04, M-05 bleiben erhalten
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import yfinance as yf
from scipy import stats

from modules.config import cfg
from modules.exit_sim import register_exit_sim, summarize_exit_sim, trim_exit_sim, update_exit_sim_entry
from modules import candidate_ledger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")

TRADIER_BASE    = "https://api.tradier.com/v1"
TRADIER_TIMEOUT = 10


# ── Tradier Hilfsfunktionen ───────────────────────────────────────────────────

def _tradier_headers() -> dict:
    """Authorization-Header für Tradier Live-API."""
    api_key = os.environ.get("TRADIER_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
    }


def _use_tradier() -> bool:
    """Gibt True zurück wenn TRADIER_API_KEY gesetzt ist."""
    return bool(os.environ.get("TRADIER_API_KEY", "").strip())


# ── History I/O ───────────────────────────────────────────────────────────────

def load_history() -> dict:
    if not HISTORY_PATH.exists():
        log.error("history.json nicht gefunden.")
        sys.exit(1)
    with open(HISTORY_PATH) as f:
        return json.load(f)


def save_history(history: dict) -> None:
    from modules.atomic_io import atomic_write_json
    atomic_write_json(HISTORY_PATH, history, indent=2)          # atomar (Crash-Recovery, Audit 2026-10-04)
    log.info("history.json aktualisiert.")


# ── Preis-Abruf: Aktienkurs ───────────────────────────────────────────────────

def get_current_price(ticker: str) -> float:
    """
    Aktueller Aktienkurs: Tradier Primary → yfinance Fallback.

    Tradier /v1/markets/quotes liefert Real-Time-Kurse ohne Delay.
    yfinance als Fallback wenn Tradier nicht erreichbar.
    """
    if _use_tradier():
        price = _tradier_stock_price(ticker)
        if price > 0:
            return price
        log.debug(f"[{ticker}] Tradier Aktienkurs fehlgeschlagen → yfinance")

    # yfinance Fallback
    try:
        info = yf.Ticker(ticker).info
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    except Exception:
        return 0.0


def _tradier_stock_price(ticker: str) -> float:
    """
    Aktienkurs via Tradier /v1/markets/quotes.

    Response-Struktur:
        {"quotes": {"quote": {"last": 150.25, "bid": ..., "ask": ...}}}
    """
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/quotes",
            params={"symbols": ticker, "greeks": "false"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data  = resp.json()
        quote = data.get("quotes", {}).get("quote", {})

        # Mehrere Symbole → Liste; einzelnes Symbol → Dict
        if isinstance(quote, list):
            quote = next((q for q in quote if q.get("symbol") == ticker), {})

        # "last" bevorzugt; Fallback auf Mid aus Bid/Ask
        last = quote.get("last")
        if last and float(last) > 0:
            return float(last)

        bid = float(quote.get("bid") or 0)
        ask = float(quote.get("ask") or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 4)

        return 0.0

    except Exception as e:
        log.debug(f"Tradier Aktienkurs [{ticker}]: {e}")
        return 0.0


# ── Preis-Abruf: Optionspreis ─────────────────────────────────────────────────

def get_option_quote(ticker: str, option: dict, strategy: str) -> dict | None:
    """Bid/Ask einer Options-Position: Tradier Primary -> yfinance Fallback.
    None = kein Quote gefunden. Der Gegentyp (Put statt Call) wird NUR bei
    fehlender Strategie-Info versucht -- vorher lieferte ein nicht gefundener
    Call still den Preis des PUTS gleichen Strikes (falsches Instrument)."""
    if not option:
        return None
    strike, expiry = option.get("strike"), option.get("expiry")
    if not strike or not expiry:
        return None
    option_type = _option_type_from_strategy(strategy)
    types = [option_type] + ([("put" if option_type == "call" else "call")] if not strategy else [])
    for t in types:
        q = _tradier_option_quote(ticker, strike, expiry, t) if _use_tradier() else None
        if q is None:
            q = _yfinance_option_quote(ticker, strike, expiry, t)
        if q is not None:
            return q
    return None


def get_current_option_price(ticker: str, option: dict, strategy: str, side: str = "sell") -> float:
    """REALISIERBARER Preis (Audit 2026-09-29): Verkauf zum Bid, Kauf zum Ask.
    Vorher Mid (bzw. Ask, wenn kein Bid) -> Exit ohne halben Spread, jedes
    Outcome im Median ~2,5 Prozentpunkte zu gut (Spread ~5 % des Ask).
    0.0 = kein Quote ODER Bid 0 (siehe compute_outcome zur Unterscheidung)."""
    q = get_option_quote(ticker, option, strategy)
    if q is None:
        return 0.0
    return float(q["bid"] if side == "sell" else q["ask"])


def _option_type_from_strategy(strategy: str) -> str:
    """
    Leitet "call" oder "put" aus der Trade-Strategie ab.

    "LONG_CALL", "BULL_CALL_SPREAD" → "call"
    "LONG_PUT",  "BEAR_PUT_SPREAD"  → "put"
    ""                              → "call" (Standard-Fallback; wird in
                                      _tradier_option_price auch als Put versucht)
    """
    s = strategy.upper()
    if "PUT" in s or "BEAR" in s:
        return "put"
    return "call"  # Default: Call (häufiger Fall)


def _tradier_option_quote(ticker: str, strike: float, expiry: str, option_type: str) -> dict | None:
    """Bid/Ask via Tradier /v1/markets/options/chains (Strike ± 0.01)."""
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/chains",
            params={"symbol": ticker, "expiration": expiry, "greeks": "false"},
            headers=_tradier_headers(), timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        options = (resp.json().get("options") or {}).get("option") or []
        if isinstance(options, dict):
            options = [options]
        for o in options:
            if o.get("option_type") != option_type:
                continue
            if abs(float(o.get("strike", 0)) - float(strike)) > 0.01:
                continue
            bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
            if bid > 0 or ask > 0:
                return {"bid": bid, "ask": ask, "source": "tradier"}
        return None
    except Exception as e:
        log.warning(f"Tradier Options-Chain [{ticker} {expiry}]: {e}")
        return None


def _yfinance_option_quote(ticker: str, strike: float, expiry: str, option_type: str) -> dict | None:
    """Bid/Ask via yfinance (Fallback)."""
    try:
        t = yf.Ticker(ticker)
        if expiry not in t.options:
            return None
        chain = t.option_chain(expiry)
        opts = chain.calls if option_type == "call" else chain.puts
        matches = opts[(opts["strike"] == strike) & ((opts["ask"] > 0) | (opts["bid"] > 0))]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {"bid": float(row["bid"] or 0), "ask": float(row["ask"] or 0), "source": "yfinance"}
    except Exception as e:
        log.warning(f"yfinance Options-Preis Fehler für {ticker}: {e}")
        return None


# ── Spread-Preis (beide Legs) ─────────────────────────────────────────────────

def _expired_spread_intrinsic(ticker: str, option: dict) -> float | None:
    """
    Intrinsic-Wert eines Bull Call Spreads am Verfallstag via yfinance-History.

    Returns None  → historische Daten nicht verfügbar (Outcome bleibt 0.0).
    Returns 0.0   → Spread verfallen wertlos (OTM). Outcome = -100%.
    Returns width → Spread voll im Geld. Outcome = Max-Gewinn.
    """
    try:
        expiry_str = option.get("expiry", "")
        long_k     = float(option.get("strike", 0))
        sl         = option.get("spread_leg") or {}
        short_k    = float(sl.get("strike", 0))
        is_put     = short_k < long_k      # Bear-Put-Spread: Long-Put-Strike > Short-Put-Strike
        width      = round(abs(short_k - long_k), 2) if long_k > 0 and short_k > 0 else 0

        if width <= 0 or long_k <= 0:
            return None

        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        end_str   = (expiry_dt + timedelta(days=4)).strftime("%Y-%m-%d")
        hist      = yf.Ticker(ticker).history(start=expiry_str, end=end_str, auto_adjust=True)
        if hist.empty:
            return None

        close = float(hist["Close"].iloc[0])  # Schlusskurs am Verfallstag

        if is_put:     # vorher nur Bull-Call -> verfallene Put-Spreads fielen still aus
            intrinsic = round(min(max(long_k - close, 0.0), width), 4)
        elif close <= long_k:
            intrinsic = 0.0
        elif close >= short_k:
            intrinsic = width
        else:
            intrinsic = round(close - long_k, 4)

        log.info(
            f"    [{ticker}] Expired Spread ({expiry_str}): "
            f"stock=${close:.2f} | long_k=${long_k} short_k=${short_k} "
            f"→ intrinsic=${intrinsic:.2f}"
        )
        return intrinsic

    except Exception as e:
        log.warning(f"    [{ticker}] Expired-Spread Fehler: {e}")
        return None


def get_current_spread_price(ticker: str, option: dict, strategy: str, detail: dict | None = None) -> float | None:
    """
    Aktueller AUSFÜHRBARER Net-Wert eines Spreads: long.bid − short.ask (= combo_bid).
    (Docstring bis 2026-10-09 sprach fälschlich von Mid – der Code rechnet executable.)
    detail: optional, erhält die Leg-Quotes (für modules/spread_execution, keine Zusatz-Calls).

    Returns None  → Preis nicht abrufbar (kein verwertbares Outcome).
    Returns float → Net-Wert inkl. 0.0 (Spread wertlos / vollständig verloren).

    Abgelaufene Kontrakte: Tradier liefert keine Chain mehr → Fallback auf
    yfinance-History für Intrinsic-Value-Berechnung am Verfallstag.
    """
    sl = option.get("spread_leg") or {}
    if not sl:
        log.warning(f"    [{ticker}] Spread: kein spread_leg in option-Dict")
        return None

    expiry    = option.get("expiry", "")
    long_str  = option.get("strike", "?")
    short_str = sl.get("strike", "?")

    # Abgelaufene Option → Intrinsic-Wert via yfinance statt Live-Chain
    try:
        if expiry and datetime.strptime(expiry, "%Y-%m-%d").date() < datetime.utcnow().date():
            log.info(f"    [{ticker}] Spread-Expiry {expiry} abgelaufen → Intrinsic-Berechnung")
            return _expired_spread_intrinsic(ticker, option)
    except ValueError:
        pass

    # Realisierbar schließen: Long-Leg zum Bid verkaufen, Short-Leg zum Ask zurückkaufen.
    long_q  = get_option_quote(ticker, option, strategy)
    short_q = get_option_quote(ticker, {"strike": sl.get("strike"), "expiry": expiry}, strategy)
    long_mid  = long_q["bid"] if long_q else 0.0
    short_mid = short_q["ask"] if short_q else 0.0

    if detail is not None:
        detail["spread_quotes"] = {"long": long_q, "short": short_q}
    if long_q is not None and short_q is not None and short_mid > 0:
        net = round(long_mid - short_mid, 4)
        log.info(
            f"    [{ticker}] Spread-Legs: "
            f"long(k={long_str})=${long_mid:.2f} | "
            f"short(k={short_str})=${short_mid:.2f} | net=${net:.2f}"
        )
        return max(net, 0.0)

    log.warning(
        f"    [{ticker}] Spread-Leg nicht abrufbar: "
        f"long(k={long_str})={long_mid:.2f} | "
        f"short(k={short_str})={short_mid:.2f} | "
        f"expiry={expiry}"
    )
    return None


# ── Outcome-Berechnung ────────────────────────────────────────────────────────

from modules.outcomes import RELIABLE_OUTCOME_METHODS  # noqa: E402


def compute_outcome(trade: dict, current_stock_price: float, meta: dict | None = None) -> float | None:
    """
    Berechnet Trade-Outcome (Return) für das RL-Training.
    Returns None wenn kein verwertbarer Preis ermittelbar ist (statt 0.0,
    das sonst als echtes Ergebnis ins Lern-System fließen würde).

    Prioritäten:
      Spread:       Net-Spread-Preis (long − short). Kein Stock-Fallback.
      Long Option:  Echter Options-Preis → Delta-Approx → Stock-Fallback.
      Unbekannt:    Stock-Return als letzter Ausweg.

    Entry-Debit:
      Explizit gespeichertes entry_debit hat Vorrang.
      Fallback: net_debit (Spread) oder ask (Long).

    Args:
        trade: Trade dict
        current_stock_price: Aktueller Aktienkurs
        meta: Optional dict to record which method computed the outcome.
              When provided, sets meta["method"] to one of:
              "spread_quote", "option_quote", "delta_approx",
              "delta_approx_clipped", "stock_fallback"
    """
    ticker    = trade["ticker"]
    option    = trade.get("option") or {}
    strategy  = trade.get("strategy", "")
    sim       = trade.get("simulation") or {}
    is_spread = "SPREAD" in strategy

    # ── Entry-Debit ermitteln ────────────────────────────────────────────────
    entry_debit = float(trade.get("entry_debit") or 0)
    if entry_debit <= 0:
        if is_spread:
            sl  = option.get("spread_leg") or {}
            nd  = option.get("net_debit")
            if nd:
                entry_debit = float(nd)
            else:
                la = float(option.get("ask", 0))
                sb = float(sl.get("bid", 0))
                entry_debit = round(la - sb, 2) if la > 0 and sb > 0 else la
        else:
            entry_debit = float(option.get("ask", 0)) or float(option.get("last", 0))

    # ── Spread: beide Legs repricing, kein Stock-Fallback ───────────────────
    if is_spread:
        if entry_debit <= 0:
            log.warning(f"    [{ticker}] Spread ohne Entry-Debit → Outcome nicht verwertbar")
            return None
        current_spread = get_current_spread_price(ticker, option, strategy, detail=meta)
        if current_spread is None:
            # Preis wirklich nicht ermittelbar — nicht als 0.0 ins Training
            log.warning(f"    [{ticker}] Spread-Preis nicht abrufbar → Outcome nicht verwertbar")
            return None
        # current_spread == 0.0 ist valide: Spread verfallen wertlos → -100%
        result = (current_spread - entry_debit) / entry_debit
        if meta is not None:
            meta["method"] = "spread_quote"
        log.info(
            f"    Spread-P&L: entry=${entry_debit:.2f} → "
            f"current=${current_spread:.2f} = {result:+.2%}"
        )
        return result

    # ── Long Option: echter Preis → Delta-Approx ────────────────────────────
    entry_stock = float(sim.get("current_price", 0))
    stock_return = 0.0
    if entry_stock > 0 and current_stock_price > 0:
        stock_return = (current_stock_price - entry_stock) / entry_stock

    if entry_debit > 0:
        current_option = get_current_option_price(ticker, option, strategy)
        if current_option <= 0:
            q = get_option_quote(ticker, option, strategy)
            if q is not None and q.get("ask", 0) > 0 and q.get("bid", 0) <= 0:
                # Quote vorhanden, aber kein Bid: realisierbar 0 -> -100 %,
                # NICHT die (optimistische) Delta-Näherung.
                if meta is not None:
                    meta["method"] = "option_quote_bid_zero"
                log.info(f"    Options-P&L: entry=${entry_debit:.2f} → Bid 0 (realisierbar) = -100.00%")
                return -1.0
        if current_option > 0:
            result = (current_option - entry_debit) / entry_debit
            if meta is not None:
                meta["method"] = "option_quote"
            log.info(
                f"    Options-P&L: entry=${entry_debit:.2f} → "
                f"current=${current_option:.2f} = {result:+.2%}"
            )
            return result
        if entry_stock > 0:
            leverage = (entry_stock / entry_debit) * 0.65
            # Put: Aktie fällt -> Put gewinnt. Vorher OHNE Vorzeichenwechsel ->
            # jeder Long-Put-Outcome der Näherung war invertiert (Audit 2026-09-29:
            # AMAT +43.6 % Aktie -> "+500 %" Put; META -5.9 % -> "-100 %").
            direction = -1.0 if _option_type_from_strategy(strategy) == "put" else 1.0
            result   = direction * stock_return * leverage
            unclipped_result = result
            result   = max(-1.0, min(result, 5.0))   # Options: Max-Verlust=-100%, Cap=+500%
            # Record whether clipping was applied
            if meta is not None:
                if abs(unclipped_result - result) > 1e-6:
                    meta["method"] = "delta_approx_clipped"
                else:
                    meta["method"] = "delta_approx"
            log.info(f"    Delta-approx: {stock_return:+.2%} × {leverage:.1f} = {result:+.2%}")
            return result

    # ── Letzter Fallback: Stock-Return (nur wenn kein Debit bekannt) ─────────
    if meta is not None:
        meta["method"] = "stock_fallback"
    log.info(f"    Stock-Return Fallback: {stock_return:+.2%}")
    return stock_return


# ── Regelbasierte Exits ───────────────────────────────────────────────────────
# Die Empfehlungs-Mails enthalten TP/SL/Time-Exit-Regeln (reporter.py).
# Der Lern-Loop muss dieselben Regeln anwenden — sonst lernt das System
# aus "Preis nach 45 Tagen" statt aus der tatsächlich empfohlenen Strategie.

def check_exit_rules(trade: dict, outcome: float, today: datetime) -> str | None:
    """
    Prüft TP/SL/Time-Exit für einen aktiven Trade.
    Returns Exit-Grund ("take_profit"/"stop_loss"/"time_exit") oder None.
    """
    strategy  = trade.get("strategy", "")
    option    = trade.get("option") or {}
    is_spread = "SPREAD" in strategy

    # Schwellen analog reporter.compute_exit_rules
    if is_spread:
        sl_threshold = -0.50
        entry = float(trade.get("entry_debit") or option.get("net_debit") or 0)
        long_k  = float(option.get("strike") or 0)
        short_k = float((option.get("spread_leg") or {}).get("strike") or 0)
        width   = short_k - long_k if short_k > long_k > 0 else 0
        if width > 0 and entry > 0:
            tp_threshold = (width - entry) * 0.70 / entry   # 70% des Max-Gewinns
        else:
            tp_threshold = 0.50
    else:
        sl_threshold = -0.45
        tp_threshold = 0.50

    if outcome <= sl_threshold:
        return "stop_loss"
    if outcome >= tp_threshold:
        return "take_profit"

    # Time-Exit: 50% der Laufzeit verstrichen und Gewinn < +20%
    expiry_str = option.get("expiry", "")
    try:
        expiry    = datetime.strptime(expiry_str, "%Y-%m-%d")
        entry_dt  = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        dte_total = (expiry - entry_dt).days
        remaining = (expiry - today).days
        if dte_total > 0 and remaining <= dte_total * 0.5 and outcome < 0.20:
            return "time_exit"
    except (ValueError, KeyError):
        pass

    return None


def _spread_execution_monitor(trade: dict, meta: dict, underlying: float, today: datetime) -> dict | None:
    """Nur Beobachtung + Ledger; jede Ausnahme bleibt folgenlos für Produktion."""
    if "SPREAD" not in str(trade.get("strategy") or "") or not meta.get("spread_quotes"):
        return None
    try:
        from modules import spread_execution as se
        q = meta["spread_quotes"]
        obs = se.monitor(trade, q.get("long"), q.get("short"), underlying)
        hist = (trade.setdefault("spread_monitor", []) + [obs])[-10:]
        trade["spread_monitor"] = hist
        trade["spread_stop_shadow"] = se.stop_shadow(hist)
        se.record("monitor", trade["ticker"], today.strftime("%Y-%m-%d"),
                  {"entry_date": trade.get("entry_date"), **obs, "stop_shadow": trade["spread_stop_shadow"]})
        return obs
    except Exception as e:  # noqa: BLE001
        log.warning(f"  [{trade.get('ticker')}] Spread-Execution-Monitoring Fehler (ignoriert): {e}")
        return None


def _spread_execution_exit(trade: dict, obs: dict | None, today: datetime) -> None:
    if "SPREAD" not in str(trade.get("strategy") or ""):
        return
    try:
        from modules import spread_execution as se
        entry = trade.get("spread_execution_entry") or {}
        e_price = trade.get("entry_debit")
        mid_in, mid_out = entry.get("combo_mid"), (obs or {}).get("combo_mid")
        bid_out = (obs or {}).get("combo_bid")
        exit_rec = {"combo_bid": bid_out, "combo_ask": (obs or {}).get("combo_ask"), "combo_mid": mid_out,
                    "actual_exit_fill": None, "assumed_exit_fill": bid_out, "exit_reason": trade.get("close_reason"),
                    "quote_quality": (obs or {}).get("quote_quality"),
                    "slippage_vs_mid": round((mid_out - bid_out) / mid_out, 4) if mid_out and bid_out is not None else None,
                    "realized_roundtrip_cost": (round(((e_price - mid_in) + (mid_out - bid_out)) / e_price, 4)
                                                if e_price and mid_in is not None and mid_out is not None and bid_out is not None
                                                else None),
                    "stop_shadow": trade.get("spread_stop_shadow")}
        trade["spread_execution_exit"] = exit_rec
        se.record("exit", trade["ticker"], today.strftime("%Y-%m-%d"), {"entry_date": trade.get("entry_date"), **exit_rec})
    except Exception as e:  # noqa: BLE001
        log.warning(f"  [{trade.get('ticker')}] Spread-Execution-Exit Fehler (ignoriert): {e}")


def evaluate_shadow_trades(history: dict, today: datetime) -> None:
    """
    Bewertet Schatten-Trades (von Gates verworfene Signale) nach Ablauf
    der Haltedauer — validiert kostenlos, ob die Gates Gewinner wegfiltern.
    """
    shadows = history.get("shadow_trades", [])
    for st in shadows:
        if st.get("outcome") is not None:
            continue
        try:
            entry_dt = datetime.strptime(st["entry_date"][:10], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        if (today - entry_dt).days < cfg.learning.close_after_days:
            continue
        current = get_current_price(st["ticker"])
        if current <= 0:
            continue
        _m = {}
        outcome = compute_outcome(st, current, _m)
        if outcome is None:
            continue
        st["outcome"]    = round(outcome, 4)
        st["outcome_method"] = _m.get("method", "unknown")
        st["close_date"] = today.strftime("%Y-%m-%d")
        try:
            update_feature_stats_external(history, st)
        except Exception as e:
            log.warning(f"  [SHADOW {st['ticker']}] feature_stats_external-Update Fehler (ignoriert): {e}")
        log.info(f"  [SHADOW {st['ticker']}] ({st.get('reject_reason','?')}) Outcome={outcome:+.2%}")
    # Liste begrenzen: nur die letzten 300 behalten. Ältere Einträge werden NICHT
    # verworfen, sondern append-only archiviert — bewertete Schatten-Outcomes sind
    # die einzige Evidenz für Gate-Audits (ROI-Gate, Final-MC) und dürfen nie verloren gehen.
    if len(shadows) > 300:
        archive_shadow_trades(shadows[:-300])
        history["shadow_trades"] = shadows[-300:]


SHADOW_ARCHIVE = Path("outputs/shadow_trades_archive.jsonl")


def archive_shadow_trades(dropped: list[dict], path: Path | None = None) -> None:
    """Hängt aus history.json verdrängte Schatten-Trades an das Archiv an (append-only)."""
    path = path or SHADOW_ARCHIVE
    if not dropped:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for t in dropped:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    log.info(f"  {len(dropped)} Schatten-Trades nach {path} archiviert")


# ── Counterfactual: von der Intelligence blockierte Champion-Trades ──────────
# Gleicher Lebenszyklus wie echte Trades (Preisquelle, Exit-Regeln TP/SL/Time,
# Haltedauer, Outcome-Methode), aber OHNE Lern-Updates (keine Bins, kein RL,
# kein closed_trades-Eintrag). Nur so ist "blockiert vs. durchgelassen" fair.

def advance_counterfactual_trades(history: dict, today: datetime, price_fn=None) -> int:
    price_fn = price_fn or get_current_price
    open_cf = history.get("counterfactual_trades") or []
    still, closed = [], 0
    for t in open_cf:
        try:
            entry_dt = datetime.strptime(t["entry_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        current = price_fn(t["ticker"])
        if current <= 0:
            still.append(t)
            continue
        meta = {}
        outcome = compute_outcome(t, current, meta)
        if outcome is None:
            still.append(t)
            continue
        t["peak_return"] = round(max(float(t.get("peak_return") or outcome), outcome), 4)
        reason = check_exit_rules(t, outcome, today)
        if reason or (today - entry_dt).days >= cfg.learning.close_after_days:
            t.update(outcome=round(outcome, 4), close_date=today.strftime("%Y-%m-%d"), close_price=current,
                     close_reason=reason or "max_holding_period", outcome_method=meta.get("method", "unknown"))
            history.setdefault("counterfactual_closed", []).append(t)
            closed += 1
        else:
            t["current_return"] = round(outcome, 4)
            still.append(t)
    history["counterfactual_trades"] = still
    return closed


# ── Trailing-Stop-Paralleltest ────────────────────────────────────────────────
# Der harte Take-Profit bei +50% kappt den rechten Tail, von dem die
# Asymmetrie-These lebt (TP-Exits: Ø +120% — die Gewinner laufen weit über die
# Schwelle hinaus). Bevor die Exit-Regel geändert wird: jeden echten TP-Exit
# virtuell weiterführen und erst bei Rückfall auf TRAIL_FACTOR × Peak oder am
# Verfall schließen. trailing_outcome vs. tp_outcome liefert die Datenbasis
# für den nächsten Tuning-Slot. Kein Geld im Spiel, keine Regel verändert.

TRAIL_FACTOR = 0.65


def register_trailing_sim(history: dict, trade: dict, today: datetime) -> None:
    """Startet die virtuelle Weiterführung eines per Take-Profit geschlossenen Trades."""
    sims = history.setdefault("trailing_sim", [])
    if any(s.get("ticker") == trade["ticker"] and s.get("entry_date") == trade.get("entry_date")
           for s in sims):
        return
    tp_outcome = float(trade.get("outcome") or 0)
    sims.append({
        "ticker":           trade["ticker"],
        "strategy":         trade.get("strategy", ""),
        "option":           trade.get("option") or {},
        "simulation":       trade.get("simulation") or {},
        "entry_debit":      trade.get("entry_debit"),
        "entry_date":       trade.get("entry_date"),
        "tp_date":          today.strftime("%Y-%m-%d"),
        "tp_outcome":       tp_outcome,
        "peak":             max(float(trade.get("peak_return") or 0), tp_outcome),
        "trailing_outcome": None,
    })
    log.info(
        f"  [TRAIL {trade['ticker']}] TP-Exit {tp_outcome:+.2%} → "
        f"virtuelle Weiterführung gestartet (Exit bei {TRAIL_FACTOR:.0%} des Peaks)"
    )


def evaluate_trailing_sims(history: dict, today: datetime) -> None:
    """Führt offene Trailing-Simulationen fort und schließt sie regelbasiert."""
    for sim in history.get("trailing_sim", []):
        if sim.get("trailing_outcome") is not None:
            continue
        current = get_current_price(sim["ticker"])
        if current <= 0:
            continue
        outcome = compute_outcome(sim, current)
        if outcome is None:
            continue
        peak = max(float(sim.get("peak") or 0), outcome)
        sim["peak"]           = round(peak, 4)
        sim["current_return"] = round(outcome, 4)

        expiry_str = (sim.get("option") or {}).get("expiry", "")
        try:
            expired = datetime.strptime(expiry_str, "%Y-%m-%d") <= today
        except ValueError:
            expired = False

        if outcome <= peak * TRAIL_FACTOR or expired:
            sim["trailing_outcome"] = round(outcome, 4)
            sim["close_date"]       = today.strftime("%Y-%m-%d")
            sim["close_reason"]     = "expiry" if expired else "trail_stop"
            delta = outcome - float(sim.get("tp_outcome") or 0)
            log.info(
                f"  [TRAIL {sim['ticker']}] {sim['close_reason']}: "
                f"trailing={outcome:+.2%} vs. TP={float(sim.get('tp_outcome') or 0):+.2%} "
                f"(Δ={delta:+.2%})"
            )


# ── Multi-Variant Exit-Counterfactual (Erweiterung des Trailing-Paralleltests) ──
# Verallgemeinert register_trailing_sim/evaluate_trailing_sims (oben) auf sechs
# Exit-Varianten, die für JEDEN aktiven Trade parallel mitgeführt werden — nicht
# nur für per Take-Profit geschlossene. Reine Simulation, siehe modules/exit_sim.py
# für die Varianten-Logik und den Hinweis zur (groben) Pfad-Granularität. Ändert
# nichts an der echten Exit-Logik/trailing_sim — beide bleiben unverändert aktiv.

def evaluate_exit_sims(history: dict, today: datetime) -> None:
    """Aktualisiert alle offenen exit_sim-Einträge mit dem aktuellen Options-Return."""
    for entry in history.get("exit_sim", []):
        if entry.get("closed"):
            continue
        try:
            current = get_current_price(entry["ticker"])
            if current <= 0:
                continue
            outcome = compute_outcome(entry, current)
            if outcome is None:
                continue
            update_exit_sim_entry(entry, outcome, today)
        except Exception as e:
            log.warning(f"  [EXIT-SIM {entry.get('ticker', '?')}] Update-Fehler: {e}")
    trim_exit_sim(history)


# ── Bin-Updates (Legacy, für Backward-Kompatibilität) ─────────────────────────

def update_bin(stats_dict: dict, feature: str, bin_label: str, outcome: float) -> None:
    bin_data = stats_dict.setdefault(feature, {}).setdefault(
        bin_label, {"count": 0, "avg_return": 0.0}
    )
    old_avg = bin_data["avg_return"]
    old_cnt = bin_data["count"]
    new_cnt = old_cnt + 1
    new_avg = (old_avg * old_cnt + outcome) / new_cnt
    bin_data["count"]      = new_cnt
    bin_data["avg_return"] = round(new_avg, 6)


# ── Externer Kontext: Observational Feature-Stats (NIEMALS ins Lern-System) ──
# history["feature_stats_external"] ist PURE Observability für den Monats-Report
# (siehe modules/external/research.py + monthly_report.py Abschnitt "Externer
# Kontext (SHADOW)"). Es wird AUSSCHLIESSLICH aus dem beim Trade-Entry
# EINGEFRORENEN trade["external_context_entry"] gespeist — nie aus einer
# Neuberechnung/einem Live-Snapshot zum Zeitpunkt des Closes (der externe
# Snapshot hätte sich bis dahin revidiert/weiterentwickelt; das würde die
# Point-in-Time-Garantie des externen Kontexts verletzen).
#
# HARTE GARANTIE: Diese Stats fließen NIE in compute_pearson_weights(),
# history["model_weights"], QuasiML-Scoring oder den RL-Agenten (Trainings-
# Environment/Reward) ein. Sie sind ausschließlich deskriptiv für Menschen.
# (Siehe tests/test_feature_stats_external.py für den entsprechenden Guard-Test.)
#
# Streaming/Approximation: count und mean sind über ALLE Outcomes seit je exakt
# (Welford-Online-Update). Da eine unbegrenzt wachsende Rohwerte-Liste den
# history.json-Speicherbedarf unkontrolliert wachsen ließe, werden für
# Median/Std nur die letzten EXTERNAL_BUCKET_OUTCOMES_CAP Outcomes je Bucket
# behalten — Median/Std sind daher ab mehr als CAP Outcomes eine (im
# Zweifel leicht verzerrte) Streaming-Approximation über das jüngste Fenster,
# nicht über die komplette Historie.

EXTERNAL_BUCKET_OUTCOMES_CAP = 200


def _shipping_breadth_bucket(breadth) -> str | None:
    """Methodologische Bins für shipping_negative_breadth (0..1): <0.33,
    0.33-0.66, >0.66. None wenn kein Wert vorliegt."""
    if not isinstance(breadth, (int, float)):
        return None
    if breadth < 0.33:
        return "<0.33"
    if breadth < 0.66:
        return "0.33-0.66"
    return ">0.66"


def _divergence_bucket(div_z) -> str | None:
    """Grobe Magnitude-Bins für road_shipping_divergence_z (|z|): low (<0.5),
    medium (<1.5), high (sonst). None wenn kein Wert vorliegt."""
    if not isinstance(div_z, (int, float)):
        return None
    mag = abs(div_z)
    if mag < 0.5:
        return "low"
    if mag < 1.5:
        return "medium"
    return "high"


def external_feature_buckets(ext: dict) -> dict:
    """
    Extrahiert die methodologischen Buckets aus einem EINGEFRORENEN
    external_context_entry-Dict (row["external"]-Format, siehe
    modules/candidate_ledger.py-Docstring / modules/external/*). Tolerant
    gegenüber fehlenden Feldern/Keys — liefert None für nicht ableitbare
    Buckets, statt zu werfen.

    Bucket-Dimensionen (siehe Spezifikation):
      freight_state, maritime_state, weather_operational_risk,
      external_relation, road_shipping_agreement, shipping_breadth_bucket,
      divergence_bucket.
    """
    if not isinstance(ext, dict):
        ext = {}
    states     = ext.get("states") or {}
    primitives = ext.get("primitives") or {}
    relation   = ext.get("relation") or {}

    wdi = primitives.get("weather_disruption_index")
    if primitives.get("active_tropical_system") or (isinstance(wdi, (int, float)) and wdi >= 1.5):
        weather_bucket = "elevated"
    elif wdi is not None or "active_tropical_system" in primitives:
        weather_bucket = "normal"
    else:
        weather_bucket = None

    return {
        "freight_state":            states.get("global_freight_state"),
        "maritime_state":           states.get("global_maritime_state"),
        "weather_operational_risk": weather_bucket,
        "external_relation":        relation.get("relation"),
        "road_shipping_agreement":  primitives.get("road_shipping_agreement"),
        "shipping_breadth_bucket":  _shipping_breadth_bucket(primitives.get("shipping_negative_breadth")),
        "divergence_bucket":        _divergence_bucket(primitives.get("road_shipping_divergence_z")),
    }


def _update_external_bucket(dim_bucket: dict, key: str, outcome: float, strat_ret: float | None) -> None:
    b = dim_bucket.setdefault(str(key), {
        "count": 0, "mean": 0.0, "_m2": 0.0, "wins": 0,
        "outcomes": [], "strat_ret_sum": 0.0, "strat_ret_count": 0,
    })
    b["count"] += 1
    delta = outcome - b["mean"]
    b["mean"] += delta / b["count"]
    b["_m2"] += delta * (outcome - b["mean"])
    if outcome > 0:
        b["wins"] += 1
    b["outcomes"].append(round(outcome, 6))
    if len(b["outcomes"]) > EXTERNAL_BUCKET_OUTCOMES_CAP:
        del b["outcomes"][:-EXTERNAL_BUCKET_OUTCOMES_CAP]
    if strat_ret is not None:
        b["strat_ret_sum"] += strat_ret
        b["strat_ret_count"] += 1


def update_feature_stats_external(history: dict, trade: dict) -> None:
    """
    Observational Update von history["feature_stats_external"] beim
    Trade-Close — AUSSCHLIESSLICH aus trade["external_context_entry"]
    (beim Entry eingefroren, NIE neu berechnet). No-op wenn kein
    external_context_entry vorhanden ist (externes Modul deaktiviert / Trade
    älter als die External-Integration) oder kein Outcome vorliegt — bricht
    den Feedback-Loop nie (kein Raise).

    WICHTIG: Siehe Modul-Docstring oben — diese Funktion füttert NIE
    compute_pearson_weights()/history["model_weights"]/QuasiML/RL.
    """
    ext = trade.get("external_context_entry")
    if not isinstance(ext, dict) or not ext:
        return
    outcome = trade.get("outcome")
    if outcome is None:
        return
    try:
        outcome = float(outcome)
    except (TypeError, ValueError):
        return

    strat_ret = trade.get("real_strat_ret")
    strat_ret = float(strat_ret) if isinstance(strat_ret, (int, float)) else outcome

    buckets = external_feature_buckets(ext)
    stats_all = history.setdefault("feature_stats_external", {})
    for dim, key in buckets.items():
        if key is None:
            continue
        dim_bucket = stats_all.setdefault(dim, {})
        _update_external_bucket(dim_bucket, key, outcome, strat_ret)


# ── RL-Training ───────────────────────────────────────────────────────────────

def maybe_notify_rl_arming(history: dict) -> None:
    """
    Sendet eine EINMALIGE Email, sobald genug closed_trades unter den neuen Regeln
    (entry_date >= rl.arm_since) vorliegen, um den RL-Agenten scharfzustellen.

    Relevant nur solange das RL-Veto deaktiviert ist (Option A). Der Flag
    history['rl_arm_notified'] verhindert wiederholten Versand.
    """
    rl_cfg = cfg.rl
    if rl_cfg.get("veto_enabled", True):
        return  # RL bereits scharf → nichts zu tun
    if history.get("rl_arm_notified"):
        return  # bereits benachrichtigt

    threshold = int(rl_cfg.get("arm_threshold", 30))
    since     = str(rl_cfg.get("arm_since", "2026-06-11"))
    closed    = history.get("closed_trades", [])

    relevant = [
        t for t in closed
        if t.get("outcome") is not None and str(t.get("entry_date", ""))[:10] >= since
    ]
    n = len(relevant)
    if n < threshold:
        log.info(f"RL-Arming: {n}/{threshold} closed_trades seit {since} — noch nicht erreicht.")
        return

    wins     = sum(1 for t in relevant if t["outcome"] > 0)
    win_rate = wins / n if n else 0.0
    try:
        from modules.email_reporter import send_rl_arming_email
        send_rl_arming_email(n, threshold, win_rate, since)
        history["rl_arm_notified"] = True
        log.info(f"RL-Arming-Email gesendet: {n} Trades seit {since}, Win-Rate {win_rate:.0%}.")
    except Exception as e:
        log.error(f"RL-Arming-Email-Fehler: {e}")


def retrain_rl_agent(history: dict) -> None:
    """
    Trainiert den PPO-Agenten inkrementell auf allen closed_trades.

    Continual Learning: 2.000 Steps pro Feedback-Lauf (~5s auf CPU).
    GitHub-Actions-tauglich: Modell als .zip committed, nächster Run nutzt es.
    """
    try:
        from modules.rl_agent import train_agent
    except ImportError as e:
        log.warning(f"RL-Agent nicht importierbar: {e} → Training übersprungen")
        return

    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        log.info(
            f"Nur {len(closed)} closed_trades → RL-Training übersprungen "
            f"(Minimum: 5)."
        )
        return

    log.info(f"Starte RL-Nachtraining auf {len(closed)} closed_trades...")
    success = train_agent(
        history         = history,
        total_timesteps = 2_000,
        force_retrain   = False,
    )

    if success:
        log.info("RL-Agent erfolgreich nachtrainiert.")
    else:
        log.warning("RL-Nachtraining fehlgeschlagen (nicht kritisch).")


# ── Pearson-Gewichte (Legacy-Support) ────────────────────────────────────────

def compute_pearson_weights(history: dict) -> dict:
    from modules.outcomes import is_reliable_outcome
    closed = [t for t in history.get("closed_trades", []) if is_reliable_outcome(t)]
    if len(closed) < 5:
        return history.get("model_weights", {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20})

    outcomes, impacts, mismatches, drifts, entry_days = [], [], [], [], set()
    for t in closed:
        outcome = t.get("outcome")
        if outcome is None:
            continue
        feat = t.get("features", {})
        outcomes.append(outcome)
        entry_days.add(str(t.get("entry_date", ""))[:10])
        impacts.append(_bin_to_num("impact",    feat.get("bin_impact",    "mid")))
        mismatches.append(_bin_to_num("mismatch", feat.get("bin_mismatch",  "good")))
        drifts.append(_bin_to_num("eps_drift", feat.get("bin_eps_drift", "noise")))

    if len(outcomes) < 5:
        return history.get("model_weights", {})

    outcomes_arr = np.array(outcomes)
    correlations = {}
    for name, arr in [("impact", np.array(impacts)),
                       ("mismatch", np.array(mismatches)),
                       ("eps_drift", np.array(drifts))]:
        # Konstante Spalte (z.B. alle bin_eps_drift="noise") → pearsonr=NaN
        if np.std(arr) == 0 or np.std(outcomes_arr) == 0:
            r = 0.0
        else:
            r, _ = stats.pearsonr(arr, outcomes_arr)
        if not np.isfinite(r):
            r = 0.0
        # Sicherheitsbremse (Audit 2026-09-29): nur statistisch belastbare
        # positive Korrelationen zählen. Vorher bekam ein Feature mit zufällig
        # r=+0.02 das Zielgewicht 100 % und setzte sich bei 2 Läufen/Werktag
        # mit Lernrate 0.05 binnen Wochen durch (Overfitting auf Rauschen).
        # Gleiche Regel wie modules/factor_monitor.py: >= MIN_EFF_N
        # unabhängige Entry-Tage UND 90%-KI (Fisher-z) oberhalb 0.
        from modules.factor_monitor import MIN_EFF_N, fisher_ci
        lo, _hi = fisher_ci(float(r), len(entry_days))
        if len(entry_days) < MIN_EFF_N or lo is None or lo <= 0:
            r = 0.0
        correlations[name] = max(r, 0)

    total = sum(correlations.values()) or 1.0
    old_w    = history.get("model_weights", {})
    defaults = {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20}
    new_w = {}
    for feat, corr in correlations.items():
        raw_new = corr / total
        old     = old_w.get(feat, defaults.get(feat, 1/3))
        if not isinstance(old, (int, float)) or not np.isfinite(old):
            old = defaults.get(feat, 1/3)
        new_w[feat] = round(old + cfg.learning.learning_rate * (raw_new - old), 4)

    total_w = sum(new_w.values())
    return {k: round(v / total_w, 4) for k, v in new_w.items()}


def _bin_to_num(feature: str, bin_label: str) -> float:
    mapping = {
        "impact":    {"low": 0.0, "mid": 0.5, "high": 1.0},
        "mismatch":  {"weak": 0.0, "good": 0.5, "strong": 1.0},
        "eps_drift": {"noise": 0.0, "relevant": 0.5, "massive": 1.0},
    }
    return mapping.get(feature, {}).get(bin_label, 0.5)


# ── Haupt-Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Feedback-Loop v5.0 gestartet ===")
    log.info(f"Tradier: {'aktiv' if _use_tradier() else 'KEIN KEY → yfinance Fallback'}")

    history      = load_history()
    today        = datetime.utcnow()
    active       = history.get("active_trades", [])
    still_active = []
    newly_closed = 0

    exit_alerts: list[dict] = []

    for trade in active:
        ticker     = trade["ticker"]
        entry_date = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        age_days   = (today - entry_date).days

        # exit_sim: Multi-Variant-Counterfactual anlegen (idempotent, rein simulativ,
        # läuft unabhängig vom echten Trade-Close weiter — siehe evaluate_exit_sims).
        try:
            register_exit_sim(history, trade, today)
        except Exception as e:
            log.warning(f"  [EXIT-SIM {ticker}] Register-Fehler: {e}")

        current = get_current_price(ticker)
        if current <= 0:
            still_active.append(trade)
            continue

        meta = {}
        outcome = compute_outcome(trade, current, meta)
        if outcome is None:
            log.warning(f"  [{ticker}] Outcome nicht ermittelbar → bleibt aktiv, kein Lern-Update")
            still_active.append(trade)
            continue
        log.info(f"  [{ticker}] Alter={age_days}d Outcome={outcome:+.2%}")

        # Peak-Return mitschreiben (Grundlage für den Trailing-Paralleltest)
        trade["peak_return"] = round(max(float(trade.get("peak_return") or outcome), outcome), 4)

        # Spread-Execution-Monitoring (SHADOW, 2026-10-09): Fair Value vs. Executable vs. Underlying.
        # Der Produktions-Stop unten bleibt unverändert (executable, -50 %).
        _spread_obs = _spread_execution_monitor(trade, meta, current, today)

        # ── Regelbasierter Exit (TP/SL/Time-Exit) — auch für junge Trades ────
        exit_reason = check_exit_rules(trade, outcome, today)
        if exit_reason:
            exit_alerts.append({
                "ticker":   ticker,
                "strategy": trade.get("strategy", ""),
                "reason":   exit_reason,
                "outcome":  outcome,
                "age_days": age_days,
                "option":   trade.get("option") or {},
            })

        if exit_reason or age_days >= cfg.learning.close_after_days:
            # Bin-Updates NUR beim Close — sonst wird derselbe Trade bei jedem
            # Feedback-Lauf erneut gezählt und verzerrt die Lernstatistik massiv.
            # Lernen NUR aus verlässlichen Outcomes (echte Options-/Spread-Quotes).
            # Näherungen (Delta-Approx, gekappt bei +500 %, Aktien-Fallback) verzerren
            # Bins/Gewichte; sie werden geschlossen und gekennzeichnet, aber nicht gelernt.
            reliable = meta.get("method") in RELIABLE_OUTCOME_METHODS
            trade["outcome_reliable"] = reliable
            feat = trade.get("features", {})
            for f_name, bin_key in [("impact",    "bin_impact"),
                                      ("mismatch",  "bin_mismatch"),
                                      ("eps_drift", "bin_eps_drift")]:
                bin_label = feat.get(bin_key)
                if bin_label and reliable:
                    update_bin(history["feature_stats"], f_name, bin_label, outcome)

            trade["outcome"]        = round(outcome, 4)
            trade["close_date"]     = today.strftime("%Y-%m-%d")
            trade["close_price"]    = current
            trade["close_reason"]   = exit_reason or "max_holding_period"
            trade["outcome_method"] = meta.get("method", "unknown")
            _spread_execution_exit(trade, _spread_obs, today)
            history.setdefault("closed_trades", []).append(trade)
            if reliable:
                try:
                    update_feature_stats_external(history, trade)
                except Exception as e:
                    log.warning(f"  [{ticker}] feature_stats_external-Update Fehler (ignoriert): {e}")
            log.info(
                f"  [{ticker}] Trade abgeschlossen "
                f"({trade['close_reason']}, Return={outcome:+.2%})"
            )
            newly_closed += 1
            if trade["close_reason"] == "take_profit":
                register_trailing_sim(history, trade, today)
        else:
            trade["last_price"]     = current
            trade["current_return"] = round(outcome, 4)
            still_active.append(trade)

    history["active_trades"] = still_active
    history["model_weights"] = compute_pearson_weights(history)

    # Schatten-Trades bewerten (Gate-Validierung, kein Geld im Spiel)
    evaluate_shadow_trades(history, today)
    # UNIVERSE_V2-Kandidaten: theoretische + realistische Netto-Outcomes (append-only, eigene Datei)
    try:
        from modules import universe_v2_ledger
        _n_v2 = universe_v2_ledger.resolve_outcomes(today=today.date())
        if _n_v2:
            log.info(f"  UNIVERSE_V2-Ledger: {_n_v2} Horizont-Outcome(s) aufgelöst")
    except Exception as e:  # noqa: BLE001 – Messung darf den Feedback-Lauf nie brechen
        log.warning(f"UNIVERSE_V2-Outcomes Fehler (ignoriert): {e}")
    # Final-MC-Population: fällige 20/45/60-Tage-Outcomes + MFE/MAE (append-only, eigene Datei)
    try:
        from modules import final_mc_ledger
        _n_fm = final_mc_ledger.resolve_outcomes(today=today.date())
        if _n_fm:
            log.info(f"  Final-MC-Ledger: {_n_fm} Horizont-Outcome(s) aufgelöst")
    except Exception as e:  # noqa: BLE001 – Messung darf den Feedback-Lauf nie brechen
        log.warning(f"Final-MC-Outcomes Fehler (ignoriert): {e}")

    # Von der Intelligence blockierte Champion-Trades counterfactual fortführen
    try:
        n_cf = advance_counterfactual_trades(history, today)
        if n_cf:
            log.info(f"  {n_cf} counterfactual (blockierte) Trade(s) geschlossen")
    except Exception as e:  # noqa: BLE001 – darf den Lern-Loop nie brechen
        log.error(f"Counterfactual-Fortführung Fehler: {e}")

    # Trailing-Paralleltest fortführen (virtuelle TP-Weiterführungen)
    evaluate_trailing_sims(history, today)

    # Multi-Variant Exit-Counterfactual fortführen (pure Simulation, kein Trade-Impact)
    try:
        evaluate_exit_sims(history, today)
        summary = summarize_exit_sim(history)
        for key, stats_row in summary.items():
            if "note" in stats_row:
                log.info(f"  [EXIT-SIM] {key}: {stats_row['note']}")
            else:
                log.info(
                    f"  [EXIT-SIM] {key}: n={stats_row['n']} "
                    f"mean={stats_row['mean']:+.2%} median={stats_row['median']:+.2%} "
                    f"win_rate={stats_row['win_rate']:.0%} "
                    f"total_loss_rate={stats_row['total_loss_rate']:.0%}"
                )
    except Exception as e:
        log.error(f"Exit-Sim-Fehler: {e}")

    # RL-Scharfstellung: Email sobald genug closed_trades unter neuen Regeln vorliegen
    maybe_notify_rl_arming(history)

    save_history(history)

    # Exit-Alarme als E-Mail (TP/SL/Time-Exit erreicht → Handlungsaufforderung)
    if exit_alerts:
        try:
            from modules.email_reporter import send_exit_alert_email
            send_exit_alert_email(exit_alerts, today.strftime("%Y-%m-%d"))
        except Exception as e:
            log.error(f"Exit-Alert-Email-Fehler: {e}")

    if newly_closed > 0:
        # PPO nur nachtrainieren, wenn er Entscheidungen beeinflussen darf (rl.veto_enabled).
        # Bei Veto aus ist das Modell ungenutzt (Audit 2026-10-03: zuletzt degenerierte
        # Immer-SKIP-Policy) – Training wäre reine Rechenzeit + Commit-Rauschen.
        if bool(cfg.rl.get("veto_enabled", False)):
            log.info(f"{newly_closed} neue closed_trades → starte RL-Nachtraining...")
            retrain_rl_agent(history)
        else:
            log.info("RL-Veto aus → PPO-Nachtraining übersprungen (Modell ungenutzt)")
        # Robuster PPO-Challenger (SHADOW): chronologisch, fester Seed,
        # Walk-forward-Diagnose in outputs/models/ppo_robust_shadow_meta.json
        try:
            from modules.rl_robust_shadow import train_robust_shadow
            meta = train_robust_shadow(history)
            if meta:
                wf = meta["walk_forward"]["test"]
                log.info(f"Robustes PPO (Shadow): walk-forward n={wf['n']} "
                         f"Aktionen={wf['action_counts']} kollabiert={wf['collapsed']}")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Robustes PPO-Shadow-Training fehlgeschlagen (ignoriert): {e}")
    else:
        log.info("Keine neuen closed_trades → RL-Training übersprungen.")

    # Candidate-Ledger: Counterfactual-Outcomes nachtragen (reine Observability,
    # darf den Feedback-Loop nie brechen).
    try:
        candidate_ledger.update_outcomes(today.strftime("%Y-%m-%d"))
    except Exception as e:
        log.warning(f"candidate_ledger.update_outcomes Fehler (ignoriert): {e}")

    log.info("=== Feedback-Loop abgeschlossen ===")


if __name__ == "__main__":
    main()
