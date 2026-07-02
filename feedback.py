"""
feedback.py – Adaptive Lern-Loop v5.0

Änderungen v5.0:
    - Tradier Live-API als primäre Datenquelle für Optionspreise (P&L-Tracking)
      Endpoint: /v1/markets/options/chains (Optionspreis via Strike-Filter)
      Endpoint: /v1/markets/quotes        (Aktienkurs Real-Time)
    - get_current_price():        Tradier Primary → yfinance Fallback
    - get_current_option_price(): Tradier Primary → yfinance Fallback
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
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
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

def get_current_option_price(
    ticker: str, option: dict, strategy: str = ""
) -> float:
    """
    Aktueller Mid-Price einer Options-Position: Tradier Primary → yfinance Fallback.

    Args:
        ticker:   Ticker-Symbol (z.B. "AAPL")
        option:   Option-Dict aus history.json (strike, expiry, ...)
        strategy: Trade-Strategie (z.B. "LONG_CALL", "BEAR_PUT_SPREAD")
                  → bestimmt ob Call oder Put gesucht wird.
                  Leer → versucht Call zuerst, dann Put.
    """
    if not option:
        return 0.0
    strike = option.get("strike")
    expiry = option.get("expiry")
    if not strike or not expiry:
        return 0.0

    # Option-Type aus Strategie ableiten
    option_type = _option_type_from_strategy(strategy)

    # ── Versuch 1: Tradier ────────────────────────────────────────────────────
    if _use_tradier():
        price = _tradier_option_price(ticker, strike, expiry, option_type)
        if price > 0:
            log.debug(
                f"[{ticker}] Tradier Options-Mid: strike={strike} "
                f"expiry={expiry} → ${price:.2f}"
            )
            return price
        log.debug(f"[{ticker}] Tradier Options-Preis fehlgeschlagen → yfinance")

    # ── Versuch 2: yfinance Fallback ──────────────────────────────────────────
    return _yfinance_option_price(ticker, strike, expiry)


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


def _tradier_option_price(
    ticker: str, strike: float, expiry: str, option_type: str
) -> float:
    """
    Options-Mid-Price via Tradier /v1/markets/options/chains.

    Filtert die Chain nach Strike ± 0.01 und option_type.
    Wenn option_type="call" und nichts gefunden → versucht "put" (Fallback
    bei alten Trades ohne Strategy-Info in history.json).
    """
    def _fetch_mid(o_type: str) -> float:
        try:
            resp = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                params={
                    "symbol":     ticker,
                    "expiration": expiry,
                    "greeks":     "false",
                },
                headers=_tradier_headers(),
                timeout=TRADIER_TIMEOUT,
            )
            resp.raise_for_status()
            data    = resp.json()
            options = data.get("options", {}).get("option", []) or []

            # Einzelner Kontrakt kommt als Dict
            if isinstance(options, dict):
                options = [options]

            for o in options:
                if o.get("option_type") != o_type:
                    continue
                # Strike-Vergleich mit Float-Toleranz
                if abs(float(o.get("strike", 0)) - float(strike)) > 0.01:
                    continue

                bid = float(o.get("bid") or 0)
                ask = float(o.get("ask") or 0)
                if bid > 0 and ask > 0:
                    return round((bid + ask) / 2, 4)
                # Nur Ask vorhanden
                if ask > 0:
                    return float(ask)

            return 0.0

        except Exception as e:
            log.debug(f"Tradier Options-Chain [{ticker} {expiry}]: {e}")
            return 0.0

    # Primärer Versuch
    price = _fetch_mid(option_type)
    if price > 0:
        return price

    # Fallback: anderer Option-Type (für alte Trades ohne Strategy-Info)
    other_type = "put" if option_type == "call" else "call"
    return _fetch_mid(other_type)


def _yfinance_option_price(
    ticker: str, strike: float, expiry: str
) -> float:
    """Options-Mid-Price via yfinance (Fallback, unveränderte v4.0-Logik)."""
    try:
        t = yf.Ticker(ticker)
        if expiry not in t.options:
            return 0.0
        chain   = t.option_chain(expiry)
        # Versuche Calls zuerst, dann Puts
        for opts in [chain.calls, chain.puts]:
            matches = opts[(opts["strike"] == strike) & (opts["ask"] > 0)]
            if not matches.empty:
                row = matches.iloc[0]
                return float((row["bid"] + row["ask"]) / 2)
        return 0.0
    except Exception as e:
        log.debug(f"yfinance Options-Preis Fehler für {ticker}: {e}")
        return 0.0


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
        width      = round(short_k - long_k, 2) if short_k > long_k > 0 else 0

        if width <= 0 or long_k <= 0:
            return None

        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        end_str   = (expiry_dt + timedelta(days=4)).strftime("%Y-%m-%d")
        hist      = yf.Ticker(ticker).history(start=expiry_str, end=end_str, auto_adjust=True)
        if hist.empty:
            return None

        close = float(hist["Close"].iloc[0])  # Schlusskurs am Verfallstag

        if close <= long_k:
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
        log.debug(f"    [{ticker}] Expired-Spread Fehler: {e}")
        return None


def get_current_spread_price(ticker: str, option: dict, strategy: str) -> float | None:
    """
    Aktueller Net-Wert eines Spreads: long_leg_mid − short_leg_mid.

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

    long_mid  = get_current_option_price(ticker, option, strategy)
    short_mid = get_current_option_price(
        ticker, {"strike": sl.get("strike"), "expiry": expiry}, strategy
    )

    if long_mid > 0 and short_mid > 0:
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

def compute_outcome(trade: dict, current_stock_price: float) -> float | None:
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
        current_spread = get_current_spread_price(ticker, option, strategy)
        if current_spread is None:
            # Preis wirklich nicht ermittelbar — nicht als 0.0 ins Training
            log.warning(f"    [{ticker}] Spread-Preis nicht abrufbar → Outcome nicht verwertbar")
            return None
        # current_spread == 0.0 ist valide: Spread verfallen wertlos → -100%
        result = (current_spread - entry_debit) / entry_debit
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
        if current_option > 0:
            result = (current_option - entry_debit) / entry_debit
            log.info(
                f"    Options-P&L: entry=${entry_debit:.2f} → "
                f"current=${current_option:.2f} = {result:+.2%}"
            )
            return result
        if entry_stock > 0:
            leverage = (entry_stock / entry_debit) * 0.65
            result   = stock_return * leverage
            result   = max(-1.0, min(result, 5.0))   # Options: Max-Verlust=-100%, Cap=+500%
            log.info(f"    Delta-approx: {stock_return:+.2%} × {leverage:.1f} = {result:+.2%}")
            return result

    # ── Letzter Fallback: Stock-Return (nur wenn kein Debit bekannt) ─────────
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
        outcome = compute_outcome(st, current)
        if outcome is None:
            continue
        st["outcome"]    = round(outcome, 4)
        st["close_date"] = today.strftime("%Y-%m-%d")
        log.info(f"  [SHADOW {st['ticker']}] ({st.get('reject_reason','?')}) Outcome={outcome:+.2%}")
    # Liste begrenzen: nur die letzten 300 behalten
    if len(shadows) > 300:
        history["shadow_trades"] = shadows[-300:]


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
    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        return history.get("model_weights", {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20})

    outcomes, impacts, mismatches, drifts = [], [], [], []
    for t in closed:
        outcome = t.get("outcome")
        if outcome is None:
            continue
        feat = t.get("features", {})
        outcomes.append(outcome)
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

        current = get_current_price(ticker)
        if current <= 0:
            still_active.append(trade)
            continue

        outcome = compute_outcome(trade, current)
        if outcome is None:
            log.warning(f"  [{ticker}] Outcome nicht ermittelbar → bleibt aktiv, kein Lern-Update")
            still_active.append(trade)
            continue
        log.info(f"  [{ticker}] Alter={age_days}d Outcome={outcome:+.2%}")

        # Peak-Return mitschreiben (Grundlage für den Trailing-Paralleltest)
        trade["peak_return"] = round(max(float(trade.get("peak_return") or outcome), outcome), 4)

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
            feat = trade.get("features", {})
            for f_name, bin_key in [("impact",    "bin_impact"),
                                      ("mismatch",  "bin_mismatch"),
                                      ("eps_drift", "bin_eps_drift")]:
                bin_label = feat.get(bin_key)
                if bin_label:
                    update_bin(history["feature_stats"], f_name, bin_label, outcome)

            trade["outcome"]      = round(outcome, 4)
            trade["close_date"]   = today.strftime("%Y-%m-%d")
            trade["close_price"]  = current
            trade["close_reason"] = exit_reason or "max_holding_period"
            history.setdefault("closed_trades", []).append(trade)
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

    # Trailing-Paralleltest fortführen (virtuelle TP-Weiterführungen)
    evaluate_trailing_sims(history, today)

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
        log.info(f"{newly_closed} neue closed_trades → starte RL-Nachtraining...")
        retrain_rl_agent(history)
    else:
        log.info("Keine neuen closed_trades → RL-Training übersprungen.")

    log.info("=== Feedback-Loop abgeschlossen ===")


if __name__ == "__main__":
    main()
