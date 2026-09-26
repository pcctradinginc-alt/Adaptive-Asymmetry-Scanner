"""
modules/exit_sim.py – Multi-Variant Exit-Counterfactual (rein simulativ)

Generalisiert den bestehenden Trailing-vs-TP-Schattentest aus feedback.py
(`trailing_sim`, das ausschließlich per Take-Profit geschlossene Trades
virtuell weiterführt) auf einen vollständigen Satz alternativer Exit-Regeln,
die für JEDEN echten aktiven Trade parallel mitgeführt werden — unabhängig
davon, wie/ob der echte Trade am Ende geschlossen wird.

Rein simulativ: Es wird keine reale Order ausgelöst, keine Exit-Regel der
Produktion verändert und `trailing_sim` bleibt unverändert bestehen (siehe
feedback.py `register_trailing_sim`/`evaluate_trailing_sims`, weiterhin von
backtest_thresholds.py und modules/engine_monitor.py gelesen).

Varianten (je Trade unabhängig voneinander offen/geschlossen):
    A "tp50_sl45"  – Produktionsregel: Stop-Loss/Take-Profit exakt wie
                     feedback.check_exit_rules (nutzt die tatsächlichen
                     Trade-Schwellen inkl. Spread-Sonderfall).
    B "no_tp"      – gleicher Stop-Loss wie A, aber KEIN Take-Profit —
                     hält bis Time-Exit/Max-Holding.
    C "trail35"    – wie A, aber sobald Peak-Return die Aktivierungsschwelle
                     (Default +50%) erreicht, wird der harte TP durch einen
                     Trailing-Stop 35% unter dem bisherigen Peak ersetzt.
    D "trail50"    – wie C, Trailing-Abstand 50% statt 35%.
    E "partial50"  – verkauft (virtuell) die halbe Position beim Erreichen
                     der Aktivierungsschwelle, die zweite Hälfte läuft wie B
                     weiter. Outcome = Mittel beider Hälften.
    F "time_only"  – kein TP/SL, Exit ausschließlich über Time-Exit/Max-
                     Holding (reiner "buy and hold bis Ablauf"-Test).

WICHTIG – Pfad-Granularität: feedback.py läuft nur ca. 2×/Tag (Cron). Peaks,
Trailing-Trigger und Drawdowns werden deshalb ausschließlich an diesen
diskreten Checkpoints beobachtet, nicht kontinuierlich. Ein Trigger, der
zwischen zwei Läufen erreicht UND wieder verlassen wird, bleibt unsichtbar;
die hier festgehaltenen Outcomes sind daher eine konservative Näherung an den
tatsächlichen Intraday-Pfad, kein exaktes Pfad-Modell.
"""

from __future__ import annotations

import statistics
from datetime import datetime

from modules.config import cfg

VARIANT_KEYS = ["tp50_sl45", "no_tp", "trail35", "trail50", "partial50", "time_only"]

# Wie viele exit_sim-Einträge maximal behalten werden (Speicherbegrenzung,
# analog zu shadow_trades in feedback.py).
MAX_ENTRIES = 400


def _exit_sim_cfg() -> dict:
    """Liest den exit_sim-Konfigurationsblock robust mit Defaults."""
    try:
        raw = cfg.get("exit_sim", {}) or {}
        return dict(raw)
    except Exception:
        return {}


def _default_variant_state() -> dict:
    return {"outcome": None, "closed": False, "peak": 0.0, "close_date": None}


# ── Entry-Erzeugung ────────────────────────────────────────────────────────────

def register_exit_sim(history: dict, trade: dict, today: datetime) -> None:
    """
    Legt (idempotent, dedupliziert über ticker+entry_date) einen neuen
    exit_sim-Eintrag für einen aktiven Trade an. Wird pro Feedback-Lauf für
    jeden Trade in active_trades aufgerufen — läuft unabhängig vom echten
    Trade-Close weiter, bis alle Varianten geschlossen sind.
    """
    sims = history.setdefault("exit_sim", [])
    ticker     = trade["ticker"]
    entry_date = trade.get("entry_date")
    if any(s.get("ticker") == ticker and s.get("entry_date") == entry_date for s in sims):
        return
    sims.append({
        "ticker":      ticker,
        "strategy":    trade.get("strategy", ""),
        "option":      trade.get("option") or {},
        "simulation":  trade.get("simulation") or {},
        "entry_debit": trade.get("entry_debit"),
        "entry_date":  entry_date,
        "created":     today.strftime("%Y-%m-%d"),
        "mfe":         0.0,   # Max Favorable Excursion (bester je beobachteter Return)
        "mae":         0.0,   # Max Adverse Excursion (schlechtester je beobachteter Return)
        "closed":      False,
        "variants":    {k: _default_variant_state() for k in VARIANT_KEYS},
    })


def trim_exit_sim(history: dict, max_entries: int = MAX_ENTRIES) -> None:
    """Begrenzt exit_sim auf die letzten max_entries Einträge."""
    sims = history.get("exit_sim") or []
    if len(sims) > max_entries:
        history["exit_sim"] = sims[-max_entries:]


# ── Schwellen / Hilfsbedingungen (analog feedback.check_exit_rules) ──────────

def _sl_tp_thresholds(trade: dict) -> tuple[float, float]:
    """SL-/TP-Schwellen exakt wie feedback.check_exit_rules (Produktionsregel)."""
    strategy  = trade.get("strategy", "")
    option    = trade.get("option") or {}
    is_spread = "SPREAD" in strategy

    if is_spread:
        sl_threshold = -0.50
        entry   = float(trade.get("entry_debit") or option.get("net_debit") or 0)
        long_k  = float(option.get("strike") or 0)
        short_k = float((option.get("spread_leg") or {}).get("strike") or 0)
        width   = short_k - long_k if short_k > long_k > 0 else 0
        if width > 0 and entry > 0:
            tp_threshold = (width - entry) * 0.70 / entry
        else:
            tp_threshold = 0.50
    else:
        sl_threshold = -0.45
        tp_threshold = 0.50
    return sl_threshold, tp_threshold


def _time_exit_hit(trade: dict, outcome: float, today: datetime) -> bool:
    """Production-Time-Exit: 50% der Laufzeit verstrichen und Gewinn < +20%."""
    option = trade.get("option") or {}
    expiry_str = option.get("expiry", "")
    try:
        expiry    = datetime.strptime(expiry_str, "%Y-%m-%d")
        entry_dt  = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        dte_total = (expiry - entry_dt).days
        remaining = (expiry - today).days
        return dte_total > 0 and remaining <= dte_total * 0.5 and outcome < 0.20
    except (ValueError, KeyError, TypeError):
        return False


def _is_expired(trade: dict, today: datetime) -> bool:
    option = trade.get("option") or {}
    expiry_str = option.get("expiry", "")
    try:
        return datetime.strptime(expiry_str, "%Y-%m-%d") <= today
    except ValueError:
        return False


def _max_holding_hit(trade: dict, today: datetime, close_after_days: int) -> bool:
    try:
        entry_dt = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
    except (ValueError, KeyError, TypeError):
        return False
    return (today - entry_dt).days >= close_after_days


def _terminal_exit(trade: dict, outcome: float, today: datetime, close_after_days: int) -> bool:
    """Zeit-/Max-Holding-/Expiry-Exit — gilt für alle Varianten als Backstop."""
    return (
        _time_exit_hit(trade, outcome, today)
        or _max_holding_hit(trade, today, close_after_days)
        or _is_expired(trade, today)
    )


def _close(state: dict, outcome: float, today: datetime, reason: str | None = None) -> None:
    state["outcome"]    = round(outcome, 4)
    state["closed"]     = True
    state["close_date"] = today.strftime("%Y-%m-%d")
    if reason:
        state["close_reason"] = reason


# ── Varianten-Update-Funktionen ────────────────────────────────────────────────

def _update_variant_a(state: dict, trade: dict, outcome: float, today: datetime,
                       close_after_days: int) -> None:
    """A: Produktionsregel (TP 50% / SL 45%, spread-aware, Time-/Max-Holding)."""
    if state["closed"]:
        return
    state["peak"] = round(max(float(state["peak"]), outcome), 4)
    sl, tp = _sl_tp_thresholds(trade)
    if outcome <= sl:
        _close(state, outcome, today, "stop_loss")
        return
    if outcome >= tp:
        _close(state, outcome, today, "take_profit")
        return
    if _terminal_exit(trade, outcome, today, close_after_days):
        _close(state, outcome, today, "time_exit")


def _update_variant_b(state: dict, trade: dict, outcome: float, today: datetime,
                       close_after_days: int) -> None:
    """B: gleicher Stop-Loss wie A, aber kein Take-Profit."""
    if state["closed"]:
        return
    state["peak"] = round(max(float(state["peak"]), outcome), 4)
    sl, _tp = _sl_tp_thresholds(trade)
    if outcome <= sl:
        _close(state, outcome, today, "stop_loss")
        return
    if _terminal_exit(trade, outcome, today, close_after_days):
        _close(state, outcome, today, "time_exit")


def _update_variant_trail(state: dict, trade: dict, outcome: float, today: datetime,
                           close_after_days: int, trail_pct: float, activation: float) -> None:
    """C/D: SL wie A, aber ab Aktivierungsschwelle Trailing-Stop statt hartem TP."""
    if state["closed"]:
        return
    state["peak"] = round(max(float(state["peak"]), outcome), 4)
    sl, _tp = _sl_tp_thresholds(trade)
    if state["peak"] < activation:
        # Noch nicht aktiviert → verhält sich wie B (Stop-Loss aktiv, kein TP).
        if outcome <= sl:
            _close(state, outcome, today, "stop_loss")
            return
    else:
        trigger = state["peak"] * (1 - trail_pct)
        if outcome <= trigger:
            _close(state, outcome, today, "trail_stop")
            return
    if _terminal_exit(trade, outcome, today, close_after_days):
        _close(state, outcome, today, "time_exit")


def _update_variant_e(state: dict, trade: dict, outcome: float, today: datetime,
                       close_after_days: int, activation: float) -> None:
    """E: verkauft die Hälfte bei Erreichen der Aktivierungsschwelle, Rest wie B."""
    if state["closed"]:
        return
    state["peak"] = round(max(float(state["peak"]), outcome), 4)

    if not state.get("half1_locked") and outcome >= activation:
        state["half1_outcome"] = round(outcome, 4)
        state["half1_locked"]  = True

    sl, _tp = _sl_tp_thresholds(trade)
    half2_terminal = outcome <= sl or _terminal_exit(trade, outcome, today, close_after_days)
    if not half2_terminal:
        return

    # Falls die Aktivierungsschwelle nie erreicht wurde, ist half1 == half2
    # (keine Teilverkäufe erfolgt → identisch zu Variante B).
    half1 = state.get("half1_outcome", outcome)
    half2 = outcome
    combined = (half1 + half2) / 2.0
    reason = "stop_loss" if outcome <= sl else "time_exit"
    _close(state, combined, today, reason)


def _update_variant_f(state: dict, trade: dict, outcome: float, today: datetime,
                       close_after_days: int) -> None:
    """F: kein TP/SL — Exit ausschließlich über Time-Exit/Max-Holding/Expiry."""
    if state["closed"]:
        return
    state["peak"] = round(max(float(state["peak"]), outcome), 4)
    if _terminal_exit(trade, outcome, today, close_after_days):
        _close(state, outcome, today, "time_exit")


# ── Öffentliche Update-Funktion ───────────────────────────────────────────────

def update_exit_sim_entry(entry: dict, outcome: float, today: datetime, *,
                           close_after_days: int | None = None,
                           trail_pcts: list[float] | None = None,
                           activation: float | None = None) -> None:
    """
    Aktualisiert einen einzelnen exit_sim-Eintrag mit dem aktuellen Options-
    Return. Rein simulativ, keine Netzwerk-/IO-Zugriffe. `entry` dient hier
    zugleich als "trade"-artiges Dict für die Schwellen-Berechnung (enthält
    strategy/option/entry_date/entry_debit — dieselben Felder, die ein echter
    Trade an dieser Stelle hätte).

    close_after_days/trail_pcts/activation können explizit übergeben werden
    (z.B. in Tests); ansonsten werden sie robust aus config.yaml gelesen.
    """
    exit_cfg = _exit_sim_cfg()
    if not exit_cfg.get("enabled", True):
        return

    if close_after_days is None:
        try:
            close_after_days = int(cfg.learning.close_after_days)
        except Exception:
            close_after_days = 45

    if trail_pcts is None:
        trail_pcts = exit_cfg.get("trail_pcts", [0.35, 0.50])
    trail_pcts = list(trail_pcts) if trail_pcts else [0.35, 0.50]
    defaults = [0.35, 0.50]
    while len(trail_pcts) < 2:
        trail_pcts.append(defaults[len(trail_pcts)])
    trail35_pct, trail50_pct = float(trail_pcts[0]), float(trail_pcts[1])

    if activation is None:
        activation = float(exit_cfg.get("activation", 0.50))

    entry["mfe"] = round(max(float(entry.get("mfe") or 0.0), outcome), 4)
    entry["mae"] = round(min(float(entry.get("mae") or 0.0), outcome), 4)

    variants = entry.setdefault("variants", {k: _default_variant_state() for k in VARIANT_KEYS})
    for key in VARIANT_KEYS:
        variants.setdefault(key, _default_variant_state())

    _update_variant_a(variants["tp50_sl45"], entry, outcome, today, close_after_days)
    _update_variant_b(variants["no_tp"], entry, outcome, today, close_after_days)
    _update_variant_trail(variants["trail35"], entry, outcome, today, close_after_days,
                           trail35_pct, activation)
    _update_variant_trail(variants["trail50"], entry, outcome, today, close_after_days,
                           trail50_pct, activation)
    _update_variant_e(variants["partial50"], entry, outcome, today, close_after_days, activation)
    _update_variant_f(variants["time_only"], entry, outcome, today, close_after_days)

    entry["closed"] = all(variants[k].get("closed") for k in VARIANT_KEYS)


# ── Auswertung ─────────────────────────────────────────────────────────────────

def summarize_exit_sim(history: dict, min_n: int = 10) -> dict:
    """
    Fasst geschlossene exit_sim-Outcomes je Variante zusammen: n, mean,
    median, win_rate (outcome > 0), total_loss_rate (outcome <= -0.95).
    Varianten mit n < min_n liefern nur einen Hinweistext statt Kennzahlen,
    um Zufallsrauschen bei kleinen Stichproben nicht als Befund auszugeben.
    """
    sims = history.get("exit_sim") or []
    result: dict = {}
    for key in VARIANT_KEYS:
        outcomes = []
        for s in sims:
            if not isinstance(s, dict):
                continue
            variant = (s.get("variants") or {}).get(key) or {}
            if not variant.get("closed"):
                continue
            o = variant.get("outcome")
            if isinstance(o, (int, float)):
                outcomes.append(float(o))

        n = len(outcomes)
        if n < min_n:
            result[key] = {"n": n, "note": f"zu wenig Daten (n={n})"}
            continue

        wins   = sum(1 for o in outcomes if o > 0)
        losses = sum(1 for o in outcomes if o <= -0.95)
        result[key] = {
            "n":               n,
            "mean":            round(statistics.mean(outcomes), 4),
            "median":          round(statistics.median(outcomes), 4),
            "win_rate":        round(wins / n, 4),
            "total_loss_rate": round(losses / n, 4),
        }
    return result
