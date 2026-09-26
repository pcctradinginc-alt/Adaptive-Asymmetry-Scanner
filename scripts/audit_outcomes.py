#!/usr/bin/env python3
"""
scripts/audit_outcomes.py

Audits outcome methods used in closed trades. Reads outputs/history.json and:
1. Counts closed trades by outcome_method (or "unknown" for legacy)
2. For legacy trades, infers the method from outcome values
3. Flags trades where |outcome - clipped_delta_approx| < 0.01 as "likely_delta_approx"
4. Prints statistics: win rate, mean, median, sum for all/excluding/only delta_approx

Usage: python3 scripts/audit_outcomes.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import median, mean


def compute_clipped_delta_approx(entry_stock: float, close_price: float, entry_debit: float) -> float:
    """Compute what delta-approx would have been (clipped to [-1.0, 5.0])."""
    if entry_stock <= 0 or entry_debit <= 0:
        return 0.0

    stock_return = (close_price - entry_stock) / entry_stock
    leverage = (entry_stock / entry_debit) * 0.65
    result = stock_return * leverage
    return max(-1.0, min(result, 5.0))


def infer_outcome_method_legacy(trade: dict, entry_stock: float, close_price: float) -> str:
    """
    Infer outcome method for legacy trades (no outcome_method field).

    Returns one of: "likely_delta_approx", "likely_option_quote",
                    "likely_stock_fallback", "unknown"
    """
    outcome = float(trade.get("outcome") or 0)
    option = trade.get("option") or {}
    # Gleiche Entry-Debit-Reihenfolge wie feedback.compute_outcome:
    # entry_debit → net_debit (Spread) → ask/last (Long)
    entry_debit = float(trade.get("entry_debit") or 0)
    if entry_debit <= 0:
        entry_debit = float(option.get("net_debit") or 0) or float(option.get("ask") or 0) \
                      or float(option.get("last") or 0)
    is_spread = "SPREAD" in (trade.get("strategy") or "")

    if entry_debit > 0 and not is_spread and entry_stock > 0 and close_price > 0:
        clipped_delta_approx = compute_clipped_delta_approx(entry_stock, close_price, entry_debit)
        if abs(outcome - clipped_delta_approx) < 0.01:
            return "likely_delta_approx"
    if entry_debit <= 0 and entry_stock > 0 and close_price > 0:
        if abs(outcome - (close_price - entry_stock) / entry_stock) < 0.005:
            return "likely_stock_fallback"
    if abs(outcome) >= 5.0 - 1e-9:
        return "likely_delta_approx"   # +500%-Cap existiert nur im Delta-Approx-Pfad
    return "unknown"


def extract_entry_stock_and_price(trade: dict) -> tuple[float, float]:
    """Extract entry_stock (entry_price) and close_price from trade."""
    simulation = trade.get("simulation") or {}
    entry_stock = float(simulation.get("current_price") or 0)
    close_price = float(trade.get("close_price") or 0)
    return entry_stock, close_price


def load_history(history_path: Path = Path("outputs/history.json")) -> dict:
    """Load history.json."""
    if not history_path.exists():
        print(f"ERROR: {history_path} not found")
        sys.exit(1)

    with open(history_path) as f:
        return json.load(f)


def main():
    history = load_history()
    closed_trades = history.get("closed_trades", [])

    if not closed_trades:
        print("No closed trades found.")
        return

    # Count by outcome_method
    method_counts = defaultdict(int)
    legacy_trades = []
    likely_delta_approx_trades = []

    for trade in closed_trades:
        method = trade.get("outcome_method")
        if method:
            method_counts[method] += 1
        else:
            # Legacy trade — infer method
            entry_stock, close_price = extract_entry_stock_and_price(trade)
            inferred = infer_outcome_method_legacy(trade, entry_stock, close_price)
            method_counts[f"legacy_{inferred}"] += 1
            legacy_trades.append(trade)

            if inferred == "likely_delta_approx":
                likely_delta_approx_trades.append(trade)

    # Print method counts
    print("=" * 70)
    print("OUTCOME METHOD DISTRIBUTION (closed_trades)")
    print("=" * 70)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(closed_trades)
        print(f"  {method:30s}: {count:3d} ({pct:5.1f}%)")

    print()
    print(f"Total closed_trades: {len(closed_trades)}")
    print(f"Legacy (no outcome_method): {len(legacy_trades)}")
    print(f"Likely delta_approx (inferred): {len(likely_delta_approx_trades)}")
    print()

    # Compute statistics
    def compute_stats(trades: list, label: str) -> None:
        if not trades:
            print(f"{label}: N/A (0 trades)")
            return

        outcomes = [t.get("outcome", 0) for t in trades]
        wins = sum(1 for o in outcomes if o > 0)
        win_rate = wins / len(outcomes)
        mean_outcome = mean(outcomes)
        median_outcome = median(outcomes)
        sum_outcome = sum(outcomes)

        print(f"{label}:")
        print(f"  n={len(trades)}")
        print(f"  win_rate={win_rate:.1%}")
        print(f"  mean={mean_outcome:+.4f}")
        print(f"  median={median_outcome:+.4f}")
        print(f"  sum={sum_outcome:+.2f}")

    print("=" * 70)
    print("STATISTICS BY CATEGORY")
    print("=" * 70)

    # (a) All
    compute_stats(closed_trades, "(a) ALL TRADES")
    print()

    # (b) Excluding likely_delta_approx
    non_delta = [t for t in closed_trades if t not in likely_delta_approx_trades]
    compute_stats(non_delta, "(b) EXCLUDING LIKELY_DELTA_APPROX")
    print()

    # (c) Only likely_delta_approx
    compute_stats(likely_delta_approx_trades, "(c) ONLY LIKELY_DELTA_APPROX")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
