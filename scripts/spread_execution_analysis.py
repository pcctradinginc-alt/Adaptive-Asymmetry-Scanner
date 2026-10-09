"""scripts/spread_execution_analysis.py – Shadow-Analyse immediate_liquidation_loss vs. Outcome (2026-10-09).
Quellen: history.json (geschlossene/aktive Trades, Schatten-Trades) + shadow_trades_archive.jsonl.
Nur Spreads mit gespeicherten Leg-Quotes. Explorativ (überwiegend NON_RELIABLE); keine Schwellenoptimierung."""
from __future__ import annotations

import json
from pathlib import Path

from modules import spread_execution as se
from modules.atomic_io import atomic_write_json, read_jsonl

OUT = Path("outputs/research/spread_execution_analysis.json")


def main() -> int:
    h = json.loads(Path("outputs/history.json").read_text())
    arch = Path("outputs/shadow_trades_archive.jsonl")
    trades = (h.get("closed_trades") or []) + (h.get("active_trades") or []) + (h.get("shadow_trades") or []) \
        + (read_jsonl(arch) if arch.exists() else [])
    res = se.analyse(trades)
    res["sources"] = {"closed": len(h.get("closed_trades") or []), "active": len(h.get("active_trades") or []),
                      "shadow_view": len(h.get("shadow_trades") or []), "archive": arch.exists()}
    atomic_write_json(OUT, res, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
