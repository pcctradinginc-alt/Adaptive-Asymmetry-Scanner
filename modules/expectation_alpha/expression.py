"""modules/expectation_alpha/expression.py – vorab registrierte Expressions je These.

Alpha wird auf Basiswerten/ETFs gemessen (keine Optionen, keine Optionsscheine). Alle Alternativen
stehen zum Entscheidungszeitpunkt fest: UNDERLYING, SECTOR_ETF, SPY, QQQ, IWM.

Counterfactuals sind nur zwischen diesen Alternativen erlaubt; es gibt keine Hindsight-Alternativen.
Die Auswahl folgt einer deterministischen Regel (config `expressions.selection_rule`):
  starke News -> UNDERLYING; sonst positiver Kontext -> SECTOR_ETF (falls gemappt); sonst UNDERLYING.
THESIS_QUALITY (Median der Richtungsrenditen aller verfügbaren Expressions) wird getrennt von
EXPRESSION_QUALITY (gewählte minus Median) ausgewertet (evaluation.py).
"""
from __future__ import annotations

from modules.expectation_alpha.schemas import NEWS_STRONG, canonical_hash

ETFS = ("SPY", "QQQ", "IWM")


def registered(ticker: str, sector_etf: str | None, cfg: dict) -> list[dict]:
    ec = cfg.get("expressions") or {}
    cost = ec.get("cost_per_side") or {}
    out = []
    for name in ec.get("registered") or ["UNDERLYING"]:
        if name == "UNDERLYING":
            sym, cls = ticker, "stock"
        elif name == "SECTOR_ETF":
            sym, cls = sector_etf, "etf"
        else:
            sym, cls = name, "etf"
        out.append({"expression": name, "symbol": sym, "instrument": cls,
                    "cost_per_side": float(cost.get(cls, 0.001)),
                    "available": sym is not None})
    return out


def select(news: dict, context: dict, exprs: list[dict], cfg: dict) -> dict:
    by = {e["expression"]: e for e in exprs}
    default = (cfg.get("expressions") or {}).get("default", "UNDERLYING")
    if news.get("strength") == NEWS_STRONG:
        name, why = "UNDERLYING", "news_strong"
    elif context.get("context_status") == 1 and (by.get("SECTOR_ETF") or {}).get("available"):
        name, why = "SECTOR_ETF", "context_positive"
    else:
        name, why = default, "default"
    return {"selected_expression": name, "symbol": by[name]["symbol"], "rule": why,
            "registry_hash": canonical_hash(exprs)}
