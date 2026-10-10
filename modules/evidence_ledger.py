"""modules/evidence_ledger.py – einheitliche Sicht der Evidence Engine über alle Prognose-/Outcome-Speicher.

Konsolidierung über eine gemeinsame Schnittstelle statt eines Rewrites (docs/ARCHITECTURE_MAP.md, Befunde 5/9).
Die bestehenden Ledger schreiben unverändert weiter. Dieses Modul LIEST sie und bildet jeden Datensatz auf ein
Schema ab (`evidence-v1`):

  prediction_id, source_store, population, entity, decision_time, decision_date, direction (+1/−1/None),
  features (zum Entscheidungszeitpunkt eingefroren), horizon, horizon_unit (calendar|trading|holding),
  outcome (Richtungsrendite brutto), outcome_net (nach Kosten), cost, outcome_method, price_type,
  quality (RELIABLE | PROXY | APPROXIMATED | UNKNOWN), label_available_at (frühester Tag, an dem das Label bekannt war)

Regeln:
* Nur Lesen. Es wird nichts nachberechnet und nichts aufgefüllt; fehlende Outcomes fehlen.
* `labeled(..., as_of)` liefert ausschließlich Labels mit label_available_at <= as_of. Das ist der
  Leakage-Schutz für jedes Training.
* Underlying-Renditen sind PROXY für Options-Trades (price_type UNDERLYING_PROXY), nie realisierte Options-P&L.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from modules.atomic_io import read_jsonl

log = logging.getLogger(__name__)

SCHEMA = "evidence-v1"
UNDERLYING_COST = 0.002          # 2 × 10 bp Round-Trip (wie expectation_alpha), explizit je Datensatz


def _dir(d) -> int | None:
    s = str(d or "").upper()
    return 1 if s == "BULLISH" else -1 if s == "BEARISH" else None


def _day(x) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10]) if x else None
    except ValueError:
        return None


def _rec(**kw) -> dict:
    base = {"schema": SCHEMA, "features": None, "cost": None, "outcome_net": None, "meta": {}}
    base.update(kw)
    d = _day(base.get("decision_time"))
    base["decision_date"] = d.isoformat() if d else None
    return base


# ── Adapter je Speicher ─────────────────────────────────────────────────────
def from_candidate_ledger(root: Path | None = None, horizons=(5, 20, 45)) -> list[dict]:
    """Jeder Kandidat mit Richtung (nach Prescreening). Kalendertag-Horizonte, Underlying-Proxy;
    Label bekannt am Tag nach dem Horizont-Schluss."""
    from modules import candidate_ledger as cl
    root = Path(root or cl.LEDGER_ROOT)
    out = []
    for f in sorted(root.glob("*.jsonl")) if root.exists() else []:
        for r in read_jsonl(f):
            d = _dir(r.get("direction"))
            day = _day(r.get("signal_timestamp") or r.get("date"))
            if d is None or day is None:
                continue
            oc = r.get("outcomes") or {}
            for h in horizons:
                v = oc.get(f"ret_{h}d")
                g = None if v is None else d * float(v)
                out.append(_rec(prediction_id=f"{r.get('signal_id')}:{h}d", source_store="candidate_ledger",
                                population="CANDIDATE", entity=r.get("ticker"),
                                decision_time=r.get("signal_timestamp"), direction=d,
                                features=r.get("features") or {}, horizon=h, horizon_unit="calendar",
                                outcome=g, outcome_net=None if g is None else round(g - UNDERLYING_COST, 6),
                                cost=UNDERLYING_COST, outcome_method="underlying_close_daily",
                                price_type="UNDERLYING_PROXY", quality="PROXY" if g is not None else "UNKNOWN",
                                label_available_at=(day + timedelta(days=h + 1)).isoformat(),
                                meta={"signal_id": r.get("signal_id"), "status": r.get("status"),
                                      "reject_stage": r.get("reject_stage")}))
    return out


def from_final_mc(ledger_dir: Path | None = None, outcomes_path: Path | None = None) -> list[dict]:
    from modules import final_mc_ledger as fml
    rows, outs = fml.read_rows(ledger_dir), fml.read_outcomes(outcomes_path)
    out = []
    for r in rows:
        for (oid, h), e in outs.items():
            if oid != r.get("observation_id") or e.get("outcome") is None:
                continue
            out.append(_rec(prediction_id=f"{oid}:{h}d", source_store="final_mc_ledger",
                            population="FINAL_MC_SURVIVOR", entity=r.get("ticker"),
                            decision_time=r.get("timestamp"), direction=_dir(r.get("direction")),
                            features=None, horizon=int(h), horizon_unit="calendar", outcome=e["outcome"],
                            outcome_net=round(float(e["outcome"]) - UNDERLYING_COST, 6), cost=UNDERLYING_COST,
                            outcome_method=e.get("outcome_method"), price_type="UNDERLYING_PROXY", quality="PROXY",
                            label_available_at=e.get("exit_date") or e.get("resolved_at")))
    return out


def from_expectation_alpha(root: Path | None = None) -> list[dict]:
    from modules.expectation_alpha import ledger as eal
    rows = {r["observation_id"]: r for r in eal.read_rows(root)}
    out = []
    for (oid, ex, v, h), e in eal.read_outcomes(root).items():
        r = rows.get(oid)
        if not r or ex != "UNDERLYING" or v != "immediate" or e.get("outcome_net") is None:
            continue
        out.append(_rec(prediction_id=f"{oid}:{h}t", source_store="expectation_alpha", population="EA_NEWS_CANDIDATE",
                        entity=r.get("ticker"), decision_time=r.get("decision_time"),
                        direction=r.get("direction_sign"), features=r.get("env"), horizon=int(h),
                        horizon_unit="trading", outcome=e.get("outcome"), outcome_net=e.get("outcome_net"),
                        cost=None, outcome_method=e.get("outcome_method"), price_type="UNDERLYING_PROXY",
                        quality="PROXY", label_available_at=e.get("exit_date"), meta={"status": r.get("status")}))
    return out


def from_champion_trades(history: dict | None = None, history_path: Path | None = None) -> list[dict]:
    """Geschlossene Champion-Trades: realisierte Options-P&L (Preistyp aus der Outcome-Methode)."""
    import json

    from modules.outcomes import is_reliable_outcome, outcome_class
    if history is None:
        p = Path(history_path or "outputs/history.json")
        history = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    from modules import learning_ledger as ll
    out = []
    for t in history.get("closed_trades") or []:
        if t.get("outcome") is None:
            continue
        pt = t.get("outcome_price_type") or {
            "option_quote": "OBSERVED_QUOTE", "option_quote_bid_zero": "OBSERVED_QUOTE",
            "spread_quote": "OBSERVED_QUOTE", "expiry_intrinsic": "SETTLEMENT_INTRINSIC",
            "delta_approx": "MODELED", "delta_approx_clipped": "MODELED",
            "stock_fallback": "UNDERLYING_PROXY"}.get(t.get("outcome_method"), "UNKNOWN")
        cls = outcome_class(t)
        out.append(_rec(prediction_id=ll.trade_id(t), source_store="history.closed_trades",
                        population="CHAMPION_TRADE", entity=t.get("ticker"), decision_time=t.get("entry_date"),
                        direction=_dir((t.get("deep_analysis") or {}).get("direction")),
                        features=t.get("features") or {}, horizon=None, horizon_unit="holding",
                        outcome=t.get("outcome"), outcome_net=t.get("outcome_net"),
                        cost=(t.get("cost_model") or {}).get("cost_return"),
                        outcome_method=t.get("outcome_method"), price_type=pt,
                        quality="RELIABLE" if is_reliable_outcome(t) else cls,
                        label_available_at=t.get("close_date")))
    return out


SOURCES = {"CANDIDATE": from_candidate_ledger, "FINAL_MC_SURVIVOR": from_final_mc,
           "EA_NEWS_CANDIDATE": from_expectation_alpha, "CHAMPION_TRADE": from_champion_trades}


def records(populations=None) -> list[dict]:
    out = []
    for pop, fn in SOURCES.items():
        if populations and pop not in populations:
            continue
        try:
            out.extend(fn())
        except Exception as e:  # noqa: BLE001 – ein defekter Speicher darf die Sicht der anderen nicht verhindern
            log.warning(f"evidence_ledger: {pop} nicht lesbar ({type(e).__name__}: {e})")
    return out


def labeled(recs: list[dict], *, as_of, population: str | None = None, horizon: int | None = None,
            min_quality: tuple[str, ...] = ("RELIABLE", "PROXY")) -> list[dict]:
    """Nur Labels, die zum Zeitpunkt as_of bereits bekannt waren (label_available_at <= as_of), mit Outcome."""
    cut = _day(as_of.isoformat() if isinstance(as_of, (date, datetime)) else as_of)
    out = []
    for r in recs:
        la = _day(r.get("label_available_at"))
        if r.get("outcome") is None or la is None or cut is None or la > cut:
            continue
        if population and r.get("population") != population:
            continue
        if horizon is not None and r.get("horizon") != horizon:
            continue
        if r.get("quality") not in min_quality:
            continue
        out.append(r)
    return out


def summary(recs: list[dict] | None = None) -> dict:
    recs = recs if recs is not None else records()
    out: dict = {}
    for r in recs:
        s = out.setdefault(r["population"], {"n": 0, "with_outcome": 0, "by_quality": {}, "by_price_type": {}})
        s["n"] += 1
        if r.get("outcome") is not None:
            s["with_outcome"] += 1
            s["by_quality"][r["quality"]] = s["by_quality"].get(r["quality"], 0) + 1
            s["by_price_type"][r["price_type"]] = s["by_price_type"].get(r["price_type"], 0) + 1
    return {"schema": SCHEMA, "populations": out}


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(summary(), indent=1, ensure_ascii=False))
