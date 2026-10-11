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
* Alle Quell-Outcomes sind bereits richtungsbereinigt (candidate_ledger `_direction_adjusted_return`,
  final_mc `path_outcome`, EA `_path`) und werden unverändert übernommen – nie ein zweites Mal mit der Richtung
  multipliziert.
* label_available_at = Tag NACH dem Ausstiegs-Schlusskurs (der Schluss ist erst nach Handelsende bekannt).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from modules.atomic_io import read_jsonl

log = logging.getLogger(__name__)

SCHEMA = "evidence-v1"
UNDERLYING_COST = 0.002          # 2 × 10 bp Round-Trip (wie expectation_alpha), explizit je Datensatz
COMMISSION_PER_CONTRACT_SIDE = 0.65   # wie ROI-Gate/feedback (cost-v1); Bid/Ask-Kreuzung steckt schon im Outcome
OPTION_HORIZONS = (5, 20, 45, 60, 120)


def _dir(d) -> int | None:
    s = str(d or "").upper()
    return 1 if s == "BULLISH" else -1 if s == "BEARISH" else None


def _day(x) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10]) if x else None
    except ValueError:
        return None


def _after(x) -> str | None:
    """Erster Tag, an dem ein Label mit Ausstiegs-Schluss am Tag x bekannt ist (x + 1)."""
    d = _day(x)
    return (d + timedelta(days=1)).isoformat() if d else None


def _rec(**kw) -> dict:
    base = {"schema": SCHEMA, "features": None, "cost": None, "outcome_net": None, "meta": {}}
    base.update(kw)
    d = _day(base.get("decision_time"))
    base["decision_date"] = d.isoformat() if d else None
    return base


# ── Adapter je Speicher ─────────────────────────────────────────────────────
def from_candidate_ledger(root: Path | None = None, horizons=(5, 20, 45)) -> list[dict]:
    """Jeder Kandidat mit Richtung (nach Prescreening). Kalendertag-Horizonte, Underlying-Proxy.
    ret_{h}d ist bereits richtungsbereinigt. Label bekannt am Tag nach dem Horizont-Schluss, gerechnet ab dem
    effektiven Entry-Tag (next_open: nachgetragenes Handelsdatum) – genau wie candidate_ledger.update_outcomes."""
    from modules import candidate_ledger as cl
    root = Path(root or cl.LEDGER_ROOT)
    out = []
    for f in sorted(root.glob("*.jsonl")) if root.exists() else []:
        for r in read_jsonl(f):
            out.extend(_candidate_option_records(r))
            d = _dir(r.get("direction"))
            day = _day(r.get("signal_timestamp") or r.get("date"))
            if d is None or day is None:
                continue
            ent = cl._row_entry_dt(r)
            ent = ent.date() if ent else day
            oc = r.get("outcomes") or {}
            for h in horizons:
                v = oc.get(f"ret_{h}d")
                g = None if v is None else float(v)             # bereits richtungsbereinigt
                out.append(_rec(prediction_id=f"{r.get('signal_id')}:{h}d", source_store="candidate_ledger",
                                population="CANDIDATE", entity=r.get("ticker"),
                                decision_time=r.get("signal_timestamp"), direction=d,
                                features=r.get("features") or {}, horizon=h, horizon_unit="calendar",
                                outcome=g, outcome_net=None if g is None else round(g - UNDERLYING_COST, 6),
                                cost=UNDERLYING_COST, outcome_method="underlying_close_daily",
                                price_type="UNDERLYING_PROXY", quality="PROXY" if g is not None else "UNKNOWN",
                                label_available_at=(max(ent, day) + timedelta(days=h + 1)).isoformat(),
                                meta=_candidate_meta(r)))
    return out


def _candidate_meta(r: dict) -> dict:
    """Zielpopulation und Regime-/Signaltyp-Zuordnung je Kandidat (für Scorecard und Champion-Generalisierung).
    stage: CHAMPION (alle Gates, vorgeschlagen) | LATE (bis Options-Design gekommen) | EARLY (vorher verworfen)."""
    ft = r.get("features") or {}
    st, rs = r.get("status"), r.get("reject_stage")
    stage = "CHAMPION" if st == "proposed" else "LATE" if rs == "options_design" else "EARLY"
    return {"signal_id": r.get("signal_id"), "status": st, "reject_stage": rs, "stage": stage,
            "is_champion": st == "proposed", "signal_type": ft.get("prescreen_category"),
            "vix_structure": ft.get("vix_structure"), "late_mark": bool(r.get("late_mark"))}


def _late(oc: dict, key: str, h: int, row_flag: bool) -> bool:
    """Verspäteter Mark je Instrument/Horizont (seit 2026-10-11). Altzeilen ohne Einzel-Flags: Zeilen-Flag gilt."""
    per_key = [k for k in oc if "_late_" in k]
    return bool(oc.get(f"{key.replace('_ret', '')}_late_{h}d")) if per_key else bool(row_flag)


def _candidate_option_records(r: dict) -> list[dict]:
    """Echte Options-Outcomes des Candidate-Ledgers (beobachtete Tradier-Quotes: Kauf zum Ask, Verkauf zum Bid; nach
    Verfall vertragliches Settlement). CANDIDATE_OPTION = Long-Option der Signalrichtung (Call/Put, implizit
    richtungsrichtig), CANDIDATE_STRATEGY = Produktions-Strategie (Spread). Netto = minus Kommission je Kontrakt/Seite.
    Verspätete Marks (late_mark) sind kein Horizont-Outcome -> quality LATE_MARK (nie in der Default-Evidenz)."""
    d = _dir(r.get("direction"))
    day = _day(r.get("signal_timestamp") or r.get("date"))
    if d is None or day is None:
        return []
    oc = r.get("outcomes") or {}
    meta = _candidate_meta(r)
    out = []
    for pop, key, spec, legs_n, debit_key in (
            ("CANDIDATE_OPTION", "real_opt_ret", r.get("real_option") or {}, 1, "ask"),
            ("CANDIDATE_STRATEGY", "real_strat_ret", r.get("real_strategy") or {}, None, "net_debit_entry")):
        if not spec:
            continue
        n_legs = legs_n or max(1, len(spec.get("legs") or []))
        debit = spec.get(debit_key)
        cost = (2 * n_legs * COMMISSION_PER_CONTRACT_SIDE) / (float(debit) * 100) if debit and float(debit) > 0 \
            else None
        expiry = _day(spec.get("expiry"))
        for h in OPTION_HORIZONS:
            v = oc.get(f"{key}_{h}d")
            if v is None:
                continue
            mark = _day(oc.get(f"{key.replace('_ret', '')}_mark_date_{h}d"))     # real_opt_/real_strat_mark_date
            settled = expiry is not None and mark is not None and mark > expiry
            g = float(v)
            out.append(_rec(prediction_id=f"{r.get('signal_id')}:{h}d:{pop.split('_')[1].lower()}",
                            source_store="candidate_ledger", population=pop, entity=r.get("ticker"),
                            decision_time=r.get("signal_timestamp"), direction=d, features=r.get("features") or {},
                            horizon=h, horizon_unit="calendar", outcome=g,
                            outcome_net=None if cost is None else round(g - cost, 6), cost=cost,
                            outcome_method=f"{key}_quote", price_type="SETTLEMENT_INTRINSIC" if settled
                            else "OBSERVED_QUOTE",
                            quality="LATE_MARK" if _late(oc, key, h, meta["late_mark"]) else "RELIABLE"
                            if cost is not None else "UNKNOWN",
                            label_available_at=_after(mark) if mark else
                            (max(_day(spec.get("entry_date")) or day, day) + timedelta(days=h + 1)).isoformat(),
                            meta={**meta, "contract": spec.get("symbol") or spec.get("strategy")}))
    return out


def from_final_mc(ledger_dir: Path | None = None, outcomes_path: Path | None = None) -> list[dict]:
    from modules import final_mc_ledger as fml
    rows, outs = fml.read_rows(ledger_dir), fml.read_outcomes(outcomes_path)
    by_obs: dict = {}
    for (oid, h), e in outs.items():
        by_obs.setdefault(oid, []).append((h, e))
    out = []
    for r in rows:
        oid = r.get("observation_id")
        for h, e in by_obs.get(oid, []):
            if e.get("outcome") is None:
                continue
            out.append(_rec(prediction_id=f"{oid}:{h}d", source_store="final_mc_ledger",
                            population="FINAL_MC_SURVIVOR", entity=r.get("ticker"),
                            decision_time=r.get("timestamp"), direction=_dir(r.get("direction")),
                            features=None, horizon=int(h), horizon_unit="calendar", outcome=e["outcome"],
                            outcome_net=round(float(e["outcome"]) - UNDERLYING_COST, 6), cost=UNDERLYING_COST,
                            outcome_method=e.get("outcome_method"), price_type="UNDERLYING_PROXY", quality="PROXY",
                            label_available_at=_after(e.get("exit_date")) or e.get("resolved_at")))
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
                        quality="PROXY", label_available_at=_after(e.get("exit_date")), meta={"status": r.get("status")}))
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
        # Bewertung (trade-eval-v1): gespeichert ab 2026-10-11; Altbestand wird hier nur abgeleitet (ohne Quotes,
        # nie zurückgeschrieben) -> modellierte/unbekannte Preise sind UNUSABLE, nie ein Verlust
        from modules.outcomes import trade_evaluation
        ev = t.get("outcome_evaluation") or trade_evaluation(t, price_type=pt)
        out.append(_rec(prediction_id=ll.trade_id(t), source_store="history.closed_trades",
                        population="CHAMPION_TRADE", entity=t.get("ticker"), decision_time=t.get("entry_date"),
                        direction=_dir((t.get("deep_analysis") or {}).get("direction")),
                        features=t.get("features") or {}, horizon=None, horizon_unit="holding",
                        outcome=t.get("outcome"), outcome_net=t.get("outcome_net"),
                        cost=(t.get("cost_model") or {}).get("cost_return"),
                        outcome_method=t.get("outcome_method"), price_type=pt,
                        quality="RELIABLE" if is_reliable_outcome(t) and ev["status"] == "OK" else cls,
                        label_available_at=_after(t.get("close_date")),
                        meta={"evaluation": ev, "strategy": t.get("strategy"), "is_champion": True,
                              "stage": "CHAMPION", "close_reason": t.get("close_reason"),
                              "signal_type": (t.get("features") or {}).get("prescreen_category")}))
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


# ── Scorecard: zentrale, reproduzierbare wirtschaftliche Erfolgsmessung (2026-10-11) ──────────────
SCORECARD_PATH = Path("outputs/evidence/scorecard.json")
SCORECARD_VERSION = "scorecard-v1"
MIN_N, MIN_DATES = 30, 20          # darunter: NEED_MORE_DATA (keine Aussage, nie ein Vorzeichen)


def _cell(rs: list[dict], alpha: float) -> dict:
    """Kennzahlen einer Zelle (netto). Unabhängige Ereignisse = Signaltage (Bootstrap zieht ganze Tage)."""
    import statistics

    from modules import evidence_core as ec
    v = [float(r["outcome_net"]) for r in rs if r.get("outcome_net") is not None]
    g = [r["decision_date"] for r in rs if r.get("outcome_net") is not None]
    out = {"n": len(v), "n_signal_days": len(set(g)), "n_entities": len({r.get("entity") for r in rs})}
    if not v:
        out["verdict"] = "NO_DATA"
        return out
    out.update(mean_net=round(statistics.fmean(v), 6), median_net=round(statistics.median(v), 6),
               hit_rate=round(sum(1 for x in v if x > 0) / len(v), 4),
               mean_gross=round(statistics.fmean(float(r["outcome"]) for r in rs if r.get("outcome_net") is not None), 6),
               worst=round(min(v), 6))
    if len(v) < MIN_N or len(set(g)) < MIN_DATES:
        out["verdict"] = "NEED_MORE_DATA"
        return out
    lo, hi = ec.group_bootstrap_ci(v, g, alpha=alpha)
    out["ci"] = [lo, hi]
    out["verdict"] = ("NEED_MORE_DATA" if lo is None else "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0
                      else "INCONCLUSIVE")
    return out


def _diff(a: list[dict], b: list[dict], alpha: float, n: int = 2000, seed: int = 53) -> dict:
    """Mittelwert-Differenz a − b (netto) mit Bootstrap über Signaltage je Gruppe; Aussage erst ab Mindestdaten."""
    import numpy as np
    da, db = _cell(a, alpha), _cell(b, alpha)
    out = {"delta_mean_net": None if da.get("mean_net") is None or db.get("mean_net") is None
           else round(da["mean_net"] - db["mean_net"], 6)}
    if min(da["n"], db["n"]) < MIN_N or min(da["n_signal_days"], db["n_signal_days"]) < MIN_DATES:
        return {**out, "verdict": "NEED_MORE_DATA"}

    def groups(rs):
        by: dict = {}
        for r in rs:
            if r.get("outcome_net") is not None:
                by.setdefault(r["decision_date"], []).append(float(r["outcome_net"]))
        return [np.array(x) for x in by.values()]
    ga, gb, rng = groups(a), groups(b), np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sa = [ga[i] for i in rng.integers(0, len(ga), len(ga))]
        sb = [gb[i] for i in rng.integers(0, len(gb), len(gb))]
        diffs.append(np.concatenate(sa).mean() - np.concatenate(sb).mean())
    lo, hi = (round(float(x), 6) for x in np.quantile(diffs, (alpha / 2, 1 - alpha / 2)))
    return {**out, "ci": [lo, hi], "verdict": "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0 else "INCONCLUSIVE"}


def option_mark_coverage(root: Path | None = None, as_of=None) -> dict:
    """Je Horizont: fällige Real-Options-Marks (Horizont ab Entry verstrichen), gesetzt, offen, verspätet."""
    from modules import candidate_ledger as cl
    root = Path(root or cl.LEDGER_ROOT)
    cut = _day(as_of.isoformat() if isinstance(as_of, (date, datetime)) else as_of) or date.today()
    cov: dict = {}
    for f in sorted(root.glob("*.jsonl")) if root.exists() else []:
        for r in read_jsonl(f):
            oc = r.get("outcomes") or {}
            for key, spec in (("real_opt_ret", r.get("real_option")), ("real_strat_ret", r.get("real_strategy"))):
                if not spec or spec.get("entry_pending"):
                    continue
                ent = _day(spec.get("entry_date"))
                for h in OPTION_HORIZONS:
                    if ent is None or (cut - ent).days < h:
                        continue
                    c = cov.setdefault(f"{key}_{h}d", {"due": 0, "marked": 0, "pending": 0, "late_marks": 0,
                                                       "oldest_pending_days": None})
                    c["due"] += 1
                    if f"{key}_{h}d" in oc:
                        c["marked"] += 1
                        c["late_marks"] += bool(r.get("late_mark"))
                    else:
                        c["pending"] += 1
                        age = (cut - ent).days - h
                        c["oldest_pending_days"] = max(c["oldest_pending_days"] or 0, age)
    return cov


def scorecard(recs: list[dict] | None = None, *, as_of=None, alpha: float = 0.05, write: bool = False,
              path: Path | None = None) -> dict:
    """Zentrale Erfolgsmessung auf zeitlich korrekten Beobachtungen (labeled(as_of)), netto nach Kosten.
    Zellen: alle Kandidaten (Underlying-Proxy / echte Option / echte Strategie), Late-Stage, Champion-Kandidaten,
    realisierte Champion-Trades (Bewertung OK), Regime (VIX-Struktur) und Signaltyp. Bonferroni über alle Zellen.
    Baselines: Late-Stage vs. früh verworfene Kandidaten (Mehrwert der Gates), Option vs. Underlying-Proxy derselben
    Signale. Ein positives Ergebnis ist KEIN Alpha-Nachweis (Forward-Evidenz nur über registrierte Verträge)."""
    recs = recs if recs is not None else records()
    as_of = as_of or date.today()
    lab = labeled(recs, as_of=as_of, min_quality=("RELIABLE", "PROXY"))
    by_pop: dict = {}
    for r in lab:
        by_pop.setdefault((r["population"], r.get("horizon")), []).append(r)
    cells: dict = {}
    for (pop, h), rs in sorted(by_pop.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        tag = f"{pop}_{h}d" if h is not None else pop
        cells[f"{tag}|ALL"] = rs
        if pop.startswith("CANDIDATE"):
            cells[f"{tag}|LATE_STAGE"] = [r for r in rs if (r.get("meta") or {}).get("stage") in ("LATE", "CHAMPION")]
            cells[f"{tag}|CHAMPION"] = [r for r in rs if (r.get("meta") or {}).get("stage") == "CHAMPION"]
            for dim in ("vix_structure", "signal_type"):
                for val in sorted({str((r.get("meta") or {}).get(dim)) for r in rs}):
                    cells[f"{tag}|{dim}={val}"] = [r for r in rs if str((r.get("meta") or {}).get(dim)) == val]
    k = max(1, len(cells))
    a_eff = alpha / k
    out_cells = {name: _cell(rs, a_eff) for name, rs in cells.items()}
    baselines = {}
    for (pop, h), rs in by_pop.items():
        if not pop.startswith("CANDIDATE"):
            continue
        tag = f"{pop}_{h}d"
        late = [r for r in rs if (r.get("meta") or {}).get("stage") in ("LATE", "CHAMPION")]
        early = [r for r in rs if (r.get("meta") or {}).get("stage") == "EARLY"]
        baselines[f"{tag}|LATE_STAGE_vs_EARLY"] = _diff(late, early, a_eff)
    for h in OPTION_HORIZONS:                                     # Option vs. Underlying-Proxy derselben Signale
        opt = {(r.get("meta") or {}).get("signal_id"): r for r in by_pop.get(("CANDIDATE_OPTION", h), [])}
        und = [r for r in by_pop.get(("CANDIDATE", h), []) if (r.get("meta") or {}).get("signal_id") in opt]
        if opt and und:
            baselines[f"CANDIDATE_OPTION_{h}d|vs_UNDERLYING_SAME_SIGNALS"] = _diff(
                [opt[(r.get("meta") or {}).get("signal_id")] for r in und], und, a_eff)
    champ = [r for r in recs if r["population"] == "CHAMPION_TRADE" and r.get("outcome") is not None]
    ev_counts: dict = {}
    for r in champ:
        ev = (r.get("meta") or {}).get("evaluation") or {}
        key = f"{ev.get('status')}/{ev.get('tier')}"
        ev_counts[key] = ev_counts.get(key, 0) + 1
    reasons: dict = {}
    for r in champ:
        for x in ((r.get("meta") or {}).get("evaluation") or {}).get("reasons") or []:
            reasons[x] = reasons.get(x, 0) + 1
    res = {"version": SCORECARD_VERSION, "as_of": str(as_of), "alpha": alpha, "n_cells": k,
           "alpha_per_cell": round(a_eff, 6), "min_n": MIN_N, "min_signal_days": MIN_DATES,
           "note": "netto nach Kosten; Unabhängigkeit = Signaltage; positives Ergebnis ist kein Alpha-Nachweis",
           "cells": out_cells, "baselines": baselines,
           "champion_trades": {"closed": len(champ), "evaluation": ev_counts, "not_evaluable_reasons": reasons},
           "option_mark_coverage": option_mark_coverage(as_of=as_of)}
    if write:
        from modules.atomic_io import atomic_write_json
        atomic_write_json(Path(path or SCORECARD_PATH), res, indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "scorecard":
        sc = scorecard(write=True)
        print(_json.dumps({k: sc[k] for k in ("as_of", "n_cells", "champion_trades", "option_mark_coverage")},
                          indent=1, ensure_ascii=False))
    else:
        print(_json.dumps(summary(), indent=1, ensure_ascii=False))
