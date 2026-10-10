"""modules/expectation_alpha – Expectation Alpha (EA), V1 SHADOW. Architektur: docs/EXPECTATION_ALPHA_ARCHITECTURE.md

Öffentliche Hooks (alle fehlertolerant; ein Fehler ändert nie eine Champion-Entscheidung):
  enrich_candidates(analyses, ...)   pipeline.py nach Stufe 4. Liest analyses read-only und gibt nur
                                     eine Zusammenfassung zurück; Kandidaten und Entscheidungen bleiben
                                     unverändert.
  record_downstream(obs, ...)        pipeline.py Laufende. Hält beschreibend fest, was der Champion tat.
  resolve_outcomes(...)              feedback.py: fällige Outcomes, append-only.
  evaluate(...)                      feedback.py: Gruppen, Werte, Vorschläge. Es wird nie etwas
                                     automatisch angewendet.
"""
from __future__ import annotations

import copy
import logging
import statistics
import time as _time
from datetime import datetime, timezone
from pathlib import Path

from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha.schemas import (ABSTAIN, ERROR, SCHEMA_VERSION, STAGE, TRADE, WAIT,
                                               canonical_hash, rnd)

log = logging.getLogger(__name__)

__all__ = ["enrich_candidates", "build_context", "record_downstream", "resolve_outcomes", "evaluate", "STAGE"]


def _code_version() -> str:
    from modules.promotion_controller import code_version
    return code_version()


def build_context(decision_time: datetime, cfg: dict, *, loaders: dict | None = None) -> dict:
    """Ein Kontext je Lauf: Gaps, Regime, Marktsignale, Domänen-Bestätigung, Provenienz, Fehler."""
    from modules import production_intelligence_adapter as pia
    from modules.expectation_alpha import cross_asset_confirmation as cac
    from modules.expectation_alpha import data as eadata
    from modules.expectation_alpha import expectation_gap as eg
    from modules.expectation_alpha import regime_change as rc
    loaders = loaders or {}
    t0 = _time.monotonic()
    inp = eadata.load_inputs(decision_time, cfg, archive_fn=loaders.get("archive"), prices_fn=loaders.get("prices"),
                             commodity_fn=loaders.get("commodity"))
    dates = eadata.weekly_grid(inp.px)
    gaps, regime, signals, dconf = {}, {"status": "UNAVAILABLE", "dimensions": {}, "regime_uncertainty": None}, \
        {"date": None, "values": {}}, {}
    try:
        gaps = eg.domain_gaps(inp, dates, cfg)
    except Exception as e:  # noqa: BLE001 – Fehler wird sichtbar (ERROR-Kontext), nie stiller Default
        inp.errors.add("expectation_gap", e)
    try:
        if dates:
            _ind, states = rc.build_regime(inp.px, inp.archive_obs, dates)
            regime = rc.regime_snapshot(states, cfg)
    except Exception as e:  # noqa: BLE001
        inp.errors.add("regime", e)
    try:
        signals = cac.market_signals(inp.px, inp.commodity, int((cfg.get("confirmation") or {}).get("window_days", 20)),
                                     (cfg.get("warmup") or {}).get("signal_min_rows"))
        for dom, g in gaps.items():
            if g.get("status") == "OK" and g.get("sign"):
                spec, amb = cac.domain_spec(dom, g["sign"], cfg)
                dconf[dom] = cac.confirm(spec, signals.get("values") or {}, amb, dampening=cac.dampening_of(cfg))
    except Exception as e:  # noqa: BLE001
        inp.errors.add("confirmation", e)
    vix = None
    if not inp.px.empty and "^VIX" in inp.px and len(inp.px["^VIX"].dropna()):
        vix = float(inp.px["^VIX"].dropna().iloc[-1])
    t = inp.decision_time
    ctx = {"schema_version": SCHEMA_VERSION, "mode": cfg.get("mode"), "version": cfg.get("version"),
           "decision_time": t.isoformat(timespec="seconds"), "date": t.date().isoformat(),
           "grid_end": dates[-1].date().isoformat() if dates else None, "gaps": gaps, "regime": regime,
           "signals": signals, "domain_confirmation": dconf, "vix": rnd(vix, 3), "vix_regime": pia.regime_label(vix),
           "errors": inp.errors.items, "data_provenance": inp.provenance, "config_hash": eacfg.config_hash(cfg),
           "sector_map_hash": eacfg.sector_map_hash(cfg), "code_version": _code_version(),
           "build_seconds": round(_time.monotonic() - t0, 2)}
    core = {k: ctx[k] for k in ("gaps", "regime", "signals", "vix", "grid_end", "config_hash")}
    ctx["context_hash"] = canonical_hash(core)
    ctx["context_id"] = canonical_hash({"t": ctx["decision_time"], "h": ctx["context_hash"]})
    return ctx


def _run_errors(ctx: dict, budget_exceeded: bool) -> list[str]:
    errs = []
    if budget_exceeded:
        errs.append("LAUFZEITBUDGET_UEBERSCHRITTEN")
    # Kurse und Archiv sind Pflichtquellen: ohne sie wäre jede Einstufung ein Teilbild -> ERROR (nie ABSTAIN/WAIT)
    fatal = {e["where"] for e in ctx.get("errors") or []} & {"prices", "archive"}
    if fatal:
        errs.append("DATENFEHLER: " + ", ".join(sorted(fatal)))
    return errs


def _distribution(vals: list) -> dict:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": rnd(v[0], 4), "median": rnd(statistics.median(v), 4), "max": rnd(v[-1], 4)}


def _attach_claims(rows: list[dict], cands: list[dict], cfg: dict, t: datetime, loaders: dict | None,
                   deadline: float | None) -> dict:
    """Maschinell prüfbare Claims je These (SHADOW). LLM strukturiert nur Text (separater Call); Status entscheidet
    claims.verify deterministisch gegen PIT-Evidenz (< decision_time). Ergebnis nur im Ledger, nie in Entscheidungen."""
    from modules.expectation_alpha import claim_extraction as cx
    from modules.expectation_alpha import claims as cl
    llm_fn = (loaders or {}).get("claims_llm")
    counts: dict[str, int] = {}
    for a in cands:
        counts[a.get("ticker")] = counts.get(a.get("ticker"), 0) + 1
    dup = {tk for tk, n in counts.items() if n > 1}          # mehrere Events je Ticker: Zuordnung nicht eindeutig
    by_t = {a.get("ticker"): a for a in cands if a.get("ticker") not in dup}
    todo = [(r["ticker"], by_t[r["ticker"]]) for r in rows if r["status"] != ERROR and r["ticker"] in by_t]
    res = cx.extract_many(todo, cfg, llm_fn=llm_fn, deadline=deadline) if todo else {}
    stores = (loaders or {}).get("claims_stores")
    statuses: dict[str, int] = {}
    for r in rows:
        x = res.get(r["ticker"]) if r["ticker"] not in dup else None
        if x is None:
            block = {"summary": cl.summarize(None, extraction_status="NOT_RUN"),
                     "reason": "Ticker mehrfach im Lauf (Zuordnung nicht eindeutig)" if r["ticker"] in dup
                     else "Research-Status ERROR"}
        elif x["status"] != "OK":
            block = {"summary": cl.summarize(None, extraction_status=x["status"]), "reason": x.get("reason")}
        else:
            texts = cl.source_texts(by_t.get(r["ticker"]) or {})
            norm, dropped = cl.normalize(x["raw_claims"], texts, r["ticker"])
            if stores is None:
                stores = cl.load_stores()
            ev = cl.load_evidence(r["ticker"], t, stores=stores)
            ver = cl.verify(norm, ev)
            block = {"summary": cl.summarize(ver, extraction_status="OK"), "claims": ver, "dropped": dropped,
                     "evidence_sources": ev["sources"], "evidence_error": ev.get("error"), "model": x.get("model"),
                     "store_errors": (stores or {}).get("errors") or []}
        st = block["summary"]["extraction_status"]
        statuses[st] = statuses.get(st, 0) + 1
        r["claims"] = block
        r["env"]["ea_verified_claim_fraction"] = block["summary"]["verified_claim_fraction"]
    sums = [r["claims"]["summary"] for r in rows if r["claims"]["summary"]["extraction_status"] == "OK"]
    return {"extraction_status_counts": statuses,
            "n_claims": sum(x["n_claims"] for x in sums), "n_verified": sum(x["n_verified"] for x in sums),
            "n_unverified": sum(x["n_unverified"] for x in sums), "n_contradicted": sum(x["n_contradicted"] for x in sums),
            "verified_claim_fraction": _distribution([x["verified_claim_fraction"] for x in sums])}


def insufficient_history(ctx: dict) -> dict:
    """Warm-up-Zähler je Lauf: Gaps, RoC-Felder (Gap/Modell), Regime, Cross-Asset-Signale mit INSUFFICIENT_HISTORY."""
    out = {"gaps": 0, "roc_fields": 0, "regime": 0, "signals": 0}
    for g in (ctx.get("gaps") or {}).values():
        out["gaps"] += g.get("status") == "INSUFFICIENT_HISTORY"
        for roc in (g.get("gap_roc"), (g.get("model") or {}).get("roc")):
            out["roc_fields"] += len((roc or {}).get("insufficient_history_fields") or [])
    reg = ctx.get("regime") or {}
    out["regime"] = int(reg.get("regime_uncertainty_status") == "INSUFFICIENT_HISTORY") + sum(
        1 for d in (reg.get("dimensions") or {}).values() if d.get("dynamics_status") == "INSUFFICIENT_HISTORY")
    out["signals"] = sum(1 for v in ((ctx.get("signals") or {}).get("missing_reason") or {}).values()
                         if v == "INSUFFICIENT_HISTORY")
    out["total"] = sum(out.values())
    return out


def _eff_vs_raw(rows: list[dict]) -> dict:
    """Summe effektiver vs. roher Bestätigungen (familiengedämpft); Quote < 1 = redundante Evidenz entfernt."""
    raw = sum((r.get("cross_asset_confirmation") or {}).get("raw_confirmation_count") or 0 for r in rows)
    eff = sum((r.get("cross_asset_confirmation") or {}).get("effective_confirmation") or 0.0 for r in rows)
    return {"raw": raw, "effective": rnd(eff, 3), "ratio": rnd(eff / raw, 4) if raw else None}


def enrich_candidates(analyses: list[dict], *, decision_time: datetime | None = None, impact_min: int = 4,
                      surprise_min: int = 3, deadline: float | None = None, cfg: dict | None = None,
                      loaders: dict | None = None, root: Path | None = None, contracts: list[dict] | None = None,
                      registry: Path | None = None) -> dict:
    """SHADOW: je Deep-Analysis-Kandidat eine These im EA-Ledger. Rückgabe: nur Zusammenfassung.
    `analyses` wird nicht verändert (Arbeit auf einer tiefen Kopie)."""
    from modules.expectation_alpha import ledger as eal
    from modules.expectation_alpha import thesis as th
    t0 = _time.monotonic()
    cfg = cfg if cfg is not None else eacfg.load()
    if cfg.get("mode") != "shadow":
        return {"mode": cfg.get("mode"), "enabled": False}
    t = decision_time or datetime.now(timezone.utc)
    t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    cands = copy.deepcopy(list(analyses or []))
    budget = float(cfg.get("max_runtime_seconds", 150))
    summary: dict = {"mode": "shadow", "enabled": True, "production_influence": "NONE",
                     "candidate_count": len(cands)}
    if deadline is not None and _time.monotonic() + budget > deadline:
        ctx = {"schema_version": SCHEMA_VERSION, "decision_time": t.isoformat(timespec="seconds"),
               "date": t.date().isoformat(), "gaps": {}, "regime": {}, "signals": {}, "errors": [
                   {"where": "budget", "error": "Scanner-Laufzeitbudget reicht nicht für EA"}],
               "config_hash": eacfg.config_hash(cfg), "sector_map_hash": eacfg.sector_map_hash(cfg),
               "code_version": _code_version()}
        ctx["context_hash"] = canonical_hash(ctx)
        ctx["context_id"] = canonical_hash({"t": ctx["decision_time"], "h": ctx["context_hash"]})
        budget_exceeded = True
    else:
        ctx = build_context(t, cfg, loaders=loaders)
        budget_exceeded = (_time.monotonic() - t0) > budget
    errs = _run_errors(ctx, budget_exceeded)
    rows = []
    for a in cands:
        if not a.get("ticker"):
            continue
        try:
            rows.append(th.build_thesis(a, ctx, cfg, impact_min=impact_min, surprise_min=surprise_min,
                                        run_errors=errs))
        except Exception as e:  # noqa: BLE001 – Kandidat wird als ERROR protokolliert, nie verschluckt
            log.warning(f"expectation_alpha: These {a.get('ticker')} nicht ableitbar ({type(e).__name__}: {e})")
            summary.setdefault("candidate_errors", []).append({"ticker": a.get("ticker"), "error": str(e)[:200]})
    claims_sum: dict = {}
    if rows:
        try:
            cdl = t0 + budget
            claims_sum = _attach_claims(rows, cands, cfg, t, loaders, min(cdl, deadline) if deadline else cdl)
        except Exception as e:  # noqa: BLE001 – Claim-Logging darf die These nie verhindern
            log.warning(f"expectation_alpha: Claims nicht ableitbar ({type(e).__name__}: {e})")
            claims_sum = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            for r in rows:
                r.setdefault("claims", {"summary": {"extraction_status": "ERROR", "n_claims": None,
                                                    "verified_claim_fraction": None}})
                r["env"].setdefault("ea_verified_claim_fraction", None)
    eal.write_context(ctx, root)
    new = eal.record_candidates(rows, root=root, contracts=contracts, registry=registry)
    by = {s: sum(1 for r in rows if r["status"] == s) for s in (TRADE, WAIT, ABSTAIN, ERROR)}
    gaps = ctx.get("gaps") or {}
    summary.update({
        "date": ctx["date"], "context_id": ctx["context_id"], "enriched_count": len(rows), "recorded": len(new),
        "status_counts": by, "shadow_trade_count": by[TRADE], "wait_count": by[WAIT], "abstain_count": by[ABSTAIN],
        "error_count": by[ERROR],
        "missing_counts": {d: g.get("status") for d, g in gaps.items() if g.get("status") != "OK"},
        "stale_components": sorted({c for g in gaps.values() for side in ("model", "market")
                                    for c, fv in ((g.get(side) or {}).get("components") or {}).items()
                                    if fv.get("status") == "STALE"}),
        "gap_z": {d: g.get("gap_z") for d, g in gaps.items()},
        "confirmation_ratio": _distribution([r["confirmation_ratio"] for r in rows]),
        "effective_vs_raw_confirmation": _eff_vs_raw(rows),
        "claims": claims_sum,
        "insufficient_history_count": insufficient_history(ctx),
        "family_diversity": _distribution([r.get("confirmation_family_count") for r in rows]),
        "regime_uncertainty": (ctx.get("regime") or {}).get("regime_uncertainty"),
        "groups": {gname: sum(1 for r in rows if r["group"] == gname) for gname in "ABCDEX"},
        "errors": ctx.get("errors") or [], "run_errors": errs,
        "runtime_seconds": round(_time.monotonic() - t0, 2),
        "observations": [{"observation_id": r["observation_id"], "ticker": r["ticker"]} for r in new]})
    try:
        eal.record_run({k: v for k, v in summary.items() if k != "observations"} |
                       {"decision_time": ctx["decision_time"], "config_hash": ctx.get("config_hash"),
                        "code_version": ctx.get("code_version")}, root)
    except OSError as e:
        log.warning(f"expectation_alpha: Lauf-Protokoll nicht schreibbar ({e})")
    return summary


def record_downstream(observations: list[dict], *, final_tickers=(), reject_stats: dict | None = None,
                      root: Path | None = None) -> int:
    from modules import final_mc_ledger as fml
    from modules.expectation_alpha import ledger as eal
    m = fml.downstream_map(final_tickers=set(final_tickers), roi_rejects=[], reject_stats=reject_stats or {})
    return eal.record_downstream(observations, m, root=root)


def resolve_outcomes(**kw) -> dict:
    from modules.expectation_alpha import ledger as eal
    return eal.resolve_outcomes(**kw)


def evaluate(**kw) -> dict:
    from modules.expectation_alpha import evaluation as ev
    return ev.run(**kw)
