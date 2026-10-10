"""modules/expectation_alpha/evaluation.py – Research-Auswertung des EA-Ledgers (nur beschreibend).

Die Promotion-Entscheidung trifft allein der PromotionController über EA001–EA007, und nur auf
Forward-Daten. Diese Auswertung zeigt:
* NEWS × CONTEXT-Gruppen A–E/X je Horizont (Rendite brutto und netto, Trefferquote, MAE/MFE, Pfad-
  Drawdown, risikoadjustiert, 95 %-Block-Bootstrap-CI über Signaltage);
* den Wert der Abstinenz, des WAIT-Triggers, der Expression und der Kill Conditions, jeweils gepaart;
* die Kalibrierung der Bestätigungsquote;
* die regelbasierte Fehlerklassifikation;
* den Datenstatus.
Lernschleife: Es werden nur Vorschläge geschrieben (proposals.json, auto_applied=false). Kein
Schwellenwert, kein Vertrag und keine Produktionswirkung ändert sich automatisch.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modules.atomic_io import atomic_write_json, atomic_write_text, read_jsonl
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import ledger as eal
from modules.expectation_alpha.schemas import ABSTAIN, ERROR, GROUPS, STAGE, TRADE, WAIT, rnd

CONF_BUCKETS = ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0001))


def _boot_ci(items: list[tuple[str, float]], n: int, seed: int) -> list:
    """95 %-CI des Mittelwerts, Block-Bootstrap über Signaltage (vektorisiert: Summen/Zählungen je Tag)."""
    by: dict[str, list[float]] = {}
    for d, v in items:
        by.setdefault(d, []).append(v)
    if len(by) < 2:
        return [None, None]
    sums = np.array([sum(v) for v in by.values()])
    cnts = np.array([len(v) for v in by.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(by), size=(n, len(by)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [round(float(lo), 5), round(float(hi), 5)]


def _mean_opt(xs) -> float | None:
    v = [x for x in xs if x is not None]
    return rnd(statistics.fmean(v), 5) if v else None


def stats_block(items: list[dict], cfg: dict, key: str = "outcome_net") -> dict:
    ev = cfg.get("evaluation") or {}
    n = len(items)
    out = {"n": n, "independent_dates": len({i["date"] for i in items})}
    if n < int(ev.get("min_n_report", 30)):
        out["status"] = "NEED_MORE_DATA"
        return out
    v = [i[key] for i in items]
    sd = statistics.stdev(v) if n > 1 else None
    out.update(status="OK", mean=rnd(statistics.fmean(v), 5), median=rnd(statistics.median(v), 5),
               hit_rate=rnd(sum(1 for x in v if x > 0) / n, 4),
               mean_gross=rnd(statistics.fmean(i["outcome"] for i in items), 5),
               mae=_mean_opt(i.get("mae") for i in items), mfe=_mean_opt(i.get("mfe") for i in items),
               path_drawdown=_mean_opt(i.get("max_drawdown") for i in items),
               risk_adjusted=rnd(statistics.fmean(v) / sd, 4) if sd else None,
               ci95=_boot_ci([(i["date"], i[key]) for i in items], int(ev.get("bootstrap_n", 2000)),
                             int(ev.get("bootstrap_seed", 47))))
    return out


def _joined(rows: list[dict], outs: dict, expression: str, variant: str, h: int) -> list[dict]:
    res = []
    for r in rows:
        e = outs.get((r["observation_id"], expression, variant, h))
        if e and e.get("outcome_net") is not None:
            res.append({**e, "date": r["date"], "row": r})
    return res


def _paired(rows, outs, h, a: tuple, b: tuple, cfg) -> dict:
    items = []
    for r in rows:
        ea = outs.get((r["observation_id"], *a, h))
        eb = outs.get((r["observation_id"], *b, h))
        if ea and eb and ea.get("outcome_net") is not None and eb.get("outcome_net") is not None:
            items.append({"date": r["date"], "outcome_net": ea["outcome_net"] - eb["outcome_net"],
                          "outcome": ea["outcome"] - eb["outcome"], "mae": None, "mfe": None, "max_drawdown": None})
    s = stats_block(items, cfg)
    s["definition"] = f"{a} minus {b}, gleicher Ausstiegstag, netto"
    return s


def _paired_sel(rows, outs, h, cfg) -> dict:
    """Regelbasierte Expression minus Default (UNDERLYING), nur wo die Regel nicht den Default wählt."""
    items = []
    for r in rows:
        sel = (r.get("selected_expression") or {}).get("selected_expression")
        ea = outs.get((r["observation_id"], sel, "immediate", h))
        eb = outs.get((r["observation_id"], "UNDERLYING", "immediate", h))
        if ea and eb and ea.get("outcome_net") is not None and eb.get("outcome_net") is not None:
            items.append({"date": r["date"], "outcome_net": ea["outcome_net"] - eb["outcome_net"],
                          "outcome": ea["outcome"] - eb["outcome"], "mae": None, "mfe": None, "max_drawdown": None})
    return {**stats_block(items, cfg), "definition": "gewählte Expression minus UNDERLYING, netto"}


def classify_failure(r: dict, outs: dict, h: int, contexts: list[dict]) -> str | None:
    """Regelbasiert; None = kein Fehlschlag (netto > 0) oder (noch) nicht bewertbar."""
    if r.get("status") == ERROR:
        return "DATA_BAD"
    u = outs.get((r["observation_id"], "UNDERLYING", "immediate", h))
    km = outs.get((r["observation_id"], "UNDERLYING", "kill_managed", h))
    if km and km.get("entry") == "DATA_BAD":
        return "DATA_BAD"
    sel = (r.get("selected_expression") or {}).get("selected_expression") or "UNDERLYING"
    s = outs.get((r["observation_id"], sel, "immediate", h))
    if not u or not s or s.get("outcome_net") is None:
        return None
    if s["outcome_net"] > 0:
        return None
    alts = [outs.get((r["observation_id"], e["expression"], "immediate", h)) for e in r.get("expressions") or []]
    vals = [e["outcome_net"] for e in alts if e and e.get("outcome_net") is not None]
    thesis_q = statistics.median(vals) if vals else u["outcome_net"]
    if thesis_q > 0:
        return "EXPRESSION_WRONG"
    dim = r.get("regime_dimension")
    st0 = (r.get("regime_state") or {}).get(dim) if dim else None
    later = [c for c in contexts if str(c.get("date")) > r["date"] and str(c.get("date")) <= (u.get("exit_date") or "")]
    if dim and st0 and later:
        st1 = ((((later[-1].get("regime") or {}).get("dimensions") or {}).get(dim)) or {}).get("state")
        if st1 and st1 != st0:
            return "REGIME_CHANGED"
    if "catalyst_failure" in ((km or {}).get("kill_events") or {}):
        return "CATALYST_WRONG"
    if (u.get("mfe") or 0) >= 0.05:
        return "TIMING_WRONG"
    return "THESIS_WRONG"


def run(*, root: Path | None = None, cfg: dict | None = None, now: datetime | None = None,
        promotion_state: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or eacfg.load()
    now = now or datetime.now(timezone.utc)
    P = eal.paths(root)
    rows = eal.read_rows(root)
    outs = eal.read_outcomes(root)
    contexts = eal.read_contexts(root)
    horizons = [int(h) for h in (cfg.get("outcomes") or {}).get("horizons") or [20, 60, 120, 250]]
    ph = int((cfg.get("outcomes") or {}).get("primary_horizon", 60))
    valid = [r for r in rows if r.get("status") != ERROR]
    rep: dict = {"generated": now.isoformat(timespec="seconds"), "mode": "SHADOW – Research, keine Handelsempfehlung",
                 "config_hash": eacfg.config_hash(cfg), "population": eal.population_summary(rows),
                 "primary_horizon": ph, "groups": {}, "status_value": {}, "values": {}}
    for h in horizons:
        u = _joined(rows, outs, "UNDERLYING", "immediate", h)
        rep["groups"][str(h)] = {g: stats_block([i for i in u if i["row"].get("group") == g], cfg) for g in GROUPS}
        rep["status_value"][str(h)] = {s: stats_block([i for i in u if i["row"].get("status") == s], cfg)
                                       for s in (TRADE, WAIT, ABSTAIN, ERROR)}
    u = _joined(valid, outs, "UNDERLYING", "immediate", ph)
    all_m = stats_block(u, cfg)
    kept = stats_block([i for i in u if i["row"].get("status") != ABSTAIN], cfg)
    rep["values"]["abstention"] = {
        "definition": "E[nicht ABSTAIN] − E[alle gültigen] (netto, Underlying, Primärhorizont)",
        "all": all_m, "kept": kept,
        "delta": rnd(kept["mean"] - all_m["mean"], 5) if all_m.get("mean") is not None and kept.get("mean")
        is not None else None}
    waits = [r for r in rows if r.get("status") == WAIT]
    trig = [outs.get((r["observation_id"], "UNDERLYING", "triggered", ph)) for r in waits]
    trig = [t for t in trig if t]
    rep["values"]["wait"] = {"paired": _paired(waits, outs, ph, ("UNDERLYING", "triggered"),
                                               ("UNDERLYING", "immediate"), cfg),
                             "n_resolved": len(trig),
                             "no_entry_share": rnd(sum(1 for t in trig if t.get("entry") == "NO_ENTRY") / len(trig), 4)
                             if trig else None}
    sel_rows = [r for r in valid if (r.get("selected_expression") or {}).get("selected_expression") != "UNDERLYING"]
    tq, eq = [], []
    for r in valid:
        vals = {e["expression"]: outs.get((r["observation_id"], e["expression"], "immediate", ph))
                for e in r.get("expressions") or []}
        vals = {k: v["outcome_net"] for k, v in vals.items() if v and v.get("outcome_net") is not None}
        sel = (r.get("selected_expression") or {}).get("selected_expression")
        if len(vals) >= 2 and sel in vals:
            med = statistics.median(vals.values())
            tq.append({"date": r["date"], "outcome_net": med, "outcome": med, "mae": None, "mfe": None,
                       "max_drawdown": None})
            eq.append({"date": r["date"], "outcome_net": vals[sel] - med, "outcome": vals[sel] - med, "mae": None,
                       "mfe": None, "max_drawdown": None})
    rep["values"]["expression"] = {
        "thesis_quality": {**stats_block(tq, cfg), "definition": "Median der Netto-Renditen aller registrierten Expressions"},
        "expression_quality": {**stats_block(eq, cfg), "definition": "gewählte Expression minus Median"},
        "rule_vs_default": _paired_sel(sel_rows, outs, ph, cfg)}
    rep["values"]["kill_management"] = _paired(valid, outs, ph, ("UNDERLYING", "kill_managed"),
                                               ("UNDERLYING", "immediate"), cfg)
    cal = []
    for lo, hi in CONF_BUCKETS:
        b = [i for i in u if i["row"].get("confirmation_ratio") is not None and lo <= i["row"]["confirmation_ratio"] < hi]
        cal.append({"bucket": f"[{lo:.1f},{min(hi, 1.0):.1f}]", **stats_block(b, cfg)})
    rep["calibration_confirmation"] = cal
    fails: dict[str, int] = {}
    for r in rows:
        f = classify_failure(r, outs, ph, contexts)
        if f:
            fails[f] = fails.get(f, 0) + 1
    rep["failure_classes"] = fails
    runs = read_jsonl(P["runs"])[-20:]
    rep["data_status"] = {"runs": len(runs), "error_runs": sum(1 for x in runs if x.get("error_count")),
                          "missing_domains": _count([d for x in runs for d in (x.get("missing_counts") or {})]),
                          "stale_components": _count([c for x in runs for c in (x.get("stale_components") or [])]),
                          "median_runtime_seconds": rnd(statistics.median([x["runtime_seconds"] for x in runs
                                                                           if x.get("runtime_seconds") is not None]), 2)
                          if any(x.get("runtime_seconds") is not None for x in runs) else None}
    rep["contracts"] = contract_status(promotion_state)
    from modules.expectation_alpha import lead_lag
    rep["lead_lag"] = lead_lag.run(rows, outs, cfg)        # nur Research: kein Gewicht, keine Horizont-Auswahl
    props = proposals(rep, rows, outs, cfg)
    if write:
        atomic_write_json(P["evaluation_json"], rep, indent=1, ensure_ascii=False, default=str)
        atomic_write_text(P["evaluation_md"], render_md(rep))
        atomic_write_json(P["proposals"], props, indent=1, ensure_ascii=False, default=str)
    rep["proposals"] = props
    return rep


def _count(xs: list) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def contract_status(state: dict | None = None) -> list[dict]:
    if state is None:
        p = Path("outputs/intelligence/promotion_state.json")
        state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    out = []
    for k, h in sorted((state.get("hypotheses") or {}).items()):
        if h.get("eligible_stage") != STAGE:
            continue
        ev = h.get("evidence") or {}
        out.append({"key": k, "title": h.get("title"), "state": h.get("state"), "decision": h.get("decision"),
                    "influence_level": h.get("influence_level"), "n": ev.get("n_observations"),
                    "independent_dates": ev.get("n_independent_dates"), "span_days": ev.get("calendar_span_days"),
                    "regimes": ev.get("regimes"), "delta": ev.get("delta_expectancy"), "ci": ev.get("ci"),
                    "next_requirement": h.get("next_requirement"), "forward_start": h.get("forward_start")})
    return out


def proposals(rep: dict, rows: list[dict], outs: dict, cfg: dict) -> dict:
    """Konservative Lernschleife: nur Vorschläge, nie angewendet. Challenger erst ab Mindestdaten."""
    ev = cfg.get("evaluation") or {}
    n_res = sum(1 for (_o, ex, v, h) in outs if ex == "UNDERLYING" and v == "immediate"
                and h == rep["primary_horizon"])
    dates = len({r["date"] for r in rows})
    items = []
    cal = [c for c in rep.get("calibration_confirmation") or [] if c.get("status") == "OK"]
    if len(cal) >= 3 and any(cal[i]["mean"] > cal[i + 1]["mean"] for i in range(len(cal) - 1)):
        items.append({"type": "CALIBRATION_REVIEW", "what": "Bestätigungsquote nicht monoton zur Netto-Rendite",
                      "action": "menschliche Prüfung; ggf. neue Vertragsversion (nie Schwellenänderung in v1)"})
    ready = n_res >= int(ev.get("challenger_min_n", 300)) and dates >= int(ev.get("challenger_min_independent_dates", 120))
    return {"generated": rep["generated"], "auto_applied": False, "status": "PROPOSAL_ONLY",
            "challenger": {"status": "READY_FOR_HUMAN_REVIEW" if ready else "NEED_MORE_DATA",
                           "resolved": n_res, "independent_dates": dates,
                           "required": {"n": ev.get("challenger_min_n"),
                                        "independent_dates": ev.get("challenger_min_independent_dates")},
                           "note": "kombinierter Score/Challenger erst nach Mindestdaten; Registrierung nur per PR"},
            "items": items,
            "never_automatic": ["Konstitution", "Risk Limits", "PIT-Regeln", "Promotion-Kriterien",
                                "Produktionseinfluss", "Champion"]}


def _fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{x * 100:+.2f}%" if pct else f"{x}"


def render_md(rep: dict) -> str:
    ph = str(rep["primary_horizon"])
    pop = rep["population"]
    L = [f"# Expectation Alpha – {rep['mode']}", "", f"Stand {rep['generated']} · config {rep['config_hash']}", "",
         f"Population {STAGE}: n={pop['n']}, Signaltage {pop['dates']}, Ereignis-Cluster {pop['event_clusters']}, "
         f"Status {pop['by_status']}", "", f"## NEWS × CONTEXT (Underlying, netto, {ph} Handelstage)", "",
         "| Gruppe | n | Status | Mittel | Median | Treffer | MAE | MFE | CI95 |", "|---|---|---|---|---|---|---|---|---|"]
    for g, s in rep["groups"][ph].items():
        L.append(f"| {g} | {s['n']} | {s.get('status')} | {_fmt(s.get('mean'))} | {_fmt(s.get('median'))} | "
                 f"{_fmt(s.get('hit_rate'), False)} | {_fmt(s.get('mae'))} | {_fmt(s.get('mfe'))} | {s.get('ci95', '–')} |")
    L += ["", "## Werte (gepaart, netto)", ""]
    for k, v in rep["values"].items():
        if k == "expression":
            for kk, vv in v.items():
                L.append(f"- expression.{kk}: n={vv.get('n')} {vv.get('status')} Mittel {_fmt(vv.get('mean'))}")
        elif k == "abstention":
            L.append(f"- abstention: Δ {_fmt(v.get('delta'))}")
        elif k == "wait":
            L.append(f"- wait: n={v['paired'].get('n')} {v['paired'].get('status')} Δ {_fmt(v['paired'].get('mean'))}, "
                     f"ohne Einstieg {v.get('no_entry_share')}")
        else:
            L.append(f"- {k}: n={v.get('n')} {v.get('status')} Δ {_fmt(v.get('mean'))}")
    L += ["", "## Verträge EA001–EA007", "", "| Vertrag | Zustand | n | Tage | Spanne | Regime | Δ | CI | nächste Anforderung |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in rep["contracts"]:
        L.append(f"| {c['key']} | {c['state']} | {c['n']} | {c['independent_dates']} | {c['span_days']} | "
                 f"{len(c.get('regimes') or [])} | {_fmt(c.get('delta'))} | {c.get('ci')} | {c.get('next_requirement') or '–'} |")
    ll = rep.get("lead_lag") or {}
    L += ["", f"## Lead-Lag-Diagnostik (Research, Horizonte vorab: {ll.get('horizons_preregistered')}, "
              f"keine Auswahl) – Status {ll.get('lead_lag_status')}", "",
          "| Feature | Familie | " + " | ".join(f"IC {h}T (n)" for h in ll.get("horizons_preregistered") or []) + " |",
          "|---|---|" + "---|" * len(ll.get("horizons_preregistered") or [])]
    for name, f in (ll.get("features") or {}).items():
        cells = []
        for h in ll.get("horizons_preregistered") or []:
            b = f["horizons"].get(str(h)) or {}
            cells.append(f"{b.get('ic') if b.get('status') == 'OK' else '–'} ({b.get('n', 0)})")
        L.append(f"| {name} | {f['family']} | " + " | ".join(cells) + " |")
    L += ["", f"Fehlerklassen: {rep['failure_classes'] or '–'}", "",
          "Hinweis: SHADOW. Keine Handelsempfehlung, kein Einfluss auf Champion, Scores, Gates oder Sizing."]
    return "\n".join(L) + "\n"
