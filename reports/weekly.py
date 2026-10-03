"""
reports/weekly.py – Weekly Intelligence Report.

Liest vorhandene Forschungs-/Paper-Ergebnisse aus outputs/ (alle Eingaben
optional; fehlende Dateien -> Abschnitt zeigt "keine Daten"/"n/a", nie
erfundene Werte) und rendert HTML + Text + Markdown. Optionaler Versand über
modules.mailer.

    python -m reports.weekly --dry-run   # rendern + nach outputs/reports/ schreiben, kein Versand
    python -m reports.weekly --send      # zusätzlich Mail senden, Snapshot fortschreiben

WICHTIG: Forward-/Paper-Performance (outputs/history.json, echte Paper-Trades)
wird strikt getrennt von Backtest-/Walk-Forward-Ergebnissen (ml_research,
meta_learning) dargestellt und beschriftet.

Research-/Paper-Signale, keine Orderausführung, keine Anlageberatung.
"""
from __future__ import annotations

import argparse
import html as _html
import json
import logging
import statistics
import sys
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Schwellen / Konstanten ─────────────────────────────────────────────────
LOW_SAMPLE_N = 20                 # n < 20 -> "LOW SAMPLE"
DEGRADATION_MIN_N = 5             # min. Trades im 4-Wochen-Fenster für PERFORMANCE DEGRADATION
STALE_DAYS_WARN = 10              # Forschungsartefakt älter als X Tage -> Hinweis
WINDOWS = (("Seit Beginn", None), ("Letzte 12 Monate", 365),
           ("Letzte 3 Monate", 91), ("Letzte 4 Wochen", 28))
RECENT_DAYS = 28
DISCLAIMER = "Research-/Paper-Signale, keine Orderausführung, keine Anlageberatung."
NO_HC_TEXT = "No high-confidence candidates this week."
FIRST_REPORT_TEXT = "Erster Bericht – keine Vergleichsbasis."
NO_DATA = "keine Daten"
NA = "n/a"
META_METRICS = ("cagr", "sharpe", "sortino", "max_dd", "calmar", "profit_factor",
                "hit_rate", "expectancy", "brier", "ece")
# Montagsbericht: 7 Hauptabschnitte (Spezifikation), danach die Detailabschnitte als Anhang A1–A19.
MONDAY_TITLES = {
    1: "SYSTEM STATUS",
    2: "WHAT THE SYSTEM LEARNED",
    3: "HYPOTHESIS SCOREBOARD",
    4: "RESEARCH INTELLIGENCE",
    5: "CURRENT MARKET / WORLD MODEL",
    6: "TOP TRADE CANDIDATES",
    7: "PERFORMANCE",
}
NO_TRADE_TEXT = "NO HIGH-CONFIDENCE TRADE THIS WEEK."
# Berichts-Label (kein Trade-Gate): HIGH-CONFIDENCE nur, wenn das kalibrierte MC-Band des Kandidaten
# auf echten Paper-Trades belegt ist (n >= 20, Expectancy > 0, Profit Factor >= 1,2) und kein Safe Mode gilt.
HC_BAND_MIN_N, HC_BAND_MIN_PF, CAL_MIN_N = 20, 1.2, 10
SCOREBOARD_GROUPS = ("RESEARCH IDEA", "CHALLENGER", "FORWARD VALIDATED", "PROMOTED", "REJECTED")
LEARN_DAYS = 7

SECTION_TITLES = {
    1: "SYSTEM STATUS",
    2: "LIVE / FORWARD PERFORMANCE (Paper-Trades, echt – kein Backtest)",
    3: "CONFIDENCE PERFORMANCE",
    4: "CURRENT MODEL INTELLIGENCE",
    5: "META-LEARNING STATUS (Backtest, kein Forward)",
    6: "CURRENT HIGH-CONFIDENCE CANDIDATES",
    7: "RECENT SIGNAL REVIEW",
    8: "WHAT THE SYSTEM LEARNED THIS WEEK",
    9: "RESEARCH PIPELINE",
    10: "RISK / HEALTH WARNINGS",
    11: "WORLD MODEL",
    12: "META-COGNITION",
    13: "ALPHA HEALTH",
    14: "RESEARCH INTELLIGENCE",
    15: "MODEL BLIND SPOTS",
    16: "ACTIVE LEARNING",
    17: "ALTERNATIVE DATA INTELLIGENCE",
    18: "PROMOTION STATUS (Research → Production)",
    19: "RESEARCH FACTORY (Hypothesen, Richtungen, Datenlücken)",
}


# ── Laden ──────────────────────────────────────────────────────────────────
def _load_json(path: Path):
    """JSON laden; fehlend -> None (debug), kaputt -> None (Warnung)."""
    if not path.exists():
        log.debug("Eingabe fehlt: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("Eingabe nicht lesbar (%s): %s", path.name, e)
        return None


def _load_jsonl(path: Path) -> list[dict]:
    rows, bad = [], 0
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError as e:
        log.warning("JSONL nicht lesbar (%s): %s", path.name, e)
    if bad:
        log.warning("%s: %d unlesbare Zeilen übersprungen", path.name, bad)
    return rows


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN
        return None
    return float(x)


def _parse_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s[:10]).date()
    except ValueError:
        return None


# ── Forward-Metriken ───────────────────────────────────────────────────────
def forward_metrics(outcomes: list[float]) -> dict:
    """Kennzahlen aus einer Liste von Trade-Outcomes (Reihenfolge = close_date).

    Konvention Max-Drawdown: kumulierte Outcome-Kurve, additiv je Trade als
    Anteil einer Einheit (Start 0, cum += outcome); MaxDD = min(cum - laufendes
    Maximum inkl. Start 0), negativ, in Einheiten (-1.0 = eine ganze Einheit)."""
    n = len(outcomes)
    m = {"closed": n, "low_sample": n < LOW_SAMPLE_N}
    if n == 0:
        return m
    wins = [o for o in outcomes if o > 0]
    losses = [o for o in outcomes if o <= 0]
    m["win_rate"] = len(wins) / n
    m["avg_return"] = sum(outcomes) / n
    m["median"] = statistics.median(outcomes)
    m["avg_winner"] = sum(wins) / len(wins) if wins else None
    m["avg_loser"] = sum(losses) / len(losses) if losses else None
    if m["avg_winner"] is not None and m["avg_loser"]:
        m["payoff_ratio"] = m["avg_winner"] / abs(m["avg_loser"])
    else:
        m["payoff_ratio"] = None
    gl = abs(sum(losses))
    m["profit_factor"] = (sum(wins) / gl) if gl > 0 else None
    m["expectancy"] = m["avg_return"]  # = win_rate*avg_win + (1-win_rate)*avg_loss
    sd = statistics.stdev(outcomes) if n > 1 else None
    m["sharpe_per_trade"] = (m["avg_return"] / sd) if sd else None          # je Trade, nicht annualisiert
    down = [min(o, 0.0) for o in outcomes]
    dd_sd = (sum(x * x for x in down) / n) ** 0.5
    m["sortino_per_trade"] = (m["avg_return"] / dd_sd) if dd_sd > 0 else None
    cum = peak = dd = 0.0
    for o in outcomes:
        cum += o
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    m["max_dd"] = dd
    return m


def _closed_trades(history: dict) -> list[dict]:
    out = []
    for t in _d(history).get("closed_trades") or []:
        if isinstance(t, dict) and _num(t.get("outcome")) is not None:
            out.append(t)
    return out


def compute_forward(history: dict | None, today: _date) -> dict:
    """Fenster-Kennzahlen (alle / zuverlässig) + Signalanzahl."""
    if not history:
        return {"available": False}
    closed = _closed_trades(history)
    active = [t for t in _d(history).get("active_trades") or [] if isinstance(t, dict)]
    res = {"available": True, "n_closed_total": len(closed), "n_active": len(active),
           "n_reconstructed": sum(1 for t in closed if t.get("outcome_method_reconstructed") == "delta_approx"),
           "windows": []}
    for label, days in WINDOWS:
        cutoff = today - timedelta(days=days) if days else None

        def in_win(t):
            if cutoff is None:
                return True
            cd = _parse_date(t.get("close_date"))
            return cd is not None and cd >= cutoff

        sel = [t for t in closed if in_win(t)]
        sel.sort(key=lambda t: _parse_date(t.get("close_date")) or _date.max)
        rel = [t for t in sel if t.get("outcome_method_reconstructed") != "delta_approx"]
        # Signale: Trades (offen + geschlossen) mit entry_date im Fenster
        def sig(t):
            if cutoff is None:
                return True
            ed = _parse_date(t.get("entry_date"))
            return ed is not None and ed >= cutoff
        n_sig = sum(1 for t in closed + active if sig(t))
        res["windows"].append({
            "label": label, "signals": n_sig,
            "all": forward_metrics([float(t["outcome"]) for t in sel]),
            "reliable": forward_metrics([float(t["outcome"]) for t in rel]),
        })
    return res


# ── Health ─────────────────────────────────────────────────────────────────
def summarize_health(health: dict | None, today: _date) -> dict:
    if not isinstance(health, dict) or not health:
        return {"available": False}
    counts = {"PASS": 0, "WARNING": 0, "FAIL": 0, "DEFERRED": 0}
    failing, warning = [], []
    stale, last_ok = {}, []
    for sid, h in health.items():
        if not isinstance(h, dict):
            continue
        st = str(h.get("status", "")).upper()
        if st == "PASS":
            counts["PASS"] += 1
        elif st in ("FAIL", "FAILED", "ERROR"):
            counts["FAIL"] += 1
            failing.append(sid)
        elif st in ("WARN", "WARNING", "SCHEMA_CHANGED", "STALE"):
            counts["WARNING"] += 1
            warning.append(sid)
        elif st == "DEFERRED":
            counts["DEFERRED"] += 1
        sv = h.get("staleness")
        if sv and st != "DEFERRED":
            stale[str(sv)] = stale.get(str(sv), 0) + 1
        ls = _parse_date(h.get("last_success"))
        if ls and st != "DEFERRED":
            last_ok.append(ls)
    overall = "FAIL" if counts["FAIL"] else "WARNING" if counts["WARNING"] else "PASS"
    return {"available": True, "counts": counts, "overall": overall, "failing": failing,
            "warning": warning, "staleness": stale,
            "oldest_success": min(last_ok).isoformat() if last_ok else None,
            "newest_success": max(last_ok).isoformat() if last_ok else None}


# ── Drift ──────────────────────────────────────────────────────────────────
def _drift_flag(v) -> bool | None:
    """Interpretiert einen Drift-Eintrag: bool, dict mit flag/drift/alert, Text."""
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        for k in ("flag", "flagged", "drift", "drifted", "alert", "detected"):
            if isinstance(v.get(k), bool):
                return v[k]
        lvl = v.get("level") or v.get("status")
        if isinstance(lvl, str):
            return lvl.upper() in ("HIGH", "ALERT", "DRIFT", "WARN", "WARNING", "FAIL")
        return None
    if isinstance(v, str):
        return v.upper() in ("HIGH", "ALERT", "DRIFT", "WARN", "WARNING", "FAIL")
    return None


def drift_summary(drift) -> dict:
    """{'model': bool|None, 'feature': bool|None, 'flags': [namen]} – nur aus vorhandenen Feldern."""
    out = {"model": None, "feature": None, "flags": []}
    if not isinstance(drift, dict):
        return out
    for k, v in drift.items():
        f = _drift_flag(v)
        kl = str(k).lower()
        if f is None:
            continue
        if f:
            out["flags"].append(str(k))
        if "model" in kl or "performance" in kl:
            out["model"] = bool(out["model"]) or f
        elif "feature" in kl or "data" in kl or "input" in kl:
            out["feature"] = bool(out["feature"]) or f
    return out


# ── Snapshot / Diff ────────────────────────────────────────────────────────
def build_snapshot(data: dict) -> dict:
    ml, meta, hyp, st = data["ml"], data["meta"], data["hyp"], data["meta_state"]
    return {
        "date": data["date"],
        "hypotheses": {k: _d(v).get("status") for k, v in _d(_d(hyp).get("hypotheses")).items()},
        "model_verdicts": {k: _d(_d(v).get("decision")).get("verdict") for k, v in _d(_d(ml).get("models")).items()
                           if _d(v).get("decision")},
        "meta_weights": {k: _num(_d(v).get("meta_weight")) for k, v in _d(_d(meta).get("model_intelligence")).items()},
        "calibration_coverage": _num(_d(_d(ml).get("calibration")).get("coverage")),
        "champion": _d(ml).get("champion"),
        "safe_mode": safe_mode_status(data.get("safe"))[0],
        "meta_verdict": _d(_d(meta).get("decision")).get("verdict"),
    }


def data_stand(data: dict) -> str:
    """Zeitstempel je Artefakt + Warnung bei Alter > 8 Tage (Audit F11: Report
    zeigte unbemerkt Zahlen eines älteren Laufs)."""
    parts, stale = [], []
    today = _date.fromisoformat(data["date"])
    for name, key in (("ml_research", "ml"), ("meta_learning", "meta"), ("next_validation", "nextv"),
                      ("system_state", "safe")):
        g = _d(data.get(key)).get("generated") or _d(data.get(key)).get("updated") or _d(data.get(key)).get("updated_at")
        parts.append(f"{name} {g or NA}")
        try:
            if g and (today - _date.fromisoformat(str(g)[:10])).days > 8:
                stale.append(name)
        except ValueError:
            pass
    return "; ".join(parts) + (f" – VERALTET: {', '.join(stale)}" if stale else "")


def _fv(x):
    if x is None:
        return NA
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def diff_snapshots(prev: dict | None, cur: dict) -> list[str] | None:
    """Gemessene Änderungen; None = keine Vorwoche."""
    if not prev:
        return None
    out = []
    for hid, s in cur["hypotheses"].items():
        ps = _d(prev.get("hypotheses")).get(hid, "__missing__")
        if ps == "__missing__":
            out.append(f"Hypothese {hid}: neu ({_fv(s)})")
        elif ps != s:
            out.append(f"Hypothese {hid}: {_fv(ps)} -> {_fv(s)}")
    for hid in _d(prev.get("hypotheses")):
        if hid not in cur["hypotheses"]:
            out.append(f"Hypothese {hid}: nicht mehr vorhanden")
    for mid, v in cur["model_verdicts"].items():
        pv = _d(prev.get("model_verdicts")).get(mid, "__missing__")
        if pv == "__missing__":
            out.append(f"Modell {mid}: neu bewertet ({_fv(v)})")
        elif pv != v:
            out.append(f"Modell {mid}: Verdikt {_fv(pv)} -> {_fv(v)}")
    for mid in sorted(set(cur["meta_weights"]) | set(_d(prev.get("meta_weights")))):
        a, b = _d(prev.get("meta_weights")).get(mid), cur["meta_weights"].get(mid)
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
            out.append(f"Meta-Gewicht Modell {mid} {_fv(a)}->{_fv(b)}")
    a, b = prev.get("calibration_coverage"), cur["calibration_coverage"]
    if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
        out.append(f"Kalibrierung coverage {_fv(a)}->{_fv(b)}")
    for key, label in (("champion", "Champion"), ("safe_mode", "Safe Mode"), ("meta_verdict", "Meta-Verdikt")):
        if prev.get(key) != cur.get(key):
            out.append(f"{label}: {_fv(prev.get(key))} -> {_fv(cur.get(key))}")
    return out


def save_state(path: Path, snapshot: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError as e:
        log.error("Snapshot nicht geschrieben (%s): %s", path, e)
        return False


# ── Sammeln ────────────────────────────────────────────────────────────────
def _trade_memory_index(rows: list[dict]) -> dict:
    idx = {}
    for r in rows:
        idx[(r.get("ticker"), r.get("entry_date"))] = r
    return idx


def collect(root, date, state_path=None) -> dict:
    """Sammelt alle Eingaben + berechnete Kennzahlen (rein lesend)."""
    root = Path(root)
    today = date if isinstance(date, _date) else _parse_date(str(date)) or _date.today()
    out_dir = root / "outputs"
    rs = out_dir / "research"
    ml = _load_json(rs / "ml_research.json")
    meta = _load_json(rs / "meta_learning.json")
    meta_state = _load_json(rs / "meta_state.json")
    hc = _load_json(rs / "hc_candidates.json")
    ml_fwd = _load_json(rs / "ml_forward.json")
    hyp = _load_json(rs / "hypothesis_db.json")
    fail = _load_json(rs / "failure_analysis.json")
    memory = _load_jsonl(rs / "trade_memory.jsonl")
    history = _load_json(out_dir / "history.json")
    health_raw = _load_json(out_dir / "external_data" / "health" / "source_health.json")
    world = _load_json(rs / "world_model.json")
    mstate = _load_json(rs / "machine_state.json")
    # Safe Mode NUR aus dem kanonischen SystemState (gleiche Ableitung wie Scanner/HC/Adapter/Promotion)
    try:
        from modules import system_state as _ss
        sys_state = _ss.current(inputs={k: root / v for k, v in _ss.DEFAULT_INPUTS.items()},
                                state_path=root / _ss.STATE, history_path=root / _ss.HISTORY)
    except Exception as _e:  # noqa: BLE001 – unbekannt ist nie "aus"
        log.error(f"weekly: SystemState nicht ableitbar ({_e})")
        sys_state = None
    safe = sys_state
    nextv = _load_json(rs / "next_validation.json")
    director = _load_json(rs / "research_candidates.json")
    alearn = _load_json(rs / "active_learning.json")
    promo = _load_json(out_dir / "intelligence" / "promotion_state.json")
    promo_tr = _load_jsonl(out_dir / "intelligence" / "promotion_transitions.jsonl")
    promo_prop = _load_json(out_dir / "intelligence" / "contract_proposals.json")
    alt_board = _load_json(rs / "source_scoreboard.json")
    alt_val = _load_json(rs / "alt_data_validation.json")
    alt_fwd = _load_jsonl(rs / "alt_forward_ledger.jsonl")
    alt_cond = _load_json(rs / "source_conditions.json")
    alt_health = {}
    try:
        from modules.alt_data.registry import SOURCES as _ALT_SOURCES
        for _sid, _s in _ALT_SOURCES.items():
            if _s.get("health"):
                alt_health[_sid] = _load_json(root / _s["health"])
    except Exception as _e:  # noqa: BLE001 – Bericht darf an der Registry nie scheitern
        log.warning(f"weekly: Alt-Data-Registry nicht ladbar ({_e})")
    ent_rep = _load_json(out_dir / "entity" / "entity_report.json")
    fac_plan = _load_json(rs / "factory_plan.json")
    fac_res = _load_json(rs / "factory_results.json")
    fac_dirs = _load_json(rs / "research_directions.json")
    fac_ch = _load_jsonl(rs / "factory_challengers.jsonl")
    fac_fwd = _load_jsonl(rs / "factory_forward_ledger.jsonl")
    ledger_rows = ledger_open = 0
    ledger_dir = out_dir / "candidate_ledger"
    if ledger_dir.is_dir():
        for f in sorted(ledger_dir.glob("*.jsonl")):
            for r in _load_jsonl(f):
                ledger_rows += 1
                if r.get("status") != "rejected":
                    ledger_open += 1

    data = {
        "date": today.isoformat(), "root": str(root),
        "ml": ml, "meta": meta, "meta_state": meta_state, "hc": hc, "ml_fwd": ml_fwd,
        "hyp": hyp, "fail": fail, "history": history, "world": world, "mstate": mstate, "safe": safe,
        "nextv": nextv, "director": director, "alearn": alearn,
        "promo": promo, "promo_transitions": promo_tr, "promo_proposals": promo_prop,
        "factory": {"plan": fac_plan, "results": fac_res, "directions": fac_dirs, "challengers": fac_ch,
                    "forward_cohorts": {h: sum(1 for r in fac_fwd if r.get("hypothesis_id") == h)
                                        for h in sorted({r.get("hypothesis_id") for r in fac_fwd})}},
        "alt": {"board": alt_board, "validation": alt_val, "conditions": alt_cond, "health": alt_health,
                "entity": ent_rep,
                "forward_cohorts": {h: sum(1 for r in alt_fwd if r.get("hypothesis_id") == h)
                                    for h in sorted({r.get("hypothesis_id") for r in alt_fwd})}},
        "missing": [n for n, v in (("ml_research.json", ml), ("meta_learning.json", meta),
                                   ("meta_state.json", meta_state), ("hc_candidates.json", hc),
                                   ("ml_forward.json", ml_fwd), ("hypothesis_db.json", hyp),
                                   ("failure_analysis.json", fail), ("history.json", history),
                                   ("source_health.json", health_raw)) if v is None]
                   + ([] if memory else ["trade_memory.jsonl"])
                   + ([] if ledger_rows else ["candidate_ledger/*.jsonl"]),
        "health": summarize_health(health_raw, today),
        "forward": compute_forward(history, today),
        "ledger": {"rows": ledger_rows, "not_rejected": ledger_open},
    }
    # Recent Signal Review
    idx = _trade_memory_index(memory)
    cutoff = today - timedelta(days=RECENT_DAYS)
    recent = []
    for t in _closed_trades(_d(history)):
        cd = _parse_date(t.get("close_date"))
        if cd is None or cd < cutoff:
            continue
        o = float(t["outcome"])
        mem = idx.get((t.get("ticker"), t.get("entry_date")))
        cause = None
        if o <= 0:
            if mem:
                f = _d(mem.get("failure"))
                www = mem.get("what_went_wrong") or []
                cause = f.get("primary") or (www[0] if www else None) or "unbekannt (kein Label)"
            else:
                cause = "unbekannt (kein trade_memory-Eintrag)"
        recent.append({"ticker": t.get("ticker"), "entry_date": t.get("entry_date"),
                       "close_date": t.get("close_date"), "outcome": o, "worked": o > 0,
                       "approx": t.get("outcome_method_reconstructed") == "delta_approx",
                       "cause": cause})
    recent.sort(key=lambda r: r["close_date"] or "")
    data["recent"] = recent
    # Snapshot / Learning
    sp = Path(state_path) if state_path else out_dir / "reports" / "weekly_state.json"
    data["state_path"] = str(sp)
    prev = _load_json(sp)
    data["snapshot"] = build_snapshot(data)
    data["prev_snapshot_date"] = _d(prev).get("date") if prev else None
    data["learned"] = diff_snapshots(prev if isinstance(prev, dict) else None, data["snapshot"])
    data["warnings"] = compute_warnings(data)
    data["week_proposals"] = _week_proposals(out_dir, today)
    data["mc_calibration"] = _d(_load_json(rs / "paper_performance_analysis.json")).get("mc_hit_rate_calibration")
    data["rl_status"] = _load_json(rs / "rl_promotion.json")
    data["surprise"] = _load_json(rs / "surprise_study.json")
    data["inquiry"] = _load_json(rs / "inquiry_chains.json")
    return data


def _week_proposals(out_dir: Path, today: _date) -> list[dict]:
    """Produktions-Trade-Vorschläge (vollständige Pipeline bestanden) der letzten 7 Tage aus den
    Tagesreports, angereichert um die Ledger-Merkmale desselben Tages (ML-Schätzungen = Research)."""
    rep_dir = out_dir / "daily_reports"
    if not rep_dir.is_dir():
        return []
    cutoff = today - timedelta(days=LEARN_DAYS)
    props = []
    for f in sorted(rep_dir.glob("*.json")):
        d = _parse_date(f.stem)
        if d is None or not (cutoff < d <= today):
            continue
        for p in _d(_load_json(f)).get("proposals") or []:
            if isinstance(p, dict) and p.get("ticker"):
                props.append({**p, "_date": d.isoformat()})
    if not props:
        return []
    want = {(p["_date"], p["ticker"]) for p in props}
    feats = {}
    led = out_dir / "candidate_ledger"
    if led.is_dir():
        for f in sorted(led.glob("*.jsonl")):
            for r in _load_jsonl(f):
                k = (str(r.get("date"))[:10], r.get("ticker"))
                if k in want:
                    feats[k] = _d(r.get("features"))
    for p in props:
        p["_ledger_features"] = feats.get((p["_date"], p["ticker"]), {})
    return props


# ── Warnungen ──────────────────────────────────────────────────────────────
def compute_warnings(data: dict) -> list[dict]:
    w = []

    def add(code, detail):
        w.append({"code": code, "detail": detail})

    h = data["health"]
    if h.get("available") and h["counts"]["FAIL"]:
        add("DATA PIPELINE FAILURE", f"{h['counts']['FAIL']} Quelle(n) mit FAIL: {', '.join(h['failing'][:8])}")
    meta, ml = _d(data["meta"]), _d(data["ml"])
    ds = drift_summary(meta.get("drift"))
    if ds["feature"]:
        add("DATA DRIFT", "Feature-/Daten-Drift geflaggt (" + ", ".join(ds["flags"]) + ")")
    if ds["model"]:
        add("MODEL DRIFT", "Modell-Drift geflaggt (" + ", ".join(ds["flags"]) + ")")
    cal = _d(ml.get("calibration"))
    if cal.get("interval_calibrated") is False:
        add("CALIBRATION FAILURE", f"Intervall nicht kalibriert (coverage {_fv(_num(cal.get('coverage')))} vs Ziel {_fv(_num(cal.get('target')))})")
    over = [str(b.get("bucket")) for b in _buckets(meta) if b.get("flag") == "overconfident"]
    if over:
        add("CALIBRATION FAILURE", "Überkonfidente Buckets: " + ", ".join(over))
    fw = data["forward"]
    if fw.get("available"):
        base = fw["windows"][0]["reliable"]
        if base.get("closed", 0) < LOW_SAMPLE_N:
            add("LOW SAMPLE SIZE", f"Forward (zuverlässig, ohne delta_approx): n={base.get('closed', 0)} < {LOW_SAMPLE_N}")
        short = [f"{x['label']} (n={x['reliable']['closed']})" for x in fw["windows"][1:]
                 if 0 < x["reliable"]["closed"] < LOW_SAMPLE_N]
        if short:
            add("LOW SAMPLE SIZE", "Kurze Forward-Fenster mit n<" + str(LOW_SAMPLE_N) + ": " + ", ".join(short))
    low_b = [str(b.get("bucket")) for b in _buckets(meta) if b.get("flag") == "low_n"]
    if low_b:
        add("LOW SAMPLE SIZE", "Buckets mit zu kleinem n: " + ", ".join(low_b))
    reg = _d(meta.get("current_regime"))
    if str(reg.get("confidence", "")).upper() == "LOW":
        add("REGIME UNCERTAINTY", "Regime-Confidence LOW")
    if str(_d(meta.get("disagreement")).get("current_level", "")).upper() == "HIGH":
        add("EXCESSIVE MODEL DISAGREEMENT", "Modell-Disagreement-Level HIGH")
    if fw.get("available"):
        wl, wr = fw["windows"][3], fw["windows"][0]
        for key in ("reliable", "all"):
            r4, lt = wl[key], wr[key]
            if r4.get("closed", 0) >= DEGRADATION_MIN_N and lt.get("closed", 0) > r4["closed"]:
                if r4["expectancy"] < 0 and r4["expectancy"] < lt["expectancy"]:
                    add("PERFORMANCE DEGRADATION",
                        f"Forward 4W-Expectancy {r4['expectancy']:+.3f} (n={r4['closed']}, {key}) "
                        f"< langfristig {lt['expectancy']:+.3f} (n={lt['closed']})")
                break
    active, _txt = safe_mode_status(data.get("safe"))        # kanonischer SystemState
    if active is not False:
        add("SAFE MODE", "Safe Mode " + ("aktiv" if active else "UNBEKANNT") + " (keine HC-Alerts, stabiler Champion): "
            + "; ".join(map(str, _d(data.get("safe")).get("safe_mode_reason") or [])))
    wc = _d(_d(data.get("world")).get("current"))
    if _num(wc.get("uncertainty")) is not None and wc["uncertainty"] >= 0.6:
        add("REGIME UNCERTAINTY", f"World-Model-Unsicherheit {wc['uncertainty']}")
    return w


def _bucket_key(meta) -> str | None:
    """Kalibrierung des AKTIVEN Ensembles (Audit F11: vorher das verworfene primary_meta)."""
    meta = _d(meta)
    cb = meta.get("calibration_buckets")
    if not isinstance(cb, dict):
        return None
    for key in (meta.get("active_ensemble"), "static_equal", meta.get("primary_meta")):
        if key in cb:
            return key
    return None


def _buckets(meta) -> list[dict]:
    meta = _d(meta)
    cb = meta.get("calibration_buckets")
    if isinstance(cb, dict):
        key = _bucket_key(meta)
        cb = cb.get(key) if key else None
    return [b for b in (cb or []) if isinstance(b, dict)]


def safe_mode_status(state) -> tuple[bool | None, str]:
    """Safe Mode ausschließlich aus dem kanonischen SystemState (modules/system_state.py) –
    derselbe Zustand, den Scanner, HC-Scanner, Adapter und PromotionController lesen.
    Fehlt er -> unbekannt, nie 'aus'."""
    if not isinstance(state, dict) or "safe_mode" not in state:
        return None, "UNBEKANNT (SystemState nicht ableitbar – HC-Alerts gesperrt)"
    ver = f" [State v{state.get('state_version')}]"
    if state.get("safe_mode"):
        return True, "AKTIV: " + "; ".join(map(str, state.get("safe_mode_reason") or [])) + ver
    return False, "aus" + ver


# ── Formatierung ───────────────────────────────────────────────────────────
def _f(x, nd=3, pct=False, sign=False):
    v = _num(x)
    if v is None:
        return NA
    if pct:
        return f"{v * 100:+.1f}%" if sign else f"{v * 100:.1f}%"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


ARROWS = {"improving": "↑", "stable": "→", "deteriorating": "↓"}

# Block-Typen: ("para", text) ("kv", [(k, v)]) ("table", headers, rows, [row_class]) ("list", [items]) ("note", text)


def _fwd_table(kind: str, windows: list[dict]) -> tuple:
    hdr = ["Fenster (close_date)", "Signals", "Closed", "Win Rate", "Avg Ret", "Median", "Avg Win", "Avg Loss",
           "Payoff", "PF", "Expectancy", "MaxDD (Einh.)", "Hinweis"]
    rows, cls = [], []
    for w in windows:
        m = w[kind]
        if not m.get("closed"):
            rows.append([w["label"], str(w["signals"]), "0"] + [NA] * 9 + [NO_DATA])
            cls.append("")
            continue
        rows.append([w["label"], str(w["signals"]), str(m["closed"]), _f(m["win_rate"], pct=True),
                     _f(m["avg_return"], pct=True, sign=True), _f(m["median"], pct=True, sign=True),
                     _f(m["avg_winner"], pct=True, sign=True), _f(m["avg_loser"], pct=True, sign=True),
                     _f(m["payoff_ratio"], 2), _f(m["profit_factor"], 2), _f(m["expectancy"], pct=True, sign=True),
                     _f(m["max_dd"], 2), "LOW SAMPLE" if m["low_sample"] else ""])
        cls.append("warn" if m["low_sample"] else "")
    return ("table", hdr, rows, cls)


def build_sections(data: dict) -> list[tuple[str, str, list]]:
    """Montagsbericht: 7 Hauptabschnitte, danach alle Detailabschnitte als Anhang (A1–A19)."""
    main = monday_sections(data)
    appendix = [(f"A{num}", title, blocks) for num, title, blocks in detail_sections(data)]
    return [(str(n), t, b) for n, t, b in main] + appendix


def detail_sections(data: dict) -> list[tuple[int, str, list]]:
    ml, meta, st, hc = _d(data["ml"]), _d(data["meta"]), _d(data["meta_state"]), data["hc"]
    secs = []

    # 1 System Status
    h, cal = data["health"], _d(ml.get("calibration"))
    mdec = _d(meta.get("decision"))
    reg = meta.get("current_regime")
    if isinstance(reg, dict):
        reg_s = f"{reg.get('name') or reg.get('label') or reg.get('regime') or NA} (Confidence {reg.get('confidence', NA)})"
    else:
        reg_s = str(reg) if reg else NA
    ds = drift_summary(meta.get("drift"))
    fmt_flag = lambda v: NA if v is None else ("GEFLAGGT" if v else "ok")
    if h.get("available"):
        c = h["counts"]
        health_s = f"{h['overall']} (PASS {c['PASS']} / WARNING {c['WARNING']} / FAIL {c['FAIL']}; DEFERRED {c['DEFERRED']})"
        fresh_s = ("letzter Erfolg: " + (h["newest_success"] or NA) + ", ältester: " + (h["oldest_success"] or NA)
                   + "; Staleness: " + (", ".join(f"{k} {v}" for k, v in sorted(h["staleness"].items())) or NA))
    else:
        health_s = fresh_s = NO_DATA
    cal_parts = []
    if cal:
        cal_parts.append(f"ML-Intervall: coverage {_f(cal.get('coverage'))} vs Ziel {_f(cal.get('target'))}, "
                         f"kalibriert={cal.get('interval_calibrated', NA)}, Brier {_f(cal.get('brier'), 4)} "
                         f"(Basis {_f(cal.get('brier_base'), 4)}), p_up_skill {_f(cal.get('p_up_skill'), 3, sign=True)}")
    mcal = meta.get("calibration")
    if isinstance(mcal, dict):
        cal_parts.append("Meta: " + ", ".join(f"{k}={_fv(v)}" for k, v in mcal.items() if not isinstance(v, (dict, list))))
    secs.append((1, SECTION_TITLES[1], [
        ("kv", [
            ("Current Champion", str(ml.get("champion")) if ml.get("champion") else ("keiner" if ml else NO_DATA)),
            ("Current Meta Model", (f"{meta.get('primary_meta', NA)} – Version {meta.get('meta_version', NA)}, "
                                    f"Verdikt {mdec.get('verdict', NA)}") if meta else NO_DATA),
            ("Last Training / Validation", f"ml_research {ml.get('generated', NO_DATA)} (Modus {ml.get('mode', NA)}); "
                                           f"meta_learning {meta.get('generated', NO_DATA) if meta else NO_DATA}"),
            ("Current Regime", reg_s if meta else NO_DATA),
            ("Data pipelines", health_s),
            ("Data freshness", fresh_s),
            ("Model drift", fmt_flag(ds["model"]) if meta else NO_DATA),
            ("Feature drift", fmt_flag(ds["feature"]) if meta else NO_DATA),
            ("Calibration status", " | ".join(cal_parts) if cal_parts else NO_DATA),
            ("Safe Mode", safe_mode_status(data.get("safe"))[1]
             + (f" · aktives Ensemble {st.get('active_ensemble')}" if st.get("active_ensemble") else "")),
            ("Datenstand der Artefakte", data_stand(data)),
            ("Tägliche Scanner-Mail", "wird NICHT vom Research-/Intelligenz-Stack gesteuert (LLM-Pipeline, "
                                      "eigene Gates; Audit F13)"),
        ])]))

    # 2 Forward
    fw = data["forward"]
    b2 = [("note", "Quelle: outputs/history.json (echte Paper-Trades). Strikt getrennt vom Backtest. "
                   "Outcome = Rendite je Trade als Anteil (0.10 = +10%). MaxDD: kumulierte Outcome-Kurve, "
                   "additiv je Trade (Einheit = 1 Position), Reihenfolge close_date. Signals = Trades (inkl. offen) mit entry_date im Fenster.")]
    if not fw.get("available"):
        b2.append(("para", f"FORWARD: {NO_DATA} (outputs/history.json fehlt)."))
    else:
        b2.append(("para", f"Geschlossen gesamt: {fw['n_closed_total']}, offen: {fw['n_active']}, davon "
                           f"rekonstruiert (delta_approx): {fw['n_reconstructed']}."))
        b2.append(("para", "A) Nur zuverlässige Outcomes (ohne delta_approx)"))
        b2.append(_fwd_table("reliable", fw["windows"]))
        b2.append(("para", "B) Alle Outcomes (inkl. delta_approx, Näherung)"))
        b2.append(_fwd_table("all", fw["windows"]))
    lg = data["ledger"]
    b2.append(("para", f"Candidate-Ledger: {lg['rows']} Einträge, davon {lg['not_rejected']} nicht abgelehnt." if lg["rows"] else f"Candidate-Ledger: {NO_DATA}."))
    secs.append((2, SECTION_TITLES[2], b2))

    # 3 Confidence
    bk = _buckets(meta)
    b3 = [("note", f"Quelle: meta_learning.calibration_buckets[{_bucket_key(meta) or NA}] – aktives Ensemble, "
                   f"Walk-Forward-OOS mit Vorjahres-Kalibrierung (Backtest, kein Forward).")]
    if bk:
        rows, cls = [], []
        for b in bk:
            exp = b.get("expected_return", b.get("predicted"))
            rows.append([str(b.get("bucket", NA)), _fv(b.get("n")), _f(b.get("win_rate"), pct=True),
                         _f(b.get("avg_return"), pct=True, sign=True), _f(exp, pct=True, sign=True),
                         _f(b.get("calibration_error"), 3, sign=True), str(b.get("flag", NA))])
            cls.append("bad" if b.get("flag") in ("overconfident", "underconfident") else "")
        b3.append(("table", ["Bucket", "n", "Win Rate", "Avg Return", "Expected", "Calibration Error", "Flag"], rows, cls))
    else:
        b3.append(("para", NO_DATA))
    secs.append((3, SECTION_TITLES[3], b3))

    # 4 Model intelligence
    mi = _d(meta.get("model_intelligence"))
    b4 = [("note", "Pfeile nur aus dem Feld trend (metrisch, mit trend_t): ↑ improving, → stable, ↓ deteriorating.")]
    if mi:
        rows = []
        for mid, v in mi.items():
            v = _d(v)
            tr = v.get("trend")
            rows.append([mid, _f(v.get("oos_ic"), 4), _f(v.get("recent_ic"), 4),
                         f"{ARROWS.get(tr, '?')} {tr if tr else NA}", _f(v.get("trend_t"), 2),
                         _fv(v.get("calibration")), _fv(v.get("contribution")), _f(v.get("meta_weight"), 3)])
        b4.append(("table", ["Modell", "OOS IC", "Recent IC", "Trend", "trend_t", "Calibration", "Contribution", "Meta-Gewicht"], rows, []))
    else:
        b4.append(("para", NO_DATA))
    secs.append((4, SECTION_TITLES[4], b4))

    # 5 Meta learning
    b5 = []
    appr = _d(meta.get("approaches"))
    pm, ref = meta.get("primary_meta"), meta.get("reference")
    if appr:
        b5.append(("note", "Alle Zahlen aus Backtest/Walk-Forward – kein Forward-Ergebnis."))
        cols = [k for k in META_METRICS if any(k in _d(_d(a).get("metrics")) for a in appr.values())]
        names = [n for n in (pm, ref) if n in appr] or list(appr)
        deltas = _d(meta.get("deltas"))
        rows = []
        for k in cols:
            row = [k] + [_f(_d(_d(appr[n]).get("metrics")).get(k), 4) for n in names]
            d = _d(deltas.get(k))
            row.append(f"{_f(d.get('delta'), 4, sign=True)} [{_f(d.get('ci_low'), 4)}; {_f(d.get('ci_high'), 4)}]" if d else NA)
            rows.append(row)
        b5.append(("table", ["Metrik"] + [f"{n}{' (meta)' if n == pm else ' (static/ref)' if n == ref else ''}" for n in names] + ["Delta [CI]"], rows, []))
        b5.append(("para", f"Verdikt: {mdec.get('verdict', NA)}"))
        if mdec.get("reasons"):
            b5.append(("list", [str(r) for r in mdec["reasons"]]))
    else:
        b5.append(("para", f"Meta-Learning: {NO_DATA}."))
    models = _d(ml.get("models"))
    if models:
        rows = []
        for mid, m in models.items():
            m = _d(m)
            base, ic = _d(_d(m.get("wf")).get("base")), _d(_d(m.get("wf")).get("ic"))
            rows.append([mid, str(m.get("role", NA)), _f(base.get("mean"), pct=True, sign=True), _f(base.get("t_months"), 2),
                         _f(base.get("sharpe_ann"), 2), _f(base.get("max_dd"), pct=True), _f(ic.get("mean_ic"), 4),
                         str(_d(m.get("decision")).get("verdict", NA))])
        b5.append(("para", "ML-Modelle – BACKTEST (Walk-Forward netto, ml_research.json)"))
        b5.append(("table", ["Modell", "Rolle", "WF Mean/Kohorte", "t", "Sharpe", "MaxDD", "IC", "Verdikt"], rows, []))
    fwd = _d(_d(data["ml_fwd"]).get("forward"))
    if fwd:
        rows = [[mid, _fv(_d(_d(v).get("base")).get("n_months")), _f(_d(_d(v).get("base")).get("mean"), pct=True, sign=True),
                 _f(_d(_d(v).get("base")).get("t_months"), 2), _f(_d(_d(v).get("ic")).get("mean_ic"), 4)]
                for mid, v in fwd.items()]
        b5.append(("para", "ML-Modelle – FORWARD-SHADOW (ml_forward.json, keine Trades)"))
        b5.append(("table", ["Modell", "n Monate", "Mean", "t", "IC"], rows, []))
    else:
        b5.append(("para", f"ML-Forward-Shadow (ml_forward.json): {NO_DATA}."))
    secs.append((5, SECTION_TITLES[5], b5))

    # 6 HC candidates
    b6 = []
    cands = _d(hc).get("candidates") if hc else None
    if not hc or not _d(hc).get("enabled", True) or not cands:
        b6.append(("para", NO_HC_TEXT))
        if _d(hc).get("disabled_reason"):
            b6.append(("para", f"Grund: {_d(hc)['disabled_reason']}"))
        if not hc:
            b6.append(("note", "hc_candidates.json nicht vorhanden."))
    else:
        rows = [[str(c.get("ticker", NA)), _f(c.get("calibrated_probability"), pct=True),
                 _f(c.get("expected_return_20d"), pct=True, sign=True), _f(c.get("expected_downside"), pct=True, sign=True),
                 _f(c.get("asymmetry_ratio"), 2), _fv(c.get("model_agreement")), _fv(c.get("regime_compatibility")),
                 str(c.get("confidence", NA))] for c in cands if isinstance(c, dict)]
        b6.append(("table", ["Ticker", "Probability", "Expected Return (20d)", "Expected Downside", "Asymmetry",
                             "Model Agreement", "Regime Match", "Confidence"], rows, []))
    secs.append((6, SECTION_TITLES[6], b6))

    # 7 Recent
    b7 = [("note", f"Trades, die in den letzten {RECENT_DAYS} Tagen geschlossen wurden (history.json); Ursachen aus trade_memory.jsonl.")]
    if data["recent"]:
        rows, cls = [], []
        for r in data["recent"]:
            rows.append([str(r["ticker"]), str(r["entry_date"]), str(r["close_date"]), _f(r["outcome"], pct=True, sign=True),
                         "funktioniert" if r["worked"] else "nicht", (r["cause"] or "") + (" [delta_approx]" if r["approx"] else "")])
            cls.append("" if r["worked"] else "bad")
        b7.append(("table", ["Ticker", "Entry", "Close", "Outcome", "Ergebnis", "Primäre Ursache (Verlierer)"], rows, cls))
    elif data["forward"].get("available"):
        b7.append(("para", "Keine in den letzten 4 Wochen geschlossenen Trades."))
    else:
        b7.append(("para", NO_DATA))
    secs.append((7, SECTION_TITLES[7], b7))

    # 8 Learned
    if data["learned"] is None:
        b8 = [("para", FIRST_REPORT_TEXT)]
    elif not data["learned"]:
        b8 = [("para", f"Keine gemessenen Änderungen seit Snapshot vom {data['prev_snapshot_date'] or NA}.")]
    else:
        b8 = [("note", f"Gemessene Änderungen seit Snapshot vom {data['prev_snapshot_date'] or NA}:"),
              ("list", data["learned"])]
    secs.append((8, SECTION_TITLES[8], b8))

    # 9 Research pipeline
    hyp = _d(data["hyp"])
    hs = _d(hyp.get("hypotheses"))
    b9 = []
    if hs:
        cnt = {"accepted": 0, "rejected": 0, "inconclusive": 0, "currently testing": 0, "sonstige": 0}
        for v in hs.values():
            s = str(_d(v).get("status", ""))
            cs = str(_d(v).get("canonical_status", ""))
            if cs == "RETEST_LATER" or s in ("testing", "running", "in_test", "currently_testing"):
                cnt["currently testing"] += 1
            elif cs == "INCONCLUSIVE":
                cnt["inconclusive"] += 1
            elif cs == "ACCEPTED" or s.startswith("accepted"):
                cnt["accepted"] += 1
            elif cs == "REJECTED":
                cnt["rejected"] += 1
            elif s.startswith("accepted"):
                cnt["accepted"] += 1
            elif s.startswith("rejected"):
                cnt["rejected"] += 1
            elif s in ("not_significant_after_fdr", "passed_pending_locked"):
                cnt["inconclusive"] += 1
            elif s in ("testing", "running", "in_test", "currently_testing"):
                cnt["currently testing"] += 1
            else:
                cnt["sonstige"] += 1
        disc = _d(hyp.get("discovery"))
        b9.append(("kv", [(k, str(v)) for k, v in cnt.items()] +
                   [("Getestet gesamt (n_tested_total)", _fv(hyp.get("n_tested_total"))),
                    ("Discovery", (f"{disc.get('n_tested', NA)} getestet, {disc.get('n_survivors', NA)} überlebt, Fenster {disc.get('window', NA)}") if disc else NO_DATA)]))
        newest = sorted((v for v in hs.values() if _d(v).get("created_at")), key=lambda v: v["created_at"], reverse=True)[:5]
        if newest:
            b9.append(("table", ["Angelegt", "ID", "Titel", "Status"],
                       [[str(v["created_at"]), str(v.get("id", NA)), str(v.get("title", NA)), str(v.get("status", NA))] for v in newest], []))
        fa = _d(data["fail"])
        prim = _d(_d(fa.get("reliable")).get("primary"))
        if prim:
            b9.append(("para", "Failure-Analyse (zuverlässige Outcomes, Anteil primäre Ursache): " +
                       ", ".join(f"{k} {_f(_d(v).get('share'), pct=True)} (n={_d(v).get('n')})" for k, v in prim.items())))
    else:
        b9.append(("para", NO_DATA))
    secs.append((9, SECTION_TITLES[9], b9))

    # 10 Warnings
    if data["warnings"]:
        b10 = [("table", ["Warnung", "Detail"], [[w["code"], w["detail"]] for w in data["warnings"]], ["bad"] * len(data["warnings"]))]
    else:
        b10 = [("para", "Keine Warnungen aus den vorhandenen Daten.")]
    if data["missing"]:
        b10.append(("note", "Fehlende Eingaben: " + ", ".join(data["missing"])))
    secs.append((10, SECTION_TITLES[10], b10))
    secs.extend(intelligence_sections(data))
    return secs


# ── Montagsbericht: 7 Hauptabschnitte ──────────────────────────────────────
_PROMO_GROUP = {"IDEA": "RESEARCH IDEA", "HISTORICAL_RESEARCH": "RESEARCH IDEA",
                "HISTORICALLY_VALIDATED": "RESEARCH IDEA", "PROSPECTIVE_CHALLENGER": "CHALLENGER",
                "FORWARD_VALIDATED": "FORWARD VALIDATED", "GUARDED_PRODUCTION": "PROMOTED",
                "LIMITED_PRODUCTION": "PROMOTED", "FULL_PRODUCTION": "PROMOTED",
                "DEMOTED": "REJECTED", "REJECTED": "REJECTED", "EXPIRED": "REJECTED"}


def _recent(ts, today: _date, days: int = LEARN_DAYS) -> bool:
    d = _parse_date(str(ts)[:10]) if ts else None
    return d is not None and today - timedelta(days=days) < d <= today


def scoreboard(data: dict) -> dict[str, list[list[str]]]:
    """Hypothesen gruppiert nach RESEARCH IDEA / CHALLENGER / FORWARD VALIDATED / PROMOTED / REJECTED.
    Quelle der Wahrheit für Status/Forward-Evidenz: PromotionController; ergänzt um Fabrik und
    Hypothesen-DB (nur historisch -> nie mehr als RESEARCH IDEA bzw. REJECTED)."""
    groups: dict[str, list[list[str]]] = {g: [] for g in SCOREBOARD_GROUPS}
    seen = set()
    for k, h in _d(_d(data.get("promo")).get("hypotheses")).items():
        h, ev = _d(h), _d(_d(h).get("evidence"))
        ci = ev.get("ci") or [None, None]
        hid = h.get("hypothesis_id") or k
        seen.add(hid)
        exp = _d(ev.get("fired")).get("expectancy", _d(ev.get("policy")).get("expectancy"))
        span = (f"{str(ev.get('first_observation'))[:10]} – {str(ev.get('last_observation'))[:10]}"
                if ev.get("first_observation") else f"ab {str(h.get('forward_start') or NA)[:10]}")
        groups[_PROMO_GROUP.get(str(h.get("state")), "RESEARCH IDEA")].append([
            k, str(h.get("title") or h.get("description") or "")[:60], str(h.get("state")),
            str(ev.get("n_observations", 0)), str(ev.get("n_independent_dates", 0)), span,
            _fv(ev.get("delta_expectancy")), f"[{_fv(ci[0])}, {_fv(ci[1])}]", _fv(exp),
            _fv(ev.get("ece") if ev.get("ece") is not None else ev.get("brier")),
            str(h.get("influence_level") or "NONE")])
    fac = _d(data.get("factory"))
    fc = _d(fac.get("forward_cohorts"))
    for c in fac.get("challengers") or []:
        hid = c.get("hypothesis_id")
        if not hid or hid in seen:
            continue
        seen.add(hid)
        groups["CHALLENGER"].append([hid, str(c.get("title") or c.get("signal") or "")[:60], "PROSPECTIVE_CHALLENGER (Fabrik)",
                                     str(fc.get(hid, 0)), NA, f"ab {str(c.get('forward_start') or NA)[:10]}",
                                     NA, NA, NA, NA, "NONE"])
    res = _d(_d(fac.get("results")).get("results"))
    for hid, r in res.items():
        if hid in seen:
            continue
        seen.add(hid)
        st = str(_d(r).get("status"))
        grp = "REJECTED" if st == "REJECTED" else "RESEARCH IDEA"
        spec = _d(_d(r).get("spec"))
        groups[grp].append([hid, str(spec.get("title") or spec.get("signal") or "")[:60], f"{st} (historisch)",
                            "0", "0", "kein Forward", NA, NA, NA, NA, "NONE"])
    for hid, h in _d(_d(data.get("hyp")).get("hypotheses")).items():
        if hid in seen:
            continue
        cs = str(_d(h).get("canonical_status") or "")
        grp = "REJECTED" if cs == "REJECTED" else ("RESEARCH IDEA" if cs in ("ACCEPTED", "INCONCLUSIVE", "RETEST_LATER")
                                                   else None)
        if grp is None:
            continue
        groups[grp].append([hid, str(_d(h).get("title") or "")[:60], f"{cs or _d(h).get('status')} (historisch)",
                            "0", "0", "kein Forward", NA, NA, NA, NA, "NONE"])
    return groups


def _band(mc_hit, calib) -> tuple[str | None, dict]:
    v = _num(mc_hit)
    if v is None or not _d(calib):
        return None, {}
    band = "<0.55" if v < 0.55 else "0.55-0.65" if v < 0.65 else "0.65-0.75" if v < 0.75 else ">=0.75"
    return band, _d(_d(calib).get(band))


def _calibrated_p(mc_hit, calib) -> str:
    """Kalibrierte Trefferquote = realisierte Win Rate des MC-Hit-Rate-Bands (echte Paper-Trades)."""
    band, b = _band(mc_hit, calib)
    if band is None:
        return NA
    if (b.get("n") or 0) < CAL_MIN_N:
        return f"{NA} (Band {band}: n={b.get('n', 0)} < {CAL_MIN_N})"
    return f"{b['win_rate'] * 100:.0f}% (Band {band}, n={b['n']}; Modell sagte {_num(mc_hit) * 100:.0f}%)"


def is_high_confidence(p: dict, data: dict) -> bool:
    """Berichts-Label: kalibriertes Band auf echten Paper-Trades belegt und kein Safe Mode."""
    if safe_mode_status(data.get("safe"))[0] is not False:
        return False
    _, b = _band(p.get("mc_hit_rate") or _d(p.get("simulation")).get("hit_rate"), data.get("mc_calibration"))
    return ((b.get("n") or 0) >= HC_BAND_MIN_N and (_num(b.get("mean")) or 0) > 0
            and (_num(b.get("profit_factor")) or 0) >= HC_BAND_MIN_PF)


def _candidate_blocks(p: dict, data: dict) -> list:
    da, opt, sim, lf = _d(p.get("deep_analysis")), _d(p.get("option")), _d(p.get("simulation")), _d(p.get("_ledger_features"))
    ts, ex, rt = _d(p.get("trade_score")), _d(p.get("exit_rules")), _d(da.get("red_team"))
    feats = _d(p.get("features"))
    st = _d(data.get("safe"))
    hyps = [k for k, h in _d(_d(data.get("promo")).get("hypotheses")).items()
            if str(_d(h).get("influence_level") or "NONE") != "NONE"]
    risks = [str(rt.get(f"argument_{i}"))[:160] for i in (1, 2, 3) if rt.get(f"argument_{i}")]
    inval = []
    if ex:
        inval.append("Exit-Regeln: " + ", ".join(f"{k}={v}" for k, v in ex.items() if not isinstance(v, (dict, list)))[:200])
    if da.get("bear_case"):
        inval.append("Bear Case: " + str(da["bear_case"])[:160])
    opt_idea = (f"{p.get('strategy', NA)} Strike {opt.get('strike', NA)} Verfall {opt.get('expiry', NA)} "
                f"(DTE {opt.get('dte', NA)}, Ask {opt.get('ask', NA)}, IV {_fv(opt.get('implied_vol'))})") if opt else NA
    kv = [
        ("Richtung", str(p.get("direction") or da.get("direction") or NA)),
        ("Aktueller Kurs (Scan)", _fv(sim.get("current_price"))),
        ("Erwartete Rendite 20d", NA + " (kein produktiv validiertes 20d-Modell)"),
        ("Erwartete Rendite 60d", (f"{_f(lf.get('ml_exp_ret_60'), pct=True, sign=True)} "
                                   f"[q10 {_f(lf.get('ml_q10_ret_60'), pct=True, sign=True)}, q90 "
                                   f"{_f(lf.get('ml_q90_ret_60'), pct=True, sign=True)}] – ML-Research-Schätzung, "
                                   f"nicht produktiv validiert") if lf.get("ml_exp_ret_60") is not None else NA),
        ("Erwartete Rendite 120d", NA + " (kein Modell)"),
        ("Kalibrierte Wahrscheinlichkeit", _calibrated_p(p.get("mc_hit_rate") or sim.get("hit_rate"), data.get("mc_calibration"))),
        ("Erwarteter Drawdown / MAE", (f"{_f(lf.get('ml_exp_dd_60'), pct=True, sign=True)} (60d, ML-Research)"
                                       if lf.get("ml_exp_dd_60") is not None else NA)),
        ("MFE", NA + " (erst nach Outcome messbar)"),
        ("Asymmetrie", (f"Modell-Move {_fv(p.get('model_move_pct'))}% vs. Implied {_fv(p.get('implied_move_pct'))}% "
                        f"(Edge {_fv(p.get('edge_vs_implied'))}), Break-even {_fv(p.get('trade_bep_pct'))}%")),
        ("Confidence (Trade Score)", f"{_fv(ts.get('total'))} · Catalyst-Confidence {_fv(da.get('catalyst_confidence'))}/10"),
        ("Regime", str(da.get("macro_regime") or NA)),
        ("Model Agreement", (f"Disagreement-SD {_fv(lf.get('ml_disagreement_sd'))} (ML-Research)"
                             if lf.get("ml_disagreement_sd") is not None else NA)),
        ("Data Quality", f"{_fv(_d(st.get('data_health')).get('data_quality'))} (System) · Analyse: {da.get('data_confidence', NA)}"),
        ("Relevante promotete Hypothesen", ", ".join(hyps) or "keine (keine Hypothese mit Produktionseinfluss)"),
        ("Wichtigste positive Evidenz", str(da.get("catalyst") or NA)[:200] + (" – " + str(da.get("asymmetry_reasoning"))[:300]
                                                                              if da.get("asymmetry_reasoning") else "")),
        ("Counterfactual Fragility", (_fv(feats.get("risk_counterfactual_fragility")) + " (Anteil knapp bestandener Gates)")
                                     if feats.get("risk_counterfactual_fragility") is not None else NA),
        ("Abstention-Risiko (SHADOW)", ("abstain_score " + _fv(feats.get("risk_abstain_score")) + " · " +
                                        ", ".join(f"{k[5:]} {_fv(v)}" for k, v in feats.items()
                                                  if k.startswith("risk_") and k != "risk_abstain_score" and v is not None))
                                       if feats.get("risk_abstain_score") is not None else NA),
        ("Optionsidee", opt_idea),
    ]
    return [("para", f"{p.get('ticker')} – Scan {p.get('_date')} (Produktionspipeline bestanden, Rang {p.get('trade_rank', NA)})"),
            ("kv", kv), ("para", "Risiken:"), ("list", risks or [NA]),
            ("para", "Invalidation Conditions:"), ("list", inval or [NA])]


def _perf_row(label: str, m: dict) -> list[str]:
    if not m.get("closed"):
        return [label, "0"] + [NA] * 7
    return [label, str(m["closed"]), _f(m["expectancy"], pct=True, sign=True), _f(m["win_rate"], pct=True),
            _f(m["profit_factor"], 2), _f(m.get("sharpe_per_trade"), 2), _f(m.get("sortino_per_trade"), 2),
            _f(m["max_dd"], 2), "LOW SAMPLE" if m["low_sample"] else ""]


def _inquiry_blocks(data: dict) -> list:
    """KERNFRAGE: Was verstehe ich nicht -> Erklärung -> Daten -> neue Daten -> Verhaltensänderung."""
    inq = _d(data.get("inquiry"))
    if not inq:
        return [("para", "KERNFRAGE: " + NO_DATA + " (modules.inquiry noch nicht gelaufen)")]
    rows = []
    for c in inq.get("chains") or []:
        q1, ex = _d(c.get("question_1_not_understood")), c.get("question_2_explanations") or []
        rows.append([str(q1.get("finding"))[:70],
                     "; ".join(f"{e.get('id')} ({e.get('status')})" for e in ex[:3]) or "keine",
                     "ja" if ex and all(e.get("data_available") for e in ex) else ("teilweise" if ex else "–"),
                     c.get("status"), c.get("question_5_behaviour"), c.get("next_step")])
    lc = _d(inq.get("research_learning"))
    cal = _d(lc.get("priority_calibration"))
    curve = ", ".join(f"{q}: {v.get('success_rate')} (n={v.get('tested')})" for q, v in _d(lc.get("by_quarter")).items())
    return [("para", "KERNFRAGE – Was verstehe ich nicht, welche Erklärung, welche Daten, hält sie auf neuen Daten, "
                     "ändert sie mein Verhalten? Status: " + ", ".join(f"{k} {v}" for k, v in
                                                                       _d(inq.get("status_counts")).items())),
            ("table", ["Nicht verstanden", "Erklärung(en) (Status)", "Daten da", "Kettenstatus", "Verhalten",
                       "Nächster Schritt"], rows[:12], []),
            ("para", f"Lernt die Forschung? Erfolgsquote je Quartal: {curve or NO_DATA}. Kalibrierung der "
                     f"Priorisierung (Spearman Priorität vs. Erfolg): {_fv(cal.get('spearman'))} (n={cal.get('n', 0)}).")]


def monday_sections(data: dict) -> list[tuple[int, str, list]]:
    today = _parse_date(data["date"]) or _date.today()
    st, meta, ml = _d(data.get("safe")), _d(data.get("meta")), _d(data.get("ml"))
    groups = scoreboard(data)
    secs = []

    # 1 SYSTEM STATUS
    dh, dr, cv, ps = _d(st.get("data_health")), _d(st.get("drift_state")), _d(st.get("champion_version")), _d(st.get("promotion_state"))
    cnt = _d(dh.get("counts"))
    secs.append((1, MONDAY_TITLES[1], [("kv", [
        ("Health", f"{dh.get('status', NA)} · Data Quality {_fv(dh.get('data_quality'))}" if dh else NO_DATA),
        ("Safe Mode", safe_mode_status(st)[1]),
        ("Drift", f"{dr.get('level', NA)}" + (" – " + "; ".join(map(str, dr.get("reasons") or [])) if dr.get("reasons") else "")),
        ("Datenquellen", ", ".join(f"{k} {v}" for k, v in cnt.items()) if cnt else NO_DATA),
        ("Champion-Version", f"{cv.get('version', NA)} (ML-Champion: {cv.get('ml_champion') or 'keiner'})" if cv else NO_DATA),
        ("Meta-Modell", f"{meta.get('primary_meta', NA)} v{meta.get('meta_version', NA)}, Verdikt "
                        f"{_d(meta.get('decision')).get('verdict', NA)} (SHADOW)" if meta else NO_DATA),
        ("Aktive Hypothesen (Research)", str(len(groups["RESEARCH IDEA"]))),
        ("Challenger (prospektiv)", str(len(groups["CHALLENGER"]))),
        ("Promotete Hypothesen", str(len(groups["PROMOTED"])) + (f" – mit Einfluss: {', '.join(ps.get('with_influence') or [])}"
                                                                 if ps.get("with_influence") else " – kein Produktionseinfluss")),
        ("Drift/Fehler (Warnungen)", ", ".join(sorted({w["code"] for w in data["warnings"]})) or "keine"),
    ])]))

    # 2 WHAT THE SYSTEM LEARNED
    tr = [t for t in data.get("promo_transitions") or [] if _recent(t.get("timestamp"), today)]
    confirmed = [f"{t.get('key')}: {t.get('previous_state')} -> {t.get('new_state')} ({t.get('reason')})" for t in tr
                 if t.get("new_state") in ("FORWARD_VALIDATED", "GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")]
    rejected = [f"{t.get('key')}: {t.get('decision')} – {t.get('reason')}" for t in tr
                if t.get("decision") in ("REJECT", "DEMOTE", "ROLLBACK")]
    fres = _d(_d(data.get("factory")).get("results"))
    if _recent(fres.get("generated"), today):
        rejected += [f"{hid}: REJECTED (historischer Walk-Forward) – {'; '.join(map(str, _d(r).get('reasons') or []))[:120]}"
                     for hid, r in _d(fres.get("results")).items() if _d(r).get("status") == "REJECTED"]
    nv = _d(data.get("nextv"))
    blind = [f"{c.get('id')}: n={c.get('n')}, typischer Fehler {_fv(c.get('typical_error'))}, {c.get('common_properties')}"
             for c in nv.get("blind_spot_clusters") or []]
    ms = _d(data.get("mstate"))
    decay = [str(x) for x in (ms.get("which_features_are_decaying") or [])][:6]
    secs.append((2, MONDAY_TITLES[2], [
        ("note", f"Zeitraum: letzte {LEARN_DAYS} Tage. Nur gemessene Änderungen."),
        ("para", "Neu bestätigte Erkenntnisse (Forward):"), ("list", confirmed or ["keine"]),
        ("para", "Verworfene Hypothesen:"), ("list", rejected[:10] or ["keine"]),
        ("para", "Blind Spots (signifikante Fehlercluster):"), ("list", blind[:5] or ["keine"]),
        ("para", "Alpha Decay:"), ("list", decay or ["keine gemessene Abschwächung"]),
        *_inquiry_blocks(data),
        ("para", "Daten-/Research-Erkenntnisse (Änderungen seit letztem Bericht):"),
        ("list", (data.get("learned") or [FIRST_REPORT_TEXT if data.get("learned") is None else "keine"])[:10]),
    ]))

    # 3 HYPOTHESIS SCOREBOARD
    hdr = ["ID", "Kurzbeschreibung", "Status", "Forward N", "Unabh. Tage", "Zeitraum", "Effekt (Δ Exp.)", "CI",
           "Expectancy", "Calibration", "Produktionswirkung"]
    b3 = [("note", "Forward N/Tage/Effekt zählen ausschließlich prospektive Forward-Daten; historische Ergebnisse "
                   "führen höchstens zu RESEARCH IDEA.")]
    for g in SCOREBOARD_GROUPS:
        rows = groups[g]
        b3.append(("para", f"{g} ({len(rows)})"))
        b3.append(("table", hdr, rows[:12], []) if rows else ("para", "keine"))
    secs.append((3, MONDAY_TITLES[3], b3))

    # 4 RESEARCH INTELLIGENCE
    plan = _d(_d(data.get("factory")).get("plan"))
    ideas = [_d(h) for h in plan.get("ideas") or []]
    new = [f"{h.get('id')}: {h.get('title')} ({h.get('family')}, Priorität {_fv(h.get('priority'))})"
           for h in ideas if h.get("plan_status") == "SELECTED"]
    unorth = [f"{h.get('id')}: {h.get('title')} – {str(h.get('mechanism') or '')[:120]}"
              for h in ideas if h.get("exploratory") or h.get("idea_source") == "cross_domain"]
    gaps = [f"{h.get('title')}: Quellen {', '.join(_d(x).get('name', str(x)) if isinstance(x, dict) else str(x) for x in _d(h.get('readiness')).get('free_sources') or []) or '–'}"
            for h in ideas if h.get("plan_status") == "DATA_GAP"]
    gaps += [f"{a.get('source')} (Info-Gewinn {_fv(a.get('expected_information_gain'))})"
             for a in (_d(data.get("alearn")).get("data_gaps") or [])[:3]]
    sp = _d(data.get("surprise"))
    gaps += [f"Surprise Engine: {g}" for g in (sp.get("data_gaps") or [])]
    surprise_lines = [f"{k}: {_d(_d(v).get('decision')).get('verdict', NA)} – Mittel "
                      f"{_fv(_d(_d(_d(v).get('h20')).get('base_cost')).get('mean'))}, t "
                      f"{_fv(_d(_d(_d(v).get('h20')).get('base_cost')).get('t_months'))}, Placebo-p "
                      f"{_fv(_d(_d(v).get('h20')).get('placebo_p'))}"
                      for k, v in _d(sp.get("hypotheses")).items()]
    dc = sorted((_d(c) for c in _d(data.get("director")).get("candidates") or []),
                key=lambda c: -(_num(c.get("priority")) or 0))[:5]
    questions = [f"{c.get('question') or c.get('hypothesis') or c.get('title') or c.get('research_id')} "
                 f"(Priorität {_fv(c.get('priority'))}, "
                 f"EIG {_fv(c.get('expected_information_gain'))})" for c in dc]
    dirs = _d(_d(_d(data.get("factory")).get("directions")).get("directions"))
    best = sorted(((k, d) for k, d in dirs.items() if (_num(_d(d).get("tested")) or 0) > 0),
                  key=lambda kv: -(_num(_d(kv[1]).get("posterior_success")) or 0))[:5]
    secs.append((4, MONDAY_TITLES[4], [
        ("para", "Neue Hypothesen (zum Test ausgewählt):"), ("list", new or ["keine"]),
        ("para", "Unorthodoxe Cross-Domain-Ideen:"), ("list", unorth[:6] or ["keine"]),
        ("para", "Data Gaps (nie simuliert; kostenlose Quellen):"), ("list", gaps[:8] or ["keine"]),
        ("para", "Wichtigste Forschungsfragen (Priorität = EIG × Relevanz × Neuheit × Datenqualität ÷ Kosten ÷ Overfit):"),
        ("list", questions or [NO_DATA]),
        ("para", "Expectation/Surprise Engine (Fundamental vs. Marktreaktion, historischer Walk-Forward, "
                 "zählt nicht als Forward-Evidenz):"),
        ("list", surprise_lines or ["noch kein Studienlauf"]),
        ("para", "Forschungsbereiche mit höchstem nachgewiesenem Informationswert (Posterior Erfolg):"),
        ("list", [f"{k}: Posterior {_fv(_d(d).get('posterior_success'))}, getestet {_d(d).get('tested')}, "
                  f"Erfolg {_d(d).get('success')}" for k, d in best] or ["noch kein Bereich mit getesteten Hypothesen"]),
    ]))

    # 5 WORLD MODEL
    w = _d(data.get("world"))
    cur, prev = _d(w.get("current")), _d(w.get("previous"))
    b5 = []
    if cur:
        rows = [[k[:-6], str(v), _fv(cur.get(k[:-6] + "_score")), _fv(cur.get(k[:-6] + "_uncertainty")),
                 str(prev.get(k, NA))] for k, v in cur.items() if k.endswith("_state")]
        b5 += [("kv", [("Stichtag", str(cur.get("date", NA))), ("Gesamt-Unsicherheit", _fv(cur.get("uncertainty"))),
                       ("Regime (Meta-Learning)", str(_d(meta.get("current_regime")).get("name") or
                                                      _d(meta.get("current_regime")).get("label") or NA))]),
               ("table", ["Dimension", "Zustand", "Score", "Unsicherheit", "Vorwoche"], rows, []),
               ("para", "Relevante Veränderungen:"), ("list", [str(x) for x in w.get("changes") or []] or ["keine"])]
    else:
        b5.append(("para", NO_DATA))
    secs.append((5, MONDAY_TITLES[5], b5))

    # 6 TOP TRADE CANDIDATES – ausschließlich Produktionspipeline
    props = sorted(data.get("week_proposals") or [], key=lambda p: (str(p.get("_date")), -(_num(_d(p.get("trade_score")).get("total")) or 0)),
                   reverse=True)
    b6 = [("note", "Nur Kandidaten, die die vollständige Produktionspipeline bestanden haben (Tagesreports der letzten "
                   f"{LEARN_DAYS} Tage). Research-/Backtest-Kandidaten erscheinen hier nie (siehe Anhang A6).")]
    active, _ = safe_mode_status(st)
    if active is not False:
        b6.append(("para", "Safe Mode aktiv oder unbekannt: keine positiven Intelligence-Boosts; nur Champion-Entscheidungen."))
    hc = [p for p in props if is_high_confidence(p, data)]
    b6.append(("note", f"HIGH-CONFIDENCE (Berichts-Label, kein Gate): kalibriertes MC-Band auf echten Paper-Trades "
                       f"mit n >= {HC_BAND_MIN_N}, Expectancy > 0, Profit Factor >= {HC_BAND_MIN_PF}; kein Safe Mode."))
    if not hc:
        b6.append(("para", NO_TRADE_TEXT))
    for p in hc[:5]:
        b6 += _candidate_blocks(p, data)
    rest = [p for p in props if p not in hc]
    if rest:
        b6.append(("para", f"Weitere Produktions-Kandidaten der Woche (Pipeline bestanden, NICHT high-confidence): {len(rest)}"))
        for p in rest[:5]:
            b6 += _candidate_blocks(p, data)
    secs.append((6, MONDAY_TITLES[6], b6))

    # 7 PERFORMANCE – Forward / Walk-Forward OOS / Backtest strikt getrennt
    fw = data["forward"]
    hdr7 = ["Fenster", "N", "Expectancy", "Win Rate", "Profit Factor", "Sharpe/Trade", "Sortino/Trade", "MaxDD (Einh.)", "Hinweis"]
    b7 = [("para", "A) ECHTE FORWARD-/PAPER-PERFORMANCE (Champion, nur zuverlässige Outcomes)")]
    if fw.get("available"):
        b7.append(("table", hdr7, [_perf_row(x["label"], x["reliable"]) for x in fw["windows"]], []))
    else:
        b7.append(("para", NO_DATA))
    ev = _d(_d(data.get("promo")).get("evaluation"))
    ch, ad = _d(ev.get("CHAMPION_ONLY")), _d(ev.get("ADAPTIVE_ACTUAL"))
    b7.append(("para", "Champion vs. Adaptive Intelligence (prospektiv, gleiche Trades):"))
    b7.append(("table", ["Variante", "N", "Expectancy", "Win Rate", "Sharpe", "Sortino", "MaxDD", "Brier"],
               [[n, str(v.get("n", 0)), _fv(v.get("expectancy")), _fv(v.get("win_rate")), _fv(v.get("sharpe")),
                 _fv(v.get("sortino")), _fv(v.get("max_drawdown")), _fv(v.get("brier"))]
                for n, v in (("Champion only", ch), ("Adaptive (tatsächlich)", ad))], [])
               if ch.get("n") else ("para", "Noch keine aufgelösten prospektiven Entscheidungen."))
    cal = _d(data.get("mc_calibration"))
    if cal:
        b7.append(("para", "Calibration (Forward): vorhergesagte MC-Trefferquote vs. realisierte Win Rate"))
        b7.append(("table", ["Band", "n", "vorhergesagt", "realisiert", "Expectancy", "PF"],
                   [[k, str(_d(v).get("n")), _f(_d(v).get("predicted_hit_rate"), pct=True), _f(_d(v).get("win_rate"), pct=True),
                     _f(_d(v).get("mean"), pct=True, sign=True), _fv(_d(v).get("profit_factor"))] for k, v in cal.items()], []))
    models = _d(ml.get("models"))
    b7.append(("para", "B) WALK-FORWARD OOS (Research-Modelle, keine Trades)"))
    b7.append(("table", ["Modell", "WF Mean/Kohorte", "t", "Sharpe", "MaxDD", "IC", "Verdikt"],
               [[mid, _f(_d(_d(_d(m).get("wf")).get("base")).get("mean"), pct=True, sign=True),
                 _f(_d(_d(_d(m).get("wf")).get("base")).get("t_months"), 2),
                 _f(_d(_d(_d(m).get("wf")).get("base")).get("sharpe_ann"), 2),
                 _f(_d(_d(_d(m).get("wf")).get("base")).get("max_dd"), pct=True),
                 _f(_d(_d(_d(m).get("wf")).get("ic")).get("mean_ic"), 4), str(_d(_d(m).get("decision")).get("verdict", NA))]
                for mid, m in models.items()], []) if models else ("para", NO_DATA))
    appr = _d(meta.get("approaches"))
    b7.append(("para", "C) BACKTEST (Meta-Learning, historisch rekonstruiert – kein Forward, zählt nicht als Produktionsevidenz)"))
    if appr:
        pm = meta.get("primary_meta")
        mm = _d(_d(appr.get(pm)).get("metrics"))
        b7.append(("kv", [(k, _fv(mm.get(k))) for k in META_METRICS if k in mm] or [("Metriken", NO_DATA)]))
    else:
        b7.append(("para", NO_DATA))
    rl = _d(data.get("rl_status"))
    b7.append(("para", f"RL-Agent: {rl.get('status', 'keine Bewertung')}" + (f" – {rl.get('summary')}" if rl.get("summary") else "")))
    secs.append((7, MONDAY_TITLES[7], b7))
    return secs


def intelligence_sections(data: dict) -> list:
    """Abschnitte 11–16 (nächste Intelligenz-Stufe). Nur gemessene Inhalte."""
    out = []
    w = _d(data.get("world"))
    cur, prev = _d(w.get("current")), _d(w.get("previous"))
    if cur:
        rows = [[k[:-6], str(v), _fv(cur.get(k[:-6] + "_score")), _fv(cur.get(k[:-6] + "_uncertainty")),
                 str(prev.get(k, NA))] for k, v in cur.items() if k.endswith("_state")]
        val = _d(w.get("validation"))
        b = [("kv", [("Stichtag", str(cur.get("date", NA))), ("Gesamt-Unsicherheit", _fv(cur.get("uncertainty"))),
                     ("Validierung gegen Regime-Engine", f"{val.get('verdict', NA)} (besser: {val.get('better')}, schlechter: {val.get('worse')})")]),
             ("table", ["Dimension", "Zustand", "Score", "Unsicherheit", "Vorwoche"], rows, [])]
        if w.get("changes"):
            b.append(("list", [str(x) for x in w["changes"]]))
    else:
        b = [("para", NO_DATA)]
    out.append((11, SECTION_TITLES[11], b))
    ms = _d(data.get("mstate"))
    if ms:
        sa = _d(ms.get("self_assessment"))
        b = [("kv", [(k, _fv(v)) for k, v in sa.items()]),
             ("para", "Stärken (was wir wissen):"), ("list", [str(x) for x in (ms.get("what_do_we_know") or [])[:6]] or [NO_DATA]),
             ("para", "Schwächen (wo wir systematisch irren):"),
             ("list", [str(x) for x in (ms.get("where_are_we_systematically_wrong") or [])[:6]] or [NO_DATA]),
             ("para", "Größte Unsicherheiten:"), ("list", [str(x) for x in (ms.get("what_are_we_uncertain_about") or [])[:5]] or [NO_DATA])]
    else:
        b = [("para", NO_DATA)]
    out.append((12, SECTION_TITLES[12], b))
    meta = _d(data.get("meta"))
    mi = _d(meta.get("model_intelligence"))
    strong = sorted(((k, _d(v)) for k, v in mi.items()), key=lambda kv: -(kv[1].get("recent_ic") or -9))[:3]
    decay = (ms.get("which_features_are_decaying") or []) if ms else []
    hyp = _d(_d(data.get("hyp")).get("hypotheses"))
    newc = [f"{k}: {_d(v).get('title')} ({_d(v).get('canonical_status')})" for k, v in hyp.items()
            if _d(v).get("source") in ("director", "discovery")][:5]
    out.append((13, SECTION_TITLES[13], [
        ("para", "Stärkste Alpha-Quellen (jüngster 13-Wochen-IC):"),
        ("list", [f"{k}: IC {v.get('recent_ic')} (Trend {v.get('trend')}, t={v.get('trend_t')})" for k, v in strong] or [NO_DATA]),
        ("para", "Schwächer werdend (Decay/Strukturbruch, gemessen):"), ("list", [str(x) for x in decay[:6]] or ["keine gemessene Abschwächung"]),
        ("para", "Neue Alpha-Kandidaten (Director/Discovery):"), ("list", newc or ["keine"])]))
    counts = _d(_d(data.get("hyp")).get("status_counts"))
    nv = _d(data.get("nextv"))
    insight = None
    ab = _d(nv.get("abstention_confirmation"))
    if ab:
        fw = _d(ab.get("forward"))
        insight = (f"Abstinenz-Regel: historisch (Status {ab.get('status', 'n/a')}, in-sample, zählt nicht) aktiv "
                   f"{ab.get('active_expectancy')} vs. inaktiv {ab.get('inactive_expectancy')} (t={ab.get('diff_t')}); "
                   f"VORWÄRTS ab {fw.get('forward_from', 'n/a')}: {fw.get('active_cohorts', 0)} aktive / "
                   f"{fw.get('inactive_cohorts', 0)} inaktive Kohorten, Status {fw.get('status', 'n/a')}")
    out.append((14, SECTION_TITLES[14], [
        ("kv", [(k, str(v)) for k, v in counts.items()] + [("Gesamtvalidierung", str(nv.get("decision", NA))),
                                                             ("G-Komponenten", str(nv.get("G_components", NA)))]),
        ("para", "Größte Erkenntnis: " + (insight or NO_DATA))]))
    cl = nv.get("blind_spot_clusters") or []
    out.append((15, SECTION_TITLES[15], [("table", ["Cluster", "n", "typischer Fehler", "Lift", "Eigenschaften", "Abdeckung"],
                                         [[c.get("id"), str(c.get("n")), _fv(c.get("typical_error")), _fv(c.get("lift")),
                                           str(c.get("common_properties")), str(c.get("existing_model_coverage"))] for c in cl], [])]
                if cl else [("para", "Keine signifikanten Fehlercluster.")]))
    al = _d(data.get("alearn")).get("data_gaps") or []
    out.append((16, SECTION_TITLES[16], [("table", ["Quelle", "Dimensionen", "Info-Gewinn", "Kosten", "Priorität", "Status"],
                                         [[a.get("source"), str(a.get("dimensions")), _fv(a.get("expected_information_gain")),
                                           str(a.get("acquisition_cost")), _fv(a.get("priority")),
                                           "ungeprüft (Aufnahmeprüfung offen)"] for a in al[:6]], [])]
                if al else [("para", NO_DATA)]))
    out.append((17, SECTION_TITLES[17], alt_data_section(data)))
    out.append((18, SECTION_TITLES[18], promotion_section(data)))
    out.append((19, SECTION_TITLES[19], factory_section(data)))
    return out


def factory_section(data: dict) -> list:
    """Scientific Hypothesis Factory: Plan, Ergebnisse, Prospective Challenger, Datenlücken,
    Meta-Learning über Forschungsrichtungen. Nur gemessene Werte; kein Produktionseinfluss."""
    f = _d(data.get("factory"))
    plan, res = _d(f.get("plan")), _d(_d(f.get("results")).get("results"))
    if not plan and not res:
        return [("para", NO_DATA)]
    ideas = plan.get("ideas") or []
    counts: dict = {}
    for h in ideas:
        counts[_d(h).get("plan_status")] = counts.get(_d(h).get("plan_status"), 0) + 1
    blocks = [("kv", [("Ideen", str(plan.get("n_ideas", NA))),
                      ("Status", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=str)) or NA),
                      ("Budget", str(plan.get("budget", NA)))])]
    sel = [h for h in ideas if _d(h).get("plan_status") == "SELECTED"]
    blocks.append(("table", ["ID", "Familie", "Herkunft", "Signal", "Priorität", "Ergebnis", "Gründe"],
                   [[h.get("id"), str(h.get("family")), str(h.get("idea_source")), str(h.get("signal"))[:48],
                     _fv(h.get("priority")),
                     str(_d(res.get(h.get("id"))).get("status", "ausstehend")),
                     "; ".join(map(str, _d(res.get(h.get("id"))).get("reasons") or []))[:80] or "–"] for h in sel], []))
    gaps = [f"{h.get('family')}: {', '.join(_d(s).get('name', str(s)) if isinstance(s, dict) else str(s) for s in _d(h.get('readiness')).get('free_sources') or []) or '–'}"
            for h in ideas if _d(h).get("plan_status") == "DATA_GAP"]
    blocks += [("para", "DATENLÜCKEN (nie simuliert; kostenlose Quellen):"), ("list", gaps or ["keine"])]
    ch = f.get("challengers") or []
    fc = _d(f.get("forward_cohorts"))
    blocks += [("para", "PROSPECTIVE CHALLENGER (nur Forward-Kohorten zählen):"),
               ("list", [f"{c.get('hypothesis_id')}: {c.get('signal')} ab {c.get('forward_start')} – "
                         f"{fc.get(c.get('hypothesis_id'), 0)} Forward-Kohorten" for c in ch] or ["keine"])]
    dirs = _d(_d(f.get("directions")).get("directions"))
    if dirs:
        top = sorted(dirs.items(), key=lambda kv: -(_d(kv[1]).get("tested") or 0))[:8]
        blocks.append(("table", ["Richtung", "getestet", "Erfolg", "prospektiv", "verworfen", "Posterior", "Bewertung"],
                       [[k, str(d.get("tested")), str(d.get("success")), str(d.get("prospective")), str(d.get("rejected")),
                         _fv(d.get("posterior_success")), str(d.get("assessment"))] for k, d in top], []))
    blocks.append(("note", "Fabrik = SHADOW/RESEARCH. Produktionseinfluss nur über Champion-Vertrag (PR) -> "
                           "PromotionController -> Adapter."))
    return blocks


def promotion_section(data: dict, today: _date | None = None) -> list:
    """Promotion-Status je Hypothese, aktive Intelligence-Wirkung (nur Forward-Daten),
    Demotions, Promotion-Kandidaten, Need-more-data. Keine Aussage ohne Messung."""
    st = _d(data.get("promo"))
    hyps = _d(st.get("hypotheses"))
    if not hyps:
        return [("para", NO_DATA)]
    rows, need, cands = [], [], []
    for k, h in hyps.items():
        h, ev = _d(h), _d(_d(h).get("evidence"))
        ci = ev.get("ci") or [None, None]
        rows.append([k, str(h.get("title") or h.get("description") or "")[:48], str(h.get("state")),
                     str(ev.get("n_observations", 0)), str(ev.get("n_independent_dates", 0)),
                     f"{ev.get('calendar_span_days', 0)} T", _fv(ev.get("delta_expectancy")),
                     f"[{_fv(ci[0])}, {_fv(ci[1])}]", str(h.get("influence_level")),
                     str(h.get("next_requirement") or h.get("recommendation") or "–")[:70]])
        if any("NEED_MORE_DATA" in str(r) for r in h.get("reasons") or []):
            need.append(f"{k}: {h.get('next_requirement')}")
    for n in st.get("notices") or []:
        cands.append(f"{n.get('hypothesis')}: {n.get('current_level')} -> {n.get('proposed_level')} "
                     f"({'automatisch begrenzt' if n.get('automatic') else 'Empfehlung – Mensch/PR'}) · "
                     f"Forward N {n.get('forward_n')}, Spanne {n.get('calendar_span_days')} T, "
                     f"Δ Expectancy {n.get('delta_expectancy')} CI {n.get('ci')}")
    blocks = [("kv", [("Automatische Obergrenze", str(st.get("max_automatic_influence"))),
                      ("Policy", f"{st.get('policy_version')} ({st.get('policy_hash')})"),
                      ("Hypothesen getestet (gesamt)", str(_d(st.get("multiple_testing")).get("number_of_hypotheses_tested"))),
                      ("Integrität", "OK" if not any(_d(st.get("integrity")).values()) else str(st.get("integrity")))]),
              ("table", ["ID", "Beschreibung", "Status", "Forward N", "Unabh. Tage", "Spanne", "Δ Effekt", "CI",
                         "Prod.-Einfluss", "Nächste Anforderung"], rows, [])]
    ev_all = _d(st.get("evaluation"))
    ch, ad = _d(ev_all.get("CHAMPION_ONLY")), _d(ev_all.get("ADAPTIVE_ACTUAL"))
    blocks.append(("para", "ACTIVE INTELLIGENCE EFFECT (nur prospektive Forward-Daten):"))
    if ch.get("n"):
        blocks.append(("kv", [("Champion-Trades (aufgelöst)", str(ch.get("trade_count"))),
                              ("blockiert", str(ch.get("trade_count", 0) - ad.get("trade_count", 0))),
                              ("gerettet (verhinderte Verlierer)", str(ad.get("avoided_losers"))),
                              ("verpasste Gewinner", str(ad.get("missed_winners"))),
                              ("inkrementelle Expectancy", _fv((ad.get("expectancy") or 0) - (ch.get("expectancy") or 0))
                               if ad.get("n") else NA),
                              ("Δ Win Rate", _fv((ad.get("win_rate") or 0) - (ch.get("win_rate") or 0)) if ad.get("n") else NA),
                              ("Δ Max Drawdown", _fv((ad.get("max_drawdown") or 0) - (ch.get("max_drawdown") or 0))
                               if ad.get("n") else NA),
                              ("Δ Brier", _fv((ad.get("brier") or 0) - (ch.get("brier") or 0))
                               if ad.get("brier") is not None and ch.get("brier") is not None else NA)]))
    else:
        blocks.append(("para", "Noch keine aufgelösten Forward-Entscheidungen – kein Effekt messbar."))
    dem = [f"{e.get('key')}: {e.get('previous_state')} -> {e.get('new_state')} ({e.get('reason')})"
           for e in data.get("promo_transitions") or [] if e.get("decision") in ("DEMOTE", "ROLLBACK", "REJECT")]
    blocks += [("para", "DEMOTIONS:"), ("list", dem[-6:] or ["keine"]),
               ("para", "PROMOTION CANDIDATES:"), ("list", cands or ["keine"]),
               ("para", "NEED MORE DATA:"), ("list", need or ["keine"])]
    pp = _d(data.get("promo_proposals"))
    if pp:
        props = [f"{_d(p.get('walk_forward')).get('rule')}: Kalibrierung Δ {_d(p.get('walk_forward')).get('calibration_delta')}, "
                 f"Test Δ {_d(p.get('walk_forward')).get('test_delta')} (n={_d(p.get('walk_forward')).get('test_n_tail')}) "
                 f"– Entwurf, Registrierung nur per PR" for p in pp.get("proposals") or []]
        blocks += [("para", f"NEUE HYPOTHESEN-VORSCHLÄGE (historischer Walk-Forward, {pp.get('n_trades')} verlässliche "
                            f"Trades, {pp.get('candidates_tested')} Regeln getestet, {len(pp.get('rejected') or [])} "
                            f"verworfen):"), ("list", props or ["keine – kein Muster übersteht den Walk-Forward"])]
    return blocks


def alt_data_section(data: dict) -> list:
    """Alternative Data (SHADOW): Quellen-Scoreboard, inkrementeller Nutzen, Forward-Kohorten.
    Nur gemessene Werte; kein Produktionseinfluss."""
    alt = _d(data.get("alt"))
    board = _d(_d(alt.get("board")).get("sources"))
    if not board:
        return [("para", NO_DATA)]
    rows = [[sid, _f(b.get("coverage"), pct=True), _fv(b.get("freshness")), _fv(b.get("data_quality")),
             ", ".join(b.get("active_features") or []) or "–", _fv(b.get("oos_value")),
             _fv(b.get("forward_value")) if b.get("forward_value") is not None else "noch keine Forward-Daten",
             _fv(b.get("source_value_score")), str(b.get("status", NA))] for sid, b in board.items()]
    blocks = [("table", ["Quelle", "Coverage", "Freshness", "Data Quality", "Active Features", "OOS Value",
                         "Forward Value", "Source Value Score", "Status"], rows, [])]
    val = _d(_d(alt.get("validation")).get("sources"))
    for sid, r in val.items():
        r = _d(r)
        deltas = []
        for bid, b in _d(r.get("baselines")).items():
            bs = _d(_d(b).get("bootstrap"))
            if bs:
                deltas.append(f"{bid}: Δ Monatsrendite {_fv(bs.get('delta_monthly_mean'))} "
                              f"(CI {bs.get('ci_monthly_mean')}), Δ Brier {_fv(_d(_d(b).get('delta')).get('brier'))}")
        blocks.append(("para", f"{sid}: {r.get('verdict', NA)} – {r.get('verdict_reason', NA)}"))
        if deltas:
            blocks.append(("list", deltas))
    blocks += alt_health_blocks(alt, board)
    fc = _d(alt.get("forward_cohorts"))
    blocks.append(("para", "Prospective Challenger (Forward-Kohorten je Vertrag): " +
                   (", ".join(f"{k} {v}" for k, v in fc.items()) if fc else "noch keine")))
    blocks.append(("note", "Alle Quellen SHADOW/RESEARCH. Produktionseinfluss nur über PromotionController nach "
                           "Forward-Validierung und menschlicher Freigabe."))
    return blocks


def alt_health_blocks(alt: dict, board: dict) -> list:
    """Data Source Health, Quellen mit/ohne Mehrwert, Alpha Decay, bedingte Befunde (gemessen)."""
    out = []
    hl = _d(alt.get("health"))
    if hl:
        rows = [[sid, str(_d(h).get("last_observation") or NA)[:10], _f(_d(h).get("coverage"), pct=True)
                 if _d(h).get("coverage") is not None else NA, _fv(_d(h).get("error_rate")),
                 str(_d(h).get("schema_errors") if _d(h).get("schema_errors") is not None else "–"),
                 str(_d(h).get("checked_at") or NA)[:16]] for sid, h in hl.items()]
        out += [("para", "DATA SOURCE HEALTH:"),
                ("table", ["Quelle", "letzte Beobachtung", "Abdeckung", "Fehlerquote", "Schema-Fehler", "geprüft"], rows, [])]
    ent = _d(_d(alt.get("entity")).get("gleif"))
    if ent:
        out.append(("para", f"Entity Resolution (GLEIF): {ent.get('HIGH', 0)} HIGH / {ent.get('MEDIUM', 0)} MEDIUM / "
                            f"{ent.get('LOW', 0)} LOW in diesem Lauf, {ent.get('remaining', NA)} offen, "
                            f"Töchter: {_d(_d(alt.get('entity')).get('gleif_children')).get('children', 0)}"))
    with_fwd = [f"{k}: Forward {_fv(_d(b).get('forward_value'))} ({_d(b).get('forward_cohorts')} Kohorten)"
                for k, b in board.items() if (_d(b).get("forward_value") or 0) > 0
                and (_d(b).get("forward_cohorts") or 0) >= 26]
    without = [f"{k}: {_d(b).get('verdict')}" for k, b in board.items() if _d(b).get("verdict") == "REJECT"]
    decay = [f"{k}/{f}: {st}" for k, b in board.items() for f, st in (_d(b).get("alpha_decay") or {}).items()
             if st in ("decaying", "reversed")]
    out += [("para", "QUELLEN MIT BESTÄTIGTEM FORWARD-MEHRWERT (>= 26 Kohorten):"), ("list", with_fwd or ["keine"]),
            ("para", "QUELLEN OHNE MEHRWERT:"), ("list", without or ["keine"]),
            ("para", "QUELLEN MIT ALPHA DECAY:"), ("list", decay or ["keine"])]
    cells = []
    for sid, src in _d(_d(alt.get("conditions")).get("sources")).items():
        for f, c in _d(_d(src).get("dev")).items():
            for kind in ("sector", "regime"):
                for key, hz in _d(_d(c).get(kind)).items():
                    for h, st in _d(hz).items():
                        t = _d(st).get("t_months")
                        if t is not None and abs(t) >= 2.5:
                            cells.append((abs(t), f"{sid}/{f} × {key} × {h} T: IC {_fv(_d(st).get('mean_ic'))}, t {t}"))
    out += [("para", "BEDINGTE BEFUNDE (Quelle × Sektor/Regime × Horizont, |t| >= 2,5, beschreibend, nicht BH-korrigiert):"),
            ("list", [c for _, c in sorted(cells, reverse=True)[:8]] or ["keine"])]
    return out


def subject_for(date_s: str) -> str:
    return f"Adaptive Asymmetry Scanner – Monday Intelligence Report – {date_s}"


# ── Renderer ───────────────────────────────────────────────────────────────
def _text_table(headers, rows) -> list[str]:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()
    return [line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows]


def render_text(data: dict) -> str:
    L = [subject_for(data["date"]), DISCLAIMER, ""]
    for num, title, blocks in build_sections(data):
        L += [f"{num}. {title}", "=" * (len(title) + 4)]
        for b in blocks:
            if b[0] in ("para", "note"):
                L.append(b[1])
            elif b[0] == "kv":
                L += [f"  {k}: {v}" for k, v in b[1]]
            elif b[0] == "list":
                L += [f"  - {i}" for i in b[1]]
            elif b[0] == "table":
                L += ["  " + s for s in _text_table(b[1], b[2])]
        L.append("")
    return "\n".join(L)


def render_md(data: dict) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    L = [f"# {subject_for(data['date'])}", "", f"_{DISCLAIMER}_", ""]
    for num, title, blocks in build_sections(data):
        L += [f"## {num}. {title}", ""]
        for b in blocks:
            if b[0] == "para":
                L += [b[1], ""]
            elif b[0] == "note":
                L += [f"_{b[1]}_", ""]
            elif b[0] == "kv":
                L += [f"- **{k}**: {v}" for k, v in b[1]] + [""]
            elif b[0] == "list":
                L += [f"- {i}" for i in b[1]] + [""]
            elif b[0] == "table":
                L += ["| " + " | ".join(esc(h) for h in b[1]) + " |", "|" + "---|" * len(b[1])]
                L += ["| " + " | ".join(esc(c) for c in r) + " |" for r in b[2]] + [""]
    return "\n".join(L)


_CSS = ("body{font-family:Arial,Helvetica,sans-serif;color:#0f172a;max-width:960px;margin:0 auto;padding:16px}"
        "h1{font-size:20px}h2{font-size:16px;border-bottom:1px solid #cbd5e1;padding-bottom:4px;margin-top:24px}"
        "table{border-collapse:collapse;font-size:12px;margin:6px 0 12px}"
        "th,td{border:1px solid #cbd5e1;padding:3px 6px;text-align:left}th{background:#f1f5f9}"
        "tr.warn td{background:#fef9c3}tr.bad td{background:#fee2e2}"
        ".note{color:#64748b;font-size:12px}.disc{background:#f1f5f9;padding:6px 10px;font-size:12px}")


def render_html(data: dict) -> str:
    e = _html.escape
    P = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(subject_for(data['date']))}</title>"
         f"<style>{_CSS}</style></head><body>",
         f"<h1>{e(subject_for(data['date']))}</h1><p class='disc'>{e(DISCLAIMER)}</p>"]
    for num, title, blocks in build_sections(data):
        P.append(f"<h2>{num}. {e(title)}</h2>")
        for b in blocks:
            if b[0] == "para":
                P.append(f"<p>{e(b[1])}</p>")
            elif b[0] == "note":
                P.append(f"<p class='note'>{e(b[1])}</p>")
            elif b[0] == "kv":
                P.append("<table>" + "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in b[1]) + "</table>")
            elif b[0] == "list":
                P.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b[1]) + "</ul>")
            elif b[0] == "table":
                cls = b[3] if len(b) > 3 and b[3] else [""] * len(b[2])
                P.append("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in b[1]) + "</tr>" +
                         "".join(f"<tr class='{c}'>" + "".join(f"<td>{e(str(x))}</td>" for x in r) + "</tr>"
                                 for r, c in zip(b[2], cls)) + "</table>")
    P.append("</body></html>")
    return "\n".join(P)


# ── CLI ────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reports.weekly", description="Weekly Intelligence Report")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="rendern + schreiben, nicht senden (Default)")
    g.add_argument("--send", action="store_true", help="Mail senden und Snapshot fortschreiben")
    ap.add_argument("--date", help="Berichtsdatum YYYY-MM-DD (Default: heute UTC)")
    ap.add_argument("--root", default=str(REPO_ROOT), help="Basisverzeichnis der Eingaben (Default: Repo-Root)")
    ap.add_argument("--out-dir", help="Ausgabeverzeichnis (Default: <root>/outputs/reports)")
    ap.add_argument("--save-state", action="store_true", help="Vorwochen-Snapshot auch ohne --send aktualisieren")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.date:
        d = _parse_date(args.date)
        if d is None:
            ap.error("--date muss YYYY-MM-DD sein")
    else:
        d = datetime.now(timezone.utc).date()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs" / "reports"
    state_path = out_dir / "weekly_state.json"

    data = collect(root, d, state_path=state_path)
    html, text, md = render_html(data), render_text(data), render_md(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, content in (("html", html), ("txt", text), ("md", md)):
        p = out_dir / f"weekly_{data['date']}.{ext}"
        p.write_text(content, encoding="utf-8")
    log.info("Report geschrieben: %s/weekly_%s.{html,txt,md}", out_dir, data["date"])

    rc, save = 0, args.save_state
    if args.send:
        from modules.mailer import send_mail
        res = send_mail(subject_for(data["date"]), html, text)
        log.info("Mail-Status: %s (Versuche %s)", res["status"], res["attempts"])
        if res["status"] == "sent":
            save = True
        elif res["status"] == "failed":
            rc = 1
    if save:
        if save_state(state_path, data["snapshot"]):
            log.info("Snapshot aktualisiert: %s", state_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
