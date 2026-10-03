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
                      ("safe_mode", "safe")):
        g = _d(data.get(key)).get("generated") or _d(data.get(key)).get("updated")
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
    safe = _load_json(rs / "safe_mode.json")
    nextv = _load_json(rs / "next_validation.json")
    director = _load_json(rs / "research_candidates.json")
    alearn = _load_json(rs / "active_learning.json")
    alt_board = _load_json(rs / "source_scoreboard.json")
    alt_val = _load_json(rs / "alt_data_validation.json")
    alt_fwd = _load_jsonl(rs / "alt_forward_ledger.jsonl")
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
        "alt": {"board": alt_board, "validation": alt_val,
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
    return data


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
    sm = _d(data.get("safe"))
    if sm.get("active") is True:
        add("SAFE MODE", "Safe Mode aktiv (keine HC-Alerts, stabiler Champion): " + "; ".join(map(str, sm.get("reasons") or [])))
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


def safe_mode_status(safe) -> tuple[bool | None, str]:
    """Safe Mode ausschließlich aus safe_mode.json (Audit F11: vorher meta_state.safe_mode,
    das seit der Entkopplung immer false war). Fehlt die Datei -> unbekannt, nie 'aus'."""
    if not isinstance(safe, dict) or "active" not in safe:
        return None, "UNBEKANNT (safe_mode.json fehlt/unlesbar – HC-Alerts gesperrt)"
    if safe.get("active"):
        return True, "AKTIV: " + "; ".join(map(str, safe.get("reasons") or []))
    return False, "aus"


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


def build_sections(data: dict) -> list[tuple[int, str, list]]:
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
    return out


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
    fc = _d(alt.get("forward_cohorts"))
    blocks.append(("para", "Prospective Challenger (Forward-Kohorten je Vertrag): " +
                   (", ".join(f"{k} {v}" for k, v in fc.items()) if fc else "noch keine")))
    blocks.append(("note", "Alle Quellen SHADOW/RESEARCH. Produktionseinfluss nur über PromotionController nach "
                           "Forward-Validierung und menschlicher Freigabe."))
    return blocks


def subject_for(date_s: str) -> str:
    return f"Adaptive Asymmetry Scanner – Weekly Intelligence Report – {date_s}"


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
