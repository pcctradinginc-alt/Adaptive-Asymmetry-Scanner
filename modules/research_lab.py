"""
modules/research_lab.py – Hypothesen-Datenbank, Experiment-Lab und
Discovery-Engine (NUR SHADOW)

    python -m modules.research_lab      (CI: .github/workflows/ml_research.yml, full)

Hypothesen kommen aus zwei Quellen:
  * config/hypotheses.yaml – formuliert von Mensch oder Research-Agent
    (Aussage, Signal-Ausdruck, erwartete Richtung, Begründung).
  * Discovery-Engine – testet systematisch Divergenzen/Relationen der
    PIT-Merkmale, aber NUR auf dem Discovery-Fenster (Labels enden vor
    periods.discovery_end). Überlebende nach Benjamini-Hochberg werden als
    Hypothese mit fester Richtung registriert und anschließend auf den nie
    gesehenen Jahren danach geprüft.

Jede Hypothese durchläuft dieselbe feste Prüfkette (config/research_protocol.yaml,
geschützt; dieses Modul liest sie nur):
  1. Ausdruck/Leakage: nur PIT-Merkmale aus dem Feature-Store, keine Labels
  2. Duplikat: |Rank-Korrelation| zu bereits getesteter Hypothese -> kein Neutest
  3. Abdeckung
  4. Walk-Forward (festes Signal, keine Anpassung -> jedes Testjahr OOS)
  5. Kosten (Stress), Stabilität (Jahre, Hälften), Regime (VIX, Trend)
  6. Mehrfachtest: Benjamini-Hochberg über ALLE jemals getesteten Hypothesen
  7. Locked-Holdout – höchstens EINMAL je Hypothese, nur nach Bestehen von 1–6
Ergebnis in outputs/research/hypothesis_db.json (Gedächtnis: verhindert, dass
dieselbe Idee immer wieder "neu entdeckt" wird). Annahme ("accepted") heißt nur:
darf als Merkmal in einen NEU registrierten Challenger – nie direkt in Produktion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules import ml_research as ml

log = logging.getLogger(__name__)

HYP_CONFIG = Path("config/hypotheses.yaml")
DB_PATH = ml.OUT_DIR / "hypothesis_db.json"
ALLOWED_NAMES = frozenset(ml.ALL_FEATURES)
ALLOWED_FUNCS = frozenset({"rank", "sign", "abs"})
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


# ── Signal-Ausdrücke (eingeschränkte, PIT-sichere Sprache) ───────────────────

class SignalError(ValueError):
    pass


def validate_expr(expr: str) -> ast.Expression:
    """Erlaubt: PIT-Merkmale, Zahlen, + - * /, unäres -, rank()/sign()/abs().
    Alles andere (Labels wie fwd_*, mfe_*, Attribute, Aufrufe) -> SignalError."""
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except SyntaxError as e:
        raise SignalError(f"Syntax: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and node.id not in ALLOWED_FUNCS:
                raise SignalError(f"nicht erlaubtes Merkmal '{node.id}' (Leakage-Schutz: nur PIT-Merkmale)")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS or len(node.args) != 1 \
                    or node.keywords:
                raise SignalError("nur rank(x), sign(x), abs(x) erlaubt")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                raise SignalError("nur + - * / erlaubt")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise SignalError("nur unäres - erlaubt")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise SignalError("nur Zahlen-Konstanten")
        elif not isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop)):
            raise SignalError(f"nicht erlaubter Ausdruck: {type(node).__name__}")
    return tree


def eval_signal(panel: pd.DataFrame, expr: str) -> pd.Series:
    tree = validate_expr(expr)
    dates = panel["date"]

    def rank(s):
        s = s if isinstance(s, pd.Series) else pd.Series(s, index=panel.index)
        return s.groupby(dates).rank(pct=True) - 0.5
    ns = {n: panel[n] for n in ALLOWED_NAMES if n in panel.columns}
    ns.update({"rank": rank, "sign": np.sign, "abs": np.abs})
    out = eval(compile(tree, "<signal>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307 – AST vorab auf Whitelist geprüft
    if not isinstance(out, pd.Series):
        raise SignalError("Ausdruck ergibt kein Merkmal je Zeile")
    return out.replace([np.inf, -np.inf], np.nan)


def signal_key(expr: str) -> str:
    return hashlib.sha256(ast.dump(validate_expr(expr)).encode()).hexdigest()[:12]


# ── Statistik ────────────────────────────────────────────────────────────────

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_one_sided(t: float | None) -> float:
    return 1.0 - _phi(t) if t is not None and math.isfinite(t) else 1.0


def benjamini_hochberg(pvals: dict[str, float], q: float) -> set[str]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    k_max = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            k_max = i
    return {k for k, _ in items[:k_max]}


def window(panel: pd.DataFrame, start, end, label_end_before) -> pd.DataFrame:
    le = panel["label_end_20"]
    return panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] < pd.Timestamp(end))
                 & le.notna() & (le < pd.Timestamp(label_end_before))]


def evaluate_signal(df: pd.DataFrame, score: pd.Series, panel: pd.DataFrame) -> dict:
    d = df.assign(score=score.loc[df.index])
    coh = ml.cohort_returns(d, "score")
    base, stress = ml.perf(coh, ml.COST_BASE), ml.perf(coh, ml.COST_STRESS)
    ic = ml.ic_stats(ml.daily_ic(d, "score"))
    halves = {}
    if not coh.empty:
        mid = coh["date"].iloc[len(coh) // 2]
        for name, part in (("first", coh[coh["date"] < mid]), ("second", coh[coh["date"] >= mid])):
            halves[name] = ml._r(part["top_vs_univ"].mean() - 2 * ml.COST_BASE)
    return {"base": base, "stress": stress, "ic": ic, "halves": halves,
            "regimes": ml.regime_attribution(coh, panel), "p_value": ml._r(p_one_sided(base.get("t_months")), 6)}


# ── Prüfkette ────────────────────────────────────────────────────────────────

FINGERPRINT_N = 40_000
MIN_LOCKED_COHORTS = 10


def _fingerprint(s: pd.Series) -> np.ndarray:
    """Ränge auf einer festen, deterministischen Zeilenstichprobe (float32)."""
    if len(s) > FINGERPRINT_N:
        s = s.iloc[np.linspace(0, len(s) - 1, FINGERPRINT_N).astype(int)]
    return s.rank().to_numpy(np.float32)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) != len(b):
        return None
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 1000:
        return None
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum()) / den if den > 0 else None


def _regime_verdicts(regimes: dict) -> dict:
    out = {}
    for k, v in regimes.items():
        m, n = v.get("mean"), v.get("n_cohorts") or 0
        out[k] = "n/a" if n < 10 or m is None else ("works" if m > 0 else "fails")
    return out


def evaluate_hypothesis(panel: pd.DataFrame, hyp: dict, tested_signals: dict, protocol: dict) -> dict:
    """Stufen 1–5 (FDR und Locked folgen datenbankweit in run())."""
    acc = protocol["hypothesis_acceptance"]
    per = protocol["periods"]
    rec = {"id": hyp["id"], "title": hyp.get("title"), "statement": hyp.get("statement"),
           "signal": hyp.get("signal"), "direction": int(hyp.get("direction", 1)), "source": hyp.get("source", "config"),
           "created_at": hyp.get("created_at"), "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "stages": {}, "reasons": []}
    try:
        sig = eval_signal(panel, hyp["signal"]) * rec["direction"]
        rec["signal_key"] = signal_key(hyp["signal"])
        rec["stages"]["leakage"] = "pass"
    except (SignalError, KeyError, TypeError) as e:
        rec.update(status="rejected_leakage_or_invalid", reasons=[str(e)])
        rec["stages"]["leakage"] = "fail"
        return rec
    test = window(panel, f"{per['first_test_year']}-01-01", per["locked_from"], per["locked_from"])
    s_test = sig.loc[test.index]
    # Duplikat-Prüfung gegen alle bereits getesteten Signale (feste Stichprobe des Testfensters)
    mine = _fingerprint(s_test)
    for other_id, other in tested_signals.items():
        if other_id == hyp["id"]:
            continue
        rho = _rank_corr(mine, other)
        if rho is not None and abs(rho) >= acc["duplicate_rank_corr"]:
            rec.update(status="duplicate", duplicate_of=other_id, duplicate_rank_corr=ml._r(rho, 3))
            rec["stages"]["duplicate"] = "fail"
            return rec
    rec["stages"]["duplicate"] = "pass"
    rec["_fingerprint"] = mine
    cov = float(s_test.notna().mean()) if len(s_test) else 0.0
    rec["coverage"] = ml._r(cov, 3)
    if cov < acc["min_coverage"]:
        rec.update(status="insufficient_coverage", reasons=[f"Abdeckung {cov:.2f} < {acc['min_coverage']}"])
        rec["stages"]["coverage"] = "fail"
        return rec
    rec["stages"]["coverage"] = "pass"
    ev = evaluate_signal(test, sig, panel)
    rec["walk_forward"] = ev
    rec["p_value"] = ev["p_value"]
    b = ev["base"]
    reasons = []
    if (b.get("t_months") or -9) < acc["min_t_months"] or (b.get("mean") or -1) <= 0:
        reasons.append(f"Walk-Forward netto {b.get('mean')} mit t={b.get('t_months')}")
    if (b.get("years_positive_share") or 0) < acc["min_years_positive"]:
        reasons.append(f"nur {b.get('years_positive_share')} der Jahre positiv")
    if acc.get("stress_must_be_positive") and (ev["stress"].get("mean") or -1) <= 0:
        reasons.append("bei Stresskosten nicht positiv")
    h = ev["halves"]
    if acc.get("halves_same_sign") and not ((h.get("first") or -1) > 0 and (h.get("second") or -1) > 0):
        reasons.append(f"instabil über Hälften {h}")
    rv = _regime_verdicts(ev["regimes"])
    rec["regimes"] = rv
    # k von len(rv) Regimen: höchstens len(rv)-k dürfen scheitern; Regime ohne
    # ausreichend Kohorten (n/a) zählen weder dafür noch dagegen, mind. 2 müssen tragen.
    n_fail = sum(v == "fails" for v in rv.values())
    n_work = sum(v == "works" for v in rv.values())
    if n_fail > len(rv) - acc["min_regimes_same_sign"] or n_work < 2:
        reasons.append(f"Regime: {rv}")
    if (b.get("t_months") or 0) <= -acc["min_t_months"]:
        rec["counter_hypothesis"] = "Gegenrichtung signifikant – wäre eigene, neu zu registrierende Hypothese"
    rec["stages"]["walk_forward"] = "pass" if not reasons else "fail"
    rec["reasons"] = reasons
    rec["status"] = "rejected" if reasons else "passed_pre_fdr"
    return rec


def locked_check(panel: pd.DataFrame, hyp: dict) -> dict:
    per = ml.PROTOCOL["periods"]
    lk = panel[(panel["date"] >= pd.Timestamp(per["locked_from"])) & panel["label_end_20"].notna()]
    sig = eval_signal(panel, hyp["signal"]) * int(hyp.get("direction", 1))
    ev = evaluate_signal(lk, sig, panel)
    return {"base": ev["base"], "ic": ev["ic"], "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ── Discovery-Engine ─────────────────────────────────────────────────────────

def discovery_candidates() -> list[str]:
    """Begrenzter, vorab definierter Suchraum (Anzahl fließt in BH ein)."""
    f = list(ml.STOCK_FEATURES)
    cands = list(f)
    cands += [f"rank({a}) - rank({b})" for a, b in combinations(f, 2)]          # Divergenzen
    cands += [f"{a} * sign(spy_trend_200)" for a in f]                            # trendabhängig
    cands += [f"{a} * sign(vix - 20)" for a in f]                                 # volatilitätsabhängig
    return cands


def discover(panel: pd.DataFrame, protocol: dict) -> dict:
    per, q = protocol["periods"], protocol["multiple_testing"]["fdr_q"]
    disc = window(panel, per["train_start"], per["discovery_end"], per["discovery_end"])
    if disc.empty:
        return {"n_tested": 0, "survivors": []}
    stats, pv = {}, {}
    for expr in discovery_candidates():
        sig = eval_signal(panel, expr)
        ic = ml.daily_ic(disc.assign(_s=sig.loc[disc.index]), "_s")
        st = ml.ic_stats(ic)
        t = st.get("t_months")
        stats[expr] = st
        if t is None:                      # nicht auswertbar zählt trotzdem als Test (konservatives BH)
            pv[expr] = 1.0
            continue
        pv[expr] = min(1.0, 2 * p_one_sided(abs(t)))                              # zweiseitig
    sig_set = benjamini_hochberg(pv, q)
    survivors = []
    for expr in sorted(sig_set, key=lambda e: pv[e]):
        direction = 1 if (stats[expr]["mean_ic"] or 0) > 0 else -1
        survivors.append({"id": f"DISC-{signal_key(expr)}", "title": f"Discovery: {expr}", "signal": expr,
                          "direction": direction, "source": "discovery",
                          "statement": f"Im Discovery-Fenster (bis {per['discovery_end']}) Rank-IC "
                                       f"{stats[expr]['mean_ic']} (t {stats[expr]['t_months']}), BH q={q}",
                          "discovery": {"p_two_sided": ml._r(pv[expr], 6), **stats[expr]}})
    return {"n_tested": len(pv), "n_survivors": len(survivors), "survivors": survivors,
            "window": f"{per['train_start']}..{per['discovery_end']}"}


# ── Datenbank + Lauf ─────────────────────────────────────────────────────────

def load_hypotheses(path: Path = HYP_CONFIG) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("hypotheses") or []


def load_db(path: Path = DB_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"hypotheses": {}}


def run(panel: pd.DataFrame, protocol: dict | None = None, hyp_path: Path = HYP_CONFIG,
        db_path: Path = DB_PATH, with_discovery: bool = True) -> dict:
    protocol = protocol or ml.PROTOCOL
    db = load_db(db_path)
    recs: dict = db.setdefault("hypotheses", {})
    hyps = load_hypotheses(hyp_path)
    if with_discovery:
        d = discover(panel, protocol)
        db["discovery"] = {k: v for k, v in d.items() if k != "survivors"}
        db["discovery"]["survivor_ids"] = [s["id"] for s in d["survivors"]]
        hyps = hyps + d["survivors"]
    tested_signals: dict = {}
    for hyp in hyps:
        status = hyp.get("status")
        if status in ("blocked_data", "prior_result"):
            recs[hyp["id"]] = {**{k: v for k, v in hyp.items()}, "evaluated_at": None}
            continue
        prev = recs.get(hyp["id"])
        try:
            key = signal_key(hyp["signal"])
        except (SignalError, KeyError):
            key = None                                          # Stufe 1 meldet den Fehler
        if prev and prev.get("signal_key") and prev.get("signal_key") != key:
            recs[hyp["id"]] = {**prev, "status": "invalid_modified",
                               "reasons": ["Signal nach Test geändert – neue id nötig"]}
            continue
        rec = evaluate_hypothesis(panel, hyp, tested_signals, protocol)
        if prev and prev.get("locked"):
            rec["locked"] = prev["locked"]                     # Locked nie erneut
        recs[hyp["id"]] = rec
        fp = rec.pop("_fingerprint", None)
        if fp is not None and rec.get("stages", {}).get("coverage") == "pass":
            tested_signals[hyp["id"]] = fp
    # Mehrfachtest über alle je getesteten Hypothesen
    pv = {k: r["p_value"] for k, r in recs.items() if isinstance(r.get("p_value"), (int, float))}
    q = protocol["multiple_testing"]["fdr_q"]
    sig = benjamini_hochberg(pv, q)
    db["n_tested_total"] = len(pv)
    hyp_by_id = {h["id"]: h for h in hyps}
    for k, r in recs.items():
        if r.get("status") != "passed_pre_fdr":
            continue
        if k not in sig:
            r["status"] = "not_significant_after_fdr"
            r["reasons"] = [f"p={r['p_value']} nicht signifikant nach BH (q={q}, n={len(pv)})"]
            continue
        if not r.get("locked") and k in hyp_by_id:
            lk = locked_check(panel, hyp_by_id[k])
            if (lk["base"].get("n_cohorts") or 0) < MIN_LOCKED_COHORTS:
                r["status"] = "passed_pending_locked"          # Locked noch zu dünn -> später, weiterhin nur einmal
                r["reasons"] = [f"Locked-Holdout: erst {lk['base'].get('n_cohorts', 0)} fertige Kohorten"]
                continue
            r["locked"] = lk
        lb = (r.get("locked") or {}).get("base", {})
        ok = (lb.get("mean") or -1) > 0 or not protocol["hypothesis_acceptance"].get("locked_must_be_positive")
        r["status"] = "accepted" if ok else "rejected_locked"
    db["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(db, indent=2, ensure_ascii=False, default=str))
    (db_path.parent / "hypothesis_db.md").write_text(render_md(db), encoding="utf-8")
    return db


def render_md(db: dict) -> str:
    L = [f"# Hypothesen-Datenbank – {db.get('generated')}", "",
         f"Getestet (mit p-Wert): {db.get('n_tested_total')} · Mehrfachtest: Benjamini-Hochberg über alle · "
         f"Discovery: {db.get('discovery', {}).get('n_tested')} Relationen im Fenster "
         f"{db.get('discovery', {}).get('window')}, {db.get('discovery', {}).get('n_survivors')} überlebt", "",
         "| ID | Titel | Quelle | Status | netto (WF) | t | p | Regime | Gründe |", "|---|---|---|---|---|---|---|---|---|"]
    order = {"accepted": 0, "passed_pre_fdr": 1, "rejected_locked": 2, "not_significant_after_fdr": 3}
    for k, r in sorted(db.get("hypotheses", {}).items(), key=lambda kv: (order.get(kv[1].get("status"), 9), kv[0])):
        b = (r.get("walk_forward") or {}).get("base", {})
        reasons = "; ".join(r.get("reasons") or ([r.get("evidence")] if r.get("evidence") else []))[:160]
        L.append(f"| {k} | {r.get('title')} | {r.get('source')} | {r.get('status')} | {b.get('mean')} | "
                 f"{b.get('t_months')} | {r.get('p_value')} | {r.get('regimes', '')} | {reasons} |")
    return "\n".join(L) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    import pickle
    import os
    cache = os.environ.get("ML_PANEL_CACHE")
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
    else:
        panel = ml.build_research_panel()
    db = run(panel)
    print(render_md(db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
