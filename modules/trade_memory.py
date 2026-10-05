"""
modules/trade_memory.py – episodisches Trade-Gedächtnis mit deterministischem
Failure Analyzer.

    python -m modules.trade_memory [--history outputs/history.json] [--out-dir outputs/research]

Reine Auswertung (Forschung). Ändert NIE Produktion (Scores, Gates, PPO,
Gewichte, history.json). Ergebnisse:
  outputs/research/trade_memory.jsonl     ein Fall je geschlossenem Trade
  outputs/research/failure_analysis.json  Aggregation (maschinenlesbar)
  outputs/research/failure_analysis.md    kompakte Zusammenfassung (deutsch)

Grundsätze:
  * Regeln und Schwellen sind VORAB festgelegt (Modulkonstanten, Begründung in
    docs/research/FAILURE_TAXONOMY.md) und werden NICHT auf Ergebnisse optimiert.
    Klassifikation ist rein deterministisch (kein LLM, kein Zufall).
  * Multi-Label + genau eine primäre Ursache nach fester Priorität
    (PRIMARY_PRIORITY). Nur Verlusttrades (outcome < 0) werden klassifiziert.
  * Nicht-RELIABLE Outcomes (modules/outcomes.py: UNKNOWN, RECONSTRUCTED,
    APPROXIMATED): Ursache "unreliable_outcome", keine weitere Klassifikation,
    getrennte Ausweisung in der Aggregation.
  * Kein Blick in die Zukunft: similar_cases() nutzt bei query_date nur Fälle,
    die zu diesem Datum bereits abgeschlossen waren.
  * Fehlende Felder sind der Normalfall (Shadow-Trades, alte Einträge): alles
    über .get(); ein Label wird nur vergeben, wenn seine Eingabefelder da sind.
  * Kleine Stichproben: Anteile sind Beschreibung, keine Signifikanzaussage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from modules.outcomes import is_reliable_outcome, reliability_stamp

log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
OUT_DIR = Path("outputs/research")

# ── Regel-Schwellen (vorab festgelegt, siehe FAILURE_TAXONOMY.md) ────────────
SIGNAL_WRONG_K = 0.5            # Gegenbewegung > K · sigma_daily · sqrt(Handelstage)
FALLBACK_ADVERSE_MOVE = -0.02   # ohne Sigma: Underlying-Rendite (signiert) < -2 %
TRADING_DAY_FACTOR = 5.0 / 7.0  # Kalendertage -> Handelstage (Näherung)
GAVE_BACK_PEAK = 0.30           # Options-Zwischenhoch >= +30 % und Endverlust
ENTRY_SPREAD_HIGH = 0.10        # entry_quote.spread_pct (Anteil, 0.05 = 5 %) > 10 %
LOW_CONFIDENCE_LEVELS = ("low",)
CORE_FEATURES = ("impact", "surprise", "mismatch", "z_score", "sigma_30d")
REGIME_MAX_LAG_DAYS = 3         # Regime-Label höchstens so alt (nur rückwärts, PIT)

# Feste Priorität der primären Ursache (erste zutreffende gewinnt).
PRIMARY_PRIORITY = (
    "unreliable_outcome",
    "exit_gave_back",
    "timing_too_early",
    "signal_wrong",
    "structure_decay",
    "weak_adverse_move",
    "underlying_unknown",
    "entry_cost_high",
    "low_data_confidence",
    "regime_changed",
    "other",
)

NUMERIC_FEATURES = (
    "impact", "surprise", "mismatch", "z_score", "sigma_30d",
    "price_move_48h", "eps_drift", "iv_rank", "quick_mc_hit_rate",
)

BULL_STRATEGIES = ("LONG_CALL", "BULL_CALL_SPREAD")
BEAR_STRATEGIES = ("LONG_PUT", "BEAR_PUT_SPREAD")

MIN_SHARED_FEATURES = 3         # similar_cases: Mindestzahl gemeinsamer Features
DIRECTION_PENALTY = 1.0         # Distanzzuschlag bei entgegengesetzter Richtung
CATALYST_MAX_CHARS = 160


# ── Helfer ───────────────────────────────────────────────────────────────────

def _num(x):
    """Endliche Zahl als float, sonst None (bool zählt nicht)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _parse_date(s):
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _direction(trade: dict):
    """+1 bullisch / -1 bärisch: deep_analysis.direction, sonst Strategie."""
    d = str((trade.get("deep_analysis") or {}).get("direction") or "").upper()
    if d == "BULLISH":
        return 1
    if d == "BEARISH":
        return -1
    strat = trade.get("strategy")
    if strat in BULL_STRATEGIES:
        return 1
    if strat in BEAR_STRATEGIES:
        return -1
    return None


def _ttm_days(ttm) -> float | None:
    """'4-8 Wochen' / '2-3 Monate' / '6 Monate' -> Untergrenze in Tagen.
    Untergrenze = frühester erwarteter Zeitpunkt (konservativ für 'zu früh')."""
    if not isinstance(ttm, str):
        return None
    m = re.match(r"\s*(\d+)(?:\s*-\s*\d+)?\s*(woche|monat|tag)", ttm.lower())
    if not m:
        return None
    n = int(m.group(1))
    return float(n * {"tag": 1, "woche": 7, "monat": 30}[m.group(2)])


def _regime_on(regimes: dict, d: str | None) -> dict | None:
    """Regime am Tag d (oder letzter Tag davor, max. REGIME_MAX_LAG_DAYS)."""
    dd = _parse_date(d)
    if dd is None or not regimes:
        return None
    for lag in range(REGIME_MAX_LAG_DAYS + 1):
        key = date.fromordinal(dd.toordinal() - lag).isoformat()
        if regimes.get(key):
            return dict(regimes[key])
    return None


def make_case_id(ticker, entry_date, strategy) -> str:
    raw = f"{ticker}|{entry_date}|{strategy}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ── A) Fälle bauen ───────────────────────────────────────────────────────────

def build_cases(history: dict, regimes_by_date: dict | None = None) -> list[dict]:
    """Ein Fall je geschlossenem Trade (closed + Shadow mit close_date).
    Doppelte case_id (gleicher Ticker/Datum/Strategie): erster Treffer gilt,
    closed_trades vor shadow_trades."""
    regimes_by_date = regimes_by_date or {}
    cases, seen = [], set()
    for source in ("closed_trades", "shadow_trades"):
        for t in history.get(source) or []:
            if not isinstance(t, dict) or not t.get("close_date"):
                continue
            outcome = _num(t.get("outcome"))
            if outcome is None:
                continue
            case = _build_case(t, source, outcome, regimes_by_date)
            if case["case_id"] in seen:
                log.debug(f"trade_memory: doppelte case_id {case['case_id']} ignoriert")
                continue
            seen.add(case["case_id"])
            cases.append(case)
    cases.sort(key=lambda c: (c.get("close_date") or "", c["case_id"]))
    return cases


def _build_case(t: dict, source: str, outcome: float, regimes: dict) -> dict:
    feats_raw = t.get("features") or {}
    features = {k: _num(feats_raw.get(k)) for k in NUMERIC_FEATURES}
    features = {k: v for k, v in features.items() if v is not None}
    sim = t.get("simulation") or {}
    deep = t.get("deep_analysis") or {}
    quote = t.get("entry_quote") or {}
    opt = t.get("option") or {}

    d_entry, d_close = _parse_date(t.get("entry_date")), _parse_date(t.get("close_date"))
    holding = (d_close - d_entry).days if d_entry and d_close else None
    direction = _direction(t)

    p0, p1 = _num(sim.get("current_price")), _num(t.get("close_price"))
    und_ret = None
    if direction is not None and p0 and p0 > 0 and p1 is not None:
        und_ret = direction * (p1 / p0 - 1.0)

    sigma = _num(sim.get("sigma"))
    if sigma is None:
        sigma = _num(feats_raw.get("sigma_30d"))
    if sigma is not None and sigma <= 0:
        sigma = None

    # Spread: entry_quote.spread_pct, Fallback option.spread_ratio (gleiche Skala: Anteil)
    spread = _num(quote.get("spread_pct"))
    if spread is None:
        spread = _num(opt.get("spread_ratio"))

    reliable = is_reliable_outcome(t)   # eine Definition; Altbestand ohne Methode = UNKNOWN
    catalyst = deep.get("catalyst")
    if isinstance(catalyst, str) and len(catalyst) > CATALYST_MAX_CHARS:
        catalyst = catalyst[:CATALYST_MAX_CHARS - 1] + "…"

    reg_in = _regime_on(regimes, t.get("entry_date"))
    reg_out = _regime_on(regimes, t.get("close_date"))

    case = {
        "case_id": make_case_id(t.get("ticker"), t.get("entry_date"), t.get("strategy")),
        "source": source,
        "ticker": t.get("ticker"),
        "entry_date": t.get("entry_date"),
        "close_date": t.get("close_date"),
        "holding_days": holding,
        "strategy": t.get("strategy"),
        "direction": direction,
        "regime_entry": reg_in,
        "regime_exit": reg_out,
        "macro_regime_llm": deep.get("macro_regime"),
        "features": features,
        "reason_for_trade": {
            "catalyst_type": t.get("catalyst_type"),
            "catalyst": catalyst,
        },
        "model_predictions": {
            "sim_hit_rate": _num(sim.get("hit_rate")),
            "quick_mc_hit_rate": _num(feats_raw.get("quick_mc_hit_rate")),
            "impact": _num(deep.get("impact")),
            "surprise": _num(deep.get("surprise")),
            "catalyst_confidence": _num(deep.get("catalyst_confidence")),
        },
        "position": {
            "entry_debit": _num(t.get("entry_debit")),
            "spread_pct": spread,
        },
        "outcome": outcome,
        "outcome_reliable": reliable,
        "underlying_return_signed": und_ret,
        "max_upside": _num(t.get("peak_return")),
        "exit_reason": t.get("close_reason"),
        "diagnostics": {
            "sigma_daily": sigma,
            "ttm_days": _ttm_days(deep.get("time_to_materialization")),
            "data_confidence": deep.get("data_confidence"),
        },
    }
    fail = classify_failure(case) if outcome < 0 else None
    case["failure"] = fail
    case["what_went_wrong"] = list(fail["labels"]) if fail else []
    return case


# ── B) Failure Analyzer ──────────────────────────────────────────────────────

# Nur Kern-Regime zählen: über ~45 Tage ändert sich fast immer irgendein
# Neben-Label (Öl, Dollar, Strom ...), das wäre unspezifisch (91 % aller Verluste).
REGIME_CORE_KEYS = ("fin_conditions", "credit", "trend", "vix")


def _regime_changed(reg_in, reg_out) -> bool:
    if not reg_in or not reg_out:
        return False
    keys = set(reg_in) & set(reg_out) & set(REGIME_CORE_KEYS)
    return any(reg_in[k] != reg_out[k] for k in keys)


def classify_failure(case: dict) -> dict:
    """Deterministische Ursachen-Klassifikation eines Falls (siehe
    docs/research/FAILURE_TAXONOMY.md). Rückgabe:
        {"primary": code|None, "labels": [codes], "details": {...}}
    Nicht-Verluste (outcome >= 0 oder fehlend) -> primary None, labels []."""
    outcome = _num(case.get("outcome"))
    if outcome is None or outcome >= 0:
        return {"primary": None, "labels": [], "details": {}}
    if case.get("outcome_reliable") is False:
        return {"primary": "unreliable_outcome", "labels": ["unreliable_outcome"], "details": {}}

    diag = case.get("diagnostics") or {}
    und = _num(case.get("underlying_return_signed"))
    hold = _num(case.get("holding_days"))
    sigma = _num(diag.get("sigma_daily"))
    peak = _num(case.get("max_upside"))
    ttm = _num(diag.get("ttm_days"))
    spread = _num((case.get("position") or {}).get("spread_pct"))

    labels, details = set(), {}

    if und is None:
        labels.add("underlying_unknown")
    else:
        if sigma and hold is not None:
            days = max(hold * TRADING_DAY_FACTOR, 1.0)
            thr = -SIGNAL_WRONG_K * sigma * math.sqrt(days)
        else:
            thr = FALLBACK_ADVERSE_MOVE
        details["signal_threshold"] = thr
        if und < thr:
            labels.add("signal_wrong")
        elif und < 0:
            labels.add("weak_adverse_move")
        else:
            labels.add("structure_decay")
        # Zu früh nur bei schwacher Gegenbewegung: eine deutliche Gegenbewegung
        # ist ein falsches Signal, unabhängig vom erwarteten Zeithorizont.
        if thr <= und < 0 and ttm is not None and hold is not None and ttm > hold:
            labels.add("timing_too_early")
            details["ttm_days"] = ttm

    if peak is not None and peak >= GAVE_BACK_PEAK:
        labels.add("exit_gave_back")
    if spread is not None and spread > ENTRY_SPREAD_HIGH:
        labels.add("entry_cost_high")

    feats = case.get("features") or {}
    conf = str(diag.get("data_confidence") or "").lower()
    if conf in LOW_CONFIDENCE_LEVELS or any(feats.get(k) is None for k in CORE_FEATURES):
        labels.add("low_data_confidence")
    if _regime_changed(case.get("regime_entry"), case.get("regime_exit")):
        labels.add("regime_changed")

    if not labels:
        labels.add("other")
    primary = next(c for c in PRIMARY_PRIORITY if c in labels)
    ordered = [c for c in PRIMARY_PRIORITY if c in labels]
    return {"primary": primary, "labels": ordered, "details": details}


# ── C) Aggregation ───────────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _cause_table(losses: list[dict]) -> dict:
    n = len(losses)
    prim, lab = Counter(), Counter()
    outs: dict[str, list[float]] = {}
    for c in losses:
        f = c.get("failure") or {}
        p = f.get("primary") or "other"
        prim[p] += 1
        outs.setdefault(p, []).append(c["outcome"])
        for code in f.get("labels") or []:
            lab[code] += 1
    return {
        "n": n,
        "mean_outcome": _mean([c["outcome"] for c in losses]),
        "primary": {
            code: {"n": k, "share": k / n, "mean_outcome": _mean(outs[code])}
            for code, k in sorted(prim.items(), key=lambda kv: (-kv[1], kv[0]))
        } if n else {},
        "labels": {
            code: {"n": k, "share": k / n}
            for code, k in sorted(lab.items(), key=lambda kv: (-kv[1], kv[0]))
        } if n else {},
    }


def _recent(cases: list[dict], n: int) -> list[dict]:
    return sorted(cases, key=lambda c: (c.get("close_date") or "", c["case_id"]))[-n:] if n > 0 else []


def aggregate_failures(cases: list[dict], last_n: int = 100) -> dict:
    """Anteile der Ursachen über die letzten `last_n` Verlusttrades (zuverlässige
    und unzuverlässige getrennt) + Gewinner-vs-Verlierer-Features (nur
    zuverlässige Outcomes, je Gruppe die letzten `last_n`)."""
    losses = [c for c in cases if c["outcome"] < 0]
    rel_l = _recent([c for c in losses if c.get("outcome_reliable", False)], last_n)
    unr_l = _recent([c for c in losses if not c.get("outcome_reliable", False)], last_n)
    wins = _recent([c for c in cases if c["outcome"] > 0 and c.get("outcome_reliable", False)], last_n)

    comparison = {}
    for k in NUMERIC_FEATURES:
        w = [c["features"][k] for c in wins if k in c.get("features", {})]
        l = [c["features"][k] for c in rel_l if k in c.get("features", {})]
        if not w or not l:
            continue
        sd_all = statistics.pstdev(w + l) if len(w + l) > 1 else 0.0
        diff = _mean(w) - _mean(l)
        comparison[k] = {
            "mean_winners": _mean(w), "mean_losers": _mean(l),
            "n_winners": len(w), "n_losers": len(l),
            "diff": diff,
            "effect_size": diff / sd_all if sd_all > 0 else None,  # standardisiert (Cohen-d-artig)
        }
    return {
        "last_n": last_n,
        "n_cases": len(cases),
        "n_losses_total": len(losses),
        "n_wins_total": sum(1 for c in cases if c["outcome"] > 0),
        "reliable": _cause_table(rel_l),
        "unreliable": _cause_table(unr_l),
        "winner_vs_loser": comparison,
        "note": "Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.",
    }


# ── D) Episodisches Gedächtnis ───────────────────────────────────────────────

def similar_cases(query_features: dict, cases: list[dict], k: int = 10,
                  query_date: str | None = None, query_direction: int | None = None) -> dict:
    """k ähnlichste Fälle: z-standardisierte euklidische Distanz (RMS über die
    gemeinsamen numerischen Features, min. MIN_SHARED_FEATURES). Mit query_date
    nur Fälle, die an diesem Tag bereits abgeschlossen waren (entry_date UND
    close_date < query_date) – kein Blick in die Zukunft. Gleiche Richtung
    bevorzugt (Zuschlag DIRECTION_PENALTY bei entgegengesetzter Richtung).
    Rückgabe: {"cases": [...mit "distance"], "summary": {...}}."""
    q = {f: _num(v) for f, v in (query_features or {}).items()}
    q = {f: v for f, v in q.items() if v is not None}
    qd = _parse_date(query_date) if query_date else None
    pool = []
    for c in cases:
        if qd is not None:
            e, cl = _parse_date(c.get("entry_date")), _parse_date(c.get("close_date"))
            if e is None or cl is None or e >= qd or cl >= qd:
                continue
        pool.append(c)

    stats = {}
    for f in q:
        vals = [c["features"][f] for c in pool if f in c.get("features", {})]
        if len(vals) >= 2:
            sd = statistics.pstdev(vals)
            if sd > 0:
                stats[f] = (statistics.fmean(vals), sd)

    scored = []
    for c in pool:
        sq = []
        for f, (mu, sd) in stats.items():
            v = c.get("features", {}).get(f)
            if v is not None:
                sq.append(((q[f] - v) / sd) ** 2)
        if len(sq) < MIN_SHARED_FEATURES:
            continue
        dist = math.sqrt(sum(sq) / len(sq))
        if query_direction in (1, -1) and c.get("direction") in (1, -1) and c["direction"] != query_direction:
            dist += DIRECTION_PENALTY
        scored.append((dist, c["case_id"], c))
    scored.sort(key=lambda x: (x[0], x[1]))
    top = [dict(c, distance=d) for d, _, c in scored[:max(k, 0)]]

    outs = [c["outcome"] for c in top]
    causes = Counter((c.get("failure") or {}).get("primary") for c in top if c.get("failure"))
    summary = {
        "n": len(top),
        "n_pool": len(pool),
        "n_unreliable": sum(1 for c in top if not c.get("outcome_reliable", False)),
        "win_share": sum(1 for o in outs if o > 0) / len(outs) if outs else None,
        "median_outcome": statistics.median(outs) if outs else None,
        "top_failure_cause": causes.most_common(1)[0][0] if causes else None,
    }
    return {"cases": top, "summary": summary}


# ── E) Ausgabe ───────────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f} %"


def _cause_md(title: str, tab: dict) -> list[str]:
    out = [f"### {title} (n = {tab['n']}, mittleres Outcome {_fmt(tab['mean_outcome'])})", ""]
    if not tab["n"]:
        return out + ["Keine Fälle.", ""]
    out += ["| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |",
            "|---|---:|---:|---:|---:|"]
    for code, r in tab["primary"].items():
        lab = tab["labels"].get(code, {}).get("share")
        out.append(f"| {code} | {r['n']} | {_pct(r['share'])} | {_fmt(r['mean_outcome'])} | {_pct(lab)} |")
    extra = [c for c in tab["labels"] if c not in tab["primary"]]
    for code in extra:
        out.append(f"| ({code}) | 0 | 0 % | – | {_pct(tab['labels'][code]['share'])} |")
    return out + [""]


def render_markdown(agg: dict) -> str:
    L = ["# Failure-Analyse (Trade-Gedächtnis)", "",
         f"Fälle gesamt: {agg['n_cases']} (Gewinner {agg['n_wins_total']}, Verlierer {agg['n_losses_total']}). "
         f"Auswertung über die letzten {agg['last_n']} Verlusttrades je Gruppe.", "",
         "> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades "
         "sind Unterschiede zwischen Ursachen statistisch nicht belastbar.",
         "> Hinweis Outcome-Klassen: Nur RELIABLE (Quote-basiert) wird klassifiziert. UNKNOWN "
         "(Altbestand ohne Preismethode), RECONSTRUCTED (`delta_approx`) und APPROXIMATED werden "
         "nicht weiter klassifiziert und getrennt ausgewiesen.",
         f"> Outcome-Klassen (geschlossen): {agg.get('outcome_classes', {})}", ""]
    L += _cause_md("RELIABLE Verlusttrades", agg["reliable"])
    L += _cause_md("Nicht-RELIABLE Verlusttrades (UNKNOWN/RECONSTRUCTED/APPROXIMATED, explorativ)", agg["unreliable"])
    L += ["### Gewinner vs. Verlierer (nur RELIABLE-Outcomes)", ""]
    wl = agg["winner_vs_loser"]
    if wl:
        L += ["| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |",
              "|---|---:|---:|---:|---:|---:|"]
        for f, r in wl.items():
            L.append(f"| {f} | {_fmt(r['mean_winners'])} | {_fmt(r['mean_losers'])} | {_fmt(r['diff'])} "
                     f"| {_fmt(r['effect_size'], 2)} | {r['n_winners']}/{r['n_losers']} |")
    else:
        L.append("Keine vergleichbaren Features.")
    L += ["", agg["note"], ""]
    return "\n".join(L)


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"nicht serialisierbar: {type(o)}")


def _load_regimes_for(cases_dates: list[str]) -> dict:
    """Regime je Tag (offline): Tagesreports (VIX) + lokales Makro-Archiv."""
    from modules.factor_monitor import load_regimes, macro_regimes, merge_regimes
    return merge_regimes(load_regimes(), macro_regimes(sorted(set(cases_dates))))


def run(history_path=HISTORY_PATH, out_dir=OUT_DIR, last_n: int = 100) -> dict:
    """Fälle bauen, klassifizieren, aggregieren und Dateien schreiben."""
    try:
        history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.error(f"trade_memory: history nicht lesbar ({history_path}): {e}")
        raise

    dates = [t.get(k) for src in ("closed_trades", "shadow_trades") for t in history.get(src) or []
             if isinstance(t, dict) for k in ("entry_date", "close_date") if t.get(k)]
    # Regime pro Tag samt Rückblick-Fenster laden, damit _regime_on Lücken füllen kann
    wanted = set()
    for d in dates:
        dd = _parse_date(d)
        if dd:
            wanted.update(date.fromordinal(dd.toordinal() - i).isoformat() for i in range(REGIME_MAX_LAG_DAYS + 1))
    try:
        regimes = _load_regimes_for(sorted(wanted))
    except (OSError, ValueError, ImportError) as e:
        log.warning(f"trade_memory: Regime nicht verfügbar, Fälle ohne Regime: {e}")
        regimes = {}

    cases = build_cases(history, regimes)
    agg = aggregate_failures(cases, last_n=last_n)
    agg.update(reliability_stamp(history.get("closed_trades") or []))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trade_memory.jsonl", "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False, default=_json_default) + "\n")
    (out / "failure_analysis.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (out / "failure_analysis.md").write_text(render_markdown(agg), encoding="utf-8")
    log.info(f"trade_memory: {len(cases)} Fälle -> {out}")
    return {"n_cases": len(cases), "out_dir": str(out), "aggregate": agg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Episodisches Trade-Gedächtnis + Failure Analyzer")
    ap.add_argument("--history", default=str(HISTORY_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--last-n", type=int, default=100)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(a.history, a.out_dir, a.last_n)
    log.info(f"{res['n_cases']} Fälle geschrieben nach {res['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
