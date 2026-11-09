"""
monthly_report.py – Monatlicher Performance-Report per E-Mail

Läuft am 1. jedes Monats (GitHub Actions: monthly_report.yml) und sendet:
  - A. Verifizierte Performance (nur RELIABLE, modules/outcomes.py) mit INSUFFICIENT_EVIDENCE bei zu wenig Daten
  - B. Historische Performance – EXPLORATIV (alle Evidenzklassen; kein Alpha-Nachweis): Win-Rate,
    Trade-Anzahl, Mean/Median-Return, Totalverluste (<= -95 %) des Vormonats (close_date im Vormonat)
  - C. Forschungs- und Lernfortschritt aus vorhandenen Forschungsdateien (fehlend = fehlend, nie 0)
  - Rollende Statistiken (30er, 60er Fenster, alle) mit 90%-Konfidenzintervallen über unabhängige
    Haltefenster (dedupliziert je Kontrakt) — kein KI bei zu wenigen unabhängigen Fenstern
  - Gesamt-Statistik über alle closed_trades
  - Signal-Funnel: Wie viele Tage hatten 0 Trades und welche Gates blockierten

Nutzt dieselben GMAIL_SENDER / GMAIL_APP_PW / NOTIFY_EMAIL Secrets wie der
tägliche Scanner. Keine zusätzlichen Kosten, keine neuen API-Keys.
"""

import json
import logging
import math
import os
import random
from html import escape
import statistics
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from modules.email_reporter import _send_smtp
from modules.outcomes import TOTAL_LOSS_THRESHOLD, class_counts, is_total_loss, outcome_class


# ── Field Mapping: suggestion gate names → ledger field paths ────────────────
# Maps the 'field' returned by suggest_thresholds to the ledger field path used
# in challenger.yaml rules. These must match the actual storage format in the
# candidate_ledger JSON.
LEDGER_FIELD_MAP = {
    "mismatch": "features.mismatch",
    "impact": "features.impact",
    "surprise": "features.surprise",
    "score": "features.trade_score",
    "dte": None,  # no direct ledger field for DTE
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
REPORTS_DIR  = Path("outputs/daily_reports")
REPO_NAME    = os.getenv("GITHUB_REPOSITORY", "Adaptive-Asymmetry-Scanner")


def trade_identity(trade: dict) -> tuple:
    """Identität eines Trades = Underlying, Entry-Tag, Strategie und KONTRAKT (Strike(s), Verfall, Symbol).
    Zwei verschiedene Kontrakte desselben Tickers am selben Tag (z. B. Put und Call) sind zwei Trades."""
    o = trade.get("option") if isinstance(trade.get("option"), dict) else {}
    leg = o.get("spread_leg") if isinstance(o.get("spread_leg"), dict) else {}
    return (trade.get("ticker", "?"), str(trade.get("entry_date", "?"))[:10], trade.get("strategy"),
            o.get("symbol"), o.get("strike"), o.get("expiry") or o.get("expiration"), leg.get("strike"))


def dedup_closed_trades(closed: list[dict]) -> list[dict]:
    """
    Deduplicate trades: keep the first occurrence of each trade identity (trade_identity).
    Audit 2026-11-01: früher nur (ticker, entry_date) – das legte zwei verschiedene UNH-Kontrakte vom
    2026-04-11 (Put 310 / Call 330) zusammen und verwarf einen davon (116 statt 117 Trades).
    """
    seen = set()
    result = []
    for trade in closed:
        key = trade_identity(trade)
        if key not in seen:
            seen.add(key)
            result.append(trade)
    return result


def rolling_stats(trades: list[dict], window: int | None = None) -> dict:
    """
    Compute rolling window stats on trades (sorted by close_date).
    Returns: n, win_rate, mean, median, total_loss_rate, mean_ex_top3, profit_factor, profit_factor_n_losses.
    If window is None, uses all trades.
    """
    if window is not None:
        trades = trades[-window:]

    outs = [float(t["outcome"]) for t in trades if t.get("outcome") is not None]

    if not outs:
        return {
            "n": 0,
            "win_rate": None,
            "mean": None,
            "median": None,
            "total_loss_rate": None,
            "mean_ex_top3": None,
            "profit_factor": None,
        }

    wins = sum(1 for o in outs if o > 0)
    win_rate = wins / len(outs)

    # total_loss_rate: Anteil Totalverluste (modules.outcomes.TOTAL_LOSS_THRESHOLD, <= -95 %)
    total_losses = sum(1 for o in outs if is_total_loss(o))
    total_loss_rate = total_losses / len(outs)

    # mean_ex_top3: mean after removing 3 best outcomes
    sorted_outs = sorted(outs, reverse=True)
    if len(sorted_outs) > 3:
        mean_ex_top3 = statistics.mean(sorted_outs[3:])
    else:
        mean_ex_top3 = None

    # profit_factor: sum of wins / abs(sum of losses)
    sum_wins = sum(o for o in outs if o > 0)
    sum_losses = sum(o for o in outs if o <= 0)
    if sum_losses >= 0 or sum_losses == 0:
        profit_factor = None
    else:
        profit_factor = sum_wins / abs(sum_losses)

    return {
        "n": len(outs),
        "win_rate": win_rate,
        "mean": statistics.mean(outs),
        "median": statistics.median(outs),
        "total_loss_rate": total_loss_rate,
        "mean_ex_top3": mean_ex_top3,
        "profit_factor": profit_factor,
    }


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float] | None:
    """
    Compute 90% confidence interval (5th/95th percentile) for the mean via bootstrap.
    Returns (lower, upper) or None if len(values) < 10.
    """
    if len(values) < 10:
        return None

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(len(values))]
        boot_means.append(statistics.mean(sample))

    boot_means.sort()
    lower_idx = int(n_boot * 0.05)
    upper_idx = int(n_boot * 0.95)
    return (boot_means[lower_idx], boot_means[upper_idx])


# ── Evidenzqualität (Audit 2026-11-01) ───────────────────────────────────────
# Eine Klassifizierung: modules/outcomes.outcome_class (RELIABLE | RECONSTRUCTED | APPROXIMATED | UNKNOWN).
# Belastbar ist nur RELIABLE; alles andere ist explorativ und wird nie als Alpha-Nachweis dargestellt.
EVIDENCE_CLASSES = ("RELIABLE", "RECONSTRUCTED", "APPROXIMATED", "UNKNOWN")
MIN_EVIDENCE_N = 20                 # wie backtest_thresholds.MIN_N_FOR_SUGGESTION: darunter keine Statistik
MIN_INDEPENDENT_WINDOWS = 5         # wie option_learner.MIN_INDEPENDENT_PERIODS: darunter kein Konfidenzintervall
HOLDING_WINDOW_DAYS = int(os.getenv("CLOSE_AFTER_DAYS", "45"))
EXPLORATIVE_BANNER = ("EXPLORATIV – enthält rekonstruierte beziehungsweise nicht ausreichend verifizierte "
                      "Options-Ergebnisse. Kein belastbarer Alpha-Nachweis.")
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
TRADE_MEMORY_PATH = Path("outputs/research/trade_memory.jsonl")
FAILURE_ANALYSIS_PATH = Path("outputs/research/failure_analysis.json")
OPTION_LEARNER_DIR = Path("outputs/research/option_learner")
ALPHA_VALIDATION_PATH = Path("outputs/research/autonomous_alpha/option_validation.json")


def _outcome(t: dict) -> float | None:
    try:
        v = float(t.get("outcome"))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _entry_day(t: dict) -> date | None:
    try:
        return date.fromisoformat(str(t.get("entry_date"))[:10])
    except ValueError:
        return None


def holding_windows(trades: list[dict]) -> list[int | None]:
    """Unabhängige Beobachtungsgruppen: nicht überlappende Haltefenster (HOLDING_WINDOW_DAYS ab dem frühesten Entry).
    Trades desselben Fensters überlappen zeitlich und sind voneinander abhängig."""
    days = [_entry_day(t) for t in trades]
    known = [d for d in days if d is not None]
    if not known:
        return [None] * len(trades)
    origin = min(known)
    return [None if d is None else (d - origin).days // HOLDING_WINDOW_DAYS for d in days]


def grouped_ci(trades: list[dict], alpha: float = 0.10) -> dict:
    """90-%-KI des Mittelwerts per Bootstrap über ganze Haltefenster (modules.evidence_core.group_bootstrap_ci).
    Weniger als MIN_INDEPENDENT_WINDOWS Fenster -> kein Intervall (statt eines scheinbar präzisen iid-KI)."""
    pairs = [(o, g) for t, g in zip(trades, holding_windows(trades))
             if (o := _outcome(t)) is not None and g is not None]
    k = len({g for _, g in pairs})
    if k < MIN_INDEPENDENT_WINDOWS:
        return {"ci": None, "independent_windows": k,
                "reason": f"nur {k} unabhängige Haltefenster ({HOLDING_WINDOW_DAYS} T; mind. {MIN_INDEPENDENT_WINDOWS})"}
    from modules import evidence_core as ec
    lo, hi = ec.group_bootstrap_ci([o for o, _ in pairs], [g for _, g in pairs], alpha=alpha)
    return {"ci": None if lo is None else (lo, hi), "independent_windows": k, "reason": None}


def evidence_stats(trades: list[dict]) -> dict:
    """Kennzahlen einer Population; fehlende Outcomes zählen nie als 0. Status INSUFFICIENT_EVIDENCE, wenn zu wenige
    Outcomes oder unabhängige Haltefenster vorliegen – die Werte sind dann nur beschreibend."""
    outs = [o for t in trades if (o := _outcome(t)) is not None]
    rs = rolling_stats([{"outcome": o} for o in outs])
    ci = grouped_ci([t for t in trades if _outcome(t) is not None])
    enough = len(outs) >= MIN_EVIDENCE_N and ci["independent_windows"] >= MIN_INDEPENDENT_WINDOWS
    return {"n_trades": len(trades), "n_evaluable": len(outs), "n_missing_outcome": len(trades) - len(outs),
            "win_rate": rs["win_rate"], "mean": rs["mean"], "median": rs["median"],
            "profit_factor": rs["profit_factor"], "total_loss_rate": rs["total_loss_rate"],
            "mean_ex_top3": rs["mean_ex_top3"], "independent_windows": ci["independent_windows"],
            "ci90": ci["ci"] if enough else None, "status": "EVALUABLE" if enough else INSUFFICIENT,
            "reason": None if enough else (f"n={len(outs)} < {MIN_EVIDENCE_N}" if len(outs) < MIN_EVIDENCE_N
                                           else ci["reason"])}


def _load_trade_memory(path: Path | None = None) -> dict:
    """case_id -> Fall aus dem Trade-Gedächtnis (Regime zum Entry); fehlt die Datei: leer."""
    path = path or TRADE_MEMORY_PATH
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line) if line.strip() else None
            if isinstance(c, dict) and c.get("case_id"):
                out[c["case_id"]] = c
    except (OSError, ValueError):
        return {}
    return out


def _regime_label(t: dict, memory: dict) -> str:
    from modules.trade_memory import make_case_id
    c = memory.get(make_case_id(t.get("ticker"), t.get("entry_date"), t.get("strategy"))) or {}
    vix = (c.get("regime_entry") or {}).get("vix") if isinstance(c.get("regime_entry"), dict) else None
    return str(vix) if vix else "unbekannt"


def _sector_label(t: dict) -> str:
    return str(t.get("sector_etf") or (t.get("features") or {}).get("sector") or "unbekannt")


def evidence_report(closed: list[dict], report_month: str | None = None, memory: dict | None = None) -> dict:
    """Alle Kennzahlen getrennt nach Evidenzklasse; Aufschlüsselungen (Struktur, Sektor, Regime) nur für RELIABLE."""
    memory = memory if memory is not None else _load_trade_memory()
    by_cls = {c: [t for t in closed if outcome_class(t) == c] for c in EVIDENCE_CLASSES}
    rel = by_cls["RELIABLE"]

    def split(key):
        g: dict = {}
        for t in rel:
            g.setdefault(key(t), []).append(t)
        return {k: evidence_stats(v) for k, v in sorted(g.items())}
    month_rel = [t for t in rel if report_month and str(t.get("close_date", ""))[:7] == report_month]
    return {"classes": {c: evidence_stats(v) for c, v in by_cls.items()}, "class_counts": class_counts(closed),
            "reliable": evidence_stats(rel), "reliable_month": evidence_stats(month_rel),
            "explorative": evidence_stats(closed),
            "by_strategy": split(lambda t: str(t.get("strategy") or "unbekannt")),
            "by_sector": split(_sector_label), "by_regime": split(lambda t: _regime_label(t, memory)),
            "total_loss_definition": f"Outcome ≤ {TOTAL_LOSS_THRESHOLD * 100:.0f} %"}


def month_key(d: date) -> str:
    return d.strftime("%Y-%m")


def prev_month(key: str) -> str:
    y, m = map(int, key.split("-"))
    return f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"


def month_stats(closed: list[dict], key: str) -> dict | None:
    """Statistik über Trades, die im Monat `key` geschlossen wurden."""
    outs = [
        float(t["outcome"]) for t in closed
        if t.get("outcome") is not None
        and str(t.get("close_date", ""))[:7] == key
    ]
    if not outs:
        return None
    wins = sum(1 for o in outs if o > 0)
    return {
        "n":            len(outs),
        "win_rate":     wins / len(outs),
        "wins":         wins,
        "mean":         statistics.mean(outs),
        "median":       statistics.median(outs),
        "total_losses": sum(1 for o in outs if is_total_loss(o)),
    }


def overall_stats(closed: list[dict]) -> dict | None:
    outs = [float(t["outcome"]) for t in closed if t.get("outcome") is not None]
    if not outs:
        return None
    wins = sum(1 for o in outs if o > 0)
    return {
        "n": len(outs), "win_rate": wins / len(outs), "wins": wins,
        "mean": statistics.mean(outs), "median": statistics.median(outs),
        "total_losses": sum(1 for o in outs if is_total_loss(o)),
    }


def funnel_summary(key: str) -> dict:
    """Aggregiert Daily-JSONs des Monats: 0-Trade-Tage + Top-Reject-Gründe."""
    days = zero_days = 0
    rejects: Counter = Counter()
    stop_reasons: Counter = Counter()
    for f in sorted(REPORTS_DIR.glob(f"{key}-*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        days += 1
        s = d.get("stats", {})
        n_trades = s.get("trades", len(d.get("proposals", []) or []))
        if not n_trades:
            zero_days += 1
            if s.get("stop_reason"):
                stop_reasons[s["stop_reason"]] += 1
        for reason, info in (d.get("rejects") or {}).items():
            rejects[reason] += info.get("count", 0)
    return {
        "days": days, "zero_days": zero_days,
        "top_rejects": rejects.most_common(5),
        "top_stops":   stop_reasons.most_common(3),
    }


def spy_return(key: str) -> float | None:
    """SPY-Return im Monat `key` via yfinance (kostenlos)."""
    try:
        import yfinance as yf
        y, m = map(int, key.split("-"))
        start = date(y, m, 1)
        end   = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        hist  = yf.Ticker("SPY").history(
            start=start.isoformat(), end=end.isoformat(), auto_adjust=True
        )
        if len(hist) < 2:
            return None
        return float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
    except Exception as e:
        log.warning(f"SPY-Benchmark nicht abrufbar: {e}")
        return None


def shadow_stats(history: dict, key: str) -> dict | None:
    """Outcome-Statistik der Schatten-Trades des Monats, GETRENNT je Reject-Grund.

    Verschiedene Gates erzeugen verschiedene Populationen (ROI-Gate: Optionsrendite
    des besten Tiers; Final-MC-Survivor: ohne Strategie). Eine gemischte Win-Rate
    wäre kein Gate-Audit. Schatten-Outcomes werden zum Halte-Ende ohne TP/SL bewertet
    -> nur innerhalb eines Grundes vergleichbar, Vergleich mit echten Trades nur Hinweis.
    """
    shadows = [
        t for t in history.get("shadow_trades", [])
        if t.get("outcome") is not None
        and str(t.get("close_date", ""))[:7] == key
    ]
    if not shadows:
        return None
    by: dict[str, list[float]] = {}
    for t in shadows:
        r = str(t.get("reject_reason") or "unbekannt")
        by.setdefault("score_gate" if r.startswith("score_") else r, []).append(float(t["outcome"]))
    groups = {}
    for r, outs in sorted(by.items()):
        wins = sum(1 for o in outs if o > 0)
        groups[r] = {"n": len(outs), "win_rate": wins / len(outs), "wins": wins, "mean": statistics.mean(outs)}
    outs = [o for v in by.values() for o in v]
    wins = sum(1 for o in outs if o > 0)
    return {"n": len(outs), "win_rate": wins / len(outs), "wins": wins,
            "mean": statistics.mean(outs), "by_reason": groups}


def current_thresholds() -> dict:
    """Aktuelle Gate-Schwellen aus config.yaml / Code-Konstanten."""
    try:
        from modules.config import cfg
        from modules.options_designer import ttm_to_dte_floor
        gate_cfg     = getattr(cfg, "gates", None)
        mismatch_cap = float(getattr(gate_cfg, "mismatch_max",
                              getattr(getattr(cfg, "pipeline", None), "max_mismatch", 7.0)))
        impact_floor = int(getattr(gate_cfg, "impact_min",
                           getattr(getattr(cfg, "pipeline", None), "min_impact_threshold", 4)))
        surprise_floor = int(getattr(gate_cfg, "surprise_min", 3))
        score_floor    = int(getattr(gate_cfg, "trade_score_min", 55))
        dte_floor      = ttm_to_dte_floor("")  # unbekanntes TTM → konservativer Default
    except Exception:
        mismatch_cap, impact_floor, surprise_floor, score_floor, dte_floor = 7.0, 4, 3, 55, 120
    return {
        "dte":      dte_floor,      # ttm_to_dte_floor-Default (options_designer.py, v11.0)
        "mismatch": mismatch_cap,
        "impact":   impact_floor,
        "surprise": surprise_floor, # Impact×Surprise-Floor (pipeline.py Stufe 4b)
        "score":    score_floor,    # Trade-Score-Gate (pipeline.py Stufe 10)
    }


def tuning_suggestions(history: dict) -> list[dict]:
    try:
        from backtest_thresholds import suggest_thresholds
        return suggest_thresholds(history, current_thresholds())
    except Exception as e:
        log.warning(f"Tuning-Vorschläge nicht berechenbar: {e}")
        return []


def slot_candidates(history: dict) -> list[dict]:
    """Kandidaten aus den laufenden Schatten-Messungen für den nächsten Slot."""
    try:
        from backtest_thresholds import slot_candidate_analysis
        return slot_candidate_analysis(history)
    except Exception as e:
        log.warning(f"Slot-Analyse nicht berechenbar: {e}")
        return []


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def challenger_snippet(suggestion: dict, today: date) -> str:
    """
    Generate a ready-to-copy YAML snippet for challengers.yaml from a suggestion
    returned by suggest_thresholds().

    Args:
        suggestion: dict with keys gate, field, mode, current, suggested, gain_pp, etc.
        today: the date to use for registered_on

    Returns:
        YAML text that can be pasted into challengers.yaml, or a comment explaining
        why the suggestion cannot be rendered (e.g., no ledger field).
    """
    field = suggestion.get("field")
    ledger_field = LEDGER_FIELD_MAP.get(field)

    if ledger_field is None:
        # Field has no ledger mapping — cannot be evaluated prospectively
        gate_label = suggestion.get("gate", field)
        return f"# {gate_label}: kein Ledger-Feld – Challenger nicht direkt auswertbar"

    # Generate a sanitized ID from the gate and new value
    gate_label = suggestion.get("gate", field).lower().replace(" ", "_").replace("-", "_")
    new_value = suggestion.get("suggested")
    try:
        new_value_str = str(new_value).replace(".", "p")
    except Exception:
        new_value_str = "unknown"

    challenger_id = f"{gate_label}_{new_value_str}"

    # Build hypothesis sentence
    gate_name = suggestion.get("gate", field)
    mode = suggestion.get("mode", "min")
    op_str = "≥" if mode == "min" else "≤"
    current = suggestion.get("current")
    suggested = suggestion.get("suggested")
    gain_pp = suggestion.get("gain_pp", 0)

    hypothesis = (
        f"Gate '{gate_name}' geändert: {op_str} {current} → {op_str} {suggested}. "
        f"Retrospektiv +{gain_pp:.0f}pp Win-Rate auf bereits gesehenen Daten. "
        f"Nur mit prospektivem Walk-forward validieren."
    )

    # Build rule: {field, op, value}
    op_map = {"min": ">=", "max": "<="}
    op = op_map.get(mode, ">=")

    # Build baseline_rule (current value)
    baseline_op = op_map.get(mode, ">=")

    tomorrow = today + timedelta(days=1)
    start_date_str = tomorrow.isoformat()
    today_str = today.isoformat()

    yaml_snippet = f"""  - id: {challenger_id}
    hypothesis: >
      {hypothesis}
    registered_on: "{today_str}"
    start_date: "{start_date_str}"
    rule:
      - field: "{ledger_field}"
        op: "{op}"
        value: {suggested}
    baseline_rule:
      - field: "{ledger_field}"
        op: "{baseline_op}"
        value: {current}
    metric: "outcomes.real_strat_ret_45d"
    min_n: 30
    horizon_days: 45
    max_duration_days: 180
    status: active
"""
    return yaml_snippet


def suggestion_evidence(history: dict | None, s: dict) -> dict:
    """Evidenzbasis eines retrospektiven Vorschlags: Klassen der Outcomes in der vorgeschlagenen Teilmenge und
    unabhängige Haltefenster der RELIABLE-Fälle. Challenger-Vorlage nur bei ausreichender verifizierter Basis."""
    try:
        from backtest_thresholds import MIN_N_FOR_SUGGESTION, _select, trade_rows
    except ImportError:
        return {"classes": {}, "reliable_n": 0, "windows": 0, "template_allowed": False}
    rows = [r for r in trade_rows(history or {}) if isinstance(r.get(s.get("field")), (int, float))]
    sel = _select(rows, s["field"], s["suggested"], s.get("mode", "min")) if rows else []
    classes = dict(Counter(r.get("outcome_class", "UNKNOWN") for r in sel))
    rel = [{"outcome": r["outcome"], "entry_date": r.get("entry_date")} for r in sel
           if r.get("outcome_class") == "RELIABLE"]
    windows = grouped_ci(rel)["independent_windows"] if rel else 0
    allowed = len(rel) >= max(MIN_N_FOR_SUGGESTION, MIN_EVIDENCE_N) and windows >= MIN_INDEPENDENT_WINDOWS
    return {"classes": classes, "reliable_n": len(rel), "windows": windows, "template_allowed": allowed}


def build_tuning_html(suggestions: list[dict], history: dict | None = None) -> str:
    today = date.today()
    if not suggestions:
        return (
            "<h3>💡 Hypothesen-Generator (retrospektiv, NICHT promotion-fähig)</h3>"
            "<p>Keine Empfehlung diesen Monat — keine Alternative erfüllt die "
            "Guardrails (≥20 Trades, ≥5pp Win-Rate-Gewinn, Ø-Return nicht schlechter). "
            "Das ist ein gutes Zeichen oder es fehlen noch Daten.</p>"
        )
    rows = ""
    yaml_snippets = ""
    any_template = False
    for s in suggestions:
        cs, ss = s["current_stats"], s["suggested_stats"]
        op = "≥" if s["mode"] == "min" else "≤"
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'><b>{s['gate']}</b></td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{op} {s['current']} "
            f"({_fmt_pct(cs['win_rate'])} Win, n={cs['n']})</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;color:#16a34a;'>"
            f"<b>{op} {s['suggested']}</b> ({_fmt_pct(ss['win_rate'])} Win, "
            f"Ø {ss['mean']:+.1%}, n={ss['n']})</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'><b>+{s['gain_pp']:.0f}pp</b></td>"
            f"</tr>"
        )
        # Audit 2026-11-01: Vorlage nur bei ausreichender VERIFIZIERTER Basis – explorative Vorschläge bekommen
        # keine kopierfertige Challenger-Registrierung (kein Testslot für unbelegte Hypothesen).
        se = suggestion_evidence(history, s)
        basis = escape(f"Evidenzbasis der Teilmenge: {se['classes'] or 'unbekannt'}; RELIABLE n={se['reliable_n']}, "
                       f"{se['windows']} unabhängige Haltefenster")
        any_template = any_template or se["template_allowed"]
        yaml_snippets += f"<p style='font-size:0.80em;margin-top:1em;'><b>{s['gate']}:</b> {basis}</p>\n"
        if se["template_allowed"]:
            snippet = challenger_snippet(s, today)
            yaml_snippets += f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;overflow-x:auto;font-size:0.75em;'>{snippet}</pre>\n"
        else:
            yaml_snippets += ("<p style='font-size:0.80em;color:#92400e;'><b>EXPLORATIV – keine Challenger-Vorlage.</b> "
                              "Die Teilmenge beruht überwiegend auf nicht verifizierten Ergebnissen (mind. "
                              f"{MIN_EVIDENCE_N} RELIABLE-Outcomes und {MIN_INDEPENDENT_WINDOWS} unabhängige "
                              "Haltefenster nötig). Kein Testslot, keine Registrierung; Schwellen bleiben unverändert. "
                              "Eine prospektive Prüfung wäre nur nach menschlicher Freigabe neu zu registrieren.</p>\n")

    return f"""
    <h3>💡 Hypothesen-Generator (retrospektiv, NICHT promotion-fähig)</h3>
    <p style="font-size:0.90em;color:#555;">
    Diese Vorschläge basieren auf retrospektiver In-Sample-Analyse bereits gesehener Daten.
    Sie sind <b>NICHT promotion-fähig</b> und dürfen <b>NICHT direkt</b> in config.yaml
    oder pipeline.py übernommen werden. Die einzige zulässige Konsequenz ist die
    <b>Registrierung eines neuen Challengers</b> in <code>challengers.yaml</code> mit
    <code>registered_on = heute</code> und <code>start_date > heute</code>.
    Der Challenger wird dann prospektiv (Walk-Forward) auf zukünftigen Daten validiert.
    </p>
    <table style="border-collapse:collapse;font-size:13px;width:100%;margin-top:1em;">
      <tr style="text-align:left;color:#64748b;">
        <th style="padding:4px 8px;">Gate</th><th style="padding:4px 8px;">Aktuell</th>
        <th style="padding:4px 8px;">Vorschlag</th><th style="padding:4px 8px;">Δ Win-Rate</th>
      </tr>
      {rows}
    </table>
    <p style="font-size:0.85em;color:#555;margin-top:1em;"><b>Challenger-Vorlagen:</b></p>
    {yaml_snippets}
    {_snippet_note(any_template)}"""


def _snippet_note(any_template: bool) -> str:
    if not any_template:
        return ("<p style='font-size:0.80em;color:#666;'><i>Keine Challenger-Vorlage: kein Vorschlag hat eine "
                "ausreichende verifizierte Datenbasis.</i></p>")
    return ("<p style='font-size:0.80em;color:#666;'><i>Diese Snippets können in challengers.yaml kopiert und angepasst "
            "werden. Jeder Challenger wird mit der heutigen Registrierung und morgen als start_date erstellt, um "
            "Walk-Forward-Validität zu gewährleisten.</i></p>")


def build_slot_html(candidates: list[dict], evidence: dict | None = None) -> str:
    """
    Entscheidungsvorlage für den nächsten Tuning-Slot — fasst die laufenden
    Schatten-Messungen (Trailing-Stop, Spreads vs. Long Calls, Sektor-
    Konzentration) zusammen. Reines Reporting, ändert nichts.
    """
    if not candidates:
        return (
            "<h3>🧭 Slot-Analyse — Kandidaten für den nächsten Tuning-Slot</h3>"
            "<p>Keine Kandidaten berechenbar (fehlende Daten).</p>"
        )
    rows = ""
    for c in candidates:
        ready = c.get("ready", False)
        style = "" if ready else "color:#94a3b8;"
        status = "✅" if ready else "⏳"
        rec = c.get("recommendation") or "—"
        rows += (
            f"<tr style='{style}'>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{status} "
            f"<b>{c.get('name', '?')}</b></td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>n={c.get('n', 0)}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{c.get('finding', '')}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'><i>{rec}</i></td>"
            f"</tr>"
        )
    return f"""
    <h3>🧭 Slot-Analyse — Kandidaten für den nächsten Tuning-Slot</h3>
    <table style="border-collapse:collapse;font-size:13px;width:100%">
      <tr style="text-align:left;color:#64748b;">
        <th style="padding:4px 8px;">Analyse</th><th style="padding:4px 8px;">n</th>
        <th style="padding:4px 8px;">Befund</th><th style="padding:4px 8px;">Empfehlung</th>
      </tr>
      {rows}
    </table>
    <p style="font-size:0.85em;color:#92400e;">⚠️ Ausgegraute Zeilen (⏳) haben noch zu wenig
    Daten — Empfehlung lautet dann "weiter Daten sammeln". Es gilt weiterhin: max. eine
    Schwellen-Änderung pro Monat (Tuning-Slot), und der User entscheidet, ob und welcher
    Kandidat umgesetzt wird.</p>{_slot_evidence_note(evidence)}"""


def _slot_evidence_note(evidence: dict | None) -> str:
    if not evidence:
        return ""
    strat = "; ".join(f"{k}: n={v['n_evaluable']}, Ø {_p(v['mean'])}, {v['status']}"
                      for k, v in evidence.get("by_strategy", {}).items()) or "keine RELIABLE-Trades"
    return (f"<p style='font-size:0.85em;background:#fef3c7;padding:4px;'><b>{EXPLORATIVE_BANNER}</b> Die Slot-Analyse "
            f"nutzt alle closed_trades (Klassen {evidence.get('class_counts')}); ✅ bedeutet nur genug Fälle, keine "
            f"verifizierte Evidenz. Verifiziert (RELIABLE) nach Struktur: {strat}.</p>")


def build_rolling_stats_html(closed: list[dict], prev_closed_stats: dict | None = None,
                             cur_stats: dict | None = None) -> str:
    """
    Build HTML table showing rolling window statistics.
    Includes 30-day, 60-day, and all-trades windows.
    """
    # Sort by close_date to have consistent ordering
    sorted_trades = sorted(
        [t for t in dedup_closed_trades(closed)
         if t.get("outcome") is not None and t.get("close_date")],
        key=lambda t: t.get("close_date", "")
    )

    if not sorted_trades:
        return "<p><i>Keine geschlossenen Trades für Rolling-Statistik.</i></p>"

    windows = [
        ("Letzte 30", 30),
        ("Letzte 60", 60),
        ("Alle Trades", None),
    ]

    rows = ""

    for label, window in windows:
        stats = rolling_stats(sorted_trades, window)

        if stats["n"] == 0:
            rows += f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{label}</td><td colspan='7' style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'><i>keine Daten</i></td></tr>"
            continue

        win_rate_str = f"{stats['win_rate'] * 100:.0f}%" if stats['win_rate'] is not None else "—"
        mean_str = f"{stats['mean']:+.1%}" if stats['mean'] is not None else "—"
        median_str = f"{stats['median']:+.1%}" if stats['median'] is not None else "—"
        mean_ex_top3_str = f"{stats['mean_ex_top3']:+.1%}" if stats['mean_ex_top3'] is not None else "—"
        total_loss_str = f"{stats['total_loss_rate'] * 100:.0f}%" if stats['total_loss_rate'] is not None else "—"

        # Audit 2026-11-01: KI über ganze, nicht überlappende Haltefenster (abhängige Trades desselben Fensters
        # zusammen gezogen); zu wenige Fenster -> kein Intervall statt eines scheinbar präzisen iid-Bootstraps.
        gci = grouped_ci(sorted_trades[-window:] if window else sorted_trades)
        if gci["ci"]:
            ci_str = f"{gci['ci'][0]:+.1%} bis {gci['ci'][1]:+.1%} ({gci['independent_windows']} Fenster)"
        else:
            ci_str = f"— ({gci['reason']})"

        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'><b>{label}</b></td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{stats['n']}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{win_rate_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{mean_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{median_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{mean_ex_top3_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{total_loss_str}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{ci_str}</td>"
            f"</tr>"
        )

    # Δ Win-Rate Monat vs. Vormonat (neutral formuliert). "Signifikant" nur,
    # wenn sich das 90%-KI der letzten 30 Trades nicht mit dem KI der
    # 30 Trades davor überschneidet.
    delta_html = ""
    if prev_closed_stats and cur_stats:
        delta = (cur_stats["win_rate"] - prev_closed_stats["win_rate"]) * 100
        last_ci = grouped_ci(sorted_trades[-30:])["ci"] if len(sorted_trades) >= 60 else None
        prior_ci = grouped_ci(sorted_trades[-60:-30])["ci"] if len(sorted_trades) >= 60 else None
        if last_ci and prior_ci and last_ci[0] > prior_ci[1]:
            significance = "signifikant besser (letzte 30 vs. 30 davor, KIs überschneiden sich nicht)"
        elif last_ci and prior_ci and last_ci[1] < prior_ci[0]:
            significance = "signifikant schlechter (letzte 30 vs. 30 davor, KIs überschneiden sich nicht)"
        else:
            significance = "kein Nachweis einer Veränderung"
        delta_html = (
            f"<p style='margin-top:1em;font-size:0.95em;color:#555;'>"
            f"<b>Δ Win-Rate ggü. Vormonat:</b> {delta:+.0f} Prozentpunkte "
            f"(n={cur_stats.get('n', '?')} vs. {prev_closed_stats.get('n', '?')}) — "
            f"bei dieser Stichprobe <b>{significance}</b>.</p>"
        )

    return f"""
    <h3>📊 Beobachtete Performance (rollierend, alle Evidenzklassen – explorativ)</h3>
    <table style="border-collapse:collapse;font-size:13px;width:100%">
      <tr style="text-align:left;color:#64748b;">
        <th style="padding:4px 8px;">Fenster</th>
        <th style="padding:4px 8px;">n</th>
        <th style="padding:4px 8px;">Win-Rate</th>
        <th style="padding:4px 8px;">Ø</th>
        <th style="padding:4px 8px;">Median</th>
        <th style="padding:4px 8px;">Ø ohne Top-3</th>
        <th style="padding:4px 8px;">Totalverlust-Quote (≤ −95 %)</th>
        <th style="padding:4px 8px;">90%-KI des Ø (über Haltefenster)</th>
      </tr>
      {rows}
    </table>
    {delta_html}
    """


def _fmt_signed_pct(x) -> str:
    return f"{x:+.1%}" if isinstance(x, (int, float)) else "–"


def build_challenger_html() -> str:
    """
    🧪 Challenger (Walk-forward): rendert evaluate_all() aus modules/challenger.py.
    Rein informativ — ändert nie config.yaml/challengers.yaml. Darf den Report
    unter keinen Umständen zum Absturz bringen (try/except).
    """
    try:
        from modules.challenger import evaluate_all
        results = evaluate_all()
        if not results:
            return ""

        rows_html = ""
        for r in results:
            verdict = r.get("verdict", "running")
            color = {
                "promote_recommended": "#16a34a",
                "reject": "#dc2626",
                "expired": "#dc2626",
                "queued": "#888",
            }.get(verdict, "#d97706")
            ci = ""
            if r.get("ci_lower") is not None and r.get("ci_upper") is not None:
                ci = f"[{_fmt_signed_pct(r['ci_lower'])}, {_fmt_signed_pct(r['ci_upper'])}]"
            rows_html += (
                f"<li><b>{r['id']}</b> — {r.get('hypothesis', '').strip()}<br>"
                f"n Baseline/Challenger: {r.get('n_baseline')}/{r.get('n_challenger')} · "
                f"Ø Baseline {_fmt_signed_pct(r.get('mean_baseline'))} vs. "
                f"Ø Challenger {_fmt_signed_pct(r.get('mean_challenger'))} · "
                f"CI(Diff) {ci} · "
                f"Verdikt: <b style='color:{color}'>{verdict}</b></li>"
            )
        return (
            "<h3>🧪 Challenger (Walk-forward)</h3>"
            "<p style='font-size:0.85em;color:#888'>Rein informativ — Promotion erfolgt "
            "ausschließlich durch einen von Menschen gemergten PR auf config.yaml.</p>"
            "<p style='font-size:0.85em;color:#888'>Hinweis: opt_ret-Metriken sind synthetisch "
            "(Black-Scholes, konstante IV, kein IV-Crush, fixer Spread) — kein echtes Options-P&amp;L.</p>"
            f"<ul>{rows_html}</ul>"
        )
    except Exception as e:
        log.debug(f"build_challenger_html Fehler (ignoriert): {e}")
        return ""


def _fmt_signed_pct2(x) -> str:
    return f"{x:+.1%}" if isinstance(x, (int, float)) else "–"


def _external_row_html(label: str, stats: dict) -> str:
    tiny = "" if not stats or stats.get("effective_n") is None or stats["effective_n"] >= 10 else \
        " <span style='color:#dc2626;'>(zu kleine Stichprobe)</span>"
    if not stats or not stats.get("n"):
        return (
            f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{label}</td>"
            f"<td colspan='7' style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>"
            f"<i>keine Daten</i></td></tr>"
        )
    ci = ""
    if stats.get("ci_lower") is not None and stats.get("ci_upper") is not None:
        ci = f"[{_fmt_signed_pct2(stats['ci_lower'])}, {_fmt_signed_pct2(stats['ci_upper'])}]"
    win = f"{stats['win_rate']:.0%}" if stats.get("win_rate") is not None else "–"
    return (
        f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{label}{tiny}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{stats.get('n', 0)}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{stats.get('independent_date_count', 0)}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{stats.get('effective_n', 0):.1f}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{win}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{_fmt_signed_pct2(stats.get('mean'))}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{_fmt_signed_pct2(stats.get('median'))}</td>"
        f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{ci}</td></tr>"
    )


def _external_bucket_table_html(title: str, group: dict) -> str:
    if not group:
        return f"<p style='font-size:0.85em;color:#94a3b8;'><i>{title}: keine Daten.</i></p>"
    rows = "".join(_external_row_html(str(k), v) for k, v in sorted(group.items()))
    return f"""
    <p style="font-size:0.90em;font-weight:bold;margin-top:1em;">{title}</p>
    <table style="border-collapse:collapse;font-size:12px;width:100%;">
      <tr style="text-align:left;color:#64748b;">
        <th style="padding:4px 8px;">Bucket</th><th style="padding:4px 8px;">n</th>
        <th style="padding:4px 8px;">Unabh. Tage</th><th style="padding:4px 8px;">Eff. n</th>
        <th style="padding:4px 8px;">Win-Rate</th><th style="padding:4px 8px;">Ø</th>
        <th style="padding:4px 8px;">Median</th><th style="padding:4px 8px;">90%-KI</th>
      </tr>
      {rows}
    </table>"""


def build_external_context_html() -> str:
    """
    🌍 Externer Kontext (SHADOW): mature Outcome-Tabellen je Freight-/Maritime-/
    Relations-Bucket, Sektor, Katalysator-Typ, Richtung; Daten-Readiness je
    Quelle, Source-Health, Feature-Coverage, promotete Features, Hypothesen-
    Vorschläge (modules/external/research.py). Rein informativ, Shadow-Mode.
    Guarded: darf den Report NIE zum Absturz bringen.
    """
    try:
        from modules.config import cfg
        mode = str(getattr(getattr(cfg, "external_context", None), "mode", "shadow"))
    except Exception:
        mode = "shadow"

    header = (
        "<h3>🌍 Externer Kontext (SHADOW)</h3>"
        if mode == "shadow" else
        f"<h3>🌍 Externer Kontext (mode={mode})</h3>"
    )
    shadow_notice = (
        "<p style='font-size:0.85em;color:#92400e;background:#fffbeb;padding:8px 10px;"
        "border-radius:4px;'><b>SHADOW — NICHT in der Produktionsentscheidung verwendet.</b> "
        "Alle Zahlen unten sind reine Observability über den externen Real-Economy-Kontext "
        "(Road/Maritime Freight, Weather); score_delta ist auf 0 begrenzt.</p>"
        if mode == "shadow" else ""
    )

    try:
        from modules.challenger import load_ledger_rows
        from modules.external.research import analyze_external_buckets, generate_hypothesis_proposals
        rows = load_ledger_rows()
    except Exception as e:
        log.debug(f"build_external_context_html: Ledger/Research nicht ladbar (ignoriert): {e}")
        return header + shadow_notice + "<p><i>Keine externen Daten.</i></p>"

    # ── Mature Outcome-Tabellen ────────────────────────────────────────────
    tables_html = ""
    try:
        buckets = analyze_external_buckets(rows)
        tables_html += _external_bucket_table_html("Freight-Zustand (global_freight_state)",
                                                     buckets.get("freight_state", {}))
        tables_html += _external_bucket_table_html("Maritime-Zustand (global_maritime_state)",
                                                     buckets.get("maritime_state", {}))
        tables_html += _external_bucket_table_html("Externe Relation (LLM)",
                                                     buckets.get("external_relation", {}))
        tables_html += _external_bucket_table_html("Sektor/Industrie", buckets.get("sector", {}))
        tables_html += _external_bucket_table_html("Katalysator-Typ", buckets.get("catalyst", {}))
        tables_html += _external_bucket_table_html("Richtung", buckets.get("direction", {}))
    except Exception as e:
        log.debug(f"build_external_context_html: analyze_external_buckets Fehler (ignoriert): {e}")
        tables_html = "<p><i>Mature-Outcome-Tabellen nicht berechenbar.</i></p>"

    # ── Daten-Readiness + Source-Health ────────────────────────────────────
    readiness_html = "<p><i>Readiness nicht berechenbar.</i></p>"
    health_html = "<p><i>Source-Health nicht verfügbar.</i></p>"
    try:
        from modules.external.registry import SourceRegistry, compute_readiness
        registry = SourceRegistry()
        health = registry.load_health()
        rows_r = ""
        rows_h = ""
        for source_id, src_cfg in sorted(registry.sources.items()):
            h = health.get(source_id, {})
            readiness = compute_readiness(src_cfg, h)
            rows_r += (
                f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{source_id}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{readiness.value}</td></tr>"
            )
            rows_h += (
                f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{source_id}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{h.get('status', '–')}</td>"
                f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{h.get('staleness', 'UNKNOWN')}</td></tr>"
            )
        if rows_r:
            readiness_html = (
                "<table style='border-collapse:collapse;font-size:12px;width:100%;'>"
                "<tr style='text-align:left;color:#64748b;'><th style='padding:4px 8px;'>Quelle</th>"
                "<th style='padding:4px 8px;'>Readiness</th></tr>" + rows_r + "</table>"
            )
        if rows_h:
            health_html = (
                "<table style='border-collapse:collapse;font-size:12px;width:100%;'>"
                "<tr style='text-align:left;color:#64748b;'><th style='padding:4px 8px;'>Quelle</th>"
                "<th style='padding:4px 8px;'>Status</th><th style='padding:4px 8px;'>Frische</th></tr>"
                + rows_h + "</table>"
            )
    except Exception as e:
        log.debug(f"build_external_context_html: Readiness/Health Fehler (ignoriert): {e}")

    # ── Feature-Coverage (% non-null je Primitiv, letzte 30 Ledger-Tage) ────
    coverage_html = "<p><i>Feature-Coverage nicht berechenbar.</i></p>"
    try:
        recent_cutoff = (date.today() - timedelta(days=30)).isoformat()
        recent = [r for r in rows if str(r.get("date", "")) >= recent_cutoff and isinstance(r.get("external"), dict)]
        if recent:
            prim_keys: set[str] = set()
            for r in recent:
                prim_keys.update(((r.get("external") or {}).get("primitives") or {}).keys())
            cov_rows = ""
            for prim in sorted(prim_keys):
                present = sum(
                    1 for r in recent
                    if ((r.get("external") or {}).get("primitives") or {}).get(prim) is not None
                )
                pct = present / len(recent)
                cov_rows += (
                    f"<tr><td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{prim}</td>"
                    f"<td style='padding:4px 8px;border-bottom:1px solid #e2e8f0;'>{pct:.0%}</td></tr>"
                )
            coverage_html = (
                "<table style='border-collapse:collapse;font-size:12px;width:100%;'>"
                "<tr style='text-align:left;color:#64748b;'><th style='padding:4px 8px;'>Primitiv</th>"
                "<th style='padding:4px 8px;'>% non-null (30d)</th></tr>" + cov_rows + "</table>"
            )
        else:
            coverage_html = "<p><i>Keine Ledger-Zeilen mit externem Kontext in den letzten 30 Tagen.</i></p>"
    except Exception as e:
        log.debug(f"build_external_context_html: Feature-Coverage Fehler (ignoriert): {e}")

    # ── Promotete externe Features ──────────────────────────────────────────
    promoted_html = "<p><i>Keine promoteten externen Features.</i></p>"
    try:
        from modules.config import cfg as _cfg
        promoted = list(getattr(getattr(getattr(_cfg, "external_context", None), "learning", None),
                                 "promoted_external_features", []) or [])
        if promoted:
            promoted_html = "<ul>" + "".join(f"<li>{p}</li>" for p in promoted) + "</ul>"
    except Exception as e:
        log.debug(f"build_external_context_html: promoted features Fehler (ignoriert): {e}")

    # ── Hypothesen-Vorschläge (research.py) ─────────────────────────────────
    hypotheses_html = "<p><i>Keine neuen Hypothesen-Vorschläge diesen Monat.</i></p>"
    try:
        proposals = generate_hypothesis_proposals(rows)
        if proposals:
            items = ""
            for p in proposals:
                items += (
                    f"<p style='font-size:0.80em;margin-top:0.8em;'><b>{p['hypothesis_id']}</b> "
                    f"({p['group_label']}):</p>"
                    f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;"
                    f"overflow-x:auto;font-size:0.75em;'>{p['snippet']}</pre>"
                )
            hypotheses_html = (
                "<p style='font-size:0.85em;color:#555;'>Retrospektiv/in-sample — NICHT "
                "promotion-fähig. Snippets können in challengers.yaml eingefügt werden "
                "(start_date liegt bereits in der Zukunft).</p>" + items
            )
    except Exception as e:
        log.debug(f"build_external_context_html: Hypothesen-Generator Fehler (ignoriert): {e}")

    return f"""
    {header}
    {shadow_notice}
    <p style="font-size:0.90em;font-weight:bold;">Mature Outcomes je externem Bucket</p>
    {tables_html}
    <p style="font-size:0.90em;font-weight:bold;margin-top:1em;">Daten-Readiness je Quelle</p>
    {readiness_html}
    <p style="font-size:0.90em;font-weight:bold;margin-top:1em;">Source-Health</p>
    {health_html}
    <p style="font-size:0.90em;font-weight:bold;margin-top:1em;">Feature-Coverage (letzte 30 Tage)</p>
    {coverage_html}
    <p style="font-size:0.90em;font-weight:bold;margin-top:1em;">Promotete externe Features (Produktion)</p>
    {promoted_html}
    <p style="font-size:0.90em;font-weight:bold;margin-top:1em;">🔬 Externe Hypothesen-Vorschläge</p>
    {hypotheses_html}
    """


def _safe_external_context_html() -> str:
    """Wrapper um build_external_context_html() — darf den Report NIE zum
    Absturz bringen (Anforderung: Guarded try/except)."""
    try:
        return build_external_context_html()
    except Exception as e:
        log.debug(f"_safe_external_context_html Fehler (ignoriert): {e}")
        return "<h3>🌍 Externer Kontext (SHADOW)</h3><p><i>keine externen Daten</i></p>"


NV = "nicht verfügbar"


def _usd(x, digits: int = 2) -> str:
    return NV if x is None else f"${x:,.{digits}f}"


def build_cost_html(report_month: str, summary: dict | None = None) -> str:
    """Abschnitt KOSTEN & EFFIZIENZ aus der Kosten-Telemetrie (modules/cost_telemetry.py).
    Nicht messbare Werte -> 'nicht verfügbar'; fehlende Telemetrie -> 'unvollständig'."""
    from modules import cost_telemetry as ct
    s = summary if summary is not None else ct.month_summary(report_month)
    status = "" if s.get("complete") else " <span style='color:#b45309'>(unvollständig)</span>"
    fam = s.get("by_family") or {}
    scope = s.get("by_scope") or {}
    has = bool(s.get("telemetry_calls"))

    def fam_cost(k):
        return _usd(fam[k]["cost_usd"]) if has and k in fam else (_usd(0.0) if has else NV)

    other_llm = (sum(v["cost_usd"] for k, v in fam.items() if k not in ("SONNET", "HAIKU")) if has else None)
    delta = NV
    if s.get("delta_usd") is not None:
        pct = f" ({s['delta_pct']:+.0%})" if s.get("delta_pct") is not None else ""
        delta = f"{s['delta_usd']:+,.2f} ${pct}"
    api = s.get("api") or {}
    api_txt = (", ".join(f"{k}: {v['requests']} Req." + (f" ({v['rate_limited']}× 429)" if v.get("rate_limited") else "")
                         for k, v in api.items()) or NV)
    ac = s.get("analysis_cache") or {}
    if ac.get("hits"):
        cache_txt = f"{ac['hits']}/{ac['lookups']} Treffer, Ersparnis {_usd(ac['saved_usd'])}"
    elif ac.get("lookups"):
        cache_txt = (f"Beobachtungsmodus: {ac.get('would_hit', 0)}/{ac['lookups']} identische Wiederholungen, "
                     f"potenziell {_usd(ac.get('potential_saved_usd'))}")
    else:
        cache_txt = NV
    val = s.get("analysis_cache_validation") or {}
    if val.get("checks"):
        cache_txt += (f"; A/B-Prüfung {val['checks']} Vergleiche, Richtung gleich "
                      f"{val['direction_agreement']:.0%} → {val['decision']}")
    pc = s.get("prompt_cache_savings_usd")
    prompt_cache_txt = (_usd(pc) if has and s.get("cache_read_input_tokens") else NV)

    routing_txt, prefilter_txt = NV, NV
    try:
        from modules import model_routing as _mr
        from modules.config import cfg as _cfg
        d = _mr.decision("deep_analysis", _cfg.models.deep_analysis,
                         prompt_version=_mr.current_version("route_pair"))
        ab = d.get("ab") or {}
        if d.get("status") != "OFF":
            routing_txt = (f"{d.get('status')}: aktiv {d.get('active')}; A/B {ab.get('pairs', 0)} Paare/"
                           f"{ab.get('days', 0)} Tage, Gate-Übereinstimmung {ab.get('gate_agreement') or NV}, "
                           f"Recall {NV if ab.get('pass_recall') is None else ab['pass_recall']}")
        pf = _mr.prefilter_decision(prompt_version=_mr.current_version("prefilter_pair"))
        if pf.get("status") != "OFF":
            prefilter_txt = (f"{pf.get('status')} ({'aktiv' if pf.get('active') else 'inaktiv'}); "
                             f"{pf.get('pairs', 0)} Paare, Verlust bestandener Kandidaten {NV if pf.get('pass_loss') is None else pf['pass_loss']}")
    except Exception as e:  # noqa: BLE001
        log.debug(f"Routing-Status nicht verfügbar: {e}")

    def scope_cost(k):
        return _usd(scope[k]["cost_usd"]) if has and k in scope else (_usd(0.0) if has else NV)

    def num(x):
        return NV if not has or x is None else f"{x:,}"

    rows = [
        ("Gesamtkosten LLM [MEASURED: API-usage × Listenpreis]", _usd(s.get("cost_usd"))),
        (f"Vormonat ({s.get('prev_month')})", _usd(s.get("prev_cost_usd"))),
        ("Veränderung", delta),
        ("Sonnet / Haiku / andere LLMs", f"{fam_cost('SONNET')} / {fam_cost('HAIKU')} / {_usd(other_llm)}"),
        ("Bezahlte APIs [ESTIMATED: Planpreis anteilig]", _usd(s.get("paid_api_cost_usd")) if s.get("api") else NV),
        ("API-Nutzung [MEASURED: Requests]", api_txt),
        ("Sonnet-Calls / Haiku-Calls", f"{num(s.get('sonnet_calls'))} / {num(s.get('haiku_calls'))}"),
        ("Fehlgeschlagene Calls", num(s.get("failed_calls"))),
        ("Input- / Output-Tokens [MEASURED]", f"{num(s.get('input_tokens'))} / {num(s.get('output_tokens'))}"),
        ("Prompt-Cache-Ersparnis [ESTIMATED vs. ohne Cache]", prompt_cache_txt),
        ("Analyse-Cache [ESTIMATED Ersparnis]", cache_txt),
        ("Batch-API-Anteil (-50 %)", (f"{s.get('batch_calls', 0)}/{s.get('telemetry_calls')} Calls" if has else NV)),
        ("Modell-Routing Deep Analysis", routing_txt),
        ("Bearish-Vorfilter", prefilter_txt),
        ("Kosten je Scan-Lauf", _usd(s.get("cost_per_scan"), 3)),
        ("Kosten je Sonnet-Analyse", _usd(s.get("cost_per_sonnet_analysis"), 4)),
        ("Kosten je finalem Trade", _usd(s.get("cost_per_final_trade"))),
        ("Produktion / Shadow / Research",
         f"{scope_cost('production')} / {scope_cost('shadow')} / {scope_cost('research')}"),
    ]
    trs = "".join(f"<tr><td style='padding:2px 8px'>{k}</td><td style='padding:2px 8px'><b>{v}</b></td></tr>"
                  for k, v in rows)
    note = ""
    if s.get("days_without_telemetry"):
        note += f"<p><i>{s['days_without_telemetry']} Scan-Tag(e) mit LLM-Stufen ohne Telemetrie.</i></p>"
    if s.get("unknown_cost_calls"):
        note += f"<p><i>{s['unknown_cost_calls']} Calls ohne berechenbare Kosten (Preis/usage fehlt).</i></p>"
    return (f"<h3>💰 KOSTEN &amp; EFFIZIENZ {report_month}{status}</h3>"
            f"<table style='font-size:0.9em;border-collapse:collapse'>{trs}</table>{note}"
            f"<p style='font-size:0.9em'>{ct.explain(s)}</p>")


def universe_html(report_month: str) -> str:
    """Kompakt: UNIVERSE_V1/V2-Größen und V2-Shadow-Kennzahlen je Bucket (Details im Montagsbericht §9)."""
    try:
        from modules import universe_v2 as uv, universe_v2_ledger as v2l
        from modules.final_mc_ledger import read_rows
        snap = uv.latest_snapshot()
        sm = (snap or {}).get("summary") or {}
        rows = [r for r in read_rows(v2l.DIR) if str(r.get("date", ""))[:7] == report_month]
        bs = v2l.bucket_summary(rows, v2l.read_outcomes())
    except Exception as e:  # noqa: BLE001 – Report darf nie abbrechen
        log.warning(f"Universe-Abschnitt nicht berechenbar: {e}")
        return ""
    def pct(x, sign=False):
        return NV if x is None else (f"{x:+.1%}" if sign else f"{x:.1%}")
    lines = "".join(f"<li>{b}: {v['signals']} Signale, Netto-Exp. {pct(v['net_expectancy'], True)}, "
                    f"Ø Kosten {pct(v['avg_execution_cost'])}</li>" for b, v in bs.items())
    return (f"<h3>🌐 UNIVERSE V1 / V2 {report_month}</h3><p>V2 (Shadow): optionierbar {sm.get('n_optionable', NV)}, "
            f"research {sm.get('n_research', NV)}, tradeable {sm.get('n_tradeable', NV)} (Stand {(snap or {}).get('as_of', NV)})."
            f"</p><ul>{lines or '<li>noch keine V2-Signale</li>'}</ul>")


def commodity_html(report_month: str) -> str:
    """Commodity Intelligence (RESEARCH): eine Statuszeile + Kernzahlen (Details im Montagsbericht §10)."""
    try:
        from modules import commodity_intelligence as cmd
        s = cmd.report_summary()
        rows = cmd.report_rows(s)
    except Exception as e:  # noqa: BLE001 – Report darf nie abbrechen
        log.warning(f"Commodity-Abschnitt nicht berechenbar: {e}")
        return ""
    import html as _h
    lis = "".join(f"<li>{_h.escape(k)}: {_h.escape(str(v))}</li>" for k, v in rows[1:])
    return (f"<h3>🛢️ COMMODITY INTELLIGENCE {report_month}</h3><p><b>{_h.escape(str(rows[0][1]))}</b></p>"
            f"<ul style='font-size:0.9em'>{lis}</ul>")


def _safe_cost_html(report_month: str) -> str:
    """Darf den Report NIE zum Absturz bringen."""
    try:
        return build_cost_html(report_month)
    except Exception as e:  # noqa: BLE001
        log.warning(f"Kostenabschnitt nicht berechenbar (ignoriert): {e}")
        return (f"<h3>💰 KOSTEN &amp; EFFIZIENZ {report_month} (unvollständig)</h3>"
                f"<p><i>Kosten-Telemetrie {NV}.</i></p>")


def _p(x, signed: bool = True) -> str:
    if not isinstance(x, (int, float)):
        return "–"
    return f"{x:+.1%}" if signed else f"{x:.0%}"


def _stats_row(label: str, s: dict) -> str:
    td = "padding:4px 8px;border-bottom:1px solid #e2e8f0;"
    pf = f"{s['profit_factor']:.2f}" if isinstance(s.get("profit_factor"), (int, float)) else "–"
    ci = f"{s['ci90'][0]:+.1%} bis {s['ci90'][1]:+.1%}" if s.get("ci90") else "–"
    label = escape(str(label))
    return (f"<tr><td style='{td}'><b>{label}</b></td><td style='{td}'>{s['n_trades']}</td>"
            f"<td style='{td}'>{s['n_evaluable']}</td><td style='{td}'>{_p(s['win_rate'], False)}</td>"
            f"<td style='{td}'>{_p(s['mean'])}</td><td style='{td}'>{_p(s['median'])}</td><td style='{td}'>{pf}</td>"
            f"<td style='{td}'>{_p(s['total_loss_rate'], False)}</td><td style='{td}'>{_p(s['mean_ex_top3'])}</td>"
            f"<td style='{td}'>{s['independent_windows']}</td><td style='{td}'>{ci}</td>"
            f"<td style='{td}'>{s['status']}{'' if not s.get('reason') else ' (' + escape(s['reason']) + ')'}</td></tr>")


_STATS_HEAD = ("<tr style='text-align:left;color:#64748b;'><th>Gruppe</th><th>Trades</th><th>auswertbar</th>"
               "<th>Win-Rate</th><th>Ø</th><th>Median</th><th>PF</th><th>Totalverlust (≤ −95 %)</th>"
               "<th>Ø ohne Top-3</th><th>unabh. Haltefenster</th><th>90%-KI</th><th>Status</th></tr>")


def build_evidence_html(ev: dict, report_month: str) -> str:
    """A. Verifizierte Performance (nur RELIABLE) + Kennzahlen je Evidenzklasse."""
    rel, mon = ev["reliable"], ev["reliable_month"]
    verdict = ("<p style='color:#92400e;'><b>INSUFFICIENT_EVIDENCE</b> – die verifizierte Datenbasis reicht für keine "
               "statistisch belastbare Aussage (mind. "
               f"{MIN_EVIDENCE_N} RELIABLE-Outcomes und {MIN_INDEPENDENT_WINDOWS} unabhängige Haltefenster). "
               "Die Werte sind rein beschreibend.</p>" if rel["status"] == INSUFFICIENT else
               "<p>Verifizierte Datenbasis ausreichend für eine beschreibende Statistik mit Haltefenster-KI "
               "(kein Alpha-Nachweis ohne prospektive Bestätigung).</p>")
    groups = ""
    for title, key in (("Optionsstruktur", "by_strategy"), ("Sektor", "by_sector"), ("Marktregime (VIX)", "by_regime")):
        rows = "".join(_stats_row(k, v) for k, v in ev[key].items()) or \
            "<tr><td colspan='12'><i>keine RELIABLE-Trades</i></td></tr>"
        groups += f"<p><b>RELIABLE nach {title}:</b></p><table style='border-collapse:collapse;font-size:12px'>" \
                  f"{_STATS_HEAD}{rows}</table>"
    classes = "".join(_stats_row(c, ev["classes"][c]) for c in EVIDENCE_CLASSES)
    return f"""
    <h3>✅ A. Verifizierte Performance (nur RELIABLE)</h3>
    <p style="font-size:0.9em;color:#555;">RELIABLE = realisierbarer Preis aus beobachteten Quotes oder vertraglichem
    Settlement (modules/outcomes.py). Nur diese Ergebnisse gelten als Options-Performance-Evidenz und dürfen ins
    autonome Lernen. Totalverlust = {ev['total_loss_definition']}.</p>
    <table style="border-collapse:collapse;font-size:12px">{_STATS_HEAD}
      {_stats_row("Alle RELIABLE", rel)}{_stats_row(f"RELIABLE geschlossen im {report_month}", mon)}
    </table>
    {verdict}
    {groups}
    <p><b>Alle Kennzahlen je Evidenzklasse</b> (Klassen: {ev['class_counts']}):</p>
    <table style="border-collapse:collapse;font-size:12px">{_STATS_HEAD}{classes}</table>"""


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def research_progress(root: Path | None = None, alpha_path: Path | None = None) -> dict:
    """C. Lernfortschritt aus vorhandenen Forschungsdateien (keine eigene Logik); fehlend bleibt fehlend (None)."""
    root = root or OPTION_LEARNER_DIR
    out: dict = {"horizons": {}, "alpha_validation": None, "gaps": [], "missing": []}
    for h in (20, 5):
        st = _read_json(root / f"status_{h}d.json")
        if st is None:
            out["missing"].append(f"option_learner/status_{h}d.json")
            out["horizons"][h] = None
            continue
        fw = st.get("forward") or {}
        out["horizons"][h] = {
            "verified_labels": st.get("verified_labels"), "signal_days": st.get("label_dates"),
            "model_instances": (st.get("model_instances") or {}).get("count"),
            "valid_forward_decisions": fw.get("valid_decisions"), "valid_forward_predictions": fw.get("valid_predictions"),
            "forward_due_evaluated": fw.get("n"), "forward_verified_outcomes": fw.get("n_selected"),
            "forward_verdict": fw.get("verdict"), "vs_reference": (fw.get("vs_reference") or {}).get("verdict"),
            "learning_state": (st.get("learning_status") or {}).get("state"),
            "technically_missing": (st.get("label_pipeline") or {}).get("technically_missing"),
            "as_of": st.get("as_of")}
    val = _read_json(alpha_path or ALPHA_VALIDATION_PATH)
    if val is None:
        out["missing"].append("autonomous_alpha/option_validation.json")
    else:
        out["alpha_validation"] = {"status": val.get("status"), "n_plans": val.get("n_plans"),
                                   "classifications": val.get("classification_counts")}
    imp = _read_json(root / "improvement.json")
    if imp is None:
        out["missing"].append("option_learner/improvement.json")
    else:
        out["gaps"] = [g.get("finding") for g in (imp.get("gaps") or [])[:3] if g.get("finding")]
    return out


def build_research_html(rp: dict) -> str:
    def v(x):
        if x is None:
            return "<i>fehlt</i>"
        return escape(", ".join(f"{k} {n}" for k, n in x.items()) or "keine") if isinstance(x, dict) else escape(str(x))
    rows = ""
    labels = (("Verifizierte Options-Labels", "verified_labels"), ("Unabhängige Signaltage", "signal_days"),
              ("Registrierte Modellinstanzen", "model_instances"),
              ("Gültige Forward-Prognosen (Entscheidungen / Prognosen)", None),
              ("Fällige Forward-Entscheidungen bewertet / mit verifiziertem Ergebnis", None),
              ("Forward-Urteil", "forward_verdict"), ("Vergleich zur eingefrorenen Referenz", "vs_reference"),
              ("Lernstatus", "learning_state"), ("Technisch fehlende Labels", "technically_missing"))
    for label, key in labels:
        cells = ""
        for h in (20, 5):
            hz = rp["horizons"].get(h)
            if hz is None:
                cells += "<td><i>fehlt</i></td>"
            elif key is None and label.startswith("Gültige"):
                cells += f"<td>{v(hz['valid_forward_decisions'])} / {v(hz['valid_forward_predictions'])}</td>"
            elif key is None:
                cells += f"<td>{v(hz['forward_due_evaluated'])} / {v(hz['forward_verified_outcomes'])}</td>"
            else:
                cells += f"<td>{v(hz[key])}</td>"
        rows += f"<tr><td>{label}</td>{cells}</tr>"
    av = rp.get("alpha_validation")
    alpha = ("<i>fehlt</i> (noch kein Lauf der prospektiven Alpha-Validierung)" if av is None else
             escape(f"{av.get('status')} – {av.get('n_plans')} Prüfpläne, Klassen {av.get('classifications') or {}}"))
    gaps = "".join(f"<li>{escape(str(g))}</li>" for g in rp.get("gaps") or []) or \
        "<li><i>keine gemessene Lücke oder Datei fehlt</i></li>"
    missing = (f"<p style='font-size:0.85em;color:#92400e;'>Fehlende Forschungsdateien: {', '.join(rp['missing'])} – "
               "Werte als fehlend gekennzeichnet, nie als 0.</p>" if rp.get("missing") else "")
    return f"""
    <h3>🔬 C. Forschungs- und Lernfortschritt</h3>
    <table style="border-collapse:collapse;font-size:12px"><tr style="color:#64748b;"><th></th><th>20 T</th><th>5 T</th></tr>
    {rows}</table>
    <p><b>Prospektive Alpha-Validierung (Alpha→Options):</b> {alpha}</p>
    <p><b>Wichtigste Datenlücken:</b></p><ul>{gaps}</ul>{missing}"""


def build_loss_causes_html(path: Path | None = None) -> str:
    """Verlustursachen aus dem bestehenden Failure Analyzer (trade_memory), getrennt nach Evidenzstärke."""
    fa = _read_json(path or FAILURE_ANALYSIS_PATH)
    if fa is None:
        return "<h3>🧩 Verlustursachen</h3><p><i>failure_analysis.json fehlt – keine Aussage.</i></p>"
    rel, unrel = fa.get("reliable") or {}, fa.get("unreliable") or {}
    n = rel.get("n") or 0
    prim = ", ".join(f"{k} {v.get('n')}" for k, v in (rel.get("primary") or {}).items()) or "keine"
    labels = ", ".join(f"{k} {v.get('n')}/{n}" for k, v in (rel.get("labels") or {}).items()) or "keine"
    supported = ("keine – n=" + str(n) + f" verifizierte Verluste (unter {MIN_EVIDENCE_N}), keine Signifikanzaussage"
                 if n < MIN_EVIDENCE_N else "nur mit eigenem, vorab registriertem Test")
    return f"""
    <h3>🧩 Verlustursachen (Failure Analyzer, deterministische Regeln)</h3>
    <ul>
      <li><b>Beobachtete Tatsache</b> ({n} RELIABLE-Verluste): primäre Ursache {escape(prim)}; Merkmale
          {escape(labels)}.</li>
      <li><b>Plausible Ursache</b>: hohe Einstiegskosten (Spread) und Zeitwertverlust bei Spreads – Regel-Labels, kein
          Kausalnachweis.</li>
      <li><b>Statistisch unterstützt</b>: {supported}.</li>
      <li><b>Ungeprüft / Datenlücke</b>: {unrel.get('n', 0)} Verluste ohne verifizierte Preisquelle werden nicht nach
          Ursachen klassifiziert; Veränderungen der impliziten Volatilität sind je Trade nicht erfasst.</li>
    </ul>"""


def build_html(report_month: str, cur: dict | None, prev: dict | None,
               total: dict | None, funnel: dict, closed: list[dict] | None = None,
               spy: float | None = None, shadow: dict | None = None,
               tuning: list[dict] | None = None,
               slot: list[dict] | None = None, *, evidence: dict | None = None,
               research: dict | None = None, history: dict | None = None) -> str:
    def stat_block(label: str, s: dict | None) -> str:
        if s is None:
            return f"<p><b>{label}:</b> keine geschlossenen Trades</p>"
        return (
            f"<p><b>{label}:</b> Win-Rate <b>{_fmt_pct(s['win_rate'])}</b> "
            f"({s['wins']}/{s['n']}) · Ø {s['mean']:+.1%} · "
            f"Median {s['median']:+.1%} · Totalverluste (≤ −95 %) {s['total_losses']}</p>"
        )

    # Rolling stats and month-over-month analysis
    rolling_html = ""
    if closed:
        rolling_html = build_rolling_stats_html(closed, prev, cur)

    # SPY nur als Marktkontext (Audit 2026-11-01): Options-Renditen beziehen sich auf das eingesetzte Prämienkapital,
    # sind gehebelt, haben andere Haltedauern und Ausführungskosten -> keine "Edge"-Differenz, kein Alpha-Nachweis.
    bench_html = ""
    if spy is not None:
        bench_html = (f"<p><b>Marktkontext:</b> SPY {spy:+.1%} im {report_month}. <i>Kein Vergleichsmaßstab für die "
                      f"Options-Renditen (Risikokapital, Hebel, Haltedauer und Ausführung nicht vergleichbar) – "
                      f"daraus folgt kein Options-Alpha.</i></p>")

    # Schatten-Trades: filtern die Gates Gewinner weg?
    shadow_html = ""
    if shadow:
        rows = []
        for r, g in (shadow.get("by_reason") or {"alle": shadow}).items():
            flag = ("⚠️ verworfene Trades besser als echte (n klein, ohne TP/SL – prüfen)"
                    if cur and g["n"] >= 10 and g["mean"] > cur["mean"]
                    and g["win_rate"] > cur["win_rate"] else
                    ("n<10 – keine Aussage" if g["n"] < 10 else "kein Hinweis auf weggefilterte Gewinner"))
            rows.append(f"{r}: Win-Rate {_fmt_pct(g['win_rate'])} ({g['wins']}/{g['n']}) · Ø {g['mean']:+.1%} — {flag}")
        shadow_html = ("<p><b>Schatten-Trades</b> (von Gates verworfen, nur getrackt; je Gate getrennt, "
                       "Bewertung zum Halte-Ende ohne TP/SL):<br>" + "<br>".join(rows) + "</p>")

    funnel_html = ""
    if funnel["days"]:
        funnel_html = (
            f"<h3>Signal-Funnel {report_month}</h3>"
            f"<p>{funnel['zero_days']} von {funnel['days']} Scan-Tagen ohne Trade-Vorschlag.</p>"
        )
        if funnel["top_rejects"]:
            items = "".join(f"<li>{r}: {c}×</li>" for r, c in funnel["top_rejects"])
            funnel_html += f"<p><b>Häufigste Reject-Gründe:</b></p><ul>{items}</ul>"
        if funnel["top_stops"]:
            items = "".join(f"<li>{r} ({c}×)</li>" for r, c in funnel["top_stops"])
            funnel_html += f"<p><b>Häufigste Stop-Gründe (0-Trade-Tage):</b></p><ul>{items}</ul>"

    evidence_html = build_evidence_html(evidence, report_month) if evidence else ""
    research_html = build_research_html(research) if research else ""
    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:640px">
      <h2>📊 Monats-Report — {REPO_NAME}</h2>
      <p>Berichtsmonat: <b>{report_month}</b> (Trades mit Close-Datum in diesem Monat)</p>
      {evidence_html}
      <hr>
      <h3>🧪 B. Historische Performance – explorativ</h3>
      <p style="background:#fef3c7;border:1px solid #f59e0b;padding:6px;"><b>{EXPLORATIVE_BANNER}</b></p>
      {stat_block(f"Monat {report_month}", cur)}
      {stat_block("Vormonat", prev)}
      {rolling_html}
      {bench_html}
      {shadow_html}
      {stat_block("Gesamt (alle closed Trades, alle Evidenzklassen)", total)}
      <hr>
      {research_html}
      {build_loss_causes_html() if evidence else ""}
      <hr>
      {build_tuning_html(tuning or [], history)}
      {build_slot_html(slot or [], evidence)}
      {build_challenger_html()}
      {_safe_external_context_html()}
      {funnel_html}
      <hr>
      {_safe_cost_html(report_month)}
      {universe_html(report_month)}
      {commodity_html(report_month)}
      <hr>
      <p style="color:#888;font-size:0.85em">
        Automatisch generiert durch {REPO_NAME} · monthly_report.py<br>
        Hinweis: Outcomes basieren auf Trades, die nach
        {os.getenv("CLOSE_AFTER_DAYS", "45")} Tagen geschlossen wurden.
      </p>
    </body></html>
    """


def main() -> None:
    if not HISTORY_PATH.exists():
        log.error("history.json nicht gefunden.")
        sys.exit(1)
    history = json.loads(HISTORY_PATH.read_text())
    # einmal deduplizieren, damit alle Abschnitte dieselbe Basis haben
    # (vorher nur die Rolling-Stats -> widersprüchliche Zahlen in einer Mail)
    closed  = dedup_closed_trades(history.get("closed_trades", []))

    # Berichtsmonat = Vormonat (Report läuft am 1. des Folgemonats)
    today        = date.today()
    report_month = month_key(today.replace(day=1) - timedelta(days=1))
    prior_month  = prev_month(report_month)

    cur    = month_stats(closed, report_month)
    prev   = month_stats(closed, prior_month)
    total  = overall_stats(closed)
    funnel = funnel_summary(report_month)
    spy    = spy_return(report_month)
    shadow = shadow_stats(history, report_month)
    tuning = tuning_suggestions(history)
    slot   = slot_candidates(history)

    evidence = evidence_report(closed, report_month)
    try:
        research = research_progress()
    except Exception as e:  # noqa: BLE001 – der Report darf an Forschungsdateien nie scheitern
        log.warning(f"Forschungsfortschritt nicht lesbar: {e}")
        research = {"horizons": {20: None, 5: None}, "alpha_validation": None, "gaps": [],
                    "missing": [f"Forschungsdateien nicht lesbar ({type(e).__name__})"]}
    rel = evidence["reliable"]
    # Betreff nach Evidenzqualität (Audit 2026-11-01): nie eine explorative Win-Rate als Schlagzeile
    rel_txt = (f"verifiziert n={rel['n_evaluable']} ({rel['status']})" if rel["n_evaluable"] else
               "verifiziert: keine RELIABLE-Trades")
    subject = (f"📊 {REPO_NAME}: Monats-Report {report_month} — {rel_txt}; "
               f"explorativ n={evidence['explorative']['n_evaluable']}")

    if tuning:
        log.info(f"{len(tuning)} Hypothesen-Vorschlag/Vorschläge gefunden (retrospektiv)")
    html = build_html(report_month, cur, prev, total, funnel, closed, spy, shadow, tuning, slot,
                      evidence=evidence, research=research, history=history)
    log.info(f"Sende Monats-Report: {subject}")
    _send_smtp(subject, html)
    log.info("=== Monats-Report abgeschlossen ===")


if __name__ == "__main__":
    main()
