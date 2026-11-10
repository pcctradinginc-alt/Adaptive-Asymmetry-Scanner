"""
modules/engine_monitor.py – Engine-Überwachung (reine Observability)

Zweck:
  Der Scanner hat mehrere Paralleltests laufen (Schatten-Trades, Trailing-Sim),
  eine mehrstufige Gate-Pipeline und einen Lern-Loop (Feature-Bins/Outcomes;
  das Pearson-Gewichts-Update auf model_weights ist seit 2026-11-09 entfernt).
  Bricht irgendwo etwas (Datenquelle tot, Gate zu scharf, NaN im
  Lern-Loop), fällt das bisher nur auf, wenn jemand manuell in die JSONs
  schaut. Dieses Modul baut daraus einen Health-Report, der jeden Lauf
  automatisch mitprüft.

WICHTIG: Dieses Modul verändert KEINE Gates, Schwellen oder Pipeline-Logik.
Es liest nur bereits vorhandene Daten (history.json, Daily-Reports) und
meldet Auffälligkeiten. Reine Funktionen, kein Netzwerk, keine LLM-Calls.

Checks in build_health_report():
  a) DÜRRE           – Streak aufeinanderfolgender 0-Trade-Tage + Killer-Gate
                        (welche Funnel-Stufe hat den Kandidaten-Fluss gestoppt)
  b) PARALLELTEST     – Reife-Übersicht der Schatten-Trades / Trailing-Sim
     -REIFE             (keine Warnung, nur Metrics für die Kalibrierung)
  c) LERN-LOOP-SANITY – NaN-/Null-Gewichte, verwaiste (überfällige) Trades,
                        Feature-Korrelationen mit Outcome (nur Diagnose)
  d) DATA-HEALTH      – Läuft der Scanner noch? Kommen sinnvolle Daten rein?
"""

import json
import logging
import math
import re
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path

from modules.config import cfg

log = logging.getLogger(__name__)

# Funnel-Reihenfolge für die Killer-Diagnose (siehe pipeline.py stats-dict).
# Bewusst reduziert auf die "Haupt-Trichter"-Stufen — Zwischenstufen wie
# sector_ok/pre_mc/roi_precheck/final_selected/after_isf sind Diagnose-Detail,
# aber für die Frage "welches Gate hat den 0-Trade-Tag verursacht" reicht
# diese grobe Kette.
FUNNEL_ORDER = [
    "universe", "prescreened", "analyzed", "mismatch_ok",
    "quick_mc", "intraday_ok", "final_mc", "roi_ok", "trades",
]

_DAILY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _parse_date(value) -> date | None:
    """Parst 'YYYY-MM-DD...'-Strings robust. None bei allem Unerwarteten."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _get_close_after_days() -> int:
    try:
        return int(cfg.learning.close_after_days)
    except Exception:
        return 45


def _bin_to_num(feature: str, bin_label: str) -> float:
    """
    Konvertiert bin-Label zu numerischem Wert (identisch mit feedback.py).
    Wird für die Pearson-Korrelations-Berechnung der Features verwendet.
    """
    mapping = {
        "impact":    {"low": 0.0, "mid": 0.5, "high": 1.0},
        "mismatch":  {"weak": 0.0, "good": 0.5, "strong": 1.0},
        "eps_drift": {"noise": 0.0, "relevant": 0.5, "massive": 1.0},
    }
    return mapping.get(feature, {}).get(bin_label, 0.5)


def _load_recent_daily_reports(reports_dir: Path, today: date, limit: int = 10) -> list[tuple[date, dict]]:
    """
    Lädt die letzten `limit` VORHANDENEN Daily-JSONs (nach Dateiname sortiert,
    neueste zuerst), robust gegen fehlende/kaputte Dateien — beide werden
    einfach übersprungen und zählen nicht zu den "vorhandenen" Reports.
    Nur Dateien mit Datum <= today werden berücksichtigt (kein Blick in die
    "Zukunft" bei Zeitreise-Tests).
    """
    if not reports_dir.exists():
        return []

    candidates: list[tuple[date, Path]] = []
    for path in reports_dir.iterdir():
        m = _DAILY_FILE_RE.match(path.name)
        if not m:
            continue
        file_date = _parse_date(m.group(1))
        if file_date is None or file_date > today:
            continue
        candidates.append((file_date, path))
    candidates.sort(key=lambda x: x[0], reverse=True)

    result: list[tuple[date, dict]] = []
    for file_date, path in candidates:
        if len(result) >= limit:
            break
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        result.append((file_date, data))
    return result


def _killer_gate(stats: dict) -> str:
    """
    Killer-Gate = die Funnel-Stufe DIREKT NACH der letzten Stufe mit Wert > 0.
    Beispiel: universe=23, prescreened=5, analyzed=3, mismatch_ok=0 →
    letzte Stufe > 0 ist "analyzed" → Killer ist "mismatch_ok".
    """
    last_positive_idx = -1
    for i, key in enumerate(FUNNEL_ORDER):
        val = stats.get(key, 0) or 0
        if isinstance(val, (int, float)) and val > 0:
            last_positive_idx = i
    if last_positive_idx == -1:
        return FUNNEL_ORDER[0]  # schon das Universum ist leer
    if last_positive_idx + 1 < len(FUNNEL_ORDER):
        return FUNNEL_ORDER[last_positive_idx + 1]
    return FUNNEL_ORDER[last_positive_idx]  # praktisch unerreichbar (trades = letzte Stufe)


# ── a) Dürre-Streak + Killer-Diagnose ────────────────────────────────────────

# Streak über mehr als 10 Reports zählen (Live 2026-11-05: "10 Tage" gemeldet, tatsächlich 13 seit dem letzten
# Trade am 16.10.); davor liegende Tage dienen als Basis der Durchsatz-Erwartung.
DUERRE_LOOKBACK_REPORTS = 120
THROUGHPUT_BASELINE_DAYS = 30           # Handelstage vor dem Streak als Vergleichsbasis
THROUGHPUT_MIN_BASE_ANALYZED = 50       # darunter keine belastbare Quote Trades je Analyse
THROUGHPUT_MIN_BASE_TRADES = 3
THROUGHPUT_COLLAPSE_SHARE = 0.25        # Durchsatz < 25 % der Basis = eingebrochen
THROUGHPUT_RECENT_DAYS = 5


def _num_or_none(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _throughput_metrics(streak_days: list[dict], base_days: list[dict]) -> dict | None:
    """Erwartete Trades im Streak bei der Umwandlungsquote der Vorperiode (Trades je LLM-Analyse).

    Der Killer-Gate-Befund nennt nur die erste Stufe mit 0. Fällt der Durchsatz (LLM-Analysen je Tag) z. B. durch
    ein Budget auf wenige Kandidaten, ist auch ein unverändertes Gate an jedem Tag "Killer", obwohl die Dürre allein
    aus der kleinen Zahl der Versuche folgt (Live 2026-10-28..11-06: 1-3 statt 28-96 Analysen/Tag). P(0 Trades) nach
    Poisson mit der Erwartung; hoch = Dürre durch den Durchsatz erklärbar, klein = Ursache liegt hinter der Analyse
    (Gate, Markt). Reine Beschreibung, keine Schwellen- oder Gate-Änderung."""
    base = [d for d in base_days if d["analyzed"] is not None and d["trades"] is not None]
    streak = [d for d in streak_days if d["analyzed"] is not None]
    base_an = sum(d["analyzed"] for d in base)
    base_tr = sum(d["trades"] for d in base)
    if not streak or base_an < THROUGHPUT_MIN_BASE_ANALYZED or base_tr < THROUGHPUT_MIN_BASE_TRADES:
        return None
    rate = base_tr / base_an
    streak_an = sum(d["analyzed"] for d in streak)
    expected = rate * streak_an
    base_per_day = base_an / len(base)
    recent = streak[:THROUGHPUT_RECENT_DAYS]          # neueste zuerst
    recent_per_day = sum(d["analyzed"] for d in recent) / len(recent)
    collapsed_since, collapsed_an = None, 0.0
    if recent_per_day < THROUGHPUT_COLLAPSE_SHARE * base_per_day:
        for d in streak:                                # zurück bis zum ersten Tag über der Schwelle
            if d["analyzed"] >= THROUGHPUT_COLLAPSE_SHARE * base_per_day:
                break
            collapsed_since, collapsed_an = d["date"], collapsed_an + d["analyzed"]
    return {
        "baseline_days": len(base), "baseline_from": base[-1]["date"], "baseline_to": base[0]["date"],
        "baseline_analyzed_per_day": round(base_per_day, 2),
        "baseline_trades_per_analysis": round(rate, 5),
        "streak_analyzed": int(streak_an), "streak_analyzed_per_day": round(streak_an / len(streak), 2),
        "recent_analyzed_per_day": round(recent_per_day, 2),
        "expected_trades": round(expected, 2), "p_zero_trades": round(math.exp(-expected), 4),
        "throughput_collapsed_since": collapsed_since,
        "expected_trades_since_collapse": round(rate * collapsed_an, 2) if collapsed_since else None,
    }


def _duerre_metrics(reports_dir: Path, today: date) -> dict:
    recent = _load_recent_daily_reports(reports_dir, today, limit=DUERRE_LOOKBACK_REPORTS)

    streak = 0
    killer_counts: dict[str, int] = {}
    days = []
    for file_date, data in recent:
        stats = data.get("stats") or {}
        days.append({"date": file_date.strftime("%Y-%m-%d"), "analyzed": _num_or_none(stats.get("analyzed")),
                     "trades": _num_or_none(stats.get("trades"))})
    for _, data in recent:
        stats = data.get("stats") or {}
        trades = stats.get("trades", 0) or 0
        if trades != 0:
            break
        streak += 1
        killer = _killer_gate(stats)
        killer_counts[killer] = killer_counts.get(killer, 0) + 1

    last_10_days = []
    for file_date, data in recent[:10]:
        stats = data.get("stats") or {}
        trades = stats.get("trades", 0) or 0
        last_10_days.append({
            "date":   file_date.strftime("%Y-%m-%d"),
            "trades": trades,
            "killer": _killer_gate(stats) if trades == 0 else None,
        })

    throughput = _throughput_metrics(days[:streak], days[streak:streak + THROUGHPUT_BASELINE_DAYS])
    return {"streak": streak, "streak_capped": streak == len(recent) and len(recent) == DUERRE_LOOKBACK_REPORTS,
            "killer_counts": killer_counts, "last_10_days": last_10_days, "throughput": throughput}


def _throughput_note(tp: dict | None) -> str:
    if not tp:
        return ""
    note = (f" Durchsatz: Ø {tp['streak_analyzed_per_day']:.1f} LLM-Analysen/Tag im Streak (letzte "
            f"{THROUGHPUT_RECENT_DAYS}: {tp['recent_analyzed_per_day']:.1f}) vs. Ø {tp['baseline_analyzed_per_day']:.1f} "
            f"in den {tp['baseline_days']} Tagen davor; bei deren Quote ({tp['baseline_trades_per_analysis']:.2%} Trades je "
            f"Analyse) wären {tp['expected_trades']:.1f} Trades zu erwarten, P(0 Trades) = {tp['p_zero_trades']:.0%}.")
    if tp.get("throughput_collapsed_since"):
        note += (f" Durchsatz seit {tp['throughput_collapsed_since']} unter "
                 f"{THROUGHPUT_COLLAPSE_SHARE:.0%} der Basis (seitdem erwartet "
                 f"{tp['expected_trades_since_collapse']:.1f} Trades): bei so wenigen Analysen ist der Killer-Befund "
                 f"dieser Tage kein Beleg für ein zu scharfes Gate.")
    return note


# ── b) Paralleltest-Reife (Shadow-Trades + Trailing-Sim) ─────────────────────

def _parallel_test_metrics(history: dict, today: date, close_after_days: int) -> dict:
    shadow_trades = history.get("shadow_trades") or []
    by_reason: dict[str, dict] = {}
    for t in shadow_trades:
        if not isinstance(t, dict):
            continue
        reason = t.get("reject_reason") or "unknown"
        g = by_reason.setdefault(reason, {
            "total": 0, "evaluated": 0, "ready_in_14d": 0, "_eval_dates": [],
        })
        g["total"] += 1
        if t.get("outcome") is not None:
            g["evaluated"] += 1
            continue
        entry_date = _parse_date(t.get("entry_date"))
        if entry_date is None:
            continue
        eval_date = entry_date + timedelta(days=close_after_days)
        g["_eval_dates"].append(eval_date)
        if (eval_date - today).days <= 14:
            g["ready_in_14d"] += 1

    shadow_by_reason = {}
    for reason, g in by_reason.items():
        eval_dates = g.pop("_eval_dates")
        next_eval = min(eval_dates).strftime("%Y-%m-%d") if eval_dates else None
        shadow_by_reason[reason] = {**g, "next_eval_date": next_eval}

    trailing_sim = history.get("trailing_sim") or []
    open_n = closed_n = 0
    deltas = []
    for s in trailing_sim:
        if not isinstance(s, dict):
            continue
        if s.get("trailing_outcome") is None:
            open_n += 1
        else:
            closed_n += 1
            tp, trail = s.get("tp_outcome"), s.get("trailing_outcome")
            if isinstance(tp, (int, float)) and isinstance(trail, (int, float)):
                deltas.append(trail - tp)
    avg_delta = round(sum(deltas) / len(deltas), 4) if deltas else None

    return {
        "shadow_by_reason": shadow_by_reason,
        "trailing_sim": {"open": open_n, "closed": closed_n, "avg_delta": avg_delta},
    }


# ── c) Lern-Loop-Sanity ───────────────────────────────────────────────────────

def _check_learn_loop_sanity(
    history: dict, today: date, close_after_days: int, warnings: list[str],
) -> dict:
    """
    Prüft Sanity des Lern-Loops: verwaiste Trades,
    und Feature-Korrelationen mit Outcome (rein Diagnose).

    Das Pearson-Gewichts-Update (feedback.compute_pearson_weights) wurde am
    2026-11-09 zusammen mit QuasiML/RL entfernt; history["model_weights"] wird
    weder aktualisiert noch zum Scoren verwendet.

    Returns dict mit:
      - weight_update_status: immer "REMOVED"
      - weight_learning: Hinweistext zum entfernten Gewichts-Lernpfad
      - learn_loop_frozen: immer False (kein Gewichts-Lernpfad, der einfrieren könnte)
      - feature_corr: RELIABLE-Korrelationen ab 20 RELIABLE (Diagnose, oder {})
      - feature_corr_exploratory, n_reliable, n_non_reliable: Diagnose
    """
    metrics = {"learn_loop_frozen": False, "feature_corr": {},
               "weight_update_status": "REMOVED",
               "weight_learning": ("REMOVED (2026-11-09: Pearson-Update mit QuasiML/RL entfernt; "
                                   "Korrelationen nur Diagnose)")}

    stale_cutoff = close_after_days + 30
    stale = []
    for t in history.get("active_trades") or []:
        if not isinstance(t, dict):
            continue
        entry_date = _parse_date(t.get("entry_date"))
        if entry_date is None:
            continue
        age = (today - entry_date).days
        if age > stale_cutoff:
            stale.append(f"{t.get('ticker', '?')} ({age}d)")
    if stale:
        warnings.append(
            f"{len(stale)} active_trade(s) älter als {stale_cutoff} Tage "
            f"(sollte(n) längst geschlossen sein): {', '.join(stale[:5])}."
        )

    # Lern-Loop-Status: Das Pearson-Gewichts-Update wurde 2026-11-09 entfernt (weight_update_status
    # = "REMOVED", nie "frozen"). Zählungen und Korrelationen bleiben als reine Diagnose erhalten;
    # Korrelationen über alle Trades (inkl. UNKNOWN/RECONSTRUCTED) sind nur EXPLORATORY.
    try:
        from modules.outcomes import MIN_RELIABLE_FOR_WEIGHT_UPDATE, is_reliable_outcome
        closed_trades = history.get("closed_trades") or []
        valid_trades = [t for t in closed_trades if isinstance(t, dict) and t.get("outcome") is not None]
        reliable = [t for t in valid_trades if is_reliable_outcome(t)]
        n_rel, n_non = len(reliable), len(valid_trades) - len(reliable)
        metrics.update(n_reliable=n_rel, n_non_reliable=n_non,
                       min_reliable_for_weight_update=MIN_RELIABLE_FOR_WEIGHT_UPDATE)

        exploratory = _feature_correlations(valid_trades) if len(valid_trades) >= 20 else {}
        metrics["feature_corr_exploratory"] = exploratory
        # Rein diagnostisch: RELIABLE-Korrelation ab 20 RELIABLE (wie bisher), keine Warnung,
        # kein Freeze-Status – es gibt keinen Gewichts-Lernpfad mehr.
        metrics["feature_corr"] = _feature_correlations(reliable) if n_rel >= 20 else {}

    except Exception as e:
        # Fehler in Korrelations-Berechnung → nur loggen, keine Warnung
        log.debug(f"Feature-Korrelations-Check fehlgeschlagen: {e}")

    return metrics


def _feature_correlations(trades: list[dict]) -> dict:
    """Pearson-Korrelation Feature-Bin vs. Outcome (Diagnose, gleiche Bin-Abbildung wie feedback)."""
    outcomes, impacts, mismatches, drifts = [], [], [], []
    for t in trades:
        outcome = t.get("outcome")
        if outcome is None:
            continue
        outcomes.append(float(outcome))
        feat = t.get("features") or {}
        impacts.append(_bin_to_num("impact", feat.get("bin_impact", "mid")))
        mismatches.append(_bin_to_num("mismatch", feat.get("bin_mismatch", "good")))
        drifts.append(_bin_to_num("eps_drift", feat.get("bin_eps_drift", "noise")))
    correlations = {}
    for feature_name, feature_vals in (("impact", impacts), ("mismatch", mismatches), ("eps_drift", drifts)):
        try:
            if len(set(feature_vals)) <= 1 or len(set(outcomes)) <= 1:
                correlations[feature_name] = 0.0
            else:
                corr = statistics.correlation(feature_vals, outcomes)
                correlations[feature_name] = round(corr, 3) if math.isfinite(corr) else 0.0
        except (ValueError, statistics.StatisticsError):
            correlations[feature_name] = 0.0
    return correlations


# ── e) Externer Kontext: Lern-Health (nur Warnungen, keine Gates) ────────────
# Reine Observability über den externen Real-Economy-Kontext (siehe
# modules/external/). Ändert nichts an mode=shadow/production, keine Gates.

EXTERNAL_PRIMITIVES_FOR_MISSINGNESS = (
    "freight_global_z", "us_freight_tsi_z", "shipping_global_z",
    "suez_z", "panama_z", "hormuz_z", "malacca_z", "bab_el_mandeb_z",
    "weather_disruption_index",
)
EXTERNAL_MISSING_RATIO_WARN = 0.80
# Ab wie vielen Tagen Konstanz ein Primitiv als "eingefroren" gilt — je nach
# Aktualisierungstakt der Quelle. Vorher: jede Konstanz über >= 2 Zeilen
# -> Fehlalarm am 2026-10-01 nach nur 4 Ledger-Tagen, obwohl PortWatch
# wöchentlich und BTS-TSI/Destatis/Eurostat/e-Stat monatlich publizieren.
EXTERNAL_CONSTANT_MIN_DAYS = {
    "freight_global_z": 75, "us_freight_tsi_z": 75,          # Monatsdaten
    "shipping_global_z": 21, "suez_z": 21, "panama_z": 21,   # PortWatch wöchentlich
    "hormuz_z": 21, "malacca_z": 21, "bab_el_mandeb_z": 21,
    "weather_disruption_index": 14,
}
EXTERNAL_TINY_EFFECTIVE_N   = 10


def _external_ledger_rows_last_30d(today: date) -> list[dict]:
    """Lädt Candidate-Ledger-Zeilen der letzten 30 Kalendertage (read-only,
    über modules.challenger.load_ledger_rows). Nie ein Fehler nach außen."""
    try:
        from modules.challenger import load_ledger_rows
        cutoff = today - timedelta(days=30)
        rows = load_ledger_rows()
        out = []
        for r in rows:
            d = _parse_date(r.get("date"))
            if d is not None and d >= cutoff:
                out.append(r)
        return out
    except Exception:
        return []


def _check_external_missingness(rows: list[dict], warnings: list[str]) -> dict:
    """Primitive, die über 30 Tage konstant ODER >80% missing sind."""
    result: dict[str, dict] = {}
    ext_rows = [r for r in rows if isinstance(r.get("external"), dict)]
    if not ext_rows:
        return result
    for prim in EXTERNAL_PRIMITIVES_FOR_MISSINGNESS:
        values = []
        n_total = len(ext_rows)
        for r in ext_rows:
            v = ((r.get("external") or {}).get("primitives") or {}).get(prim)
            values.append(v)
        present = [v for v in values if v is not None]
        missing_ratio = 1 - (len(present) / n_total) if n_total else 1.0
        dates = [d for d in (_parse_date(r.get("date")) for r, v in zip(ext_rows, values)
                             if v is not None) if d is not None]
        span_days = (max(dates) - min(dates)).days if dates else 0
        is_constant = len(present) >= 2 and len(set(present)) == 1
        # Wetter-Index 0.0 = keine schwere Warnung -> Normalzustand, kein Einfrieren.
        if prim == "weather_disruption_index" and is_constant and present[0] == 0:
            is_constant = False
        stale = is_constant and span_days >= EXTERNAL_CONSTANT_MIN_DAYS.get(prim, 21)
        result[prim] = {"missing_ratio": round(missing_ratio, 3), "constant": is_constant,
                         "constant_days": span_days if is_constant else 0,
                         "stale": stale, "n": n_total}
        if missing_ratio > EXTERNAL_MISSING_RATIO_WARN:
            warnings.append(
                f"Externes Primitiv '{prim}' ist über die letzten 30 Ledger-Tage zu "
                f"{missing_ratio:.0%} missing (>{EXTERNAL_MISSING_RATIO_WARN:.0%}-Schwelle)."
            )
        elif stale:
            warnings.append(
                f"Externes Primitiv '{prim}' ist seit {span_days} Tagen konstant "
                f"({present[0]!r}, Takt-Schwelle {EXTERNAL_CONSTANT_MIN_DAYS.get(prim, 21)} T) "
                f"— evtl. Quelle eingefroren/gestuckt."
            )
    return result


def _check_stale_high_criticality_sources(warnings: list[str]) -> dict:
    """Liest health/source_health.json (falls vorhanden) und warnt bei
    STALE-Status hoher Kritikalität. Fehlende Datei/Registry → kein Fehler."""
    try:
        from modules.config import cfg as _cfg
        archive_root = getattr(
            getattr(getattr(_cfg, "external_context", None), "archive", None),
            "root", "outputs/external_data",
        )
        from modules.external.registry import load_health, load_source_configs
        health = load_health(archive_root)
        sources = load_source_configs()
    except Exception:
        return {}

    stale_high = []
    for source_id, h in (health or {}).items():
        src_cfg = sources.get(source_id) or {}
        criticality = h.get("criticality") or src_cfg.get("criticality", "low")
        if criticality == "high" and h.get("staleness") == "STALE":
            stale_high.append(source_id)
    if stale_high:
        warnings.append(
            f"{len(stale_high)} hoch-kritische externe Quelle(n) sind STALE: "
            f"{', '.join(sorted(stale_high))}."
        )
    return {"stale_high_criticality": stale_high}


def _check_feature_stats_external_reliability(history: dict, rows: list[dict],
                                                warnings: list[str]) -> dict:
    """Warnt, wenn feature_stats_external-Buckets mit sehr kleinem
    effective_n dennoch im Monats-Report als belastbar erscheinen könnten
    (nur ein Hinweis, kein Gate)."""
    stats_ext = history.get("feature_stats_external") or {}
    tiny: list[str] = []
    try:
        from modules.external.research import analyze_external_buckets
        buckets = analyze_external_buckets(rows) if rows else {}
    except Exception:
        buckets = {}
    for dim, values in stats_ext.items():
        if not isinstance(values, dict):
            continue
        for key, b in values.items():
            count = int(b.get("count", 0)) if isinstance(b, dict) else 0
            eff_n = None
            group_stats = (buckets.get(dim) or {}).get(str(key)) if buckets else None
            if isinstance(group_stats, dict):
                eff_n = group_stats.get("effective_n")
            if count > 0 and (eff_n is not None and eff_n < EXTERNAL_TINY_EFFECTIVE_N):
                tiny.append(f"{dim}={key} (effective_n={eff_n:.1f})")
    if tiny:
        warnings.append(
            f"feature_stats_external: {len(tiny)} Bucket(s) mit sehr kleinem "
            f"effective_n (<{EXTERNAL_TINY_EFFECTIVE_N}) werden im Report angezeigt — "
            f"Vorsicht bei der Interpretation: {', '.join(tiny[:5])}."
        )
    return {"tiny_effective_n_buckets": tiny}


def _check_promoted_external_dominance(warnings: list[str]) -> dict:
    """Hinweis-Check: sind externe Features promotet (learning.
    promoted_external_features nicht leer)? Seit 2026-11-09 gibt es keinen
    Lern-Loop mehr, der model_weights schreibt; ein Gewichts-Dominanz-Check
    ist daher nicht möglich. Die Warnung verweist auf die manuelle Prüfung."""
    try:
        from modules.config import cfg as _cfg
        learning = getattr(getattr(_cfg, "external_context", None), "learning", None)
        promoted = list(getattr(learning, "promoted_external_features", []) or [])
    except Exception:
        promoted = []
    if not promoted:
        return {"promoted_external_features": []}
    warnings.append(
        f"{len(promoted)} externe Feature(s) sind promotet "
        f"({', '.join(promoted)}) — Einfluss im Production Intelligence Adapter manuell prüfen "
        f"(kein automatischer Dominanz-Check: model_weights wird nicht mehr gelernt)."
    )
    return {"promoted_external_features": promoted}


def _check_external_health(history: dict, today: date, warnings: list[str]) -> dict:
    """Baut metrics['external'] — reine Warnungen, kein Gate, kein Netzwerk,
    kein Schreibzugriff. Jede Sub-Prüfung ist einzeln try/except-geschützt."""
    metrics: dict = {}
    try:
        rows = _external_ledger_rows_last_30d(today)
    except Exception:
        rows = []
    try:
        metrics["missingness"] = _check_external_missingness(rows, warnings)
    except Exception as e:
        log.debug(f"engine_monitor: external missingness Fehler (ignoriert): {e}")
    try:
        metrics["source_health"] = _check_stale_high_criticality_sources(warnings)
    except Exception as e:
        log.debug(f"engine_monitor: external source_health Fehler (ignoriert): {e}")
    try:
        metrics["feature_stats_external"] = _check_feature_stats_external_reliability(
            history, rows, warnings)
    except Exception as e:
        log.debug(f"engine_monitor: external feature_stats Fehler (ignoriert): {e}")
    try:
        metrics["promoted_dominance"] = _check_promoted_external_dominance(warnings)
    except Exception as e:
        log.debug(f"engine_monitor: external promoted-dominance Fehler (ignoriert): {e}")
    return metrics


# ── d) Data-Health ────────────────────────────────────────────────────────────

def _check_data_health(reports_dir: Path, today: date, warnings: list[str]) -> None:
    latest = _load_recent_daily_reports(reports_dir, today, limit=1)
    if not latest:
        warnings.append("Keine Daily-Reports gefunden (Scanner-Historie leer oder Verzeichnis fehlt).")
        return

    latest_date, data = latest[0]
    age_days = (today - latest_date).days
    if age_days > 5:
        warnings.append(
            f"Letzter Daily-Report ist {age_days} Tage alt ({latest_date.strftime('%Y-%m-%d')}) "
            f"→ Scanner läuft vermutlich nicht mehr."
        )

    stats = data.get("stats") or {}
    if not stats.get("universe"):
        warnings.append(
            f"Neuester Report ({latest_date.strftime('%Y-%m-%d')}): universe=0 (keine Ticker im Universum)."
        )
    if stats.get("vix") is None:
        warnings.append(f"Neuester Report ({latest_date.strftime('%Y-%m-%d')}): VIX fehlt.")


# ── Öffentliche API ───────────────────────────────────────────────────────────

def build_health_report(history: dict, reports_dir, today: date) -> dict:
    """
    Baut den Engine-Health-Report. Reine Funktion — kein Netzwerk, keine
    Seiteneffekte, verändert weder history noch die Daily-Reports.

    Returns:
        {"status": "OK"|"WARN", "warnings": [str, ...], "metrics": {...}}
    """
    reports_dir = Path(reports_dir)
    if isinstance(today, datetime):
        today = today.date()

    warnings: list[str] = []
    close_after_days = _get_close_after_days()

    duerre = _duerre_metrics(reports_dir, today)
    if duerre["streak"] >= 5:
        if duerre["killer_counts"]:
            dominant = max(duerre["killer_counts"], key=duerre["killer_counts"].get)
            warnings.append(
                f"Dürre-Streak: {duerre['streak']} Tage ohne Trade in Folge "
                f"(dominanter Killer: {dominant}, {duerre['killer_counts'][dominant]}x)."
                + _throughput_note(duerre.get("throughput"))
            )
        else:
            warnings.append(f"Dürre-Streak: {duerre['streak']} Tage ohne Trade in Folge."
                            + _throughput_note(duerre.get("throughput")))

    parallel_tests = _parallel_test_metrics(history, today, close_after_days)

    learn_loop_metrics = _check_learn_loop_sanity(history, today, close_after_days, warnings)
    _check_data_health(reports_dir, today, warnings)
    external_metrics = _check_external_health(history, today, warnings)

    return {
        "status":   "WARN" if warnings else "OK",
        "warnings": warnings,
        "metrics":  {
            "duerre":          duerre,
            "parallel_tests":  parallel_tests,
            "learn_loop":      learn_loop_metrics,
            "external":        external_metrics,
        },
    }


def append_markdown_section(md_path, health: dict) -> None:
    """
    Hängt einen "## Engine-Status"-Abschnitt an eine bestehende Daily-Markdown
    an. Rein additiv — verändert keine bestehenden Zeilen. Falls die Datei
    nicht existiert, passiert nichts (Reporter läuft nur bei Trade-Vorschlägen
    bzw. nicht auf jedem Exit-Pfad).

    Fail-safe: JEDE Exception wird nur geloggt, nie weitergereicht — das
    Monitoring darf die Pipeline nicht zum Absturz bringen.
    """
    try:
        md_path = Path(md_path)
        if not md_path.exists():
            log.debug(f"Engine-Monitor: {md_path} existiert nicht → Abschnitt übersprungen")
            return

        status   = health.get("status", "OK")
        warnings = health.get("warnings") or []
        metrics  = health.get("metrics") or {}
        duerre   = metrics.get("duerre", {})
        parallel = metrics.get("parallel_tests", {})
        trailing = parallel.get("trailing_sim", {})
        shadow   = parallel.get("shadow_by_reason", {})

        lines = ["", "---", "## Engine-Status", "", f"**Status:** {status}", ""]

        if warnings:
            lines.append("**Warnungen:**")
            lines += [f"- {w}" for w in warnings]
        else:
            lines.append("_Keine Warnungen._")
        lines.append("")

        lines.append("**Reife-Übersicht (Paralleltests):**  ")
        lines.append(f"- Dürre-Streak: {duerre.get('streak', 0)} Tag(e) ohne Trade")
        if shadow:
            parts = [
                f"{reason} {g.get('evaluated', 0)}/{g.get('total', 0)} reif"
                for reason, g in shadow.items()
            ]
            lines.append(f"- Shadow-Trades: {', '.join(parts)}")
        else:
            lines.append("- Shadow-Trades: keine")
        avg_delta = trailing.get("avg_delta")
        avg_delta_str = f"{avg_delta:+.2%}" if isinstance(avg_delta, (int, float)) else "n/a"
        lines.append(
            f"- Trailing-Sim: {trailing.get('open', 0)} offen, "
            f"{trailing.get('closed', 0)} geschlossen (Ø Δ vs. TP: {avg_delta_str})"
        )
        lines.append("")

        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    except Exception as e:
        log.error(f"Engine-Monitor: append_markdown_section-Fehler: {e}")
