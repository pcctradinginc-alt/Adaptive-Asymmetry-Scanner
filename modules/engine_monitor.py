"""
modules/engine_monitor.py – Engine-Überwachung (reine Observability)

Zweck:
  Der Scanner hat mehrere Paralleltests laufen (Schatten-Trades, Trailing-Sim),
  eine mehrstufige Gate-Pipeline und einen Lern-Loop, der auf model_weights
  schreibt. Bricht irgendwo etwas (Datenquelle tot, Gate zu scharf, NaN im
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
  c) LERN-LOOP-SANITY – NaN-/Null-Gewichte, verwaiste (überfällige) Trades
  d) DATA-HEALTH      – Läuft der Scanner noch? Kommen sinnvolle Daten rein?
"""

import json
import logging
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from modules.config import cfg

log = logging.getLogger(__name__)

# Funnel-Reihenfolge für die Killer-Diagnose (siehe pipeline.py stats-dict).
# Bewusst reduziert auf die "Haupt-Trichter"-Stufen — Zwischenstufen wie
# sector_ok/pre_mc/roi_precheck/rl_scored/after_isf sind Diagnose-Detail,
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

def _duerre_metrics(reports_dir: Path, today: date) -> dict:
    recent = _load_recent_daily_reports(reports_dir, today, limit=10)

    streak = 0
    killer_counts: dict[str, int] = {}
    for _, data in recent:
        stats = data.get("stats") or {}
        trades = stats.get("trades", 0) or 0
        if trades != 0:
            break
        streak += 1
        killer = _killer_gate(stats)
        killer_counts[killer] = killer_counts.get(killer, 0) + 1

    last_10_days = []
    for file_date, data in recent:
        stats = data.get("stats") or {}
        trades = stats.get("trades", 0) or 0
        last_10_days.append({
            "date":   file_date.strftime("%Y-%m-%d"),
            "trades": trades,
            "killer": _killer_gate(stats) if trades == 0 else None,
        })

    return {"streak": streak, "killer_counts": killer_counts, "last_10_days": last_10_days}


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
) -> None:
    weights = history.get("model_weights") or {}
    if isinstance(weights, dict) and weights:
        nan_feats, numeric_vals = [], []
        for k, v in weights.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                nan_feats.append(k)
            else:
                numeric_vals.append(fv)
        if nan_feats:
            warnings.append(f"model_weights enthält NaN: {', '.join(sorted(nan_feats))}.")
        elif numeric_vals and all(v == 0 for v in numeric_vals):
            warnings.append("Alle model_weights sind exakt 0 (Lern-Loop liefert kein Signal mehr).")

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
            )
        else:
            warnings.append(f"Dürre-Streak: {duerre['streak']} Tage ohne Trade in Folge.")

    parallel_tests = _parallel_test_metrics(history, today, close_after_days)

    _check_learn_loop_sanity(history, today, close_after_days, warnings)
    _check_data_health(reports_dir, today, warnings)

    return {
        "status":   "WARN" if warnings else "OK",
        "warnings": warnings,
        "metrics":  {
            "duerre":          duerre,
            "parallel_tests":  parallel_tests,
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
