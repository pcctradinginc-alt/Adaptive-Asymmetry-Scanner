"""
monthly_report.py – Monatlicher Performance-Report per E-Mail

Läuft am 1. jedes Monats (GitHub Actions: monthly_report.yml) und sendet:
  - Win-Rate, Trade-Anzahl, Mean/Median-Return, Totalverluste des Vormonats
    (= Trades mit close_date im Vormonat)
  - Vergleich zum Monat davor (Δ Win-Rate) → wird das System besser?
  - Gesamt-Statistik über alle closed_trades
  - Signal-Funnel: Wie viele Tage hatten 0 Trades und welche Gates blockierten

Nutzt dieselben GMAIL_SENDER / GMAIL_APP_PW / NOTIFY_EMAIL Secrets wie der
tägliche Scanner. Keine zusätzlichen Kosten, keine neuen API-Keys.
"""

import json
import logging
import os
import statistics
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from modules.email_reporter import _send_smtp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
REPORTS_DIR  = Path("outputs/daily_reports")
REPO_NAME    = os.getenv("GITHUB_REPOSITORY", "Adaptive-Asymmetry-Scanner")


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
        "total_losses": sum(1 for o in outs if o <= -0.99),
    }


def overall_stats(closed: list[dict]) -> dict | None:
    outs = [float(t["outcome"]) for t in closed if t.get("outcome") is not None]
    if not outs:
        return None
    wins = sum(1 for o in outs if o > 0)
    return {
        "n": len(outs), "win_rate": wins / len(outs), "wins": wins,
        "mean": statistics.mean(outs), "median": statistics.median(outs),
        "total_losses": sum(1 for o in outs if o <= -0.99),
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


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def build_html(report_month: str, cur: dict | None, prev: dict | None,
               total: dict | None, funnel: dict) -> str:
    def stat_block(label: str, s: dict | None) -> str:
        if s is None:
            return f"<p><b>{label}:</b> keine geschlossenen Trades</p>"
        return (
            f"<p><b>{label}:</b> Win-Rate <b>{_fmt_pct(s['win_rate'])}</b> "
            f"({s['wins']}/{s['n']}) · Ø {s['mean']:+.1%} · "
            f"Median {s['median']:+.1%} · Totalverluste {s['total_losses']}</p>"
        )

    if cur and prev:
        delta = (cur["win_rate"] - prev["win_rate"]) * 100
        trend = (
            f"<p style='font-size:1.1em'><b>Trend: {'📈' if delta >= 0 else '📉'} "
            f"{delta:+.0f} Prozentpunkte Win-Rate vs. Vormonat</b> — "
            f"das System wird {'besser' if delta > 0 else 'schlechter' if delta < 0 else 'nicht besser'}.</p>"
        )
    else:
        trend = "<p><i>Noch kein Vormonats-Vergleich möglich (zu wenige Daten).</i></p>"

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

    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:640px">
      <h2>📊 Monats-Report — {REPO_NAME}</h2>
      <p>Berichtsmonat: <b>{report_month}</b> (Trades mit Close-Datum in diesem Monat)</p>
      {stat_block(f"Monat {report_month}", cur)}
      {stat_block("Vormonat", prev)}
      {trend}
      <hr>
      {stat_block("Gesamt (alle closed Trades)", total)}
      {funnel_html}
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
    closed  = history.get("closed_trades", [])

    # Berichtsmonat = Vormonat (Report läuft am 1. des Folgemonats)
    today        = date.today()
    report_month = month_key(today.replace(day=1) - timedelta(days=1))
    prior_month  = prev_month(report_month)

    cur    = month_stats(closed, report_month)
    prev   = month_stats(closed, prior_month)
    total  = overall_stats(closed)
    funnel = funnel_summary(report_month)

    if cur and prev:
        delta   = (cur["win_rate"] - prev["win_rate"]) * 100
        subject = (
            f"📊 {REPO_NAME}: Win-Rate {_fmt_pct(cur['win_rate'])} im {report_month} "
            f"({delta:+.0f}pp vs. Vormonat)"
        )
    elif cur:
        subject = f"📊 {REPO_NAME}: Win-Rate {_fmt_pct(cur['win_rate'])} im {report_month}"
    else:
        subject = f"📊 {REPO_NAME}: Monats-Report {report_month} — keine geschlossenen Trades"

    html = build_html(report_month, cur, prev, total, funnel)
    log.info(f"Sende Monats-Report: {subject}")
    _send_smtp(subject, html)
    log.info("=== Monats-Report abgeschlossen ===")


if __name__ == "__main__":
    main()
