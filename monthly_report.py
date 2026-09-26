"""
monthly_report.py – Monatlicher Performance-Report per E-Mail

Läuft am 1. jedes Monats (GitHub Actions: monthly_report.yml) und sendet:
  - Win-Rate, Trade-Anzahl, Mean/Median-Return, Totalverluste des Vormonats
    (= Trades mit close_date im Vormonat)
  - Rollende Statistiken (30er, 60er Fenster, alle) mit ehrlichen Metriken und
    90%-Konfidenzintervallen (dedupliziert) — objektive Bewertung ohne Übertreibung
  - Gesamt-Statistik über alle closed_trades
  - Signal-Funnel: Wie viele Tage hatten 0 Trades und welche Gates blockierten

Nutzt dieselben GMAIL_SENDER / GMAIL_APP_PW / NOTIFY_EMAIL Secrets wie der
tägliche Scanner. Keine zusätzlichen Kosten, keine neuen API-Keys.
"""

import json
import logging
import os
import random
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


def dedup_closed_trades(closed: list[dict]) -> list[dict]:
    """
    Deduplicate trades: keep one trade per (ticker, entry_date).
    Returns only the first occurrence of each unique (ticker, entry_date) pair.
    """
    seen = set()
    result = []
    for trade in closed:
        ticker = trade.get("ticker", "?")
        entry_date = trade.get("entry_date", "?")
        key = (ticker, entry_date)
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

    # total_loss_rate: fraction of outcomes <= -0.95
    total_losses = sum(1 for o in outs if o <= -0.95)
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
    """Outcome-Statistik der Schatten-Trades (verworfene Signale) des Monats."""
    shadows = [
        t for t in history.get("shadow_trades", [])
        if t.get("outcome") is not None
        and str(t.get("close_date", ""))[:7] == key
    ]
    if not shadows:
        return None
    outs = [float(t["outcome"]) for t in shadows]
    wins = sum(1 for o in outs if o > 0)
    return {"n": len(outs), "win_rate": wins / len(outs), "wins": wins,
            "mean": statistics.mean(outs)}


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


def build_tuning_html(suggestions: list[dict]) -> str:
    if not suggestions:
        return (
            "<h3>🔧 Schwellen-Tuning</h3>"
            "<p>Keine Empfehlung diesen Monat — keine Alternative erfüllt die "
            "Guardrails (≥20 Trades, ≥5pp Win-Rate-Gewinn, Ø-Return nicht schlechter). "
            "Das ist ein gutes Zeichen oder es fehlen noch Daten.</p>"
        )
    rows = ""
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
    return f"""
    <h3>🔧 Schwellen-Tuning-Vorschlag (echte + Schatten-Trades)</h3>
    <table style="border-collapse:collapse;font-size:13px;width:100%">
      <tr style="text-align:left;color:#64748b;">
        <th style="padding:4px 8px;">Gate</th><th style="padding:4px 8px;">Aktuell</th>
        <th style="padding:4px 8px;">Vorschlag</th><th style="padding:4px 8px;">Δ Win-Rate</th>
      </tr>
      {rows}
    </table>
    <p style="font-size:0.85em;color:#92400e;">⚠️ Nur eine Empfehlung — nichts wurde
    automatisch geändert. Anpassung in config.yaml bzw. pipeline.py, idealerweise
    max. eine Schwelle pro Monat (sonst ist der Effekt nicht zuordenbar).</p>"""


def build_slot_html(candidates: list[dict]) -> str:
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
    Kandidat umgesetzt wird.</p>"""


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

        outcomes_for_ci = [float(t["outcome"]) for t in sorted_trades[-window:] if t.get("outcome") is not None] if window else [float(t["outcome"]) for t in sorted_trades if t.get("outcome") is not None]
        ci = bootstrap_ci(outcomes_for_ci)
        if ci:
            ci_str = f"{ci[0]:+.1%} bis {ci[1]:+.1%}"
        else:
            ci_str = "—"

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
        outs = [float(t["outcome"]) for t in sorted_trades]
        last_ci = bootstrap_ci(outs[-30:]) if len(outs) >= 60 else None
        prior_ci = bootstrap_ci(outs[-60:-30]) if len(outs) >= 60 else None
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
    <h3>📊 Beobachtete Performance (rollierend)</h3>
    <table style="border-collapse:collapse;font-size:13px;width:100%">
      <tr style="text-align:left;color:#64748b;">
        <th style="padding:4px 8px;">Fenster</th>
        <th style="padding:4px 8px;">n</th>
        <th style="padding:4px 8px;">Win-Rate</th>
        <th style="padding:4px 8px;">Ø</th>
        <th style="padding:4px 8px;">Median</th>
        <th style="padding:4px 8px;">Ø ohne Top-3</th>
        <th style="padding:4px 8px;">Totalverlust-Quote</th>
        <th style="padding:4px 8px;">90%-KI des Ø</th>
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


def build_html(report_month: str, cur: dict | None, prev: dict | None,
               total: dict | None, funnel: dict, closed: list[dict] | None = None,
               spy: float | None = None, shadow: dict | None = None,
               tuning: list[dict] | None = None,
               slot: list[dict] | None = None) -> str:
    def stat_block(label: str, s: dict | None) -> str:
        if s is None:
            return f"<p><b>{label}:</b> keine geschlossenen Trades</p>"
        return (
            f"<p><b>{label}:</b> Win-Rate <b>{_fmt_pct(s['win_rate'])}</b> "
            f"({s['wins']}/{s['n']}) · Ø {s['mean']:+.1%} · "
            f"Median {s['median']:+.1%} · Totalverluste {s['total_losses']}</p>"
        )

    # Rolling stats and month-over-month analysis
    rolling_html = ""
    if closed:
        rolling_html = build_rolling_stats_html(closed, prev, cur)

    # SPY-Benchmark: schlägt das System buy-and-hold?
    bench_html = ""
    if spy is not None and cur:
        edge  = cur["mean"] - spy
        color = "#16a34a" if edge > 0 else "#dc2626"
        bench_html = (
            f"<p><b>Benchmark:</b> Ø Trade-Return {cur['mean']:+.1%} vs. "
            f"SPY {spy:+.1%} → Edge "
            f"<b style='color:{color}'>{edge:+.1%}</b> "
            f"<i>(Achtung: Options-Returns sind gehebelt — fairer Vergleich nur "
            f"über das eingesetzte Risikokapital)</i></p>"
        )
    elif spy is not None:
        bench_html = f"<p><b>Benchmark:</b> SPY {spy:+.1%} im {report_month}</p>"

    # Schatten-Trades: filtern die Gates Gewinner weg?
    shadow_html = ""
    if shadow:
        shadow_html = (
            f"<p><b>Schatten-Trades</b> (von Gates verworfen, nur getrackt): "
            f"Win-Rate {_fmt_pct(shadow['win_rate'])} ({shadow['wins']}/{shadow['n']}) "
            f"· Ø {shadow['mean']:+.1%} — "
            f"{'⚠️ Gates filtern evtl. Gewinner weg!' if cur and shadow['win_rate'] > cur['win_rate'] else 'Gates arbeiten korrekt.'}</p>"
        )

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
      {rolling_html}
      {bench_html}
      {shadow_html}
      <hr>
      {stat_block("Gesamt (alle closed Trades)", total)}
      {build_tuning_html(tuning or [])}
      {build_slot_html(slot or [])}
      {build_challenger_html()}
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
    spy    = spy_return(report_month)
    shadow = shadow_stats(history, report_month)
    tuning = tuning_suggestions(history)
    slot   = slot_candidates(history)

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

    if tuning:
        log.info(f"{len(tuning)} Tuning-Vorschlag/Vorschläge gefunden")
    html = build_html(report_month, cur, prev, total, funnel, closed, spy, shadow, tuning, slot)
    log.info(f"Sende Monats-Report: {subject}")
    _send_smtp(subject, html)
    log.info("=== Monats-Report abgeschlossen ===")


if __name__ == "__main__":
    main()
