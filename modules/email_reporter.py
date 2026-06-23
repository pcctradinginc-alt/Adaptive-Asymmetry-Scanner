"""
modules/email_reporter.py v8.3

Änderungen v8.3:
    #5  Greeks-Block im Trade-Card:
        Delta (P(ITM)-Approximation), Theta/Tag, Vega-Exposure, Breakeven.
        Trader sieht jetzt nicht nur ROI, sondern auch wie empfindlich
        der Trade auf Seitwärtsbewegung und IV-Crush reagiert.

    #7  MC-Probabilitäten im Report:
        Hit-Rate aus Monte Carlo (P(Kurs > Ziel)) und Catalyst-Confidence
        werden als separater Block angezeigt. Macht Wahrscheinlichkeits-
        grundlage für den ROI transparent.

Änderungen v8.2:
    - Integration der Exit-Regeln (Take-Profit, Stop-Loss, Time-Exit)
    - Textlimit: [:700] für Best Argument For/Against
    - Score-Schwelle für Email: >= 50
    - Fix: pipeline_stats werden auch im Kein-Trade-Fall durchgereicht
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


def send_status_email(pipeline_stats: dict, today: str) -> None:
    trades  = pipeline_stats.get("trades", 0)
    subject = (
        f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
        if trades > 0
        else f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"
    )
    html = _build_status_email(pipeline_stats, today)
    _send_smtp(subject, html)


def send_email(proposals: list[dict], today: str, pipeline_stats: dict | None = None) -> None:
    pipeline_stats = pipeline_stats or {}
    proposals = [p for p in proposals
                 if p.get("trade_score", {}).get("total", 0) >= 65]
    if proposals:
        html    = _build_trade_email(proposals, today)
        subject = f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
    else:
        stats   = {**pipeline_stats, "trades": 0}
        html    = _build_status_email(stats, today)
        subject = f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"
    _send_smtp(subject, html)


def send_exit_alert_email(alerts: list[dict], today: str) -> None:
    """
    Sofort-Alarm wenn aktive Trades ihre Exit-Regel erreicht haben.
    Wird von feedback.py aufgerufen — der wichtigste Mail-Typ, denn hier
    geht es um offene Positionen mit echtem Geld.
    """
    LABELS = {
        "take_profit": ("🎯 TAKE-PROFIT", "#16a34a", "Gewinn mitnehmen — Position (oder Hälfte) schliessen"),
        "stop_loss":   ("🛑 STOP-LOSS",   "#dc2626", "Position liquidieren — Verlust begrenzen"),
        "time_exit":   ("⏳ TIME-EXIT",   "#ca8a04", "Theta-Decay: halbe Laufzeit um, Gewinn < +20% — schliessen"),
    }
    rows = ""
    for a in alerts:
        label, color, action = LABELS.get(a["reason"], (a["reason"], "#0f172a", ""))
        opt = a.get("option") or {}
        rows += f"""
        <div style="margin:10px 0;padding:12px 14px;border-left:4px solid {color};background:#f8fafc;border-radius:4px;">
          <b style="color:{color};">{label}</b> — <b>{a['ticker']}</b> {a.get('strategy','')}
          <br>Strike ${opt.get('strike','?')} · Expiry {opt.get('expiry','?')} ·
          aktueller P&amp;L: <b style="color:{color};">{a['outcome']:+.1%}</b> ·
          Haltedauer {a['age_days']}d
          <br><i>{action}</i>
        </div>"""
    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:640px">
      <h2>🚨 Exit-Alarm — {len(alerts)} Position(en) haben ihre Exit-Regel erreicht</h2>
      {rows}
      <p style="color:#888;font-size:0.85em">Automatisch generiert · {today} ·
      Diese Positionen wurden im Lern-System mit dem heutigen Stand geschlossen.</p>
    </body></html>"""
    n_tp = sum(1 for a in alerts if a["reason"] == "take_profit")
    n_sl = sum(1 for a in alerts if a["reason"] == "stop_loss")
    subject = f"🚨 Exit-Alarm: {len(alerts)} Position(en) ({n_tp}× TP, {n_sl}× SL) – {today}"
    _send_smtp(subject, html)


# Copy-Paste-Prompt für Claude Code: setzt den RL-Agenten von Option A (Veto aus)
# auf Option B (gelernter Filter mit Regret-Reward) um. Bewusst self-contained,
# damit der User ihn direkt einfügen kann, ohne weiteren Kontext.
RL_ARMING_PROMPT = (
    "Stelle den RL-Agenten des Adaptive-Asymmetry-Scanners scharf (Übergang Option A → B). "
    "Es liegen jetzt genug closed_trades unter den neuen Regeln vor. Mach Folgendes und "
    "committe erst nach meiner Freigabe:\n"
    "1) modules/rl_environment.py: Reward auf Regret-Shaping umbauen — SKIP darf nicht mehr "
    "gratis sein. reward(SKIP) = -max(0, outcome) (Wegskippen eines Gewinners wird bestraft), "
    "NORMAL = outcome, BOOST = outcome*1.5 wie bisher. KEIN Win-Capping (verschärft den SKIP-Kollaps).\n"
    "2) config.yaml: rl.timesteps_per_update von 2000 auf mindestens 50000 erhöhen.\n"
    "3) feedback.py retrain_rl_agent: einmalig force_retrain=True, damit nicht auf der alten "
    "degenerierten Policy aufgesetzt wird.\n"
    "4) Anti-Degenerations-Check ergänzen: nach dem Training auf den closed_trades evaluieren; "
    "wenn der Agent >90% skippt, rl.veto_enabled auf false lassen und mich warnen statt scharfzustellen.\n"
    "5) Backtest auf closed_trades: Win-Rate des getradeten Subsets vs. aller Trades berichten.\n"
    "6) Erst nach grünem Anti-Degenerations-Check rl.veto_enabled: true setzen.\n"
    "Berichte mir das Ergebnis vor dem Commit."
)


def send_rl_arming_email(n_trades: int, threshold: int,
                         win_rate: float, since: str) -> None:
    """
    Einmalige Email, sobald genug closed_trades unter den neuen Regeln vorliegen,
    um den RL-Agenten scharfzustellen (Option A → B). Enthält einen fertigen
    Prompt, den der User direkt an Claude Code geben kann.
    """
    html = _build_rl_arming_email(n_trades, threshold, win_rate, since)
    subject = f"🟢 RL scharfstellen bereit: {n_trades} Trades erreicht (Schwelle {threshold})"
    _send_smtp(subject, html)


def _build_rl_arming_email(n_trades: int, threshold: int,
                           win_rate: float, since: str) -> str:
    import html as _html
    prompt_escaped = _html.escape(RL_ARMING_PROMPT)
    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;margin:0;padding:0;background:#f8fafc;">
<div style="max-width:640px;margin:30px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
  <div style="background:#16a34a;padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:6px;">🟢</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">RL-Agent bereit zum Scharfstellen</div>
    <div style="color:rgba(255,255,255,0.85);font-size:16px;margin-top:4px;">Datenschwelle erreicht</div>
  </div>
  <div style="padding:24px 32px;color:#0f172a;font-size:14px;line-height:1.55;">
    <p><b>Trade-Anzahl für RL-Scharfstellung erreicht.</b></p>
    <table style="width:100%;border-collapse:collapse;margin:14px 0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">
      <tr><td style="padding:9px 16px;background:#f0fdf4;border-bottom:1px solid #e2e8f0;">✅&nbsp; Geschlossene Trades seit {since}</td>
          <td style="padding:9px 16px;background:#f0fdf4;border-bottom:1px solid #e2e8f0;text-align:right;"><b>{n_trades}</b> / {threshold}</td></tr>
      <tr><td style="padding:9px 16px;">📊&nbsp; Win-Rate dieser Trades</td>
          <td style="padding:9px 16px;text-align:right;"><b>{win_rate:.0%}</b></td></tr>
    </table>
    <p>Der RL-Agent läuft aktuell im <b>Option-A-Modus</b> (PPO-Veto aus, QuasiML-Ranking).
    Es liegen jetzt genug Outcomes unter den neuen Regeln vor, um ihn als echten gelernten
    Filter scharfzustellen (Option B: Regret-Reward + Voll-Retraining + Anti-Degenerations-Check).</p>
    <p style="margin-top:18px;"><b>Diesen Prompt direkt an Claude Code geben:</b></p>
    <pre style="white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#e2e8f0;
                padding:16px 18px;border-radius:8px;font-size:12.5px;line-height:1.5;
                font-family:Menlo,Consolas,monospace;user-select:all;">{prompt_escaped}</pre>
    <p style="color:#64748b;font-size:12px;">Hinweis: Claude soll <b>erst nach deiner Freigabe</b> committen und
    bei einem fehlgeschlagenen Anti-Degenerations-Check das Veto ausgelassen lassen.</p>
  </div>
  <div style="padding:14px 32px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;text-align:center;">
    Adaptive Asymmetry-Scanner &nbsp;·&nbsp; {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </div>
</div></body></html>"""


def _build_status_email(stats: dict, today: str) -> str:
    vix         = stats.get("vix")
    trades      = stats.get("trades", 0)
    header_col  = "#16a34a" if trades > 0 else "#0f172a"
    status_icon = "🎯" if trades > 0 else "📊"
    status_text = "Trade Empfehlung" if trades > 0 else "Kein Trade heute"

    funnel = [
        (f"{stats.get('universe', 0)} Ticker im Universum", "📋", True),
        (f"{stats.get('candidates', 0)} nach Hard-Filter (Cap>2B, Vol>1M)", "🔍", stats.get("candidates", 0) > 0),
        (f"{stats.get('sector_ok', stats.get('candidates', 0))} nach Sector-Momentum", "📈", stats.get("sector_ok", stats.get("candidates", 0)) > 0),
        (f"{stats.get('prescreened', 0)} nach Prescreening (Haiku)", "🤖", stats.get("prescreened", 0) > 0),
        (f"{stats.get('roi_precheck', 0)} nach ROI Pre-Check", "💰", stats.get("roi_precheck", 0) > 0),
        (f"{stats.get('pre_mc', stats.get('roi_precheck', 0))} nach Pre-MC-Gate", "⚡", stats.get("pre_mc", stats.get("roi_precheck", 0)) > 0),
        (f"{stats.get('analyzed', 0)} nach Deep Analysis (Sonnet)", "🧠", stats.get("analyzed", 0) > 0),
        (f"{stats.get('after_isf', stats.get('analyzed', 0))} nach Impact×Surprise-Floor", "🎯", stats.get("after_isf", stats.get("analyzed", 0)) > 0),
        (f"{stats.get('quick_mc', 0)} nach Quick Monte Carlo", "🎲", stats.get("quick_mc", 0) > 0),
        (f"{trades} finale Trade-Vorschläge", "🏆" if trades > 0 else "❌", trades > 0),
    ]

    rows = ""
    for label, icon, active in funnel:
        bg    = "#f0fdf4" if active else "#fef2f2"
        color = "#16a34a" if active else "#dc2626"
        rows += f"""<tr><td style="padding:9px 16px;font-size:13px;color:{color};background:{bg};border-bottom:1px solid #e2e8f0;">{icon}&nbsp; {label}</td></tr>"""

    vix_str = f"{float(vix):.2f}" if vix else "–"

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;margin:0;padding:0;background:#f8fafc;">
<div style="max-width:620px;margin:30px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
  <div style="background:{header_col};padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:6px;">{status_icon}</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">Adaptive Asymmetry-Scanner</div>
    <div style="color:rgba(255,255,255,0.85);font-size:16px;margin-top:4px;">{status_text}</div>
    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:6px;">{today} &nbsp;·&nbsp; VIX {vix_str} &nbsp;·&nbsp; v8.3</div>
  </div>
  <div style="padding:24px 32px;">
    <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;">{rows}</table>
  </div>
  <div style="padding:14px 32px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;text-align:center;">
    Adaptive Asymmetry-Scanner v8.3 &nbsp;·&nbsp; {datetime.utcnow().strftime('%H:%M UTC')}
  </div>
</div></body></html>"""


def _build_trade_email(proposals: list[dict], today: str) -> str:
    cards = ""
    for i, p in enumerate(proposals, 1):
        ticker   = p.get("ticker", "?")
        strategy = p.get("strategy", "?")
        da       = p.get("deep_analysis", {}) or {}
        sim      = p.get("simulation", {}) or {}
        option   = p.get("option", {}) or {}
        roi      = p.get("roi_analysis", {}) or {}
        exit_r   = p.get("exit_rules", {}) or {}
        ts       = p.get("trade_score", {}) or {}

        ts_total = ts.get("total", 0)
        ts_grade = ts.get("grade", "–")
        best_for = (ts.get("best_argument_for", "") or "")[:700]
        best_ag  = (ts.get("best_argument_against", "") or "")[:700]

        score_color = "#16a34a" if ts_total >= 75 else "#ca8a04" if ts_total >= 60 else "#ea580c"

        # ── v8.3 #5: Greeks-Block ─────────────────────────────────────────────
        delta          = roi.get("delta", 0) or 0
        theta_day_pct  = roi.get("theta_daily_pct", 0) or 0
        vega_loss      = roi.get("vega_loss", 0) or 0
        breakeven      = roi.get("breakeven", 0) or 0
        breakeven_pct  = roi.get("breakeven_pct", 0) or 0
        mc_weight      = roi.get("mc_weight", 0) or 0
        ttm            = p.get("time_to_maturation", da.get("time_to_materialization", "–"))

        theta_color = "#dc2626" if theta_day_pct > 0.025 else "#ca8a04" if theta_day_pct > 0.015 else "#16a34a"
        delta_str   = f"{delta:.2f}" if delta else "–"
        theta_str   = f"{theta_day_pct:.1%}/Tag" if theta_day_pct else "–"
        be_str      = f"${breakeven:.2f} (+{breakeven_pct:.1%})" if breakeven else "–"
        vega_str    = f"{vega_loss:.1%}" if vega_loss else "–"

        greeks_html = f"""
        <div style="margin-top:12px;padding:10px 14px;background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;font-size:12px;">
          <b style="color:#0369a1;">📐 Greeks &amp; Optionsmechanik</b>
          <table style="width:100%;margin-top:6px;font-size:12px;color:#0c4a6e;border-collapse:collapse;">
            <tr>
              <td style="padding:2px 8px 2px 0;"><b>Delta:</b> {delta_str}</td>
              <td style="padding:2px 8px 2px 0;"><b>Theta/Tag:</b> <span style="color:{theta_color};">{theta_str}</span></td>
            </tr>
            <tr>
              <td style="padding:2px 8px 2px 0;"><b>Breakeven:</b> {be_str}</td>
              <td style="padding:2px 8px 2px 0;"><b>Vega-Exposure:</b> {vega_str}</td>
            </tr>
          </table>
        </div>"""

        # ── v8.3 #7 / v10.0 #5: MC-Wahrscheinlichkeits-Block + Implied Move ────
        mc_hit_rate_pct = p.get("mc_hit_rate", 0) or 0
        cat_conf        = da.get("catalyst_confidence", None)
        dte_val         = option.get("dte", "–")
        catalyst_type   = p.get("catalyst_type", "OTHER")
        implied_move    = p.get("implied_move_pct")   # z.B. 8.2 (%)
        model_move      = p.get("model_move_pct", 0)  # z.B. 12.4 (%)
        edge_implied    = p.get("edge_vs_implied")     # z.B. 4.2 (%)

        mc_color = "#16a34a" if mc_hit_rate_pct >= 0.65 else "#ca8a04" if mc_hit_rate_pct >= 0.50 else "#dc2626"
        cat_str  = f"{cat_conf}/10" if cat_conf is not None else "–"

        # Implied Move Row
        if implied_move is not None:
            if edge_implied is not None and edge_implied > 5.0:
                edge_color = "#16a34a"
                edge_sign  = f"+{edge_implied:.1f}%"
            elif edge_implied is not None and edge_implied < -2.0:
                edge_color = "#dc2626"
                edge_sign  = f"{edge_implied:.1f}%"
            else:
                edge_color = "#ca8a04"
                edge_sign  = f"{edge_implied:+.1f}%" if edge_implied is not None else "–"
            implied_row = f"""
            <tr>
              <td style="padding:4px 8px 2px 0;" colspan="2">
                <b>Market-Implied:</b> ±{implied_move:.1f}% &nbsp;|&nbsp;
                <b>Model-Target:</b> +{model_move:.1f}% &nbsp;|&nbsp;
                <b>Edge:</b> <span style="color:{edge_color};font-weight:bold;">{edge_sign}</span>
                &nbsp;<span style="color:#64748b;font-size:11px;">({catalyst_type})</span>
              </td>
            </tr>"""
        else:
            implied_row = ""

        prob_html = f"""
        <div style="margin-top:8px;padding:10px 14px;background:#fefce8;border:1px solid #fde68a;border-radius:6px;font-size:12px;">
          <b style="color:#92400e;">📊 Wahrscheinlichkeiten &amp; Katalysator</b>
          <table style="width:100%;margin-top:6px;font-size:12px;color:#78350f;border-collapse:collapse;">
            <tr>
              <td style="padding:2px 8px 2px 0;"><b>MC Hit-Rate:</b> <span style="color:{mc_color};font-weight:bold;">{mc_hit_rate_pct:.0%}</span> (P Kurs &gt; Ziel)</td>
              <td style="padding:2px 8px 2px 0;"><b>Catalyst-Konfidenz:</b> {cat_str}</td>
            </tr>
            <tr>
              <td style="padding:2px 8px 2px 0;"><b>Thesis-Horizont:</b> {ttm}</td>
              <td style="padding:2px 8px 2px 0;"><b>Option-Laufzeit:</b> {dte_val}d</td>
            </tr>
            {implied_row}
          </table>
        </div>"""

        # ── Ausführungs-Block (Tradier-Sprache) ──────────────────────────────
        is_spread = strategy == "BULL_CALL_SPREAD"
        sl        = option.get("spread_leg") or {}

        # ── Exit-Block (strategie-spezifisch, direkt nach Entry) ─────────────
        exit_html = ""
        if exit_r and exit_r.get("entry_cost", 0) > 0:
            vol_crush_pts = exit_r.get("vol_crush_pts", 30)
            underlying_be = exit_r.get("underlying_breakeven", 0)
            delta_exit_v  = exit_r.get("delta_exit")
            entry_c       = exit_r["entry_cost"]
            entry_usd     = round(entry_c * 100)
            sl_price      = exit_r["stop_loss_price"]
            tp1_price     = exit_r["take_profit_price"]
            tp2_price     = exit_r["full_profit_price"]
            sl_pnl        = round((sl_price  - entry_c) * 100)
            tp1_pnl       = round((tp1_price - entry_c) * 100)
            tp2_pnl       = round((tp2_price - entry_c) * 100)

            if is_spread:
                tp_label  = f"+{exit_r['take_profit_pct']}% Max-Gewinn"
                tp2_label = f"+{exit_r['full_profit_pct']}% Max-Gewinn"
                sl_label  = f"−{abs(exit_r['stop_loss_pct'])}% Net-Debit"
                sl_action = "Sell to Close Spread · Limit GTC"
                warn_row  = f"""
                <tr><td colspan="3" style="padding:5px 0 0;font-size:11px;color:#92400e;">
                  ⚠️ Warnschwellen: IV-Crush &gt;{vol_crush_pts} Pkte &nbsp;|&nbsp;
                  Kurs &lt; Break-even + 1%&nbsp;(${underlying_be:.2f})
                </td></tr>"""
                title = "🚪 EXIT – Bull Call Spread (pro Kontrakt)"
            else:
                tp_label  = f"+{exit_r['take_profit_pct']}% ROI"
                tp2_label = f"+{exit_r['full_profit_pct']}% ROI"
                sl_label  = f"−{abs(exit_r['stop_loss_pct'])}% Debit"
                sl_action = "Buy to Close · Limit GTC"
                delta_str = f"{delta_exit_v:.2f}" if delta_exit_v else "0.80"
                warn_row  = f"""
                <tr><td colspan="3" style="padding:5px 0 0;font-size:11px;color:#92400e;">
                  ⚠️ Warnschwellen: IV-Crush &gt;{vol_crush_pts} Pkte &nbsp;|&nbsp;
                  Delta &gt; {delta_str} (ITM – Roll prüfen)
                </td></tr>"""
                title = "🚪 EXIT – Long Call (pro Kontrakt)"
            exit_html = f"""
            <div style="margin-top:10px;padding:12px 14px;background:#fff7ed;border:1px dashed #ed8936;border-radius:6px;font-size:12px;">
              <b style="color:#c05621;">{title}</b>
              <table style="width:100%;margin-top:6px;font-size:12px;color:#7b341e;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #fed7aa;">
                  <td style="padding:2px 8px 2px 0;width:38%;font-size:11px;color:#92400e;"></td>
                  <td style="padding:2px 8px 2px 0;font-size:11px;color:#92400e;">Optionspreis</td>
                  <td style="padding:2px 0;font-size:11px;color:#92400e;">P&amp;L / Kontrakt</td>
                </tr>
                <tr style="border-bottom:1px solid #fed7aa;">
                  <td style="padding:4px 8px 4px 0;"><b>Entry-Kosten</b></td>
                  <td style="padding:4px 8px 4px 0;">${entry_c:.2f}</td>
                  <td style="padding:4px 0;font-weight:600;">${entry_usd:.0f}</td>
                </tr>
                <tr style="border-bottom:1px solid #fed7aa;">
                  <td style="padding:4px 8px 4px 0;"><b>Stop ({sl_label})</b><br>
                    <span style="font-size:11px;color:#dc2626;">{sl_action}</span></td>
                  <td style="padding:4px 8px 4px 0;color:#dc2626;font-weight:bold;">${sl_price:.2f}</td>
                  <td style="padding:4px 0;color:#dc2626;font-weight:bold;">{sl_pnl:+.0f}$</td>
                </tr>
                <tr style="border-bottom:1px solid #fed7aa;">
                  <td style="padding:4px 8px 4px 0;"><b>TP1 ({tp_label})</b><br>
                    <span style="font-size:11px;color:#15803d;">Hälfte schliessen</span></td>
                  <td style="padding:4px 8px 4px 0;color:#15803d;font-weight:bold;">${tp1_price:.2f}</td>
                  <td style="padding:4px 0;color:#15803d;font-weight:bold;">{tp1_pnl:+.0f}$</td>
                </tr>
                <tr style="border-bottom:1px solid #fed7aa;">
                  <td style="padding:4px 8px 4px 0;"><b>TP2 ({tp2_label})</b><br>
                    <span style="font-size:11px;color:#15803d;">Rest schliessen</span></td>
                  <td style="padding:4px 8px 4px 0;color:#15803d;font-weight:bold;">${tp2_price:.2f}</td>
                  <td style="padding:4px 0;color:#15803d;font-weight:bold;">{tp2_pnl:+.0f}$</td>
                </tr>
                <tr>
                  <td colspan="3" style="padding:4px 0;"><b>Time-Exit:</b> {exit_r['time_exit_date']} ({exit_r['time_exit_dte_remaining']} DTE Rest) — wenn Gewinn &lt; +{exit_r['time_exit_min_profit_pct']}%</td>
                </tr>
                {warn_row}
              </table>
            </div>"""

        # ── Positionsgrößen-Block (Fractional Kelly) ─────────────────────────
        sizing      = p.get("position_sizing") or {}
        sizing_html = ""
        if sizing:
            s_contracts = sizing.get("contracts", 0)
            s_note      = sizing.get("note", "")
            s_color     = "#16a34a" if s_contracts > 0 else "#dc2626"
            note_html   = f"<br><i style='color:#92400e;'>{s_note}</i>" if s_note else ""
            sizing_html = f"""
            <div style="margin-top:10px;padding:12px 14px;background:#eef2ff;border:1px solid #c7d2fe;border-radius:6px;font-size:12px;">
              <b style="color:#3730a3;">💼 Positionsgröße (¼-Kelly, Depot ${sizing.get('portfolio_usd',0):,.0f})</b>
              <table style="width:100%;margin-top:6px;font-size:12px;color:#312e81;border-collapse:collapse;">
                <tr>
                  <td style="padding:2px 8px 2px 0;"><b>Empfehlung:</b>
                    <span style="color:{s_color};font-weight:bold;">{s_contracts} Kontrakt(e)</span>
                    à ${sizing.get('cost_per_contract',0):,.0f}</td>
                  <td style="padding:2px 8px 2px 0;"><b>Budget:</b> ${sizing.get('position_usd',0):,.0f}
                    ({sizing.get('position_pct',0):.1%} Depot)</td>
                </tr>
                <tr>
                  <td style="padding:2px 8px 2px 0;"><b>Max. Risiko (bei SL):</b> ${sizing.get('max_risk_usd',0):,.0f}</td>
                  <td style="padding:2px 8px 2px 0;"><b>Voll-Kelly:</b> {sizing.get('kelly_raw',0):.2f}</td>
                </tr>
              </table>
              {note_html}
            </div>"""

        if is_spread and sl:
            long_k     = float(option.get("strike", 0))
            short_k    = float(sl.get("strike", 0))
            net_debit  = float(option.get("net_debit", 0))
            width      = round(short_k - long_k, 2) if short_k > long_k else 0
            max_profit = round((width - net_debit) * 100, 2) if width > 0 and net_debit > 0 else 0
            max_loss   = round(net_debit * 100, 2)
            bep_price  = round(long_k + net_debit, 2)
            max_roi    = round((width - net_debit) / net_debit, 4) if net_debit > 0 else 0
            execution_html = f"""
            <div style="margin-top:10px;padding:12px 14px;background:#f0fdf4;border:1px solid #86efac;border-radius:6px;font-size:12px;">
              <b style="color:#15803d;">📋 Ausführung (Tradier)</b>
              <table style="width:100%;margin-top:6px;font-size:12px;color:#14532d;border-collapse:collapse;">
                <tr>
                  <td style="padding:3px 8px 3px 0;width:40%;"><b>Buy to Open:</b></td>
                  <td style="padding:3px 0;">${long_k:.2f} CALL &nbsp;@&nbsp; ask&nbsp;<b>${option.get('ask',0):.2f}</b></td>
                </tr>
                <tr>
                  <td style="padding:3px 8px 3px 0;"><b>Sell to Open:</b></td>
                  <td style="padding:3px 0;">${short_k:.2f} CALL &nbsp;@&nbsp; bid&nbsp;<b>${sl.get('bid',0):.2f}</b></td>
                </tr>
                <tr style="border-top:1px solid #bbf7d0;">
                  <td style="padding:5px 8px 3px 0;"><b>Net Debit:</b></td>
                  <td style="padding:5px 0;"><b>${net_debit:.2f}</b> &nbsp;·&nbsp; Max Verlust: ${max_loss:.0f}</td>
                </tr>
                <tr>
                  <td style="padding:3px 8px 3px 0;"><b>Break-even:</b></td>
                  <td style="padding:3px 0;">${bep_price:.2f} &nbsp;(+{(bep_price / long_k - 1) * 100:.1f}% über Buy-Strike)</td>
                </tr>
                <tr>
                  <td style="padding:3px 8px 3px 0;"><b>Max Gewinn:</b></td>
                  <td style="padding:3px 0;color:#15803d;font-weight:600;">${max_profit:.0f} pro Kontrakt ({max_roi:.1%} ROI auf Debit)</td>
                </tr>
              </table>
            </div>"""
            strike_cell  = f"<div><b>Long / Short:</b> ${long_k:.2f} / ${short_k:.2f} CALL</div>"
            bidask_cell  = f"<div><b>Expiry:</b> {option.get('expiry','–')} ({option.get('dte','–')}d)</div>"
        else:
            ask_lc        = float(option.get("ask", 0))
            strike_lc     = float(option.get("strike", 0))
            entry_usd_lc  = round(ask_lc * 100)
            be_underlying = round(strike_lc + ask_lc, 2)
            be_pct_lc     = round((ask_lc / strike_lc) * 100, 1) if strike_lc > 0 else 0
            if ask_lc > 0:
                execution_html = f"""
                <div style="margin-top:10px;padding:12px 14px;background:#f0fdf4;border:1px solid #86efac;border-radius:6px;font-size:12px;">
                  <b style="color:#15803d;">📋 Ausführung (Tradier)</b>
                  <table style="width:100%;margin-top:6px;font-size:12px;color:#14532d;border-collapse:collapse;">
                    <tr>
                      <td style="padding:3px 8px 3px 0;width:40%;"><b>Buy to Open:</b></td>
                      <td style="padding:3px 0;">${strike_lc:.2f} CALL &nbsp;@&nbsp; ask&nbsp;<b>${ask_lc:.2f}</b></td>
                    </tr>
                    <tr style="border-top:1px solid #bbf7d0;">
                      <td style="padding:5px 8px 3px 0;"><b>Entry-Kosten:</b></td>
                      <td style="padding:5px 0;">${ask_lc:.2f} /Share &nbsp;·&nbsp; <b>${entry_usd_lc:.0f}</b> pro Kontrakt</td>
                    </tr>
                    <tr>
                      <td style="padding:3px 8px 3px 0;"><b>Break-even:</b></td>
                      <td style="padding:3px 0;">${be_underlying:.2f} Kurs &nbsp;(+{be_pct_lc:.1f}% über Strike)</td>
                    </tr>
                    <tr>
                      <td style="padding:3px 8px 3px 0;"><b>Expiry:</b></td>
                      <td style="padding:3px 0;">{option.get('expiry','–')} &nbsp;({option.get('dte','–')} DTE)</td>
                    </tr>
                  </table>
                </div>"""
            else:
                execution_html = ""
            strike_cell = f"<div><b>Strike:</b> ${option.get('strike','–')} CALL</div>"
            bidask_cell = f"<div><b>Bid/Ask:</b> ${option.get('bid','–')} / ${option.get('ask','–')}</div>"

        cards += f"""
        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:20px;margin-bottom:24px;background:#fff;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-size:22px;font-weight:bold;color:#0f172a;">#{i} {ticker}</span>
            <span style="background:{score_color};color:#fff;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;">
              {ts_grade}&nbsp;·&nbsp;{ts_total}/100
            </span>
          </div>

          <div style="background:#f8fafc;border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:12px;color:#334155;border-left:3px solid {score_color};">
            <span style="color:#16a34a;font-weight:600;">✅ Für:</span> {best_for}<br>
            <span style="color:#dc2626;font-weight:600;">⚠️ Gegen:</span> {best_ag}
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
            <div><b>Strategie:</b> {strategy}</div>
            <div><b>Richtung:</b> {da.get('direction','–')}</div>
            <div><b>IV-Rank:</b> {p.get('iv_rank','–')}%</div>
            <div><b>Ziel:</b> ${sim.get('target_price',0):.2f}</div>
            {strike_cell}
            {bidask_cell}
            <div><b>IV %:</b> {option.get('implied_vol', 0):.1%}</div>
            <div><b>ROI netto:</b> <span style="color:#16a34a;font-weight:600;">{roi.get('roi_net',0):.1%}</span></div>
          </div>

          {execution_html}
          {sizing_html}
          {exit_html}
          {greeks_html}
          {prob_html}
        </div>"""

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f8fafc;margin:0;padding:0;">
<div style="max-width:640px;margin:30px auto;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;">
  <div style="background:#16a34a;padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:6px;">🎯</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">Adaptive Asymmetry-Scanner</div>
    <div style="color:rgba(255,255,255,0.85);font-size:16px;margin-top:4px;">Trade Empfehlung — {len(proposals)} Signal(e)</div>
    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:6px;">{today} &nbsp;·&nbsp; v8.3</div>
  </div>
  <div style="padding:24px 32px;">{cards}</div>
</div></body></html>"""


def _send_smtp(subject: str, html: str) -> None:
    sender   = os.getenv("GMAIL_SENDER", "")
    password = os.getenv("GMAIL_APP_PW", "")
    receiver = os.getenv("NOTIFY_EMAIL", sender)
    if not sender or not password:
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, receiver
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, receiver, msg.as_string())
        log.info(f"Email gesendet: {subject}")
    except Exception as e:
        log.error(f"SMTP-Fehler: {e}")
