"""
modules/email_reporter.py v9.0

Änderungen v9.0:
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
                 if p.get("trade_score", {}).get("total", 0) >= 50]
    if proposals:
        html    = _build_trade_email(proposals, today)
        subject = f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
    else:
        stats   = {**pipeline_stats, "trades": 0}
        html    = _build_status_email(stats, today)
        subject = f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"
    _send_smtp(subject, html)


def _build_status_email(stats: dict, today: str) -> str:
    vix         = stats.get("vix")
    trades      = stats.get("trades", 0)
    header_col  = "#16a34a" if trades > 0 else "#0f172a"
    status_icon = "🎯" if trades > 0 else "📊"
    status_text = "Trade Empfehlung" if trades > 0 else "Kein Trade heute"

    funnel = [
        (f"{stats.get('universe', 0)} Ticker im Universum", "📋", True),
        (f"{stats.get('candidates', 0)} nach Hard-Filter (Cap>2B, Vol>1M)", "🔍", stats.get("candidates", 0) > 0),
        (f"{stats.get('prescreened', 0)} nach Prescreening (Haiku)", "🤖", stats.get("prescreened", 0) > 0),
        (f"{stats.get('roi_precheck', 0)} nach ROI Pre-Check", "💰", stats.get("roi_precheck", 0) > 0),
        (f"{stats.get('analyzed', 0)} nach Deep Analysis (Sonnet)", "🧠", stats.get("analyzed", 0) > 0),
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
    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:6px;">{today} &nbsp;·&nbsp; VIX {vix_str} &nbsp;·&nbsp; v9.0</div>
  </div>
  <div style="padding:24px 32px;">
    <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;border:1px solid #e2e8f0;">{rows}</table>
  </div>
  <div style="padding:14px 32px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;text-align:center;">
    Adaptive Asymmetry-Scanner v9.0 &nbsp;·&nbsp; {datetime.utcnow().strftime('%H:%M UTC')}
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

        # ── v9.0 #5: Greeks-Block ─────────────────────────────────────────────
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

        # ── v9.0 #7: MC-Wahrscheinlichkeits-Block ────────────────────────────
        mc_hit_rate_pct = p.get("mc_hit_rate", 0) or 0
        cat_conf        = da.get("catalyst_confidence", None)
        dte_val         = option.get("dte", "–")

        mc_color = "#16a34a" if mc_hit_rate_pct >= 0.65 else "#ca8a04" if mc_hit_rate_pct >= 0.50 else "#dc2626"
        cat_str  = f"{cat_conf}/10" if cat_conf is not None else "–"

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
          </table>
        </div>"""

        # ── Exit-Regeln Block ─────────────────────────────────────────────────
        exit_html = ""
        if exit_r and exit_r.get("entry_cost", 0) > 0:
            exit_html = f"""
            <div style="margin-top:10px;padding:12px;background:#fff7ed;border:1px dashed #ed8936;border-radius:6px;font-size:12px;">
                <b style="color:#c05621;">🚪 EXIT-STRATEGIE (pro Kontrakt)</b><br>
                <table style="width:100%;margin-top:5px;font-size:12px;color:#7b341e;">
                    <tr><td><b>Entry:</b> ${exit_r['entry_cost']:.2f}</td><td><b>Stop-Loss:</b> <span style="color:#dc2626;">${exit_r['stop_loss_price']:.2f}</span></td></tr>
                    <tr><td><b>TP 50%:</b> ${exit_r['take_profit_price']:.2f}</td><td><b>TP 100%:</b> ${exit_r['full_profit_price']:.2f}</td></tr>
                    <tr><td colspan="2"><b>Time-Exit:</b> {exit_r['time_exit_date']} (wenn Gewinn &lt; {exit_r['time_exit_min_profit_pct']}%)</td></tr>
                </table>
            </div>"""

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
            <div><b>Strike:</b> ${option.get('strike','–')}</div>
            <div><b>Expiry:</b> {option.get('expiry','–')} ({option.get('dte','–')}d)</div>
            <div><b>Bid/Ask:</b> ${option.get('bid','–')} / ${option.get('ask','–')}</div>
            <div><b>ROI netto:</b> <span style="color:#16a34a;font-weight:600;">{roi.get('roi_net',0):.1%}</span></div>
          </div>

          {greeks_html}
          {prob_html}
          {exit_html}
        </div>"""

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f8fafc;margin:0;padding:0;">
<div style="max-width:640px;margin:30px auto;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;">
  <div style="background:#16a34a;padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:6px;">🎯</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">Adaptive Asymmetry-Scanner</div>
    <div style="color:rgba(255,255,255,0.85);font-size:16px;margin-top:4px;">Trade Empfehlung — {len(proposals)} Signal(e)</div>
    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:6px;">{today} &nbsp;·&nbsp; v9.0</div>
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
