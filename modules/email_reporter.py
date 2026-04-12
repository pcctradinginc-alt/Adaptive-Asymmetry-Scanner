"""
modules/email_reporter.py
Generiert eine HTML-Trading-Card-Email und versendet sie via Gmail SMTP.
Wird am Ende von pipeline.py aufgerufen, wenn Trade-Vorschläge vorhanden sind.

Benötigte GitHub Secrets:
  GMAIL_SENDER   → deine Gmail-Adresse (z.B. scanner@gmail.com)
  GMAIL_APP_PW   → Gmail App-Passwort (nicht dein normales PW!)
  NOTIFY_EMAIL   → Empfänger-Adresse
"""

import os
import smtplib
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


# ── HTML-Template ─────────────────────────────────────────────────────────────

def _badge_color(direction: str) -> tuple[str, str]:
    """Gibt (bg, text) Farben für Bullish/Bearish zurück."""
    if direction == "BULLISH":
        return "#30D158", "#000"
    return "#FF453A", "#fff"


def _gate_check(label: str, ok: bool) -> str:
    color = "#30D158" if ok else "#FF453A"
    icon  = "M4 7l2 2 4-4" if ok else "M4 4l6 6M10 4l-6 6"
    return f"""
    <td style="padding:0 12px 0 0; white-space:nowrap;">
      <span style="display:inline-flex;align-items:center;gap:5px;">
        <svg width="14" height="14" viewBox="0 0 14 14">
          <circle cx="7" cy="7" r="6" fill="{color}"/>
          <path d="{icon}" stroke="#fff" stroke-width="1.5"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span style="font-size:12px;color:#1d1d1f;">{label}</span>
      </span>
    </td>"""


def _metric(label: str, value: str, color: str = "#ffffff") -> str:
    return f"""
    <td style="text-align:center;padding:0 4px;">
      <p style="font-size:11px;color:#8e8e93;text-transform:uppercase;
                letter-spacing:0.06em;margin:0 0 3px;">{label}</p>
      <p style="font-size:20px;font-weight:600;color:{color};margin:0;">{value}</p>
    </td>"""


def _option_cell(label: str, value: str) -> str:
    return f"""
    <td style="text-align:center;padding:0 6px;">
      <p style="font-size:11px;color:#86868b;margin:0 0 2px;">{label}</p>
      <p style="font-size:15px;font-weight:600;color:#1d1d1f;margin:0;">{value}</p>
    </td>"""


def build_html(proposal: dict, today: str) -> str:
    da        = proposal.get("deep_analysis", {})
    sim       = proposal.get("simulation", {})
    feat      = proposal.get("features", {})
    option    = proposal.get("option", {}) or {}
    ticker    = proposal.get("ticker", "—")
    direction = da.get("direction", "BULLISH")
    strategy  = proposal.get("strategy", "LONG_CALL")
    score     = proposal.get("final_score", 0)
    iv_rank   = proposal.get("iv_rank", "—")

    badge_bg, badge_fg = _badge_color(direction)
    current_price = sim.get("current_price", 0)
    target_price  = sim.get("target_price", 0)
    hit_rate      = sim.get("hit_rate", 0)

    mismatch   = feat.get("mismatch", 0)
    impact     = feat.get("impact", 0)
    eps_drift  = feat.get("eps_drift", 0)
    z_score    = feat.get("z_score", 0)

    catalyst  = da.get("catalyst", "—")
    ttm       = da.get("time_to_materialization", "—")
    mispricing = da.get("mispricing_logic", "—")
    bear_case  = da.get("bear_case", "—")
    bear_sev   = da.get("bear_case_severity", 0)

    strike    = option.get("strike", 0)
    expiry    = option.get("expiry", "—")
    dte       = option.get("dte", "—")
    bid       = option.get("bid", 0)
    ask       = option.get("ask", 0)
    oi        = option.get("open_interest", 0)
    iv        = option.get("implied_vol", 0)
    spread_r  = option.get("spread_ratio", 0)

    # Gate-Status
    vix       = sim.get("vix", 18.0)
    gates_html = "".join([
        _gate_check(f"VIX {vix:.1f}", vix < 35),
        _gate_check("Earnings &gt;7d", True),
        _gate_check(f"OI {oi:,}", oi >= 100),
        _gate_check(f"Spread {spread_r:.1%}", spread_r < 0.10),
        _gate_check(f"Bear {bear_sev}/10", bear_sev <= 7),
    ])

    now_str = datetime.utcnow().strftime("%A, %d. %B %Y · %H:%M MEZ")
    mismatch_color = "#FF9F0A" if mismatch >= 5 else "#ffffff"
    hit_color      = "#30D158" if hit_rate >= 0.75 else "#FF9F0A"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asymmetry Scanner · {ticker}</title>
</head>
<body style="margin:0;padding:0;background:#f5f5f7;
             font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',
             'Helvetica Neue',Arial,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background:#f5f5f7;padding:32px 16px;">
<tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" border="0"
       style="max-width:560px;width:100%;">

  <!-- ── HEADER ── -->
  <tr><td align="center" style="padding-bottom:20px;">
    <table cellpadding="0" cellspacing="0" border="0">
      <tr>
        <td style="background:#000;border-radius:12px;padding:8px 16px;">
          <span style="color:#fff;font-size:13px;font-weight:500;
                       letter-spacing:0.02em;">
            &#x25CF;&nbsp;
            <span style="color:#30D158;">&#x2713;</span>
            &nbsp;Asymmetry Scanner · Signal detected
          </span>
        </td>
      </tr>
    </table>
    <p style="color:#86868b;font-size:12px;margin:10px 0 0;">{now_str}</p>
  </td></tr>

  <!-- ── MAIN CARD ── -->
  <tr><td style="background:#fff;border-radius:20px;overflow:hidden;
                 border:1px solid #e0e0e5;">

    <!-- Card Header (dark) -->
    <table width="100%" cellpadding="0" cellspacing="0" border="0"
           style="background:#000;padding:24px 28px 20px;">
      <tr>
        <td style="vertical-align:top;">
          <p style="color:#86868b;font-size:12px;margin:0 0 4px;
                    text-transform:uppercase;letter-spacing:0.08em;">Top Signal heute</p>
          <p style="color:#fff;font-size:36px;font-weight:700;
                    margin:0;letter-spacing:-0.02em;">{ticker}</p>
          <p style="color:#86868b;font-size:13px;margin:4px 0 0;">
            {da.get("direction","")}-Signal · {ttm}
          </p>
        </td>
        <td style="vertical-align:top;text-align:right;">
          <span style="background:{badge_bg};color:{badge_fg};font-size:12px;
                       font-weight:600;padding:4px 12px;border-radius:20px;
                       display:inline-block;">{direction}</span>
          <p style="color:#fff;font-size:22px;font-weight:600;
                    margin:8px 0 0;">${current_price:.2f}</p>
          <p style="color:#30D158;font-size:13px;margin:2px 0 0;">
            Ziel ${target_price:.2f}
          </p>
        </td>
      </tr>
    </table>

    <!-- Score Bar (near-black) -->
    <table width="100%" cellpadding="16" cellspacing="0" border="0"
           style="background:#1c1c1e;border-collapse:collapse;">
      <tr>
        {_metric("Final Score", f"{score:.4f}")}
        <td style="width:1px;background:#2c2c2e;padding:0;"></td>
        {_metric("Mismatch", f"{mismatch:.1f}", mismatch_color)}
        <td style="width:1px;background:#2c2c2e;padding:0;"></td>
        {_metric("Impact", f"{impact}/10")}
        <td style="width:1px;background:#2c2c2e;padding:0;"></td>
        {_metric("MC Hit-Rate", f"{hit_rate:.0%}", hit_color)}
      </tr>
    </table>

    <!-- Body -->
    <table width="100%" cellpadding="0" cellspacing="0" border="0"
           style="padding:24px 28px;">
      <tr><td>

        <!-- Mispricing -->
        <table width="100%" cellpadding="16" cellspacing="0" border="0"
               style="background:#f5f5f7;border-radius:12px;margin-bottom:16px;">
          <tr><td>
            <p style="font-size:11px;color:#86868b;text-transform:uppercase;
                      letter-spacing:0.06em;margin:0 0 6px;">Mispricing-Logik</p>
            <p style="font-size:14px;color:#1d1d1f;margin:0;line-height:1.6;">
              {mispricing}
            </p>
          </td></tr>
        </table>

        <!-- Catalyst + TTM -->
        <table width="100%" cellpadding="0" cellspacing="12" border="0"
               style="margin-bottom:16px;">
          <tr>
            <td width="48%" style="background:#f5f5f7;border-radius:12px;padding:14px;">
              <p style="font-size:11px;color:#86868b;text-transform:uppercase;
                        letter-spacing:0.06em;margin:0 0 4px;">Katalysator</p>
              <p style="font-size:13px;color:#1d1d1f;font-weight:500;margin:0;">
                {catalyst}
              </p>
            </td>
            <td width="4%"></td>
            <td width="48%" style="background:#f5f5f7;border-radius:12px;padding:14px;">
              <p style="font-size:11px;color:#86868b;text-transform:uppercase;
                        letter-spacing:0.06em;margin:0 0 4px;">EPS-Drift</p>
              <p style="font-size:13px;color:#1d1d1f;font-weight:500;margin:0;">
                {eps_drift:+.2%}
                &nbsp;·&nbsp;Z={z_score:.2f}
              </p>
            </td>
          </tr>
        </table>

        <!-- Options Box -->
        <table width="100%" cellpadding="16" cellspacing="0" border="0"
               style="border:1.5px solid #0071e3;border-radius:12px;
                      margin-bottom:16px;">
          <tr><td>
            <table width="100%" cellpadding="0" cellspacing="0" border="0"
                   style="margin-bottom:12px;">
              <tr>
                <td style="font-size:11px;color:#0071e3;text-transform:uppercase;
                            letter-spacing:0.06em;font-weight:600;">
                  Options-Vorschlag
                </td>
                <td align="right">
                  <span style="background:#e8f1fb;color:#0071e3;font-size:11px;
                               font-weight:600;padding:3px 10px;border-radius:20px;">
                    {strategy.replace("_", " ")}
                  </span>
                </td>
              </tr>
            </table>
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
              <tr>
                {_option_cell("Strike", f"${strike:.0f}")}
                {_option_cell("Expiry", expiry[:10] if expiry != "—" else "—")}
                {_option_cell("DTE", f"{dte}d")}
                {_option_cell("IV-Rank", f"{iv_rank:.0f}%")}
                {_option_cell("Impl. Vol", f"{iv:.0%}")}
              </tr>
            </table>
            <table width="100%" cellpadding="0" cellspacing="0" border="0"
                   style="border-top:1px solid #e0e0e5;margin-top:12px;
                          padding-top:10px;">
              <tr>
                <td style="font-size:12px;color:#86868b;">Bid / Ask</td>
                <td style="font-size:13px;font-weight:500;color:#1d1d1f;">
                  ${bid:.2f} / ${ask:.2f}
                </td>
                <td style="font-size:12px;color:#86868b;">Open Interest</td>
                <td style="font-size:13px;font-weight:500;color:#1d1d1f;">
                  {oi:,}
                </td>
                <td style="font-size:12px;color:#86868b;">Spread</td>
                <td style="font-size:13px;font-weight:500;
                           color:{'#30D158' if spread_r < 0.05 else '#FF9F0A'};">
                  {spread_r:.1%}
                </td>
              </tr>
            </table>
          </td></tr>
        </table>

        <!-- Bear Case -->
        <table width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td width="3" style="background:#FF453A;border-radius:2px;"></td>
            <td style="padding:10px 14px;background:#fff5f5;border-radius:0 8px 8px 0;">
              <p style="font-size:11px;color:#FF453A;text-transform:uppercase;
                        letter-spacing:0.06em;margin:0 0 4px;font-weight:600;">
                Bear Case · Severity {bear_sev}/10
              </p>
              <p style="font-size:13px;color:#3a1010;margin:0;line-height:1.5;">
                {bear_case}
              </p>
            </td>
          </tr>
        </table>

      </td></tr>
    </table>

    <!-- Risk Gates -->
    <table width="100%" cellpadding="14" cellspacing="0" border="0"
           style="background:#f5f5f7;">
      <tr>{gates_html}</tr>
    </table>

    <!-- Footer -->
    <table width="100%" cellpadding="16" cellspacing="0" border="0"
           style="border-top:1px solid #e0e0e5;">
      <tr><td align="center">
        <p style="font-size:11px;color:#86868b;margin:0;line-height:1.6;">
          Adaptive Asymmetry-Scanner v3.5 &nbsp;·&nbsp;
          Nur zu Informationszwecken &nbsp;·&nbsp; Keine Anlageberatung<br>
          pcctradinginc-alt/news-mirofish &nbsp;·&nbsp; GitHub Actions
        </p>
      </td></tr>
    </table>

  </td></tr>
  <!-- ── END CARD ── -->

  <!-- Disclaimer -->
  <tr><td align="center" style="padding-top:16px;">
    <p style="font-size:11px;color:#86868b;margin:0;line-height:1.6;">
      Diese E-Mail wurde automatisch generiert.
      Kein Handelsauftrag. Papier-Trading empfohlen.
    </p>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ── SMTP-Versand ──────────────────────────────────────────────────────────────

def send_email(proposals: list[dict], today: str) -> None:
    """Versendet für den Top-Vorschlag eine HTML-Trading-Card-Email."""
    sender   = os.getenv("GMAIL_SENDER", "")
    app_pw   = os.getenv("GMAIL_APP_PW", "")
    recipient = os.getenv("NOTIFY_EMAIL", sender)

    if not sender or not app_pw:
        log.warning("GMAIL_SENDER oder GMAIL_APP_PW nicht gesetzt – kein Email-Versand.")
        return

    if not proposals:
        log.info("Keine Vorschläge – kein Email-Versand.")
        return

    # Nur Top-Signal (höchster FinalScore)
    top = proposals[0]
    ticker = top.get("ticker", "Signal")

    html_body = build_html(top, today)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{ticker} · {top.get('strategy','—').replace('_', ' ')} · news-mirofish"
    msg["From"]    = f"Asymmetry Scanner <{sender}>"
    msg["To"]      = recipient

    # Plaintext-Fallback
    plain = (
        f"Asymmetry Scanner – {today}\n\n"
        f"Top Signal: {ticker}\n"
        f"Strategie:  {top.get('strategy','—')}\n"
        f"Richtung:   {top.get('deep_analysis',{}).get('direction','—')}\n"
        f"FinalScore: {top.get('final_score',0):.4f}\n"
        f"Mismatch:   {top.get('features',{}).get('mismatch',0):.2f}\n\n"
        f"Details im GitHub-Repo:\n"
        f"https://github.com/pcctradinginc-alt/news-mirofish/tree/main/outputs/daily_reports"
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, app_pw)
            smtp.sendmail(sender, recipient, msg.as_string())
        log.info(f"Email erfolgreich gesendet an {recipient} ({ticker})")
    except Exception as e:
        log.error(f"Email-Versand fehlgeschlagen: {e}")
