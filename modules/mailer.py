"""
modules/mailer.py – modularer Mailversand (Provider austauschbar).

Konfiguration ausschließlich über Umgebungsvariablen (keine Credentials im Repo):

    MAIL_PROVIDER   "gmail_smtp" (Default) | "smtp"
    GMAIL_SENDER, GMAIL_APP_PW      Gmail (SMTP_SSL smtp.gmail.com:465)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_STARTTLS   generischer SMTP
    MAIL_FROM       Absender (Default: GMAIL_SENDER bzw. SMTP_USER)
    MAIL_TO         Empfänger, komma-getrennt (Fallback: NOTIFY_EMAIL)

Öffentliche API: `send_mail(subject, html, text, ...) -> dict`
(status: "sent" | "dry_run" | "not_configured" | "failed").

Sicherheit: Passwörter und Klartext-Adressen erscheinen nie im Log; Empfänger
werden maskiert (a***@domain). Retry ist begrenzt (max_retries) mit
linearem Backoff; die Wartefunktion ist über `_sleep` austauschbar.
"""
from __future__ import annotations

import logging
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

# Für Tests monkeypatchbar (kein echtes Warten).
_sleep = time.sleep


def mask_address(addr: str) -> str:
    """'alice@example.com' -> 'a***@example.com' (nie die volle Adresse loggen)."""
    addr = (addr or "").strip()
    if "@" not in addr:
        return "***"
    local, _, domain = addr.rpartition("@")
    return f"{local[:1]}***@{domain}"


def _split_addresses(raw: str) -> list[str]:
    return [a.strip() for a in (raw or "").split(",") if a.strip()]


class MailConfig:
    """Aus der Umgebung gelesene Konfiguration (Werte nur im Speicher)."""

    def __init__(self, env: dict | None = None):
        e = os.environ if env is None else env
        self.provider = (e.get("MAIL_PROVIDER") or "gmail_smtp").strip().lower()
        self.gmail_sender = e.get("GMAIL_SENDER", "")
        self.gmail_pw = e.get("GMAIL_APP_PW", "")
        self.smtp_host = e.get("SMTP_HOST", "")
        self.smtp_port = e.get("SMTP_PORT", "")
        self.smtp_user = e.get("SMTP_USER", "")
        self.smtp_password = e.get("SMTP_PASSWORD", "")
        self.smtp_starttls = (e.get("SMTP_STARTTLS", "") or "").strip().lower() in ("1", "true", "yes", "on")
        default_from = self.gmail_sender if self.provider == "gmail_smtp" else self.smtp_user
        self.sender = e.get("MAIL_FROM") or default_from
        self.recipients = _split_addresses(e.get("MAIL_TO") or e.get("NOTIFY_EMAIL") or "")


class GmailSmtpProvider:
    name = "gmail_smtp"

    def __init__(self, cfg: MailConfig):
        self.cfg = cfg

    def configured(self) -> bool:
        return bool(self.cfg.gmail_sender and self.cfg.gmail_pw)

    def send(self, msg: MIMEMultipart) -> None:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(self.cfg.gmail_sender, self.cfg.gmail_pw)
            smtp.sendmail(msg["From"], self.cfg.recipients, msg.as_string())


class SmtpProvider:
    name = "smtp"

    def __init__(self, cfg: MailConfig):
        self.cfg = cfg

    def configured(self) -> bool:
        return bool(self.cfg.smtp_host)

    def send(self, msg: MIMEMultipart) -> None:
        c = self.cfg
        port = int(c.smtp_port) if str(c.smtp_port).isdigit() else (587 if c.smtp_starttls else 465)
        if c.smtp_starttls:
            with smtplib.SMTP(c.smtp_host, port, timeout=30) as smtp:
                smtp.starttls()
                if c.smtp_user:
                    smtp.login(c.smtp_user, c.smtp_password)
                smtp.sendmail(msg["From"], c.recipients, msg.as_string())
        else:
            with smtplib.SMTP_SSL(c.smtp_host, port, timeout=30) as smtp:
                if c.smtp_user:
                    smtp.login(c.smtp_user, c.smtp_password)
                smtp.sendmail(msg["From"], c.recipients, msg.as_string())


# Registry: Name -> Klasse mit configured() und send(msg)
PROVIDERS: dict[str, type] = {
    GmailSmtpProvider.name: GmailSmtpProvider,
    SmtpProvider.name: SmtpProvider,
}


def build_message(subject: str, html: str, text: str, sender: str, recipients: list[str]) -> MIMEMultipart:
    """multipart/alternative: text/plain zuerst, text/html zuletzt (bevorzugt)."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def _safe_err(exc: Exception, cfg: MailConfig) -> str:
    """Fehlertext ohne Secrets/Adressen (Klasse + bereinigte Meldung)."""
    s = f"{type(exc).__name__}: {exc}"
    for secret in (cfg.gmail_pw, cfg.smtp_password, cfg.gmail_sender, cfg.smtp_user, cfg.sender, *cfg.recipients):
        if secret:
            s = s.replace(secret, "***")
    return s[:300]


def send_mail(subject: str, html: str, text: str, *, dry_run: bool = False,
              max_retries: int = 3, backoff_s: float = 2.0, env: dict | None = None) -> dict:
    """Sendet eine HTML+Text-Mail. Wirft nie; Ergebnis als dict
    {status, attempts, error, provider, recipients (maskiert), preview?}."""
    cfg = MailConfig(env)
    result = {"status": "not_configured", "attempts": 0, "error": None,
              "provider": cfg.provider, "recipients": [mask_address(a) for a in cfg.recipients]}

    cls = PROVIDERS.get(cfg.provider)
    if cls is None:
        log.warning("Mail-Provider '%s' unbekannt (verfügbar: %s) – kein Versand",
                    cfg.provider, ", ".join(sorted(PROVIDERS)))
        result["error"] = "unknown_provider"
        return result
    provider = cls(cfg)
    if not provider.configured() or not cfg.recipients or not cfg.sender:
        log.warning("Mail nicht konfiguriert (Provider %s) – kein Versand", cfg.provider)
        return result

    msg = build_message(subject, html, text, cfg.sender, cfg.recipients)
    if dry_run:
        result["status"] = "dry_run"
        result["preview"] = {"subject": subject, "text_head": text[:500], "html_len": len(html)}
        log.info("Mail dry_run: '%s' an %s (nicht gesendet)", subject, result["recipients"])
        return result

    max_retries = max(1, int(max_retries))
    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            provider.send(msg)
            result["status"] = "sent"
            result["error"] = None
            log.info("Mail gesendet: '%s' an %s (Versuch %d)", subject, result["recipients"], attempt)
            return result
        except (smtplib.SMTPException, OSError) as e:
            result["error"] = _safe_err(e, cfg)
            log.warning("Mail-Versand Versuch %d/%d fehlgeschlagen: %s", attempt, max_retries, result["error"])
            if attempt < max_retries:
                _sleep(backoff_s * attempt)
        except Exception as e:  # unerwarteter Fehler: ebenfalls begrenzt wiederholen, aber sichtbar loggen
            result["error"] = _safe_err(e, cfg)
            log.error("Mail-Versand Versuch %d/%d unerwarteter Fehler: %s", attempt, max_retries, result["error"])
            if attempt < max_retries:
                _sleep(backoff_s * attempt)
    result["status"] = "failed"
    log.error("Mail-Versand endgültig fehlgeschlagen nach %d Versuchen", max_retries)
    return result
