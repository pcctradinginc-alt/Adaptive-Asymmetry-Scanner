"""
modules/external/alerts.py – Alarmierung ausschließlich für definierte
Ausnahmefälle (KEINE Routine-Erfolgsmails):
  - SCHEMA_CHANGED
  - AUTH_MISSING NEU aufgetreten
  - >= 3 aufeinanderfolgende Fehlversuche
  - STALE bei hoher Kritikalität
  - Archiv-Fehler
  - PIT-Integritätsverletzung (available_at > retrieved_at, naive Datetimes;
    available_at < observation_time bei Nicht-Prognosen ist dagegen ERLAUBT)
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from modules.external.pit import Observation

SourceStatus_SCHEMA_CHANGED = "SCHEMA_CHANGED"
SourceStatus_AUTH_MISSING = "AUTH_MISSING"


def check_pit_integrity(observations: Iterable[Observation]) -> list[str]:
    """Prüft Beobachtungen auf PIT-Integritätsverletzungen. Gibt eine Liste
    von Fehlermeldungen zurück (leer = alles ok)."""
    errors = []
    for o in observations:
        for name in ("observation_time", "available_at", "retrieved_at",
                      "source_release_time", "vintage_time",
                      "forecast_issue_time", "forecast_valid_time"):
            dt = getattr(o, name)
            if isinstance(dt, datetime) and dt.tzinfo is None:
                errors.append(f"{o.identity_key()}: {name} ist naiv (keine Zeitzone)")
        if o.available_at is not None and o.retrieved_at is not None \
                and o.available_at > o.retrieved_at:
            errors.append(f"{o.identity_key()}: available_at > retrieved_at")
    return errors


def decide_alerts(health_before: dict[str, dict], health_after: dict[str, dict]) -> list[dict]:
    """Vergleicht Health-Snapshots vor/nach einem Run und liefert NUR Alarme
    für die dokumentierten Bedingungen."""
    alerts: list[dict] = []
    for source_id, after in health_after.items():
        before = health_before.get(source_id, {})

        if after.get("status") == SourceStatus_SCHEMA_CHANGED:
            alerts.append({"source_id": source_id, "type": "SCHEMA_CHANGED",
                            "message": after.get("message", "")})

        if after.get("status") == SourceStatus_AUTH_MISSING \
                and before.get("status") != SourceStatus_AUTH_MISSING:
            alerts.append({"source_id": source_id, "type": "AUTH_MISSING",
                            "message": "Auth-Env-Variable fehlt neu."})

        if int(after.get("consecutive_failures", 0) or 0) >= 3:
            alerts.append({"source_id": source_id, "type": "CONSECUTIVE_FAILURES",
                            "message": f"{after['consecutive_failures']} Fehlversuche in Folge."})

        if after.get("staleness") == "STALE" and after.get("criticality") == "high":
            alerts.append({"source_id": source_id, "type": "STALE_HIGH_CRITICALITY",
                            "message": after.get("message", "")})

        if after.get("archive_error"):
            alerts.append({"source_id": source_id, "type": "ARCHIVE_FAILURE",
                            "message": after["archive_error"]})

        if int(after.get("pit_integrity_failures", 0) or 0) > 0:
            alerts.append({"source_id": source_id, "type": "PIT_INTEGRITY_FAILURE",
                            "message": after.get("message", "")})
    return alerts


def _alerts_email_enabled() -> bool:
    try:
        from modules.config import cfg
        return bool(getattr(getattr(cfg, "external_context", None), "alerts", {}).get("email", False))
    except Exception:
        return False


def format_alert_email(alerts: list[dict]) -> tuple[str, str]:
    subject = f"Adaptive Asymmetry-Scanner – External Data Alert ({len(alerts)})"
    rows = "".join(
        f"<tr><td>{a['source_id']}</td><td>{a['type']}</td><td>{a.get('message','')}</td></tr>"
        for a in alerts
    )
    html = (
        "<html><body><h3>External-Data Alerts</h3>"
        "<table border='1' cellpadding='6'><tr><th>Source</th><th>Type</th><th>Message</th></tr>"
        f"{rows}</table></body></html>"
    )
    return subject, html


def send_alerts(alerts: list[dict]) -> bool:
    """Sendet NUR wenn alerts nicht leer UND external_context.alerts.email true."""
    if not alerts or not _alerts_email_enabled():
        return False
    subject, html = format_alert_email(alerts)
    from modules.email_reporter import _send_smtp
    _send_smtp(subject, html)
    return True
