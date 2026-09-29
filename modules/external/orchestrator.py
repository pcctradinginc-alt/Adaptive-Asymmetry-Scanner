"""
modules/external/orchestrator.py – zieht alle registrierten, freigegebenen
Quellen ab, archiviert Roh-/Normalisierte Daten, aktualisiert Health und
schreibt ein Manifest je Run.

CLI:
    python -m modules.external.orchestrator ingest    [--family X] [--json out.json]
    python -m modules.external.orchestrator preflight [--family X] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime

from modules.external.alerts import check_pit_integrity, decide_alerts, send_alerts
from modules.external.archive import ExternalArchive
from modules.external.pit import ensure_utc, utc_now
from modules.external.registry import (
    SourceRegistry, SourceHealth, evaluate_staleness, gate_source,
)
from modules.external.sources.base import SourceStatus


def _now_iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


# Wall-Clock-Budget je Quelle. Einzel-Requests haben Timeouts, aber Quellen mit
# hunderten Requests (PortWatch-Paginierung, 45 NWS-Locations) summierten sich
# bei langsamem Upstream auf > Job-Timeout: Preflight/Ingestion 2026-09-29
# 04:00 UTC wurden nach 20 min ohne eine Log-Zeile abgebrochen, KEINE Quelle
# bekam Health, niemand sah welche hing. Jetzt: Überschreitung -> FAIL
# "timeout", die übrigen Quellen laufen weiter. Override je Quelle:
# max_fetch_seconds in der Registry.
DEFAULT_FETCH_BUDGET_S = 300
DEFAULT_PREFLIGHT_BUDGET_S = 120


class SourceTimeout(Exception):
    pass


def _call_with_deadline(fn, seconds: float):
    """fn() in einem Daemon-Thread; nach `seconds` SourceTimeout. Der Thread
    läuft ggf. weiter, sein Ergebnis wird verworfen (Archivierung passiert erst
    im Orchestrator NACH fetch -> ein verspäteter Abruf schreibt nie)."""
    import threading
    box: dict = {}

    def run():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise SourceTimeout(f"Zeitbudget {seconds:.0f}s überschritten")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_ingestion(now: datetime | None = None, families: list[str] | None = None,
                   dry_run: bool = False,
                   registry: SourceRegistry | None = None) -> dict:
    """Zieht alle enabled/registrierten Quellen mit vorhandenem Konnektor ab.
    Konnektor-Exceptions sind NIE fatal für den Gesamtlauf."""
    now = ensure_utc(now) or utc_now()
    registry = registry or SourceRegistry()
    archive = ExternalArchive(registry.archive_root)

    health_before = registry.load_health()
    health = {k: dict(v) for k, v in health_before.items()}
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    manifest_entries = []
    summary = {"run_id": run_id, "now": _now_iso(now), "sources": {}}

    for source_cfg in registry.iter_sources():
        source_id = source_cfg["source_id"]
        if families is not None and source_cfg.get("family") not in families:
            continue

        h = SourceHealth.from_dict(health.get(source_id, {"source_id": source_id}))
        h.source_id = source_id
        h.criticality = source_cfg.get("criticality", "low")
        h.auth_optional = bool(source_cfg.get("auth_optional", False))
        h.expected_cadence = source_cfg.get("expected_update_cadence")

        # Statische Quellen (z.B. NCEI-Normalwerte 1991-2020) nicht bei jedem
        # Lauf neu abrufen: min_refetch_days seit dem letzten Erfolg -> sichtbar
        # übersprungen (NCEI brach 2026-09-29 zweimal am Zeitbudget ab, obwohl
        # sich die Daten nie ändern).
        _min_refetch = source_cfg.get("min_refetch_days")
        if _min_refetch and h.last_success:
            try:
                _age = (now - ensure_utc(datetime.fromisoformat(h.last_success))).total_seconds() / 86400
            except (TypeError, ValueError):
                _age = None
            if _age is not None and _age < float(_min_refetch):
                h.status = SourceStatus.PASS.value
                h.message = (f"übersprungen: statisch, letzter Erfolg vor {_age:.1f} T "
                             f"(< min_refetch_days={_min_refetch})")
                health[source_id] = h.to_dict()
                summary["sources"][source_id] = {"status": h.status, "skipped": "min_refetch_days"}
                _log(f"[ingest] {source_id}: {h.message}")
                continue

        h.last_attempt = _now_iso(now)

        fetchable, reason = gate_source(source_cfg)
        if not fetchable:
            h.status = (SourceStatus.AUTH_MISSING.value if reason == "AUTH_MISSING"
                        else SourceStatus.DEFERRED.value if reason.startswith("status_override")
                        else SourceStatus.REVIEW_REQUIRED.value if "REVIEW_REQUIRED" in reason
                        else SourceStatus.DEFERRED.value)
            h.message = reason
            # Kein Abrufversuch -> keine Fehlerserie fortschreiben
            h.consecutive_failures = 0
            health[source_id] = h.to_dict()
            summary["sources"][source_id] = {"status": h.status, "reason": reason}
            continue

        connector = registry.build_connector(source_id)
        if connector is None:
            h.status = "NO_CONNECTOR"
            h.message = "Kein Konnektor registriert."
            health[source_id] = h.to_dict()
            summary["sources"][source_id] = {"status": h.status}
            continue

        budget = float(source_cfg.get("max_fetch_seconds") or DEFAULT_FETCH_BUDGET_S)
        _t0 = time.monotonic()
        _log(f"[ingest] {source_id}: start (Budget {budget:.0f}s)")
        try:
            result = _call_with_deadline(lambda: connector.fetch(now), budget)
            _log(f"[ingest] {source_id}: fertig in {time.monotonic() - _t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - Konnektor-Fehler ist nicht fatal
            _log(f"[ingest] {source_id}: FEHLER nach {time.monotonic() - _t0:.1f}s: {e!r}")
            h.status = SourceStatus.FAIL.value
            h.message = repr(e)
            h.consecutive_failures = int(h.consecutive_failures or 0) + 1
            health[source_id] = h.to_dict()
            summary["sources"][source_id] = {"status": h.status, "error": repr(e)}
            continue

        h.status = result.status.value if hasattr(result.status, "value") else str(result.status)
        h.message = result.message
        h.parse_failures = result.parse_failures
        if result.latest_observation_time is not None:
            h.latest_observation = _now_iso(result.latest_observation_time)
        if result.latest_release_time is not None:
            h.latest_release = _now_iso(result.latest_release_time)

        pit_errors = check_pit_integrity(result.observations)
        h.pit_integrity_failures = len(pit_errors)
        if pit_errors:
            h.message = (h.message + " | " if h.message else "") + "; ".join(pit_errors[:3])

        # Data-Quality-Gate: kein PASS nur weil keine Exception auftrat.
        try:
            from modules.external import data_quality
            _hist = archive.as_of(source_id, now) if result.observations else None
            dq = data_quality.assess(result.observations, source_cfg, now, history=_hist)
        except Exception as e:  # noqa: BLE001
            dq = {"issues": ["DQ_CHECK_FAILED"], "error": repr(e), "severe": False}
        h.dq = dq
        if dq.get("severe") and h.status == SourceStatus.PASS.value:
            h.status = SourceStatus.WARN.value
            h.message = (h.message + " | " if h.message else "") + "DQ: " + ",".join(dq["issues"])
        _log(f"[ingest] {source_id}: DQ n={dq.get('n_observations')} issues={dq.get('issues')}")

        counts = {"new": 0, "duplicate": 0, "revision": 0}
        archive_error = ""
        bytes_written = 0
        guard_blocked = None
        if not dry_run:
            try:
                policy = None
                for raw in result.raw:
                    archive.store_raw(raw, policy=policy)
                bytes_before = archive.normalized_bytes_written_estimate(source_id)
                counts = archive.store_observations(
                    result.observations,
                    max_backfill_bytes=source_cfg.get("max_backfill_bytes"))
                bytes_written = archive.normalized_bytes_written_estimate(source_id) - bytes_before
                guard_blocked = archive.last_guard_blocked.get(source_id)
            except Exception as e:  # noqa: BLE001
                archive_error = repr(e)

        if guard_blocked is not None:
            # Volumen-Guard hat für diese Quelle in diesem Run NICHTS
            # geschrieben (nie stillschweigend kürzen) -> WARN + Alert über
            # die bestehende ARCHIVE_FAILURE-Alarmierung (archive_error).
            h.status = SourceStatus.WARN.value
            h.message = (h.message + " | " if h.message else "") + "volume guard"
            archive_error = "volume guard"

        h.archive_error = archive_error
        h.revision_count = int(h.revision_count or 0) + counts["revision"]
        h.duplicate_count = int(h.duplicate_count or 0) + counts["duplicate"]
        h.rows_ingested = int(h.rows_ingested or 0) + counts["new"] + counts["revision"]
        h.bytes_downloaded = int(h.bytes_downloaded or 0) + sum(r.bytes for r in result.raw)

        if result.status == SourceStatus.PASS or (hasattr(result.status, "value")
                                                    and result.status.value == "PASS"):
            h.last_success = _now_iso(now)
            h.consecutive_failures = 0
        elif h.status in (SourceStatus.FAIL.value,):
            h.consecutive_failures = int(h.consecutive_failures or 0) + 1

        h.staleness = evaluate_staleness(result.latest_observation_time, h.expected_cadence, now,
                                         frequency=source_cfg.get("frequency"),
                                         max_age_days=source_cfg.get("max_staleness_days"))

        health[source_id] = h.to_dict()
        summary["sources"][source_id] = {
            "status": h.status, "counts": counts, "pit_errors": len(pit_errors),
            "archive_error": archive_error, "dq_issues": (h.dq or {}).get("issues", []),
        }

        if not dry_run and (result.raw or result.observations):
            manifest_entries.append({
                "source_id": source_id,
                "dataset": result.raw[0].dataset if result.raw else "",
                "fingerprint": result.raw[0].fingerprint if result.raw else "",
                "retrieved_at": _now_iso(now),
                "content_hash": result.raw[0].content_hash if result.raw else "",
                "content_type": result.raw[0].content_type if result.raw else "",
                "bytes": sum(r.bytes for r in result.raw),
                "bytes_written": bytes_written,
                "parser_version": connector.parser_version,
                "feature_version": None,
                "counts": counts,
            })

    if not dry_run:
        if manifest_entries:
            archive.write_manifest(run_id, manifest_entries, now=now)

        # Optionaler Object-Storage-Sync (external_context.archive.backend:
        # s3): Upload + Verifikation + Retention-Löschung abgeschlossener,
        # alter normalisierter Monate. Bei backend=git (Default) oder ohne
        # konfiguriertes Backend ist dies ein No-Op (nie Löschung). Fehler
        # sind NIE fatal für den Ingestion-Run -> WARN + bestehende
        # ARCHIVE_FAILURE-Alarmierung je betroffener Quelle.
        offload_result = archive.offload_normalized_months(now=now)
        summary["offload"] = {
            "enabled": offload_result.get("enabled", False),
            "backend": offload_result.get("backend"),
            "uploaded": len(offload_result.get("uploaded", [])),
            "offloaded": len(offload_result.get("offloaded", [])),
        }
        for source_id, messages in (offload_result.get("failures") or {}).items():
            message = "; ".join(messages)
            if source_id in health:
                h = SourceHealth.from_dict(health[source_id])
            else:
                h = SourceHealth(source_id=source_id)
            h.status = SourceStatus.WARN.value
            h.message = (h.message + " | " if h.message else "") + f"offload: {message}"
            h.archive_error = (h.archive_error + " | " if h.archive_error else "") + f"offload: {message}"
            health[source_id] = h.to_dict()
            if source_id not in summary["sources"]:
                summary["sources"][source_id] = {}
            summary["sources"][source_id]["status"] = h.status
            summary["sources"][source_id]["archive_error"] = h.archive_error

        registry.save_health(health)
        archive.storage_telemetry()

    alerts = decide_alerts(health_before, health)
    summary["alerts"] = alerts
    if not dry_run:
        send_alerts(alerts)
    return summary


def preflight(now: datetime | None = None, families: list[str] | None = None,
              registry: SourceRegistry | None = None) -> list[dict]:
    """Live-Check ALLER registrierten Quellen (inkl. disabled/deferred) — NIE
    archivieren, nur Status/Grund berichten."""
    now = ensure_utc(now) or utc_now()
    registry = registry or SourceRegistry()
    out = []
    for source_cfg in registry.iter_sources():
        source_id = source_cfg["source_id"]
        if families is not None and source_cfg.get("family") not in families:
            continue
        fetchable, reason = gate_source(source_cfg)
        # Lizenz-Review betrifft nur die ARCHIVIERUNG. Ein lesender Live-Check
        # (nichts wird gespeichert) ist zulässig und nötig, um Endpoint/Schema
        # zu verifizieren. Overrides (DEFERRED/REVIEW_REQUIRED für den Zugang),
        # disabled und AUTH_MISSING bleiben übersprungen.
        license_review_only = (not fetchable and reason == "license_status=REVIEW_REQUIRED"
                               and not source_cfg.get("status_override"))
        if not fetchable and not license_review_only:
            out.append({"source_id": source_id, "status": reason, "reason": reason,
                        "family": source_cfg.get("family")})
            continue
        connector = registry.build_connector(source_id)
        if connector is None:
            out.append({"source_id": source_id, "status": "NO_CONNECTOR",
                        "family": source_cfg.get("family")})
            continue
        budget = float(source_cfg.get("max_preflight_seconds") or DEFAULT_PREFLIGHT_BUDGET_S)
        _t0 = time.monotonic()
        _log(f"[preflight] {source_id}: start (Budget {budget:.0f}s)")
        try:
            report = _call_with_deadline(lambda: connector.preflight(now), budget)
        except Exception as e:  # noqa: BLE001
            report = {"source_id": source_id, "status": "FAIL", "reason": repr(e)}
        _log(f"[preflight] {source_id}: {report.get('status')} in {time.monotonic() - _t0:.1f}s")
        report["family"] = source_cfg.get("family")
        report["license_status"] = source_cfg.get("license_status", "OK")
        if license_review_only:
            report["archiving"] = "blocked_until_license_review"
        out.append(report)
    return out


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="modules.external.orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--family", action="append", default=None)
    p_ingest.add_argument("--json", default=None)
    p_ingest.add_argument("--dry-run", action="store_true")

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--family", action="append", default=None)
    p_pre.add_argument("--json", default=None)

    args = parser.parse_args()
    now = utc_now()

    if args.command == "ingest":
        result = run_ingestion(now, families=args.family, dry_run=args.dry_run)
    else:
        result = preflight(now, families=args.family)

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
