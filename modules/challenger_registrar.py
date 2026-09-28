"""
modules/challenger_registrar.py – registriert eingefrorene Challenger-
Vorschläge automatisch, sobald der Candidate Ledger reif ist.

Ablauf (täglich im Feedback-Workflow, `python -m modules.challenger_registrar`):
  1. config/challenger_proposals.yaml lesen; jede Spezifikation gegen
     config/challenger_proposals.lock.json prüfen (SHA-256 der kanonischen
     Spezifikation vom Einfrier-Tag). Abweichung -> NICHT registrieren.
  2. Ledger-Reife: erster Ledger-Tag bis heute >= activation.min_ledger_days
     UND >= activation.min_ledger_rows Zeilen. Nur Zählung -- die Regel
     selbst wird auf den Vor-Registrierungsdaten NIE ausgewertet.
  3. Noch nicht registrierte Vorschläge an challengers.yaml ANHÄNGEN mit
     registered_on = heute, registered_at = jetzt (UTC), start_date = morgen,
     status active, proposal_sha256. challenger.py wertet nur Zeilen nach
     start_date bzw. mit signal_timestamp > registered_at aus.

Ändert nie Produktion; Promotion bleibt ein menschlicher PR.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PROPOSALS_PATH = Path("config/challenger_proposals.yaml")
LOCK_PATH = Path("config/challenger_proposals.lock.json")
REGISTRY_PATH = Path("challengers.yaml")
LEDGER_DIR = Path("outputs/candidate_ledger")


def spec_sha256(proposal: dict) -> str:
    canonical = json.dumps(proposal, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ledger_coverage(ledger_dir: Path = LEDGER_DIR) -> dict:
    """Erster/letzter Ledger-Tag und Zeilenzahl (nur Zählung)."""
    dates, rows = [], 0
    for f in sorted(Path(ledger_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            if r.get("date"):
                dates.append(str(r["date"])[:10])
    return {"rows": rows, "first_date": min(dates) if dates else None,
            "last_date": max(dates) if dates else None,
            "n_dates": len(set(dates))}


def _registered_ids(registry_path: Path) -> set[str]:
    if not registry_path.exists():
        return set()
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return {c.get("id") for c in data.get("challengers") or []}


def _entry(p: dict, today: date, now: datetime, sha: str) -> dict:
    return {
        "id": p["id"],
        "hypothesis": " ".join(str(p["hypothesis"]).split()),
        "source": f"auto-registriert aus config/challenger_proposals.yaml "
                  f"({p.get('source_hypothesis', '-')}), eingefroren, spec_sha256 geprüft",
        "proposal_sha256": sha,
        "registered_on": today.isoformat(),
        "registered_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_date": (today + timedelta(days=1)).isoformat(),
        "rule": p["rule"],
        "baseline_rule": p["baseline_rule"],
        "metric": p["metric"],
        "min_n": p["min_n"],
        "min_clusters": p.get("min_clusters"),
        "horizon_days": p["horizon_days"],
        "max_duration_days": p["max_duration_days"],
        "status": "active",
    }


def run(today: date | None = None, proposals_path: Path = PROPOSALS_PATH,
        lock_path: Path = LOCK_PATH, registry_path: Path = REGISTRY_PATH,
        ledger_dir: Path = LEDGER_DIR, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    today = today or now.date()
    cfg = yaml.safe_load(Path(proposals_path).read_text(encoding="utf-8")) or {}
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8")) if Path(lock_path).exists() else {}
    act = cfg.get("activation") or {}
    cov = ledger_coverage(ledger_dir)
    report = {"coverage": cov, "registered": [], "skipped": {}}

    if not cov["first_date"]:
        report["reason"] = "Ledger leer"
        return report
    days = (today - date.fromisoformat(cov["first_date"])).days
    report["ledger_days"] = days
    if days < int(act.get("min_ledger_days", 70)) or cov["rows"] < int(act.get("min_ledger_rows", 0)):
        report["reason"] = (f"Ledger noch nicht reif: {days} Tage / {cov['rows']} Zeilen "
                            f"(benötigt {act.get('min_ledger_days')} / {act.get('min_ledger_rows')})")
        return report

    existing = _registered_ids(registry_path)
    new_entries = []
    for p in cfg.get("proposals") or []:
        pid = p.get("id")
        if pid in existing:
            report["skipped"][pid] = "bereits registriert"
            continue
        sha = spec_sha256(p)
        if lock.get(pid) != sha:
            report["skipped"][pid] = "Spezifikation seit dem Einfrieren geändert (Hash) -> nicht registriert"
            log.error(f"Challenger-Vorschlag {pid}: spec_sha256 weicht vom Lock ab -> nicht registriert")
            continue
        new_entries.append(_entry(p, today, now, sha))

    if new_entries:
        block = yaml.safe_dump(new_entries, sort_keys=False, allow_unicode=True, width=100)
        indented = "\n".join(("  " + ln) if ln else ln for ln in block.splitlines())
        text = Path(registry_path).read_text(encoding="utf-8").rstrip("\n")
        Path(registry_path).write_text(
            text + "\n\n  # ── automatisch registriert (modules/challenger_registrar.py) ──\n"
            + indented + "\n", encoding="utf-8")
        report["registered"] = [e["id"] for e in new_entries]
    return report


AUTO_PROPOSALS_PATH = Path("config/challenger_proposals_auto.yaml")
MAX_ACTIVE_AUTO = 6


def run_auto(today: date | None = None, proposals_path: Path = AUTO_PROPOSALS_PATH,
             registry_path: Path = REGISTRY_PATH, now: datetime | None = None) -> dict:
    """Registriert Vorschläge aus modules/alpha_discovery.py. Jeder Vorschlag
    trägt seinen eigenen spec_sha256 vom Entdeckungstag; stimmt er nicht mehr,
    wird nicht registriert. start_date = Folgetag (nur zukünftige Daten).
    Höchstens MAX_ACTIVE_AUTO aktive Auto-Challenger gleichzeitig."""
    now = now or datetime.now(timezone.utc)
    today = today or now.date()
    report = {"registered": [], "skipped": {}}
    if not Path(proposals_path).exists():
        return report
    cfg = yaml.safe_load(Path(proposals_path).read_text(encoding="utf-8")) or {}
    reg = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
    entries = reg.get("challengers") or []
    existing = {c.get("id") for c in entries}
    active_auto = sum(1 for c in entries if str(c.get("id", "")).startswith("auto_")
                      and c.get("status") == "active")
    new_entries = []
    for p in cfg.get("proposals") or []:
        pid = p.get("id")
        if pid in existing:
            continue
        spec = {k: v for k, v in p.items() if k != "spec_sha256"}
        if spec_sha256(spec) != p.get("spec_sha256"):
            report["skipped"][pid] = "Spezifikation seit dem Einfrieren geändert (Hash)"
            continue
        if active_auto + len(new_entries) >= MAX_ACTIVE_AUTO:
            report["skipped"][pid] = f"Deckel {MAX_ACTIVE_AUTO} aktive Auto-Challenger erreicht"
            continue
        new_entries.append(_entry(p, today, now, p["spec_sha256"]))
    if new_entries:
        block = yaml.safe_dump(new_entries, sort_keys=False, allow_unicode=True, width=100)
        indented = "\n".join(("  " + ln) if ln else ln for ln in block.splitlines())
        text = Path(registry_path).read_text(encoding="utf-8").rstrip("\n")
        Path(registry_path).write_text(
            text + "\n\n  # ── automatisch entdeckt + registriert (alpha_discovery) ──\n"
            + indented + "\n", encoding="utf-8")
        report["registered"] = [e["id"] for e in new_entries]
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps({"curated": run(), "auto": run_auto()}, indent=2, ensure_ascii=False))
