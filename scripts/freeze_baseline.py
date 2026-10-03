"""scripts/freeze_baseline.py – friert den aktuellen Stand als versionierte Baseline ein.

    python scripts/freeze_baseline.py --label post-pr91

Schreibt outputs/state/baselines/<label>.json (einmalig; existiert die Datei, wird NICHT
überschrieben) mit Commit, Hashes der produktionsrelevanten Konfiguration, der Verträge,
der Policies und der Registry-/Transition-Ketten. Spätere Evidenz kann so eindeutig einem
eingefrorenen Stand zugeordnet werden. Liest nur, ändert keine Konfiguration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ("config.yaml", "config/promotion_hypotheses.yaml", "config/promotion_policy.yaml",
         "config/promotion_approvals.yaml", "config/drift_policy.yaml", "config/source_health.yaml",
         "config/cost_policy.yaml", "config/factory_protocol.yaml", "config/model_registry.yaml",
         "outputs/intelligence/contract_registry.jsonl", "outputs/intelligence/promotion_transitions.jsonl")
OUT = Path("outputs/state/baselines")


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def manifest(root: Path = ROOT, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    files = {f: sha(root / f) for f in FILES}
    contracts = []
    try:
        sys.path.insert(0, str(root))
        from modules import hypothesis_contract as hc
        contracts = [{"key": hc.key(c), "spec_hash": hc.spec_hash(c),
                      "eligible_stage": c.get("eligible_stage", "CHAMPION_TRADE"),
                      "forward_start": c.get("forward_start")}
                     for c in hc.load(root / "config/promotion_hypotheses.yaml")]
    except Exception as e:  # noqa: BLE001 – Manifest bleibt nutzbar, Fehler wird festgehalten
        contracts = [{"error": str(e)}]
    m = {"frozen_at": now.isoformat(timespec="seconds"), "commit": commit, "working_tree_dirty": dirty,
         "files": files, "contracts": contracts}
    m["manifest_hash"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    path = ROOT / OUT / f"{a.label}.json"
    if path.exists():
        print(f"{path} existiert bereits – Baselines sind unveränderlich.")
        return 1
    m = manifest()
    if m["working_tree_dirty"]:
        print("Arbeitsbaum hat uncommittete Änderungen – Baseline nur von sauberem Commit.")
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(f"Baseline {a.label}: commit {m['commit'][:12]} · {len(m['contracts'])} Verträge · {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
