"""Run OCHA/HDX PortWatch canary independently of ArcGIS ingestion.

python scripts/portwatch_hdx_shadow.py --all
No CSV/Observation is imported or promoted to the source health registry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Executed from the project root by GitHub Actions; also work as python script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.external.sources.portwatch_hdx_shadow import RESOURCES, check_resources  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OCHA HDX PortWatch SHADOW-only validation")
    parser.add_argument("--all", action="store_true",
                        help="Probe eight known country CSVs and the global chokepoint CSV")
    parser.add_argument("--output", default="outputs/external_data/health/portwatch_hdx_shadow.json")
    args = parser.parse_args(argv)
    keys = tuple(RESOURCES) if args.all else ("djibouti", "chokepoints")
    result = check_resources(keys=keys)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    summary = {k: {"status": v["status"], "reason": v.get("error"),
                   "latest_observation_date": v.get("latest_observation_date"),
                   "rows": v.get("rows")}
               for k, v in result["resources"].items()}
    print(json.dumps({"status": result["status"], "mode": "SHADOW_ONLY",
                      "production_fallback_enabled": False,
                      "resources": summary}, sort_keys=True))
    return 0 if result["status"] == "PASS_SHADOW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
