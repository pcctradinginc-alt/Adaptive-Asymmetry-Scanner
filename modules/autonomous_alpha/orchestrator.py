"""Safe research-cycle orchestrator. Reuses the repository's *existing* lab.

The orchestrator invokes fixed, allowlisted modules with a shared PIT research
panel. No LLM, arbitrary shell, self-edited validation protocols or orders.
CI uses ingest-only mode after its existing full research workflow so there
is no duplicate market-data download or re-evaluation of locked holdouts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import registry as reg

STAGES = (
    ("ml_panel_and_models", ("modules.ml_research", "--mode", "full")),
    ("research_director", ("modules.research_director",)),
    ("hypothesis_factory_plan", ("modules.hypothesis_factory", "plan")),
    ("research_lab", ("modules.research_lab",)),
    ("hypothesis_factory_evaluate", ("modules.hypothesis_factory", "evaluate")),
)


def run_cycle(root: Path = Path("."), *, panel_cache: Path | None = None,
              timeout_seconds: int = 3600, runner=None) -> dict:
    root = Path(root).resolve()
    if timeout_seconds < 30:
        raise ValueError("timeout_seconds must be >= 30")
    if not (root / "config/research_protocol.yaml").exists():
        raise reg.EvidenceIntegrityError("not a research repository (protected protocol missing)")
    if panel_cache is not None and not Path(panel_cache).is_file():
        raise FileNotFoundError("provided panel cache does not exist")
    stages = []
    with tempfile.TemporaryDirectory(prefix="alpha_research_") as work:
        cache = Path(panel_cache).resolve() if panel_cache is not None else Path(work) / "panel.pkl"
        env = dict(os.environ)
        env["ML_PANEL_CACHE"] = str(cache)
        # Opt-in external DATA SNAPSHOT remains supported by the existing ML
        # command. Never load arbitrary generated Python source/LLM text.
        commands = STAGES[1:] if panel_cache is not None else STAGES
        for name, args in commands:
            command = [sys.executable, "-m", args[0], *args[1:]]
            try:
                call = runner or subprocess.run
                result = call(command, cwd=str(root), env=env,
                              timeout=timeout_seconds, capture_output=True,
                              text=True, check=False)
                details = {"stage": name, "command": command[2:],
                           "exit_code": int(result.returncode),
                           "stdout_tail": (result.stdout or "")[-1800:],
                           "stderr_tail": (result.stderr or "")[-1800:]}
            except subprocess.TimeoutExpired:
                details = {"stage": name, "command": command[2:],
                           "exit_code": 124, "error": "timeout"}
            stages.append(details)
            if details["exit_code"] != 0:
                break
        ok = len(stages) == len(commands) and all(s["exit_code"] == 0 for s in stages)
        # Ingest partial experimental evidence, but NEVER mark a failed cycle as complete.
        ingest = reg.ingest(root)
        report = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "research_only", "success": ok, "stages": stages,
            "journal": ingest, "production_effect": "NONE",
            "panel_from_existing_cache": panel_cache is not None,
        }
        reg._atomic_json(root / reg.OUTPUT / "last_run.json", report)
        return report
