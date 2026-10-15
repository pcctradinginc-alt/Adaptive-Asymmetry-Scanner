"""Alle Workflow-Einstiegspunkte (C5, 2026-10-15): generisch aus .github/workflows/*.yml gelesen.

1. Jeder `python -m <modul>` / `python <skript>.py`-Aufruf zeigt auf eine existierende Datei (Umbenennung fällt
   sonst erst im nächtlichen Lauf auf).
2. Schlanke Jobs (ohne `pip install -r requirements.txt`) importieren ihren Einstiegspunkt in einem Subprozess,
   in dem jedes Drittpaket aus requirements.txt blockiert ist, das der Job NICHT selbst installiert
   (Regression 2026-10-12/13: Upstream-Guard und Watchdog brachen mit ModuleNotFoundError ab).
3. Alle offiziellen Actions laufen auf Node 24 (Node-20-Majors sind abgekündigt).
Neue Jobs/Workflows sind automatisch abgedeckt.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
# pip-Name -> Import-Name (nur wo abweichend)
IMPORT_NAME = {"pyyaml": "yaml", "scikit-learn": "sklearn", "python-box": "box", "stable-baselines3":
               "stable_baselines3"}
# kleinste Major-Version mit `runs.using: node24` (geprüft an den action.yml der Tags, 2026-10-15)
NODE24_MIN = {"actions/checkout": 5, "actions/setup-python": 6, "actions/cache": 5, "actions/upload-artifact": 6,
              "actions/download-artifact": 7}
_ENTRY_RE = re.compile(r"\bpython3?\s+(?:-m\s+([\w.]+)|([\w./-]+\.py))")


def _requirements() -> list[str]:
    out = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0].lower()
        out.append(IMPORT_NAME.get(name, name.replace("-", "_")))
    return out


def _jobs():
    for wf in WORKFLOWS:
        data = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        for name, job in (data.get("jobs") or {}).items():
            yield wf.name, name, job


def _runs(job) -> list[str]:
    return [s.get("run") or "" for s in job.get("steps") or []]


def _entrypoints(job) -> list[str]:
    """Pfade relativ zum Repo (Module als Datei bzw. Paket-__init__)."""
    out = []
    for run in _runs(job):
        for mod, script in _ENTRY_RE.findall(run):
            if mod in ("pytest", "pip"):
                continue
            out.append(script or mod)
    return out


def _module_path(entry: str) -> Path | None:
    if entry.endswith(".py"):
        return ROOT / entry
    base = ROOT.joinpath(*entry.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py", base / "__main__.py"):
        if p.exists():
            return p
    return None


def _slim_installs(job) -> list[str] | None:
    """None = voller Job (requirements.txt); sonst die explizit installierten Pakete (Import-Namen)."""
    pkgs: list[str] = []
    for run in _runs(job):
        for line in run.splitlines():
            if "pip install" not in line:
                continue
            if "requirements" in line:
                return None
            for tok in line.split("pip install", 1)[1].split():
                if not tok.startswith("-"):
                    name = re.split(r"[<>=!~\[]", tok, maxsplit=1)[0].lower()
                    pkgs.append(IMPORT_NAME.get(name, name.replace("-", "_")))
    return pkgs


ALL_ENTRIES = sorted({(wf, job, e) for wf, job, j in _jobs() for e in _entrypoints(j)})
SLIM = sorted((wf, job, e, tuple(inst)) for wf, job, j in _jobs()
              if (inst := _slim_installs(j)) is not None for e in _entrypoints(j))


def test_workflows_were_parsed():
    assert len(WORKFLOWS) >= 10 and len(ALL_ENTRIES) >= 30
    assert {(wf, job) for wf, job, *_ in SLIM} >= {("scanner.yml", "guard"), ("watchdog.yml", "watchdog"),
                                                    ("options_capture.yml", "capture")}


@pytest.mark.parametrize("wf, job, entry", ALL_ENTRIES)
def test_entrypoint_exists(wf, job, entry):
    assert _module_path(entry) is not None, f"{wf}:{job} ruft {entry} auf – Datei/Modul existiert nicht"


@pytest.mark.parametrize("wf, job, entry, installed", SLIM)
def test_slim_job_entrypoint_imports_with_installed_packages_only(wf, job, entry, installed):
    blocked = sorted(set(_requirements()) - set(installed))
    path = _module_path(entry)
    code = ("import sys, runpy\n"
            f"for m in {blocked!r}:\n    sys.modules[m] = None\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"runpy.run_path({str(path)!r}, run_name='__slim_import_check__')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{wf}:{job} {entry} ohne {blocked}:\n{r.stderr[-2000:]}"


def test_official_actions_run_on_node24():
    old = []
    for wf in WORKFLOWS:
        for m in re.finditer(r"uses:\s*(actions/[\w-]+)@v(\d+)", wf.read_text(encoding="utf-8")):
            action, major = m.group(1), int(m.group(2))
            if major < NODE24_MIN.get(action, 0):
                old.append(f"{wf.name}: {action}@v{major} (Node 24 ab v{NODE24_MIN[action]})")
    assert not old, old
