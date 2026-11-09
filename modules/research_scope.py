"""Forschungsumfang: eine Bewertungsachse, eingefrorene Generatoren (Aktionsplan 2026-11-09, Punkt 7).

Befund: mindestens zwölf parallele Lern-/Research-Engines, 0 Promotions, Forschungs-Erfolgsquote 2026-Q4 0,0
(n = 100). Mehr parallele Generatoren erhöhen das Mehrfachtest-Problem, nicht die statistische Power. Daher:

  * Achse (läuft weiter): Candidate Ledger + Quote-Archiv -> Options-Lernmodell -> globaler Alpha-Ledger ->
    Challenger/Promotion/Champion-Protokoll, dazu die fehlergetriebene Forschung (Power-Gate, Ledger).
  * Eingefrorene Generatoren (config/research_scope.yaml, frozen_generators): erzeugen keine NEUEN Hypothesen
    mehr – ihre Workflow-Schritte werden übersprungen. Bereits registrierte Hypothesen werden unverändert weiter
    ausgewertet (Forward-, Bewertungs- und Ledger-Schritte bleiben aktiv).
  * Eingefrorener Code (frozen_code): kein Ausbau; die Dateien sind per SHA-256 gepinnt
    (tests/test_research_scope.py). Eine Änderung verlangt eine bewusste Änderung dieser Konfiguration im PR.
  * Gesundheitsanzeigen melden eingefrorene Generatoren als FROZEN statt als veraltet.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml

SCOPE = Path("config/research_scope.yaml")


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def load(path: Path = SCOPE) -> dict:
    return _load(str(path))


def frozen_steps(path: Path = SCOPE) -> dict[str, dict]:
    """Workflow-Schritt-ID -> Eintrag aller eingefrorenen Generatoren."""
    return {g["step"]: {"name": n, **g} for n, g in (load(path).get("frozen_generators") or {}).items()}


def health_frozen(key: str, path: Path = SCOPE) -> dict | None:
    """Eingefrorener Generator, der die Gesundheitskennzahl `key` speist (oder None)."""
    for n, g in (load(path).get("frozen_generators") or {}).items():
        if key in (g.get("health_keys") or []):
            return {"name": n, "since": load(path).get("registered_on"), "reason": g.get("reason")}
    return None


def file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_drift(path: Path = SCOPE) -> dict[str, str]:
    """Eingefrorene Dateien, deren Inhalt vom gepinnten Stand abweicht: {Pfad: Befund}."""
    out = {}
    for f, sha in sorted((load(path).get("frozen_code") or {}).items()):
        if not Path(f).exists():
            out[f] = "fehlt"
        elif file_sha(f) != sha:
            out[f] = "geändert"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--github-output", action="store_true", help="frozen_<schritt>=true nach $GITHUB_OUTPUT")
    args = ap.parse_args(argv)
    lines = [f"frozen_{step}=true" for step in sorted(frozen_steps())]
    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    drift = code_drift()
    for f, why in drift.items():
        print(f"::warning::Eingefrorener Code {why}: {f} (config/research_scope.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
