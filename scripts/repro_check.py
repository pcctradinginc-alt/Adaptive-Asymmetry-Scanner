"""Reproduzierbarkeits-Vergleich zweier Läufe (Audit F14 / Remediation P2-5).

    python scripts/repro_check.py RUN_A_DIR RUN_B_DIR [--tol 1e-9]

Vergleicht die Kernmetriken aus meta_learning.json und next_validation.json
zweier Ausgabeverzeichnisse (gleicher Commit, gleicher Panel-Snapshot, gleiche
Seeds). Exit 0 = identisch innerhalb der Toleranz, 1 = Abweichung, 2 = Eingabefehler.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        out[prefix] = obj


def core(d: Path) -> dict:
    """Nur deterministische Kernfelder (keine Zeitstempel)."""
    out: dict = {}
    meta = json.loads((d / "meta_learning.json").read_text())
    _flatten("meta.panel_hash", meta.get("panel_hash"), out)
    _flatten("meta.decision", (meta.get("decision") or {}).get("verdict"), out)
    for name, a in (meta.get("approaches") or {}).items():
        _flatten(f"meta.{name}", a.get("metrics"), out)
    nv = d / "next_validation.json"
    if nv.exists():
        n = json.loads(nv.read_text())
        _flatten("next.decision", n.get("decision"), out)
        for k, m in (n.get("metrics") or {}).items():
            _flatten(f"next.{k}", {kk: vv for kk, vv in m.items() if not kk.startswith("_")}, out)
        _flatten("next.verdicts", n.get("component_verdicts"), out)
    return out


def compare(a: dict, b: dict, tol: float) -> list[str]:
    diffs = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not (math.isclose(x, y, rel_tol=tol, abs_tol=tol) or (x != x and y != y)):
                diffs.append(f"{k}: {x} != {y}")
        elif x != y:
            diffs.append(f"{k}: {x!r} != {y!r}")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    try:
        a, b = core(Path(args.a)), core(Path(args.b))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Eingabefehler: {e}")
        return 2
    diffs = compare(a, b, args.tol)
    print(f"{len(a)} Kernwerte verglichen, {len(diffs)} Abweichungen (tol={args.tol})")
    for d in diffs[:50]:
        print("  " + d)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
