"""Unveränderliche Hypothesen-Verträge für Alternative Data.

Pflichtfelder (Auftrag): hypothesis_id, source_features, population, exposure,
signal, threshold, direction, horizon, baseline, primary_metric,
economic_rationale, registered_at, forward_start, promotion_criteria,
failure_condition (+ spec_hash, berechnet).

* spec_hash über alle inhaltlichen Felder. Die erste Registrierung wird
  append-only in outputs/research/hypothesis_contracts.jsonl festgehalten;
  derselbe hypothesis_id mit anderem Hash -> INVALID_MODIFIED (nie getestet).
  Änderung = neue ID.
* Das Signal läuft durch die geprüfte Signal-Sprache des Research-Labs
  (nur registrierte PIT-Features); source_features müssen im Signal vorkommen
  und registrierte Alt-Features sein.
* Die historische Prüfung macht das Research-Lab (gleiche Kette, BH über alle
  Hypothesen). Ab forward_start schreibt record_forward() je Stichtag die
  Signal-Top-Dezile (append-only) = Prospective Challenger.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from modules.alt_data.registry import ALT_FEATURES

CONFIG = Path("config/alt_hypotheses.yaml")
REGISTRY = Path("outputs/research/hypothesis_contracts.jsonl")
FORWARD_LEDGER = Path("outputs/research/alt_forward_ledger.jsonl")
REQUIRED = ("hypothesis_id", "source_features", "population", "exposure", "signal", "threshold", "direction",
            "horizon", "baseline", "primary_metric", "economic_rationale", "registered_at", "forward_start",
            "promotion_criteria", "failure_condition")
HASHED = tuple(f for f in REQUIRED if f != "registered_at")


def spec_hash(c: dict) -> str:
    return hashlib.sha256(json.dumps({k: c.get(k) for k in HASHED}, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def validate(c: dict) -> list[str]:
    from modules.research_lab import validate_expr
    errs = [f"Pflichtfeld fehlt: {f}" for f in REQUIRED if c.get(f) in (None, "", [])]
    if errs:
        return errs
    try:
        names = {n.id for n in ast.walk(validate_expr(c["signal"])) if isinstance(n, ast.Name)}
    except ValueError as e:
        return [f"Signal ungültig: {e}"]
    for f in c["source_features"]:
        if f not in ALT_FEATURES:
            errs.append(f"source_feature nicht registriert: {f}")
        elif f not in names:
            errs.append(f"source_feature {f} kommt im Signal nicht vor")
    if int(c["direction"]) not in (-1, 1):
        errs.append("direction muss +1/-1 sein")
    if int(c["horizon"]) != 20:
        errs.append("horizon: Research-Lab prüft 20 Handelstage (andere Horizonte -> eigener Vertragstyp)")
    return errs


def load(config: Path | None = None) -> list[dict]:
    config = config or CONFIG
    return (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("contracts") or [] if config.exists() else []


def register(contracts: list[dict], registry: Path | None = None, now: str | None = None) -> dict[str, dict]:
    """-> {id: {"status": VALID|INVALID|INVALID_MODIFIED, "spec_hash", "errors"}}; schreibt Erstregistrierungen."""
    registry = registry or REGISTRY
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    known = {}
    if registry.exists():
        for line in registry.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                known.setdefault(r["hypothesis_id"], r)
    out = {}
    for c in contracts:
        hid, h = c.get("hypothesis_id"), spec_hash(c)
        errs = validate(c)
        if errs:
            out[hid] = {"status": "INVALID", "spec_hash": h, "errors": errs}
            continue
        if hid in known and known[hid]["spec_hash"] != h:
            out[hid] = {"status": "INVALID_MODIFIED", "spec_hash": h, "registered_hash": known[hid]["spec_hash"],
                        "errors": ["Vertrag nach Registrierung verändert – neue hypothesis_id nötig"]}
            continue
        if hid not in known:
            registry.parent.mkdir(parents=True, exist_ok=True)
            with open(registry, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"hypothesis_id": hid, "spec_hash": h, "first_registered": now,
                                     "contract": c}, ensure_ascii=False, sort_keys=True) + "\n")
            known[hid] = {"spec_hash": h}
        out[hid] = {"status": "VALID", "spec_hash": h, "errors": []}
    return out


def lab_hypotheses(contracts: list[dict], status: dict[str, dict]) -> list[dict]:
    """Gültige Verträge -> Hypothesen im Format des Research-Labs (gleiche Prüfkette)."""
    out = []
    for c in contracts:
        st = status.get(c["hypothesis_id"], {})
        if st.get("status") != "VALID":
            continue
        out.append({"id": c["hypothesis_id"], "title": c.get("title") or c["hypothesis_id"],
                    "statement": c["economic_rationale"], "signal": c["signal"], "direction": int(c["direction"]),
                    "source": f"alt_data:{ALT_FEATURES[c['source_features'][0]]['source']}",
                    "created_at": c["registered_at"], "contract_hash": st["spec_hash"]})
    return out


def record_forward(panel: pd.DataFrame, contracts: list[dict], status: dict[str, dict],
                   ledger: Path | None = None) -> int:
    """Prospective Challenger: je gültigem Vertrag und Stichtag >= forward_start das
    Top-Dezil des Signals (append-only, je (id, Datum) nur einmal)."""
    from modules.research_lab import eval_signal
    ledger = ledger or FORWARD_LEDGER
    seen = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add((r["hypothesis_id"], r["date"]))
    n = 0
    for c in contracts:
        st = status.get(c["hypothesis_id"], {})
        if st.get("status") != "VALID":
            continue
        start = pd.Timestamp(c["forward_start"])
        sig = eval_signal(panel, c["signal"]) * int(c["direction"])
        df = panel[["date", "ticker"]].assign(s=sig)
        for d, g in df[df["date"] >= start].groupby("date"):
            key = (c["hypothesis_id"], str(pd.Timestamp(d).date()))
            g = g.dropna(subset=["s"])
            if key in seen or len(g) < 30:
                continue
            top = g[g["s"] >= g["s"].quantile(0.9)]["ticker"].tolist()
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"hypothesis_id": c["hypothesis_id"], "spec_hash": st["spec_hash"],
                                     "date": key[1], "top_decile": top, "n_cross_section": int(len(g)),
                                     "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
            seen.add(key)
            n += 1
    return n
