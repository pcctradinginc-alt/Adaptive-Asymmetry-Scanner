"""Ein Fehlerbudget für alle neuen Bestätigungstests (Aktionsplan 2026-11-09, Punkt 7).

Bisher hatten Challenger (challengers.yaml, alpha 0,10/(n·Looks)) und Promotion-Verträge
(config/promotion_policy.yaml, Familie 0,05) eigene Budgets neben dem globalen Alpha-Ledger
(config/alpha_ledger.yaml, 0,05). Ab LEDGER_CUTOFF erhalten NEUE Challenger, NEUE Promotion-Verträge und das
Champion-Protokoll ihr Alpha ausschließlich aus dem globalen Ledger:

  * Zuteilung genau einmal je Test (idempotent, Hash-Kette, fail closed), geschrieben von genau einem Schreiber
    (``python -m modules.ledger_registration`` im ML-Research-Workflow). Leser (Monats-/Wochenbericht,
    PromotionController) lesen nur nach – ohne Zuteilung gibt es keine Bestätigung (fail closed).
  * Innerhalb eines Tests wird die Zuteilung gleichmäßig (Bonferroni) auf die geplanten Prüfzeitpunkte verteilt.
  * Vor LEDGER_CUTOFF registrierte Challenger und Verträge bleiben unverändert (registrierte Hypothesen werden
    nicht umgeschrieben); sie werden als Altfamilie AUSSERHALB des globalen Budgets ausgewiesen. Die systemweite
    Schranke ist damit: 0,05 (Ledger) + Altfamilien bis zu deren Ende – offen dokumentiert, nie verschwiegen.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

LEDGER_CUTOFF = date(2026, 11, 9)
CHAMPION_PROTOCOL = Path("config/champion_protocol.yaml")
CHALLENGER_SUBSYSTEM = "challenger"
PROMOTION_SUBSYSTEM = "promotion_contract"
CHAMPION_SUBSYSTEM = "champion"
LEGACY_FAMILIES = {
    "challenger": {"alpha_family": 0.10, "rule": "0,10/(n_aktiv·Looks), n_aktiv zählt nur die Altfamilie",
                   "ends": "mit Ablauf der vor 2026-11-09 registrierten Challenger"},
    "promotion_contract": {"alpha_family": 0.05, "rule": "0,05/(Familiengröße·12 Looks)",
                           "ends": "mit Ablauf der vor 2026-11-09 registrierten Verträge"},
}


def spec_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _day(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def uses_ledger(registered) -> bool:
    """Ab LEDGER_CUTOFF registriert -> Alpha nur aus dem globalen Ledger."""
    d = _day(registered)
    return d is not None and d >= LEDGER_CUTOFF


def challenger_test_id(c: dict) -> str:
    return f"CHALLENGER:{c['id']}"


def challenger_spec(c: dict) -> dict:
    keys = ("id", "hypothesis", "registered_on", "registered_at", "start_date", "rule", "baseline_rule", "metric",
            "metric_fallback", "min_n", "min_clusters", "horizon_days", "max_duration_days", "planned_looks")
    return {k: c.get(k) for k in keys}


def promotion_test_id(key: str) -> str:
    return f"PROMOTION:{key}"


def champion_test_id(protocol: dict) -> str:
    return f"CHAMPION:{protocol['version']}"


def _ledger_path(path: Path | None):
    from modules import option_learner as ol
    return Path(path) if path is not None else ol.ALPHA_LEDGER


def ledger_alpha(test_id: str, path: Path | None = None) -> dict | None:
    """Bestehende Zuteilung (nur lesen, nie schreiben) oder None. Ein Ledger mit Befund (verletzte Kette) gilt als
    nicht lesbar -> None (fail closed)."""
    from modules import option_learner as ol
    try:
        st = ol.alpha_ledger_state(_ledger_path(path), persist=False)
    except Exception as e:  # noqa: BLE001 – unlesbar = keine Zuteilung
        log.warning(f"alpha_ledger nicht lesbar ({e}) – keine Bestätigung möglich")
        return None
    if st["problems"]:
        return None
    return st["allocations"].get(test_id)


def per_look_alpha(alloc: dict | None, looks: int) -> float | None:
    if not alloc:
        return None
    return float(alloc["alpha"]) / max(1, int(looks))


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def pending_registrations(registry_path: Path = Path("challengers.yaml"),
                          contracts_path: Path = Path("config/promotion_hypotheses.yaml"),
                          champion_path: Path = CHAMPION_PROTOCOL) -> list[tuple[str, str, str]]:
    """(test_id, Teilsystem, spec_hash) aller Tests, die ihr Alpha aus dem Ledger beziehen müssen – in fester,
    ergebnisunabhängiger Reihenfolge (Registrierungsdatum, dann ID)."""
    out = []
    for c in _load_yaml(registry_path).get("challengers") or []:
        if c.get("status", "active") == "active" and uses_ledger(c.get("registered_on")):
            out.append((str(c.get("registered_on")), challenger_test_id(c), CHALLENGER_SUBSYSTEM,
                        spec_hash(challenger_spec(c))))
    from modules import hypothesis_contract as hc
    for c in (_load_yaml(contracts_path).get("contracts") or _load_yaml(contracts_path).get("hypotheses") or []):
        if isinstance(c, dict) and uses_ledger(c.get("registered_at")):
            try:
                out.append((str(c.get("registered_at")), promotion_test_id(hc.key(c)), PROMOTION_SUBSYSTEM,
                            hc.spec_hash(c)))
            except Exception as e:  # noqa: BLE001 – unvollständiger Vertrag: keine Zuteilung
                log.warning(f"Promotion-Vertrag ohne gültigen Schlüssel übersprungen: {e}")
    proto = _load_yaml(champion_path)
    if proto.get("version") and uses_ledger(proto.get("registered_on")):
        out.append((str(proto["registered_on"]), champion_test_id(proto), CHAMPION_SUBSYSTEM, spec_hash(proto)))
    return [(t, s, h) for _, t, s, h in sorted(out)]


def allocate_all(now: datetime | None = None, *, path: Path | None = None, persist: bool = True,
                 **paths) -> list[dict]:
    """Einziger Schreiber: teilt jedem ausstehenden Test einmal Alpha zu (idempotent). Weicht der spec_hash einer
    bestehenden Zuteilung ab, wird NICHT neu zugeteilt (Befund: Vorregistrierung nachträglich geändert)."""
    from modules import option_learner as ol
    now = now or datetime.now(timezone.utc)
    lp = _ledger_path(path)
    out = []
    for test_id, sub, h in pending_registrations(**paths):
        old = ledger_alpha(test_id, lp)
        if old is not None:
            out.append({"test_id": test_id, "status": "ALLOCATED" if old.get("spec_hash") == h else "SPEC_CHANGED",
                        "alpha": old.get("alpha"), "class_index": old.get("class_index")})
            continue
        ev = ol.alpha_allocate(test_id, sub, h, now, path=lp, persist=persist)
        out.append({"test_id": test_id, "status": "NEW" if ev else "NO_BUDGET",
                    "alpha": (ev or {}).get("alpha"), "class_index": (ev or {}).get("class_index")})
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = allocate_all()
    print(json.dumps(res, indent=1, ensure_ascii=False))
    bad = [r for r in res if r["status"] in ("SPEC_CHANGED", "NO_BUDGET")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
