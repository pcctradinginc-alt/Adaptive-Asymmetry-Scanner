"""modules/hypothesis_contract.py – einheitlicher, unveränderlicher Hypothesen-Vertrag.

Jede Hypothese, die jemals Produktionswirkung erhalten KÖNNTE, braucht einen
Vertrag (config/promotion_hypotheses.yaml). Regeln:

* Pflichtfelder (Auftrag) + falsifizierbare Aussage (population, exposure,
  signal, threshold, direction, horizon, baseline, metric, failure_condition).
* spec_hash = SHA-256 über die KOMPLETTE kanonische Spezifikation. Die erste
  Registrierung wird append-only in outputs/intelligence/contract_registry.jsonl
  festgehalten (Hash-Kette über alle Einträge -> Manipulation erkennbar).
* Nach registered_at ist die Spezifikation eingefroren: gleicher
  hypothesis_id+version mit anderem Hash -> INVALID_MODIFIED (wird nie
  bewertet, nie angewendet). Neue Schwelle/Richtung/Featuredefinition =
  neue hypothesis_id oder neue version.
* Mindest-Evidenz (Stichprobe, unabhängige Tage, Kalenderspanne) darf die
  Policy-Untergrenzen nicht unterschreiten.
* Similarity-Check gegen ALLE registrierten Verträge (auch abgelehnte/
  abgelaufene): sehr ähnlich -> nur mit begründetem `distinct_from`.
* H0 / H_alt werden deterministisch aus der Spezifikation erzeugt.
* Das Signal ist ein sicherer Ausdruck (AST-Whitelist) über benannte
  Kandidaten-/Kontextmerkmale – kein eval, keine Attribute, keine Aufrufe
  außer abs/min/max.

Dieses Modul erzeugt KEINE Hypothesen (keine neue Intelligenz) – es
normiert, friert ein und prüft.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONFIG = Path("config/promotion_hypotheses.yaml")
POLICY = Path("config/promotion_policy.yaml")
REGISTRY = Path("outputs/intelligence/contract_registry.jsonl")

REQUIRED = ("hypothesis_id", "version", "created_at", "registered_at", "created_by", "source_type",
            "research_question", "economic_rationale", "signal_definition", "direction", "universe",
            "sector_scope", "regime_scope", "features", "thresholds", "outcome_horizon", "primary_metric",
            "secondary_metrics", "baseline", "minimum_sample_size", "minimum_independent_dates",
            "minimum_calendar_span", "promotion_criteria", "demotion_criteria", "maximum_initial_influence",
            "production_class", "forward_start",
            # falsifizierbare Aussage
            "population", "exposure", "failure_condition")
SOURCE_TYPES = ("blind_spot", "alpha_discovery", "regime_failure", "model_drift", "feature_drift",
                "active_learning", "causal_research", "alternative_data", "historical_analogue",
                "meta_learning_failure", "unknown_unknown", "manual")
# Produktionsklasse = höchste Wirkungsart, für die der Vertrag überhaupt gedacht ist
PRODUCTION_CLASSES = ("research_only", "abstention", "rerank", "score", "weight")
INFLUENCE_LEVELS = ("NONE", "ABSTENTION_ONLY", "RERANK_ONLY", "SCORE_LIMITED", "WEIGHT_10", "WEIGHT_25")
_CLASS_MAX_LEVEL = {"research_only": "NONE", "abstention": "ABSTENTION_ONLY", "rerank": "RERANK_ONLY",
                    "score": "SCORE_LIMITED", "weight": "WEIGHT_25"}
_OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
SIMILARITY_LIMIT = 0.80
DERIVED = ("H0", "H_alt", "spec_hash")


class ContractError(ValueError):
    pass


# ── Policy ──────────────────────────────────────────────────────────────────
def load_policy(path: Path | None = None) -> dict:
    p = path or POLICY
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


# ── Hash ────────────────────────────────────────────────────────────────────
def canonical(c: dict) -> str:
    return json.dumps({k: v for k, v in c.items() if k not in DERIVED}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def spec_hash(c: dict) -> str:
    return hashlib.sha256(canonical(c).encode()).hexdigest()


def key(c: dict) -> str:
    return f"{c.get('hypothesis_id')}@v{c.get('version')}"


# ── Signal-Ausdruck (sicher) ────────────────────────────────────────────────
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant, ast.Compare,
                  ast.BoolOp, ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
                  ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Call, ast.IfExp)
_FUNCS = {"abs": abs, "min": min, "max": max}


def parse_signal(expr: str) -> ast.Expression:
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as e:
        raise ContractError(f"Signal nicht parsebar: {e}") from e
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED_NODES):
            raise ContractError(f"Signal: nicht erlaubtes Element {type(n).__name__}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in _FUNCS):
            raise ContractError("Signal: nur abs/min/max erlaubt")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float, bool, str)):
            raise ContractError("Signal: nur Zahl-/Bool-/String-Konstanten")
    return tree


def signal_names(expr: str) -> set[str]:
    t = parse_signal(expr)
    return {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and n.id not in _FUNCS}


def _ev(node, env):
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        v = env.get(node.id)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise KeyError(node.id)          # fehlend -> Regel feuert NICHT (nie 0 als Ersatz)
        return v
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else (+v if isinstance(node.op, ast.UAdd) else (not v))
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left, env), _ev(node.right, env)
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                ast.Div: lambda: a / b if b else float("nan")}[type(node.op)]()
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, env)
            sym = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}[type(op)]
            if not _OPS[sym](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*[_ev(a, env) for a in node.args])
    raise ContractError(type(node).__name__)


def evaluate_signal(c: dict, env: dict):
    """-> numerischer Signalwert oder None (fehlende Merkmale -> None, nie 0)."""
    try:
        v = _ev(parse_signal(c["signal_definition"]), env)
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def fires(c: dict, env: dict) -> bool | None:
    """Trifft die Regel zu? thresholds = {op, value}; Richtung -1 kehrt den Vergleich NICHT um
    (die Richtung beschreibt die erwartete Wirkung, nicht die Regel). None = nicht auswertbar."""
    v = evaluate_signal(c, env)
    if v is None:
        return None
    th = c["thresholds"]
    return bool(_OPS[th["op"]](v, float(th["value"])))


def in_scope(c: dict, sector: str | None, regime: str | None) -> bool:
    """Wirkung nur im validierten Bereich: Sektor/Regime müssen BEKANNT und im Scope sein
    (Scope 'all' = global)."""
    ss, rs = c.get("sector_scope") or ["all"], c.get("regime_scope") or ["all"]
    if "all" not in ss and (sector is None or sector not in ss):
        return False
    if "all" not in rs and (regime is None or regime not in rs):
        return False
    return True


# ── H0 / H_alt ──────────────────────────────────────────────────────────────
def hypotheses_pair(c: dict) -> tuple[str, str]:
    sig, th = c.get("signal_definition"), c.get("thresholds") or {}
    rule = f"{sig} {th.get('op')} {th.get('value')}"
    h0 = (f"H0: In {c.get('population')} hat die Regel [{rule}] keinen inkrementellen Effekt auf "
          f"{c.get('primary_metric')} über {c.get('outcome_horizon')} gegenüber {c.get('baseline')}.")
    word = "besser" if int(c.get("direction", 1)) > 0 else "schlechter"
    alt_word = "schlechter" if word == "besser" else "besser"
    h_alt = (f"H_alt: Fälle mit [{rule}] schneiden {alt_word} statt {word} ab als die Baseline "
             f"(Gegenrichtung). Gewinnt H_alt, wird {key(c)} REJECTED – kein Vorzeichenwechsel, "
             f"sondern neue Hypothese mit neuer Registrierung.")
    return h0, h_alt


# ── Validierung ─────────────────────────────────────────────────────────────
def validate(c: dict, policy: dict | None = None) -> list[str]:
    policy = policy if policy is not None else load_policy()
    floors = policy.get("evidence_floors") or {}
    errs = [f"Pflichtfeld fehlt: {f}" for f in REQUIRED if c.get(f) in (None, "", [])]
    if errs:
        return errs
    if c["source_type"] not in SOURCE_TYPES:
        errs.append(f"source_type unbekannt: {c['source_type']}")
    if c["production_class"] not in PRODUCTION_CLASSES:
        errs.append(f"production_class unbekannt: {c['production_class']}")
    if int(c["direction"]) not in (-1, 1):
        errs.append("direction muss +1/-1 sein")
    try:
        names = signal_names(c["signal_definition"])
    except ContractError as e:
        return errs + [str(e)]
    feats = set(c["features"])
    if not names <= feats:
        errs.append(f"Signal nutzt nicht deklarierte Merkmale: {sorted(names - feats)}")
    th = c["thresholds"]
    if not isinstance(th, dict) or th.get("op") not in _OPS or not isinstance(th.get("value"), (int, float)):
        errs.append("thresholds muss {op, value(numerisch)} sein")
    for f, floor_key in (("minimum_sample_size", "min_observations"),
                         ("minimum_independent_dates", "min_independent_dates"),
                         ("minimum_calendar_span", "min_calendar_span_days")):
        if int(c[f]) < int(floors.get(floor_key, 0)):
            errs.append(f"{f}={c[f]} unter Policy-Untergrenze {floors.get(floor_key)}")
    mx = c["maximum_initial_influence"]
    if mx not in INFLUENCE_LEVELS:
        errs.append(f"maximum_initial_influence unbekannt: {mx}")
    elif INFLUENCE_LEVELS.index(mx) > INFLUENCE_LEVELS.index(_CLASS_MAX_LEVEL.get(c["production_class"], "NONE")):
        errs.append(f"maximum_initial_influence {mx} überschreitet Klasse {c['production_class']}")
    elif INFLUENCE_LEVELS.index(mx) > INFLUENCE_LEVELS.index("WEIGHT_10"):
        errs.append("maximum_initial_influence darf höchstens WEIGHT_10 sein (25 % erst nach weiterer Evidenz)")
    try:
        reg = datetime.fromisoformat(str(c["registered_at"]).replace("Z", "+00:00"))
        fwd = datetime.fromisoformat(str(c["forward_start"]).replace("Z", "+00:00"))
        if fwd.tzinfo is None:
            fwd = fwd.replace(tzinfo=timezone.utc)
        if reg.tzinfo is None:
            reg = reg.replace(tzinfo=timezone.utc)
        if fwd <= reg:
            errs.append("forward_start muss NACH registered_at liegen")
    except ValueError:
        errs.append("registered_at/forward_start kein ISO-Datum")
    dc = c["demotion_criteria"]
    if not isinstance(dc, dict) or not dc:
        errs.append("demotion_criteria muss vor der Promotion strukturiert festgelegt sein")
    pc = c["promotion_criteria"]
    if not isinstance(pc, dict) or "delta_expectancy_min" not in pc:
        errs.append("promotion_criteria braucht delta_expectancy_min")
    errs += validate_stage(c)
    return errs


ELIGIBLE_STAGES = ("CHAMPION_TRADE", "FINAL_MC_SURVIVOR")


def validate_stage(c: dict) -> list[str]:
    """Populations-Verträge (eligible_stage gesetzt) müssen ihre Population, den Horizont, die
    Baseline innerhalb derselben Population und Cluster-Untergrenzen explizit festlegen. Ein
    mitgespeicherter spec_hash muss zur Spezifikation passen. v1-Verträge ohne Feld: unverändert."""
    st = c.get("eligible_stage")
    if st is None:
        return []
    errs = []
    if st not in ELIGIBLE_STAGES:
        errs.append(f"eligible_stage unbekannt: {st}")
    for f in ("population_definition", "horizon_days", "baseline", "minimum_independent_event_clusters"):
        if c.get(f) in (None, "", []):
            errs.append(f"Pflichtfeld für Populations-Vertrag fehlt: {f}")
    if c.get("baseline_population") != st:
        errs.append("baseline_population muss gleich eligible_stage sein (Vergleich nur innerhalb der Population)")
    if (c.get("promotion_criteria") or {}).get("min_fired_event_clusters") is None:
        errs.append("promotion_criteria braucht min_fired_event_clusters")
    if c.get("spec_hash") and c["spec_hash"] != spec_hash(c):
        errs.append("gespeicherter spec_hash passt nicht zur Spezifikation")
    return errs


# ── Similarity (Hypothesis Recycling) ───────────────────────────────────────
def _tokens(c: dict) -> set[str]:
    try:
        names = signal_names(c.get("signal_definition", ""))
    except ContractError:
        names = set()
    th = c.get("thresholds") or {}
    toks = {f"f:{n}" for n in names} | {f"op:{th.get('op')}", f"dir:{c.get('direction')}",
                                        f"cls:{c.get('production_class')}"}
    toks |= {f"sec:{s}" for s in c.get("sector_scope") or []} | {f"reg:{r}" for r in c.get("regime_scope") or []}
    return toks


def similarity(a: dict, b: dict) -> float:
    ta, tb = _tokens(a), _tokens(b)
    j = len(ta & tb) / max(1, len(ta | tb))
    na = re.sub(r"\s+", "", str(a.get("signal_definition")))
    nb = re.sub(r"\s+", "", str(b.get("signal_definition")))
    if na == nb and int(a.get("direction", 0)) == int(b.get("direction", 0)):
        j = max(j, 0.9)
    return round(j, 3)


def similar_existing(c: dict, others: list[dict]) -> list[tuple[str, float]]:
    out = []
    for o in others:
        if o.get("hypothesis_id") == c.get("hypothesis_id"):
            continue
        s = similarity(c, o)
        if s >= SIMILARITY_LIMIT:
            out.append((key(o), s))
    return out


# ── Registry (append-only, Hash-Kette) ──────────────────────────────────────
def _entry_hash(prev: str, entry: dict) -> str:
    body = json.dumps({k: v for k, v in entry.items() if k != "entry_hash"}, sort_keys=True, ensure_ascii=False,
                      default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


def read_registry(registry: Path | None = None) -> tuple[list[dict], list[str]]:
    """-> (Einträge, Integritätsfehler). Prüft Kette UND dass gespeicherte Spec zum Hash passt."""
    registry = registry or REGISTRY
    entries, problems, prev = [], [], ""
    if not registry.exists():
        return [], []
    for i, line in enumerate(registry.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("prev_hash") != prev or _entry_hash(prev, e) != e.get("entry_hash"):
            problems.append(f"Registry-Kette gebrochen bei Zeile {i + 1} ({e.get('key')})")
        if spec_hash(e.get("contract") or {}) != e.get("spec_hash"):
            problems.append(f"Registry-Spec manipuliert: {e.get('key')}")
        prev = e.get("entry_hash", "")
        entries.append(e)
    return entries, problems


def load(config: Path | None = None) -> list[dict]:
    p = config or CONFIG
    if not p.exists():
        return []
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("contracts") or []


def register(contracts: list[dict], registry: Path | None = None, policy: dict | None = None,
             now: str | None = None) -> dict[str, dict]:
    """-> {key: {status VALID|INVALID|INVALID_MODIFIED|DUPLICATE_SIMILAR|REGISTRY_TAMPERED, spec_hash, errors}}."""
    registry = registry or REGISTRY
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries, problems = read_registry(registry)
    known = {e["key"]: e for e in entries}
    out: dict[str, dict] = {}
    if problems:   # Fail closed: manipulierte Registry -> nichts gilt als gültig
        for c in contracts:
            out[key(c)] = {"status": "REGISTRY_TAMPERED", "spec_hash": spec_hash(c), "errors": problems}
        return out
    prev = entries[-1]["entry_hash"] if entries else ""
    registered_specs = [e["contract"] for e in entries]
    for c in contracts:
        k, h = key(c), spec_hash(c)
        errs = validate(c, policy)
        if errs:
            out[k] = {"status": "INVALID", "spec_hash": h, "errors": errs}
            continue
        if k in known:
            if known[k]["spec_hash"] != h:
                out[k] = {"status": "INVALID_MODIFIED", "spec_hash": h, "registered_hash": known[k]["spec_hash"],
                          "errors": ["Spezifikation nach Registrierung verändert – neue hypothesis_id/version nötig"]}
            else:
                out[k] = {"status": "VALID", "spec_hash": h, "errors": []}
            continue
        sims = similar_existing(c, registered_specs)
        just = c.get("distinct_from") or {}
        if sims and not (isinstance(just, dict) and just.get("justification")
                         and {s[0] for s in sims} <= set(just.get("ids") or [])):
            out[k] = {"status": "DUPLICATE_SIMILAR", "spec_hash": h, "similar_to": sims,
                      "errors": ["sehr ähnlich zu bestehender Hypothese – fortführen oder distinct_from begründen"]}
            continue
        h0, halt = hypotheses_pair(c)
        entry = {"key": k, "hypothesis_id": c["hypothesis_id"], "version": c["version"], "spec_hash": h,
                 "first_registered": now, "H0": h0, "H_alt": halt, "contract": c, "prev_hash": prev}
        entry["entry_hash"] = _entry_hash(prev, entry)
        registry.parent.mkdir(parents=True, exist_ok=True)
        with open(registry, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        prev = entry["entry_hash"]
        known[k] = entry
        registered_specs.append(c)
        out[k] = {"status": "VALID", "spec_hash": h, "errors": []}
    return out


def research_inventory() -> list[dict]:
    """Bestehende Research-Hypothesen (Research-Lab-DB inkl. Alt-Data-Verträge) als Zählbasis
    für Multiple Testing. Sie sind production_class research_only: ihre Population ist das
    ML-Querschnittspanel, nicht die Champion-Trades -> keine direkte Produktionswirkung."""
    p = Path("outputs/research/hypothesis_db.json")
    if not p.exists():
        return []
    try:
        db = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for hid, r in (db.get("hypotheses") or {}).items():
        out.append({"hypothesis_id": hid, "source": r.get("source"), "status": r.get("canonical_status"),
                    "production_class": "research_only", "max_state": "HISTORICALLY_VALIDATED"
                    if r.get("canonical_status") == "ACCEPTED" else "HISTORICAL_RESEARCH"})
    return out
