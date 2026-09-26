"""
modules/challenger.py – Pre-registered Challenger → Shadow → Walk-forward → Promote

Zweck: Diese Datei ist reine BEOBACHTUNG/AUSWERTUNG. Sie liest den
Candidate-Ledger (outputs/candidate_ledger/*.jsonl) und die Registry
(challengers.yaml) und berechnet einen deterministischen Walk-forward-
Vergleich zwischen dem Produktions-Arm (baseline_rule, i.d.R.
status == "proposed") und einem vorregistrierten Challenger-Arm (rule).

WICHTIG — HARTE GARANTIE:
  Dieses Modul verändert NIEMALS config.yaml oder challengers.yaml.
  Es gibt in diesem Modul keinen Schreibzugriff auf diese beiden Dateien —
  weder direkt noch über eine Hilfsfunktion. Eine Promotion (Übernahme
  eines Challengers in die Produktion) ist ausschließlich ein von einem
  Menschen gemergter Pull-Request, der `gates:` in config.yaml ändert.
  evaluate()/evaluate_all() liefern nur eine Empfehlung ("promote_recommended"),
  nie eine Aktion.

Auswertungslogik (siehe evaluate()):
  - Nur Ledger-Zeilen mit date >= challenger.start_date UND deren Horizont-
    Metrik bereits vorliegt (Walk-forward: kein Blick auf Daten, die vor
    der Registrierung lagen oder deren Outcome noch nicht feststeht).
  - Pro Arm: n, mean, median, total_loss_rate (Anteil Outcomes <= -0.95).
  - Bootstrap-Konfidenzintervall (seeded, deterministisch) der Differenz
    der Mittelwerte (Challenger − Baseline), mit unabhängigem Resampling
    beider Arme.
  - Signifikanzniveau alpha = 0.10 / n_active (Bonferroni über die aktiven
    Challenger); da einseitig getestet wird, werden die alpha- und
    (1-alpha)-Perzentile der Bootstrap-Differenzverteilung als CI-Grenzen
    verwendet.
  - Verdikt (deterministisch):
      "running"             wenn ein Arm n < min_n hat und noch nicht expired
      "promote_recommended" wenn CI-Untergrenze > 0 UND
                             challenger.total_loss_rate <= baseline.total_loss_rate + 0.02
      "reject"              wenn CI-Obergrenze < 0, ODER expired ohne Promotion
      sonst                 "running"
  - Hard-Guard MAX_ACTIVE = 3: sind mehr als 3 Challenger mit status=="active"
    registriert, werden nur die ersten 3 (sortiert nach registered_on)
    ausgewertet; die übrigen erscheinen als "queued" (nicht ausgewertet).
"""

from __future__ import annotations

import hashlib
import logging
import json
import random
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

REGISTRY_PATH = Path("challengers.yaml")
LEDGER_ROOT   = Path("outputs/candidate_ledger")

MAX_ACTIVE   = 3
ALPHA_BASE   = 0.10
N_BOOT       = 2000
LOSS_FLOOR   = -0.95
LOSS_MARGIN  = 0.02

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "in": lambda a, b: a in b,
}


# ── Registry / Ledger loading (read-only) ────────────────────────────────────

def load_registry(path: Path | str = REGISTRY_PATH) -> list[dict]:
    """Reads challengers.yaml and returns the list of challenger dicts. Never writes."""
    import yaml

    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    challengers = data.get("challengers") or []
    out = []
    for c in challengers:
        c = dict(c)
        c.setdefault("baseline_rule", [{"field": "status", "op": "==", "value": "proposed"}])
        c.setdefault("status", "active")
        c.setdefault("metric_fallback", None)
        out.append(c)
    return out


def load_ledger_rows(root: Path | str = LEDGER_ROOT) -> list[dict]:
    """Reads all outputs/candidate_ledger/*.jsonl rows across all months. Never writes."""
    root = Path(root)
    rows: list[dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            log.debug(f"challenger.load_ledger_rows: {path} nicht lesbar (ignoriert): {e}")
    return rows


# ── Field access / selection ─────────────────────────────────────────────────

def _get_path(row: dict, path: str):
    """Dot-path lookup, e.g. 'features.trade_score'. Returns None if missing."""
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match_condition(row: dict, cond: dict) -> bool:
    field = cond.get("field")
    op = cond.get("op")
    value = cond.get("value")
    if field is None or op not in _OPS:
        return False
    actual = _get_path(row, field)
    if actual is None:
        return False
    try:
        return bool(_OPS[op](actual, value))
    except Exception:
        return False


def select(rows: list[dict], rule: list[dict]) -> list[dict]:
    """AND-combination of a list of {field, op, value} conditions. Unknown/missing → excluded."""
    if not rule:
        return list(rows)
    return [r for r in rows if all(_match_condition(r, cond) for cond in rule)]


def metric_value(row: dict, metric: str, fallback: str | None = None):
    """Reads `metric` from a row; only an EXPLICIT `fallback` is used.
    Kein automatischer opt_ret_→ret_-Fallback: Options- und Aktienrenditen
    in einem Arm zu mischen würde die Mittelwerte verzerren."""
    val = _get_path(row, metric)
    if val is not None:
        return val
    if fallback:
        return _get_path(row, fallback)
    return None


# ── Stats ─────────────────────────────────────────────────────────────────────

def _arm_stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "total_loss_rate": None}
    losses = sum(1 for v in values if v <= LOSS_FLOOR)
    return {
        "n": n,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "total_loss_rate": losses / n,
    }


def _deterministic_seed(challenger_id: str) -> int:
    """Stable seed derived from the challenger id (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256(challenger_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _bootstrap_diff_ci(challenger_vals: list[float], baseline_vals: list[float],
                        alpha: float, n_boot: int, seed: int) -> tuple[float, float]:
    """Bootstrap CI of (mean(challenger) - mean(baseline)) with independent resampling.
    One-sided percentiles at alpha and 1-alpha."""
    rng = random.Random(seed)
    diffs = []
    nc, nb = len(challenger_vals), len(baseline_vals)
    for _ in range(n_boot):
        c_sample = [challenger_vals[rng.randrange(nc)] for _ in range(nc)]
        b_sample = [baseline_vals[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(statistics.mean(c_sample) - statistics.mean(b_sample))
    diffs.sort()
    lower_idx = max(0, min(n_boot - 1, int(n_boot * alpha)))
    upper_idx = max(0, min(n_boot - 1, int(n_boot * (1 - alpha))))
    return diffs[lower_idx], diffs[upper_idx]


def _parse_date(d) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(challenger: dict, rows: list[dict], today: date, n_active: int) -> dict:
    """Evaluates a single pre-registered challenger against the ledger rows.
    Never writes anything — purely computes a recommendation."""
    cid = challenger["id"]
    start_date = _parse_date(challenger["start_date"])
    metric = challenger["metric"]
    fallback = challenger.get("metric_fallback")
    min_n = int(challenger.get("min_n", 30))
    max_duration_days = int(challenger.get("max_duration_days", 180))

    # Walk-forward: only rows at/after start_date whose horizon has elapsed
    # (i.e. the metric — or its fallback — is already present).
    eligible = []
    for r in rows:
        try:
            r_date = _parse_date(r.get("date"))
        except Exception:
            continue
        if r_date < start_date:
            continue
        if metric_value(r, metric, fallback) is None:
            continue
        eligible.append(r)

    baseline_rows = select(eligible, challenger.get("baseline_rule") or [])
    challenger_rows = select(eligible, challenger.get("rule") or [])

    baseline_vals = [v for v in (metric_value(r, metric, fallback) for r in baseline_rows) if v is not None]
    challenger_vals = [v for v in (metric_value(r, metric, fallback) for r in challenger_rows) if v is not None]

    base_stats = _arm_stats(baseline_vals)
    chal_stats = _arm_stats(challenger_vals)

    registered_on = _parse_date(challenger.get("registered_on", challenger["start_date"]))
    expired = today > registered_on + timedelta(days=max_duration_days)

    result = {
        "id": cid,
        "hypothesis": challenger.get("hypothesis", ""),
        "metric": metric,
        "n_baseline": base_stats["n"],
        "n_challenger": chal_stats["n"],
        "mean_baseline": base_stats["mean"],
        "mean_challenger": chal_stats["mean"],
        "median_baseline": base_stats["median"],
        "median_challenger": chal_stats["median"],
        "total_loss_rate_baseline": base_stats["total_loss_rate"],
        "total_loss_rate_challenger": chal_stats["total_loss_rate"],
        "ci_lower": None,
        "ci_upper": None,
        "alpha": ALPHA_BASE / max(n_active, 1),
        "expired": expired,
        "verdict": "running",
    }

    if base_stats["n"] < min_n or chal_stats["n"] < min_n:
        result["verdict"] = "reject" if expired else "running"
        return result

    alpha = ALPHA_BASE / max(n_active, 1)
    seed = _deterministic_seed(cid)
    lower, upper = _bootstrap_diff_ci(challenger_vals, baseline_vals, alpha, N_BOOT, seed)
    result["ci_lower"] = lower
    result["ci_upper"] = upper

    promote = lower > 0 and chal_stats["total_loss_rate"] <= base_stats["total_loss_rate"] + LOSS_MARGIN
    if promote:
        result["verdict"] = "promote_recommended"
    elif upper < 0 or expired:
        result["verdict"] = "reject"
    else:
        result["verdict"] = "running"

    return result


def evaluate_all(registry_path: Path | str = REGISTRY_PATH,
                  ledger_root: Path | str = LEDGER_ROOT,
                  today: date | None = None) -> list[dict]:
    """Loads the registry + ledger and evaluates all registered challengers.
    Enforces MAX_ACTIVE = 3: only the first 3 active challengers (by
    registered_on) are evaluated; the rest are reported as 'queued'.
    Never writes to challengers.yaml or config.yaml."""
    today = today or date.today()
    registry = load_registry(registry_path)
    rows = load_ledger_rows(ledger_root)

    active = [c for c in registry if c.get("status") == "active"]
    inactive = [c for c in registry if c.get("status") != "active"]

    active_sorted = sorted(active, key=lambda c: _parse_date(c["registered_on"]))
    to_evaluate = active_sorted[:MAX_ACTIVE]
    queued = active_sorted[MAX_ACTIVE:]

    n_active = len(to_evaluate)
    results = [evaluate(c, rows, today, n_active) for c in to_evaluate]

    for c in queued:
        results.append({
            "id": c["id"],
            "hypothesis": c.get("hypothesis", ""),
            "metric": c.get("metric"),
            "n_baseline": None,
            "n_challenger": None,
            "mean_baseline": None,
            "mean_challenger": None,
            "median_baseline": None,
            "median_challenger": None,
            "total_loss_rate_baseline": None,
            "total_loss_rate_challenger": None,
            "ci_lower": None,
            "ci_upper": None,
            "alpha": None,
            "expired": False,
            "verdict": "queued",
        })

    for c in inactive:
        results.append({
            "id": c["id"],
            "hypothesis": c.get("hypothesis", ""),
            "metric": c.get("metric"),
            "n_baseline": None,
            "n_challenger": None,
            "mean_baseline": None,
            "mean_challenger": None,
            "median_baseline": None,
            "median_challenger": None,
            "total_loss_rate_baseline": None,
            "total_loss_rate_challenger": None,
            "ci_lower": None,
            "ci_upper": None,
            "alpha": None,
            "expired": False,
            "verdict": c.get("status"),
        })

    return results


def _fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{x:+.1%}" if pct else f"{x}"


def _print_table(results: list[dict]) -> None:
    header = f"{'id':<24} {'n_base':>7} {'n_chal':>7} {'mean_base':>10} {'mean_chal':>10} {'ci_lower':>9} {'ci_upper':>9} {'verdict':<20}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['id']:<24} "
            f"{str(r['n_baseline']):>7} {str(r['n_challenger']):>7} "
            f"{_fmt(r['mean_baseline']):>10} {_fmt(r['mean_challenger']):>10} "
            f"{_fmt(r['ci_lower']):>9} {_fmt(r['ci_upper']):>9} "
            f"{r['verdict']:<20}"
        )


if __name__ == "__main__":
    _print_table(evaluate_all())
