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

Pre-Registrierung (siehe load_registry()/evaluate_all()):
  - start_date muss ECHT NACH registered_on liegen (strikt größer), sonst
    ist der Eintrag "invalid" (kein Post-hoc-Overfitting durch nachträglich
    vordatierte Registrierung).
  - Optionales Feld `registered_at` (ISO-UTC-Zeitstempel): ist es gesetzt,
    sind Ledger-Zeilen mit einem `signal_timestamp`-Feld nur eligible, wenn
    dieser Zeitstempel ECHT NACH registered_at liegt. Zeilen ohne
    signal_timestamp fallen auf den date >= start_date-Vergleich zurück.
  - Ein "invalid" Eintrag wird NIE ausgewertet und fließt nicht in n_active
    (Bonferroni) ein.

Auswertungslogik (siehe evaluate()):
  - Nur Ledger-Zeilen mit date >= challenger.start_date (bzw. dem strengeren
    registered_at-Vergleich über signal_timestamp, siehe oben) UND deren
    Horizont-Metrik bereits vorliegt (Walk-forward: kein Blick auf Daten,
    die vor der Registrierung lagen oder deren Outcome noch nicht feststeht).
  - Pro Arm: n, mean, median, total_loss_rate (Anteil Outcomes <= -0.95).
  - Bootstrap-Konfidenzintervall (seeded, deterministisch) der Differenz
    der Mittelwerte (Challenger − Baseline) als POLICY-Bootstrap: die Arme
    überlappen sich (z.B. baseline score>=55 enthält alle challenger
    score>=61 Zeilen), daher werden nicht beide Arme unabhängig resampelt.
    Statt dessen wird pro Replikat die gesamte eligible Ledger-Population
    (mit Zurücklegen) resampelt, und darauf werden Challenger- und
    Baseline-Regel jeweils neu angewendet ("policy bootstrap"). Ist einer
    der beiden Arme in einem Replikat leer, wird das Replikat verworfen;
    werden mehr als 10% aller Replikate verworfen, ist die Datenlage zu
    dünn für ein CI und evaluate() liefert (None, None) zurück (Verdikt
    bleibt "running").
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

def _validate_challenger(c: dict) -> str | None:
    """Pre-registration guard: returns None if `c` is valid, else an invalid-reason string.

    - start_date muss ECHT NACH registered_on liegen (kein post-hoc-Overfitting
      durch eine nachträglich vordatierte/gleichgesetzte Registrierung).
    - registered_at (falls vorhanden) muss ein gültiger ISO-Zeitstempel sein.
    """
    try:
        registered_on = _parse_date(c["registered_on"])
    except Exception:
        return "registered_on fehlt oder ist ungültig"
    try:
        start_date = _parse_date(c["start_date"])
    except Exception:
        return "start_date fehlt oder ist ungültig"
    if not (start_date > registered_on):
        return "start_date muss strikt nach registered_on liegen"
    registered_at = c.get("registered_at")
    if registered_at is not None:
        try:
            datetime.fromisoformat(str(registered_at).replace("Z", "+00:00"))
        except Exception:
            return "registered_at ist kein gültiger ISO-Zeitstempel"
    return None


def load_registry(path: Path | str = REGISTRY_PATH) -> list[dict]:
    """Reads challengers.yaml and returns the list of challenger dicts. Never writes.

    Jeder Eintrag wird gegen die Pre-Registrierungs-Garantie geprüft
    (start_date strikt nach registered_on, gültiges registered_at); ein
    ungültiger Eintrag bekommt `_invalid_reason` gesetzt (None sonst).
    evaluate_all() wertet solche Einträge nie aus."""
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
        c.setdefault("registered_at", None)
        c["_invalid_reason"] = _validate_challenger(c)
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


def _bootstrap_diff_ci(chal_flags: list[bool], base_flags: list[bool], values: list[float],
                        alpha: float, n_boot: int, seed: int) -> tuple[float | None, float | None]:
    """Policy bootstrap CI of (mean(challenger) - mean(baseline)).

    Der Challenger- und der Baseline-Arm überlappen sich (der Baseline-Arm
    enthält typischerweise auch alle Challenger-Zeilen, z.B. score>=55
    umfasst score>=61). Unabhängiges Resampling beider Arme ignoriert diese
    Überlappung und liefert ein künstlich zu breites CI. Statt dessen wird
    hier die gesamte eligible Population (mit Zurücklegen) resampelt und
    Challenger-/Baseline-Zugehörigkeit (`chal_flags`/`base_flags`, pro Zeile
    vorab berechnet) sowie der Metrik-Wert (`values`) auf das Replikat
    übertragen; Mittelwertdifferenz je Replikat = mean(challenger im
    Replikat) - mean(baseline im Replikat).

    Replikate, in denen einer der beiden Arme leer ist, werden verworfen
    (gezählt). Werden mehr als 10% aller Replikate verworfen, ist die
    Datenlage zu dünn für ein belastbares CI und es wird (None, None)
    zurückgegeben.

    One-sided Perzentile bei alpha und 1-alpha, deterministischer Seed."""
    rng = random.Random(seed)
    n = len(values)
    diffs = []
    skipped = 0
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        c_vals = [values[i] for i in idxs if chal_flags[i]]
        b_vals = [values[i] for i in idxs if base_flags[i]]
        if not c_vals or not b_vals:
            skipped += 1
            continue
        diffs.append(statistics.mean(c_vals) - statistics.mean(b_vals))

    if n_boot > 0 and skipped / n_boot > 0.10:
        return None, None
    if not diffs:
        return None, None

    diffs.sort()
    n_diffs = len(diffs)
    lower_idx = max(0, min(n_diffs - 1, int(n_diffs * alpha)))
    upper_idx = max(0, min(n_diffs - 1, int(n_diffs * (1 - alpha))))
    return diffs[lower_idx], diffs[upper_idx]


def _parse_date(d) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


# ── Evaluation ────────────────────────────────────────────────────────────────

def _row_is_time_eligible(row: dict, start_date: date, registered_at: str | None) -> bool:
    """Walk-forward time gate for a single ledger row.

    Wenn `registered_at` gesetzt ist UND die Zeile ein `signal_timestamp`
    trägt, entscheidet ausschließlich der Vergleich signal_timestamp >
    registered_at (strikt). Zeilen ohne signal_timestamp (oder wenn kein
    registered_at vorregistriert wurde) fallen auf date >= start_date zurück.
    """
    ts = row.get("signal_timestamp")
    if registered_at and ts:
        try:
            return str(ts) > str(registered_at)
        except Exception:
            return False
    try:
        r_date = _parse_date(row.get("date"))
    except Exception:
        return False
    return r_date >= start_date


def evaluate(challenger: dict, rows: list[dict], today: date, n_active: int) -> dict:
    """Evaluates a single pre-registered challenger against the ledger rows.
    Never writes anything — purely computes a recommendation."""
    cid = challenger["id"]
    start_date = _parse_date(challenger["start_date"])
    registered_at = challenger.get("registered_at")
    metric = challenger["metric"]
    fallback = challenger.get("metric_fallback")
    min_n = int(challenger.get("min_n", 30))
    max_duration_days = int(challenger.get("max_duration_days", 180))

    # Walk-forward: only rows at/after start_date (or, if registered_at is
    # pre-registered and the row carries a signal_timestamp, only rows whose
    # signal_timestamp is strictly after registered_at) whose horizon has
    # elapsed (i.e. the metric — or its fallback — is already present).
    eligible = []
    for r in rows:
        if not _row_is_time_eligible(r, start_date, registered_at):
            continue
        if metric_value(r, metric, fallback) is None:
            continue
        eligible.append(r)

    baseline_rule = challenger.get("baseline_rule") or []
    rule = challenger.get("rule") or []

    # Precompute per-row membership flags + metric values once, for both the
    # arm stats below and the policy bootstrap (avoids re-filtering per
    # bootstrap replicate).
    values = [metric_value(r, metric, fallback) for r in eligible]
    base_flags = [all(_match_condition(r, cond) for cond in baseline_rule) for r in eligible]
    chal_flags = [all(_match_condition(r, cond) for cond in rule) for r in eligible]

    baseline_vals = [v for v, f in zip(values, base_flags) if f]
    challenger_vals = [v for v, f in zip(values, chal_flags) if f]

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
    lower, upper = _bootstrap_diff_ci(chal_flags, base_flags, values, alpha, N_BOOT, seed)
    result["ci_lower"] = lower
    result["ci_upper"] = upper

    if lower is None or upper is None:
        # Too many bootstrap replicates had an empty arm (>10%) — the data
        # is too thin for a reliable CI; keep the verdict "running".
        result["verdict"] = "running"
        return result

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
    Pre-registration guard: entries failing _validate_challenger() (e.g.
    start_date not strictly after registered_on) get verdict "invalid",
    are never evaluated, and do not count towards n_active.
    Never writes to challengers.yaml or config.yaml."""
    today = today or date.today()
    registry = load_registry(registry_path)
    rows = load_ledger_rows(ledger_root)

    invalid = [c for c in registry if c.get("_invalid_reason")]
    valid = [c for c in registry if not c.get("_invalid_reason")]

    active = [c for c in valid if c.get("status") == "active"]
    inactive = [c for c in valid if c.get("status") != "active"]

    active_sorted = sorted(active, key=lambda c: _parse_date(c["registered_on"]))
    to_evaluate = active_sorted[:MAX_ACTIVE]
    queued = active_sorted[MAX_ACTIVE:]

    n_active = len(to_evaluate)
    results = [evaluate(c, rows, today, n_active) for c in to_evaluate]

    for c in invalid:
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
            "verdict": "invalid",
            "invalid_reason": c.get("_invalid_reason"),
        })

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
    print("Hinweis: opt_ret-Metriken sind synthetisch (Black-Scholes, konstante IV, "
          "kein IV-Crush, fixer Spread) — kein echtes Options-P&L.")
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
