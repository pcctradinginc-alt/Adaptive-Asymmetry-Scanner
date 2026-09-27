"""
modules/external/research.py – Retrospektiver Hypothesen-Generator über den
externen Real-Economy-Kontext (Road/Maritime Freight, Weather, siehe
modules/external/).

Zweck: analysiert MATURE Candidate-Ledger-Zeilen (Outcome-Horizont bereits
verstrichen), gruppiert nach externen Kontext-Buckets × (Sektor/Industrie,
Katalysator-Typ, Richtung, Relevanz-Level) und schlägt — gegen einen
vorregistrierten Mechanismus-Katalog (config/external_hypotheses.yaml,
H1–H12) — NEUE Challenger für challengers.yaml vor.

HARTE GARANTIEN:
  - Dieses Modul wird NUR aus backtest_thresholds.py / dem Hypothesen-
    Generator-Pfad von monthly_report.py aufgerufen — NIEMALS aus der
    Produktions-Pipeline (pipeline.py).
  - Es gibt in diesem Modul KEINEN Schreibzugriff auf challengers.yaml oder
    config.yaml — jede Funktion hier liefert nur Text/Daten zurück.
  - Jeder Vorschlag ist retrospektiv/in-sample (bereits gesehene Daten) und
    daher NICHT promotion-fähig; jedes generierte Snippet trägt das Label
    "[retrospektiv/in-sample, NICHT promotion-fähig]" in der Hypothese.
  - Konkurrierende Hypothesen (z.B. H1A/H1B) werden aus demselben Katalog
    unabhängig evaluiert — es wird nie nur eine Richtung eines Features
    vorgeschlagen, ohne dass die Gegenrichtung ebenfalls im Katalog geprüft
    worden wäre.
  - Budget: config external_context.research.max_external_hypotheses_per_month
    (Default 3), max. 2 Vorschläge je Feature/Monat, keine >2-Wege-
    Interaktionen (dieser Generator gruppiert immer nur nach EINEM externen
    Bucket zzgl. der Hypothesen-eigenen challenger_rule — siehe
    analyze_external_buckets()).
"""

from __future__ import annotations

import random
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HYPOTHESES_PATH = Path("config/external_hypotheses.yaml")

MIN_N                    = 30
MIN_INDEPENDENT_DATES    = 15
MIN_SPAN_DAYS            = 60
DEFAULT_MAX_PER_MONTH    = 3
MAX_THRESHOLDS_PER_FEATURE = 2
N_BOOT                   = 2000

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "in": lambda a, b: a in b,
}


def _get_path(row: dict, path: str):
    """Dot-path lookup, z.B. 'external.states.global_freight_state'."""
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match_rule(row: dict, rule: list[dict]) -> bool:
    for cond in rule or []:
        field = cond.get("field")
        op = cond.get("op")
        if field is None or op not in _OPS:
            return False
        actual = _get_path(row, field)
        if actual is None:
            return False
        try:
            if not bool(_OPS[op](actual, cond.get("value"))):
                return False
        except Exception:
            return False
    return True


# ── Katalog-Loader (rein lesend) ─────────────────────────────────────────────

def load_hypothesis_catalog(path: Path | str = DEFAULT_HYPOTHESES_PATH) -> list[dict]:
    """Lädt config/external_hypotheses.yaml. Fehlt die Datei → []. Schreibt nie."""
    path = Path(path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("hypotheses") or [])


def is_mature_row(row: dict, horizon: int = 45) -> bool:
    """Zeile ist 'mature' für `horizon`, wenn outcomes.real_strat_ret_{h}d
    (bevorzugt) oder outcomes.ret_{h}d bereits vorliegt (Outcome-Horizont
    verstrichen)."""
    outcomes = row.get("outcomes") or {}
    return f"real_strat_ret_{horizon}d" in outcomes or f"ret_{horizon}d" in outcomes


def metric_value(row: dict, metric: str):
    return _get_path(row, metric)


# ── Cluster (Handelstag) / effektive Stichprobengröße ────────────────────────

def _cluster_by_date(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get("date", "")), []).append(r)
    return out


def estimate_intra_date_rho(rows: list[dict], metric: str) -> float:
    """
    Intra-Tag-Korrelation von `metric` als One-way-ANOVA-ICC-Schätzer:
    rho = (MS_between - MS_within) / (MS_between + (n0-1) * MS_within).

    Interpretation: wie stark hängen Outcomes am selben Handelstag
    zusammen (gemeinsame Marktbewegung)? Ein negativer Schätzer hat hier
    keine sinnvolle Design-Effekt-Interpretation und wird auf 0 geclippt;
    Werte > 1 (numerisches Artefakt bei extrem kleinen Gruppen) werden auf 1
    geclippt.
    """
    clusters = _cluster_by_date(rows)
    groups = [
        [metric_value(r, metric) for r in g if isinstance(metric_value(r, metric), (int, float))]
        for g in clusters.values()
    ]
    groups = [g for g in groups if g]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return 0.0
    grand_mean = statistics.fmean(v for g in groups for v in g)
    ss_between = sum(len(g) * (statistics.fmean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum((v - statistics.fmean(g)) ** 2 for g in groups for v in g)
    df_between = k - 1
    df_within = n - k
    if df_between <= 0 or df_within <= 0:
        return 0.0
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / df_between
    denom = ms_between + (n0 - 1) * ms_within
    if n0 <= 0 or denom <= 0:
        return 0.0
    rho = (ms_between - ms_within) / denom
    return max(0.0, min(1.0, rho))


def effective_n(rows: list[dict], metric: str) -> dict:
    """
    n, independent_date_count und ein design-effekt-basiertes effective_n:
        effective_n = n / (1 + (m - 1) * rho)
    mit m = mittlere Cluster-Größe (n / Anzahl distinct Tage) und rho aus
    estimate_intra_date_rho() (auf [0, 1] geclippt). Dokumentation der
    Näherung: dies ist der klassische Design-Effekt für Cluster-Sampling
    (Kish 1965) — eine Approximation, kein exaktes Maß bei stark
    unbalancierten Clustern.
    """
    n = len(rows)
    dates = sorted(set(str(r.get("date", "")) for r in rows))
    n_dates = len(dates)
    if n == 0 or n_dates == 0:
        return {"n": 0, "independent_date_count": 0, "effective_n": 0.0, "rho": 0.0}
    m = n / n_dates
    rho = estimate_intra_date_rho(rows, metric)
    design_effect = 1 + (m - 1) * rho
    eff_n = n / design_effect if design_effect > 0 else float(n)
    return {
        "n": n,
        "independent_date_count": n_dates,
        "effective_n": round(eff_n, 2),
        "rho": round(rho, 4),
    }


def cluster_bootstrap_ci(rows: list[dict], metric: str, alpha: float = 0.05,
                          n_boot: int = N_BOOT, seed: int = 1234) -> tuple[float, float] | None:
    """
    Cluster-(Handelstag-)Bootstrap-KI des Mittelwerts von `metric` — dieselbe
    Resampling-Idee wie modules.challenger._bootstrap_diff_ci (Cluster- statt
    Zeilen-Resampling, um Intra-Tag-Korrelation zu respektieren), hier für den
    Mittelwert EINER Gruppe statt einer Differenz zweier Arme. Deterministisch
    (fixer seed). None wenn <2 distinct Cluster oder keine gültigen Werte.
    """
    values_by_cluster: dict[str, list[float]] = {}
    for r in rows:
        v = metric_value(r, metric)
        if isinstance(v, (int, float)):
            values_by_cluster.setdefault(str(r.get("date", "")), []).append(float(v))
    clusters = list(values_by_cluster.keys())
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        drawn = [rng.choice(clusters) for _ in clusters]
        vals = [v for c in drawn for v in values_by_cluster[c]]
        if vals:
            means.append(statistics.fmean(vals))
    if not means:
        return None
    means.sort()
    n_m = len(means)
    lo = max(0, min(n_m - 1, int(n_m * alpha)))
    hi = max(0, min(n_m - 1, int(n_m * (1 - alpha))))
    return means[lo], means[hi]


# ── Gruppen-Analyse ───────────────────────────────────────────────────────────

BUCKET_PATHS = {
    "freight_state":     "external.states.global_freight_state",
    "maritime_state":    "external.states.global_maritime_state",
    "external_relation": "external.relation.relation",
}

GROUP_DIMENSIONS = {
    "sector":    "external.ticker_exposure.industry",
    "catalyst":  "catalyst_type",
    "direction": "direction",
    "relevance": "external.ticker_exposure.maritime_relevance",
}


def analyze_group(rows: list[dict], metric: str) -> dict:
    """n, independent_date_count, effective_n, calendar span, Sektor-Anzahl,
    mean/median/win_rate/std, geclustertes 90%-Bootstrap-KI des Mittelwerts."""
    values = [v for r in rows if isinstance((v := metric_value(r, metric)), (int, float))]
    if not values:
        return {
            "n": 0, "independent_date_count": 0, "effective_n": 0.0, "rho": 0.0,
            "span_days": None, "sector_count": 0, "mean": None, "median": None,
            "std": None, "win_rate": None, "ci_lower": None, "ci_upper": None,
        }
    n = len(values)
    wins = sum(1 for v in values if v > 0)
    dates = sorted(set(str(r.get("date", "")) for r in rows if r.get("date")))
    span_days = None
    if len(dates) >= 2:
        try:
            span_days = (date.fromisoformat(dates[-1][:10]) - date.fromisoformat(dates[0][:10])).days
        except Exception:
            span_days = None
    eff = effective_n(rows, metric)
    ci = cluster_bootstrap_ci(rows, metric)
    sectors = {_get_path(r, GROUP_DIMENSIONS["sector"]) for r in rows} - {None}
    return {
        "n": n,
        "independent_date_count": eff["independent_date_count"],
        "effective_n": eff["effective_n"],
        "rho": eff["rho"],
        "span_days": span_days,
        "sector_count": len(sectors),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if n >= 2 else 0.0,
        "win_rate": wins / n,
        "ci_lower": ci[0] if ci else None,
        "ci_upper": ci[1] if ci else None,
    }


def mature_external_rows(rows: list[dict], horizon: int = 45) -> list[dict]:
    """Filtert auf mature Zeilen MIT externem Kontext (row['external'] gesetzt).
    Tolerant: eine Ledger ohne 'external'-Feld (anderer Agent/anderer Stand)
    liefert einfach eine leere Liste."""
    return [r for r in rows if isinstance(r, dict) and r.get("external") and is_mature_row(r, horizon)]


def analyze_external_buckets(rows: list[dict], metric: str = "outcomes.real_strat_ret_45d",
                              horizon: int = 45) -> dict:
    """
    Gruppiert mature Ledger-Zeilen nach jedem externen Bucket in BUCKET_PATHS
    sowie zusätzlich nach Sektor/Katalysator-Typ/Richtung/Relevanz-Level
    (GROUP_DIMENSIONS) und liefert je Gruppe analyze_group().
    {bucket_name: {bucket_value: stats}, ...}. Nie ein Fehler nach außen —
    fehlende/kaputte Zeilen führen höchstens zu leeren Gruppen.
    """
    mature = mature_external_rows(rows, horizon)
    out: dict[str, dict] = {}
    for name, path in {**BUCKET_PATHS, **GROUP_DIMENSIONS}.items():
        groups: dict[str, list[dict]] = {}
        for r in mature:
            val = _get_path(r, path)
            if val is None:
                continue
            groups.setdefault(str(val), []).append(r)
        out[name] = {val: analyze_group(g, metric) for val, g in groups.items()}
    return out


# ── Hypothesen-Vorschläge (challengers.yaml-Snippets, nie ein Write) ─────────

def _maturity_ok(stats: dict) -> bool:
    """Mindest-Reife: n>=30, unabhängige Tage>=15, Kalenderspanne>=60 Tage UND
    das In-Sample-KI schließt 0 nicht ein (|Effekt| mit belastbarem KI)."""
    if stats.get("n", 0) < MIN_N:
        return False
    if stats.get("independent_date_count", 0) < MIN_INDEPENDENT_DATES:
        return False
    if (stats.get("span_days") or 0) < MIN_SPAN_DAYS:
        return False
    lo, hi = stats.get("ci_lower"), stats.get("ci_upper")
    if lo is None or hi is None:
        return False
    return not (lo <= 0 <= hi)


def _research_config() -> dict:
    try:
        from modules.config import cfg
        r = getattr(getattr(cfg, "external_context", None), "research", None)
        return dict(r) if r else {}
    except Exception:
        return {}


def _yaml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_yaml_value(x) for x in v) + "]"
    return str(v)


def _rule_yaml(rule: list[dict], indent: str = "    ") -> str:
    lines = []
    for cond in rule or []:
        lines.append(f'{indent}  - field: "{cond.get("field")}"')
        lines.append(f'{indent}    op: "{cond.get("op")}"')
        lines.append(f'{indent}    value: {_yaml_value(cond.get("value"))}')
    return "\n".join(lines)


def hypothesis_snippet(hyp: dict, group_label: str, stats: dict, today: date) -> str:
    """Rendert EIN challengers.yaml-Eintrag (als Text, kein Write) für einen
    gereiften Hypothesen-Treffer. registered_on=heute, start_date=morgen
    (Walk-forward-Pflicht, siehe modules/challenger.py). Das Label
    "[retrospektiv/in-sample, NICHT promotion-fähig]" steht in der Hypothese."""
    tomorrow = today + timedelta(days=1)
    hid = str(hyp.get("id", "hX"))
    metric = hyp.get("primary_metric", "outcomes.real_strat_ret_45d")
    min_n = int(hyp.get("min_n", MIN_N))
    horizon = int(hyp.get("horizon", 45))
    max_duration = int(hyp.get("max_duration", 180))
    mean = stats.get("mean")
    ci_lo, ci_hi = stats.get("ci_lower"), stats.get("ci_upper")
    mean_str = f"{mean:+.2%}" if isinstance(mean, (int, float)) else "–"
    ci_str = (f"[{ci_lo:+.2%}, {ci_hi:+.2%}]"
              if isinstance(ci_lo, (int, float)) and isinstance(ci_hi, (int, float)) else "–")

    mechanism = (hyp.get("mechanism") or "").strip().replace("\n", " ")
    hypothesis_text = (
        f"[retrospektiv/in-sample, NICHT promotion-fähig] {mechanism} "
        f"Befund ({group_label}): n={stats.get('n')}, "
        f"unabhängige Tage={stats.get('independent_date_count')}, "
        f"effective_n={stats.get('effective_n')}, Ø={mean_str}, 90%-KI={ci_str}."
    )

    return (
        f"  - id: research_{hid.lower()}\n"
        f"    hypothesis: >\n"
        f"      {hypothesis_text}\n"
        f'    registered_on: "{today.isoformat()}"\n'
        f'    start_date: "{tomorrow.isoformat()}"\n'
        f"    rule:\n"
        f"{_rule_yaml(hyp.get('challenger_rule'))}\n"
        f"    baseline_rule:\n"
        f"{_rule_yaml(hyp.get('baseline_rule'))}\n"
        f'    metric: "{metric}"\n'
        f"    min_n: {min_n}\n"
        f"    horizon_days: {horizon}\n"
        f"    max_duration_days: {max_duration}\n"
        f"    status: active\n"
    )


def generate_hypothesis_proposals(rows: list[dict], today: date | None = None,
                                   catalog_path: Path | str = DEFAULT_HYPOTHESES_PATH) -> list[dict]:
    """
    Retrospektiver/in-sample Hypothesen-Generator über mature Candidate-
    Ledger-Zeilen mit externem Kontext, geprüft gegen den vorregistrierten
    Mechanismus-Katalog (H1–H12). NIEMALS ein Write auf challengers.yaml/
    config.yaml — liefert nur eine Liste von
    {"hypothesis_id", "group_label", "mechanism", "stats", "snippet"}.

    Budget-Gates:
      - config external_context.research.max_external_hypotheses_per_month
        (Default 3) — Gesamtzahl der Vorschläge in diesem Aufruf.
      - max. MAX_THRESHOLDS_PER_FEATURE (2) Vorschläge je `feature`-Gruppe.
      - keine >2-Wege-Interaktionen: die challenger_rule/baseline_rule jeder
        Katalog-Hypothese wird 1:1 verwendet (keine zusätzliche Verschachtelung
        durch dieses Modul).
    Reifegrad-Gate (siehe _maturity_ok): n>=30, unabhängige Tage>=15,
    Kalenderspanne>=60 Tage, In-Sample-KI schließt 0 nicht ein UND ein
    Mechanismus-Eintrag ist im Katalog vorhanden (sonst kein Vorschlag).
    Konkurrierende Hypothesen (…A/…B) werden unabhängig evaluiert — beide
    Richtungen können gleichzeitig vorgeschlagen werden.
    Wirft nie eine Exception nach außen (leere Liste bei jedem Fehler).
    """
    try:
        today = today or date.today()
        catalog = load_hypothesis_catalog(catalog_path)
        if not catalog:
            return []

        research_cfg = _research_config()
        if research_cfg.get("enable_hypothesis_generation", True) is False:
            return []
        budget = int(research_cfg.get("max_external_hypotheses_per_month", DEFAULT_MAX_PER_MONTH))
        if budget <= 0:
            return []

        proposals: list[dict] = []
        per_feature_count: dict[str, int] = {}

        for hyp in catalog:
            if len(proposals) >= budget:
                break
            if not (hyp.get("mechanism") or "").strip():
                continue  # kein Mechanismus-Eintrag im Katalog → nie vorschlagen
            feature = hyp.get("feature", hyp.get("id", ""))
            if per_feature_count.get(feature, 0) >= MAX_THRESHOLDS_PER_FEATURE:
                continue

            metric = hyp.get("primary_metric", "outcomes.real_strat_ret_45d")
            horizon = int(hyp.get("horizon", 45))
            min_n = int(hyp.get("min_n", MIN_N))

            mature = mature_external_rows(rows, horizon)
            selected = [r for r in mature if _match_rule(r, hyp.get("challenger_rule"))]
            if not selected:
                continue
            stats = analyze_group(selected, metric)
            if stats.get("n", 0) < max(MIN_N, min_n):
                continue
            if not _maturity_ok(stats):
                continue

            group_label = str(hyp.get("id", "h")).lower()
            proposals.append({
                "hypothesis_id": hyp.get("id"),
                "group_label": group_label,
                "mechanism": hyp.get("mechanism"),
                "stats": stats,
                "snippet": hypothesis_snippet(hyp, group_label, stats, today),
            })
            per_feature_count[feature] = per_feature_count.get(feature, 0) + 1

        return proposals
    except Exception:
        return []
