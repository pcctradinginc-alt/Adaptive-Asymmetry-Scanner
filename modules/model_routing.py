"""
modules/model_routing.py – evidenzbasiertes Modell-Routing (Kostenoptimierung Prio 4).

Ein günstigeres Challenger-Modell (z.B. Haiku statt Sonnet für die Deep Analysis) wird NUR aktiv,
wenn ein gepaarter A/B-Test auf identischen Prompts die vorab festgelegten Qualitätskriterien
erfüllt. Messung: je Lauf eine kleine Stichprobe wird zusätzlich mit dem jeweils anderen Modell
analysiert (im selben Message Batch, -50 %); verglichen werden die ENTSCHEIDUNGEN, die die
Pipeline aus der Analyse ableitet (Gate bestanden ja/nein, Richtung, Red-Team-Verdikt).

Kriterien (config/cost_policy.yaml model_routing.<workflow>), alle müssen gelten:
  - min_pairs Paare an min_days verschiedenen Tagen
  - Gate-Entscheidungs-Übereinstimmung >= min_gate_agreement
  - Recall: Anteil der vom Champion bestandenen Kandidaten, die auch der Challenger besteht
    >= min_pass_recall (der Challenger darf gute Kandidaten nicht verlieren)
  - Richtungs-Übereinstimmung >= min_direction_agreement
Nach der Umschaltung prüft eine Champion-Stichprobe weiter (Monitoring). Fallen die letzten
monitor_window Paare unter die Schwellen -> automatischer Rückfall auf den Champion (Demotion).
mode: observe = nur messen; auto = nach Kriterien umschalten; off = nichts.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any

from modules import cost_telemetry as ct

log = logging.getLogger(__name__)


def settings(workflow: str, pol: dict | None = None) -> dict | None:
    s = ((pol or ct.policy()).get("model_routing") or {}).get(workflow)
    if not s or str(s.get("mode", "off")) == "off" or not s.get("challenger"):
        return None
    return s


def gate_pass(result: dict | None, impact_min: int = 4, surprise_min: int = 3) -> bool:
    """Entscheidung, die die Pipeline aus einer Deep Analysis ableitet (Stufe 4/4a/4b)."""
    if not isinstance(result, dict):
        return False
    rt = result.get("red_team") or {}
    if rt.get("red_team_verdict") == "VETO":
        return False
    if result.get("direction") != "BULLISH":
        return False
    imp, sur = result.get("impact"), result.get("surprise")
    return (isinstance(imp, (int, float)) and imp >= impact_min
            and isinstance(sur, (int, float)) and sur >= surprise_min)


def max_tokens_for(workflow: str, model: str, pol: dict | None = None, default: int = 1600) -> int:
    s = ((pol or ct.policy()).get("model_routing") or {}).get(workflow) or {}
    return int((s.get("max_tokens_by_model") or {}).get(model) or s.get("max_tokens_default") or default)


def pair_row(workflow: str, ticker: str | None, reference_model: str, other_model: str,
             ref: dict | None, other: dict | None, impact_min: int = 4, surprise_min: int = 3,
             ref_stop_reason: str | None = None, other_stop_reason: str | None = None) -> dict:
    gr, go = gate_pass(ref, impact_min, surprise_min), gate_pass(other, impact_min, surprise_min)
    truncated = "max_tokens" in (ref_stop_reason, other_stop_reason)

    def g(r, k):
        return (r or {}).get(k)
    return {"kind": "route_pair", "workflow": workflow, "ticker": ticker,
            "reference_model": reference_model, "other_model": other_model,
            "gate_ref": gr, "gate_other": go, "gate_equal": gr == go,
            "direction_equal": g(ref, "direction") == g(other, "direction"),
            "verdict_equal": (g(ref, "red_team") or {}).get("red_team_verdict")
                             == (g(other, "red_team") or {}).get("red_team_verdict"),
            "impact_abs_diff": (abs(float(g(ref, "impact")) - float(g(other, "impact")))
                                if isinstance(g(ref, "impact"), (int, float))
                                and isinstance(g(other, "impact"), (int, float)) else None),
            "other_parse_ok": isinstance(other, dict),
            "ref_stop_reason": ref_stop_reason, "other_stop_reason": other_stop_reason,
            # TRUNCATED = INVALID_FOR_MODEL_COMPARISON: abgeschnittene Antworten sind keine Modellleistung
            "comparison_status": "TRUNCATED" if truncated else "VALID"}


def annotate_truncation(pairs: list[dict], rows: list[dict]) -> list[dict]:
    """Ältere Paare ohne comparison_status: Abbruchgrund aus den LLM-Zeilen desselben Laufs (run_id,
    ticker, model) nachtragen. Ohne Zuordnung bleibt das Paar VALID (wie bisher gewertet)."""
    idx = {}
    for r in rows:
        if r.get("kind") == "llm" and r.get("stop_reason"):
            idx.setdefault((r.get("run_id"), r.get("ticker"), r.get("model")), []).append(r["stop_reason"])
    out = []
    for p in pairs:
        if p.get("comparison_status"):
            out.append(p)
            continue
        sr = idx.get((p.get("run_id"), p.get("ticker"), p.get("other_model"))) or []
        rr = idx.get((p.get("run_id"), p.get("ticker"), p.get("reference_model"))) or []
        trunc = "max_tokens" in sr or "max_tokens" in rr
        out.append({**p, "comparison_status": "TRUNCATED" if trunc else "VALID",
                    "other_stop_reason": sr[0] if sr else None, "ref_stop_reason": rr[0] if rr else None})
    return out


def valid_pairs(pairs: list[dict]) -> list[dict]:
    return [p for p in pairs if p.get("comparison_status", "VALID") == "VALID"]


def comparison_report(rows: list[dict], workflow: str = "deep_analysis") -> dict:
    """Faire Vergleichsbasis: abgeschnittene Antworten getrennt, nie als Modellfehler gezählt."""
    pairs = annotate_truncation([r for r in rows if r.get("kind") == "route_pair" and r.get("workflow") == workflow], rows)
    v = valid_pairs(pairs)
    llm = [r for r in rows if r.get("kind") == "llm" and r.get("stage") == workflow and r.get("cost_usd")]
    cost = {}
    for r in llm:
        cost.setdefault(r.get("model"), []).append(float(r["cost_usd"]))
    mean = {m: round(sum(c) / len(c), 5) for m, c in cost.items()}
    return {"pairs_total": len(pairs), "valid_comparisons": len(v),
            "truncated": sum(1 for p in pairs if p["comparison_status"] == "TRUNCATED"),
            "equal_gate_decision": sum(1 for p in v if p.get("gate_equal")),
            "challenger_stricter": sum(1 for p in v if p.get("gate_ref") and not p.get("gate_other")),
            "challenger_looser": sum(1 for p in v if p.get("gate_other") and not p.get("gate_ref")),
            "direction_disagreements": sum(1 for p in v if not p.get("direction_equal")),
            "metrics_valid": metrics(v), "mean_cost_per_call_usd": mean,
            "note": "Kein Outcome-Ground-Truth: 'stricter/looser' beschreibt Gate-Abweichung, nicht 'besser'."}


def metrics(pairs: list[dict]) -> dict:
    n = len(pairs)
    if not n:
        return {"pairs": 0, "days": 0}
    days = len({str(p.get("ts", ""))[:10] for p in pairs})
    ref_pass = [p for p in pairs if p.get("gate_ref")]
    return {"pairs": n, "days": days,
            "gate_agreement": round(sum(1 for p in pairs if p.get("gate_equal")) / n, 3),
            "direction_agreement": round(sum(1 for p in pairs if p.get("direction_equal")) / n, 3),
            "pass_recall": (round(sum(1 for p in ref_pass if p.get("gate_other")) / len(ref_pass), 3)
                            if ref_pass else None),
            "ref_pass_n": len(ref_pass),
            "parse_ok": round(sum(1 for p in pairs if p.get("other_parse_ok")) / n, 3)}


def _meets(m: dict, s: dict, min_pairs: int, min_days: int) -> bool:
    return (m.get("pairs", 0) >= min_pairs and m.get("days", 0) >= min_days
            and m.get("gate_agreement", 0) >= float(s.get("min_gate_agreement", 0.92))
            and m.get("direction_agreement", 0) >= float(s.get("min_direction_agreement", 0.90))
            and m.get("pass_recall") is not None and m["pass_recall"] >= float(s.get("min_pass_recall", 0.90))
            and m.get("parse_ok", 0) >= 0.95)


def decision(workflow: str, champion: str, rows: list[dict] | None = None, pol: dict | None = None) -> dict:
    """Aktives Modell + Begründung. Wirft nie: im Zweifel Champion."""
    try:
        s = settings(workflow, pol)
        if s is None:
            return {"active": champion, "status": "OFF"}
        chal = s["challenger"]
        rows = rows if rows is not None else ct.load_ledger()
        pairs = valid_pairs(annotate_truncation(
            [r for r in rows if r.get("kind") == "route_pair" and r.get("workflow") == workflow], rows))
        # A/B: Champion als Referenz, Challenger als "other"
        ab = [p for p in pairs if p.get("reference_model") == champion and p.get("other_model") == chal]
        m = metrics(ab)
        out = {"active": champion, "challenger": chal, "ab": m, "mode": s.get("mode")}
        validated = _meets(m, s, int(s.get("min_pairs", 80)), int(s.get("min_days", 10)))
        if not validated:
            out["status"] = "AB_RUNNING"
            return out
        # Nach Umschaltung: Monitoring-Paare (Challenger aktiv, Champion-Stichprobe als "other")
        mon = [p for p in pairs if p.get("reference_model") == chal and p.get("other_model") == champion]
        win = int(s.get("monitor_window", 40))
        # Monitoring-Paare spiegeln: Referenz = Champion-Entscheidung
        mirrored = [{**p, "gate_ref": p.get("gate_other"), "gate_other": p.get("gate_ref")} for p in mon[-win:]]
        mm = metrics(mirrored)
        out["monitor"] = mm
        if mm.get("pairs", 0) >= win and not _meets(mm, s, win, 1):
            out["status"] = "DEMOTED"
            return out
        out["status"] = "VALIDATED"
        if s.get("mode") == "auto":
            out["active"] = chal
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"model_routing.decision Fehler -> Champion: {e}")
        return {"active": champion, "status": "ERROR"}


def sample(keys: list[str], n: int, salt: str | None = None) -> set[str]:
    """Deterministische, nicht-selektive Stichprobe (Hash, nicht Qualität) für den A/B."""
    salt = salt or date.today().isoformat()
    ranked = sorted(keys, key=lambda k: hashlib.sha256(f"{salt}:{k}".encode()).hexdigest())
    return set(ranked[:max(int(n), 0)])


def summary(champion_by_workflow: dict[str, str], rows: list[dict] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else ct.load_ledger()
    return {wf: decision(wf, champ, rows) for wf, champ in champion_by_workflow.items()}


# ── Bearish-Vorfilter (Prescreen-Richtung vor der teuren Deep Analysis) ─────
def prefilter_pair(ticker: str | None, prescreen_direction: str | None, analysis: dict | None,
                   impact_min: int = 4, surprise_min: int = 3) -> dict:
    return {"kind": "prefilter_pair", "workflow": "deep_analysis", "ticker": ticker,
            "prescreen_direction": prescreen_direction,
            "da_direction": (analysis or {}).get("direction") if isinstance(analysis, dict) else None,
            "gate_pass": gate_pass(analysis, impact_min, surprise_min)}


def prefilter_decision(rows: list[dict] | None = None, pol: dict | None = None) -> dict:
    """Aktiv nur, wenn die Prescreen-Richtung BEARISH praktisch nie einen Kandidaten trifft, den
    die Deep Analysis bestehen lässt: Verlust = Anteil der Gate-bestandenen Kandidaten mit
    Prescreen-BEARISH <= max_pass_loss (bei >= min_pass_n bestandenen an >= min_days Tagen)."""
    try:
        s = ((pol or ct.policy()).get("bearish_prefilter") or {})
        mode = str(s.get("mode", "off"))
        if mode == "off":
            return {"active": False, "status": "OFF"}
        rows = rows if rows is not None else ct.load_ledger()
        pairs = [r for r in rows if r.get("kind") == "prefilter_pair" and r.get("prescreen_direction")]
        passed = [p for p in pairs if p.get("gate_pass")]
        bear = [p for p in pairs if p.get("prescreen_direction") == "BEARISH"]
        days = len({str(p.get("ts", ""))[:10] for p in pairs})
        loss = (sum(1 for p in passed if p.get("prescreen_direction") == "BEARISH") / len(passed)) if passed else None
        out = {"pairs": len(pairs), "days": days, "pass_n": len(passed), "pass_loss": None if loss is None
               else round(loss, 3), "bearish_share": round(len(bear) / len(pairs), 3) if pairs else None,
               "mode": mode, "active": False}
        ok = (len(passed) >= int(s.get("min_pass_n", 40)) and days >= int(s.get("min_days", 10))
              and loss is not None and loss <= float(s.get("max_pass_loss", 0.05)))
        out["status"] = "VALIDATED" if ok else "AB_RUNNING"
        out["active"] = ok and mode == "auto"
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"prefilter_decision Fehler -> inaktiv: {e}")
        return {"active": False, "status": "ERROR"}


def apply_bearish_prefilter(candidates: list[dict], allow_bearish: bool, decision_: dict | None = None,
                            monitor_n: int | None = None) -> tuple[list[dict], list[str]]:
    """Entfernt Prescreen-BEARISH-Kandidaten vor der Deep Analysis, wenn Bearish-Trades aus sind
    UND der Vorfilter validiert ist. Eine kleine Hash-Stichprobe läuft zur Überwachung weiter."""
    if allow_bearish:
        return candidates, []
    d = decision_ if decision_ is not None else prefilter_decision()
    if not d.get("active"):
        return candidates, []
    s = (ct.policy().get("bearish_prefilter") or {})
    n = int(monitor_n if monitor_n is not None else s.get("monitor_sample_per_run", 2))
    bears = [c.get("ticker") for c in candidates if c.get("prescreen_direction") == "BEARISH"]
    keep_monitor = sample(bears, n)
    skipped = [t for t in bears if t not in keep_monitor]
    return [c for c in candidates if c.get("ticker") not in skipped], skipped


# ── Kompaktes Prescreen-Format (nur Ausgabeformat; Entscheidungen müssen gleich bleiben) ─
def _decisions(results: list) -> dict:
    return {r.get("ticker"): r.get("decision") for r in results or [] if isinstance(r, dict) and r.get("ticker")}


def prescreen_pair(ref_results: list, other_results: list, compact_is_reference: bool) -> dict:
    a, b = _decisions(ref_results), _decisions(other_results)
    common = [t for t in a if t in b]
    return {"kind": "prescreen_pair", "workflow": "prescreening", "compact_is_reference": compact_is_reference,
            "n": len(a), "n_common": len(common), "agree": sum(1 for t in common if a[t] == b[t]),
            "yes_ref": sum(1 for t in a if a[t] == "[YES]"), "yes_other": sum(1 for t in b if b[t] == "[YES]")}


def compact_decision(rows: list[dict] | None = None, pol: dict | None = None) -> dict:
    """Aktiv, wenn die YES/NO-Entscheidungen beider Vorlagen auf denselben Batches zu
    >= min_agreement übereinstimmen (>= min_decisions Ticker an >= min_days Tagen) und die
    YES-Zahl nicht systematisch sinkt (Recall >= min_yes_ratio)."""
    s = ((pol or ct.policy()).get("prescreen_compact") or {})
    mode = str(s.get("mode", "off"))
    if mode == "off":
        return {"active": False, "status": "OFF"}
    rows = rows if rows is not None else ct.load_ledger()
    pairs = [r for r in rows if r.get("kind") == "prescreen_pair"]
    n = sum(int(r.get("n_common") or 0) for r in pairs)
    # Paare vereinheitlichen: Referenz = bewährte Vorlage
    yes_legacy = sum(int(r.get("yes_other" if r.get("compact_is_reference") else "yes_ref") or 0) for r in pairs)
    yes_compact = sum(int(r.get("yes_ref" if r.get("compact_is_reference") else "yes_other") or 0) for r in pairs)
    agree = sum(int(r.get("agree") or 0) for r in pairs)
    missing = sum(int(r.get("n") or 0) - int(r.get("n_common") or 0) for r in pairs)
    days = len({str(r.get("ts", ""))[:10] for r in pairs})
    out = {"decisions": n, "days": days, "agreement": round(agree / n, 3) if n else None,
           "yes_legacy": yes_legacy, "yes_compact": yes_compact, "missing": missing, "mode": mode}
    ok = (n >= int(s.get("min_decisions", 200)) and days >= int(s.get("min_days", 5))
          and out["agreement"] is not None and out["agreement"] >= float(s.get("min_agreement", 0.95))
          and (yes_legacy == 0 or yes_compact / yes_legacy >= float(s.get("min_yes_ratio", 0.9)))
          and missing <= 0.02 * max(n, 1))
    out["status"] = "VALIDATED" if ok else "AB_RUNNING"
    out["active"] = ok and mode == "auto"
    return out
