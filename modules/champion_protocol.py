"""Champion als vorregistriertes Experiment (Aktionsplan 2026-11-09, Punkt 1).

Bisher wurden die Produktions-Gates nachträglich auf gesehenen Trades nachjustiert (z. B. Mismatch-Cap aus n = 4,
DTE-Floor aus 98 Trades, Bearish-Sperre aus 0/6). Ab der Registrierung gilt stattdessen:

  * EINE Primärkennzahl: verlässliche Netto-Optionsrendite je geschlossenem Champion-Trade
    (history.json closed_trades, outcome_method in RELIABLE_OUTCOME_METHODS, outcome_net = Entry Ask, Exit Bid,
    abzüglich Kommission), nur Trades mit entry_date >= start_date.
  * Die entscheidungsrelevante Konfiguration ist per Hash eingefroren (decision_config_sha256): Gates, Filter,
    Risiko, Optionen, Pipeline, Modelle, Kosten, Haltedauer, economics-first-Budget und die LLM-Prompt-Versionen.
    Weicht der aktuelle Hash ab, ist das Protokoll gebrochen (PROTOCOL_BROKEN) – eine Gate-Änderung braucht eine
    neue Protokollversion (neuer Start, neue Ledger-Zuteilung). tests/test_champion_protocol.py erzwingt das.
  * Test: einseitig H0 Expectancy <= 0, Beobachtungen = Mittel je Entry-Tag (abhängige Trades desselben Tages
    zählen einmal), t-Test; geplante Prüfungen bei 30/60/90 verlässlichen Trades, Alpha aus dem globalen Ledger
    (Klasse alpha_hypothesis, Teilsystem champion) zu gleichen Teilen auf die Prüfungen verteilt (Bonferroni).
    Bestätigung nur bei p < Teil-Alpha; frühes Verwerfen, wenn die einseitige Obergrenze zum Teil-Alpha < 0 liegt.
    Nach der letzten Prüfung bzw. max_duration_days ohne Bestätigung: NOT_CONFIRMED.
  * Prüfberichte werden je Prüfung einmal eingefroren (create-only); spätere Läufe rechnen nie neu.

Nur Messung und Bericht: das Protokoll erzeugt, blockiert oder verändert keine Trades.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PROTOCOL = Path("config/champion_protocol.yaml")
CONFIG = Path("config.yaml")
ECONOMICS_FIRST = Path("config/economics_first.yaml")
HISTORY = Path("outputs/history.json")
OUT = Path("outputs/research/champion_protocol")
DECISION_SECTIONS = ("gates", "filters", "risk", "options", "pipeline", "models", "costs", "learning")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def prompt_versions() -> dict:
    """Versionen der LLM-Prompts (Vorlagen + Sampling temperature=0@2023-06-01), die über die Kandidaten
    entscheiden. Ein Sampling-Fallback zur Laufzeit steht je Ergebnis in llm_audit.sampling."""
    from modules import deep_analysis, prescreener
    return {"deep_analysis": deep_analysis.PROMPT_VERSION, "prescreener": prescreener.PROMPT_VERSION}


def decision_config(config: Path = CONFIG, economics_first: Path = ECONOMICS_FIRST) -> dict:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    ef = yaml.safe_load(Path(economics_first).read_text(encoding="utf-8")) or {}
    return {"config_yaml": {k: cfg.get(k) for k in DECISION_SECTIONS}, "economics_first": ef,
            "prompts": prompt_versions()}


def decision_config_sha(config: Path = CONFIG, economics_first: Path = ECONOMICS_FIRST) -> str:
    return _sha(decision_config(config, economics_first))


def load_protocol(path: Path = PROTOCOL) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def reliable_trades(history: dict, start: date) -> list[dict]:
    """Verlässliche, geschlossene Champion-Trades ab start (Reihenfolge: Schließdatum, dann Ticker)."""
    from modules.outcomes import is_reliable_outcome
    out = []
    for t in history.get("closed_trades") or []:
        try:
            entry = date.fromisoformat(str(t.get("entry_date"))[:10])
        except ValueError:
            continue
        y = t.get("outcome_net")
        if entry < start or not is_reliable_outcome(t) or not isinstance(y, (int, float)) or not math.isfinite(y):
            continue
        out.append({"ticker": t.get("ticker"), "entry_date": entry.isoformat(),
                    "close_date": str(t.get("close_date") or "")[:10], "y": float(y)})
    return sorted(out, key=lambda r: (r["close_date"], r["ticker"] or "", r["entry_date"]))


def day_means(trades: list[dict]) -> list[float]:
    by: dict[str, list[float]] = {}
    for t in trades:
        by.setdefault(t["entry_date"], []).append(t["y"])
    return [sum(v) / len(v) for _, v in sorted(by.items())]


def one_sided_test(xs: list[float], alpha: float) -> dict:
    """Einseitiger t-Test H0: Mittel <= 0 über Tagesmittel; p-Wert und einseitige Grenzen zum Teil-Alpha."""
    from scipy import stats
    n = len(xs)
    if n < 3:
        return {"n_days": n, "mean": None, "p_value": None, "lower": None, "upper": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    if sd <= 1e-12 * max(1.0, abs(m)):
        # Zero sample variance does NOT establish zero sampling uncertainty.
        # Never declare statistical confirmation from perfectly repeated few-day returns.
        return {"n_days": n, "mean": round(m, 6), "p_value": None,
                "lower": None, "upper": None}
    se = sd / math.sqrt(n)
    tq = float(stats.t.ppf(1 - alpha, n - 1))
    return {"n_days": n, "mean": round(m, 6), "p_value": float(stats.t.sf(m / se, n - 1)),
            "lower": round(m - tq * se, 6), "upper": round(m + tq * se, 6)}


FEASIBILITY_POWER = 0.80
FEASIBILITY_MIN_DAYS = 20      # erst ab so vielen Handelstagen ist die Trade-Rate schätzbar


def feasibility(history: dict, trades: list[dict], start: date, today: date, looks: list[int],
                look_alpha: float | None, max_duration_days: int) -> dict:
    """Ist der vorregistrierte Test bei der beobachteten Trade-Rate überhaupt entscheidbar? (Audit 2026-11-27)

    Nur Messung, ändert weder Test noch Prüfzeitpunkte. Rate = verlässliche Trades je NYSE-Handelstag seit Start.
    Kleinster nachweisbarer Effekt je Trade (MDE) bei Prüfung n: (z_{1-α_look} + z_{Power}) · SD / √n, SD aus den
    verlässlichen Paper-Outcomes VOR dem Start (unabhängig von der Teststichprobe; Tagesmittel-Aggregation ignoriert
    -> eher optimistisch)."""
    from statistics import NormalDist, stdev

    from modules.market_snapshot import is_trading_day
    end = date.fromordinal(start.toordinal() + int(max_duration_days))
    elapsed = sum(1 for o in range(start.toordinal(), min(today, end).toordinal() + 1)
                  if is_trading_day(date.fromordinal(o)))
    remaining = sum(1 for o in range(max(today, start).toordinal() + 1, end.toordinal() + 1)
                    if is_trading_day(date.fromordinal(o)))
    prior = [t["y"] for t in reliable_trades(history, date.min) if date.fromisoformat(t["entry_date"]) < start]
    sd = stdev(prior) if len(prior) >= 3 else None
    rate = len(trades) / elapsed if elapsed else None
    z = NormalDist().inv_cdf
    rows = []
    for n in looks:
        need = max(0, n - len(trades))
        days = (need / rate) if rate else None
        mde = ((z(1 - look_alpha) + z(FEASIBILITY_POWER)) * sd / math.sqrt(n)) if (sd and look_alpha) else None
        rows.append({"n_trades": n, "missing_trades": need,
                     "projected_trading_days": round(days, 1) if days is not None else None,
                     "reachable_before_end": (need == 0) or (days is not None and days <= remaining),
                     "mde_mean_return_per_trade": round(mde, 4) if mde is not None else None})
    first = rows[0] if rows else {}
    verdict = ("TOO_EARLY_TO_ESTIMATE" if elapsed < FEASIBILITY_MIN_DAYS else
               "INFEASIBLE_AT_CURRENT_RATE" if not first.get("reachable_before_end") else "FEASIBLE")
    return {"trading_days_elapsed": elapsed, "trading_days_remaining": remaining,
            "reliable_trades_per_trading_day": round(rate, 4) if rate is not None else None,
            "sd_basis": {"n_prior_reliable_outcomes": len(prior), "sd": round(sd, 4) if sd else None},
            "power": FEASIBILITY_POWER, "looks": rows, "verdict": verdict,
            "note": "Nur Messung; Test, Alpha und Prüfzeitpunkte bleiben wie registriert."}


def _freeze(out: Path, look: int, report: dict) -> dict:
    """Prüfbericht je Prüfung einmal einfrieren (create-only); ein bestehender Bericht gilt unverändert."""
    p = out / f"look_{look}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {**report, "status": "LOOK_REPORT_UNREADABLE"}
    out.mkdir(parents=True, exist_ok=True)
    rep = {**report, "report_sha256": _sha(report)}
    p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return rep


def evaluate(today: date | None = None, *, protocol_path: Path = PROTOCOL, history_path: Path = HISTORY,
             out: Path = OUT, config: Path = CONFIG, economics_first: Path = ECONOMICS_FIRST,
             alpha_ledger_path: Path | None = None, persist: bool = True) -> dict:
    from modules import ledger_registration as lr
    today = today or datetime.now(timezone.utc).date()
    proto = load_protocol(protocol_path)
    start = date.fromisoformat(str(proto["start_date"]))
    looks = [int(n) for n in proto["looks_at_n"]]
    current_sha = decision_config_sha(config, economics_first)
    alloc = lr.ledger_alpha(lr.champion_test_id(proto), alpha_ledger_path)
    look_alpha = lr.per_look_alpha(alloc, len(looks))
    try:
        history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        history = {}
    trades = reliable_trades(history, start)
    res = {"protocol": proto.get("version"), "as_of": today.isoformat(), "start_date": start.isoformat(),
           "primary_metric": proto.get("primary_metric"), "decision_config_sha256": current_sha,
           "decision_config_pinned": proto.get("decision_config_sha256"),
           "ledger_alpha": (alloc or {}).get("alpha"), "look_alpha": look_alpha, "looks_at_n": looks,
           "reliable_trades": len(trades), "looks": [], "production_influence": "NONE"}
    try:
        res["feasibility"] = feasibility(history, trades, start, today, looks, look_alpha,
                                         int(proto.get("max_duration_days", 365)))
    except Exception as e:  # noqa: BLE001 – reine Diagnose, darf die Auswertung nie verhindern
        log.warning("Champion-Machbarkeitsdiagnose nicht berechenbar: %s", e)
        res["feasibility"] = {"verdict": "UNKNOWN", "error": f"{type(e).__name__}: {e}"[:160]}
    if current_sha != proto.get("decision_config_sha256"):
        res["status"] = "PROTOCOL_BROKEN"
        res["reason"] = ("Entscheidungs-Konfiguration weicht vom registrierten Stand ab – neue Protokollversion "
                         "registrieren (neuer Start, neue Ledger-Zuteilung); bisherige Trades zählen nicht weiter.")
        return res
    if look_alpha is None:
        res["status"] = "AWAITING_ALLOCATION"
        res["reason"] = "Noch keine Zuteilung aus dem globalen Alpha-Ledger – keine Bestätigung möglich."
        return res
    status = "NEED_MORE_DATA"
    for k, n in enumerate(looks, start=1):
        if len(trades) < n:
            break
        subset = trades[:n]
        test = one_sided_test(day_means(subset), look_alpha)
        verdict = ("CONFIRMED_POSITIVE" if test["p_value"] is not None and test["p_value"] < look_alpha
                   and (test["mean"] or 0) > 0 else
                   "REJECTED_NEGATIVE" if test["upper"] is not None and test["upper"] < 0 else "CONTINUE")
        report = {"protocol": proto.get("version"), "look": k, "n_trades": n,
                  "trade_ids": [f"{t['ticker']}|{t['entry_date']}|{t['close_date']}" for t in subset],
                  "test": test, "look_alpha": look_alpha, "verdict": verdict, "evaluated_on": today.isoformat()}
        rep = _freeze(out, k, report) if persist else report
        res["looks"].append({k2: rep.get(k2) for k2 in ("look", "n_trades", "test", "verdict", "evaluated_on")})
        status = rep.get("verdict", verdict)
        if status in ("CONFIRMED_POSITIVE", "REJECTED_NEGATIVE"):
            break
    end = start.toordinal() + int(proto.get("max_duration_days", 365))
    if status in ("CONTINUE", "NEED_MORE_DATA") and (len(res["looks"]) == len(looks) or today.toordinal() > end):
        status = "NOT_CONFIRMED"
    res["status"] = status
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = evaluate()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("protocol", "status", "reliable_trades", "look_alpha")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
