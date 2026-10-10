"""modules/expectation_alpha/timing.py – deterministische Research-Entscheidung (kein LLM).

Reihenfolge (vorab festgelegt, config `decision`):
  ERROR    Technischer Fehler oder Datenfehler: News-Edge oder Richtung fehlen, Laufbudget überschritten
           oder gar kein Kontext verfügbar. ERROR ist nie ABSTAIN.
  ABSTAIN  Gründe (mehrere möglich):
           - schwache News ohne positiven Kontext;
           - negativer Kontext;
           - Regime-Unsicherheit > Grenze;
           - effektiver Widerspruchsanteil (familiengedämpft) >= Grenze.
  TRADE    effective_confirmation_ratio >= Grenze bei mindestens `min_confirmation_families` verfügbaren
           Evidenz-Familien und (starke News oder positiver Kontext).
  WAIT     sonst. Konkreter Trigger (any_of), höchstens max_wait_trading_days.
Kill Conditions werden zum Entscheidungszeitpunkt eingefroren (kill_hash). Spätere Prüfungen nutzen nur
diese Fassung; ein abweichender Hash macht die Beobachtung zu DATA_BAD.
"""
from __future__ import annotations

import re

from modules.expectation_alpha.schemas import (ABSTAIN, ERROR, NEWS_STRONG, TRADE, WAIT, canonical_hash, finite)


def ttm_upper_days(ttm) -> float | None:
    """'4-8 Wochen' -> 56, '2-3 Monate' -> 90, '6 Monate' -> 180 (Obergrenze = spätester erwarteter Eintritt)."""
    if not isinstance(ttm, str):
        return None
    m = re.match(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*(woche|monat|tag)", ttm.lower())
    if not m:
        return None
    n = int(m.group(2) or m.group(1))
    return float(n * {"tag": 1, "woche": 7, "monat": 30}[m.group(3)])


def kill_conditions(*, ttm, primary_domain: str | None, gap_sign: int | None, cfg: dict) -> dict:
    kc = (cfg.get("decision") or {}).get("kill_conditions") or {}
    days = ttm_upper_days(ttm)
    cat_td = int(round(days * 5 / 7)) if days else int((kc.get("catalyst_failure") or {}).get("default_days", 20))
    k = {
        "economic_invalidation": {"domain": primary_domain, "gap_sign_at_decision": gap_sign,
                                  "rule": (kc.get("economic_invalidation") or {}).get("rule"),
                                  "evaluable": primary_domain is not None and gap_sign not in (None, 0)},
        "market_confirmation_failure": {**dict(kc.get("market_confirmation_failure") or {}),
                                        "family_dampening": float((cfg.get("confirmation") or {})
                                                                  .get("family_dampening", 0.5))},
        "catalyst_failure": {"deadline_trading_days": cat_td, "ttm": ttm,
                             "source": "ttm_upper_bound" if days else "default",
                             "rule": (kc.get("catalyst_failure") or {}).get("rule")},
        "time_invalidation": dict(kc.get("time_invalidation") or {}),
        "risk_stop": dict(kc.get("risk_stop") or {}),
    }
    return {"conditions": k, "kill_hash": canonical_hash(k), "immutable": True}


def verify_kill(kill: dict) -> bool:
    """True, wenn die eingefrorenen Kill Conditions unverändert sind."""
    return bool(kill) and canonical_hash(kill.get("conditions")) == kill.get("kill_hash")


def wait_trigger(cfg: dict) -> dict:
    wt = (cfg.get("decision") or {}).get("wait_trigger") or {}
    t = {"any_of": [dict(x) for x in wt.get("any_of") or []],
         "max_wait_trading_days": int(wt.get("max_wait_trading_days", 10)),
         "min_confirmation_families": int((cfg.get("decision") or {}).get("min_confirmation_families", 3)),
         "family_dampening": float((cfg.get("confirmation") or {}).get("family_dampening", 0.5)),
         "evaluation": "je Schlusskurs ab Entscheidungstag, nur mit bis dahin bekannten Kursen; Einstieg zum Schluss "
                       "des Folgetags nach dem Triggertag (nie Signal und Ausführung am selben Schlusskurs)",
         "entry_lag_days": 1}
    t["trigger_hash"] = canonical_hash(t)
    return t


def decide(*, news: dict, context: dict, confirmation: dict, regime_uncertainty, errors: list[str],
           cfg: dict) -> dict:
    dc = cfg.get("decision") or {}
    if errors:
        return {"status": ERROR, "reasons": list(errors), "wait_trigger": None}
    strong = news.get("strength") == NEWS_STRONG
    ctx = context.get("context_status")
    # familienbewusst: unabhängige Evidenz-Familien statt Rohsignalen (korrelierte Signale zählen gedämpft)
    n_av = int(confirmation.get("family_count") or 0)
    ratio = finite(confirmation.get("effective_confirmation_ratio"))
    conflict = finite(confirmation.get("effective_conflict_share"))
    ru = finite(regime_uncertainty)
    if ctx is None and n_av == 0 and ru is None:
        return {"status": ERROR, "reasons": ["KEIN_KONTEXT: Gaps, Bestätigung und Regime nicht verfügbar"],
                "wait_trigger": None}
    min_av = int(dc.get("min_confirmation_families", 3))
    why = []
    if not strong and ctx != 1:
        why.append("WEAK_NEWS_NO_POSITIVE_CONTEXT")
    if ctx == -1:
        why.append("NEGATIVE_CONTEXT")
    if ru is not None and ru > float(dc.get("max_regime_uncertainty", 0.6)):
        why.append("REGIME_UNCERTAIN")
    if n_av >= min_av and conflict is not None and conflict >= float(dc.get("conflict_share_abstain", 0.5)):
        why.append("CONFIRMATION_CONFLICT")
    if why:
        return {"status": ABSTAIN, "reasons": why, "wait_trigger": None}
    if n_av >= min_av and ratio is not None and ratio >= float(dc.get("trade_min_confirmation_ratio", 0.6)) \
            and (strong or ctx == 1):
        return {"status": TRADE, "reasons": ["CONFIRMED"], "wait_trigger": None}
    reason = "CONFIRMATION_INSUFFICIENT" if n_av < min_av else "CONFIRMATION_PENDING"
    return {"status": WAIT, "reasons": [reason], "wait_trigger": wait_trigger(cfg)}
