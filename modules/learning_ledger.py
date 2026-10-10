"""modules/learning_ledger.py – Consume-once-Garantie für Lern-Updates aus gereiften Outcomes.

Jede eindeutige Beobachtung wird je (observation_id, Horizont, learning_target) genau EINMAL verarbeitet,
z. B. `T3a1…:close:feature_stats` oder `shadow:ab12…:45d:feature_stats_external`.

Crash-/Retry-Sicherheit: Die Markierung liegt im SELBEN Dokument wie der gelernte Zustand (history.json).
`feedback.save_history` schreibt atomar, Update und Markierung werden also gemeinsam persistiert oder gehen
gemeinsam verloren:
  * Crash vor dem Speichern: Beides fehlt; der Retry verarbeitet genau einmal.
  * Crash nach dem Speichern: Die Markierung ist vorhanden; der Retry überspringt (duplicate_learning_skip).
  * Fehler im Update selbst: Die betroffenen Abschnitte werden auf den Stand vor dem Update zurückgesetzt
    (Transaktion im Speicher), es wird keine Markierung gesetzt, und es gibt kein halbes Update.
Verschiedene Horizonte oder Lernziele derselben Beobachtung sind getrennte Schlüssel und ausdrücklich erlaubt.
Gilt nur für inkrementelle Akkumulatoren. Voll neu berechnete Größen (Pearson-Gewichte, Kalibrierung,
Ledger-Evaluation) zählen jede Beobachtung je Berechnung ohnehin genau einmal.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger(__name__)

FIELD = "processed_outcomes"            # {key: processed_at (ISO-Datum)}
META = "processed_outcomes_meta"        # {"duplicate_skips": n, "since": ISO}
_RUN = {"processed": 0, "duplicate_skips": 0}


def key(observation_id: str, horizon: str | int, target: str) -> str:
    if not observation_id or not target:
        raise ValueError("observation_id und learning_target sind Pflicht")
    h = f"{horizon}d" if isinstance(horizon, int) else str(horizon)
    return f"{observation_id}:{h}:{target}"


def trade_id(trade: dict) -> str:
    """Stabile Identität eines Champion-Trades (Ticker, Entry-Datum, Strategie, Kontrakt)."""
    opt = trade.get("option") or {}
    raw = "|".join(str(x) for x in (trade.get("ticker"), str(trade.get("entry_date") or "")[:10], trade.get("strategy"),
                                     opt.get("expiry") or opt.get("expiration"), opt.get("strike"),
                                     opt.get("short_strike") or opt.get("strike_short")))
    return "T" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def is_processed(state: dict, k: str) -> bool:
    return k in (state.get(FIELD) or {})


def consume(state: dict, observation_id: str, horizon: str | int, target: str, fn: Callable[[], None], *,
            sections: tuple[str, ...] = (), now: datetime | None = None) -> bool:
    """Führt `fn` genau einmal je Schlüssel aus und markiert den Schlüssel in `state`.
    sections: Schlüssel von `state`, die `fn` verändert. Bei einem Fehler werden sie zurückgesetzt.
    -> True = verarbeitet, False = bereits verarbeitet (übersprungen). Fehler in fn werden weitergereicht."""
    k = key(observation_id, horizon, target)
    done = state.setdefault(FIELD, {})
    if k in done:
        _RUN["duplicate_skips"] += 1
        meta = state.setdefault(META, {"duplicate_skips": 0})
        meta["duplicate_skips"] = int(meta.get("duplicate_skips", 0)) + 1
        log.info(f"learning_ledger: {k} bereits verarbeitet -> übersprungen (kein Doppel-Update)")
        return False
    snap = {s: copy.deepcopy(state.get(s)) for s in sections}
    try:
        fn()
    except Exception:
        for s, v in snap.items():
            if v is None:
                state.pop(s, None)
            else:
                state[s] = v
        raise
    done[k] = (now or datetime.now(timezone.utc)).date().isoformat()
    state.setdefault(META, {"duplicate_skips": 0}).setdefault("since", done[k])
    _RUN["processed"] += 1
    return True


def _sig(t: dict) -> tuple:
    return trade_id(t), str(t.get("close_date") or ""), t.get("outcome")


def quarantine_duplicate_trades(state: dict, now: datetime | None = None) -> int:
    """Exakte Duplikate in closed_trades (gleiche Trade-Identität, gleicher Close-Tag, gleicher Outcome) zählen nur
    einmal: Spätere Kopien wandern nach `closed_trades_quarantine` (nicht gelöscht, mit Grund). Idempotent.
    Hintergrund: Altbestand 2026-04-11 enthält mehrfach gebuchte, identische Trades. Jede Kopie floss bisher erneut
    in Pearson-Gewichte, Robust-PPO-Training, Abstention-Vorschläge und Forward-Statistiken ein."""
    closed = state.get("closed_trades") or []
    seen: set = set()
    keep, moved = [], []
    stamp = (now or datetime.now(timezone.utc)).date().isoformat()
    for t in closed:
        sg = _sig(t)
        if sg in seen:
            moved.append({**t, "_quarantine": {"reason": "exact_duplicate_closed_trade", "trade_id": sg[0],
                                               "quarantined_at": stamp}})
        else:
            seen.add(sg)
            keep.append(t)
    if moved:
        state["closed_trades"] = keep
        state.setdefault("closed_trades_quarantine", []).extend(moved)
        log.warning(f"learning_ledger: {len(moved)} exakte Duplikate aus closed_trades in Quarantäne verschoben")
    return len(moved)


def already_closed(state: dict, trade: dict) -> bool:
    """Ist dieselbe Trade-Identität bereits geschlossen? (doppelte active_trades-Einträge)"""
    tid = trade_id(trade)
    return any(trade_id(t) == tid for t in state.get("closed_trades") or [])


def run_stats(reset: bool = False) -> dict:
    """Zähler dieses Prozesses (für das Lauf-Log: duplicate_learning_skips)."""
    out = dict(_RUN)
    if reset:
        _RUN.update(processed=0, duplicate_skips=0)
    return out
