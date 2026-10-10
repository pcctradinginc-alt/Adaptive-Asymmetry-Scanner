"""modules/request_cache.py – Request-Deduplizierung innerhalb eines Laufs (API-Kosten-Optimierung 2026-10-09).

Mehrere Module rufen im selben Scanner-Lauf exakt dieselbe Anfrage ab (z. B. Tradier
/markets/options/expirations für denselben Ticker: alpha_sources-Skew, options_designer,
market_snapshot). Ein Prozess-lokaler Cache liefert die zweite identische Anfrage aus dem Speicher.

Regeln (config/data_freshness.yaml):
  * Schlüssel = Provider + Endpoint + Parameter (sortiert) – nie Header/Keys (keine Secrets im Speicher-Key).
  * Nur Endpoints mit cache_in_run: true und max_staleness_s > 0. Quotes/Chains (REALTIME_OR_DAILY_CRITICAL)
    werden NIE wiederverwendet -> Finalisten sehen immer frische Quotes.
  * Ablauf nach max_staleness_s, neuer UTC-Tag invalidiert immer.
  * Fehler/Exceptions werden nie gecacht (nächster Aufruf fragt erneut).
  * Unlesbarer/beschädigter Eintrag -> Miss (frisch abrufen statt alten Wert erzwingen).
  * Der Wert wird als Kopie ausgegeben (kein geteilter, veränderbarer Zustand zwischen Consumern).
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

log = logging.getLogger(__name__)

CONFIG = Path("config/data_freshness.yaml")
_LOCK = threading.Lock()
_STORE: dict[str, dict] = {}
STATS: Counter = Counter()
_CFG: dict | None = None


def _cfg() -> dict:
    global _CFG
    if _CFG is None:
        try:
            _CFG = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            log.debug(f"data_freshness.yaml nicht ladbar ({e}) -> kein Request-Cache")
            _CFG = {}
    return _CFG


def endpoint_spec(provider: str, endpoint: str) -> dict:
    return ((_cfg().get("endpoints") or {}).get(f"{provider}/{endpoint}")) or {}


def cache_ttl_s(provider: str, endpoint: str) -> float:
    """0 = nie wiederverwenden (auch für unbekannte Endpoints – im Zweifel frisch)."""
    s = endpoint_spec(provider, endpoint)
    if not s.get("cache_in_run"):
        return 0.0
    try:
        return max(0.0, float(s.get("max_staleness_s") or 0))
    except (TypeError, ValueError):
        return 0.0


def request_key(provider: str, endpoint: str, params: dict | None) -> str:
    return f"{provider}/{endpoint}?" + json.dumps(params or {}, sort_keys=True, default=str)


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def cached(provider: str, endpoint: str, params: dict | None, fetch: Callable[[], Any],
           now: Callable[[], float] = time.time) -> Any:
    """Liefert fetch() oder – bei identischer, noch frischer Anfrage im selben Prozess – die gespeicherte Kopie.
    fetch() darf werfen; Exceptions werden weitergereicht und nie gecacht."""
    ttl = cache_ttl_s(provider, endpoint)
    if ttl <= 0:
        STATS["bypass"] += 1
        return fetch()
    key = request_key(provider, endpoint, params)
    t = now()
    with _LOCK:
        e = _STORE.get(key)
    if e is not None:
        try:
            fresh = (t - float(e["t"])) <= ttl and _day(float(e["t"])) == _day(t)
            if fresh:
                val = copy.deepcopy(e["value"])
                STATS["hit"] += 1
                return val
            STATS["expired"] += 1
        except (KeyError, TypeError, ValueError):            # beschädigter Eintrag -> Miss
            STATS["corrupt"] += 1
        with _LOCK:
            _STORE.pop(key, None)
    STATS["miss"] += 1
    value = fetch()
    with _LOCK:
        _STORE[key] = {"t": t, "value": copy.deepcopy(value)}
    return value


def invalidate(provider: str | None = None, symbol: str | None = None) -> int:
    """Gezielte Invalidierung (Corporate Action, neues Listing, Schemaänderung). Ohne Argumente: alles."""
    with _LOCK:
        keys = [k for k in _STORE if (provider is None or k.startswith(provider + "/"))
                and (symbol is None or f'"symbol": "{symbol}"' in k)]
        for k in keys:
            _STORE.pop(k, None)
    return len(keys)


def reset() -> None:
    global _CFG
    with _LOCK:
        _STORE.clear()
    STATS.clear()
    _CFG = None


def stats() -> dict:
    s = dict(STATS)
    lookups = s.get("hit", 0) + s.get("miss", 0)
    s["hit_rate"] = round(s.get("hit", 0) / lookups, 3) if lookups else None
    return s
