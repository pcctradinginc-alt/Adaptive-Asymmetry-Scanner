"""modules/source_health.py – täglicher Source Health Check für ALLE externen Datenquellen.

    python -m modules.source_health check [--no-mail] [--no-probes]   (CI: source_health.yml, vor dem Scanner)
    python -m modules.source_health report                             (Daily Data Health Report aus Snapshot)

Kernregel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten
Daten. Unbekannt != gesund – ohne aktuellen Snapshot gilt jede Quelle als UNVALIDATED.

Quellen (keine Parallel-Registry – alles aus Bestehendem):
  * Champion-Pflichtdaten des Scanners (config/source_health.yaml core_sources): aktive Probes
    (Kurse, VIX, Optionsketten inkl. offiziellem Fallback-Provider, Nachrichten).
  * Externe Registry-Quellen (config/external_sources/*.yaml, täglicher Orchestrator):
    outputs/external_data/health/source_health.json (Status, Freshness, Data-Quality-Befund).
  * Alternative Daten (eigene Wochenjobs): leichte Probe + Health-Datei + Store-Statistik.

Prüfungen je Quelle: Erreichbarkeit, Authentifizierung, HTTP-/API-Status, Rate Limit,
Antwortzeit, Schema, leere Antwort, letzte erfolgreiche Aktualisierung, observation_time,
available_at, retrieved_at, Datenalter gegen quellenabhängige Grenze (Frequenz!), Coverage,
Missing-Rate, Duplikate, Ausreißer, Plausibilität (Zeitstempel in der Zukunft), neue Daten
seit dem letzten Check, nachgelagerte Nutzung (Features, Modelle, Entscheidungen, Hypothesen).

Status: HEALTHY | DEGRADED | STALE | BROKEN | UNVALIDATED.
Recovery: nach STALE/BROKEN erst nach `recovery_passes_required` vollständig bestandenen
Prüfungen in Folge wieder HEALTHY (bis dahin DEGRADED, "RECOVERING").

Ausgaben (outputs/health/):
  source_health_snapshot.json   aktueller Zustand (Scanner liest NUR diesen)
  source_health_history.jsonl   append-only Historie (wiederkehrende Instabilität)
  feature_availability.json     je Feature available/stale/data_quality/source_actual
  data_safe_mode.json           blockierte Entscheidungspfade, deaktivierte Signale, globaler Safe Mode
  daily_data_health.md          kompakter Report
  health_notified.json          zuletzt gemeldeter Zustand (Mail nur bei relevanter Änderung)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

CONFIG = Path("config/source_health.yaml")
OUT = Path("outputs/health")
SNAPSHOT = OUT / "source_health_snapshot.json"
HISTORY = OUT / "source_health_history.jsonl"
FEATURES = OUT / "feature_availability.json"
DATA_SAFE_MODE = OUT / "data_safe_mode.json"
REPORT_MD = OUT / "daily_data_health.md"
NOTIFIED = OUT / "health_notified.json"
REGISTRY_HEALTH = Path("outputs/external_data/health/source_health.json")

HEALTHY, DEGRADED, STALE, BROKEN, UNVALIDATED = "HEALTHY", "DEGRADED", "STALE", "BROKEN", "UNVALIDATED"
STATUSES = (HEALTHY, DEGRADED, STALE, BROKEN, UNVALIDATED)
UNHEALTHY = (STALE, BROKEN, UNVALIDATED)
SEVERITY = {HEALTHY: 0, DEGRADED: 1, UNVALIDATED: 2, STALE: 3, BROKEN: 4}
# Alter der letzten erfolgreichen Aktualisierung (Registry-Quellen; Orchestrator läuft täglich,
# selten aktualisierte Quellen dürfen länger ohne neuen Erfolg sein).
SUCCESS_AGE_DAYS_BY_FREQUENCY = {"hourly": 2, "daily": 3, "weekly": 10, "monthly": 35, "quarterly": 100,
                                 "annual": 400, "static": 400, "event": 10}
CRIT_WEIGHT = {"CRITICAL": 3.0, "IMPORTANT": 2.0, "NON_CRITICAL": 1.0}
HEADERS = {"sec": {"User-Agent": "AdaptiveAsymmetryScanner research@pcctrading.com", "Accept-Encoding": "gzip, deflate"},
           "jsonapi": {"Accept": "application/vnd.api+json"}}


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG).read_text(encoding="utf-8"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(x) -> datetime | None:
    if x in (None, "", "None"):
        return None
    try:
        t = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).to_pydatetime()


def _iso(t: datetime | None) -> str | None:
    return t.isoformat(timespec="seconds") if t else None


def _age_days(t: datetime | None, now: datetime) -> float | None:
    return None if t is None else round((now - t).total_seconds() / 86400, 2)


# ── Probes (Erreichbarkeit, Auth, Status, Rate Limit, Latenz, Schema, leer) ────
def _probe_result(**kw) -> dict:
    base = {"ok": False, "reachable": None, "auth_ok": None, "http_status": None, "rate_limited": False,
            "timeout": False, "latency_s": None, "schema_ok": None, "empty": None, "error": None,
            "latest_observation": None, "n_items": None, "missing_rate": None}
    base.update(kw)
    return base


def _classify_exception(e: Exception) -> dict:
    from modules.external.http import AuthError, SchemaError, redact_secrets
    msg = redact_secrets(f"{type(e).__name__}: {e}")[:300]
    low = msg.lower()
    if isinstance(e, AuthError):
        return {"reachable": True, "auth_ok": False, "error": msg}
    if isinstance(e, SchemaError):
        return {"reachable": True, "schema_ok": False, "error": msg}
    if "429" in low:
        return {"reachable": True, "rate_limited": True, "http_status": 429, "error": msg}
    if "timeout" in low or "timed out" in low:
        return {"reachable": False, "timeout": True, "error": msg}
    status = next((int(c) for c in ("500", "502", "503", "504", "404", "400") if c in low), None)
    return {"reachable": status is not None, "http_status": status, "error": msg}


def run_probe(spec: dict, now: datetime, *, fetch=None, yf_module=None, env: dict | None = None) -> dict:
    """Leichte Probe mit begrenztem exponentiellem Backoff (http.fetch: retries=3, 2/4 s)."""
    import os
    env = os.environ if env is None else env
    kind = spec.get("kind")
    t0 = time.monotonic()
    try:
        if kind in ("http_json", "http_csv"):
            if fetch is None:
                from modules.external.http import fetch as _f
                fetch = _f
            headers = dict(HEADERS.get(spec.get("headers"), {}) if isinstance(spec.get("headers"), str)
                           else spec.get("headers") or {})
            params = dict(spec.get("params") or {})
            for k, d in (spec.get("date_params") or {}).items():
                params[k] = (now + timedelta(days=int(d))).date().isoformat()
            if spec.get("auth_env"):
                key = env.get(spec["auth_env"], "")
                if not key:
                    return _probe_result(auth_ok=False, error=f"AUTH_MISSING ({spec['auth_env']} nicht gesetzt)",
                                         auth_missing=True)
                if spec.get("auth_header") == "bearer":
                    headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
                else:
                    params[spec.get("auth_param", "token")] = key
            r = fetch(spec["url"], params=params or None, headers=headers, method=spec.get("method", "GET"),
                      json_body=spec.get("json_body"), timeout=int(spec.get("timeout", 20)), retries=3, backoff=2.0)
            lat = round(time.monotonic() - t0, 2)
            if kind == "http_csv":
                return _parse_csv_probe(r, spec, lat)
            try:
                payload = r.json()
            except ValueError as e:
                return _probe_result(reachable=True, auth_ok=True, http_status=getattr(r, "status", 200), latency_s=lat,
                                     schema_ok=False, error=f"kein JSON: {e}"[:200])
            return _check_json(payload, spec, lat, getattr(r, "status", 200))
        if kind == "yfinance_history":
            yf = yf_module or __import__("yfinance")
            h = yf.Ticker(spec["symbol"]).history(period="10d")
            lat = round(time.monotonic() - t0, 2)
            if h is None or len(h) == 0:
                return _probe_result(reachable=True, auth_ok=True, latency_s=lat, schema_ok=True, empty=True,
                                     error="leere Kurshistorie")
            if "Close" not in h:
                return _probe_result(reachable=True, auth_ok=True, latency_s=lat, schema_ok=False, error="Spalte Close fehlt")
            close = pd.to_numeric(h["Close"], errors="coerce")
            ok_rows = close[close.notna() & (close > 0)]
            last = _ts(ok_rows.index.max()) if len(ok_rows) else None
            return _probe_result(ok=last is not None, reachable=True, auth_ok=True, latency_s=lat, schema_ok=True,
                                 empty=len(ok_rows) == 0, latest_observation=_iso(last), n_items=int(len(h)),
                                 missing_rate=round(1 - len(ok_rows) / len(h), 3))
        if kind == "yfinance_options":
            yf = yf_module or __import__("yfinance")
            exps = yf.Ticker(spec["symbol"]).options
            lat = round(time.monotonic() - t0, 2)
            n = len(exps or ())
            return _probe_result(ok=n > 0, reachable=True, auth_ok=True, latency_s=lat, schema_ok=True, empty=n == 0,
                                 n_items=n, latest_observation=_iso(now) if n else None,
                                 error=None if n else "keine Verfallstermine")
        return _probe_result(error=f"unbekannte Probe-Art {kind}")
    except Exception as e:  # noqa: BLE001 – jede Störung wird klassifiziert, nie verschluckt
        return _probe_result(latency_s=round(time.monotonic() - t0, 2), **_classify_exception(e))


def _check_json(payload, spec: dict, lat: float, status) -> dict:
    req = spec.get("required_keys") or []
    if spec.get("required_type") == "list":
        if not isinstance(payload, list):
            return _probe_result(reachable=True, auth_ok=True, http_status=status, latency_s=lat, schema_ok=False,
                                 error=f"Liste erwartet, {type(payload).__name__} erhalten")
        return _probe_result(ok=len(payload) > 0, reachable=True, auth_ok=True, http_status=status, latency_s=lat,
                             schema_ok=True, empty=len(payload) == 0, n_items=len(payload),
                             error=None if payload else "leere Antwort")
    if not isinstance(payload, dict) or any(k not in payload for k in req):
        got = sorted(payload)[:8] if isinstance(payload, dict) else type(payload).__name__
        return _probe_result(reachable=True, auth_ok=True, http_status=status, latency_s=lat, schema_ok=False,
                             error=f"Schema: erwartet {req}, erhalten {got}")
    vals = [payload[k] for k in req]
    empty = any(v in (None, "", [], {}) for v in vals)
    n = next((len(v) for v in vals if isinstance(v, (list, dict))), None)
    latest = None
    ld = spec.get("latest_date")                      # {list, date, value}: jüngster gültiger Wert
    if ld and isinstance(payload.get(ld["list"]), list):
        dates = [_ts(o.get(ld["date"])) for o in payload[ld["list"]]
                 if isinstance(o, dict) and str(o.get(ld.get("value", "value"), "")).strip() not in ("", ".")]
        dates = [d for d in dates if d is not None]
        latest = _iso(max(dates)) if dates else None
        if not dates:
            empty = True
    return _probe_result(ok=not empty, reachable=True, auth_ok=True, http_status=status, latency_s=lat, schema_ok=True,
                         empty=empty, n_items=n, latest_observation=latest, error="leere Antwort" if empty else None)


def _parse_csv_probe(r, spec: dict, lat: float) -> dict:
    text = r.content.decode("utf-8", errors="replace") if hasattr(r, "content") else str(r)
    rows = [l.split(",") for l in text.strip().splitlines()[1:] if "," in l]
    if not rows:
        return _probe_result(reachable=True, auth_ok=True, latency_s=lat, schema_ok=True, empty=True, error="leere CSV")
    valid = [x for x in rows if len(x) > int(spec.get("value_col", 1))
             and x[int(spec.get("value_col", 1))].strip() not in ("", ".")]
    if not valid:
        return _probe_result(reachable=True, auth_ok=True, latency_s=lat, schema_ok=True, empty=True,
                             n_items=len(rows), missing_rate=1.0, error="nur fehlende Werte")
    last = _ts(valid[-1][int(spec.get("date_col", 0))])
    if last is None:
        return _probe_result(reachable=True, auth_ok=True, latency_s=lat, schema_ok=False, error="Datumsspalte unlesbar")
    return _probe_result(ok=True, reachable=True, auth_ok=True, latency_s=lat, schema_ok=True, empty=False,
                         latest_observation=_iso(last), n_items=len(rows),
                         missing_rate=round(1 - len(valid) / len(rows), 3))


# ── Store-Statistik (Duplikate, Missing, Ausreißer, Plausibilität) ───────────
def store_stats(directory: Path | str, now: datetime, cfg: dict, window_days: int = 120, **kw) -> dict:
    d = Path(directory)
    parts = sorted(d.glob("*.csv.gz")) if d.exists() else []
    if not parts:
        return {"exists": False}
    df = pd.concat([pd.read_csv(p, dtype=str) for p in parts[-2:]], ignore_index=True)
    return frame_stats(df, now, cfg, window_days, **kw)


NEW_ROWS_DAYS = 3        # Datenfehler gelten als NEU, wenn die Zeile in den letzten Tagen abgerufen wurde


def frame_stats(df: pd.DataFrame, now: datetime, cfg: dict, window_days: int = 120, *,
                value_optional: bool = False, log_values: bool = False, outlier_check: bool = True,
                quarantined_parser_versions: tuple = ()) -> dict:
    """value_optional: fehlende Werte sind fachlich vorgesehen (z.B. Zuschlag ohne EUR-Betrag) ->
    keine Missing-Rate. log_values: rechtsschiefe Beträge -> Ausreißer auf log1p(|x|)-Skala."""
    if df is None or not len(df):
        return {"exists": True, "n_rows": 0}
    th = cfg["thresholds"]
    avail = pd.to_datetime(df.get("available_at"), utc=True, errors="coerce")
    retr = pd.to_datetime(df.get("retrieved_at"), utc=True, errors="coerce")
    obs = pd.to_datetime(df.get("observation_time"), utc=True, errors="coerce")
    recent = df[avail >= pd.Timestamp(now) - pd.Timedelta(days=window_days)] if avail is not None else df
    vals = pd.to_numeric(recent.get("value"), errors="coerce") if "value" in recent else pd.Series(dtype=float)
    dup = float(df["series_id"].duplicated().mean()) if "series_id" in df else 0.0
    future = int((avail > retr + pd.Timedelta(days=1)).sum())                     # PIT-Verletzung
    fut_mask = obs > pd.Timestamp(now) + pd.Timedelta(days=1)                      # Datenfehler (Periode in der Zukunft)
    future_obs_total = int(fut_mask.sum())
    # Alte, bekannte Fehlzeilen liegen append-only im Store, werden aber vom Verbraucher ignoriert
    # (Quarantäne) -> nur NEU abgerufene zählen für den Status; Gesamtzahl bleibt als Info sichtbar.
    if quarantined_parser_versions and "parser_version" in df:   # Defekt vom aktuellen Parser behoben, Verbraucher ignoriert
        fut_mask_new = fut_mask & ~df["parser_version"].astype(str).isin(set(quarantined_parser_versions))
    else:
        fut_mask_new = fut_mask
    future_obs = int((fut_mask_new & (retr >= pd.Timestamp(now) - pd.Timedelta(days=NEW_ROWS_DAYS))).sum())
    out_rate = 0.0
    if outlier_check and len(vals.dropna()) >= 20 and "metric" in recent:
        zs = []
        for _, g in recent.assign(_v=vals).dropna(subset=["_v"]).groupby("metric"):
            v = g["_v"].to_numpy()
            if log_values:
                v = np.sign(v) * np.log1p(np.abs(v))
            mad = np.median(np.abs(v - np.median(v)))
            if mad > 0:
                zs.append(np.abs(v - np.median(v)) / (1.4826 * mad) > th["outlier_z"])
        out_rate = float(np.concatenate(zs).mean()) if zs else 0.0
    return {"exists": True, "n_rows": int(len(df)), "n_recent": int(len(recent)),
            "latest_available": _iso(_ts(avail.max())) if avail is not None and avail.notna().any() else None,
            "latest_observation": _iso(_ts(obs[obs <= pd.Timestamp(now) + pd.Timedelta(days=1)].max()))
            if obs is not None and obs.notna().any() else None,
            "latest_retrieved": _iso(_ts(retr.max())) if retr is not None and retr.notna().any() else None,
            "missing_rate": None if value_optional else (round(float(vals.isna().mean()), 4) if len(vals) else None),
            "duplicate_rate": round(dup, 4), "future_timestamps": future, "outlier_rate": round(out_rate, 4),
            "future_observations": future_obs, "future_observation_rate": round(future_obs / len(df), 6),
            "future_observations_quarantined": future_obs_total - future_obs}


# ── Quelleninventar + Checks ────────────────────────────────────────────────
def _effective_criticality(declared: str, decisions: list) -> str:
    """CRITICAL nur, wenn die Quelle einen produktiven Entscheidungspfad speist;
    reine SHADOW-/Research-Quellen höchstens IMPORTANT."""
    if declared == "CRITICAL" and not decisions:
        return "IMPORTANT"
    return declared


def downstream(cfg: dict) -> dict[str, dict]:
    """source_id -> {features, models, hypotheses, decisions, other} (aus Registries abgeleitet)."""
    out: dict[str, dict] = {}

    def slot(s):
        return out.setdefault(s, {"features": [], "models": [], "hypotheses": [], "decisions": [], "other": []})
    for c in cfg.get("core_sources") or []:
        s = slot(c["source_id"])
        s["features"] += c.get("features") or []
        s["decisions"] += c.get("decisions") or []
    try:
        from modules.alt_data import contracts as ac
        from modules.alt_data.registry import ALT_FEATURES, SOURCES
        alt_c = ac.load()
        try:
            from modules import ml_research as ml
            specs = ml.load_registry().get("models") or []
        except Exception:  # noqa: BLE001 – Registry optional für die Abhängigkeitsliste
            specs = []
        for sid, src in SOURCES.items():
            fc = src.get("feature_contracts")
            for cid in src["contracts"]:
                s = slot(cid)
                feats = [f for f in src["features"] if cid in fc.get(f, [])] if fc else list(src["features"])
                s["features"] += feats
                s["other"].append(f"feature_store:{sid}")
                s["models"] += [m["id"] for m in specs if set(m.get("extra_features") or []) & set(feats)]
                s["hypotheses"] += [c["hypothesis_id"] for c in alt_c
                                    if any(ALT_FEATURES.get(f, {}).get("source") == sid for f in c.get("source_features") or [])]
    except Exception as e:  # noqa: BLE001
        log.warning(f"source_health: Alt-Data-Abhängigkeiten nicht ableitbar ({e})")
    try:
        from modules import ml_research as _ml
        _specs = _ml.load_registry().get("models") or []
    except Exception:  # noqa: BLE001 – Registry optional für die Abhängigkeitsliste
        _specs = []

    def _uses(m, feats):
        if m.get("model") == "rule":
            return m.get("rule_feature") in feats
        fs = m.get("features")
        if fs in ("all", None):
            return True
        if isinstance(fs, str):
            fs = (_ml.FEATURE_GROUPS.get(fs) if hasattr(_ml, "FEATURE_GROUPS") else None) or [fs]
        return bool(set(fs or []) & set(feats))
    for sid, feats in (cfg.get("feature_sources") or {}).items():
        slot(sid)["features"] += list(feats)
        slot(sid)["models"] += [m["id"] for m in _specs if _uses(m, feats)]
    for sid, extra in (cfg.get("downstream_extra") or {}).items():
        slot(sid)["other"] += list(extra)
    for v in out.values():
        for k in v:
            v[k] = sorted(set(v[k]))
    return out


def gather(cfg: dict, now: datetime, *, probes: bool = True, fetch=None, yf_module=None, env=None,
           registry_health: dict | None = None, contracts: dict | None = None) -> list[dict]:
    """-> je Quelle ein Check-Datensatz (noch ohne Status)."""
    from modules.external.registry import MAX_AGE_DAYS_BY_FREQUENCY, evaluate_staleness, gate_source
    if contracts is None:
        from modules.alt_data.registry import load_contracts
        contracts = load_contracts()
    if registry_health is None:
        registry_health = json.loads(REGISTRY_HEALTH.read_text()) if REGISTRY_HEALTH.exists() else {}
    checks = []
    for c in cfg.get("core_sources") or []:
        pr = run_probe(c["probe"], now, fetch=fetch, yf_module=yf_module, env=env) if probes else None
        checks.append({"source_id": c["source_id"], "kind": "core", "criticality": c["criticality"],
                       "fallback": c.get("fallback"), "fallback_only": bool(c.get("fallback_only")),
                       "probe": pr, "latest_observation": (pr or {}).get("latest_observation"),
                       "last_success": _iso(now) if pr and pr.get("ok") else None,
                       "max_observation_age_days": c.get("max_observation_age_days"),
                       "expected_freshness": f"{c.get('max_observation_age_days', 'n/a')} T (Probe täglich)",
                       "missing_rate": (pr or {}).get("missing_rate")})
    cmap = cfg["criticality_map"]
    alt_ids = {a["source_id"] for a in cfg.get("alt_sources") or []}
    for a in cfg.get("alt_sources") or []:
        con = contracts.get(a["source_id"]) or {}
        pr = run_probe(a["probe"], now, fetch=fetch, yf_module=yf_module, env=env) if probes and a.get("probe") else None
        hf = Path(a["health_file"]) if a.get("health_file") else None
        h = json.loads(hf.read_text()) if hf and hf.exists() else None
        st = store_stats(a["store"], now, cfg, value_optional=bool(a.get("value_optional")),
                         log_values=a.get("value_scale") == "log",
                         outlier_check=a.get("outlier_check", True),
                         quarantined_parser_versions=tuple(a.get("quarantined_parser_versions") or ())) \
            if a.get("store") else None
        last_ok = None
        if h:
            if a["source_id"] == "gleif_lei":
                last_ok = _ts(h.get("date")) if not h.get("errors") or len(h.get("errors")) < 50 else None
            else:
                last_ok = _ts(h.get("last_success") or (h.get("checked_at") if (h.get("error_rate") or 0) < 1 else None))
        latest = (st or {}).get("latest_observation") or (h or {}).get("last_observation") or (h or {}).get("last_observation")
        checks.append({"source_id": a["source_id"], "kind": "alt", "criticality": a.get("criticality") or
                       cmap.get(str(con.get("criticality", "low")), "NON_CRITICAL"), "probe": pr, "health": h and {
                           k: h.get(k) for k in ("error_rate", "coverage", "schema_errors", "budget_exhausted",
                                                 "errors_sample", "checked_at")},
                       "store": st, "last_success": _iso(last_ok), "latest_observation": latest,
                       "max_observation_age_days": a.get("max_observation_age_days"),
                       "max_success_age_days": a.get("max_success_age_days"),
                       "expected_freshness": f"Beobachtung <= {a.get('max_observation_age_days', 'n/a')} T, "
                                             f"Ingest <= {a.get('max_success_age_days')} T",
                       "schema_version": (h or {}).get("schema_version") or (h or {}).get("parser_version"),
                       "missing_rate": (st or {}).get("missing_rate"), "coverage": (h or {}).get("coverage")})
    for sid, con in contracts.items():
        if sid in alt_ids:
            continue
        rh = registry_health.get(sid) or {}
        fetchable, reason = gate_source(con)
        freq = str(con.get("frequency") or "")
        max_age = con.get("max_staleness_days") or MAX_AGE_DAYS_BY_FREQUENCY.get(freq)
        stale = evaluate_staleness(_ts(rh.get("latest_observation")), con.get("expected_update_cadence"), now,
                                   freq, con.get("max_staleness_days")) if rh.get("latest_observation") else "UNKNOWN"
        dq = rh.get("dq") or {}
        checks.append({"source_id": sid, "kind": "registry",
                       "criticality": cmap.get(str(con.get("criticality", "low")), "NON_CRITICAL"),
                       "disabled": not fetchable and reason != "AUTH_MISSING", "disabled_reason": None if fetchable else reason,
                       # Auth-Befund des Orchestrators zählt (er läuft mit den Secrets), nicht die lokale Umgebung
                       "auth_missing": rh.get("status") == "AUTH_MISSING" or (reason == "AUTH_MISSING" and not rh),
                       "registry_status": rh.get("status"), "registry_message": str(rh.get("message") or "")[:200],
                       "last_success": rh.get("last_success"), "last_attempt": rh.get("last_attempt"),
                       "latest_observation": rh.get("latest_observation"), "staleness": stale,
                       "max_observation_age_days": max_age,
                       "max_success_age_days": SUCCESS_AGE_DAYS_BY_FREQUENCY.get(freq.split(" ")[0], 3),
                       "expected_freshness": f"{freq or 'n/a'}: Beobachtung <= {max_age or '2×Kadenz'} T, Ingest täglich",
                       "registry_failures": int(rh.get("consecutive_failures") or 0),
                       "missing_rate": dq.get("null_rate", rh.get("missingness")),
                       "store": {"duplicate_rate": None, "future_timestamps": int(dq.get("future_observations") or 0),
                                 "outlier_rate": (dq.get("out_of_range") or 0) / max(1, dq.get("n_observations") or 1),
                                 "duplicate_conflicts": int(dq.get("duplicate_conflicts") or 0), "severe": dq.get("severe")},
                       "schema_version": con.get("schema_version"),
                       # Research-Quellen (z. B. Commodity Intelligence): Ausfall macht nur ihre Features
                       # unavailable, zählt aber nie für den globalen Safe Mode
                       "research_only": bool(con.get("research_only"))})
    return checks


# ── Klassifikation inkl. Recovery ───────────────────────────────────────────
def classify(chk: dict, prev: dict | None, cfg: dict, now: datetime) -> dict:
    th = cfg["thresholds"]
    reasons, raw = [], HEALTHY
    pr = chk.get("probe")
    st = chk.get("store") or {}

    def worse(s):
        nonlocal raw
        if SEVERITY[s] > SEVERITY[raw]:
            raw = s
    failed = False
    if chk.get("disabled"):
        return {"status": UNVALIDATED, "raw_status": UNVALIDATED, "reasons": [f"nicht aktiv: {chk.get('disabled_reason')}"],
                "failed": False, "consecutive_failures": 0, "recovery_passes": 0}
    if chk.get("auth_missing") or (pr and pr.get("auth_missing")):
        reasons.append("AUTH_MISSING: Zugangsdaten fehlen (Env/Secret)")
        worse(UNVALIDATED)
    if pr is not None and not pr.get("auth_missing"):
        if pr.get("auth_ok") is False:
            reasons.append(f"Authentifizierung fehlgeschlagen: {pr.get('error')}")
            worse(BROKEN)
            failed = True
        elif pr.get("schema_ok") is False:
            reasons.append(f"Schema: {pr.get('error')}")
            worse(BROKEN)
            failed = True
        elif pr.get("rate_limited"):
            reasons.append("Rate Limit (429) trotz Backoff")
            worse(DEGRADED)
            failed = True
        elif pr.get("timeout"):
            reasons.append(f"Timeout: {pr.get('error')}")
            worse(DEGRADED)
            failed = True
        elif pr.get("reachable") is False or (pr.get("reachable") is None and not pr.get("ok")):
            reasons.append(f"nicht erreichbar: {pr.get('error')}")
            worse(BROKEN)
            failed = True
        elif pr.get("empty"):
            reasons.append(f"leere Antwort: {pr.get('error')}")
            worse(DEGRADED)
            failed = True
        elif not pr.get("ok"):
            reasons.append(f"Probe nicht ok: {pr.get('error')}")
            worse(DEGRADED)
            failed = True
        if pr.get("latency_s") is not None and pr["latency_s"] > th["latency_degraded_s"]:
            reasons.append(f"Antwortzeit {pr['latency_s']} s > {th['latency_degraded_s']} s")
            worse(DEGRADED)
        if (pr.get("missing_rate") or 0) > th["missing_rate_degraded"]:
            reasons.append(f"Missing-Rate {pr['missing_rate']:.0%}")
            worse(DEGRADED)
    rs = chk.get("registry_status")
    if rs:
        if rs == "SCHEMA_CHANGED":
            reasons.append("Schema geändert (Orchestrator)")
            worse(BROKEN)
            failed = True
        elif rs in ("FAIL", "ERROR", "STALE"):
            reasons.append(f"Ingest {rs}: {chk.get('registry_message')}")
            worse(BROKEN if (chk.get("registry_failures") or 0) >= th["broken_after_consecutive_failures"] else DEGRADED)
            failed = True
        elif rs == "WARN":
            reasons.append(f"Ingest WARN: {chk.get('registry_message')}")
            worse(DEGRADED)
        elif rs in ("DEFERRED", "BLOCKED", "REVIEW_REQUIRED") and not chk.get("last_success"):
            return {"status": UNVALIDATED, "raw_status": UNVALIDATED, "reasons": [f"Orchestrator: {rs}"],
                    "failed": False, "consecutive_failures": 0, "recovery_passes": 0}
    h = chk.get("health") or {}
    if (h.get("error_rate") or 0) > th["error_rate_degraded"]:
        reasons.append(f"Fehlerquote letzter Ingest {h['error_rate']:.0%}")
        worse(DEGRADED)
    if h.get("schema_errors"):
        reasons.append(f"{h['schema_errors']} Schema-Fehler im letzten Ingest")
        worse(DEGRADED)
    # Plausibilität / Integrität
    if st.get("future_timestamps"):
        reasons.append(f"PIT-Verletzung: {st['future_timestamps']} Beobachtungen mit available_at > retrieved_at")
        worse(BROKEN)
        failed = True
    if st.get("future_observations"):                 # einzelne fehlerhafte Fakten: Qualitätsmangel, kein Totalausfall
        reasons.append(f"{st['future_observations']} Beobachtungen mit Periode in der Zukunft (Datenfehler)")
        worse(BROKEN if (st.get("future_observation_rate") or 0) > th["missing_rate_degraded"] else DEGRADED)
    if st.get("severe"):
        reasons.append("Data-Quality-Befund schwerwiegend")
        worse(DEGRADED)
    if (st.get("duplicate_rate") or 0) > th["duplicate_rate_degraded"] or st.get("duplicate_conflicts"):
        reasons.append(f"Duplikate {st.get('duplicate_rate')} / Konflikte {st.get('duplicate_conflicts')}")
        worse(DEGRADED)
    if (st.get("outlier_rate") or 0) > th["outlier_rate_degraded"]:
        reasons.append(f"Ausreißer-Anteil {st['outlier_rate']:.1%}")
        worse(DEGRADED)
    if (chk.get("missing_rate") or 0) > th["missing_rate_degraded"] and not pr:
        reasons.append(f"Missing-Rate {chk['missing_rate']:.0%}")
        worse(DEGRADED)
    # Freshness (quellenabhängig)
    lo, ls = _ts(chk.get("latest_observation")), _ts(chk.get("last_success"))
    mo, ms = chk.get("max_observation_age_days"), chk.get("max_success_age_days")
    if chk.get("staleness") == "STALE":
        reasons.append("veraltet (Frequenz-Grenze der Registry)")
        worse(STALE)
    elif lo is not None and mo and _age_days(lo, now) > float(mo):
        reasons.append(f"jüngste Beobachtung {_age_days(lo, now)} T alt > {mo} T")
        worse(STALE)
    if ms and (ls is None or _age_days(ls, now) > float(ms)):
        reasons.append(f"letzte erfolgreiche Aktualisierung {_age_days(ls, now) if ls else 'nie'} T > {ms} T")
        worse(STALE if ls else UNVALIDATED)
    if lo is None and ls is None and pr is None:
        reasons.append("kein Beleg für erfolgreiche Lieferung")
        worse(UNVALIDATED)
    # neue Daten seit dem letzten Check?
    new_data = None
    if prev:
        pl = _ts(prev.get("latest_observation"))
        new_data = bool(lo and (pl is None or lo > pl))
    cons = (int((prev or {}).get("consecutive_failures") or 0) + 1) if failed else 0
    if failed and cons >= th["broken_after_consecutive_failures"] and raw != UNVALIDATED:
        reasons.append(f"{cons} Fehlschläge in Folge")
        raw = BROKEN
    status, rec = raw, 0
    prev_status = (prev or {}).get("status")
    if raw == HEALTHY and (prev_status in (STALE, BROKEN) or (prev or {}).get("recovering")):
        rec = int((prev or {}).get("recovery_passes") or 0) + 1
        if rec < th["recovery_passes_required"]:
            status = DEGRADED
            reasons.append(f"RECOVERING {rec}/{th['recovery_passes_required']}: vollständiger Abruf ok, "
                           f"Bestätigung ausstehend")
    return {"status": status, "raw_status": raw, "reasons": reasons, "failed": failed, "consecutive_failures": cons,
            "recovery_passes": rec if status == DEGRADED and rec else 0, "recovering": status == DEGRADED and rec > 0,
            "new_data": new_data}


# ── Snapshot, Fallbacks, Features, Safe Mode ────────────────────────────────
def build_snapshot(checks: list[dict], prev_snap: dict | None, cfg: dict, now: datetime,
                   deps: dict | None = None) -> dict:
    deps = deps if deps is not None else downstream(cfg)
    prev_src = (prev_snap or {}).get("sources") or {}
    srcs = {}
    for chk in checks:
        sid = chk["source_id"]
        prev = prev_src.get(sid)
        c = classify(chk, prev, cfg, now)
        d = deps.get(sid) or {"features": [], "models": [], "hypotheses": [], "decisions": [], "other": []}
        crit = _effective_criticality(chk["criticality"], d.get("decisions") or [])
        pr = chk.get("probe") or {}
        srcs[sid] = {
            "source_id": sid, "kind": chk["kind"], "status": c["status"], "raw_status": c["raw_status"],
            "criticality": crit, "checked_at": _iso(now),
            "last_success": chk.get("last_success") or (prev or {}).get("last_success"),
            "latest_observation": chk.get("latest_observation"),
            "data_age_days": _age_days(_ts(chk.get("latest_observation")), now),
            "expected_freshness": chk.get("expected_freshness"), "latency": pr.get("latency_s"),
            "http_status": pr.get("http_status"), "rate_limited": pr.get("rate_limited"),
            "missing_rate": chk.get("missing_rate"), "coverage": chk.get("coverage"),
            "schema_version": chk.get("schema_version"), "new_data": c.get("new_data"),
            "error": "; ".join(c["reasons"]) or None, "reasons": c["reasons"],
            "consecutive_failures": c["consecutive_failures"], "recovery_passes": c["recovery_passes"],
            "recovering": c.get("recovering", False), "downstream_dependencies": d,
            "fallback": chk.get("fallback"), "fallback_only": chk.get("fallback_only", False),
            "store": chk.get("store"), "failed_since": None, "research_only": bool(chk.get("research_only"))}
        s = srcs[sid]
        if c["status"] in UNHEALTHY and (prev or {}).get("status") not in UNHEALTHY:
            s["failed_since"] = _iso(now)
        elif c["status"] != HEALTHY:
            s["failed_since"] = (prev or {}).get("failed_since") or (_iso(now) if c["status"] in UNHEALTHY else None)
    # Fallback-Provider: nur explizit deklariert, nur wenn selbst HEALTHY; Wechsel sichtbar
    fallbacks = []
    for sid, s in srcs.items():
        s["source_primary"], s["source_actual"], s["fallback_active"] = sid, sid, False
        fb = s.get("fallback")
        if fb and s["status"] != HEALTHY and srcs.get(fb, {}).get("status") == HEALTHY:
            s.update(source_actual=fb, fallback_active=True)
            fallbacks.append({"primary": sid, "actual": fb, "reason": s["error"]})
        elif fb and s["status"] != HEALTHY:
            s["source_actual"] = None
    snap = {"generated": _iso(now), "config_version": cfg["version"], "sources": srcs, "fallbacks": fallbacks}
    snap["features"] = feature_availability(snap, cfg, now)
    snap["safe_mode"] = data_safe_mode(snap, cfg)
    snap["counts"] = {st: sum(1 for s in srcs.values() if s["status"] == st) for st in STATUSES}
    return snap


def _usable(s: dict) -> bool:
    return s["status"] in (HEALTHY, DEGRADED) or bool(s.get("fallback_active"))


def feature_availability(snap: dict, cfg: dict, now: datetime) -> dict:
    """Je Feature: available / stale / data_quality / source_primary / source_actual.
    Cache nur mit expliziter zulässiger Datenalterung (cache_max_age_days); nie 0 als Ersatz."""
    q = cfg["status_quality"]
    cache = cfg.get("cache_max_age_days") or {}
    try:
        from modules.alt_data.registry import SOURCES as ALT
    except Exception:  # noqa: BLE001
        ALT = {}
    feat_source = {}
    for sid, src in ALT.items():
        fc = src.get("feature_contracts")
        for cid in src["contracts"]:
            for f in src["features"]:
                if fc and cid not in fc.get(f, []):
                    continue
                feat_source.setdefault(f, []).append((cid, sid))
    out = {}
    for sid, s in snap["sources"].items():
        for f in s["downstream_dependencies"].get("features") or []:
            groups = [g for _, g in feat_source.get(f, [])] or [sid]
            max_age = max((cache.get(g) for g in groups if cache.get(g)), default=None) or cache.get(sid)
            age = s.get("data_age_days")
            prevf = out.get(f)
            if _usable(s):
                entry = {"available": True, "stale": False, "data_quality": q[s["status"]] if not s.get("fallback_active")
                         else q[HEALTHY] * 0.8}
            elif max_age and age is not None and age <= float(max_age):
                entry = {"available": True, "stale": True, "data_quality": round(0.5 * q[HEALTHY], 2),
                         "cache_age_days": age, "cache_max_age_days": max_age}
            else:
                entry = {"available": False, "stale": None, "data_quality": 0.0}
            entry.update(source_primary=s["source_primary"], source_actual=s["source_actual"],
                         source_status=s["status"], criticality=s["criticality"])
            # mehrere Verträge je Feature: das schwächste Glied zählt
            if prevf is None or entry["data_quality"] < prevf["data_quality"]:
                out[f] = entry
    return out


def data_safe_mode(snap: dict, cfg: dict) -> dict:
    g = cfg["thresholds"]["global_safe_mode"]
    srcs = snap["sources"]
    blocked, disabled_signals, disabled_feats, reasons = set(), set(), set(), []
    for sid, s in srcs.items():
        if s.get("fallback_only") or _usable(s):
            continue
        dep = s["downstream_dependencies"]
        if s["criticality"] == "CRITICAL":
            blocked |= set(dep.get("decisions") or [])
            reasons.append(f"CRITICAL {sid} {s['status']}: Entscheidungspfade {dep.get('decisions')} blockiert")
        if s["criticality"] in ("CRITICAL", "IMPORTANT"):
            disabled_signals |= set(dep.get("hypotheses") or [])
        disabled_feats |= set(dep.get("features") or [])
    imp = [sid for sid, s in srcs.items() if s["criticality"] in ("CRITICAL", "IMPORTANT") and not s.get("fallback_only")
           and not s.get("research_only")
           and s["status"] in (STALE, BROKEN) and not s.get("fallback_active")]
    w = [(CRIT_WEIGHT[s["criticality"]], cfg["status_quality"][s["status"]] if not s.get("fallback_active") else 0.8)
         for s in srcs.values() if not s.get("fallback_only") and not s.get("research_only")
         and not (s["status"] == UNVALIDATED and s["kind"] == "registry")]
    dq = round(sum(a * b for a, b in w) / sum(a for a, _ in w), 3) if w else 0.0
    glob = []
    if blocked:
        glob.append(f"kritische Pflichtdaten fehlen: {sorted(blocked)}")
    if len(imp) >= g["max_unhealthy_important"]:
        glob.append(f"{len(imp)} unabhängige wichtige Quellen ausgefallen: {sorted(imp)}")
    if dq < g["min_data_quality"]:
        glob.append(f"gewichtete Data Quality {dq} < {g['min_data_quality']}")
    return {"active": bool(glob), "global_reasons": glob, "reasons": reasons, "data_quality": dq,
            "blocked_decisions": sorted(blocked), "disabled_signals": sorted(disabled_signals),
            "unavailable_features": sorted(f for f, v in snap.get("features", {}).items() if not v["available"]),
            "stale_features": sorted(f for f, v in snap.get("features", {}).items() if v.get("stale")),
            "effects": ["keine neuen High-Confidence-Labels", "keine positiven Intelligence-Boosts",
                        "keine Promotion neuer Hypothesen", "keine Signale mit fehlenden Pflichtdaten",
                        "Champion nur mit HEALTHY-Pflichtdaten"] if glob else []}


# ── Leser für Verbraucher (Scanner, HC, Adapter, PromotionController) ────────
def load_snapshot(path: Path | None = None, now: datetime | None = None, cfg: dict | None = None) -> dict:
    """Snapshot oder {"unknown": True,...}. Zu alt/fehlend/unlesbar = unbekannt (nie 'verfügbar')."""
    path = path or SNAPSHOT
    now = now or _now()
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"unknown": True, "reason": f"kein lesbarer Health-Snapshot ({path})"}
    max_h = ((cfg or {}).get("thresholds") or {}).get("snapshot_max_age_hours", 30)
    gen = _ts(snap.get("generated"))
    if gen is None or (now - gen).total_seconds() > max_h * 3600:
        return {"unknown": True, "reason": f"Health-Snapshot veraltet ({snap.get('generated')})", "stale_snapshot": snap}
    return snap


def effective_safe_mode(out_dir: Path | None = None, now: datetime | None = None,
                        snapshot_path: Path | None = None, meta_path: Path | None = None) -> dict:
    """Sicht auf den KANONISCHEN SystemState (modules/system_state.py) – keine eigene Logik mehr.
    Pfade nur für Tests/abweichende Ablagen; Produktion nutzt die Standard-Eingaben (persistiert)."""
    from modules import system_state as ss
    inputs = {}
    if out_dir is not None:
        inputs["model_health"] = Path(out_dir) / "safe_mode.json"
        inputs["meta_learning"] = Path(out_dir) / "meta_learning.json"
    if meta_path is not None:
        inputs["model_health"] = Path(meta_path)
    if snapshot_path is not None:
        inputs["data_health"] = Path(snapshot_path)
    default = all(Path(v) == ss.DEFAULT_INPUTS[k] for k, v in inputs.items())
    return ss.safe_mode_view(ss.current(inputs=inputs or None, now=now, persist=default))


def scanner_preflight(now: datetime | None = None, *, cfg: dict | None = None, live=None) -> dict:
    """Scanner: liest den Snapshot; fehlt/veraltet er, werden NUR die Champion-Pflichtdaten live geprüft.
    -> {proceed, blocked_decisions, safe_mode, data_quality, reasons, source}"""
    now = now or _now()
    cfg = cfg or load_config()
    snap = load_snapshot(now=now, cfg=cfg)
    source = "snapshot"
    if snap.get("unknown"):
        source = "live_core_probe"
        core_cfg = {**cfg, "alt_sources": []}
        checks = (live or gather)(core_cfg, now, registry_health={}, contracts={})
        snap = build_snapshot(checks, None, core_cfg, now, deps=downstream(core_cfg))
        snap["safe_mode"]["global_reasons"] = (snap["safe_mode"].get("global_reasons") or []) + \
            ["Health-Snapshot fehlt/veraltet – nur Pflichtdaten live geprüft, Intelligence aus"]
        snap["safe_mode"]["active"] = True
    sm = snap["safe_mode"]
    core_blocked = [d for d in sm.get("blocked_decisions") or [] if d in ("scanner_candidates", "risk_gates",
                                                                         "options_design")]
    return {"proceed": not core_blocked, "blocked_decisions": sm.get("blocked_decisions") or [],
            "safe_mode": bool(sm.get("active")), "data_quality": sm.get("data_quality"),
            "reasons": (sm.get("global_reasons") or []) + (sm.get("reasons") or []), "source": source,
            "fallbacks": snap.get("fallbacks") or []}


# ── Historie, Report, Benachrichtigung ──────────────────────────────────────
def append_history(snap: dict, path: Path | None = None) -> int:
    path = path or HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for sid, s in snap["sources"].items():
            fh.write(json.dumps({"checked_at": snap["generated"], "source_id": sid, "status": s["status"],
                                 "consecutive_failures": s["consecutive_failures"], "latency": s["latency"],
                                 "latest_observation": s["latest_observation"], "fallback_active": s["fallback_active"],
                                 "error": (s["error"] or "")[:200] or None}, ensure_ascii=False) + "\n")
    return len(snap["sources"])


def instability(history_path: Path | None = None, days: int = 30, now: datetime | None = None) -> dict:
    """Wiederkehrende Instabilität: Statuswechsel und Nicht-HEALTHY-Anteil je Quelle (letzte `days` Tage)."""
    path = history_path or HISTORY
    if not path.exists():
        return {}
    now = now or _now()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if _ts(r["checked_at"]) and (now - _ts(r["checked_at"])).days <= days]
    out = {}
    for sid, g in pd.DataFrame(rows).groupby("source_id") if rows else []:
        st = g.sort_values("checked_at")["status"].tolist()
        flips = sum(1 for a, b in zip(st, st[1:]) if a != b)
        bad = sum(1 for s in st if s != HEALTHY) / len(st)
        out[sid] = {"checks": len(st), "flips": flips, "unhealthy_share": round(bad, 3),
                    "recurring": flips >= 4 or (0.2 <= bad < 1 and flips >= 2)}
    return out


def changes(prev: dict | None, snap: dict) -> list[dict]:
    """Relevante Änderungen (Mail-Auslöser): Statuswechsel von/zu HEALTHY, neue Fehler,
    Fallback an/aus, Safe Mode an/aus, Recovery abgeschlossen."""
    p = (prev or {}).get("sources") or {}
    out = []
    for sid, s in snap["sources"].items():
        ps = p.get(sid, {}).get("status")
        if ps is None:
            if s["status"] in (STALE, BROKEN):
                out.append({"source_id": sid, "type": "new_failure", "from": None, "to": s["status"], "error": s["error"]})
            continue
        if ps != s["status"] and (ps == HEALTHY or s["status"] == HEALTHY or s["status"] in (STALE, BROKEN)):
            kind = "recovery" if s["status"] == HEALTHY else "degradation"
            out.append({"source_id": sid, "type": kind, "from": ps, "to": s["status"], "error": s["error"]})
        if bool(p.get(sid, {}).get("fallback_active")) != bool(s.get("fallback_active")):
            out.append({"source_id": sid, "type": "fallback_on" if s.get("fallback_active") else "fallback_off",
                        "from": p.get(sid, {}).get("source_actual"), "to": s.get("source_actual"), "error": s["error"]})
    psm = bool(((prev or {}).get("safe_mode") or {}).get("active"))
    if psm != bool(snap["safe_mode"]["active"]):
        out.append({"source_id": "*", "type": "safe_mode_on" if snap["safe_mode"]["active"] else "safe_mode_off",
                    "from": psm, "to": snap["safe_mode"]["active"], "error": "; ".join(snap["safe_mode"]["global_reasons"])})
    return out


def render_report(snap: dict, chg: list[dict], inst: dict | None = None) -> str:
    sm = snap["safe_mode"]
    L = [f"# Daily Data Health Report – {snap['generated'][:10]}", "",
         f"**Safe Mode (Daten): {'AKTIV' if sm['active'] else 'aus'}** – gewichtete Data Quality {sm['data_quality']}"
         + (f" – {'; '.join(sm['global_reasons'])}" if sm["global_reasons"] else ""), "",
         "Status: " + " · ".join(f"{k} {v}" for k, v in snap["counts"].items()), ""]
    if chg:
        L += ["## Neue Änderungen", ""] + [f"- {c['type']}: {c['source_id']} {c['from']} → {c['to']}"
                                           + (f" ({(c.get('error') or '')[:160]})" if c.get("error") else "") for c in chg] + [""]
    bad = [s for s in snap["sources"].values() if s["status"] != HEALTHY and not (s["status"] == UNVALIDATED
                                                                                and s["kind"] == "registry")]
    L += ["## Nicht gesunde Quellen", "", "| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | "
          "betroffene Features/Modelle/Entscheidungen | Grund |", "|---|---|---|---|---|---|---|"]
    for s in sorted(bad, key=lambda s: (STATUSES.index(s["status"]) * -1, s["source_id"])):
        d = s["downstream_dependencies"]
        dep = ", ".join((d.get("features") or [])[:4] + (d.get("models") or [])[:2] + (d.get("decisions") or []))
        L.append(f"| {s['source_id']} | {s['status']} | {s['criticality']} | {s['consecutive_failures']} | "
                 f"{(s['latest_observation'] or '–')[:10]} | {dep or '–'} | {(s['error'] or '')[:140]} |")
    if not bad:
        L.append("| – | alle HEALTHY | | | | | |")
    L += ["", "## Fallbacks", ""] + ([f"- {f['primary']} → {f['actual']} ({(f.get('reason') or '')[:120]})"
                                       for f in snap["fallbacks"]] or ["- keine"])
    L += ["", "## Blockierte Entscheidungspfade / deaktivierte Signale", "",
          f"- blockiert: {sm['blocked_decisions'] or 'keine'}", f"- deaktivierte Signale: {sm['disabled_signals'] or 'keine'}",
          f"- nicht verfügbare Features: {sm['unavailable_features'][:20] or 'keine'}",
          f"- Cache (stale) genutzt: {sm['stale_features'] or 'keine'}"]
    rec = [k for k, v in (inst or {}).items() if v.get("recurring")]
    L += ["", f"Wiederkehrende Instabilität (30 T): {rec or 'keine'}", "",
          "_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._"]
    return "\n".join(L)


def notify(chg: list[dict], snap: dict, report: str, *, send=None, dry_run: bool = False) -> dict:
    """Mail NUR bei relevanten Änderungen – kein Versand bei unverändert gesundem Zustand."""
    if not chg:
        return {"status": "no_change"}
    if send is None:
        from modules.mailer import send_mail as send
    kinds = sorted({c["type"] for c in chg})
    subject = f"Data Health: {', '.join(kinds)} – {snap['generated'][:10]}"
    html = "<pre style='font-family:monospace'>" + report.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    return send(subject, html, report, dry_run=dry_run)


def check(*, now: datetime | None = None, cfg: dict | None = None, probes: bool = True, mail: bool = True,
          fetch=None, yf_module=None, env=None, send=None, out_dir: Path | None = None) -> dict:
    now = now or _now()
    cfg = cfg or load_config()
    od = out_dir or OUT
    snap_p, hist_p = od / SNAPSHOT.name, od / HISTORY.name
    prev = json.loads(snap_p.read_text()) if snap_p.exists() else None
    checks = gather(cfg, now, probes=probes, fetch=fetch, yf_module=yf_module, env=env)
    snap = build_snapshot(checks, prev, cfg, now)
    od.mkdir(parents=True, exist_ok=True)
    snap_p.write_text(json.dumps(snap, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    append_history(snap, hist_p)
    (od / FEATURES.name).write_text(json.dumps({"generated": snap["generated"], "features": snap["features"]},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
    (od / DATA_SAFE_MODE.name).write_text(json.dumps({"generated": snap["generated"], **snap["safe_mode"]},
                                                     indent=1, ensure_ascii=False), encoding="utf-8")
    chg = changes(prev, snap)
    inst = instability(hist_p, now=now)
    report = render_report(snap, chg, inst)
    (od / REPORT_MD.name).write_text(report, encoding="utf-8")
    res = {"status": "no_change"}
    if mail:
        res = notify(chg, snap, report, send=send)
    (od / NOTIFIED.name).write_text(json.dumps({"at": snap["generated"], "changes": chg, "mail": res.get("status")},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
    if out_dir is None:                              # kanonischen SystemState mit neuer Data Health fortschreiben
        try:
            from modules import system_state as ss
            ss.current(now=now)
        except Exception as e:  # noqa: BLE001 – Health-Ergebnis bleibt gültig; State wird beim Lesen neu abgeleitet
            log.error(f"SystemState-Aktualisierung fehlgeschlagen: {e}")
    return {"snapshot": snap, "changes": chg, "mail": res, "report": report}


def feature_dependencies(cfg: dict | None = None) -> dict[str, list[str]]:
    """Feature -> Quellen (Panel-/Champion-Features aus feature_sources, Alt-Features aus der Registry)."""
    cfg = cfg or load_config()
    out: dict[str, set] = {}
    for sid, d in downstream(cfg).items():
        for f in d.get("features") or []:
            out.setdefault(f, set()).add(sid)
    return {f: sorted(v) for f, v in sorted(out.items())}


def render_feature_dependencies(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    deps = downstream(cfg)
    L = ["# Feature Source Dependencies (generiert: python -m modules.source_health deps)", "",
         "Fällt eine Quelle aus (SCHEMA_CHANGED/BROKEN, STALE, UNVALIDATED), sind ihre Features **unavailable**",
         "(NaN, Verfügbarkeit 0 – nie alte Werte, nie 0), Data Quality/Confidence sinken, und produktive",
         "Signale, die zwingend davon abhängen, werden blockiert oder laufen über einen explizit getesteten Fallback.", "",
         "| Feature | Quelle(n) | Fallback | Entscheidungspfade der Quelle | Modelle/Hypothesen der Quelle |", "|---|---|---|---|---|"]
    fb = {c["source_id"]: c.get("fallback") for c in cfg.get("core_sources") or []}
    for f, srcs in feature_dependencies(cfg).items():
        dec = sorted({x for s in srcs for x in (deps.get(s) or {}).get("decisions") or []})
        mh = sorted({x for s in srcs for x in ((deps.get(s) or {}).get("models") or [])
                     + ((deps.get(s) or {}).get("hypotheses") or [])})
        L.append(f"| {f} | {', '.join(srcs)} | {', '.join(fb[s] for s in srcs if fb.get(s)) or '–'} | "
                 f"{', '.join(dec) or '–'} | {', '.join(mh[:6]) or '–'} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "report", "deps"])
    ap.add_argument("--no-mail", action="store_true")
    ap.add_argument("--no-probes", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "deps":
        Path("docs/FEATURE_DEPENDENCIES.md").write_text(render_feature_dependencies(), encoding="utf-8")
        return 0
    if a.cmd == "report":
        snap = load_snapshot()
        print(render_report(snap, []) if not snap.get("unknown") else snap["reason"])
        return 0
    r = check(probes=not a.no_probes, mail=not a.no_mail)
    print(r["report"])
    print(f"\nÄnderungen: {len(r['changes'])} · Mail: {r['mail'].get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

