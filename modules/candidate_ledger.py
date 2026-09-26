"""
modules/candidate_ledger.py – Universal Candidate Ledger (Observability)

Zweck: JEDER Kandidat, der die Pipeline durchläuft — auch frühe Rejects —
wird mit seinen Feature-Werten und (später) dem tatsächlichen
Underlying-Kursverlauf getrackt. So lässt sich retrospektiv auswerten,
welche Gates Geld gekostet bzw. gespart haben (Counterfactual-Analyse).

WICHTIG: Dieses Modul ist REINE Observability.
  - Es verändert NIEMALS eine Gate-Entscheidung.
  - Jede öffentliche Funktion ist defensiv (try/except + Logging) und darf
    die Pipeline unter keinen Umständen zum Absturz bringen.

Datenformat: outputs/candidate_ledger/YYYY-MM.jsonl (eine Zeile pro Ticker/Tag)
    {
      "date": "2026-09-26",
      "ticker": "AAPL",
      "pipeline_version": "v8.3",
      "config_hash": "abcdef012345",
      "status": "rejected" | "proposed",
      "reject_stage": "quick_mc" | null,
      "reject_reason": "mc_below_threshold" | null,
      "direction": "BULLISH" | "BEARISH" | null,
      "features": {...},
      "entry_price": 123.45 | null,
      "outcomes": {
          "ret_5d": 0.012, "ret_20d": ..., "ret_45d": ..., "ret_120d": ...,
          "mfe": ..., "mae": ...
      }
    }
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

LEDGER_ROOT   = Path("outputs/candidate_ledger")
CONFIG_PATH   = Path(__file__).resolve().parent.parent / "config.yaml"
PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"

HORIZONS = (5, 20, 45, 120)

# ── In-Memory-Run-State ──────────────────────────────────────────────────────

_state = {
    "date":             None,
    "config_hash":      "unknown",
    "pipeline_version": "unknown",
    "entries":          {},   # ticker -> dict
    "flushed":          False,
}


def _safe(fn):
    """Kleiner Guard-Wrapper: fängt alles ab, loggt, gibt nie einen Fehler weiter."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.debug(f"candidate_ledger: {fn.__name__} Fehler (ignoriert): {e}")
            return None
    return wrapped


def _compute_config_hash() -> str:
    try:
        raw = CONFIG_PATH.read_bytes()
        return hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        return "unknown"


def _detect_pipeline_version() -> str:
    """Liest z.B. 'v8.3' aus dem Docstring-Kopf von pipeline.py."""
    try:
        text = PIPELINE_PATH.read_text(encoding="utf-8", errors="ignore")
        import re
        m = re.search(r"pipeline\.py\s+(v[0-9]+(?:\.[0-9]+)*)", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def start_run(today: str) -> None:
    """Setzt den In-Memory-State für einen neuen Pipeline-Lauf zurück."""
    try:
        _state["date"]             = today
        _state["config_hash"]      = _compute_config_hash()
        _state["pipeline_version"] = _detect_pipeline_version()
        _state["entries"]          = {}
        _state["flushed"]          = False
    except Exception as e:
        log.debug(f"candidate_ledger.start_run Fehler (ignoriert): {e}")
        _state["date"]             = today
        _state["config_hash"]      = "unknown"
        _state["pipeline_version"] = "unknown"
        _state["entries"]          = {}
        _state["flushed"]          = False


def _entry(ticker: str) -> dict:
    e = _state["entries"].get(ticker)
    if e is None:
        e = {
            "ticker":        ticker,
            "status":        "seen",
            "stage":         None,
            "reject_stage":  None,
            "reject_reason": None,
            "direction":     None,
            "features":      {},
        }
        _state["entries"][ticker] = e
    return e


def note(ticker, stage: str | None = None, **fields) -> None:
    """Merkt Feature-Werte/Stage für einen Ticker vor (upsert)."""
    try:
        if not ticker or not isinstance(ticker, str):
            return
        e = _entry(ticker)
        if stage:
            e["stage"] = stage
        for k, v in fields.items():
            if k == "direction":
                e["direction"] = v
            else:
                try:
                    json.dumps(v, default=str)  # nur JSON-serialisierbare Werte
                    e["features"][k] = v
                except Exception:
                    e["features"][k] = str(v)
    except Exception as e:
        log.debug(f"candidate_ledger.note Fehler (ignoriert): {e}")


def mark_rejected(ticker, reason: str) -> None:
    try:
        if not ticker or not isinstance(ticker, str):
            return
        e = _entry(ticker)
        e["status"]        = "rejected"
        e["reject_reason"] = reason
        e["reject_stage"]  = e.get("stage")
    except Exception as e:
        log.debug(f"candidate_ledger.mark_rejected Fehler (ignoriert): {e}")


def mark_passed(ticker) -> None:
    try:
        if not ticker or not isinstance(ticker, str):
            return
        e = _entry(ticker)
        e["status"] = "proposed"
    except Exception as e:
        log.debug(f"candidate_ledger.mark_passed Fehler (ignoriert): {e}")


def _fetch_prices_batch(tickers: list[str]) -> dict:
    """Ein gebündelter yfinance-Call für die Entry-Preise. Tolerant gegen Fehler."""
    prices = {t: None for t in tickers}
    if not tickers:
        return prices
    try:
        import yfinance as yf
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True, group_by="ticker")
        if data is None or data.empty:
            return prices
        for t in tickers:
            try:
                if len(tickers) == 1:
                    close = data["Close"]
                else:
                    close = data[t]["Close"]
                close = close.dropna()
                if len(close) > 0:
                    prices[t] = float(close.iloc[-1])
            except Exception:
                continue
    except Exception as e:
        log.debug(f"candidate_ledger: Batch-Preisabruf fehlgeschlagen: {e}")
    return prices


def flush(reports_dir_root: Path = LEDGER_ROOT) -> None:
    """Schreibt alle Kandidaten des aktuellen Runs als JSONL-Zeilen (dedup je Tag)."""
    try:
        if _state.get("flushed"):
            return
        if not _state.get("date") or not _state.get("entries"):
            _state["flushed"] = True
            return

        today   = _state["date"]
        entries = _state["entries"]

        root = Path(reports_dir_root)
        root.mkdir(parents=True, exist_ok=True)
        month_file = root / f"{today[:7]}.jsonl"

        existing_keys = set()
        if month_file.exists():
            try:
                with open(month_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            existing_keys.add((row.get("date"), row.get("ticker")))
                        except Exception:
                            continue
            except Exception as e:
                log.debug(f"candidate_ledger: Ledger-Datei nicht lesbar: {e}")

        tickers_to_price = [t for t in entries.keys() if (today, t) not in existing_keys]
        prices = _fetch_prices_batch(tickers_to_price)

        new_lines = []
        for ticker, e in entries.items():
            if (today, ticker) in existing_keys:
                continue
            # Ohne reject()-Aufruf ausgeschieden (z.B. Prescreening) → als
            # "dropped" mit letzter erreichter Stufe markieren statt "seen".
            if e.get("status", "seen") == "seen":
                e["status"]        = "dropped"
                e["reject_stage"]  = e.get("stage")
                e["reject_reason"] = f"unlabeled_after_{e.get('stage') or 'unknown'}"
            row = {
                "date":             today,
                "ticker":           ticker,
                "pipeline_version": _state.get("pipeline_version", "unknown"),
                "config_hash":      _state.get("config_hash", "unknown"),
                "status":           e.get("status", "seen"),
                "reject_stage":     e.get("reject_stage"),
                "reject_reason":    e.get("reject_reason"),
                "direction":        e.get("direction"),
                "features":        e.get("features", {}),
                "entry_price":      prices.get(ticker),
                "outcomes":         {},
            }
            new_lines.append(row)

        if new_lines:
            with open(month_file, "a", encoding="utf-8") as f:
                for row in new_lines:
                    f.write(json.dumps(row, default=str) + "\n")
            log.info(f"candidate_ledger: {len(new_lines)} Kandidat(en) geloggt → {month_file}")

        _state["flushed"] = True
    except Exception as e:
        log.error(f"candidate_ledger.flush Fehler (ignoriert): {e}")
        _state["flushed"] = True


# ── Outcome-Update (aus feedback.py) ─────────────────────────────────────────

def _iter_relevant_files(root: Path, today_dt: datetime):
    """Nur Dateien der letzten 5 Monate berücksichtigen."""
    if not root.exists():
        return []
    cutoff_months = set()
    d = today_dt
    for _ in range(5):
        cutoff_months.add(d.strftime("%Y-%m"))
        # einen Monat zurück (grob, unabhängig von Monatslänge)
        first_of_month = d.replace(day=1)
        d = first_of_month - timedelta(days=1)
    files = sorted(root.glob("*.jsonl"))
    return [f for f in files if f.stem in cutoff_months]


def _direction_adjusted_return(entry_price: float, price: float, direction: str | None) -> float:
    ret = (price / entry_price) - 1.0
    if direction and str(direction).upper() == "BEARISH":
        ret = -ret
    return ret


def update_outcomes(today: str, root: Path = LEDGER_ROOT) -> None:
    """
    Füllt fehlende Return-Horizonte (5/20/45/120 Kalendertage) sowie
    MFE/MAE für Ledger-Zeilen, deren Horizont bereits verstrichen ist.
    Verarbeitet nur Dateien der letzten 5 Monate. Rewrite atomisch.
    """
    try:
        root = Path(root)
        try:
            today_dt = datetime.strptime(today, "%Y-%m-%d")
        except Exception:
            today_dt = datetime.utcnow()

        files = _iter_relevant_files(root, today_dt)
        if not files:
            return

        for path in files:
            try:
                _update_outcomes_in_file(path, today_dt)
            except Exception as e:
                log.debug(f"candidate_ledger.update_outcomes: {path} Fehler (ignoriert): {e}")
    except Exception as e:
        log.error(f"candidate_ledger.update_outcomes Fehler (ignoriert): {e}")


def _update_outcomes_in_file(path: Path, today_dt: datetime) -> None:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return

    # Welche Zeilen brauchen überhaupt einen Update-Versuch?
    pending_tickers = set()
    for row in rows:
        entry_dt = None
        try:
            entry_dt = datetime.strptime(row.get("date", ""), "%Y-%m-%d")
        except Exception:
            continue
        outcomes = row.setdefault("outcomes", {})
        for h in HORIZONS:
            key = f"ret_{h}d"
            if key in outcomes:
                continue
            if (today_dt - entry_dt).days >= h:
                pending_tickers.add(row["ticker"])
                break

    if not pending_tickers:
        return

    max_days = max((today_dt - datetime.strptime(r["date"], "%Y-%m-%d")).days
                   for r in rows if r.get("ticker") in pending_tickers)
    period_days = max(max_days + 5, 10)

    histories = _fetch_history_batch(sorted(pending_tickers), period_days)

    changed = False
    for row in rows:
        ticker = row.get("ticker")
        if ticker not in pending_tickers:
            continue
        try:
            entry_dt = datetime.strptime(row.get("date", ""), "%Y-%m-%d")
        except Exception:
            continue

        hist = histories.get(ticker)
        if hist is None or len(hist) == 0:
            continue

        # Entry-Preis fehlte beim flush (API-Fehler) → aus Historie nachtragen
        entry_price = row.get("entry_price")
        if entry_price in (None, 0):
            entry_price = _price_on_or_before(hist, entry_dt)
            if entry_price in (None, 0):
                continue
            row["entry_price"] = round(float(entry_price), 4)
            changed = True

        direction = row.get("direction")
        outcomes  = row.setdefault("outcomes", {})

        filled_any = False
        for h in HORIZONS:
            key = f"ret_{h}d"
            if key in outcomes:
                continue
            if (today_dt - entry_dt).days < h:
                continue
            target_dt = entry_dt + timedelta(days=h)
            price = _price_on_or_before(hist, target_dt)
            if price is None:
                continue
            outcomes[key] = round(_direction_adjusted_return(entry_price, price, direction), 4)
            filled_any = True

        if filled_any:
            # MFE/MAE über den bislang gefüllten Zeitraum (bis zum letzten
            # tatsächlich erreichten Horizont) neu berechnen.
            latest_horizon = max(
                (h for h in HORIZONS if f"ret_{h}d" in outcomes),
                default=None,
            )
            if latest_horizon is not None:
                end_dt = entry_dt + timedelta(days=latest_horizon)
                window = _prices_between(hist, entry_dt, end_dt)
                if window:
                    rets = [_direction_adjusted_return(entry_price, p, direction) for p in window]
                    outcomes["mfe"] = round(max(rets), 4)
                    outcomes["mae"] = round(min(rets), 4)
            changed = True

    if changed:
        _atomic_write_jsonl(path, rows)


def _fetch_history_batch(tickers: list[str], period_days: int) -> dict:
    """Batched yfinance-History-Download für mehrere Ticker."""
    out = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
        period = f"{max(int(period_days), 5)}d"
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True, group_by="ticker")
        if data is None or data.empty:
            return out
        for t in tickers:
            try:
                if len(tickers) == 1:
                    close = data["Close"]
                else:
                    close = data[t]["Close"]
                close = close.dropna()
                if len(close) > 0:
                    out[t] = list(zip(close.index.to_pydatetime(), close.values.tolist()))
            except Exception:
                continue
    except Exception as e:
        log.debug(f"candidate_ledger: History-Batch-Abruf fehlgeschlagen: {e}")
    return out


def _price_on_or_before(hist, target_dt: datetime):
    """Letzter verfügbarer Preis <= target_dt (nächster Handelstag davor)."""
    best = None
    best_dt = None
    for ts, price in hist:
        ts_naive = ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts
        if ts_naive <= target_dt:
            if best_dt is None or ts_naive > best_dt:
                best_dt = ts_naive
                best = price
    return float(best) if best is not None else None


def _prices_between(hist, start_dt: datetime, end_dt: datetime):
    out = []
    for ts, price in hist:
        ts_naive = ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts
        if start_dt <= ts_naive <= end_dt:
            out.append(float(price))
    return out


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    dir_ = path.parent
    fd, tmp_path = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=str(dir_))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


# ── Summary ──────────────────────────────────────────────────────────────────

def summarize(root: Path = LEDGER_ROOT) -> dict:
    """
    Aggregiert je reject_reason (und 'proposed'): n, mean ret_20d, mean ret_45d,
    Anteil positiv (auf Basis von ret_20d).
    """
    buckets: dict = {}
    try:
        root = Path(root)
        if not root.exists():
            return {}
        for path in sorted(root.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        status = row.get("status")
                        key = "proposed" if status == "proposed" else (row.get("reject_reason") or "unknown")
                        b = buckets.setdefault(key, {"n": 0, "ret_20d": [], "ret_45d": [], "pos": 0, "pos_n": 0})
                        b["n"] += 1
                        outcomes = row.get("outcomes") or {}
                        r20 = outcomes.get("ret_20d")
                        r45 = outcomes.get("ret_45d")
                        if isinstance(r20, (int, float)):
                            b["ret_20d"].append(r20)
                            b["pos_n"] += 1
                            if r20 > 0:
                                b["pos"] += 1
                        if isinstance(r45, (int, float)):
                            b["ret_45d"].append(r45)
            except Exception as e:
                log.debug(f"candidate_ledger.summarize: {path} Fehler (ignoriert): {e}")

        result = {}
        for key, b in buckets.items():
            result[key] = {
                "n":              b["n"],
                "mean_ret_20d":   round(sum(b["ret_20d"]) / len(b["ret_20d"]), 4) if b["ret_20d"] else None,
                "mean_ret_45d":   round(sum(b["ret_45d"]) / len(b["ret_45d"]), 4) if b["ret_45d"] else None,
                "share_positive": round(b["pos"] / b["pos_n"], 4) if b["pos_n"] else None,
            }
        return result
    except Exception as e:
        log.error(f"candidate_ledger.summarize Fehler (ignoriert): {e}")
        return buckets
