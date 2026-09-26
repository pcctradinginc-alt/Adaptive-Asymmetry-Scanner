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
      "signal_timestamp": "2026-09-26T14:03:07+00:00",  # UTC, Sekundenpräzision;
          # gesetzt beim ersten note() dieses Kandidaten im Lauf
      "outcomes": {
          "ret_5d": 0.012, "ret_20d": ..., "ret_45d": ..., "ret_120d": ...,
          "mfe": ..., "mae": ...
      }
    }
"""

import hashlib
import json
import logging
import math
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules import market_snapshot
from modules.bs_pricing import bs_price

log = logging.getLogger(__name__)

LEDGER_ROOT   = Path("outputs/candidate_ledger")
CONFIG_PATH   = Path(__file__).resolve().parent.parent / "config.yaml"
PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"

HORIZONS = (5, 20, 45, 120)

# ── P0-A/B/P2 Defaults (Observability, siehe Docstring oben) ─────────────────
DEFAULT_MAX_OPTION_SNAPSHOTS = 40   # API-Call-Budget für echte Options-Snapshots/Lauf
LATE_MARK_DAYS               = 7    # ab wie vielen verspäteten Tagen real_opt_ret als late_mark markiert wird

# ── Hypothetischer Options-Kontrakt (Counterfactual, Observability) ──────────
# Gates verwerfen/akzeptieren Kandidaten auf Basis von Underlying-Returns,
# aber Trades laufen über Optionen. Um zu sehen, was ein Gate in ECHTEM
# Options-P&L gekostet/gespart hätte, wird pro Kandidat mit bekannter
# Richtung ein hypothetischer ATM-Kontrakt (Black-Scholes) mitgeführt.
# Annahmen (bewusst simpel, rein zur Observability):
#   - konstante IV über die Haltedauer (KEIN IV-Crush-Modell)
#   - halber Spread beim Entry, halber Spread beim Exit
#   - r = 4%
DEFAULT_IV      = 0.35
HYPO_SPREAD_COST = 0.05


def _bs_dte(dte) -> int | None:
    try:
        return int(dte)
    except (TypeError, ValueError):
        return None


def _resolve_hypo_iv(features: dict) -> tuple[float, str]:
    """Wählt IV für den hypothetischen Kontrakt: implied > realized > default."""
    for key in ("implied_vol", "atm_iv", "iv"):
        v = features.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), "implied"
    sigma_30d = features.get("sigma_30d")
    if isinstance(sigma_30d, (int, float)) and sigma_30d > 0:
        try:
            return float(sigma_30d) * math.sqrt(252), "realized"
        except Exception:
            pass
    return DEFAULT_IV, "default"


def _build_hypo_option(entry: dict, entry_price: float) -> dict | None:
    """
    Baut den hypothetischen ATM-Kontrakt für einen Kandidaten mit bekannter
    Richtung (BULLISH/BEARISH) und bekanntem Entry-Preis. Gibt None zurück,
    wenn Richtung, DTE oder Entry-Preis fehlen — nie ein Fehler nach außen.
    """
    try:
        direction = entry.get("direction")
        if direction not in ("BULLISH", "BEARISH"):
            return None
        if entry_price in (None, 0):
            return None
        features = entry.get("features") or {}
        ttm = features.get("ttm")
        try:
            from modules.options_designer import ttm_to_dte_floor
            dte = ttm_to_dte_floor(ttm) if ttm else 120
        except Exception:
            dte = 120
        dte = _bs_dte(dte) or 120

        iv, iv_source = _resolve_hypo_iv(features)
        kind = "call" if direction == "BULLISH" else "put"
        strike = round(float(entry_price))

        entry_premium = bs_price(
            float(entry_price), strike, dte / 365.0, iv, r=0.04, kind=kind,
        )
        if entry_premium is None or entry_premium <= 0:
            return None

        return {
            "kind":           kind,
            "strike":         strike,
            "dte":            dte,
            "iv":             round(float(iv), 4),
            "iv_source":      iv_source,
            "entry_premium":  round(float(entry_premium), 4),
            "spread_cost":    HYPO_SPREAD_COST,
        }
    except Exception as e:
        log.debug(f"candidate_ledger._build_hypo_option Fehler (ignoriert): {e}")
        return None

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


def _max_option_snapshots() -> int:
    """config.yaml ledger.max_option_snapshots, Default 40. Nie ein Fehler
    nach außen (fehlende/kaputte config.yaml → Default)."""
    try:
        from modules.config import cfg
        ledger_cfg = getattr(cfg, "ledger", None)
        return int(getattr(ledger_cfg, "max_option_snapshots", DEFAULT_MAX_OPTION_SNAPSHOTS))
    except Exception:
        return DEFAULT_MAX_OPTION_SNAPSHOTS


def _compute_event_id(ticker: str, date: str, features: dict) -> str:
    """
    event_id = sha1(ticker + Katalysator-Text)[:12], falls die Pipeline über
    note(..., event_key=...) einen Katalysator/Headline-Text mitgegeben hat
    (siehe pipeline.py Stufe 4 „Deep Analysis"). Ohne event_key: Fallback
    sha1(ticker + date)[:12] (bisheriges Verhalten — ein Event/Tag/Ticker).

    Dedup beim flush() erfolgt über (date, ticker, event_id): zwei
    verschiedene Events für denselben Ticker am selben Tag bleiben beide
    erhalten; ein erneuter Lauf über dasselbe Event überschreibt sich nicht
    (wird als Duplikat verworfen).
    """
    try:
        event_key = (features or {}).get("event_key")
        if event_key:
            raw = f"{ticker}:{event_key}"
        else:
            raw = f"{ticker}:{date}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return hashlib.sha1(f"{ticker}:{date}".encode("utf-8")).hexdigest()[:12]


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
            "ticker":           ticker,
            "status":           "seen",
            "stage":            None,
            "reject_stage":     None,
            "reject_reason":    None,
            "direction":        None,
            "features":         {},
            # Erster Zeitpunkt, zu dem dieser Kandidat in diesem Lauf notiert
            # wurde (UTC, Sekundenpräzision) — für die Pre-Registrierungs-
            # Walk-forward-Prüfung im Challenger-Modul (registered_at).
            "signal_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # P2 Signal-Identität: eindeutige ID je Kandidat/Lauf (uuid4).
            "signal_id":        uuid.uuid4().hex,
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


def _parse_iso_dt(ts: str):
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


_EMPTY_UNDERLYING = {
    "bid": None, "ask": None, "mid": None, "last": None,
    "prev_close": None, "open": None, "quote_ts": None, "source": None,
}


def _resolve_entry(ticker: str, e: dict, underlying_quotes: dict, yf_prices: dict) -> dict:
    """
    P0-A Entry-Policy (siehe Modul-Docstring): bestimmt entry_price +
    entry_basis + Session, NIEMALS der letzte yfinance-Schlusskurs bei
    Pre-/Post-/Closed-Signalen (das wäre eine unfaire Vorab-Bewegung).

        regular + Quote-Mid vorhanden  → entry_price=mid,  basis="quote_mid"
        pre/post/closed                → entry_price=None, basis="next_open"
        regular, aber keine Quote      → entry_price=yfinance-Preis, basis="yf_last"

    Gibt nie einen Fehler nach außen (Default = next_open bei Unsicherheit).
    """
    underlying = dict(_EMPTY_UNDERLYING)
    session = "closed"
    try:
        signal_dt = _parse_iso_dt(e.get("signal_timestamp") or "")
        session = market_snapshot.us_market_session(signal_dt) if signal_dt else "closed"

        quote = underlying_quotes.get(ticker)
        if quote:
            underlying.update(quote)

        if session == "regular" and underlying.get("mid") is not None:
            entry_price, entry_basis = underlying["mid"], "quote_mid"
        elif session in ("pre", "post", "closed"):
            entry_price, entry_basis = None, "next_open"
        else:
            entry_price = yf_prices.get(ticker)
            entry_basis = "yf_last"

        gap_at_entry = None
        prev_close = underlying.get("prev_close")
        if entry_price not in (None, 0) and prev_close not in (None, 0):
            try:
                gap_at_entry = round(float(entry_price) / float(prev_close) - 1.0, 4)
            except Exception:
                gap_at_entry = None

        return {
            "entry_price":  entry_price,
            "entry_basis":  entry_basis,
            "session":      session,
            "underlying":   underlying,
            "gap_at_entry": gap_at_entry,
        }
    except Exception as ex:
        log.debug(f"candidate_ledger._resolve_entry Fehler (ignoriert): {ex}")
        return {
            "entry_price":  None,
            "entry_basis":  "next_open",
            "session":      session,
            "underlying":   underlying,
            "gap_at_entry": None,
        }


def _build_real_option(e: dict, spot) -> dict | None:
    """
    P0-B: echter Options-Kontrakt (Tradier-Chain) als Ergänzung zum rein
    synthetischen hypo_option (Black-Scholes). None, wenn Richtung/Spot
    fehlen oder Tradier degradiert (kein Fehler nach außen).
    """
    try:
        direction = e.get("direction")
        if direction not in ("BULLISH", "BEARISH"):
            return None
        if spot in (None, 0):
            return None
        features = e.get("features") or {}
        ttm = features.get("ttm")
        try:
            from modules.options_designer import ttm_to_dte_floor
            dte_floor = ttm_to_dte_floor(ttm) if ttm else 120
        except Exception:
            dte_floor = 120
        return market_snapshot.select_contract(e.get("ticker") or "", direction, dte_floor, spot)
    except Exception as ex:
        log.debug(f"candidate_ledger._build_real_option Fehler (ignoriert): {ex}")
        return None


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

        # P2: event_id je Kandidat bestimmen (braucht die evtl. bis hier
        # gesammelten features, inkl. event_key aus pipeline.py note()).
        for ticker, e in entries.items():
            e["event_id"] = _compute_event_id(ticker, today, e.get("features", {}))

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
                            # Dedup-Schlüssel inkl. event_id (P2): zwei
                            # verschiedene Events desselben Tickers am selben
                            # Tag bleiben beide erhalten. Ältere Zeilen ohne
                            # event_id werden über ihren (fehlenden) Wert
                            # weiterhin eindeutig identifiziert.
                            existing_keys.add((row.get("date"), row.get("ticker"), row.get("event_id")))
                        except Exception:
                            continue
            except Exception as e:
                log.debug(f"candidate_ledger: Ledger-Datei nicht lesbar: {e}")

        tickers_to_process = [
            t for t, e in entries.items()
            if (today, t, e.get("event_id")) not in existing_keys
        ]
        prices            = _fetch_prices_batch(tickers_to_process)
        underlying_quotes = market_snapshot.fetch_underlying_quotes(tickers_to_process)

        max_snapshots       = _max_option_snapshots()
        real_option_budget  = max_snapshots

        new_lines = []
        for ticker, e in entries.items():
            if (today, ticker, e.get("event_id")) in existing_keys:
                continue
            # Ohne reject()-Aufruf ausgeschieden (z.B. Prescreening) → als
            # "dropped" mit letzter erreichter Stufe markieren statt "seen".
            if e.get("status", "seen") == "seen":
                e["status"]        = "dropped"
                e["reject_stage"]  = e.get("stage")
                e["reject_reason"] = f"unlabeled_after_{e.get('stage') or 'unknown'}"

            resolved    = _resolve_entry(ticker, e, underlying_quotes, prices)
            entry_price = resolved["entry_price"]

            row = {
                "date":             today,
                "ticker":           ticker,
                "signal_id":        e.get("signal_id"),
                "event_id":         e.get("event_id"),
                "pipeline_version": _state.get("pipeline_version", "unknown"),
                "config_hash":      _state.get("config_hash", "unknown"),
                "status":           e.get("status", "seen"),
                "reject_stage":     e.get("reject_stage"),
                "reject_reason":    e.get("reject_reason"),
                "direction":        e.get("direction"),
                "features":        e.get("features", {}),
                "entry_price":      entry_price,
                "entry_basis":      resolved["entry_basis"],
                "session":          resolved["session"],
                "underlying":       resolved["underlying"],
                "gap_at_entry":     resolved["gap_at_entry"],
                "signal_timestamp": e.get("signal_timestamp"),
                "outcomes":         {},
            }

            hypo = _build_hypo_option(e, entry_price)
            if hypo is not None:
                row["hypo_option"] = hypo

            # P0-B: echter Options-Kontrakt — nur für Kandidaten mit bekannter
            # Richtung, begrenzt auf `ledger.max_option_snapshots` API-Calls/Lauf.
            if e.get("direction") in ("BULLISH", "BEARISH"):
                if real_option_budget <= 0:
                    row["real_option_skip_reason"] = "max_snapshots_reached"
                else:
                    spot = entry_price
                    if spot in (None, 0):
                        spot = resolved["underlying"].get("last") or resolved["underlying"].get("mid")
                    if spot in (None, 0):
                        spot = prices.get(ticker)
                    # Jeder Versuch (Erfolg oder nicht) zählt gegen das
                    # API-Call-Budget, da er selbst bei einem Miss bereits
                    # einen Tradier-Call (Expirations/Chain) ausgelöst hat.
                    real_option_budget -= 1
                    real_option = _build_real_option(e, spot)
                    if real_option is not None:
                        row["real_option"] = real_option
                    else:
                        row["real_option_skip_reason"] = (
                            "no_spot" if spot in (None, 0)
                            else ("no_api_key" if not os.environ.get("TRADIER_API_KEY", "").strip()
                                  else "no_contract_found")
                        )

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


def _parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _row_entry_dt(row: dict):
    """
    Effektives Entry-Datum für Horizont-Berechnungen: bei next_open-Zeilen,
    deren Open bereits nachgetragen wurde, das nachgetragene Handelsdatum
    (entry_effective_date) — sonst (bisheriges Verhalten) das Signal-Datum.
    """
    eff = row.get("entry_effective_date")
    if eff:
        dt = _parse_date(eff)
        if dt is not None:
            return dt
    return _parse_date(row.get("date", ""))


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

    changed = False
    try:
        changed |= _fill_next_open_entries(rows, today_dt)
    except Exception as e:
        log.debug(f"candidate_ledger: next_open-Fill Fehler (ignoriert): {e}")
    try:
        changed |= _fill_return_horizons(rows, today_dt)
    except Exception as e:
        log.debug(f"candidate_ledger: Horizont-Fill Fehler (ignoriert): {e}")
    try:
        changed |= _fill_real_option_marks(rows, today_dt)
    except Exception as e:
        log.debug(f"candidate_ledger: real_opt_ret-Marks Fehler (ignoriert): {e}")

    if changed:
        _atomic_write_jsonl(path, rows)


def _fill_next_open_entries(rows: list[dict], today_dt: datetime) -> bool:
    """
    P0-A: für Zeilen mit entry_basis="next_open" und noch fehlendem
    entry_price wird der Open-Preis der ersten regulären Session STRIKT NACH
    signal_timestamp nachgetragen (Signal pre-market an einem Handelstag →
    Open desselben Tages; post/closed → Open des nächsten Handelstags).
    Setzt entry_effective_date, entry_price, gap_at_entry (falls prev_close
    bekannt) und trägt hypo_option nach, falls noch nicht vorhanden.
    """
    pending = [
        r for r in rows
        if r.get("entry_basis") == "next_open" and r.get("entry_price") in (None, 0)
    ]
    if not pending:
        return False

    tickers = sorted({r["ticker"] for r in pending if r.get("ticker")})
    dates = [_parse_date(r.get("date", "")) for r in pending]
    dates = [d for d in dates if d is not None]
    if not dates:
        return False
    max_days = max((today_dt - d).days for d in dates)
    period_days = max(max_days + 10, 15)

    histories = _fetch_history_open_batch(tickers, period_days)

    changed = False
    for row in pending:
        try:
            hist = histories.get(row.get("ticker"))
            if not hist:
                continue
            signal_dt = _parse_iso_dt(row.get("signal_timestamp") or "")
            if signal_dt is None:
                continue
            local = signal_dt.astimezone(market_snapshot.NY_TZ)
            session = row.get("session")
            floor_date = local.date() if session == "pre" else local.date() + timedelta(days=1)

            candidate = None
            for d, open_px, _close in sorted(hist, key=lambda x: x[0]):
                if d >= floor_date:
                    candidate = (d, open_px)
                    break
            if candidate is None:
                continue

            entry_date, entry_open = candidate
            if entry_open in (None, 0):
                continue

            entry_price = round(float(entry_open), 4)
            row["entry_price"]          = entry_price
            row["entry_effective_date"] = entry_date.isoformat()

            prev_close = (row.get("underlying") or {}).get("prev_close")
            if prev_close not in (None, 0):
                try:
                    row["gap_at_entry"] = round(entry_price / float(prev_close) - 1.0, 4)
                except Exception:
                    pass

            if "hypo_option" not in row:
                try:
                    hypo = _build_hypo_option(row, entry_price)
                    if hypo is not None:
                        row["hypo_option"] = hypo
                except Exception as e:
                    log.debug(f"candidate_ledger: hypo_option-Backfill (next_open) Fehler (ignoriert): {e}")

            changed = True
        except Exception as e:
            log.debug(f"candidate_ledger._fill_next_open_entries Fehler (ignoriert): {e}")
    return changed


def _fill_return_horizons(rows: list[dict], today_dt: datetime) -> bool:
    """Bestehende Logik: füllt ret_{h}d/opt_ret_{h}d/mfe/mae. Nutzt das
    effektive Entry-Datum (entry_effective_date bei next_open-Zeilen, sonst
    das Signal-Datum) statt blind row["date"]."""
    pending_tickers = set()
    for row in rows:
        entry_dt = _row_entry_dt(row)
        if entry_dt is None:
            continue
        if row.get("entry_price") in (None, 0) and row.get("entry_basis") == "next_open":
            continue  # Open noch nicht verfügbar — nächster Lauf versucht es erneut
        outcomes = row.setdefault("outcomes", {})
        for h in HORIZONS:
            key = f"ret_{h}d"
            if key in outcomes:
                continue
            if (today_dt - entry_dt).days >= h:
                pending_tickers.add(row["ticker"])
                break

    if not pending_tickers:
        return False

    candidate_days = [
        (today_dt - _row_entry_dt(r)).days for r in rows
        if r.get("ticker") in pending_tickers and _row_entry_dt(r) is not None
    ]
    if not candidate_days:
        return False
    period_days = max(max(candidate_days) + 5, 10)

    histories = _fetch_history_batch(sorted(pending_tickers), period_days)

    changed = False
    for row in rows:
        ticker = row.get("ticker")
        if ticker not in pending_tickers:
            continue
        entry_dt = _row_entry_dt(row)
        if entry_dt is None:
            continue

        hist = histories.get(ticker)
        if hist is None or len(hist) == 0:
            continue

        # Entry-Preis fehlte beim flush (API-Fehler) → aus Historie nachtragen.
        # next_open-Zeilen NICHT hier nachtragen (das übernimmt ausschließlich
        # _fill_next_open_entries mit der Open-Regel — niemals der letzte
        # verfügbare Schlusskurs, sonst wäre die Pre-Signal-Bewegung wieder
        # unfair eingepreist).
        entry_price = row.get("entry_price")
        if entry_price in (None, 0):
            if row.get("entry_basis") == "next_open":
                continue
            entry_price = _price_on_or_before(hist, entry_dt)
            if entry_price in (None, 0):
                continue
            row["entry_price"] = round(float(entry_price), 4)
            changed = True
            if "hypo_option" not in row:
                try:
                    hypo = _build_hypo_option(row, entry_price)
                    if hypo is not None:
                        row["hypo_option"] = hypo
                except Exception as e:
                    log.debug(f"candidate_ledger: hypo_option-Backfill Fehler (ignoriert): {e}")

        direction = row.get("direction")
        outcomes  = row.setdefault("outcomes", {})
        hypo      = row.get("hypo_option")

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
            opt_key = f"opt_ret_{h}d"
            if opt_key not in outcomes and hypo:
                opt_ret = _compute_opt_ret(hypo, price, h)
                if opt_ret is not None:
                    outcomes[opt_key] = opt_ret

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

    return changed


def _fill_real_option_marks(rows: list[dict], today_dt: datetime) -> bool:
    """
    P0-B: für Zeilen mit real_option und verstrichenem Horizont h wird
    real_opt_ret_{h}d (konservativ: Kauf zum Ask, Verkauf zum Bid) sowie
    real_opt_ret_mid_{h}d (Mid/Mid) nachgetragen. Für bereits verfallene
    Kontrakte wird der Intrinsic-Wert aus dem Underlying-Schlusskurs am
    Expiry-Tag verwendet statt einer (nicht mehr existierenden) Live-Quote.
    Marks passieren erst beim ERSTEN feedback.py-Lauf nach Ablauf des
    Horizonts (Lag dokumentiert über real_opt_mark_date_{h}d). Zeilen, deren
    Horizont schon >7 Tage verstrichen ist, werden trotzdem jetzt mit der
    aktuellen Quote markiert, aber zusätzlich mit late_mark=true geflaggt.
    """
    tasks = []  # (row, h, entry_dt)
    for row in rows:
        real_option = row.get("real_option")
        if not real_option:
            continue
        entry_dt = _row_entry_dt(row)
        if entry_dt is None:
            continue
        outcomes = row.setdefault("outcomes", {})
        for h in HORIZONS:
            if f"real_opt_ret_{h}d" in outcomes:
                continue
            if (today_dt - entry_dt).days >= h:
                tasks.append((row, h, entry_dt))

    if not tasks:
        return False

    symbols = sorted({
        t[0]["real_option"].get("symbol") for t in tasks
        if t[0]["real_option"].get("symbol")
    })
    quotes = market_snapshot.fetch_option_quotes(symbols)

    underlying_tickers = sorted({t[0].get("ticker") for t in tasks if t[0].get("ticker")})
    max_days = max((today_dt - t[2]).days for t in tasks)
    underlying_hist = _fetch_history_batch(underlying_tickers, max(max_days + 10, 15))

    changed = False
    for row, h, entry_dt in tasks:
        try:
            real_option = row.get("real_option") or {}
            entry_ask = real_option.get("ask")
            entry_mid = real_option.get("mid")
            if entry_ask in (None, 0) or entry_mid in (None, 0):
                continue

            symbol     = real_option.get("symbol")
            expiry_str = real_option.get("expiry")
            expiry_dt  = _parse_date(expiry_str) if expiry_str else None

            if expiry_dt is not None and today_dt.date() > expiry_dt.date():
                hist = underlying_hist.get(row.get("ticker"))
                spot_at_expiry = _price_on_or_before(hist, expiry_dt) if hist else None
                if spot_at_expiry is None:
                    continue
                strike = real_option.get("strike") or 0
                kind = "call" if row.get("direction") == "BULLISH" else "put"
                intrinsic = (max(spot_at_expiry - strike, 0.0) if kind == "call"
                             else max(strike - spot_at_expiry, 0.0))
                exit_bid = exit_mid = intrinsic
            else:
                q = quotes.get(symbol)
                if not q:
                    continue
                exit_bid, exit_mid = q.get("bid"), q.get("mid")
                if exit_bid in (None,) or exit_mid in (None,):
                    continue

            outcomes = row.setdefault("outcomes", {})
            outcomes[f"real_opt_ret_{h}d"]       = round(exit_bid / entry_ask - 1.0, 4)
            outcomes[f"real_opt_ret_mid_{h}d"]   = round(exit_mid / entry_mid - 1.0, 4)
            outcomes[f"real_opt_mark_date_{h}d"] = today_dt.strftime("%Y-%m-%d")

            if (today_dt - (entry_dt + timedelta(days=h))).days > LATE_MARK_DAYS:
                row["late_mark"] = True

            changed = True
        except Exception as e:
            log.debug(f"candidate_ledger._fill_real_option_marks Fehler (ignoriert): {e}")
    return changed


def _compute_opt_ret(hypo: dict, price_h: float, h: int):
    """
    Options-Counterfactual-Return für Horizont h Tage: bewertet den
    hypothetischen Kontrakt zum Horizont-Preis mit KONSTANTER IV (kein
    IV-Crush-Modell — das ist eine bewusste Vereinfachung), abzüglich
    halbem Spread bei Entry und Exit. Nach unten auf -1.0 gecapped
    (Optionsverlust kann nicht schlimmer als Totalverlust sein), nach
    oben offen.
    """
    try:
        kind          = hypo.get("kind")
        strike        = hypo.get("strike")
        dte           = hypo.get("dte")
        iv            = hypo.get("iv")
        entry_premium = hypo.get("entry_premium")
        spread_cost   = hypo.get("spread_cost", HYPO_SPREAD_COST)
        if not entry_premium or entry_premium <= 0:
            return None
        remaining_days = dte - h
        T_years = max(remaining_days, 0) / 365.0
        exit_price = bs_price(float(price_h), float(strike), T_years, float(iv), r=0.04, kind=kind)
        exit_net  = exit_price * (1 - spread_cost / 2)
        entry_net = entry_premium * (1 + spread_cost / 2)
        if entry_net <= 0:
            return None
        opt_ret = (exit_net / entry_net) - 1.0
        return round(max(opt_ret, -1.0), 4)
    except Exception as e:
        log.debug(f"candidate_ledger._compute_opt_ret Fehler (ignoriert): {e}")
        return None


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


def _fetch_history_open_batch(tickers: list[str], period_days: int) -> dict:
    """
    Batched yfinance-Tageshistorie INKLUSIVE Open (für den next_open
    Entry-Preis-Fill in _fill_next_open_entries). Getrennt von
    _fetch_history_batch (Close-only), damit dessen bestehendes Format
    (und die darauf aufbauenden Tests) unverändert bleibt.

    Returns: {ticker: [(date, open, close), ...]}
    """
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
                frame = data if len(tickers) == 1 else data[t]
                frame = frame.dropna(subset=["Open"])
                bars = []
                for ts, r in frame.iterrows():
                    d = ts.to_pydatetime()
                    d = d.date() if hasattr(d, "date") else d
                    try:
                        close_val = float(r["Close"])
                    except Exception:
                        close_val = float(r["Open"])
                    bars.append((d, float(r["Open"]), close_val))
                if bars:
                    out[t] = bars
            except Exception:
                continue
    except Exception as e:
        log.debug(f"candidate_ledger: History(Open)-Batch-Abruf fehlgeschlagen: {e}")
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
                        b = buckets.setdefault(key, {
                            "n": 0, "ret_20d": [], "ret_45d": [], "pos": 0, "pos_n": 0,
                            "opt_ret_20d": [], "opt_pos": 0, "opt_pos_n": 0, "opt_ret_45d": [],
                        })
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

                        opt20 = outcomes.get("opt_ret_20d")
                        opt45 = outcomes.get("opt_ret_45d")
                        if isinstance(opt20, (int, float)):
                            b["opt_ret_20d"].append(opt20)
                            b["opt_pos_n"] += 1
                            if opt20 > 0:
                                b["opt_pos"] += 1
                        if isinstance(opt45, (int, float)):
                            b["opt_ret_45d"].append(opt45)
            except Exception as e:
                log.debug(f"candidate_ledger.summarize: {path} Fehler (ignoriert): {e}")

        result = {}
        for key, b in buckets.items():
            result[key] = {
                "n":                  b["n"],
                "mean_ret_20d":       round(sum(b["ret_20d"]) / len(b["ret_20d"]), 4) if b["ret_20d"] else None,
                "mean_ret_45d":       round(sum(b["ret_45d"]) / len(b["ret_45d"]), 4) if b["ret_45d"] else None,
                "share_positive":     round(b["pos"] / b["pos_n"], 4) if b["pos_n"] else None,
                # Options-Counterfactual (Black-Scholes, konstante IV):
                "mean_opt_ret_20d":   round(sum(b["opt_ret_20d"]) / len(b["opt_ret_20d"]), 4) if b["opt_ret_20d"] else None,
                "mean_opt_ret_45d":   round(sum(b["opt_ret_45d"]) / len(b["opt_ret_45d"]), 4) if b["opt_ret_45d"] else None,
                "opt_share_positive": round(b["opt_pos"] / b["opt_pos_n"], 4) if b["opt_pos_n"] else None,
            }
        return result
    except Exception as e:
        log.error(f"candidate_ledger.summarize Fehler (ignoriert): {e}")
        return buckets
