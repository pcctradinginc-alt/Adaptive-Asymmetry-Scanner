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

Datenformat: outputs/candidate_ledger/YYYY-MM.jsonl (eine Zeile pro SIGNAL —
P0-1: ein Ticker kann mehrere Signale/Zeilen am selben Tag haben, je ein
eigener Katalysator/Event — siehe event_key/event_id unten und den
Kommentar bei _resolve_signal_for_note)
    {
      "date": "2026-09-26",
      "ticker": "AAPL",
      "signal_id": "…",           # eindeutig je Signal, bei Erzeugung vergeben
      "event_id": "…",             # sha1(ticker+event_key)[:12] bzw. sha1(ticker+date)[:12]
      "event_key": "FDA approval PDUFA" | null,
      "pipeline_version": "v8.3",
      "config_hash": "abcdef012345",
      "status": "rejected" | "proposed",
      "reject_stage": "quick_mc" | null,
      "reject_reason": "mc_below_threshold" | null,
      "direction": "BULLISH" | "BEARISH" | null,
      "features": {...},
      "entry_price": 123.45 | null,
      "signal_timestamp": "2026-09-26T14:03:07+00:00",  # UTC, Sekundenpräzision;
          # gesetzt beim ersten note() dieses Signals im Lauf
      "real_option": {...},        # echter Long-Leg-Kontrakt (Tradier), Rückwärtskompatibilität
      "real_strategy": {           # P0-2: Produktions-Strategie-Counterfactual
          "strategy": "BULL_CALL_SPREAD" | "LONG_CALL" | ...,
          "strategy_source": "computed" | "default_long" | "spread_no_liquidity",
          "legs": [{"symbol": ..., "strike": ..., "side": "long"|"short",
                     "bid": ..., "ask": ..., "mid": ...}, ...],
          "net_debit_entry": ..., "net_mid_entry": ...,
      },
      "snapshot_eligible_n": 12, "snapshot_budget": 40, "snapshot_selected": true,
      "outcomes": {
          "ret_5d": 0.012, "ret_20d": ..., "ret_45d": ..., "ret_120d": ...,
          "real_strat_ret_45d": ..., "real_strat_ret_mid_45d": ...,
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


def _compute_event_id(ticker: str, date: str, signal: dict) -> str:
    """
    event_id = sha1(ticker + Katalysator-Text)[:12], falls dieses Signal
    über note(..., event_key=...) einen Katalysator/Headline-Text bekommen
    hat (siehe pipeline.py Stufe 4 „Deep Analysis"). Ohne event_key:
    Fallback sha1(ticker + date)[:12] (bisheriges Verhalten — ein
    Event/Tag/Ticker; P0-1: es gibt dann per Ticker/Tag höchstens EIN
    Signal ohne event_key, siehe _resolve_signal_for_note).

    Dedup beim flush() erfolgt über (date, ticker, event_id): zwei
    verschiedene Events für denselben Ticker am selben Tag bleiben beide
    erhalten (je ein eigenes Signal/eine eigene Zeile — P0-1); ein
    erneuter Lauf über dasselbe Event überschreibt sich nicht (wird als
    Duplikat verworfen).
    """
    try:
        event_key = (signal or {}).get("event_key")
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


def _compute_code_sha() -> str:
    """Commit-SHA des laufenden Codes (GitHub Actions: GITHUB_SHA, sonst git)."""
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if sha:
        return sha[:12]
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip()[:12] or "unknown"
    except Exception:
        return "unknown"


def _compute_model_ids() -> dict:
    """LLM-/ML-Modell-IDs aus config.yaml (models:, rl, finbert) für Reproduzierbarkeit.

    P1: zusätzlich sha256(erste 12 Zeichen) der PPO-Modell-Datei
    (config rl.model_path), FALLS die Datei existiert — damit ein
    Ledger-Eintrag eindeutig auf die tatsächlich verwendeten Modell-Gewichte
    zurückgeführt werden kann (model_path allein sagt nichts über einen
    Retrain aus). Fehlt die Datei/der Pfad, wird das Feld einfach
    weggelassen (kein Fehler nach außen).
    """
    ids = {}
    try:
        import yaml
        raw = yaml.safe_load(Path("config.yaml").read_text()) or {}
        for k, v in (raw.get("models") or {}).items():
            if isinstance(v, str):
                ids[f"models.{k}"] = v
        for section, key in (("rl", "model_path"), ("finbert", "model_name")):
            sec = raw.get(section) or {}
            if isinstance(sec, dict) and isinstance(sec.get(key), str):
                ids[f"{section}.{key}"] = sec[key]
                if section == "rl" and key == "model_path":
                    try:
                        model_path = Path(sec[key])
                        if model_path.exists():
                            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                            ids["rl.model_sha256"] = digest[:12]
                    except Exception:
                        pass
    except Exception:
        pass
    return ids


def start_run(today: str) -> None:
    """Setzt den In-Memory-State für einen neuen Pipeline-Lauf zurück."""
    try:
        _state["date"]             = today
        _state["config_hash"]      = _compute_config_hash()
        _state["pipeline_version"] = _detect_pipeline_version()
        _state["code_sha"]         = _compute_code_sha()
        _state["model_ids"]        = _compute_model_ids()
        _state["entries"]          = {}
        _state["flushed"]          = False
    except Exception as e:
        log.debug(f"candidate_ledger.start_run Fehler (ignoriert): {e}")
        _state["date"]             = today
        _state["config_hash"]      = "unknown"
        _state["pipeline_version"] = "unknown"
        _state["entries"]          = {}
        _state["flushed"]          = False


# ── P0-1 Signal-Identität ────────────────────────────────────────────────────
#
# _state["entries"] ist ticker -> LISTE von Signal-Dicts (statt einem Dict je
# Ticker). Grund: ein Ticker kann in einem Lauf zu mehreren, unabhängigen
# Katalysator-Events gehören (z.B. zwei separate Headlines am selben Tag) —
# das darf beim Flush nicht auf eine Zeile kollabieren.
#
# note()-Aufrufe für einen Ticker passieren zuerst OHNE event_key (Universe-
# Stufe: Feature-Werte sind noch ticker-weit, das Event ist noch nicht
# bekannt) und erst später MIT event_key (Deep-Analysis-Stufe: Katalysator-
# Text bekannt). Deshalb:
#   - note(ticker, ...) OHNE event_key aktualisiert ALLE bisherigen Signale
#     dieses Tickers (oder legt das erste an, falls noch keins existiert).
#   - note(ticker, ..., event_key=X) sucht ein Signal mit event_key==X; gibt
#     es keins, wird das erste noch event_key-lose Signal "geclaimt" (das ist
#     der Normalfall: Universe-Stufe → Deep-Analysis-Stufe desselben Events);
#     gibt es auch das nicht (alle bestehenden Signale haben schon einen
#     ANDEREN event_key), wird ein NEUES Signal angelegt — eine Kopie der
#     bisher notierten ticker-weiten Felder (features/direction/stage) des
#     ersten Signals, damit das neue Event nicht mit leeren Features startet.
#   - Ein zweites note(..., event_key=X) mit demselben X aktualisiert nur
#     dieses eine Signal (Dedup: "gleiches Event zweimal notiert → eine Zeile").
#
# mark_rejected()/mark_passed() wirken ohne event_key auf ALLE Signale des
# Tickers (bisheriges Verhalten, z.B. frühe Prescreening-Rejects, die noch
# gar kein Event kennen); mit event_key nur auf das eine passende Signal.


def _new_signal(ticker: str, event_key=None) -> dict:
    return {
        "ticker":           ticker,
        "event_key":        event_key,
        "status":           "seen",
        "stage":            None,
        "reject_stage":     None,
        "reject_reason":    None,
        "direction":        None,
        "features":         {},
        # Erster Zeitpunkt, zu dem dieses Signal in diesem Lauf notiert
        # wurde (UTC, Sekundenpräzision) — für die Pre-Registrierungs-
        # Walk-forward-Prüfung im Challenger-Modul (registered_at).
        "signal_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Eindeutige ID je Signal — bei Erzeugung vergeben (nicht erst beim
        # Flush), damit sie über note()-Aufrufe hinweg stabil bleibt.
        "signal_id":        uuid.uuid4().hex,
    }


def _signals(ticker: str) -> list:
    return _state["entries"].setdefault(ticker, [])


def _resolve_signal_for_note(ticker: str, event_key) -> list:
    """Gibt die Liste der Signale zurück, auf die ein note()-Aufruf
    angewendet werden soll (siehe Modul-weiter Kommentar oben)."""
    lst = _signals(ticker)
    if event_key is None:
        if not lst:
            lst.append(_new_signal(ticker))
        return lst
    for s in lst:
        if s.get("event_key") == event_key:
            return [s]
    unassigned = next((s for s in lst if s.get("event_key") is None), None)
    if unassigned is not None:
        unassigned["event_key"] = event_key
        return [unassigned]
    if lst:
        base = lst[0]
        new_sig = _new_signal(ticker, event_key)
        new_sig["features"]  = dict(base.get("features", {}))
        new_sig["direction"] = base.get("direction")
        new_sig["stage"]     = base.get("stage")
        lst.append(new_sig)
        return [new_sig]
    new_sig = _new_signal(ticker, event_key)
    lst.append(new_sig)
    return [new_sig]


def _resolve_signal_for_mark(ticker: str, event_key) -> list:
    """Gibt die Liste der Signale zurück, auf die mark_rejected()/
    mark_passed() angewendet werden sollen."""
    lst = _signals(ticker)
    if not lst:
        lst.append(_new_signal(ticker))
        return lst
    if event_key is None:
        return lst
    for s in lst:
        if s.get("event_key") == event_key:
            return [s]
    # Unbekannter event_key (z.B. Reject vor Deep-Analysis-Stufe, während
    # ein anderer Aufrufer schon einen event_key vergeben hat): sicherer
    # Fallback auf ALLE Signale statt zu verwerfen.
    return lst


def note(ticker, stage: str | None = None, **fields) -> None:
    """Merkt Feature-Werte/Stage für einen Ticker vor (upsert je Signal,
    siehe Kommentar oben zur P0-1 Signal-Identität)."""
    try:
        if not ticker or not isinstance(ticker, str):
            return
        event_key = fields.pop("event_key", None)
        targets = _resolve_signal_for_note(ticker, event_key)
        for e in targets:
            if event_key is not None:
                e["event_key"] = event_key
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


def mark_rejected(ticker, reason: str, event_key=None) -> None:
    try:
        if not ticker or not isinstance(ticker, str):
            return
        for e in _resolve_signal_for_mark(ticker, event_key):
            e["status"]        = "rejected"
            e["reject_reason"] = reason
            e["reject_stage"]  = e.get("stage")
    except Exception as e:
        log.debug(f"candidate_ledger.mark_rejected Fehler (ignoriert): {e}")


def mark_passed(ticker, event_key=None) -> None:
    try:
        if not ticker or not isinstance(ticker, str):
            return
        for e in _resolve_signal_for_mark(ticker, event_key):
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


# ── P0-2: Produktions-Strategie-Counterfactual (real_strategy) ──────────────
#
# real_option (oben) bewertet immer nur die Long-Leg (LONG_CALL/LONG_PUT) —
# aber die Produktion (options_designer._select_strategy /
# choose_strategy()) tradet ab einem bestimmten IV-Rank/Dealer-Gamma/VIX-
# Term-Structure-Gate stattdessen einen Debit-Spread. real_strategy
# durchläuft GENAU denselben Entscheidungspfad (choose_strategy() ist die
# aus options_designer.py extrahierte reine Funktion), damit der Ledger
# nicht nur "wie hätte ein Long-Call performt", sondern "wie hätte die
# Produktion TATSÄCHLICH gehandelt" beantwortet.
#
# iv_rank: die Produktion berechnet ihn über yfinance-Realized-Vol +
# Options-Term-Structure (OptionsDesigner._get_iv_rank) — das braucht pro
# Kandidat mehrere Netzwerk-Calls und ist hier aus Budget-/Kopplungsgründen
# NICHT reproduzierbar. Stattdessen: ein Percentile-Rank der Chain-IV des
# gewählten Long-Legs innerhalb der historischen ATM-IV-Werte desselben
# Tickers (outputs/history.json → iv_history[ticker], von
# mirofish_simulation._log_iv_today() befüllt). Das ist eine bewusste
# Vereinfachung (siehe Docstring), NICHT identisch mit _get_iv_rank — bei
# zu wenig Historie (<20 Tage) wird iv_rank als nicht bestimmbar behandelt
# und strategy_source="default_long" gesetzt (Fallback: Long-Leg, wie schon
# real_option).
#
# dealer_gamma_state/vix_structure: wenn die Produktion sie im selben Lauf
# schon berechnet hat, werden sie per note(..., dealer_gamma_state=...,
# vix_structure=...) aus pipeline.py mitgegeben (ein note()-Aufruf,
# unmittelbar vor designer.run(), siehe pipeline.py Stufe 10). Fehlen sie
# (z.B. älterer Lauf, Pipeline-Pfad ohne diesen note()-Aufruf erreicht):
# "unknown" — genau der Fallback, den choose_strategy() selbst für
# unbekannte Werte verwendet (neutrales Gate, kein VIX-Adjustment).

HISTORY_PATH = Path("outputs/history.json")


def _load_iv_history(ticker: str) -> list:
    """outputs/history.json → iv_history[ticker] (Liste von {date, atm_iv}).
    Fehlt die Datei/der Ticker: leere Liste (nie ein Fehler nach außen)."""
    try:
        if not HISTORY_PATH.exists():
            return []
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        return (data.get("iv_history") or {}).get(ticker, []) or []
    except Exception as e:
        log.debug(f"candidate_ledger._load_iv_history [{ticker}] Fehler (ignoriert): {e}")
        return []


IV_RANK_MIN_HISTORY_DAYS = 20


def _resolve_candidate_iv_rank(ticker: str, chain_iv) -> float | None:
    """Percentile-Rank der Chain-IV des Long-Legs innerhalb der letzten
    ~252 historischen ATM-IV-Werte (5%/95%-Quantile wie
    OptionsDesigner._get_iv_rank's rv_score-Komponente). None, wenn Chain-IV
    fehlt oder zu wenig Historie vorliegt (< IV_RANK_MIN_HISTORY_DAYS)."""
    try:
        if chain_iv in (None, 0):
            return None
        chain_iv = float(chain_iv)
        entries = _load_iv_history(ticker)[-252:]
        values = sorted(
            float(e["atm_iv"]) for e in entries
            if isinstance(e, dict) and isinstance(e.get("atm_iv"), (int, float))
        )
        if len(values) < IV_RANK_MIN_HISTORY_DAYS:
            return None
        lo = values[max(0, int(len(values) * 0.05))]
        hi = values[min(len(values) - 1, int(len(values) * 0.95))]
        if hi <= lo:
            return None
        rank = (chain_iv - lo) / (hi - lo) * 100.0
        return max(0.0, min(100.0, rank))
    except Exception as e:
        log.debug(f"candidate_ledger._resolve_candidate_iv_rank [{ticker}] Fehler (ignoriert): {e}")
        return None


def _leg_from_contract(contract: dict, side: str) -> dict:
    return {
        "symbol": contract.get("symbol"),
        "strike": contract.get("strike"),
        "side":   side,
        "bid":    contract.get("bid"),
        "ask":    contract.get("ask"),
        "mid":    contract.get("mid"),
    }


def _strategy_row(strategy, source, legs, iv_rank, effective_gate, expiry) -> dict:
    row = {"strategy": strategy, "strategy_source": source, "legs": legs, "expiry": expiry}
    if iv_rank is not None:
        row["iv_rank"] = round(iv_rank, 1)
    if effective_gate is not None:
        row["effective_gate"] = effective_gate
    return row


def _build_real_strategy(e: dict, ticker: str, spot, long_raw: dict | None) -> dict | None:
    """
    P0-2: baut den Produktions-Strategie-Counterfactual (siehe Kommentar
    oben) — wiederverwendet den bereits abgerufenen Long-Leg-Kontrakt
    (long_raw, aus _build_real_option) statt Tradier erneut zu befragen.
    Nur eine ZUSÄTZLICHE Chain-Abfrage für den Short-Leg, falls die
    Strategie ein Spread ist (choose_strategy() → *_SPREAD). Gibt None
    zurück (nie einen Fehler), wenn Richtung fehlt oder kein Long-Kontrakt
    vorliegt.
    """
    try:
        direction = e.get("direction")
        if direction not in ("BULLISH", "BEARISH") or long_raw is None:
            return None

        features           = e.get("features") or {}
        dealer_gamma_state = features.get("dealer_gamma_state") or {}
        vix_structure      = features.get("vix_structure") or "unknown"
        is_bullish         = direction == "BULLISH"
        default_strategy   = "LONG_CALL" if is_bullish else "LONG_PUT"
        long_leg           = _leg_from_contract(long_raw, "long")
        expiry             = long_raw.get("expiry")

        iv_rank = _resolve_candidate_iv_rank(ticker, long_raw.get("iv"))
        if iv_rank is None:
            return _strategy_row(default_strategy, "default_long", [long_leg], None, None, expiry)

        from modules.options_designer import choose_strategy

        strategy, effective_gate, _reason = choose_strategy(
            iv_rank, is_bullish, dealer_gamma_state, vix_structure,
        )

        if "SPREAD" not in strategy:
            return _strategy_row(strategy, "computed", [long_leg], iv_rank, effective_gate, expiry)

        option_type = "call" if is_bullish else "put"
        short_raw = market_snapshot.select_spread_short_leg(
            ticker, expiry, option_type, long_raw.get("strike"),
        )
        if short_raw is None or short_raw.get("bid") in (None, 0):
            # Keine Liquidität im Short-Leg-Fenster → wie Produktion
            # (_find_spread_leg liefert None → Fallback auf Long-Leg-Strategie).
            return _strategy_row(default_strategy, "spread_no_liquidity", [long_leg], iv_rank, effective_gate, expiry)

        short_leg = _leg_from_contract(short_raw, "short")
        row = _strategy_row(strategy, "computed", [long_leg, short_leg], iv_rank, effective_gate, expiry)

        long_ask, short_bid = long_leg.get("ask"), short_leg.get("bid")
        long_mid, short_mid = long_leg.get("mid"), short_leg.get("mid")
        if long_ask is not None and short_bid is not None:
            row["net_debit_entry"] = round(long_ask - short_bid, 4)
        if long_mid is not None and short_mid is not None:
            row["net_mid_entry"] = round(long_mid - short_mid, 4)
        return row
    except Exception as ex:
        log.debug(f"candidate_ledger._build_real_strategy Fehler (ignoriert): {ex}")
        return None


def _finalize_real_strategy(rs: dict, session: str, today: str) -> dict:
    """Analog zu _finalize_real_option, aber für alle Legs von real_strategy
    (Multi-Leg-Session-Gating): Optionen handeln nur in der regulären
    Session — außerhalb wandern alle Leg-Quotes nach snapshot_legs
    (informativ) und die "echten" bid/ask/mid je Leg bleiben None bis zum
    Fill in update_outcomes() (_fill_real_strategy_entries)."""
    rs = dict(rs)
    legs = [dict(l) for l in rs.get("legs", [])]
    if session == "regular":
        rs["entry_pending"] = False
        rs["entry_date"]    = today
    else:
        rs["snapshot_legs"] = [
            {"symbol": l.get("symbol"), "bid": l.get("bid"), "ask": l.get("ask"), "mid": l.get("mid")}
            for l in legs
        ]
        for l in legs:
            l["bid"] = l["ask"] = l["mid"] = None
        rs["net_debit_entry"] = None
        rs["net_mid_entry"]   = None
        rs["entry_pending"]   = True
    rs["legs"] = legs
    return rs


def _now_utc() -> datetime:
    """Isoliert für Tests (monkeypatch), damit „jetzige Session" injizierbar ist."""
    return datetime.now(timezone.utc)


def _finalize_real_option(raw_contract: dict, session: str, today: str) -> dict:
    """
    Optionen handeln NUR in der regulären Session — ein Kontrakt, der
    pre/post/closed ausgewählt wurde, bekommt seine bid/ask/mid-Snapshot-
    Quote (zum Auswahl-Zeitpunkt) NICHT als Entry unterstellt (das wäre
    exakt das Look-ahead-Problem, das für das Underlying schon gelöst
    wurde). Stattdessen:

        session == "regular" → sofortiger Entry: entry_pending=False,
            entry_quote_ts=quote_ts, entry_date=heute (Signal-Tag).
        sonst                 → entry_pending=True, bid/ask/mid der
            Auswahl wandern nach snapshot_quote (rein informativ), die
            "echten" bid/ask/mid bleiben None bis zum Fill in
            update_outcomes() (siehe _fill_real_option_entries).
    """
    contract = dict(raw_contract)
    if session == "regular":
        contract["entry_pending"] = False
        contract["entry_quote_ts"] = raw_contract.get("quote_ts")
        contract["entry_date"] = today
    else:
        contract["snapshot_quote"] = {
            "bid":      raw_contract.get("bid"),
            "ask":      raw_contract.get("ask"),
            "mid":      raw_contract.get("mid"),
            "quote_ts": raw_contract.get("quote_ts"),
        }
        contract["bid"] = None
        contract["ask"] = None
        contract["mid"] = None
        contract["entry_pending"] = True
    return contract


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
    """Schreibt alle Signale des aktuellen Runs als JSONL-Zeilen (dedup je
    Tag+Event; P0-1: ein Ticker kann mehrere Signale/Zeilen haben)."""
    try:
        if _state.get("flushed"):
            return
        if not _state.get("date") or not _state.get("entries"):
            _state["flushed"] = True
            return

        today   = _state["date"]
        entries = _state["entries"]   # ticker -> [signal, ...] (P0-1)

        root = Path(reports_dir_root)
        root.mkdir(parents=True, exist_ok=True)
        month_file = root / f"{today[:7]}.jsonl"

        all_signals = [(t, sig) for t, lst in entries.items() for sig in lst]

        # event_id je Signal bestimmen (braucht die evtl. bis hier
        # gesammelten event_key/features).
        for ticker, e in all_signals:
            e["event_id"] = _compute_event_id(ticker, today, e)

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
                            # Dedup-Schlüssel inkl. event_id (P0-1/P2): zwei
                            # verschiedene Events desselben Tickers am selben
                            # Tag bleiben beide erhalten (je eine eigene
                            # Zeile). Ältere Zeilen ohne event_id werden über
                            # ihren (fehlenden) Wert weiterhin eindeutig
                            # identifiziert.
                            existing_keys.add((row.get("date"), row.get("ticker"), row.get("event_id")))
                        except Exception:
                            continue
            except Exception as e:
                log.debug(f"candidate_ledger: Ledger-Datei nicht lesbar: {e}")

        to_process = [
            (t, e) for (t, e) in all_signals
            if (today, t, e.get("event_id")) not in existing_keys
        ]

        tickers_to_process = sorted({t for t, _ in to_process})
        prices            = _fetch_prices_batch(tickers_to_process)
        underlying_quotes = market_snapshot.fetch_underlying_quotes(tickers_to_process)

        resolved_by_id = {
            e["signal_id"]: _resolve_entry(ticker, e, underlying_quotes, prices)
            for ticker, e in to_process
        }

        # ── P1: neutraler Options-Snapshot-Budget ────────────────────────────
        # Statt die ersten `max_snapshots` Signale in Verarbeitungsreihenfolge
        # zu bedienen (systematische Verzerrung — z.B. bevorzugt Ticker, die
        # zuerst in die Pipeline einliefen), wird zuerst die vollständige
        # Menge der ELIGIBLEN Signale (bekannte Richtung + auflösbarer Spot)
        # bestimmt. Übersteigt sie das Budget, wird eine deterministische
        # Pseudo-Zufallsauswahl getroffen (seeded by date via
        # sha1(date+signal_id)) — NIEMALS "die ersten N".
        eligible = []  # (ticker, signal, spot)
        for ticker, e in to_process:
            if e.get("direction") not in ("BULLISH", "BEARISH"):
                continue
            resolved = resolved_by_id[e["signal_id"]]
            spot = resolved["entry_price"]
            if spot in (None, 0):
                spot = resolved["underlying"].get("last") or resolved["underlying"].get("mid")
            if spot in (None, 0):
                spot = prices.get(ticker)
            if spot in (None, 0):
                continue
            eligible.append((ticker, e, spot))

        max_snapshots = _max_option_snapshots()
        eligible_n    = len(eligible)
        if eligible_n > max_snapshots:
            ranked = sorted(
                eligible,
                key=lambda item: hashlib.sha1(
                    f"{today}:{item[1]['signal_id']}".encode("utf-8")
                ).hexdigest(),
            )
            selected_ids = {sig["signal_id"] for (_, sig, _) in ranked[:max_snapshots]}
        else:
            selected_ids = {sig["signal_id"] for (_, sig, _) in eligible}
        spot_by_id = {sig["signal_id"]: spot for (_, sig, spot) in eligible}

        new_lines = []
        for ticker, e in to_process:
            # Ohne reject()-Aufruf ausgeschieden (z.B. Prescreening) → als
            # "dropped" mit letzter erreichter Stufe markieren statt "seen".
            if e.get("status", "seen") == "seen":
                e["status"]        = "dropped"
                e["reject_stage"]  = e.get("stage")
                e["reject_reason"] = f"unlabeled_after_{e.get('stage') or 'unknown'}"

            resolved    = resolved_by_id[e["signal_id"]]
            entry_price = resolved["entry_price"]

            row = {
                "date":             today,
                "ticker":           ticker,
                "signal_id":        e.get("signal_id"),
                "event_id":         e.get("event_id"),
                "event_key":        e.get("event_key"),
                "pipeline_version": _state.get("pipeline_version", "unknown"),
                "config_hash":      _state.get("config_hash", "unknown"),
                "code_sha":         _state.get("code_sha", "unknown"),
                "model_ids":        _state.get("model_ids", {}),
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

            # P0-B/P0-2: echter Options-Kontrakt (+ Produktions-Strategie-
            # Counterfactual) — nur für Kandidaten mit bekannter Richtung,
            # begrenzt auf `ledger.max_option_snapshots` API-Calls/Lauf (P1:
            # neutrale Zufallsauswahl statt "erste N", siehe oben).
            if e.get("direction") in ("BULLISH", "BEARISH"):
                signal_id = e["signal_id"]
                row["snapshot_eligible_n"] = eligible_n
                row["snapshot_budget"]     = max_snapshots
                if signal_id not in spot_by_id:
                    row["snapshot_selected"]       = False
                    row["real_option_skip_reason"] = "no_spot"
                elif signal_id not in selected_ids:
                    row["snapshot_selected"]       = False
                    row["real_option_skip_reason"] = "budget_random_exclusion"
                else:
                    row["snapshot_selected"] = True
                    spot = spot_by_id[signal_id]
                    real_option_raw = _build_real_option(e, spot)
                    if real_option_raw is not None:
                        row["real_option"] = _finalize_real_option(real_option_raw, resolved["session"], today)
                        real_strategy_raw = _build_real_strategy(e, ticker, spot, real_option_raw)
                        if real_strategy_raw is not None:
                            row["real_strategy"] = _finalize_real_strategy(real_strategy_raw, resolved["session"], today)
                    else:
                        row["real_option_skip_reason"] = (
                            "no_api_key" if not os.environ.get("TRADIER_API_KEY", "").strip()
                            else "no_contract_found"
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
        changed |= _fill_real_option_entries(rows)
    except Exception as e:
        log.debug(f"candidate_ledger: real_option Entry-Fill Fehler (ignoriert): {e}")
    try:
        changed |= _fill_real_strategy_entries(rows)
    except Exception as e:
        log.debug(f"candidate_ledger: real_strategy Entry-Fill Fehler (ignoriert): {e}")
    try:
        changed |= _fill_return_horizons(rows, today_dt)
    except Exception as e:
        log.debug(f"candidate_ledger: Horizont-Fill Fehler (ignoriert): {e}")
    try:
        changed |= _fill_real_option_marks(rows, today_dt)
    except Exception as e:
        log.debug(f"candidate_ledger: real_opt_ret-Marks Fehler (ignoriert): {e}")
    try:
        changed |= _fill_real_strategy_marks(rows, today_dt)
    except Exception as e:
        log.debug(f"candidate_ledger: real_strat_ret-Marks Fehler (ignoriert): {e}")

    if changed:
        _atomic_write_jsonl(path, rows)


def _fill_real_option_entries(rows: list[dict]) -> bool:
    """
    Optionen handeln nur in der regulären Session. Ein real_option-Kontrakt,
    der pre/post/closed ausgewählt wurde (entry_pending=True), bekommt seinen
    tatsächlichen Entry-Preis erst hier — beim ERSTEN feedback.py-Lauf
    (2×/Tag), der während einer regulären Session läuft — per EINEM
    gebündelten fetch_option_quotes-Call. Läuft dieser Aufruf außerhalb der
    regulären Session, wird nichts befüllt (nächster Lauf versucht es erneut).
    """
    now = _now_utc()
    session = market_snapshot.us_market_session(now)
    if session != "regular":
        return False

    pending = [
        row for row in rows
        if isinstance(row.get("real_option"), dict) and row["real_option"].get("entry_pending")
    ]
    if not pending:
        return False

    symbols = sorted({
        row["real_option"].get("symbol") for row in pending
        if row["real_option"].get("symbol")
    })
    if not symbols:
        return False
    quotes = market_snapshot.fetch_option_quotes(symbols)

    changed = False
    for row in pending:
        try:
            ro = row["real_option"]
            q = quotes.get(ro.get("symbol"))
            if not q:
                continue
            bid, ask, mid = q.get("bid"), q.get("ask"), q.get("mid")
            if bid is None and ask is None and mid is None:
                continue
            ro["bid"], ro["ask"], ro["mid"] = bid, ask, mid
            ro["entry_pending"]   = False
            ro["entry_filled_at"] = now.isoformat(timespec="seconds")
            ro["entry_date"]      = now.strftime("%Y-%m-%d")
            changed = True
        except Exception as e:
            log.debug(f"candidate_ledger._fill_real_option_entries Fehler (ignoriert): {e}")
    return changed


def _fill_real_strategy_entries(rows: list[dict]) -> bool:
    """
    P0-2: Pendant zu _fill_real_option_entries, aber für ALLE Legs von
    real_strategy (Multi-Leg-Fill — Spread braucht Long- UND Short-Leg-Quote,
    bevor entry_pending=False gesetzt wird; nur regulare Session, EIN
    gebündelter fetch_option_quotes-Call über alle offenen Legs)."""
    now = _now_utc()
    session = market_snapshot.us_market_session(now)
    if session != "regular":
        return False

    pending = [
        row for row in rows
        if isinstance(row.get("real_strategy"), dict) and row["real_strategy"].get("entry_pending")
    ]
    if not pending:
        return False

    symbols = sorted({
        leg.get("symbol")
        for row in pending
        for leg in (row["real_strategy"].get("legs") or [])
        if leg.get("symbol")
    })
    if not symbols:
        return False
    quotes = market_snapshot.fetch_option_quotes(symbols)

    changed = False
    for row in pending:
        try:
            rs   = row["real_strategy"]
            legs = rs.get("legs") or []
            if not legs:
                continue

            all_filled = True
            for leg in legs:
                q = quotes.get(leg.get("symbol"))
                if not q or (q.get("bid") is None and q.get("ask") is None and q.get("mid") is None):
                    all_filled = False
                    continue
                leg["bid"], leg["ask"], leg["mid"] = q.get("bid"), q.get("ask"), q.get("mid")
            if not all_filled:
                continue  # nächster Regular-Session-Lauf holt die fehlenden Legs nach

            long_leg  = next((l for l in legs if l.get("side") == "long"), None)
            short_leg = next((l for l in legs if l.get("side") == "short"), None)
            if long_leg is None:
                continue

            if short_leg is not None:
                if long_leg.get("ask") is not None and short_leg.get("bid") is not None:
                    rs["net_debit_entry"] = round(long_leg["ask"] - short_leg["bid"], 4)
                if long_leg.get("mid") is not None and short_leg.get("mid") is not None:
                    rs["net_mid_entry"] = round(long_leg["mid"] - short_leg["mid"], 4)
            else:
                rs["net_debit_entry"] = long_leg.get("ask")
                rs["net_mid_entry"]   = long_leg.get("mid")

            rs["entry_pending"]   = False
            rs["entry_filled_at"] = now.isoformat(timespec="seconds")
            rs["entry_date"]      = now.strftime("%Y-%m-%d")
            changed = True
        except Exception as e:
            log.debug(f"candidate_ledger._fill_real_strategy_entries Fehler (ignoriert): {e}")
    return changed


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
    P0-B: für Zeilen mit real_option (bereits gefülltem Entry — kein
    entry_pending mehr) und verstrichenem Horizont h wird real_opt_ret_{h}d
    (konservativ: Kauf zum Ask, Verkauf zum Bid) sowie real_opt_ret_mid_{h}d
    (Mid/Mid) nachgetragen. Für bereits verfallene Kontrakte wird der
    Intrinsic-Wert aus dem Underlying-Schlusskurs am Expiry-Tag verwendet
    statt einer (nicht mehr existierenden) Live-Quote.

    Optionen handeln nur in der regulären Session — Marks passieren daher
    NUR, wenn der aktuelle Lauf (2×/Tag via feedback.py) während einer
    regulären Session läuft; sonst wird nichts markiert (nächster
    Regular-Session-Lauf holt es nach). Horizonte laufen ab dem TATSÄCHLICH
    gefüllten Entry-Datum (real_option["entry_date"]), nicht ab dem
    ursprünglichen Signal-Datum. Zeilen, deren Horizont schon >7 Tage
    verstrichen ist, werden trotzdem jetzt mit der aktuellen Quote markiert,
    aber zusätzlich mit late_mark=true geflaggt.
    """
    session = market_snapshot.us_market_session(_now_utc())
    if session != "regular":
        return False

    tasks = []  # (row, h, entry_dt)
    for row in rows:
        real_option = row.get("real_option")
        if not real_option or real_option.get("entry_pending"):
            continue
        entry_dt = _parse_date(real_option.get("entry_date") or "")
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


def _fill_real_strategy_marks(rows: list[dict], today_dt: datetime) -> bool:
    """
    P0-2: Pendant zu _fill_real_option_marks, aber für real_strategy
    (Long-Leg allein ODER Long+Short-Leg bei einem Spread):

        exit_conservative = long_bid - short_ask (floored bei 0; für
            Single-Leg-Strategien einfach long_bid, wie bisher)
        exit_mid           = long_mid - short_mid (Single-Leg: long_mid)
        real_strat_ret_{h}d      = exit_conservative / net_debit_entry - 1
        real_strat_ret_mid_{h}d  = exit_mid / net_mid_entry - 1

    Am/nach Expiry: Intrinsic-Wert BEIDER Legs aus dem Underlying-
    Schlusskurs am Expiry-Tag (Long-Intrinsic - Short-Intrinsic, ebenfalls
    bei 0 gefloort für den konservativen Wert). Session-Gating und
    late_mark wie bei real_option.
    """
    session = market_snapshot.us_market_session(_now_utc())
    if session != "regular":
        return False

    tasks = []  # (row, h, entry_dt)
    for row in rows:
        rs = row.get("real_strategy")
        if not rs or rs.get("entry_pending"):
            continue
        entry_dt = _parse_date(rs.get("entry_date") or "")
        if entry_dt is None:
            continue
        outcomes = row.setdefault("outcomes", {})
        for h in HORIZONS:
            if f"real_strat_ret_{h}d" in outcomes:
                continue
            if (today_dt - entry_dt).days >= h:
                tasks.append((row, h, entry_dt))

    if not tasks:
        return False

    symbols = sorted({
        leg.get("symbol")
        for t in tasks
        for leg in (t[0]["real_strategy"].get("legs") or [])
        if leg.get("symbol")
    })
    quotes = market_snapshot.fetch_option_quotes(symbols)

    underlying_tickers = sorted({t[0].get("ticker") for t in tasks if t[0].get("ticker")})
    max_days = max((today_dt - t[2]).days for t in tasks)
    underlying_hist = _fetch_history_batch(underlying_tickers, max(max_days + 10, 15))

    changed = False
    for row, h, entry_dt in tasks:
        try:
            rs   = row.get("real_strategy") or {}
            legs = rs.get("legs") or []
            long_leg  = next((l for l in legs if l.get("side") == "long"), None)
            short_leg = next((l for l in legs if l.get("side") == "short"), None)
            if long_leg is None:
                continue

            net_debit_entry = rs.get("net_debit_entry")
            net_mid_entry   = rs.get("net_mid_entry")
            if net_debit_entry in (None, 0) or net_mid_entry in (None, 0):
                continue

            expiry_dt = _parse_date(rs.get("expiry") or "")
            kind = "call" if row.get("direction") == "BULLISH" else "put"

            if expiry_dt is not None and today_dt.date() > expiry_dt.date():
                hist = underlying_hist.get(row.get("ticker"))
                spot_at_expiry = _price_on_or_before(hist, expiry_dt) if hist else None
                if spot_at_expiry is None:
                    continue

                def _intrinsic(leg):
                    strike = leg.get("strike") or 0
                    return (max(spot_at_expiry - strike, 0.0) if kind == "call"
                            else max(strike - spot_at_expiry, 0.0))

                long_intrinsic  = _intrinsic(long_leg)
                short_intrinsic = _intrinsic(short_leg) if short_leg is not None else 0.0
                exit_conservative = max(long_intrinsic - short_intrinsic, 0.0)
                exit_mid          = long_intrinsic - short_intrinsic
            else:
                q_long = quotes.get(long_leg.get("symbol"))
                if not q_long:
                    continue
                long_bid, long_mid_q = q_long.get("bid"), q_long.get("mid")
                if long_bid is None or long_mid_q is None:
                    continue

                if short_leg is not None:
                    q_short = quotes.get(short_leg.get("symbol"))
                    if not q_short:
                        continue
                    short_ask, short_mid_q = q_short.get("ask"), q_short.get("mid")
                    if short_ask is None or short_mid_q is None:
                        continue
                    exit_conservative = max(long_bid - short_ask, 0.0)
                    exit_mid          = long_mid_q - short_mid_q
                else:
                    exit_conservative = long_bid
                    exit_mid          = long_mid_q

            outcomes = row.setdefault("outcomes", {})
            outcomes[f"real_strat_ret_{h}d"]       = round(exit_conservative / net_debit_entry - 1.0, 4)
            outcomes[f"real_strat_ret_mid_{h}d"]   = round(exit_mid / net_mid_entry - 1.0, 4)
            outcomes[f"real_strat_mark_date_{h}d"] = today_dt.strftime("%Y-%m-%d")

            if (today_dt - (entry_dt + timedelta(days=h))).days > LATE_MARK_DAYS:
                row["late_mark"] = True

            changed = True
        except Exception as e:
            log.debug(f"candidate_ledger._fill_real_strategy_marks Fehler (ignoriert): {e}")
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
