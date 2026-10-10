"""modules/option_quotes.py – dauerhaftes, append-only Archiv BEOBACHTETER Optionsquotes.

Zweck: Optionsergebnisse (Entry, Marks, Exits) waren bisher nur über Modellpreise bzw. flüchtige
Tradier-Antworten nachvollziehbar. Dieses Modul legt jede tatsächlich beobachtete Quote (Bid/Ask/
Last/Volumen/OI/IV/Greeks) mit Herkunft und Abrufzeit ab. So entsteht ein wachsendes historisches
Optionsarchiv, und Optionsergebnisse werden reproduzierbar.

WICHTIG:
  - Gespeichert werden AUSSCHLIESSLICH beobachtete Quotes (PRICE_TYPE = "OBSERVED_QUOTE").
    Modellierte Preise (Black-Scholes, Delta-Näherungen, Interpolationen) dürfen hier NIE
    hineingeschrieben werden – `archive` verwirft jeden Record mit anderem price_type.
  - Es wird nichts erfunden: fehlende oder nicht-endliche Werte bleiben None (nie 0), ein
    unbekannter Multiplikator wird mit multiplier_assumed=True gekennzeichnet.
  - Keine öffentliche Funktion wirft in den Aufrufer (try/except + Logging); `occ_symbol` ist der
    einzige Helfer, der ValueError wirft (und `normalize` fängt ihn).

Datenformat: outputs/options_archive/quotes/YYYY-MM-DD.jsonl (UTC-Datum von `retrieved_at`),
eine Zeile je Quote, Felder siehe `normalize`. Idempotent über `record_id`.

quality_flags (sortierte Liste) und quality_status:
  CROSSED       bid > ask                              WIDE_SPREAD  spread/mid > WIDE_SPREAD_PCT
  LOCKED        bid == ask > 0                         NO_QUOTE_TIME  keine Börsen-Zeit der Quote
  ZERO_BID      bid == 0                               STALE        retrieved_at - quote_time > STALE_HOURS
  NO_BID        bid fehlt                              IV_MISSING   keine (positive) implizite Vola
  NO_ASK        ask fehlt oder 0                       OI_ZERO      open_interest == 0
  INVALID_VALUE negativer Preis/Volumen/OI/Strike/IV/Multiplikator (Wert -> None)
  MULTIPLIER_ASSUMED  Multiplikator nicht geliefert, 100 angenommen
  UNUSABLE = CROSSED | NO_ASK | INVALID_VALUE; SUSPECT = STALE | WIDE_SPREAD | ZERO_BID | LOCKED |
  NO_BID; sonst OK. Der Status filtert NICHT – auch UNUSABLE-Quotes werden archiviert (sie wurden
  beobachtet); Konsumenten entscheiden anhand von quality_status.

Point-in-time (PIT): `latest_quote(occ, as_of)` liefert die jüngste Quote mit
retrieved_at <= as_of – also nur, was zum Zeitpunkt as_of tatsächlich schon abgerufen war.
`quote_time` (Börsenzeit) dient der Qualität (STALE), nie der PIT-Auswahl.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from modules.atomic_io import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

SCHEMA_VERSION = "option-quote-v1"
ROOT = Path("outputs/options_archive/quotes")
PRICE_TYPE = "OBSERVED_QUOTE"        # einziger Preistyp dieses Archivs (nie Modellpreise)
WIDE_SPREAD_PCT = 0.5
STALE_HOURS = 26
DEFAULT_MULTIPLIER = 100

_OCC_RE = re.compile(r"^([A-Z0-9.\-]{1,10})(\d{6})([CP])(\d{8})$")
_NUMERIC_STR_RE = re.compile(r"^-?\d+(\.\d+)?$")
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------------------------
def _norm_right(v) -> str | None:
    s = str(v).strip().lower() if v is not None else ""
    return {"c": "call", "call": "call", "p": "put", "put": "put"}.get(s)


def occ_symbol(underlying: str, expiry: str, right: str, strike: float) -> str:
    """OCC/Tradier-Symbol ohne Padding der Root: ROOT + YYMMDD + C|P + Strike*1000 (8-stellig).

    Beispiel: ("AAPL", "2026-11-20", "call", 150.0) -> "AAPL261120C00150000". Wirft ValueError.
    """
    root = underlying.strip().upper() if isinstance(underlying, str) else ""
    if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", root):
        raise ValueError(f"ungültiger Underlying: {underlying!r}")
    if isinstance(expiry, datetime):
        d = expiry.date()
    elif isinstance(expiry, date):
        d = expiry
    else:
        try:
            d = date.fromisoformat(str(expiry).strip()[:10])
        except ValueError as e:
            raise ValueError(f"ungültiger Verfall: {expiry!r}") from e
    cp = _norm_right(right)
    if cp is None:
        raise ValueError(f"ungültiges Right: {right!r}")
    k = _num(strike)
    if k is None or k <= 0 or round(k * 1000) > 99_999_999:
        raise ValueError(f"ungültiger Strike: {strike!r}")
    return f"{root}{d:%y%m%d}{cp[0].upper()}{round(k * 1000):08d}"


def _num(v) -> float | None:
    """Endliche Zahl oder None (bool, None, Text, NaN/Inf -> None – nie 0)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _parse_ts(v) -> datetime | None:
    """Epoch (ms oder s), ISO-String, datetime/date -> UTC-datetime; sonst None.
    Naive Zeiten gelten als UTC."""
    if v is None or isinstance(v, bool):
        return None
    try:
        if isinstance(v, datetime):
            dt = v
        elif isinstance(v, date):
            dt = datetime(v.year, v.month, v.day)
        elif isinstance(v, str) and not _NUMERIC_STR_RE.match(v.strip()):
            dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        else:
            x = _num(v)
            if x is None:
                return None
            dt = datetime.fromtimestamp(x / 1000 if abs(x) >= 1e11 else x, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _pick(raw: dict, *keys):
    """Erster nicht-None-Wert der Schlüssel; danach im verschachtelten Tradier-`greeks`-Dict."""
    nested = raw.get("greeks")
    for src in (raw, nested if isinstance(nested, dict) else {}):
        for k in keys:
            if src.get(k) is not None:
                return src[k]
    return None


def _clean_symbol(v) -> str | None:
    """Kanonisches OCC-Symbol (Großbuchstaben, ohne Leerzeichen) oder None, wenn kein OCC-Format."""
    if not isinstance(v, str):
        return None
    s = v.strip().upper().replace(" ", "")
    if not _OCC_RE.match(s):
        return None
    try:
        _parse_occ(s)   # unmögliches Datum (z. B. Monat 13) -> kein gültiges Symbol
    except ValueError:
        return None
    return s


def _parse_occ(sym: str) -> tuple[str, str, str, float]:
    root, ymd, cp, k = _OCC_RE.match(sym).groups()
    expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])).isoformat()
    return root, expiry, "call" if cp == "C" else "put", int(k) / 1000


def _checked(v, flags: set[str]) -> float | None:
    """Preis/Volumen/OI: negative Werte -> None + INVALID_VALUE."""
    x = _num(v)
    if x is not None and x < 0:
        flags.add("INVALID_VALUE")
        return None
    return x


def _count(x: float | None) -> int | float | None:
    return int(x) if x is not None and x.is_integer() else x


def _r6(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def _payload_hash(raw: dict) -> str:
    try:
        text = json.dumps(raw, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = repr(raw)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------------------------------
def normalize(raw: dict, *, underlying: str, source: str, context: str,
              retrieved_at: datetime | str | None = None) -> dict | None:
    """Rohquote (Tradier/yfinance/…) -> Archiv-Record; None, wenn kein Optionssymbol bestimmbar.
    Wirft nie."""
    try:
        return _normalize(raw, underlying, source, context, retrieved_at)
    except Exception as e:  # noqa: BLE001 – Archiv darf den Aufrufer nie crashen
        log.warning(f"option_quotes.normalize: {type(e).__name__}: {e}")
        return None


def _normalize(raw, underlying, source, context, retrieved_at) -> dict | None:
    if not isinstance(raw, dict):
        log.debug("option_quotes.normalize: raw ist kein dict")
        return None
    ra = datetime.now(timezone.utc) if retrieved_at is None else _parse_ts(retrieved_at)
    if ra is None:
        log.debug(f"option_quotes.normalize: retrieved_at nicht lesbar: {retrieved_at!r}")
        return None
    flags: set[str] = set()
    und = underlying.strip().upper() if isinstance(underlying, str) else ""

    # Kontrakt-Identität: gegebenes OCC-Symbol, sonst aus Verfall/Right/Strike gebaut
    sym = _clean_symbol(raw.get("symbol")) or _clean_symbol(raw.get("occ_symbol"))
    p_root, p_exp, p_right, p_strike = _parse_occ(sym) if sym else (None, None, None, None)
    expiry_v = _pick(raw, "expiry", "expiration_date")
    try:
        expiry = date.fromisoformat(str(expiry_v).strip()[:10]).isoformat() if expiry_v is not None else None
    except ValueError:
        expiry = None
    expiry = expiry or p_exp
    right = _norm_right(_pick(raw, "right", "option_type")) or p_right
    strike = _num(_pick(raw, "strike"))
    if strike is not None and strike <= 0:
        flags.add("INVALID_VALUE")
        strike = None
    strike = strike if strike is not None else p_strike
    if sym is None:
        try:
            sym = occ_symbol(und, expiry, right, strike)
        except ValueError as e:
            log.debug(f"option_quotes.normalize: kein Optionssymbol bestimmbar ({e})")
            return None

    # Preise / Volumen / OI (fehlend -> None, nie 0)
    bid, ask, last = (_checked(_pick(raw, k), flags) for k in ("bid", "ask", "last"))
    volume = _count(_checked(_pick(raw, "volume"), flags))
    open_interest = _count(_checked(_pick(raw, "open_interest"), flags))
    iv = _num(_pick(raw, "iv", "mid_iv", "smv_vol", "impliedVolatility"))
    if iv is not None and iv < 0:
        flags.add("INVALID_VALUE")
    if iv is not None and iv <= 0:   # 0 = "nicht berechenbar" (Tradier) -> fehlend
        iv = None
    delta, gamma, theta, vega = (_num(_pick(raw, k)) for k in ("delta", "gamma", "theta", "vega"))

    mult = _num(_pick(raw, "multiplier", "contract_size"))
    if mult is not None and mult <= 0:
        flags.add("INVALID_VALUE")
        mult = None
    assumed = mult is None
    if assumed:
        mult = DEFAULT_MULTIPLIER
        flags.add("MULTIPLIER_ASSUMED")
    mult = int(mult) if float(mult).is_integer() else mult

    # Abgeleitete Größen
    mid = (bid + ask) / 2 if bid is not None and ask is not None and 0 < bid <= ask else None
    spread = ask - bid if bid is not None and ask is not None else None
    spread_pct = spread / mid if spread is not None and mid else None
    if bid is None:
        flags.add("NO_BID")
    elif bid == 0:
        flags.add("ZERO_BID")
    if ask is None or ask == 0:
        flags.add("NO_ASK")
    if bid is not None and ask is not None:
        if bid > ask:
            flags.add("CROSSED")
        elif bid == ask > 0:
            flags.add("LOCKED")
    if spread_pct is not None and spread_pct > WIDE_SPREAD_PCT:
        flags.add("WIDE_SPREAD")
    if iv is None:
        flags.add("IV_MISSING")
    if open_interest == 0:
        flags.add("OI_ZERO")

    qt = None
    for k in ("quote_time", "ts", "bid_date", "ask_date", "trade_date"):
        qt = _parse_ts(raw.get(k))
        if qt is not None:
            break
    if qt is None:
        flags.add("NO_QUOTE_TIME")
    elif ra - qt > timedelta(hours=STALE_HOURS):
        flags.add("STALE")

    if flags & {"CROSSED", "NO_ASK", "INVALID_VALUE"}:
        status = "UNUSABLE"
    elif flags & {"STALE", "WIDE_SPREAD", "ZERO_BID", "LOCKED", "NO_BID"}:
        status = "SUSPECT"
    else:
        status = "OK"

    qt_s, ra_s = (_iso(qt) if qt else None), _iso(ra)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": hashlib.sha256(f"{sym}|{qt_s or ra_s}|{source}|{context}".encode("utf-8")).hexdigest()[:20],
        "occ_symbol": sym,
        "underlying": und or p_root,
        "expiry": expiry, "strike": strike, "right": right,
        "multiplier": mult, "multiplier_assumed": assumed,
        "bid": bid, "ask": ask, "mid": _r6(mid), "spread": _r6(spread), "spread_pct": _r6(spread_pct),
        "last": last, "volume": volume, "open_interest": open_interest,
        "iv": iv, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega,
        "quote_time": qt_s, "retrieved_at": ra_s,
        "source": source, "context": context, "price_type": PRICE_TYPE,
        "quality_flags": sorted(flags), "quality_status": status,
        "payload_hash": _payload_hash(raw),
    }


# ---------------------------------------------------------------------------------------------
# Archiv
# ---------------------------------------------------------------------------------------------
def _day_path(base: Path, day: str) -> Path:
    return base / f"{day}.jsonl"


def _valid_day(r) -> str | None:
    """UTC-Tag (YYYY-MM-DD) eines strukturell gültigen OBSERVED_QUOTE-Records, sonst None."""
    if not isinstance(r, dict):
        return None
    if r.get("schema_version") != SCHEMA_VERSION or r.get("price_type") != PRICE_TYPE:
        log.warning("option_quotes.archive: Record mit fremdem schema_version/price_type verworfen")
        return None
    if not (isinstance(r.get("record_id"), str) and r["record_id"]
            and isinstance(r.get("occ_symbol"), str) and r["occ_symbol"]):
        return None
    ra = _parse_ts(r.get("retrieved_at"))
    return ra.date().isoformat() if ra else None


def archive(records: list[dict], *, root: Path | None = None) -> int:
    """Hängt normalisierte Records an root/YYYY-MM-DD.jsonl (UTC-Datum von retrieved_at) an.
    Idempotent über record_id; Rückgabe = neu geschriebene Zeilen; wirft nie."""
    written = 0
    try:
        base = Path(root) if root is not None else ROOT
        by_day: dict[str, list[dict]] = {}
        for r in records or []:
            day = _valid_day(r)
            if day:
                by_day.setdefault(day, []).append(r)
        for day, rows in sorted(by_day.items()):
            try:
                path = _day_path(base, day)
                try:
                    seen = {x.get("record_id") for x in read_jsonl(path) if isinstance(x, dict)}
                except Exception as e:  # noqa: BLE001 – defekte Datei: lieber Duplikate als Datenverlust
                    log.warning(f"option_quotes.archive: {path} nicht lesbar ({e}) – Idempotenz-Prüfung entfällt")
                    seen = set()
                new = []
                for r in rows:
                    if r["record_id"] not in seen:
                        seen.add(r["record_id"])
                        new.append(r)
                written += append_jsonl(path, new)
            except Exception as e:  # noqa: BLE001
                log.warning(f"option_quotes.archive: Tag {day} nicht geschrieben: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"option_quotes.archive: {type(e).__name__}: {e}")
    return written


def record(raws: list[dict], *, underlying: str, source: str, context: str,
           retrieved_at: datetime | str | None = None, root: Path | None = None) -> int:
    """Normalisiert und archiviert Rohquotes; alle Records eines Aufrufs teilen sich retrieved_at.
    Gibt die Zahl neu geschriebener Records zurück (0 bei jedem Fehler); wirft nie."""
    try:
        ra = datetime.now(timezone.utc) if retrieved_at is None else retrieved_at
        recs = [n for n in (normalize(r, underlying=underlying, source=source, context=context,
                                      retrieved_at=ra) for r in raws) if n]
        return archive(recs, root=root)
    except Exception as e:  # noqa: BLE001
        log.warning(f"option_quotes.record: {type(e).__name__}: {e}")
        return 0


def capture(raw: dict, *, underlying: str, source: str, context: str,
            retrieved_at: datetime | str | None = None, root: Path | None = None) -> str | None:
    """Eine beobachtete Quote archivieren und ihre record_id zurückgeben (für Verweise aus Trade-/Outcome-
    Datensätzen). None, wenn die Quote nicht normalisierbar ist; wirft nie."""
    try:
        n = normalize(raw, underlying=underlying, source=source, context=context,
                      retrieved_at=datetime.now(timezone.utc) if retrieved_at is None else retrieved_at)
        if not n:
            return None
        archive([n], root=root)
        return n["record_id"]
    except Exception as e:  # noqa: BLE001
        log.warning(f"option_quotes.capture: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------------------------
# Lesen
# ---------------------------------------------------------------------------------------------
def _read_day(path: Path) -> list[dict]:
    try:
        return [x for x in read_jsonl(path) if isinstance(x, dict)]
    except Exception as e:  # noqa: BLE001 – defekte Tagesdatei überspringen, aber melden
        log.warning(f"option_quotes: {path} nicht lesbar ({e}) – übersprungen")
        return []


def read(start: str, end: str | None = None, *, root: Path | None = None) -> list[dict]:
    """Alle Records der Tage start..end (inklusive, YYYY-MM-DD; end=None -> bis zur letzten Datei)."""
    try:
        base = Path(root) if root is not None else ROOT
        lo = date.fromisoformat(str(start)).isoformat()
        hi = date.fromisoformat(str(end)).isoformat() if end is not None else None
        out: list[dict] = []
        for p in sorted(base.glob("*.jsonl")):
            if _DAY_FILE_RE.match(p.stem) and p.stem >= lo and (hi is None or p.stem <= hi):
                out.extend(_read_day(p))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"option_quotes.read: {type(e).__name__}: {e}")
        return []


def latest_quote(occ_symbol: str, as_of: datetime | str, *, root: Path | None = None,
                 lookback_days: int = 10) -> dict | None:
    """Point-in-time: jüngste Quote des Kontrakts mit retrieved_at <= as_of (nie eine spätere),
    rückwärts über lookback_days Tagesdateien. Bei gleichem retrieved_at gewinnt die später
    geschriebene. Gibt auch SUSPECT/UNUSABLE-Quotes zurück – quality_status prüft der Aufrufer."""
    try:
        ts = _parse_ts(as_of)
        target = str(occ_symbol).strip().upper().replace(" ", "")
        if ts is None or not target:
            return None
        base = Path(root) if root is not None else ROOT
        for i in range(int(lookback_days) + 1):
            path = _day_path(base, (ts.date() - timedelta(days=i)).isoformat())
            if not path.exists():
                continue
            best, best_t = None, None
            for r in _read_day(path):
                if r.get("occ_symbol") != target:
                    continue
                t = _parse_ts(r.get("retrieved_at"))
                if t is not None and t <= ts and (best_t is None or t >= best_t):
                    best, best_t = r, t
            if best is not None:   # Tagesdateien sind nach retrieved_at gruppiert -> jüngster Tag gewinnt
                return best
        return None
    except Exception as e:  # noqa: BLE001
        log.warning(f"option_quotes.latest_quote: {type(e).__name__}: {e}")
        return None
