"""
modules/market_snapshot.py – Tradier-Marktdaten-Snapshots (Observability)

Zweck: liefert die Zutaten, die candidate_ledger.py braucht, um den ECHTEN
handelbaren Zustand eines Kandidaten zum Signal-Zeitpunkt festzuhalten,
statt (wie bisher) blind den letzten yfinance-Tagesschlusskurs als
Entry-Preis zu unterstellen:

  - fetch_underlying_quotes(): gebündelter Tradier-Quote-Abruf für Underlyings
    (bid/ask/mid/last/prev_close/open/quote_ts)
  - us_market_session():       NYSE-Session-Klassifikation (pre/regular/post/
    closed) für einen UTC-Zeitstempel
  - select_contract():         wählt einen ECHTEN Options-Kontrakt (Tradier
    Chain) für Richtung/DTE-Floor/Spot — Ergänzung zum rein synthetischen
    Black-Scholes-hypo_option in candidate_ledger.py
  - fetch_option_quotes():     gebündelter Tradier-Quote-Abruf für
    OCC-Options-Symbole (für real_opt_ret-Marks)

WICHTIG: Dieses Modul ist REINE Observability, genau wie candidate_ledger.py.
  - Es verändert NIEMALS eine Gate-/Trade-Entscheidung.
  - Jede öffentliche Funktion ist defensiv (try/except + Logging) und
    degradiert bei fehlendem TRADIER_API_KEY oder Netzwerkfehlern auf
    None/{} statt die Pipeline zu gefährden.
  - Netzwerkzugriffe sind bewusst in kleine "_fetch_*"-Funktionen isoliert,
    damit Tests sie ohne echtes Netzwerk monkeypatchen können.

NYSE-Feiertage und Early-Closes werden über eine regel-basierte Kalender
(nyse_holidays, nyse_early_closes) berücksichtigt.
"""

import logging
import os
from datetime import datetime, time as dt_time, timezone, date, timedelta
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)

TRADIER_BASE    = "https://api.tradier.com/v1"
TRADIER_TIMEOUT = 10
QUOTE_CHUNK     = 100  # Tradier /markets/quotes: an ~100 Symbolen chunken

NY_TZ = ZoneInfo("America/New_York")

MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
EARLY_CLOSE  = dt_time(13, 0)


# ── NYSE-Kalender (rule-based, pure functions, stdlib only) ──────────────────

def _easter_date(year: int) -> date:
    """
    Compute Easter Sunday using Gregorian/Meeus algorithm.
    Returns the date of Easter Sunday for the given year.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _third_monday(year: int, month: int) -> date:
    """Return the third Monday of the given month."""
    d = date(year, month, 1)
    # Find first Monday
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        first_monday = d
    else:
        first_monday = d + timedelta(days=days_until_monday)
    # Third Monday is 2 weeks after first Monday
    return first_monday + timedelta(weeks=2)


def _last_monday(year: int, month: int) -> date:
    """Return the last Monday of the given month."""
    # Start from the last day of the month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)

    # Back up to the last Monday
    days_back = (last_day.weekday() - 0) % 7  # 0=Monday
    return last_day - timedelta(days=days_back)


def _fourth_thursday(year: int, month: int) -> date:
    """Return the fourth Thursday of the given month."""
    d = date(year, month, 1)
    # Find first Thursday (weekday 3)
    days_until_thursday = (3 - d.weekday()) % 7
    if days_until_thursday == 0:
        first_thursday = d
    else:
        first_thursday = d + timedelta(days=days_until_thursday)
    # Fourth Thursday is 3 weeks after first Thursday
    return first_thursday + timedelta(weeks=3)


def _observe_date(d: date) -> date:
    """
    Apply NYSE observation rule: if Saturday, preceding Friday;
    if Sunday, following Monday.
    EXCEPT: New Year's Day Saturday is NOT observed on Dec 31 of prior year.
    """
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    elif d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    """
    Return the set of NYSE holiday dates for the given year.
    Includes:
    - New Year's Day (Jan 1, observed; Saturday → preceding Friday,
      Sunday → following Monday)
    - Martin Luther King Jr. Day (3rd Monday January)
    - Washington's Birthday (3rd Monday February)
    - Good Friday (Friday before Easter)
    - Memorial Day (last Monday May)
    - Juneteenth (June 19, observed; from 2022 onward)
    - Independence Day (July 4, observed)
    - Labor Day (1st Monday September)
    - Thanksgiving (4th Thursday November)
    - Christmas (Dec 25, observed)

    Special note: New Year's Day on Saturday is NOT observed on Dec 31
    of the prior year per NYSE rule.
    """
    holidays = set()

    # New Year's Day (Jan 1, observed)
    new_year = date(year, 1, 1)
    if new_year.weekday() == 5:  # Saturday
        # NYSE rule: do NOT observe on Dec 31 of prior year
        # Just observe on the Saturday itself? Or not at all?
        # According to the spec, if Saturday, observed on preceding Friday (Dec 31)
        # but NOT per NYSE rule (meaning Dec 31 is not a holiday)
        # So New Year Saturday is observed on... the Saturday? Let me check the spec again.
        # "if Saturday → NOT observed on Dec 31 of prior year per NYSE rule"
        # This suggests: if Jan 1 is Saturday, it's NOT observed on Dec 31 of the prior year,
        # but IS observed on Jan 1 (the Saturday itself) or the Friday before?
        # I think the intention is: Saturday Jan 1 is NOT a holiday at all for NYSE.
        # No, wait. Let me re-read: "if Saturday → NOT observed on Dec 31 of prior year"
        # This is saying the rule is different from typical. Normally Sat → Fri before.
        # But NYSE doesn't observe it on the Friday of prior year. So maybe Jan 1 Sat is
        # NOT a holiday? Or is it observed on Saturday?
        # Actually, re-reading more carefully with the test cases:
        # "2022 New Year Saturday → no Dec 31 2021 holiday"
        # This tells us that when Jan 1 is Saturday (in 2022), Dec 31 2021 is NOT a holiday.
        # So for NYSE, we just observe Jan 1 Saturday as-is (the Saturday), not the Friday.
        holidays.add(new_year)
    elif new_year.weekday() == 6:  # Sunday
        holidays.add(_observe_date(new_year))  # Following Monday (Jan 2)
    else:  # Weekday
        holidays.add(new_year)

    # MLK Day (3rd Monday January)
    holidays.add(_third_monday(year, 1))

    # Washington's Birthday (3rd Monday February)
    holidays.add(_third_monday(year, 2))

    # Good Friday (Friday before Easter)
    easter = _easter_date(year)
    good_friday = easter - timedelta(days=2)
    holidays.add(good_friday)

    # Memorial Day (last Monday May)
    holidays.add(_last_monday(year, 5))

    # Juneteenth (June 19, observed; from 2022 onward)
    if year >= 2022:
        juneteenth = date(year, 6, 19)
        if juneteenth.weekday() == 5:  # Saturday
            holidays.add(juneteenth - timedelta(days=1))  # Preceding Friday
        elif juneteenth.weekday() == 6:  # Sunday
            holidays.add(juneteenth + timedelta(days=1))  # Following Monday
        else:
            holidays.add(juneteenth)

    # Independence Day (July 4, observed)
    july_4 = date(year, 7, 4)
    if july_4.weekday() == 5:  # Saturday
        holidays.add(july_4 - timedelta(days=1))  # Preceding Friday (July 3)
    elif july_4.weekday() == 6:  # Sunday
        holidays.add(july_4 + timedelta(days=1))  # Following Monday (July 5)
    else:
        holidays.add(july_4)

    # Labor Day (1st Monday September)
    d = date(year, 9, 1)
    days_until_monday = (7 - d.weekday()) % 7
    if days_until_monday == 0:
        first_monday = d
    else:
        first_monday = d + timedelta(days=days_until_monday)
    holidays.add(first_monday)

    # Thanksgiving (4th Thursday November)
    holidays.add(_fourth_thursday(year, 11))

    # Christmas (Dec 25, observed)
    christmas = date(year, 12, 25)
    if christmas.weekday() == 5:  # Saturday
        holidays.add(christmas - timedelta(days=1))  # Preceding Friday (Dec 24)
    elif christmas.weekday() == 6:  # Sunday
        holidays.add(christmas + timedelta(days=1))  # Following Monday (Dec 26)
    else:
        holidays.add(christmas)

    return holidays


def nyse_early_closes(year: int) -> set[date]:
    """
    Return the set of NYSE early close dates (close at 13:00 ET) for the given year.
    Includes:
    - Day before Independence Day (July 3) if weekday and not a holiday
    - Day after Thanksgiving (usually Friday)
    - Christmas Eve (Dec 24) if weekday and not a holiday
    """
    early_closes = set()
    holidays = nyse_holidays(year)

    # Day before Independence Day (July 3)
    july_3 = date(year, 7, 3)
    if july_3.weekday() < 5 and july_3 not in holidays:  # weekday (Mon-Fri)
        early_closes.add(july_3)

    # Day after Thanksgiving (4th Thursday November)
    thanksgiving = _fourth_thursday(year, 11)
    day_after = thanksgiving + timedelta(days=1)
    if day_after.weekday() < 5:  # If it's a weekday
        early_closes.add(day_after)

    # Christmas Eve (Dec 24)
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in holidays:  # weekday and not a holiday
        early_closes.add(christmas_eve)

    return early_closes


def is_trading_day(d: date) -> bool:
    """
    Check if a date is a trading day (not a holiday or weekend).
    """
    if d.weekday() >= 5:  # Saturday or Sunday
        return False
    year = d.year
    if d in nyse_holidays(year):
        return False
    return True


def market_close_time(d: date) -> dt_time:
    """
    Return the market close time for a given date.
    13:00 ET for early close days, 16:00 ET otherwise.
    """
    year = d.year
    if d in nyse_early_closes(year):
        return EARLY_CLOSE
    return MARKET_CLOSE


def next_trading_day(d: date) -> date:
    """
    Return the next trading day after the given date.
    Skips weekends and holidays.
    """
    current = d + timedelta(days=1)
    while not is_trading_day(current):
        current = current + timedelta(days=1)
    return current


# ── Tradier Hilfsfunktionen ───────────────────────────────────────────────────

def _tradier_headers() -> dict:
    api_key = os.environ.get("TRADIER_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
    }


def _use_tradier() -> bool:
    """True, wenn TRADIER_API_KEY gesetzt ist. Ohne Key degradieren alle
    Tradier-Features hier auf None/{} (Felder bleiben leer, Reason wird von
    candidate_ledger.py protokolliert)."""
    return bool(os.environ.get("TRADIER_API_KEY", "").strip())


def _chunk(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _f(v):
    """float() mit >0-Sanity-Check; sonst None."""
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _epoch_ms_to_iso(v):
    try:
        if v in (None, "", 0):
            return None
        return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return None


# ── Session-Klassifikation ────────────────────────────────────────────────────

def us_market_session(ts_utc: datetime) -> str:
    """
    Klassifiziert einen (UTC-)Zeitpunkt in eine NYSE-Session:
        "pre"     – Wochentag, vor 09:30 America/New_York
        "regular" – Wochentag, 09:30–close time America/New_York
                    (16:00 normal, 13:00 on early-close days)
        "post"    – Wochentag, nach close time America/New_York
        "closed"  – Samstag/Sonntag oder NYSE-Feiertag

    DST wird korrekt über zoneinfo behandelt. NYSE-Feiertage und Early-Closes
    werden über die rule-based Kalender berücksichtigt.
    """
    try:
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        local = ts_utc.astimezone(NY_TZ)
        d = local.date()

        # Check for weekend
        if d.weekday() >= 5:  # Samstag=5, Sonntag=6
            return "closed"

        # Check for NYSE holiday
        if d in nyse_holidays(d.year):
            return "closed"

        # Determine close time for this date
        close_time = market_close_time(d)

        # Classify by time
        t = local.time()
        if t < MARKET_OPEN:
            return "pre"
        if t < close_time:
            return "regular"
        return "post"
    except Exception as e:
        log.debug(f"market_snapshot.us_market_session Fehler (ignoriert): {e}")
        return "closed"


# ── Underlying-Quotes (gebündelt) ─────────────────────────────────────────────

def _fetch_quotes_raw(symbols: list[str]) -> list[dict]:
    """EIN Tradier /markets/quotes-Call für einen Chunk von Symbolen.
    Netzwerk isoliert, damit Tests dies monkeypatchen können."""
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/quotes",
            params={"symbols": ",".join(symbols), "greeks": "false"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data   = resp.json()
        quotes = data.get("quotes", {}).get("quote", []) or []
        if isinstance(quotes, dict):
            quotes = [quotes]
        return quotes
    except Exception as e:
        log.debug(f"market_snapshot._fetch_quotes_raw Fehler (ignoriert): {e}")
        return []


def _quote_to_dict(q: dict) -> dict:
    bid = _f(q.get("bid"))
    ask = _f(q.get("ask"))
    mid = round((bid + ask) / 2, 4) if (bid is not None and ask is not None) else None
    ts  = (_epoch_ms_to_iso(q.get("bid_date"))
           or _epoch_ms_to_iso(q.get("ask_date"))
           or _epoch_ms_to_iso(q.get("trade_date")))
    return {
        "bid":        bid,
        "ask":        ask,
        "mid":        mid,
        "last":       _f(q.get("last")),
        "prev_close": _f(q.get("prevclose")),
        "open":       _f(q.get("open")),
        "quote_ts":   ts,
        "source":     "tradier",
    }


def fetch_underlying_quotes(tickers: list[str]) -> dict:
    """
    Gebündelter Tradier-Quote-Abruf für Underlyings: EIN Call pro Chunk von
    ~100 Symbolen (GET /markets/quotes?symbols=A,B,C). Ohne TRADIER_API_KEY
    oder bei Fehlern: leeres Dict — candidate_ledger.py fällt dann auf den
    bestehenden yfinance-Preis zurück und protokolliert die Degradation.

    Returns: {ticker: {bid, ask, mid, last, prev_close, open, quote_ts, source}}
    """
    out = {}
    tickers = [t for t in dict.fromkeys(tickers) if t]
    if not tickers or not _use_tradier():
        return out
    for chunk in _chunk(tickers, QUOTE_CHUNK):
        try:
            for q in _fetch_quotes_raw(chunk):
                sym = q.get("symbol")
                if not sym:
                    continue
                out[sym] = _quote_to_dict(q)
        except Exception as e:
            log.debug(f"market_snapshot.fetch_underlying_quotes Fehler (ignoriert): {e}")
    return out


# ── Echter Options-Kontrakt (Chain-Snapshot) ─────────────────────────────────

def _fetch_expirations(ticker: str) -> list[str]:
    """Netzwerk isoliert, damit Tests dies monkeypatchen können."""
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/expirations",
            params={"symbol": ticker, "includeAllRoots": "true"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data  = resp.json()
        dates = data.get("expirations", {}).get("date", []) or []
        if isinstance(dates, str):
            dates = [dates]
        return sorted(dates)
    except Exception as e:
        log.debug(f"market_snapshot._fetch_expirations [{ticker}] Fehler (ignoriert): {e}")
        return []


def _fetch_chain(ticker: str, expiration: str) -> list[dict]:
    """Netzwerk isoliert, damit Tests dies monkeypatchen können."""
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/chains",
            params={"symbol": ticker, "expiration": expiration, "greeks": "true"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data    = resp.json()
        options = data.get("options", {}).get("option", []) or []
        if isinstance(options, dict):
            options = [options]
        return options
    except Exception as e:
        log.debug(f"market_snapshot._fetch_chain [{ticker} {expiration}] Fehler (ignoriert): {e}")
        return []


def select_contract(ticker: str, direction: str, dte_floor: int, spot: float) -> dict | None:
    """
    Wählt einen ECHTEN Options-Kontrakt (Tradier Chain) für die
    Ledger-Momentaufnahme:
        - Expiry:      die erste Expiration mit DTE >= dte_floor
        - option_type: "call" für BULLISH, "put" für BEARISH
        - Strike:      am nächsten am übergebenen Spot-Preis

    Gibt None zurück (nie einen Fehler), wenn Richtung unbekannt, kein
    TRADIER_API_KEY gesetzt, kein Spot-Preis, keine passende Expiry oder
    kein Kontrakt in der Chain gefunden wird.
    """
    try:
        if direction not in ("BULLISH", "BEARISH"):
            return None
        if not _use_tradier():
            return None
        if spot in (None, 0):
            return None
        try:
            dte_floor = int(dte_floor)
        except (TypeError, ValueError):
            dte_floor = 120

        option_type = "call" if direction == "BULLISH" else "put"

        expirations = _fetch_expirations(ticker)
        if not expirations:
            return None

        today = datetime.now(timezone.utc).date()
        chosen_exp, chosen_dte = None, None
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            except Exception:
                continue
            dte = (exp_date - today).days
            if dte >= dte_floor:
                chosen_exp, chosen_dte = exp, dte
                break
        if chosen_exp is None:
            return None

        chain      = _fetch_chain(ticker, chosen_exp)
        candidates = [o for o in chain if o.get("option_type") == option_type]
        if not candidates:
            return None

        def _dist(o):
            try:
                return abs(float(o.get("strike", 0)) - float(spot))
            except Exception:
                return float("inf")

        best   = min(candidates, key=_dist)
        greeks = best.get("greeks") or {}
        iv     = greeks.get("mid_iv") or greeks.get("smv_vol")
        iv     = float(iv) if isinstance(iv, (int, float)) and iv > 0 else None
        delta  = greeks.get("delta")
        delta  = float(delta) if isinstance(delta, (int, float)) else None

        bid = _f(best.get("bid"))
        ask = _f(best.get("ask"))
        mid = round((bid + ask) / 2, 4) if (bid is not None and ask is not None) else None

        return {
            "symbol":        best.get("symbol"),
            "strike":        float(best.get("strike", 0)),
            "expiry":        chosen_exp,
            "dte":           chosen_dte,
            "bid":           bid,
            "ask":           ask,
            "mid":           mid,
            "iv":            iv,
            "delta":         delta,
            "open_interest": int(best.get("open_interest") or 0),
            "quote_ts":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    except Exception as e:
        log.debug(f"market_snapshot.select_contract [{ticker}] Fehler (ignoriert): {e}")
        return None


def select_spread_short_leg(
    ticker: str, expiry: str | None, option_type: str, long_strike: float,
) -> dict | None:
    """
    P0-2 (candidate_ledger real_strategy): wählt den Short-Leg eines Spreads
    aus DERSELBEN Chain/Expiry wie der bereits gewählte Long-Leg — über
    genau dieselbe Auswahlregel wie die Produktion
    (options_designer.pick_spread_leg_strike: call [1.05, 1.20]×, put
    [0.80, 0.95]× long_strike). Gibt None zurück (nie einen Fehler), wenn kein
    TRADIER_API_KEY, keine Expiry/Chain oder kein Kontrakt im Fenster liegt.
    """
    try:
        if not expiry or option_type not in ("call", "put"):
            return None
        if not _use_tradier():
            return None
        chain = _fetch_chain(ticker, expiry)
        candidates = [o for o in chain if o.get("option_type") == option_type]
        if not candidates:
            return None

        from modules.options_designer import pick_spread_leg_strike

        strikes = [float(o.get("strike", 0)) for o in candidates]
        target_strike = pick_spread_leg_strike(strikes, float(long_strike), option_type)
        if target_strike is None:
            return None

        best = min(candidates, key=lambda o: abs(float(o.get("strike", 0)) - target_strike))
        bid = _f(best.get("bid"))
        ask = _f(best.get("ask"))
        mid = round((bid + ask) / 2, 4) if (bid is not None and ask is not None) else None
        return {
            "symbol": best.get("symbol"),
            "strike": float(best.get("strike", 0)),
            "bid":    bid,
            "ask":    ask,
            "mid":    mid,
        }
    except Exception as e:
        log.debug(f"market_snapshot.select_spread_short_leg [{ticker}] Fehler (ignoriert): {e}")
        return None


def fetch_term_iv_point(ticker: str, spot: float, min_dte: int = 7) -> tuple | None:
    """
    Review-Fix (replicated iv_rank): zweiter Term-Structure-Punkt für
    candidate_ledger.py's real_strategy — die ATM-IV der NÄCHSTEN Expiration
    mit DTE >= min_dte (typischerweise deutlich kürzer als die gewählte
    Long-Leg-Expiration, die schon einen eigenen Term-Punkt liefert). Genau
    EIN zusätzlicher Chain-Call pro Kandidat (getrennt vom Long-Leg-Call);
    der Aufrufer zählt Versuche selbst (term_calls-Zähler).

    Gibt (dte, atm_iv) zurück oder None (nie einen Fehler) — z.B. ohne
    TRADIER_API_KEY, ohne Expirations/Chain oder ohne ATM-Kontrakt mit IV.
    """
    try:
        if spot in (None, 0) or not _use_tradier():
            return None
        expirations = _fetch_expirations(ticker)
        if not expirations:
            return None

        today = datetime.now(timezone.utc).date()
        chosen_exp, chosen_dte = None, None
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            except Exception:
                continue
            dte = (exp_date - today).days
            if dte >= min_dte:
                chosen_exp, chosen_dte = exp, dte
                break
        if chosen_exp is None:
            return None

        chain = _fetch_chain(ticker, chosen_exp)
        calls = [o for o in chain if o.get("option_type") == "call"]
        if not calls:
            return None

        atm_ivs = []
        for o in calls:
            try:
                strike = float(o.get("strike", 0))
            except Exception:
                continue
            if not (spot * 0.93 <= strike <= spot * 1.07):
                continue
            greeks = o.get("greeks") or {}
            iv = greeks.get("mid_iv") or greeks.get("smv_vol")
            if iv and isinstance(iv, (int, float)) and iv > 0.05:
                atm_ivs.append(float(iv))
        if not atm_ivs:
            return None

        atm_ivs.sort()
        median_iv = atm_ivs[len(atm_ivs) // 2] if len(atm_ivs) % 2 else \
            (atm_ivs[len(atm_ivs) // 2 - 1] + atm_ivs[len(atm_ivs) // 2]) / 2
        return chosen_dte, float(median_iv)
    except Exception as e:
        log.debug(f"market_snapshot.fetch_term_iv_point [{ticker}] Fehler (ignoriert): {e}")
        return None


def fetch_option_quotes(symbols: list[str]) -> dict:
    """
    Gebündelter Tradier-Quote-Abruf für OCC-Options-Symbole (für
    real_opt_ret-Marks in candidate_ledger.update_outcomes). Ohne
    TRADIER_API_KEY oder bei Fehlern: leeres Dict.

    Returns: {symbol: {bid, ask, mid, ts}}
    """
    out = {}
    symbols = [s for s in dict.fromkeys(symbols) if s]
    if not symbols or not _use_tradier():
        return out
    for chunk in _chunk(symbols, QUOTE_CHUNK):
        try:
            for q in _fetch_quotes_raw(chunk):
                sym = q.get("symbol")
                if not sym:
                    continue
                bid = _f(q.get("bid"))
                ask = _f(q.get("ask"))
                mid = round((bid + ask) / 2, 4) if (bid is not None and ask is not None) else None
                ts  = (_epoch_ms_to_iso(q.get("bid_date"))
                       or _epoch_ms_to_iso(q.get("ask_date"))
                       or _epoch_ms_to_iso(q.get("trade_date")))
                out[sym] = {"bid": bid, "ask": ask, "mid": mid, "ts": ts}
        except Exception as e:
            log.debug(f"market_snapshot.fetch_option_quotes Fehler (ignoriert): {e}")
    return out
