"""
modules/universe.py – Dynamisches Ticker-Universum

Fix: Veraltete/delistete Ticker aus _SP500_STATIC entfernt.
     Stand April 2026 — alle 8 Fehler-Ticker erklärt:

  ANSS → delisted: Synopsys-Akquisition (2025)
  CMA  → delisted: Mergers Umpqua/Columbia Banking  
  DAY  → delisted: Wurde zu WEX umbenannt
  FI   → delisted: Fiserv-Ticker; jetzt korrekt als FISV
  HES  → delisted: Chevron-Akquisition (2025)
  IPG  → delisted: Omnicom-Akquisition (2025) → 0.344 OMC
  K    → delisted: Mars-Akquisition (Kellanova, Dez 2025)
  WBA  → delisted: Private-Equity (Sycamore, Aug 2025)

Ersetzt durch aktuelle Aufnahmen:
  IBKR, ARES, APP, HOOD, EME, CVNA, FIX, CRH, XYZ (Block)
"""

from __future__ import annotations
import logging
from functools import lru_cache
from io import StringIO

import requests

log = logging.getLogger(__name__)

# Wikipedia blockiert Standard-Scraper ohne User-Agent
_WP_HEADERS = {
    "User-Agent": "AdaptiveAsymmetryScanner/9.0 (research@pcctrading.com; index-rebalancing-tracker)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Veraltete Ticker (delistet/akquiriert) ────────────────────────────────────
# Diese Ticker sollen NIE mehr im Universum auftauchen
_DELISTED = frozenset({
    "ANSS",  # → Synopsys-Akquisition
    "CMA",   # → Columbia Banking
    "DAY",   # → umbenannt
    "FI",    # → war falsch, Fiserv = FISV
    "HES",   # → Chevron-Akquisition
    "IPG",   # → Omnicom-Akquisition
    "K",     # → Mars-Akquisition (Kellanova)
    "WBA",   # → Sycamore Private Equity
    # Weitere bekannte Delistings 2025:
    "CZR",   # → aus S&P500 entfernt Sep 2025
    "ENPH",  # → aus S&P500 entfernt Sep 2025
    "MKTX",  # → aus S&P500 entfernt Sep 2025
})

# ── Statischer S&P 500 Fallback (April 2026) ──────────────────────────────────
_SP500_STATIC: list[str] = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","META","AMZN",
    "AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI",
    "AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ",
    "T","ATO","ADSK","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI",
    "BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BMY",
    "AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CPT","CPB","COF","CAH",
    "KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX",
    "CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS",
    "CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG",
    "COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI",
    "CSX","CMI","CVS","DHI","DHR","DRI","DVA","DECK","DE","DAL","DVN",
    "DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DUK","DD",
    "EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ETR",
    "EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES",
    "EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
    "FIS","FITB","FSLR","FE","FISV","FMC","F","FTNT","FTV","FOXA","FOX","BEN",
    "FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC",
    "GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY",
    "HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE",
    "IFF","IP","INTU","ISRG","IVZ","INVH","IQV","IRM","JPM","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LH","LRCX","LW",
    "LVS","LDOS","LEN","LII","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU",
    "LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC",
    "MCD","MCK","MDT","MRK","MET","MTD","MGM","MCHP","MU","MSFT","MAA",
    "MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI",
    "NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC",
    "NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL",
    "OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC",
    "PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QCOM","DGX","RL",
    "RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP",
    "ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG",
    "SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
    "SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
    "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV",
    "TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI",
    "UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI",
    "V","VST","VMC","WRB","GWW","WAB","WMT","DIS","WBD","WM","WAT",
    "WEC","WFC","WELL","WST","WDC","WY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
    # Neue Aufnahmen 2025/2026
    "IBKR","ARES","APP","HOOD","EME","CVNA","FIX","CRH","XYZ",
    "PLTR","VST","GEV","SOLV","VLTO",
]

_NASDAQ100_STATIC: list[str] = [
    "ADBE","ADP","ABNB","GOOGL","GOOG","AMZN","AMD","AEP","AMGN","ADI",
    "AAPL","AMAT","APP","ARM","ASML","TEAM","ADSK","AZN","BIIB","BKNG",
    "AVGO","CDNS","CDW","CHTR","CTAS","CSCO","CTSH","CMCSA","CEG","CPRT",
    "CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DLTR","EA","EXC","FAST",
    "FTNT","GEHC","GILD","GFS","HON","IDXX","INTC","INTU","ISRG","KDP",
    "KLAC","KHC","LRCX","LIN","MELI","META","MCHP","MU","MSFT","MRNA","MDLZ",
    "MDB","MNST","NFLX","NVDA","NXPI","ORLY","ON","ODFL","PCAR","PANW","PAYX",
    "PYPL","PDD","PEP","QCOM","REGN","ROST","SBUX","SNPS","TTWO","TMUS","TSLA",
    "TXN","TTD","VRSK","VRTX","WDAY","ZS","PLTR","ARM",
]


def _fetch_sp500() -> list[str]:
    try:
        import pandas as pd
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=_WP_HEADERS, timeout=12)
        resp.raise_for_status()
        tables  = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
        tickers = (
            tables[0]["Symbol"]
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        log.info(f"S&P 500: {len(tickers)} Ticker von Wikipedia geladen.")
        return tickers
    except Exception as e:
        log.warning(f"S&P 500 Wikipedia-Fehler: {e} → statischer Fallback.")
        return []


def _fetch_nasdaq100() -> list[str]:
    try:
        import pandas as pd
        url  = "https://en.wikipedia.org/wiki/Nasdaq-100"
        resp = requests.get(url, headers=_WP_HEADERS, timeout=12)
        resp.raise_for_status()
        all_tables = pd.read_html(StringIO(resp.text))
        for table in all_tables:
            if "Ticker" in table.columns:
                tickers = (
                    table["Ticker"]
                    .str.replace(".", "-", regex=False)
                    .str.strip()
                    .tolist()
                )
                log.info(f"Nasdaq 100: {len(tickers)} Ticker von Wikipedia geladen.")
                return tickers
        log.warning("Nasdaq-100: Keine Tabelle gefunden → statischer Fallback.")
        return []
    except Exception as e:
        log.warning(f"Nasdaq-100 Wikipedia-Fehler: {e} → statischer Fallback.")
        return []


def _clean(tickers: list[str]) -> list[str]:
    seen:   set[str] = set()
    result: list[str] = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip().upper()
        if not t or len(t) > 6:
            continue
        if not all(c.isalpha() or c == "-" for c in t):
            continue
        # FIX: Delistete Ticker herausfiltern
        if t in _DELISTED:
            continue
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


@lru_cache(maxsize=8)
def get_universe(universe: str = "", as_of: str | None = None) -> list[str]:
    """Gibt die konfigurierte Ticker-Liste zurück (ohne delistete Ticker).

    as_of (ISO-Datum): Point-in-Time-Mitgliedschaft des S&P 500 zu diesem
    Stichtag statt der heutigen Liste (Audit F01, Survivorship). Für
    historische Forschung IMMER as_of bzw. research_universe() verwenden; die
    heutige Liste ist nur für Live-Signale korrekt. Nasdaq-100 hat keine
    Änderungshistorie und ist mit as_of nicht enthalten."""
    if as_of is not None:
        hist = sp500_history()
        return sorted(t for t, iv in hist["intervals"].items() if is_member(iv, as_of))
    if not universe:
        try:
            from modules.config import cfg
            universe = cfg.filters.universe
        except Exception:
            universe = "sp500_nasdaq100"

    log.info(f"Lade Ticker-Universum: '{universe}'")

    sp500:  list[str] = []
    ndq100: list[str] = []

    if universe in ("sp500", "sp500_nasdaq100"):
        sp500 = _fetch_sp500()
        if not sp500:
            sp500 = list(_SP500_STATIC)

    if universe in ("nasdaq100", "sp500_nasdaq100"):
        ndq100 = _fetch_nasdaq100()
        if not ndq100:
            ndq100 = list(_NASDAQ100_STATIC)

    combined = _clean(sp500 + ndq100)

    # Nochmal explizit delistete rausfiltern (falls Wikipedia noch veraltet)
    removed = [t for t in (sp500 + ndq100) if t in _DELISTED]
    if removed:
        log.info(f"Delistete Ticker entfernt: {removed}")

    log.info(
        f"Universum '{universe}': {len(combined)} Ticker "
        f"(S&P500={len(sp500)}, Nasdaq100={len(ndq100)}, "
        f"nach Deduplizierung+Delisting-Filter={len(combined)})"
    )
    return combined


# ── Point-in-Time-Mitgliedschaft (Audit F01 / Remediation P0-2) ─────────────
# Quelle: Wikipedia "List of S&P 500 companies", Tabelle "Selected changes"
# (dieselbe Seite wie die heutige Liste; Lizenz CC BY-SA). Rekonstruktion:
# ausgehend von der HEUTIGEN Liste werden die Änderungen rückwärts
# angewendet (vor dem Stichtag einer Aufnahme war der Titel kein Mitglied,
# vor einer Entfernung war er es). Grenzen (dokumentiert, nicht versteckt):
#   * die Tabelle ist für frühe Jahre unvollständig ("selected");
#   * für entfernte Titel ohne Kursdaten bei Yahoo (Insolvenz/Übernahme)
#     bleibt der Survivorship-Bias bestehen -> Abdeckung wird gemessen.

def _flat_col(c) -> str:
    if isinstance(c, tuple):
        parts = [str(x) for x in c if str(x) and not str(x).startswith("Unnamed")]
        uniq = []
        for x in parts:
            if x not in uniq:
                uniq.append(x)
        return " ".join(uniq).strip().lower()
    return str(c).strip().lower()


def parse_sp500_changes(table) -> list[dict]:
    """Wikipedia-Änderungstabelle -> [{date, added, removed}] (Ticker in
    Yahoo-Schreibweise, leere Felder = None). Robust gegen MultiIndex-Header
    ("Added"/"Ticker") und flache Header ("Added Ticker")."""
    import pandas as pd
    df = table.copy()
    df.columns = [_flat_col(c) for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    add_col = next((c for c in df.columns if "added" in c and ("ticker" in c or "symbol" in c)), None)
    rem_col = next((c for c in df.columns if "removed" in c and ("ticker" in c or "symbol" in c)), None)
    if not (date_col and add_col and rem_col):
        raise ValueError(f"Änderungstabelle unbekannt: {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        d = pd.to_datetime(str(r[date_col]).split("[")[0].strip(), errors="coerce")
        if pd.isna(d):
            continue

        def tk(v):
            if v is None or (isinstance(v, float) and v != v):
                return None
            v = str(v).strip().upper().replace(".", "-")
            return v if v and v != "NAN" else None
        out.append({"date": d.date().isoformat(), "added": tk(r[add_col]), "removed": tk(r[rem_col])})
    return sorted(out, key=lambda x: x["date"])


def membership_intervals(current: list[str], changes: list[dict]) -> dict[str, list[list]]:
    """{ticker: [[start, end], ...]} mit ISO-Daten; start None = vor Beginn
    der Aufzeichnung, end None = bis heute. Rückwärts von der heutigen Liste."""
    ends: dict[str, str | None] = {t: None for t in current}      # aktuell "offene" Mitglieder -> Endedatum
    out: dict[str, list[list]] = {}
    for ch in sorted(changes, key=lambda x: x["date"], reverse=True):
        d = ch["date"]
        a, r = ch.get("added"), ch.get("removed")
        if a and a in ends:                        # vor d war a kein Mitglied: Intervall schließen
            out.setdefault(a, []).append([d, ends.pop(a)])
        if r and r not in ends:                    # vor d war r Mitglied: Intervall öffnen, endet am Tag d
            ends[r] = d
    for t, e in ends.items():                      # seit Beginn der Aufzeichnung Mitglied
        out.setdefault(t, []).append([None, e])
    for t in out:
        out[t].sort(key=lambda iv: iv[0] or "")
    return out


def is_member(intervals: list[list], as_of) -> bool:
    d = str(getattr(as_of, "date", lambda: as_of)()) if hasattr(as_of, "date") else str(as_of)
    d = d[:10]
    return any((s is None or s <= d) and (e is None or d < e) for s, e in intervals)


_WP_API = "https://en.wikipedia.org/w/api.php"
_WP_MAIN = "List_of_S&P_500_companies"
# Mögliche Auslagerungsseiten der Änderungshistorie (CI 2026-10-02: Hauptseite
# lieferte nur Mitglieder-Tabelle + Navbox). Zusätzlich werden verlinkte Seiten
# mit S&P 500 + change/histor/former im Titel geprüft.
_WP_CANDIDATES = ("Historical components of the S&P 500", "List of former S&P 500 companies",
                  "Changes to the S&P 500 index", "List of S&P 500 index changes")


def _wp_tables(page: str):
    """Tabellen einer Wikipedia-Seite über die offizielle MediaWiki-API (action=parse)."""
    import pandas as pd
    r = requests.get(_WP_API, params={"action": "parse", "page": page, "prop": "text", "format": "json",
                                      "formatversion": 2, "redirects": 1}, headers=_WP_HEADERS, timeout=20)
    r.raise_for_status()
    js = r.json()
    if "error" in js:
        raise ValueError(f"{page}: {js['error'].get('info')}")
    html = js["parse"]["text"]
    return pd.read_html(StringIO(html), flavor="lxml") if "<table" in html else []


def _wp_candidate_pages() -> tuple[list[str], dict]:
    r = requests.get(_WP_API, params={"action": "parse", "page": _WP_MAIN, "prop": "sections|links",
                                      "format": "json", "formatversion": 2, "redirects": 1},
                     headers=_WP_HEADERS, timeout=20)
    r.raise_for_status()
    p = r.json().get("parse", {})
    sections = [x.get("line") for x in p.get("sections", [])]
    links = [x.get("title", "") for x in p.get("links", []) if x.get("ns") == 0]
    linked = [t for t in links if "S&P 500" in t and any(k in t.lower() for k in ("change", "histor", "former"))]
    return list(dict.fromkeys(linked + list(_WP_CANDIDATES))), {"sections": sections, "linked_candidates": linked}


def _fetch_sp500_changes() -> list[dict]:
    """Hauptseite (HTML), dann MediaWiki-API: Hauptseite und Auslagerungsseiten.
    Bestes Ergebnis = meiste parsebare Änderungen; Fehler enthält Diagnose."""
    import pandas as pd
    diag, best = {}, []

    def attempt(name, get_tables):
        nonlocal best
        try:
            ch = select_changes_table(get_tables())
        except Exception as e:  # noqa: BLE001 – Diagnose sammeln, nächste Quelle versuchen
            diag[name] = f"{type(e).__name__}: {str(e)[:300]}"
            return
        diag[name] = f"{len(ch)} Änderungen"
        if len(ch) > len(best):
            best = ch

    def main_html():
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=_WP_HEADERS,
                            timeout=20)
        resp.raise_for_status()
        diag["main_html_bytes"] = len(resp.text)
        return pd.read_html(StringIO(resp.text), flavor="lxml")

    attempt("main_html", main_html)
    if not best:
        attempt("api:" + _WP_MAIN, lambda: _wp_tables(_WP_MAIN))
    if not best:
        try:
            pages, info = _wp_candidate_pages()
            diag.update(info)
        except Exception as e:  # noqa: BLE001
            pages = list(_WP_CANDIDATES)
            diag["candidates_error"] = f"{type(e).__name__}: {e}"
        for pg in pages[:8]:
            attempt("api:" + pg, lambda pg=pg: _wp_tables(pg))
            if best:
                break
    if not best:
        raise ValueError(f"keine S&P-500-Änderungstabelle gefunden; Diagnose: {diag}")
    log.info(f"S&P-500-Änderungen: {diag}")
    return best


def _header_from_rows(table, n_rows: int):
    """Kopf aus Spaltennamen + den ersten n_rows Datenzeilen (pandas liest
    Wikipedia-Köpfe aus <td>/<th> in tbody teils als Daten)."""
    import pandas as pd
    df = table.copy()
    heads = []
    for j, c in enumerate(df.columns):
        parts = [str(x) for x in (c if isinstance(c, tuple) else (c,))]
        parts += [str(df.iloc[i, j]) for i in range(min(n_rows, len(df)))]
        heads.append(tuple(p for p in parts if p and p.lower() != "nan" and not p.isdigit()))
    out = df.iloc[n_rows:].reset_index(drop=True)
    out.columns = pd.MultiIndex.from_tuples([h if h else (f"col{j}",) for j, h in enumerate(heads)]) \
        if any(len(h) > 1 for h in heads) else [" ".join(h) for h in heads]
    return out


def select_changes_table(tables) -> list[dict]:
    """Die Änderungstabelle über ihre Spalten finden (Datum + Added/Removed-Ticker),
    nicht über eine HTML-id (CI 2026-10-02: id-Suche schlug fehl -> html5lib-Fallback
    -> ImportError; danach fanden sich keine passenden Spaltenköpfe, weil pandas
    den Kopf je nach Markup als Datenzeilen liest). Versucht je Tabelle den
    Original-Kopf und Köpfe aus den ersten 1–2 Zeilen; nimmt die Tabelle mit den
    meisten parsebaren Änderungen."""
    best: list[dict] = []
    seen_cols = []
    for t in tables:
        seen_cols.append([_flat_col(c) for c in t.columns][:8])
        for variant in (t, _header_from_rows(t, 1), _header_from_rows(t, 2)):
            try:
                ch = parse_sp500_changes(variant)
            except (ValueError, KeyError, IndexError):
                continue
            if len(ch) > len(best):
                best = ch
    if not best:
        raise ValueError(f"keine S&P-500-Änderungstabelle auf der Seite gefunden; Spaltenköpfe: {seen_cols[:6]}")
    return best


@lru_cache(maxsize=1)
def sp500_history() -> dict:
    """{'intervals': {ticker: [[start,end],...]}, 'n_changes', 'first_change', 'source'}.
    Ohne Änderungshistorie (Netz/Format) -> Fehler statt stiller Rückfall auf
    die heutige Liste (das wäre genau der Survivorship-Bias)."""
    current = _fetch_sp500()
    if not current:
        raise RuntimeError("S&P-500-Liste nicht verfügbar – PIT-Universum nicht rekonstruierbar")
    changes = _fetch_sp500_changes()
    if not changes:
        raise RuntimeError("S&P-500-Änderungshistorie leer – PIT-Universum nicht rekonstruierbar")
    current = [t.strip().upper().replace(".", "-") for t in current]
    return {"intervals": membership_intervals(current, changes), "n_changes": len(changes),
            "first_change": changes[0]["date"], "source": "wikipedia:List_of_S&P_500_companies#changes"}


def research_universe(start: str) -> dict:
    """Alle Titel, die seit `start` IRGENDWANN Mitglied waren (inkl. später
    entfernter), plus Mitgliedschaftsintervalle für die Stichtagsmaske."""
    hist = sp500_history()
    iv = hist["intervals"]
    tickers = sorted(t for t, ivs in iv.items() if any(e is None or e > start for _, e in ivs))
    removed = sorted(t for t in tickers if all(e is not None for _, e in iv[t]))
    return {**hist, "tickers": tickers, "removed_since_start": removed, "start": start}
