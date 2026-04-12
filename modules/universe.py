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

log = logging.getLogger(__name__)

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
        url    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
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
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        all_tables = pd.read_html(url)
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


@lru_cache(maxsize=1)
def get_universe(universe: str = "") -> list[str]:
    """Gibt die konfigurierte Ticker-Liste zurück (ohne delistete Ticker)."""
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
