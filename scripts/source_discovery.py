"""
scripts/source_discovery.py – sucht in OFFIZIELLEN Datenkatalogen nach
maschinenlesbaren Quellen für zurückgestellte Registry-Einträge (Belgien/
Viapass, Österreich/ASFINAG, FAF5-Spiegel auf data.bts.gov).

Läuft im Preflight (GitHub Actions) und schreibt
outputs/external_data/health/source_discovery.json. Nur Katalog-Such-APIs
(data.europa.eu, data.gv.at CKAN, Socrata-Discovery), kein Scraping.
Keine Entscheidung im Code: Aktivierung erfolgt per PR nach Sichtung.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

UA = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (source-discovery; research)"}
OUT = Path("outputs/external_data/health/source_discovery.json")

EU_PORTAL = "https://data.europa.eu/api/hub/search/search"
AT_CKAN = "https://www.data.gv.at/katalog/api/3/action/package_search"
SOCRATA = "https://api.us.socrata.com/api/catalog/v1"

QUERIES = {
    "asfinag_at": [("eu", "ASFINAG Lkw"), ("eu", "ASFINAG Fahrleistung"), ("at", "ASFINAG"),
                   ("at", "Lkw Maut Fahrleistung")],
    "viapass_be": [("eu", "Viapass"), ("eu", "kilometerheffing vrachtwagens"),
                   ("eu", "prélèvement kilométrique poids lourds"), ("eu", "Belgium heavy goods vehicles kilometre charge")],
    "fhwa_faf": [("bts", "Freight Analysis Framework"), ("eu", "Freight Analysis Framework FAF5")],
}


def _title(v):
    if isinstance(v, dict):
        return v.get("en") or v.get("de") or v.get("nl") or v.get("fr") or next(iter(v.values()), "")
    return v or ""


def search(kind: str, q: str) -> list[dict]:
    try:
        if kind == "eu":
            r = requests.get(EU_PORTAL, params={"q": q, "filter": "dataset", "limit": 15},
                             headers=UA, timeout=30)
            res = r.json().get("result", {}).get("results", [])
            return [{"title": _title(x.get("title")), "id": x.get("id"),
                     "publisher": _title((x.get("publisher") or {}).get("name")),
                     "catalog": (x.get("catalog") or {}).get("id"),
                     "license": [((d.get("license") or {}).get("id")) for d in (x.get("distributions") or [])][:3],
                     "formats": sorted({str((d.get("format") or {}).get("id")) for d in (x.get("distributions") or [])})[:6],
                     "access_urls": [(d.get("access_url") or [None])[0] if isinstance(d.get("access_url"), list)
                                     else d.get("access_url") for d in (x.get("distributions") or [])][:3]}
                    for x in res]
        if kind == "at":
            r = requests.get(AT_CKAN, params={"q": q, "rows": 15}, headers=UA, timeout=30)
            res = r.json().get("result", {}).get("results", [])
            return [{"title": x.get("title"), "name": x.get("name"),
                     "organization": (x.get("organization") or {}).get("title"),
                     "license": x.get("license_id"),
                     "resources": [{"format": rr.get("format"), "url": rr.get("url")}
                                   for rr in (x.get("resources") or [])][:5]}
                    for x in res]
        if kind == "bts":
            r = requests.get(SOCRATA, params={"domains": "data.bts.gov", "q": q, "limit": 15},
                             headers=UA, timeout=30)
            res = r.json().get("results", [])
            return [{"name": (x.get("resource") or {}).get("name"), "id": (x.get("resource") or {}).get("id"),
                     "type": (x.get("resource") or {}).get("type"),
                     "updatedAt": (x.get("resource") or {}).get("updatedAt"),
                     "permalink": x.get("permalink")} for x in res]
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]
    return []


PORTWATCH_OWNER = "IMF-portwatch_imf_dataviz"
ARCGIS_SEARCH = "https://www.arcgis.com/sharing/rest/search"


def portwatch_catalog() -> list[dict]:
    """Alle öffentlichen ArcGIS-Items des offiziellen PortWatch-Besitzers
    (Titel, Typ, URL, Änderungsdatum, Kurzbeschreibung) -- Grundlage für die
    Auswahl weiterer PortWatch-Datensätze (keine Aktivierung)."""
    out, start = [], 1
    try:
        while start and start > 0 and len(out) < 300:
            r = requests.get(ARCGIS_SEARCH, params={"q": f'owner:"{PORTWATCH_OWNER}"', "f": "json",
                                                    "num": 100, "start": start}, headers=UA, timeout=30)
            d = r.json()
            for it in d.get("results", []):
                out.append({k: it.get(k) for k in ("id", "title", "type", "url", "modified", "snippet", "tags")})
            start = d.get("nextStart", -1)
    except Exception as e:  # noqa: BLE001
        out.append({"error": repr(e)})
    return out


PORTWATCH_MAP_KEYWORDS = ("disruption", "impact", "trade risks", "nowcast", "economy",
                          "country", "industr", "spillover", "trade network")


def portwatch_map_layers(catalog: list[dict]) -> list[dict]:
    """Daten-Layer hinter relevanten PortWatch-Web-Karten/Dashboards/Experiences
    (operationalLayers bzw. dataSources) -- welche maschinenlesbaren Dienste
    stecken hinter Disruption Monitor, Trade Nowcast, Impact Maps usw.?"""
    out = []
    for it in catalog:
        title = str(it.get("title") or "").lower()
        if it.get("type") not in ("Web Map", "Dashboard", "Web Experience") or \
                not any(k in title for k in PORTWATCH_MAP_KEYWORDS) or not it.get("id"):
            continue
        try:
            r = requests.get(f"https://www.arcgis.com/sharing/rest/content/items/{it['id']}/data",
                             params={"f": "json"}, headers=UA, timeout=30)
            txt = r.text
            urls = sorted({u for u in __import__("re").findall(
                r"https://[^\"\s]+/(?:FeatureServer|MapServer|ImageServer)(?:/\d+)?", txt)})
            out.append({"title": it.get("title"), "type": it.get("type"), "id": it["id"], "layers": urls[:20]})
        except Exception as e:  # noqa: BLE001
            out.append({"title": it.get("title"), "error": repr(e)})
    return out


IMF_DATAFLOW_URLS = [
    "https://api.imf.org/external/sdmx/2.1/dataflow",
    "https://api.imf.org/external/sdmx/3.0/structure/dataflow",
]


def imf_dataflows() -> dict:
    """Liste der Datenflüsse der offiziellen IMF-Daten-API (SDMX): ID + Name.
    Grundlage für die Frage, welche IMF-Statistiken maschinenlesbar sind."""
    import re as _re
    res = {}
    for url in IMF_DATAFLOW_URLS:
        try:
            r = requests.get(url, headers={**UA, "Accept": "application/xml"}, timeout=60)
            txt = r.text
            flows = _re.findall(r'<(?:str|structure):Dataflow[^>]*\bid="([^"]+)".*?<(?:com|common):Name[^>]*>([^<]+)<',
                                txt, flags=_re.S)
            res[url] = {"http_status": r.status_code, "n": len(flows),
                        "dataflows": [{"id": i, "name": n.strip()} for i, n in flows[:400]]}
            if flows:
                break
        except Exception as e:  # noqa: BLE001
            res[url] = {"error": repr(e)}
    return res


PW = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
SCHEMA_LAYERS = {
    "disruptions_database": f"{PW}/portwatch_disruptions_database/FeatureServer/0",
    "disruptions_with_ports": f"{PW}/disruptions_with_ports/FeatureServer/0",
    "countries_database": f"{PW}/PortWatch_countries_database/FeatureServer/0",
    "ports_database": f"{PW}/PortWatch_ports_database/FeatureServer/0",
}
IMF_PROBES = {
    "ECFIE_structure": "https://api.imf.org/external/sdmx/2.1/dataflow/all/ECFIE/latest?references=datastructure",
    "ECFIE_data_json": "https://api.imf.org/external/sdmx/3.0/data/dataflow/all/ECFIE/+/*?lastNObservations=2",
    "ECFIE_data_21": "https://api.imf.org/external/sdmx/2.1/data/ECFIE/all?lastNObservations=2",
    "PI_structure": "https://api.imf.org/external/sdmx/2.1/dataflow/all/PI/latest?references=datastructure",
    "PI_data_21": "https://api.imf.org/external/sdmx/2.1/data/PI/JPN+KOR+CHN+TWN.*.*?lastNObservations=2",
}


def layer_schemas() -> dict:
    """Felder, Datensatzzahl und 3 Beispielzeilen je Layer (Schema vor dem
    Konnektor-Bau prüfen, nichts raten)."""
    out = {}
    for name, url in SCHEMA_LAYERS.items():
        try:
            meta = requests.get(url, params={"f": "json"}, headers=UA, timeout=30).json()
            cnt = requests.get(f"{url}/query", params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
                               headers=UA, timeout=30).json()
            sample = requests.get(f"{url}/query", params={"where": "1=1", "outFields": "*", "resultRecordCount": 3,
                                                          "f": "json"}, headers=UA, timeout=30).json()
            out[name] = {"url": url, "name": meta.get("name"), "maxRecordCount": meta.get("maxRecordCount"),
                         "count": cnt.get("count"), "editing": meta.get("editingInfo"),
                         "fields": [{"name": f.get("name"), "type": f.get("type"), "alias": f.get("alias")}
                                    for f in meta.get("fields") or []],
                         "sample": [ft.get("attributes") for ft in (sample.get("features") or [])][:3],
                         "error": meta.get("error") or sample.get("error")}
        except Exception as e:  # noqa: BLE001
            out[name] = {"url": url, "error": repr(e)}
    return out


def imf_series_sample(url: str, n: int = 12) -> dict:
    """Erste Series-Elemente (alle Attribute) + deren erste/letzte Obs."""
    import re as _re
    try:
        r = requests.get(url, headers={**UA, "Accept": "application/xml"}, timeout=90)
        txt = r.text
        ds = _re.search(r"<message:DataSet[^>]*>", txt)
        series, index = [], []
        for m in _re.finditer(r"<Series ([^>]*)>(.*?)</Series>", txt, flags=_re.S):
            attrs = dict(_re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            obs = _re.findall(r"<Obs ([^>]*)/>", m.group(2))
            last = dict(_re.findall(r'(\w+)="([^"]*)"', obs[-1])) if obs else {}
            index.append("|".join(str(attrs.get(k, "-")) for k in
                                  ("COUNTRY", "PRODUCTION_INDEX", "FREQUENCY", "TYPE_OF_TRANSFORMATION"))
                         + f"|last={last.get('TIME_PERIOD')}")
            if len(series) >= n:
                continue
            series.append({"attrs": attrs, "n_obs": len(obs),
                           "first_obs": dict(_re.findall(r'(\w+)="([^"]*)"', obs[0])) if obs else None,
                           "last_obs": dict(_re.findall(r'(\w+)="([^"]*)"', obs[-1])) if obs else None})
        return {"status": r.status_code, "dataset_attrs": dict(_re.findall(r'(\w+)="([^"]*)"', ds.group(0))) if ds else None,
                "n_series_total": txt.count("<Series "), "series_index": index, "series": series}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def imf_probes() -> dict:
    out = {}
    for name, url in IMF_PROBES.items():
        for accept in ("application/vnd.sdmx.data+json;version=1.0.0", "application/xml"):
            try:
                r = requests.get(url, headers={**UA, "Accept": accept}, timeout=60)
                out[f"{name}|{accept.split(';')[0]}"] = {"url": url, "status": r.status_code,
                                                         "content_type": r.headers.get("Content-Type"),
                                                         "head": r.text[:3000]}
            except Exception as e:  # noqa: BLE001
                out[f"{name}|{accept.split(';')[0]}"] = {"url": url, "error": repr(e)}
    return out


def sec_form4_canary(tickers=("AAPL", "MMM", "ARE", "JPM", "BE")) -> dict:
    """Live-Canary für modules.alpha_sources (Form-4-Parsing, offizielle SEC-API):
    CIK-Auflösung, Kauf/Verkauf-Zählung 30 Tage, Beispieltransaktion."""
    import os
    import sys as _sys
    _sys.path.insert(0, os.getcwd())
    out = {}
    try:
        from modules import alpha_sources as al
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    for t in tickers:
        try:
            trades = al.fetch_sec_insider_trades(t, days_back=30)
            cl = al.detect_insider_cluster(t, days_back=30)
            out[t] = {"cik": al.sec_cik_for_ticker(t), "n_trades": len(trades),
                      "buys": cl["buy_count"], "sells": cl["sell_count"],
                      "cluster": cl["cluster_detected"], "sample": trades[:2]}
        except Exception as e:  # noqa: BLE001
            out[t] = {"error": repr(e)}
    return out


def main() -> int:
    out = {"retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "note": "Katalogtreffer, keine Entscheidung. Aktivierung nur per PR nach Sichtung.",
           "results": {sid: {f"{k}:{q}": search(k, q) for k, q in qs} for sid, qs in QUERIES.items()},
           "portwatch_catalog": portwatch_catalog()}
    out["portwatch_map_layers"] = portwatch_map_layers(out["portwatch_catalog"])
    out["imf_dataflows"] = imf_dataflows()
    out["layer_schemas"] = layer_schemas()
    out["imf_probes"] = imf_probes()
    out["imf_pi_series"] = imf_series_sample(
        "https://api.imf.org/external/sdmx/2.1/data/PI/CHN+KOR+TWN+JPN+IND+DEU+MEX+VNM.*.*?startPeriod=2025-01")
    out["sec_form4_canary"] = sec_form4_canary()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
