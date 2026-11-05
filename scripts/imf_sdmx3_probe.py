"""TEMP read-only discovery probe for IMF SDMX 3.0 (removed before merge)."""
import json, re, sys, xml.etree.ElementTree as ET
import requests

ROOT = "https://api.imf.org/external/sdmx/3.0"
UA = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (research; official-data-only)"}
XS = "application/vnd.sdmx.structure+xml;version=3.0.0"
JD = "application/vnd.sdmx.data+json;version=2.0.0"


def loc(e):
    return e.tag.rsplit("}", 1)[-1]


def get(url, params=None, accept=JD, cap=12_000_000, timeout=90):
    try:
        with requests.get(url if url.startswith("http") else ROOT + url, params=params,
                          headers={**UA, "Accept": accept}, timeout=(10, timeout), stream=True) as r:
            buf, n = [], 0
            for ch in r.iter_content(65536):
                n += len(ch)
                if n > cap:
                    return r.status_code, r.headers.get("Content-Type", ""), None, "TRUNC"
                buf.append(ch)
            return r.status_code, r.headers.get("Content-Type", ""), b"".join(buf), r.url
    except requests.RequestException as e:
        return 0, "", None, type(e).__name__


def p(*a):
    print(*a, flush=True)


def ename(e):
    for c in e:
        if loc(c) == "Name":
            return (c.text or "").strip()
    return ""


FLOWS = {
    "FSIC": ("IMF.STA", ["SECTOR", "INDICATOR", "FREQUENCY"], r"CFSI|NPL|AQ1|FSI688"),
    "FAS": ("IMF.STA", ["INDICATOR", "SECTOR", "COUNTERPARTY_SECTOR", "TYPE_OF_TRANSFORMATION"], r""),
    "CCI": ("IMF.STA", ["INDICATOR", "GEOGRAPHIC_MARINE_ZONES_AND_CLIMATIC_CLASSIFICATIONS"], r""),
    "MFS_MA": ("IMF.STA", ["INDICATOR", "UNIT"], r""),
    "GDD": ("IMF.FAD", ["INDICATOR"], r""),
    "FM": ("IMF.FAD", ["INDICATOR"], r""),
    "WEO": ("IMF.RES", ["INDICATOR"], r"^(NGDP_RPCH|PCPIPCH|LUR|GGXWDG_NGDP|BCA_NGDPD|GGXCNL_NGDP)$"),
}


def codelists(agency, fid, dims_wanted, used):
    st, ct, body, u = get(f"/structure/dataflow/{agency}/{fid}/+", {"references": "all"}, accept=XS)
    p("STRUCT", fid, st, len(body or b""), u)
    if st != 200 or not body:
        return
    root = ET.fromstring(body)
    cls, concepts = {}, {}
    for e in root.iter():
        if loc(e) == "Codelist":
            cls[e.get("id")] = {c.get("id"): ename(c) for c in e if loc(c) == "Code"}
        if loc(e) == "Concept":
            enum = next((x.text for x in e.iter() if loc(x) == "Enumeration"), None)
            concepts[e.get("id")] = enum
    for d in root.iter():
        if loc(d) != "Dimension" or d.get("id") not in dims_wanted:
            continue
        enum = next((x.text for x in d.iter() if loc(x) == "Enumeration"), None)
        if not enum:
            ci = next((x.text for x in d.iter() if loc(x) == "ConceptIdentity"), "") or ""
            enum = concepts.get(ci.rsplit(").", 1)[-1]) if ci else None
        clid = re.search(r"Codelist=[^:]+:([^()]+)\(", enum or "")
        clid = clid.group(1) if clid else None
        codes = cls.get(clid, {})
        p("DIM", fid, d.get("id"), "codelist", clid, "n", len(codes))
        sel = used.get(d.get("id"))
        for c, n in codes.items():
            if sel is not None and c not in sel:
                continue
            p("   ", c, "|", n[:110])


def used_codes(agency, fid, key):
    st, ct, body, u = get(f"/data/dataflow/{agency}/{fid}/+/{key}", {"c[TIME_PERIOD]": "ge:2022"})
    out = {}
    if st == 200 and body:
        d = json.loads(body)["data"]
        for x in d["structures"][0]["dimensions"]["series"]:
            out[x["id"]] = {v["id"] for v in x.get("values") or []}
    p("USED", fid, st, {k: len(v) for k, v in out.items()})
    return out


def data(label, path, params=None, accept=JD):
    st, ct, body, u = get(path, params, accept=accept, cap=3_000_000)
    info = {"label": label, "http": st, "bytes": len(body or b""), "url": str(u)[:200]}
    if body and body[:1] == b"{":
        j = json.loads(body)
        if st >= 400:
            info["err"] = json.dumps(j)[:250]
        else:
            d = j.get("data") or {}
            ds = (d.get("dataSets") or [{}])[0]
            ser = ds.get("series") or {}
            info["n_series"] = len(ser)
            dims = (d.get("structures") or [{}])[0].get("dimensions") or {}
            info["series"] = {k: len(v.get("observations") or {}) for k, v in list(ser.items())[:8]}
            info["sdims"] = [(x["id"], [v["id"] for v in x.get("values") or []][:8]) for x in dims.get("series") or []]
            info["times"] = [v.get("id") or v.get("value") for v in (dims.get("observation") or [{}])[0].get("values") or []][-8:]
    elif body:
        info["head"] = body[:200].decode("utf-8", "replace")
    p("DATA", json.dumps(info, default=str)[:1400])


def main():
    for c in ("DEU", "USA", "JPN", "CHN", "GBR", "FRA"):
        data(f"weo-{c}", f"/data/dataflow/IMF.RES/WEO/+/{c}.NGDP_RPCH.A", {"startPeriod": "2021", "endPeriod": "2031"})
    data("weo-deu-ctime", "/data/dataflow/IMF.RES/WEO/+/DEU.NGDP_RPCH.A", {"c[TIME_PERIOD]": "ge:2021+le:2031"})
    data("weo-multi", "/data/dataflow/IMF.RES/WEO/+/USA+DEU+JPN.NGDP_RPCH+PCPIPCH+LUR.A", {"c[TIME_PERIOD]": "ge:2021"})
    data("weo-vintage", "/data/dataflow/IMF.RES/WEO_2025_OCT_VINTAGE/+/USA+DEU.NGDP_RPCH.A", {"c[TIME_PERIOD]": "ge:2023"})
    data("fm-multi", "/data/dataflow/IMF.FAD/FM/+/USA+DEU+JPN.GNLB_S13_POGDP_PT+GPB_S13_POGDP_PT+G63G_S13_POGDP_PT.A", {"c[TIME_PERIOD]": "ge:2021"})
    data("gdd-multi", "/data/dataflow/IMF.FAD/GDD/+/USA+DEU+JPN.FL_PS_POGDP_PT+FL_S14_POGDP_PT+FL_S11_POGDP_PT.A", {"c[TIME_PERIOD]": "ge:2015"})
    data("fsic-usa-q", "/data/dataflow/IMF.STA/FSIC/+/USA+DEU+JPN.*.AQ12_CFSI_PT+FSI688_CFSI_PT.Q", {"c[TIME_PERIOD]": "ge:2023"})
    data("mfs-ma", "/data/dataflow/IMF.STA/MFS_MA/+/USA+DEU+JPN.BM_MAI.*.M", {"c[TIME_PERIOD]": "ge:2025-01"})
    data("mfs-ma-deu", "/data/dataflow/IMF.STA/MFS_MA/+/DEU.*.*.*", {"c[TIME_PERIOD]": "ge:2025"})
    data("cci-deu", "/data/dataflow/IMF.STA/CCI/+/DEU+USA.AMGST_dC.*.A", {"c[TIME_PERIOD]": "ge:2015"})
    data("fas-deu", "/data/dataflow/IMF.STA/FAS/+/DEU.*.*.*.*.*.*", {"c[TIME_PERIOD]": "ge:2020"})
    # AIPI routes (expected blocked or absent; recorded, not bypassed)
    for u in ("https://www.imf.org/external/datamapper/api/v1/AI_PI",
              "https://www.imf.org/external/datamapper/api/v2/AI_PI",
              "https://api.imf.org/external/datamapper/api/v1/AI_PI",
              "https://www.imf.org/external/datamapper/AI_PI@AIPI/ADVEC/EME/LIC"):
        st, ct, body, uu = get(u, accept="application/json", cap=2_000_000)
        p("AIPI", u, st, ct[:50], len(body or b""), (body or b"")[:120])
    for fid, (ag, dims, filt) in FLOWS.items():
        key = {"FSIC": "USA+DEU.*.*.*", "FAS": "USA+DEU.*.*.*.*.*.*", "CCI": "USA+DEU.*.*.*",
               "MFS_MA": "USA+DEU.*.*.*"}.get(fid, "USA+DEU.*.*")
        used = used_codes(ag, fid, key)
        if filt:
            used = {k: ({c for c in v if re.search(filt, c)} if k == "INDICATOR" else v) for k, v in used.items()}
        codelists(ag, fid, dims, used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
