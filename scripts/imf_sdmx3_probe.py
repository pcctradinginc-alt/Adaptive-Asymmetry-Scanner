"""TEMP read-only discovery probe for IMF SDMX 3.0 (removed before merge)."""
import json, re, sys, xml.etree.ElementTree as ET
import requests

ROOT = "https://api.imf.org/external/sdmx/3.0"
UA = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (research; official-data-only)"}
XS = "application/vnd.sdmx.structure+xml;version=3.0.0"
JD = "application/vnd.sdmx.data+json;version=2.0.0"
KW = re.compile(r"(FSI|WEO|^FM$|FISCAL|GDD|DEBT|MFS|MONETARY|FAS|FINANCIAL ACCESS|CCI|CLIMATE|AIPI|ARTIFICIAL|PREPARED)", re.I)


def loc(e):
    return e.tag.rsplit("}", 1)[-1]


def get(path, params=None, accept=JD, cap=6_000_000, timeout=60):
    try:
        with requests.get(ROOT + path, params=params, headers={**UA, "Accept": accept},
                          timeout=(10, timeout), stream=True) as r:
            buf, n = [], 0
            for ch in r.iter_content(65536):
                n += len(ch)
                if n > cap:
                    return r.status_code, r.headers.get("Content-Type", ""), None, "TRUNC"
                buf.append(ch)
            return r.status_code, r.headers.get("Content-Type", ""), b"".join(buf), r.url
    except requests.RequestException as e:
        return 0, "", None, type(e).__name__


def name_of(e):
    for c in e:
        if loc(c) == "Name":
            return (c.text or "").strip()
    return ""


def p(*a):
    print(*a, flush=True)


def flows():
    st, ct, body, u = get("/structure/dataflow", accept=XS, cap=20_000_000, timeout=120)
    p("DATAFLOW_LIST", st, ct, len(body or b""), u)
    out = []
    if body:
        root = ET.fromstring(body)
        for e in root.iter():
            if loc(e) == "Dataflow" and e.get("id"):
                nm = name_of(e)
                if KW.search(e.get("id")) or KW.search(nm):
                    out.append((e.get("agencyID"), e.get("id"), e.get("version"), nm))
    for f in out:
        p("FLOW", json.dumps(f))
    return out


def dsd(agency, fid):
    st, ct, body, u = get(f"/structure/dataflow/{agency}/{fid}/+", {"references": "datastructure"}, accept=XS)
    dims = []
    if st == 200 and body:
        root = ET.fromstring(body)
        for d in root.iter():
            if loc(d) == "DimensionList":
                dims = [x.get("id") for x in d if x.get("id") and loc(x) in ("Dimension",)]
                break
    p("DSD", agency, fid, st, dims)
    return dims


def data(label, path, params, accept=JD, show_codes=0):
    st, ct, body, u = get(path, params, accept=accept)
    info = {"label": label, "http": st, "ct": ct[:60], "bytes": len(body or b""), "url": str(u)[:220]}
    if body and body[:1] == b"{":
        try:
            j = json.loads(body)
        except ValueError:
            j = {}
        if st >= 400:
            info["err"] = json.dumps(j)[:300]
        else:
            d = j.get("data") or {}
            ds = (d.get("dataSets") or [{}])[0]
            info["n_series"] = len(ds.get("series") or {})
            sts = d.get("structures") or [{}]
            dims = (sts[0].get("dimensions") or {})
            info["series_dims"] = [(x.get("id"), len(x.get("values") or [])) for x in dims.get("series") or []]
            info["obs_dims"] = [(x.get("id"), [v.get("id") or v.get("value") for v in (x.get("values") or [])][-6:]) for x in dims.get("observation") or []]
            sk = next(iter(ds.get("series") or {}), None)
            if sk:
                info["first_series"] = sk
                info["first_obs"] = list((ds["series"][sk].get("observations") or {}).items())[-2:]
            p("DATA", json.dumps(info, default=str)[:1500])
            if show_codes:
                for x in dims.get("series") or []:
                    vals = x.get("values") or []
                    if len(vals) > 1:
                        p("  CODES", x.get("id"), len(vals))
                        for v in vals[:show_codes]:
                            nm = v.get("name") or v.get("names") or ""
                            if isinstance(nm, dict):
                                nm = nm.get("en") or ""
                            p("   ", v.get("id"), "|", str(nm)[:90])
            return st
    elif body:
        info["head"] = body[:300].decode("utf-8", "replace")
    p("DATA", json.dumps(info, default=str)[:1500])
    return st


def main():
    fl = flows()
    # 1) WEO variants for the 500 issue
    base = "/data/dataflow/IMF.RES/WEO/+/USA.NGDP_RPCH.A"
    data("weo-plain", base, None)
    data("weo-ctime", base, {"c[TIME_PERIOD]": "ge:2020"})
    data("weo-start", base, {"startPeriod": "2020"})
    data("weo-format", base, {"format": "jsondata"})
    data("weo-accept-xml", base, None, accept="application/vnd.sdmx.data+xml;version=3.0.0")
    data("weo-accept-any", base, None, accept="*/*")
    data("weo-tilde", "/data/dataflow/IMF.RES/WEO/~/USA.NGDP_RPCH.A", None)
    data("weo-star", "/data/dataflow/IMF.RES/WEO/*/USA.NGDP_RPCH.A", None)
    data("weo-lastn", base, {"lastNObservations": "3"})
    seen = set()
    for ag, fid, ver, nm in fl:
        if (ag, fid) in seen:
            continue
        seen.add((ag, fid))
        dims = dsd(ag, fid)
        if not dims:
            continue
        for cty in ("USA", "DEU", "US"):
            key = ".".join(cty if d in ("COUNTRY", "REF_AREA", "JURISDICTION") else "*" for d in dims)
            st = data(f"{fid}-{cty}-wild", f"/data/dataflow/{ag}/{fid}/+/{key}",
                      {"c[TIME_PERIOD]": "ge:2022"}, show_codes=120)
            if st == 200:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
