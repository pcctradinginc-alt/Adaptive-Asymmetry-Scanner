"""TEMP read-only probe: official publication metadata of IMF vintage dataflows (removed before merge)."""
import re, sys, xml.etree.ElementTree as ET
import requests
UA = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (research; official-data-only)"}
ROOT = "https://api.imf.org/external/sdmx/3.0"
XS = "application/vnd.sdmx.structure+xml;version=3.0.0"
loc = lambda e: e.tag.rsplit("}", 1)[-1]
r = requests.get(ROOT + "/structure/dataflow", headers={**UA, "Accept": XS}, timeout=120)
root = ET.fromstring(r.content)
for e in root.iter():
    if loc(e) != "Dataflow" or not re.search(r"^(WEO|FM|MFS_MA)(_|$)", e.get("id") or ""):
        continue
    print("FLOW", e.get("agencyID"), e.get("id"), e.get("version"), dict(e.attrib))
    for a in e.iter():
        if loc(a) == "Annotation":
            parts = {loc(c): (c.text or "").strip()[:160] for c in a.iter() if c is not a}
            print("   ANN", a.get("id"), parts)
for fid, ag in (("WEO_2025_OCT_VINTAGE", "IMF.RES"), ("WEO", "IMF.RES")):
    r = requests.get(f"{ROOT}/structure/dataflow/{ag}/{fid}/+", params={"references": "none", "detail": "full"},
                     headers={**UA, "Accept": XS}, timeout=60)
    print("DETAIL", fid, r.status_code, r.content[:3000].decode("utf-8", "replace"))
    r = requests.get(f"{ROOT}/data/dataflow/{ag}/{fid}/+/USA.NGDP_RPCH.A", params={"startPeriod": "2025", "endPeriod": "2025", "attributes": "all"},
                     headers={**UA, "Accept": "application/vnd.sdmx.data+json;version=2.0.0"}, timeout=60)
    j = r.json(); d = j.get("data", {})
    print("META", fid, str(j.get("meta"))[:800])
    st = (d.get("structures") or [{}])[0]
    print("ATTRS", fid, str(st.get("attributes"))[:1500])
    print("ANNOT", fid, str(st.get("annotations"))[:800], str((d.get("dataSets") or [{}])[0].get("annotations"))[:300])
