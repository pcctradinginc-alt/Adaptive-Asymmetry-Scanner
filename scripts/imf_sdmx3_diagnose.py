"""Bounded, read-only probe for IMF SDMX 3.0 official gateway.

Diagnostic only: never ingests or trains; response *headers and shape* only,
no full value payload, credentials, cookies or personal information logged.
"""
from __future__ import annotations

import json
import sys
from urllib.parse import urlsplit

import requests

ROOT = "https://api.imf.org/external/sdmx/3.0"
CASES = {
    "WEO-structure": ("/structure/dataflow/IMF.RES/WEO/+", {"references": "all"}),
    "WEO-data": ("/data/dataflow/IMF.RES/WEO/+/DEU.NGDP_RPCH.A",
                 {"startPeriod": "2025", "endPeriod": "2026", "format": "jsondata"}),
    "FSI-structure": ("/structure/dataflow/IMF.STA/FSI/+", {"references": "all"}),
    "FSI-data": ("/data/dataflow/IMF.STA/FSI/+/DE.NPLR_PT._T.Q",
                 {"startPeriod": "2024", "endPeriod": "2026", "format": "jsondata"}),
    "FM-structure": ("/structure/dataflow/IMF.FAD/FM/+", {"references": "all"}),
    "FM-data": ("/data/dataflow/IMF.FAD/FM/+/DEU.GGXWDG_NGDP.A",
                {"startPeriod": "2024", "endPeriod": "2026", "format": "jsondata"}),
    "MFS-structure": ("/structure/dataflow/IMF.STA/MFS_MA/+", {"references": "all"}),
    "FAS-structure": ("/structure/dataflow/IMF.STA/FAS/+", {"references": "all"}),
    "CCI-structure": ("/structure/dataflow/IMF.STA/CCI/+", {"references": "all"}),
    "GDD-structure": ("/structure/dataflow/IMF.STA/GDD/+", {"references": "all"}),
    "AIPI-structure": ("/structure/dataflow/IMF.STA/AIPI/+", {"references": "all"}),
    "WEO-structure-no-ref": ("/structure/dataflow/IMF.RES/WEO/+", {}),
    "WEO-structure-refs-dsd": ("/structure/dataflow/IMF.RES/WEO/+", {"references": "datastructure"}),
    "WEO-structure-no-plus": ("/structure/dataflow/IMF.RES/WEO", {"references": "all"}),
    "WEO-structure-list": ("/structure/dataflow", {}),
    "FSI-2_1-metadata": ("/structure/dataflow/IMF.STA/FSI/+", {"references": "datastructure"}),
    "FSI-data-wildcard": ("/data/dataflow/IMF.STA/FSI/+/DE.*._T.Q",
                 {"startPeriod": "2025", "endPeriod": "2026", "format": "jsondata"}),
    "MFS-data-guess-safebounded": ("/data/dataflow/IMF.STA/MFS_MA/+/DE.*.M",
                 {"startPeriod": "2026", "endPeriod": "2026", "format": "jsondata"}),
    "FSIC-3-structure": ("/structure/dataflow/IMF.STA/FSIC/+", {"references":"datastructure"}),
    "FSI-global-structure": ("/structure/dataflow/IMF/FSI/+", {"references":"datastructure"}),
    "GDD-fad-structure": ("/structure/dataflow/IMF.FAD/FAD_GDD/+", {"references":"datastructure"}),
    "GDD-fad-short-structure": ("/structure/dataflow/IMF.FAD/GDD/+", {"references":"datastructure"}),
    "AIPI-res-structure": ("/structure/dataflow/IMF.RES/AIPI/+", {"references":"datastructure"}),

}


def main():
    for name, (path, params) in CASES.items():
        assert path.startswith("/") and urlsplit(ROOT).hostname == "api.imf.org"
        url = ROOT + path
        try:
            accept = ("application/vnd.sdmx.structure+xml;version=2.1"
                      if path.startswith("/structure/") else
                      "application/vnd.sdmx.data+json;version=2.0")
            with requests.get(url, params=params,
                              headers={"Accept": accept},
                              timeout=(5, 16), stream=True) as r:
                length = int(r.headers.get("Content-Length") or 0)
                chunks, received = [], 0
                for chunk in r.iter_content(32768):
                    received += len(chunk)
                    if received > 256000:
                        break
                    chunks.append(chunk)
                content = b"".join(chunks)
                keys = []
                if r.headers.get("Content-Type", "").lower().find("json") >= 0 and content[:1] == b"{":
                    try:
                        root = json.loads(content)
                        keys = list(root)[:12] if isinstance(root, dict) else []
                    except (ValueError, UnicodeDecodeError):
                        pass
                info = {}
                if content.lstrip().startswith(b"<"):
                    import xml.etree.ElementTree as ET
                    try:
                        node = ET.fromstring(content)
                        info["xml_root"] = node.tag.rsplit("}", 1)[-1]
                        info["dataflows"] = sorted([{"id": x.get("id"), "agency": x.get("agencyID")}
                            for x in node.iter() if x.tag.rsplit("}", 1)[-1] == "Dataflow"],
                            key=lambda x: (str(x["agency"]), str(x["id"])))[:20]
                        info["dsds"] = sorted([x.get("id") for x in node.iter()
                            if x.tag.rsplit("}", 1)[-1] == "DataStructure"])[:15]
                    except ET.ParseError:
                        info["xml_status"] = "valid XML prefix but response was truncated"
                if content[:1] == b"{" and len(content) < 256000:
                    try:
                        root = json.loads(content)
                        if r.status_code >= 400:
                            info["error_code"] = str(root.get("code", ""))[:50]
                            info["error_message"] = str(root.get("message", ""))[:170]
                            info["dev_message"] = str(root.get("devMessage", ""))[:170]
                        elif r.status_code == 200:
                            d = root.get("data") or {}
                            info["data_keys"] = list(d)[:12]
                            structures = d.get("structures") or []
                            if isinstance(structures, list) and structures:
                                st = structures[0]
                                info["structure_keys"] = list(st)[:15]
                                dim = st.get("dimensions") or {}
                                info["dimensions"] = {
                                    k: [{"id": x.get("id"), "n_codes": len(x.get("values") or []),
                                         "first_codes": [v.get("id") for v in (x.get("values") or [])[:4]]}
                                        for x in (vals if isinstance(vals, list) else [])[:10]]
                                    for k, vals in dim.items()
                                } if isinstance(dim, dict) else {}
                            datasets = d.get("dataSets") or []
                            if isinstance(datasets, list) and datasets:
                                ds = datasets[0]
                                info["dataset_keys"] = list(ds)[:15]
                                info["series_count"] = len(ds.get("series") or {})
                                info["first_series_key"] = next(iter(ds.get("series") or {}), "")
                    except (ValueError, TypeError, AttributeError):
                        info["decode"] = "unsupported response structure"
                print(json.dumps({"name": name, "http": r.status_code,
                                  "type": r.headers.get("Content-Type", "")[:110],
                                  "content_length": length, "sample_bytes": received,
                                  "json_top_keys": keys, **info}, sort_keys=True), flush=True)
        except requests.RequestException as exc:
            print(json.dumps({"name": name, "error_type": type(exc).__name__}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
