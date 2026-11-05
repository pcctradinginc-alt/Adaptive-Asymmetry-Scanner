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
}


def main():
    for name, (path, params) in CASES.items():
        assert path.startswith("/") and urlsplit(ROOT).hostname == "api.imf.org"
        url = ROOT + path
        try:
            with requests.get(url, params=params,
                              headers={"Accept": "application/vnd.sdmx.structure+json;version=2.0,application/vnd.sdmx.data+json;version=2.0,*/*"},
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
                print(json.dumps({"name": name, "http": r.status_code,
                                  "type": r.headers.get("Content-Type", "")[:110],
                                  "content_length": length, "sample_bytes": received,
                                  "json_top_keys": keys}, sort_keys=True), flush=True)
        except requests.RequestException as exc:
            print(json.dumps({"name": name, "error_type": type(exc).__name__}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
