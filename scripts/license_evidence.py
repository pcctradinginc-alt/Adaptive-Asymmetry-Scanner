"""
scripts/license_evidence.py – sammelt Primärbelege zu Lizenz/Nutzungsbedingungen
externer Quellen (Governance: keine Lizenz raten, sondern belegen).

Läuft im Preflight-Workflow (GitHub Actions, dort ist Internet verfügbar) und
schreibt outputs/external_data/health/license_evidence.json mit:
  - ArcGIS-Item-Metadaten der PortWatch-Datensätze (licenseInfo,
    accessInformation, owner, orgId, url) aus der offiziellen ArcGIS-Suche
  - Textauszügen der IMF-Copyright-/Nutzungsseite und der PortWatch-FAQ /
    Data-&-Methodology-Seite, gefiltert auf lizenzrelevante Absätze

Keine Bewertung im Code: Die Entscheidung (license_status OK/REVIEW_REQUIRED)
trifft ein Mensch per PR auf config/external_sources/maritime.yaml.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import requests

UA = {"User-Agent": "AdaptiveAsymmetryScanner/1.0 (license-evidence; research)"}
OUT = Path("outputs/external_data/health/license_evidence.json")

ARCGIS_SEARCH = "https://www.arcgis.com/sharing/rest/search"
PORTWATCH_TITLES = ["Daily Ports Data", "Daily Chokepoints Data"]
PAGES = {
    "imf_copyright_and_terms": "https://www.imf.org/en/about/copyright-and-terms",
    "imf_terms_legacy": "https://www.imf.org/external/terms.htm",
    "portwatch_faqs": "https://portwatch.imf.org/pages/faqs",
    "portwatch_data_methodology": "https://portwatch.imf.org/pages/data-and-methodology",
    "estat_terms_of_use": "https://www.e-stat.go.jp/en/terms-of-use",
    "estat_api_credit": "https://www.e-stat.go.jp/api/en/api-info/credit",
}
KEYWORDS = re.compile(
    r"(licen[cs]e|terms|copyright|commercial|attribut|cite|citation|permission|"
    r"reuse|re-use|redistribut|derivative|api|download|automated|scrap)", re.I)


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>|</div>", "\n", html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _relevant_paragraphs(text: str, limit: int = 40) -> list[str]:
    out = []
    for para in text.split("\n"):
        if len(para) >= 40 and KEYWORDS.search(para):
            out.append(para[:800])
        if len(out) >= limit:
            break
    return out


def arcgis_items() -> list[dict]:
    items = []
    for title in PORTWATCH_TITLES:
        try:
            r = requests.get(ARCGIS_SEARCH, params={"q": f'title:"{title}"', "f": "json", "num": 10},
                             headers=UA, timeout=30)
            for it in r.json().get("results", []):
                if "portwatch" not in json.dumps(it).lower() and it.get("title") != title:
                    continue
                items.append({k: it.get(k) for k in (
                    "id", "title", "owner", "orgId", "url", "type", "modified",
                    "licenseInfo", "accessInformation", "snippet")})
        except Exception as e:  # noqa: BLE001
            items.append({"title": title, "error": repr(e)})
    for it in items:
        if it.get("licenseInfo"):
            it["licenseInfo_text"] = _strip_html(it["licenseInfo"])[:3000]
    return items


def pages() -> dict:
    out = {}
    for key, url in PAGES.items():
        try:
            r = requests.get(url, headers=UA, timeout=30)
            text = _strip_html(r.text)
            out[key] = {"url": url, "http_status": r.status_code,
                        "relevant_paragraphs": _relevant_paragraphs(text)}
        except Exception as e:  # noqa: BLE001
            out[key] = {"url": url, "error": repr(e)}
    return out


def main() -> int:
    evidence = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portwatch_arcgis_items": arcgis_items(),
        "pages": pages(),
        "note": "Belege, keine Entscheidung. license_status wird per menschlichem PR gesetzt.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(json.dumps(evidence, indent=2, ensure_ascii=False)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
