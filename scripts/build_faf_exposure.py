#!/usr/bin/env python3
"""
scripts/build_faf_exposure.py — Build script for config/faf_exposure.yaml
(FHWA Freight Analysis Framework structural exposure map).

Downloads the official FAF5 state-level regional database from
faf.ornl.gov, aggregates it into a compact commodity(SCTG2) <-> state
exposure table, and writes it to config/faf_exposure.yaml. This is a
*structural* exposure map (which states/commodities move which freight),
never a live feed — it is meant to be rebuilt manually/occasionally (e.g.
on a new FAF5 release), not on every scanner run.

Usage:

    python3 scripts/build_faf_exposure.py [--url <zip url>] [--out PATH]

By default the script discovers the current FAF5 state-level database
download link from the official FAF5 "Data Download" page
(https://faf.ornl.gov/faf5/) by looking for an href matching
``FAF5.*State.*\\.zip`` (case-insensitive). Pass --url to skip discovery
and point directly at a zip (useful for pinning a specific release, or
when the download page layout changes and discovery needs a stopgap).

The zip is streamed to a temporary directory and is NEVER committed —
only the compact derived YAML is written to --out.

No network access is required to import this module or to exercise the
aggregation/validation/loader logic — see tests/test_build_faf_exposure.py,
which feeds a small synthetic FAF-like CSV (zipped) straight into
aggregate_faf_csv() / build_exposure_document(), bypassing discover_faf_zip_url()
and download_zip().
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard requirement (requirements.txt)
    pd = None  # type: ignore

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]

# Official FAF5 "Data Download" landing page. The actual zip filename/path
# changes across FAF5 minor releases (e.g. FAF5.6.1_State.zip), so we do
# NOT hardcode a full URL — we discover the current link from this page.
FAF_PAGE_URL = "https://faf.ornl.gov/faf5/"

# Pattern for the official FAF5 state-level database download link.
# Matches e.g. "FAF5.6.1_State.zip", "FAF5_State_2023.zip", case-insensitive.
FAF_ZIP_LINK_RE = re.compile(
    r'href=["\']([^"\']*FAF5[^"\']*State[^"\']*\.zip)["\']',
    re.IGNORECASE,
)

DEFAULT_OUT = "config/faf_exposure.yaml"
DEFAULT_TOP_STATES = 10
DEFAULT_TOP_COMMODITIES = 8
DEFAULT_CHUNKSIZE = 200_000
MAX_OUTPUT_BYTES = 200 * 1024  # <200 KB per task spec

# --------------------------------------------------------------------------- #
# Official FAF/SCTG2 (Standard Classification of Transported Goods, 2-digit)
# commodity code names. These are standard, published codes (see the FAF5
# data dictionary / Census SCTG documentation). Codes not listed here are
# left with the code only (no name) rather than guessed.
# --------------------------------------------------------------------------- #
SCTG_NAMES: dict[str, str] = {
    "01": "Live animals and fish",
    "02": "Cereal grains",
    "03": "Other agricultural products",
    "04": "Animal feed",
    "05": "Meat/seafood",
    "06": "Milled grain products",
    "07": "Other foodstuffs",
    "08": "Alcoholic beverages",
    "09": "Tobacco products",
    "10": "Building stone",
    "11": "Natural sands",
    "12": "Gravel and crushed stone",
    "13": "Nonmetallic minerals n.e.c.",
    "14": "Metallic ores and concentrates",
    "15": "Coal",
    "16": "Crude petroleum",
    "17": "Gasoline and aviation turbine fuel",
    "18": "Fuel oils",
    "19": "Other coal and petroleum products",
    "20": "Basic chemicals",
    "21": "Pharmaceutical products",
    "22": "Fertilizers",
    "23": "Chemical products n.e.c.",
    "24": "Plastics and rubber",
    "25": "Logs and other wood in the rough",
    "26": "Wood products",
    "27": "Newsprint/paper",
    "28": "Paper articles",
    "29": "Printed products",
    "30": "Textiles, leather and articles thereof",
    "31": "Nonmetallic mineral products",
    "32": "Base metals",
    "33": "Articles of base metal",
    "34": "Machinery",
    "35": "Electronic and other electrical equipment and components",
    "36": "Motorized and other vehicles (including parts)",
    "37": "Transportation equipment n.e.c.",
    "38": "Precision instruments and apparatus",
    "39": "Furniture, mattresses and mixed articles",
    "40": "Miscellaneous manufactured products",
    "41": "Waste and scrap",
    "43": "Mixed freight",
    "99": "Unknown",
}

# Official FAF dms_mode codes.
MODE_NAMES: dict[str, str] = {
    "1": "Truck",
    "2": "Rail",
    "3": "Water",
    "4": "Air (incl. truck-air)",
    "5": "Multiple modes & mail",
    "6": "Pipeline",
    "7": "Other and unknown",
    "8": "No domestic mode",
}

REQUIRED_BASE_COLS = ["dms_origst", "dms_destst", "sctg2", "dms_mode"]

YEAR_COL_RE = re.compile(r"^tons_(\d{4})$")


class FafSchemaError(RuntimeError):
    """Raised when the downloaded/given CSV does not have the expected
    FAF5 columns (base columns or a complete tons/value/tmiles year triple).
    Fails loudly on purpose — we never want to silently aggregate over the
    wrong columns."""


# --------------------------------------------------------------------------- #
# Discovery + download
# --------------------------------------------------------------------------- #

def discover_faf_zip_url(page_url: str = FAF_PAGE_URL, timeout: int = 30) -> str:
    """Fetch the official FAF5 Data Download page and find the state-level
    database zip link (FAF5*State*.zip). Raises RuntimeError if none found."""
    if requests is None:
        raise RuntimeError("requests is required for FAF download-page discovery")
    resp = requests.get(page_url, timeout=timeout)
    resp.raise_for_status()
    html = resp.text
    matches = FAF_ZIP_LINK_RE.findall(html)
    if not matches:
        raise RuntimeError(
            f"Could not find a FAF5 state-level database zip link on {page_url} "
            f"(pattern: {FAF_ZIP_LINK_RE.pattern}). The page layout may have "
            f"changed — pass --url to override discovery."
        )
    # Prefer the last match (download pages typically list newest last, or
    # only ever list one current link); if several, take the first as the
    # most prominent/likely "current" link.
    href = matches[0]
    if href.startswith("http://") or href.startswith("https://"):
        return href
    # Relative URL: join against the page URL.
    from urllib.parse import urljoin
    return urljoin(page_url, href)


def download_zip(url: str, dest_path: Path, timeout: int = 120, chunk_size: int = 1 << 20) -> str:
    """Stream url to dest_path, returning the sha256 hex digest of the
    downloaded bytes. Never loads the whole zip into memory."""
    if requests is None:
        raise RuntimeError("requests is required to download the FAF5 zip")
    sha256 = hashlib.sha256()
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                sha256.update(chunk)
    return sha256.hexdigest()


def extract_first_csv(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the first *.csv member from zip_path into extract_dir and
    return its path. Raises FafSchemaError if no CSV member is found."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise FafSchemaError(f"No .csv member found inside {zip_path}")
        member = csv_members[0]
        zf.extract(member, path=extract_dir)
        return extract_dir / member


# --------------------------------------------------------------------------- #
# Column detection + validation
# --------------------------------------------------------------------------- #

def find_latest_year(columns: list[str]) -> int:
    """Find the latest year for which tons_<year>, value_<year> and
    tmiles_<year> ALL exist among columns. Raises FafSchemaError if none."""
    cols = set(columns)
    years = sorted({int(m.group(1)) for c in columns if (m := YEAR_COL_RE.match(c))}, reverse=True)
    for year in years:
        if {f"tons_{year}", f"value_{year}", f"tmiles_{year}"} <= cols:
            return year
    raise FafSchemaError(
        "Could not find a complete tons_<year>/value_<year>/tmiles_<year> "
        f"column triple among columns: {sorted(columns)}"
    )


def validate_columns(columns: list[str], year: int) -> list[str]:
    """Verify all required columns (base + the year triple) are present.
    Fails loudly (raises FafSchemaError) rather than silently proceeding
    with a subset — FAF column names/casing have shifted across releases."""
    required = REQUIRED_BASE_COLS + [f"tons_{year}", f"value_{year}", f"tmiles_{year}"]
    cols_lower = {c.lower(): c for c in columns}
    missing = [c for c in required if c not in columns and c.lower() not in cols_lower]
    if missing:
        raise FafSchemaError(
            f"FAF5 CSV is missing required columns for year {year}: {missing}. "
            f"Available columns: {sorted(columns)}. Refusing to aggregate over "
            f"an unverified/partial schema."
        )
    return required


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _fmt_sctg(v: Any) -> str:
    try:
        return f"{int(v):02d}"
    except (TypeError, ValueError):
        return str(v).strip()


def _fmt_state(v: Any) -> str:
    try:
        return f"{int(v):02d}"
    except (TypeError, ValueError):
        return str(v).strip()


def _fmt_mode(v: Any) -> str:
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v).strip()


def aggregate_faf_csv(csv_path: Path, year: int, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
    """Stream csv_path in chunks and accumulate:
      - tons by (sctg2, dms_origst)   -> commodity_state_shares (top-N later)
      - tons by (dms_origst, sctg2)   -> state_commodity_mix (top-N later)
      - tons by dms_mode              -> mode_shares

    Returns raw (untruncated, unnormalized) accumulator dicts plus
    grand_total_tons, so callers can apply top-N truncation and rounding.
    """
    if pd is None:
        raise RuntimeError("pandas is required to aggregate the FAF5 CSV")

    tons_col = f"tons_{year}"
    required = validate_columns(list(pd.read_csv(csv_path, nrows=0).columns), year)

    tons_by_sctg_state: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tons_by_state_sctg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tons_by_mode: dict[str, float] = defaultdict(float)
    grand_total_tons = 0.0

    usecols = REQUIRED_BASE_COLS + [tons_col]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["dms_origst", "sctg2", "dms_mode", tons_col])
        if chunk.empty:
            continue
        chunk["_sctg"] = chunk["sctg2"].map(_fmt_sctg)
        chunk["_state"] = chunk["dms_origst"].map(_fmt_state)
        chunk["_mode"] = chunk["dms_mode"].map(_fmt_mode)

        grp_ss = chunk.groupby(["_sctg", "_state"])[tons_col].sum()
        for (sctg, state), tons in grp_ss.items():
            tons_f = float(tons)
            tons_by_sctg_state[sctg][state] += tons_f
            tons_by_state_sctg[state][sctg] += tons_f

        grp_mode = chunk.groupby("_mode")[tons_col].sum()
        for mode, tons in grp_mode.items():
            tons_by_mode[str(mode)] += float(tons)

        grand_total_tons += float(chunk[tons_col].sum())

    _ = required  # validated above; kept for clarity
    return {
        "tons_by_sctg_state": {k: dict(v) for k, v in tons_by_sctg_state.items()},
        "tons_by_state_sctg": {k: dict(v) for k, v in tons_by_state_sctg.items()},
        "tons_by_mode": dict(tons_by_mode),
        "grand_total_tons": grand_total_tons,
    }


def _top_n_shares(totals: dict[str, float], n: int, round_ndigits: int = 4) -> dict[str, float]:
    """Given {key: raw_amount}, return the top-n keys as a share-of-total
    dict (share computed over ALL keys, not just the top-n — i.e. shares
    of the top-n need not sum to 1.0 if there are more than n keys)."""
    total = sum(totals.values())
    if total <= 0:
        return {}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return {k: round(v / total, round_ndigits) for k, v in ranked}


def build_exposure_document(
    agg: dict[str, Any],
    meta: dict[str, Any],
    top_states: int = DEFAULT_TOP_STATES,
    top_commodities: int = DEFAULT_TOP_COMMODITIES,
) -> dict[str, Any]:
    """Turn raw aggregation accumulators into the final compact document."""
    commodity_state_shares: dict[str, dict[str, float]] = {}
    for sctg, state_totals in agg["tons_by_sctg_state"].items():
        shares = _top_n_shares(state_totals, top_states)
        if shares:
            commodity_state_shares[sctg] = shares

    state_commodity_mix: dict[str, dict[str, float]] = {}
    for state, sctg_totals in agg["tons_by_state_sctg"].items():
        shares = _top_n_shares(sctg_totals, top_commodities)
        if shares:
            state_commodity_mix[state] = shares

    mode_totals = agg["tons_by_mode"]
    mode_total_sum = sum(mode_totals.values())
    mode_shares: dict[str, float] = {}
    if mode_total_sum > 0:
        for mode_code, tons in sorted(mode_totals.items(), key=lambda kv: kv[1], reverse=True):
            name = MODE_NAMES.get(mode_code, mode_code)
            mode_shares[name] = round(tons / mode_total_sum, 4)

    used_sctg = set(commodity_state_shares.keys())
    for mix in state_commodity_mix.values():
        used_sctg.update(mix.keys())
    sctg_names = {code: SCTG_NAMES[code] for code in sorted(used_sctg) if code in SCTG_NAMES}
    # Also include codes without a known name (code-only), per spec.
    for code in sorted(used_sctg):
        sctg_names.setdefault(code, None)

    return {
        "meta": dict(meta),
        "sctg_names": sctg_names,
        "commodity_state_shares": commodity_state_shares,
        "state_commodity_mix": state_commodity_mix,
        "mode_shares": mode_shares,
    }


def write_exposure_yaml(doc: dict[str, Any], out_path: Path) -> int:
    """Write doc as compact YAML to out_path, returning the byte size.
    Does not raise on size overrun (caller decides), but prints a warning."""
    text = yaml.safe_dump(doc, sort_keys=True, allow_unicode=True, default_flow_style=False, width=100)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    size = len(text.encode("utf-8"))
    if size > MAX_OUTPUT_BYTES:
        print(
            f"WARNING: {out_path} is {size} bytes, exceeding the {MAX_OUTPUT_BYTES} byte "
            f"target. Consider lowering --top-states/--top-commodities.",
            file=sys.stderr,
        )
    return size


# --------------------------------------------------------------------------- #
# Loader (for use by the rest of the codebase, e.g. modules/external)
# --------------------------------------------------------------------------- #

_YAML_CACHE: dict[str, Any] = {}


def _load_yaml_cached(rel_path: str) -> dict:
    if rel_path not in _YAML_CACHE:
        p = REPO_ROOT / rel_path
        with open(p, "r", encoding="utf-8") as f:
            _YAML_CACHE[rel_path] = yaml.safe_load(f) or {}
    return _YAML_CACHE[rel_path]


def load_faf_exposure(rel_path: str = "config/faf_exposure.yaml", use_cache: bool = True) -> dict:
    if not use_cache:
        p = REPO_ROOT / rel_path
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return _load_yaml_cached(rel_path)


def load_industry_exposure(rel_path: str = "config/industry_exposure.yaml", use_cache: bool = True) -> dict:
    if not use_cache:
        p = REPO_ROOT / rel_path
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return _load_yaml_cached(rel_path)


def exposure_states_for_industry(
    industry: str,
    faf_cfg: Optional[dict] = None,
    industry_cfg: Optional[dict] = None,
) -> Optional[dict[str, float]]:
    """Weighted origin-state exposure for a given industry (from
    config/industry_exposure.yaml's optional faf_commodities mapping),
    derived from config/faf_exposure.yaml's commodity_state_shares.

    Returns None (never {}) when:
      - the FAF exposure map has not been built yet (meta.built_at missing), or
      - the industry has no faf_commodities mapping, or
      - none of the mapped commodities have any state share data.

    Otherwise returns {state_fips: weight} renormalized to sum to 1.0,
    combining commodities with equal weight (mapping is relevance-only,
    no per-commodity weighting is implied by the config)."""
    faf = faf_cfg if faf_cfg is not None else load_faf_exposure()
    meta = faf.get("meta") or {}
    if not meta.get("built_at"):
        return None

    ind_cfg = industry_cfg if industry_cfg is not None else load_industry_exposure()
    industries = ind_cfg.get("industries") or {}
    entry = industries.get(industry) or {}
    commodities = entry.get("faf_commodities") or []
    if not commodities:
        return None

    commodity_state_shares = faf.get("commodity_state_shares") or {}
    combined: dict[str, float] = defaultdict(float)
    n_found = 0
    for code in commodities:
        key = _fmt_sctg(code)
        state_shares = commodity_state_shares.get(key)
        if not state_shares:
            continue
        n_found += 1
        for state, share in state_shares.items():
            combined[state] += float(share)

    if not n_found or not combined:
        return None

    total = sum(combined.values())
    if total <= 0:
        return None
    return {state: round(v / total, 4) for state, v in
            sorted(combined.items(), key=lambda kv: kv[1], reverse=True)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None,
                         help="Direct URL to the FAF5 state-level database zip "
                              "(skips discovery on the FAF5 Data Download page).")
    parser.add_argument("--page-url", default=FAF_PAGE_URL,
                         help="FAF5 Data Download page to discover the zip link from "
                              "(default: %(default)s).")
    parser.add_argument("--out", default=DEFAULT_OUT,
                         help="Output path for the compact exposure YAML (default: %(default)s).")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                         help="pandas read_csv chunksize (default: %(default)s).")
    parser.add_argument("--top-states", type=int, default=DEFAULT_TOP_STATES,
                         help="Top-N origin states kept per commodity (default: %(default)s).")
    parser.add_argument("--top-commodities", type=int, default=DEFAULT_TOP_COMMODITIES,
                         help="Top-N commodities kept per state (default: %(default)s).")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="faf5_") as tmp_str:
        tmp_dir = Path(tmp_str)
        zip_url = args.url or discover_faf_zip_url(args.page_url)
        print(f"FAF5 zip URL: {zip_url}", file=sys.stderr)

        zip_path = tmp_dir / "faf5_state.zip"
        sha256 = download_zip(zip_url, zip_path)
        print(f"Downloaded {zip_path.stat().st_size} bytes, sha256={sha256}", file=sys.stderr)

        csv_path = extract_first_csv(zip_path, tmp_dir)
        columns = list(pd.read_csv(csv_path, nrows=0).columns)
        year = find_latest_year(columns)
        validate_columns(columns, year)
        print(f"Detected latest year: {year}", file=sys.stderr)

        agg = aggregate_faf_csv(csv_path, year, chunksize=args.chunksize)

        meta = {
            "source_url": zip_url,
            "faf_version": "FAF5",
            "year": year,
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256_of_zip": sha256,
        }
        doc = build_exposure_document(
            agg, meta, top_states=args.top_states, top_commodities=args.top_commodities,
        )

        out_path = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        size = write_exposure_yaml(doc, out_path)
        print(f"Wrote {out_path} ({size} bytes)", file=sys.stderr)
        # tmp_dir (and the downloaded zip) is removed automatically on exit —
        # never committed.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
