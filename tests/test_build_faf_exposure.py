"""
tests/test_build_faf_exposure.py

Exercises scripts/build_faf_exposure.py's aggregation/validation/loader
logic against a small synthetic FAF-like CSV (zipped in-memory in the
test, mirroring the real FAF5 state-level database's structure). No
network access is used — discover_faf_zip_url()/download_zip() are not
exercised here.
"""

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_faf_exposure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_faf_exposure", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_faf_exposure"] = mod
    spec.loader.exec_module(mod)
    return mod


faf = _load_module()


SYNTHETIC_CSV_2022 = """dms_origst,dms_destst,sctg2,dms_mode,tons_2022,value_2022,tmiles_2022
6,6,36,1,1000,50000,100
6,48,36,1,500,25000,300
48,6,36,2,300,15000,250
6,6,20,1,800,40000,80
48,48,20,1,200,10000,20
36,36,2,1,400,20000,40
36,6,2,2,100,5000,300
"""

SYNTHETIC_CSV_MISSING_COLS = """dms_origst,dms_destst,sctg2,tons_2022
6,6,36,1000
"""


def _make_zip(csv_text: str, csv_name: str = "FAF5_State.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(csv_name, csv_text)
    return buf.getvalue()


@pytest.fixture()
def synthetic_csv(tmp_path):
    p = tmp_path / "FAF5_State.csv"
    p.write_text(SYNTHETIC_CSV_2022, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Zip extraction
# --------------------------------------------------------------------------- #

def test_extract_first_csv_from_zip(tmp_path):
    zip_bytes = _make_zip(SYNTHETIC_CSV_2022)
    zip_path = tmp_path / "faf5.zip"
    zip_path.write_bytes(zip_bytes)
    csv_path = faf.extract_first_csv(zip_path, tmp_path)
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8") == SYNTHETIC_CSV_2022


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #

def test_find_latest_year_detects_complete_triple():
    columns = ["dms_origst", "dms_destst", "sctg2", "dms_mode",
               "tons_2017", "value_2017", "tmiles_2017",
               "tons_2022", "value_2022", "tmiles_2022",
               "tons_2050"]  # incomplete future year, no value/tmiles
    year = faf.find_latest_year(columns)
    assert year == 2022


def test_find_latest_year_raises_when_no_complete_triple():
    with pytest.raises(faf.FafSchemaError):
        faf.find_latest_year(["dms_origst", "tons_2022"])


def test_validate_columns_missing_raises_loudly():
    with pytest.raises(faf.FafSchemaError):
        faf.validate_columns(["dms_origst", "dms_destst", "sctg2", "tons_2022"], 2022)


def test_validate_columns_ok():
    columns = ["dms_origst", "dms_destst", "sctg2", "dms_mode",
               "tons_2022", "value_2022", "tmiles_2022"]
    required = faf.validate_columns(columns, 2022)
    assert "tons_2022" in required
    assert "dms_mode" in required


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def test_aggregate_faf_csv_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(SYNTHETIC_CSV_MISSING_COLS, encoding="utf-8")
    with pytest.raises(faf.FafSchemaError):
        faf.aggregate_faf_csv(p, 2022)


def test_aggregate_faf_csv_basic(synthetic_csv):
    agg = faf.aggregate_faf_csv(synthetic_csv, 2022, chunksize=3)
    # sctg 36: state 06 has 1000+500=1500, state 48 has 300 -> total 1800
    assert agg["tons_by_sctg_state"]["36"]["06"] == pytest.approx(1500.0)
    assert agg["tons_by_sctg_state"]["36"]["48"] == pytest.approx(300.0)
    # mode totals: mode 1 (truck) = 1000+500+800+200+400=2900, mode 2 (rail) = 300+100=400
    assert agg["tons_by_mode"]["1"] == pytest.approx(2900.0)
    assert agg["tons_by_mode"]["2"] == pytest.approx(400.0)
    assert agg["grand_total_tons"] == pytest.approx(3300.0)


def test_aggregate_shares_sum_to_one_over_all_states(synthetic_csv):
    agg = faf.aggregate_faf_csv(synthetic_csv, 2022)
    meta = {"source_url": "test://x", "faf_version": "FAF5", "year": 2022,
            "built_at": "2026-01-01T00:00:00Z", "sha256_of_zip": "deadbeef"}
    doc = faf.build_exposure_document(agg, meta, top_states=10, top_commodities=10)
    for sctg, state_shares in doc["commodity_state_shares"].items():
        # with top_states=10 >= number of distinct states per commodity here,
        # shares should sum to ~1.0
        assert sum(state_shares.values()) == pytest.approx(1.0, abs=1e-6)
    total_mode_share = sum(doc["mode_shares"].values())
    assert total_mode_share == pytest.approx(1.0, abs=1e-6)


def test_top_n_truncation(synthetic_csv):
    agg = faf.aggregate_faf_csv(synthetic_csv, 2022)
    meta = {"source_url": "t", "faf_version": "FAF5", "year": 2022,
            "built_at": "2026-01-01T00:00:00Z", "sha256_of_zip": "x"}
    doc = faf.build_exposure_document(agg, meta, top_states=1, top_commodities=10)
    for sctg, state_shares in doc["commodity_state_shares"].items():
        assert len(state_shares) <= 1


def test_build_exposure_document_includes_sctg_names(synthetic_csv):
    agg = faf.aggregate_faf_csv(synthetic_csv, 2022)
    meta = {"source_url": "t", "faf_version": "FAF5", "year": 2022,
            "built_at": "2026-01-01T00:00:00Z", "sha256_of_zip": "x"}
    doc = faf.build_exposure_document(agg, meta)
    assert doc["sctg_names"]["36"] == "Motorized and other vehicles (including parts)"
    assert doc["sctg_names"]["20"] == "Basic chemicals"
    assert doc["meta"]["built_at"] == "2026-01-01T00:00:00Z"


# --------------------------------------------------------------------------- #
# YAML output size
# --------------------------------------------------------------------------- #

def test_write_exposure_yaml_is_small(tmp_path, synthetic_csv):
    agg = faf.aggregate_faf_csv(synthetic_csv, 2022)
    meta = {"source_url": "t", "faf_version": "FAF5", "year": 2022,
            "built_at": "2026-01-01T00:00:00Z", "sha256_of_zip": "x"}
    doc = faf.build_exposure_document(agg, meta)
    out_path = tmp_path / "faf_exposure.yaml"
    size = faf.write_exposure_yaml(doc, out_path)
    assert out_path.exists()
    assert size < faf.MAX_OUTPUT_BYTES
    # round-trips as valid YAML
    reloaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert reloaded["meta"]["year"] == 2022
    # sctg codes must round-trip as zero-padded strings, not ints (01 vs 1)
    assert "36" in reloaded["commodity_state_shares"]


def test_real_placeholder_config_is_small_and_unbuilt():
    p = REPO_ROOT / "config" / "faf_exposure.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["meta"]["built_at"] is None
    assert p.stat().st_size < faf.MAX_OUTPUT_BYTES


# --------------------------------------------------------------------------- #
# Loader: exposure_states_for_industry
# --------------------------------------------------------------------------- #

def _built_faf_cfg():
    return {
        "meta": {"built_at": "2026-01-01T00:00:00Z", "year": 2022,
                  "source_url": "t", "faf_version": "FAF5", "sha256_of_zip": "x"},
        "sctg_names": {"36": "Motorized and other vehicles (including parts)"},
        "commodity_state_shares": {
            "36": {"06": 0.7, "48": 0.3},
        },
        "state_commodity_mix": {},
        "mode_shares": {},
    }


def _industry_cfg_with_mapping():
    return {"industries": {"Autos": {"faf_commodities": [36]}, "Unmapped": {}}}


def test_loader_returns_none_when_not_built():
    unbuilt = {"meta": {"built_at": None}, "commodity_state_shares": {}}
    ind_cfg = _industry_cfg_with_mapping()
    result = faf.exposure_states_for_industry("Autos", faf_cfg=unbuilt, industry_cfg=ind_cfg)
    assert result is None


def test_loader_returns_none_when_no_mapping():
    built = _built_faf_cfg()
    ind_cfg = _industry_cfg_with_mapping()
    result = faf.exposure_states_for_industry("Unmapped", faf_cfg=built, industry_cfg=ind_cfg)
    assert result is None


def test_loader_returns_none_for_unknown_industry():
    built = _built_faf_cfg()
    ind_cfg = _industry_cfg_with_mapping()
    result = faf.exposure_states_for_industry("Totally Unknown Industry", faf_cfg=built, industry_cfg=ind_cfg)
    assert result is None


def test_loader_returns_weighted_states_when_built():
    built = _built_faf_cfg()
    ind_cfg = _industry_cfg_with_mapping()
    result = faf.exposure_states_for_industry("Autos", faf_cfg=built, industry_cfg=ind_cfg)
    assert result is not None
    assert result["06"] == pytest.approx(0.7)
    assert result["48"] == pytest.approx(0.3)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_loader_against_real_config_files_returns_none_until_built():
    # Real config/faf_exposure.yaml on disk is not built yet, so the
    # loader must return None even though config/industry_exposure.yaml
    # now carries faf_commodities mappings (e.g. Autos -> [36]).
    real_faf_cfg = faf.load_faf_exposure(use_cache=False)
    result = faf.exposure_states_for_industry("Autos", faf_cfg=real_faf_cfg)
    assert result is None


# --------------------------------------------------------------------------- #
# industry_exposure.yaml faf_commodities sanity
# --------------------------------------------------------------------------- #

def test_industry_exposure_yaml_faf_commodities_are_known_sctg_codes():
    ind_cfg = yaml.safe_load((REPO_ROOT / "config" / "industry_exposure.yaml").read_text(encoding="utf-8"))
    for name, entry in (ind_cfg.get("industries") or {}).items():
        for code in entry.get("faf_commodities", []) or []:
            padded = f"{int(code):02d}"
            assert padded in faf.SCTG_NAMES, f"{name}: unknown SCTG2 code {code}"
