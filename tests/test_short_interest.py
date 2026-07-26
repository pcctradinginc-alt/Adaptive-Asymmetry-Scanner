"""
Tests für die Short-Interest-Extraktion (modules/data_ingestion.py, v7.2)

Deckt den Einheiten-Bugfix ab (shortRatio ist Days-to-Cover, KEIN Prozent-
vom-Float, und darf nicht als Fallback für shortPercentOfFloat dienen),
das neue short_mom-Feature (Short-Interest-Momentum ggü. Vormonat) sowie
die Anbindung an candidate["features"] fürs Backtesting (nur vorhandene
Werte, kein 0.0-Default).

Kein Netzwerkzugriff nötig — extract_short_interest() ist eine reine,
modulweite Hilfsfunktion, die direkt mit info-Dicts getestet wird.
Ausführen: pytest tests/test_short_interest.py -v
"""

import pytest

from modules.data_ingestion import (
    extract_short_interest,
    SHORT_INTEREST_HIGH,
    SHORT_INTEREST_MED,
)


# ── (a) shortPercentOfFloat als Ratio (0.15) ───────────────────────────────

def test_short_percent_of_float_as_ratio():
    result = extract_short_interest({"shortPercentOfFloat": 0.15})
    assert result["short_float"] == pytest.approx(0.15)
    assert result["label"] == "high"   # 0.15 >= SHORT_INTEREST_HIGH
    assert result["features"]["short_pct_float"] == pytest.approx(0.15)


# ── (b) shortPercentOfFloat als Prozent (15.0) → wird zu 0.15 normalisiert ─

def test_short_percent_of_float_as_percent_gets_normalized():
    result = extract_short_interest({"shortPercentOfFloat": 15.0})
    assert result["short_float"] == pytest.approx(0.15)
    assert result["label"] == "high"
    assert result["features"]["short_pct_float"] == pytest.approx(0.15)


# ── (c) NUR shortRatio vorhanden → Einheiten-Bug darf nicht mehr auftreten ─

def test_only_short_ratio_present_no_unit_mixing():
    result = extract_short_interest({"shortRatio": 2.28})
    # shortRatio (Days-to-Cover) darf NICHT als short_float interpretiert werden
    assert result["short_float"] == 0.0
    assert result["label"] == "normal"
    assert result["short_ratio_days"] == pytest.approx(2.28)
    # Kein Feature-Key, da short_float unbekannt (nicht künstlich auf 0.0 gesetzt)
    assert "short_pct_float" not in result["features"]


# ── (d) short_mom: sharesShort=110, prior=100 → +10% ───────────────────────

def test_short_momentum_computed():
    result = extract_short_interest({
        "sharesShort": 110,
        "sharesShortPriorMonth": 100,
    })
    assert result["short_mom"] == pytest.approx(0.10)
    assert result["features"]["short_mom"] == pytest.approx(0.10)


# ── (e) short_mom: prior fehlt oder ist 0 → None, kein Feature-Key ─────────

@pytest.mark.parametrize("info", [
    {"sharesShort": 110, "sharesShortPriorMonth": 0},
    {"sharesShort": 110},   # sharesShortPriorMonth fehlt komplett
    {"sharesShortPriorMonth": 100},   # sharesShort fehlt komplett
])
def test_short_momentum_none_when_prior_missing_or_zero(info):
    result = extract_short_interest(info)
    assert result["short_mom"] is None
    assert "short_mom" not in result["features"]


# ── (f) Defensive Robustheit: leer, None-Werte, Müll-Strings ───────────────

def test_empty_info_dict_no_exception():
    result = extract_short_interest({})
    assert result["short_float"] == 0.0
    assert result["label"] == "normal"
    assert result["short_ratio_days"] is None
    assert result["short_mom"] is None
    assert result["features"] == {}


def test_none_info_no_exception():
    result = extract_short_interest(None)
    assert result["short_float"] == 0.0
    assert result["label"] == "normal"
    assert result["features"] == {}


def test_none_values_no_exception():
    result = extract_short_interest({
        "shortPercentOfFloat": None,
        "shortRatio": None,
        "sharesShort": None,
        "sharesShortPriorMonth": None,
    })
    assert result["short_float"] == 0.0
    assert result["label"] == "normal"
    assert result["short_ratio_days"] is None
    assert result["short_mom"] is None
    assert result["features"] == {}


def test_garbage_strings_no_exception():
    result = extract_short_interest({
        "shortPercentOfFloat": "n/a",
        "shortRatio": "unbekannt",
        "sharesShort": "viele",
        "sharesShortPriorMonth": "",
    })
    assert result["short_float"] == 0.0
    assert result["label"] == "normal"
    assert result["short_ratio_days"] is None
    assert result["short_mom"] is None
    assert result["features"] == {}


# ── Label-Schwellen (elevated) ──────────────────────────────────────────────

def test_label_elevated_between_thresholds():
    # Zwischen MED (0.08) und HIGH (0.15)
    result = extract_short_interest({"shortPercentOfFloat": 0.10})
    assert result["label"] == "elevated"
    assert SHORT_INTEREST_MED <= result["short_float"] < SHORT_INTEREST_HIGH


def test_label_normal_below_med_threshold():
    result = extract_short_interest({"shortPercentOfFloat": 0.03})
    assert result["label"] == "normal"


# ── Kombination: short_float + short_mom gemeinsam vorhanden ───────────────

def test_both_short_float_and_momentum_present():
    result = extract_short_interest({
        "shortPercentOfFloat": 0.20,
        "shortRatio": 4.5,
        "sharesShort": 120,
        "sharesShortPriorMonth": 100,
    })
    assert result["short_float"] == pytest.approx(0.20)
    assert result["label"] == "high"
    assert result["short_ratio_days"] == pytest.approx(4.5)
    assert result["short_mom"] == pytest.approx(0.20)
    assert result["features"]["short_pct_float"] == pytest.approx(0.20)
    assert result["features"]["short_mom"] == pytest.approx(0.20)
