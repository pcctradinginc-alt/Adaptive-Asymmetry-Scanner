"""
Tests for final_mc_shadow feature in Stage 8 Final MC.

Validates:
1. ttm_to_dte_floor function correctly maps TTM strings to DTE floors
2. compute_final_mc_shadow helper function structure
3. Code contains required fields and functions
"""

import pytest
from modules.options_designer import ttm_to_dte_floor
from pipeline import compute_final_mc_shadow


class TestTtmToDteFloor:
    """Unit tests for ttm_to_dte_floor function."""

    def test_exact_mappings(self):
        """Test exact TTM-to-DTE mappings from TTM_TO_DTE_MIN dict."""
        assert ttm_to_dte_floor("4-8 Wochen") == 120
        assert ttm_to_dte_floor("2-3 Monate") == 120
        assert ttm_to_dte_floor("6 Monate") == 140

    def test_case_insensitive_monat(self):
        """Test case-insensitive matching for 'monat'."""
        # "6 Monate" matches exact mapping
        assert ttm_to_dte_floor("6 Monate") == 140

        # Other months without exact mapping default to 120
        assert ttm_to_dte_floor("3 Monate") == 120
        assert ttm_to_dte_floor("4 Monate") == 120

    def test_case_insensitive_monat_lowercase(self):
        """Test lowercase versions."""
        assert ttm_to_dte_floor("4-8 wochen") == 120
        assert ttm_to_dte_floor("2-3 monate") == 120
        assert ttm_to_dte_floor("6 monate") == 140

    def test_unknown_format_defaults_to_120(self):
        """Test that unknown formats default to 120."""
        assert ttm_to_dte_floor("") == 120
        assert ttm_to_dte_floor(None) == 120
        assert ttm_to_dte_floor("unknown") == 120
        assert ttm_to_dte_floor("1-2 weeks") == 120

    def test_six_monat_variations(self):
        """Test variations of 6 Monate still map to 140."""
        assert ttm_to_dte_floor("6 Monat") == 140
        assert ttm_to_dte_floor("6 Monate") == 140
        assert ttm_to_dte_floor("6 monAte") == 140


class TestComputeFinalMcShadow:
    """Unit tests for compute_final_mc_shadow helper function."""

    def test_shadow_structure(self):
        """Test that compute_final_mc_shadow returns correct structure."""
        # Mock sim object with run_for_dte method
        class MockSim:
            def run_for_dte(self, s, days_to_expiry):
                return {
                    "simulation": {"hit_rate": 0.55}
                }

        # Create test signal with deep_analysis
        test_signal = {
            "ticker": "TEST",
            "deep_analysis": {
                "time_to_materialization": "4-8 Wochen"
            }
        }

        mock_sim = MockSim()
        result = compute_final_mc_shadow(
            sim=mock_sim,
            s=test_signal,
            final_dte=45,
            hit_rate=0.50,
            min_long=0.50
        )

        # Check all required keys present
        assert "dte_used" in result
        assert "hit_rate_used" in result
        assert "dte_shadow" in result
        assert "hit_rate_shadow" in result
        assert "would_pass_shadow" in result

        # Check values
        assert result["dte_used"] == 45
        assert result["hit_rate_used"] == 0.50
        assert result["dte_shadow"] == 120  # "4-8 Wochen" → 120
        assert result["hit_rate_shadow"] == 0.55
        assert result["would_pass_shadow"] is True  # 0.55 >= 0.50

    def test_shadow_fails_gate(self):
        """Test compute_final_mc_shadow when shadow fails gate."""
        class MockSim:
            def run_for_dte(self, s, days_to_expiry):
                return {
                    "simulation": {"hit_rate": 0.40}  # Below 0.50 gate
                }

        test_signal = {
            "ticker": "TEST2",
            "deep_analysis": {
                "time_to_materialization": "6 Monate"
            }
        }

        mock_sim = MockSim()
        result = compute_final_mc_shadow(
            sim=mock_sim,
            s=test_signal,
            final_dte=120,
            hit_rate=0.52,
            min_long=0.50
        )

        assert result["dte_used"] == 120
        assert result["dte_shadow"] == 140  # "6 Monate" → 140
        assert result["hit_rate_shadow"] == 0.40
        assert result["would_pass_shadow"] is False  # 0.40 < 0.50

    def test_shadow_with_no_deep_analysis(self):
        """Test compute_final_mc_shadow with missing deep_analysis."""
        class MockSim:
            def run_for_dte(self, s, days_to_expiry):
                return {
                    "simulation": {"hit_rate": 0.51}
                }

        test_signal = {"ticker": "TEST3"}  # No deep_analysis

        mock_sim = MockSim()
        result = compute_final_mc_shadow(
            sim=mock_sim,
            s=test_signal,
            final_dte=45,
            hit_rate=0.50,
            min_long=0.45
        )

        # Empty TTM should default to 120
        assert result["dte_shadow"] == 120


class TestCodeContainsFinalMcShadow:
    """Integration tests: verify code contains required elements."""

    def test_pipeline_contains_final_mc_shadow_field(self):
        """Test that pipeline.py contains 'final_mc_shadow' field references."""
        import pipeline

        # Check that _final_mc_shadow_log exists
        assert hasattr(pipeline, "_final_mc_shadow_log")
        assert isinstance(pipeline._final_mc_shadow_log, list)

        # Check that compute_final_mc_shadow function exists
        assert callable(pipeline.compute_final_mc_shadow)

    def test_pipeline_imports_ttm_to_dte_floor(self):
        """Test that pipeline.py imports ttm_to_dte_floor."""
        import pipeline

        # The import should succeed
        assert hasattr(pipeline, "ttm_to_dte_floor")
        assert callable(pipeline.ttm_to_dte_floor)

    def test_grep_final_mc_shadow_in_pipeline(self):
        """Grep-style test: verify final_mc_shadow appears in pipeline.py."""
        with open("pipeline.py", "r") as f:
            content = f.read()

        # Should contain final_mc_shadow field assignment
        assert 'final_mc_shadow' in content, "pipeline.py missing final_mc_shadow field"
        # Should contain the helper function definition
        assert 'def compute_final_mc_shadow' in content, "pipeline.py missing compute_final_mc_shadow function"
        # Should contain ttm_to_dte_floor call
        assert 'ttm_to_dte_floor(ttm)' in content, "pipeline.py missing ttm_to_dte_floor call"

    def test_grep_ttm_to_dte_floor_in_options_designer(self):
        """Grep-style test: verify ttm_to_dte_floor in options_designer.py."""
        with open("modules/options_designer.py", "r") as f:
            content = f.read()

        # Should define ttm_to_dte_floor
        assert 'def ttm_to_dte_floor' in content, "options_designer.py missing ttm_to_dte_floor definition"
        # Should have TTM_TO_DTE_MIN dict
        assert 'TTM_TO_DTE_MIN' in content, "options_designer.py missing TTM_TO_DTE_MIN"
        # Should have the correct mappings
        assert '"4-8 Wochen"' in content and '120' in content
        assert '"2-3 Monate"' in content and '120' in content
        assert '"6 Monate"' in content and '140' in content

    def test_grep_final_mc_shadow_in_options_designer(self):
        """Grep-style test: verify final_mc_shadow is passed to proposals."""
        with open("modules/options_designer.py", "r") as f:
            content = f.read()

        # Should pass final_mc_shadow from signal to proposal
        assert 'final_mc_shadow' in content, "options_designer.py missing final_mc_shadow field"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
