"""Guard (Audit 2026-09-29): keine NEUEN stillen except-Handler (nur pass /
continue ohne Logging). Jeder bestehende Eintrag unten wurde einzeln geprüft:
Parse-Schleifen über optionale Einträge, Konfigurations-/Git-Fallbacks,
Temp-Datei-Aufräumen, Forschungs-Nebenwerte. Wer einen neuen stillen Handler
einführt, muss ihn entweder loggen/als Status markieren oder hier begründet
eintragen."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REVIEWED_SILENT = {
    ("modules/alpha_sources.py", "_fetch_skew_tradier"),        # Verfallsdatum-Parse je Eintrag
    ("modules/alpha_sources.py", "_fetch_skew_yfinance"),       # dito + optionales skew_25d
    ("modules/alpha_sources.py", "estimate_dealer_gamma"),      # Verfallsdatum-Parse
    ("modules/candidate_ledger.py", "_atomic_write_jsonl"),     # Temp-Datei aufräumen, Fehler wird re-raised
    ("modules/candidate_ledger.py", "_compute_model_ids"),      # Modell-Hash optional (Provenienz)
    ("modules/candidate_ledger.py", "_detect_pipeline_version"),  # Fallback "unknown"
    ("modules/candidate_ledger.py", "_fetch_history_batch"),    # je Ticker; fehlend -> Outcome später
    ("modules/candidate_ledger.py", "_fetch_history_open_batch"),
    ("modules/candidate_ledger.py", "_fetch_prices_batch"),
    ("modules/candidate_ledger.py", "_fill_next_open_entries"),  # gap_at_entry optional
    ("modules/candidate_ledger.py", "_resolve_hypo_iv"),        # Fallback-Kette, Quelle wird mitgeschrieben
    ("modules/candidate_ledger.py", "flush"),                   # Dedup-Schlüssel alter Zeilen
    ("modules/candidate_ledger.py", "summarize"),               # reine Auswertung
    ("modules/data_ingestion.py", "run"),                       # yfinance-Crumb-Warmup
    ("modules/external/archive.py", "_config_hash"),
    ("modules/external/archive.py", "_git_sha"),
    ("modules/external/archive.py", "_manifest_entries_by_source_day"),
    ("modules/external/archive.py", "_read_offloaded_jsonl"),   # Cache-Miss -> Neuladen
    ("modules/external/features.py", "_states_config"),
    ("modules/external/http.py", "fetch"),                      # nur Fehlertext-Detail
    ("modules/external/orchestrator.py", "run_ingestion"),      # loggt via _log(), Health=FAIL
    ("modules/external/registry.py", "_readiness_config"),
    ("modules/external/sources/weather.py", "_forecast_days"),
    ("modules/external/sources/weather.py", "configured_user_agent_contact"),
    ("modules/external/sources/weather.py", "fetch"),           # Manifest/Advisory optional
    ("modules/factor_monitor.py", "load_regimes"),              # unlesbarer Tagesreport -> kein Regime
    ("modules/market_snapshot.py", "fetch_term_iv_point"),
    ("modules/market_snapshot.py", "select_contract"),
    ("modules/mirofish_simulation.py", "preload_hist_params"),  # Fehler je Ticker loggt _get_hist_params
    ("modules/mirofish_simulation.py", "run_for_dte"),          # danach Preis-Check mit Warnung
    ("modules/options_designer.py", "_term_structure_yfinance"),
    ("pipeline.py", "main"),                                    # reject(...) vor continue
}


def _silent_handlers() -> set[tuple[str, str]]:
    found = set()
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "outputs/", ".")) or "__pycache__" in rel:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.ExceptHandler):
                continue
            if n.type is not None and "Exception" not in ast.unparse(n.type):
                continue
            silent = all(isinstance(b, (ast.Pass, ast.Continue)) for b in n.body)
            if not silent:
                continue
            fn = n
            while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parents[fn]
            found.add((rel, getattr(fn, "name", "<module>")))
    return found


def test_no_new_silent_exception_handlers():
    new = _silent_handlers() - REVIEWED_SILENT
    assert not new, f"Neue stille except-Handler (loggen oder begründet aufnehmen): {sorted(new)}"


def test_corrupt_ledger_lines_survive_outcome_rewrite(tmp_path, monkeypatch):
    from modules import candidate_ledger as cl
    f = tmp_path / "2026-09.jsonl"
    good = {"date": "2026-09-01", "ticker": "AAA", "direction": "BULLISH", "entry_price": 100.0,
            "entry_basis": "quote_mid", "outcomes": {}}
    f.write_text(json.dumps(good) + "\n{kaputt\n")
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, d: {
        "AAA": [(__import__("datetime").datetime(2026, 9, 1) + __import__("datetime").timedelta(days=i), 100.0 + i)
                for i in range(15)]})
    cl._update_outcomes_in_file(f, __import__("datetime").datetime(2026, 9, 12))
    lines = f.read_text().splitlines()
    assert "{kaputt" in lines                       # nicht verloren
    assert json.loads(lines[0])["outcomes"].get("ret_5d") is not None
