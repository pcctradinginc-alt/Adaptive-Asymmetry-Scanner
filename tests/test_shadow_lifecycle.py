"""Shadow-Lifecycle (2026-10-09): status-/horizontbasierte Retention statt FIFO-300.
Kein Schatten-Trade darf vor Auflösung aller Horizonte verloren gehen oder archiviert werden."""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta

import pytest

import feedback
from modules import shadow_ledger as sl

D0 = date(2026, 8, 1)


def snap(i, day=D0, reason="roi_gate", fail_gates=None, price=100.0):
    return {"ticker": f"T{i:04d}", "entry_date": day.isoformat(), "reject_reason": reason,
            "strategy": "LONG_CALL", "option": {"strike": 100, "expiry": "2027-01-15", "ask": 5.0},
            "simulation": {"current_price": price, "hit_rate": 0.7}, "features": {"impact": 5},
            "fail_gates": fail_gates, "deep_analysis": {"direction": "BULLISH"}, "outcome": None}


def closes_fn(path_by_day=None, fail=False):
    """Schlusskurse ab entry: +1 % je Tag (oder Ausfall)."""
    calls = []

    def fn(ticker, start, end):
        calls.append((ticker, start, end))
        if fail:
            raise ConnectionError("provider down")
        s, e = date.fromisoformat(start), date.fromisoformat(end) + timedelta(days=30)   # liefert MEHR als bis end
        return [((s + timedelta(days=k)).isoformat(), 100.0 * (1 + 0.01 * k)) for k in range((e - s).days + 1)]
    fn.calls = calls
    return fn


def live_fn(value=0.25, method="option_quote"):
    return lambda snapshot: (value, method)


def run_feedback(history, today, tmp_path, price=None, live=None):
    return feedback.evaluate_shadow_trades(history, datetime.combine(today, datetime.min.time()),
                                           ledger_dir=tmp_path / "ledger", archive_path=tmp_path / "archive.jsonl",
                                           price_history_fn=price or closes_fn(), live_outcome_fn=live or live_fn())


def test_more_than_300_pending_none_lost(tmp_path):
    hist = {"shadow_trades": [snap(i, D0 + timedelta(days=i // 40)) for i in range(450)]}
    run_feedback(hist, D0 + timedelta(days=12), tmp_path)                    # nichts fällig
    assert len(hist["shadow_trades"]) == 450                                   # Ansicht behält alle ungeklärten
    h = sl.health(D0 + timedelta(days=12), tmp_path / "ledger", tmp_path / "archive.jsonl", hist["shadow_trades"])
    assert h["records"] == 450 and h["lost_before_evaluation"] == 0 and h["status"]["PENDING"] == 450
    assert not (tmp_path / "archive.jsonl").exists()


def test_long_horizon_beyond_old_fifo_capacity_is_evaluated(tmp_path):
    hist = {"shadow_trades": [snap(i, D0 + timedelta(days=i // 40)) for i in range(450)]}
    run_feedback(hist, D0 + timedelta(days=12), tmp_path)
    first = hist["shadow_trades"][0]
    run_feedback(hist, D0 + timedelta(days=46), tmp_path)                    # 45 T für die ersten fällig
    assert first["outcome"] == 0.25 and first["outcome_method"] == "option_quote"   # Legacy-Feld wie bisher
    # erst NACH Auflösung des Legacy-Horizonts aus der Ansicht exportiert; ungeklärte bleiben
    exported = [json.loads(l) for l in (tmp_path / "archive.jsonl").read_text().splitlines()]
    assert exported and all(e["outcome"] is not None for e in exported) and first["ticker"] in {e["ticker"] for e in exported}
    assert all(t["outcome"] is not None or t not in exported for t in hist["shadow_trades"])
    recs = sl._records(tmp_path / "ledger")
    ix = sl._index(sl._events(tmp_path / "ledger"))
    sid = sl.shadow_id(first)
    assert sl.horizon_state(recs[sid], ix.get(sid), 45, D0 + timedelta(days=46)) == "EVALUATED"


def test_pending_cannot_be_archived(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    with pytest.raises(sl.ShadowIntegrityError):
        sl.archive([s], D0 + timedelta(days=5), tmp_path)
    with pytest.raises(sl.ShadowIntegrityError):
        feedback.archive_shadow_trades([s], tmp_path / "a.jsonl", ledger_dir=tmp_path, today=D0 + timedelta(days=5))
    assert not (tmp_path / "a.jsonl").exists()
    with pytest.raises(sl.ShadowIntegrityError):                              # nicht registriert -> nie verdrängen
        feedback.archive_shadow_trades([snap(2)], tmp_path / "a.jsonl", ledger_dir=tmp_path, today=D0)


def test_partial_horizons_stay_unresolved(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    t = D0 + timedelta(days=21)
    sl.run_worker(t, closes_fn(), live_fn(), tmp_path)
    recs, ix = sl._records(tmp_path), sl._index(sl._events(tmp_path))
    sid = sl.shadow_id(s)
    assert sl.record_status(recs[sid], ix.get(sid), t) == "PARTIALLY_EVALUATED"
    assert not sl.is_resolved(s, tmp_path, t)
    with pytest.raises(sl.ShadowIntegrityError):
        sl.archive([s], t, tmp_path)


def test_full_evaluation_allows_archive(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    for d in (21, 46, 61):
        sl.run_worker(D0 + timedelta(days=d), closes_fn(), live_fn(), tmp_path)
    t = D0 + timedelta(days=61)
    recs, ix = sl._records(tmp_path), sl._index(sl._events(tmp_path))
    assert sl.record_status(recs[sl.shadow_id(s)], ix.get(sl.shadow_id(s)), t) == "EVALUATED"
    assert sl.archive([s], t, tmp_path) == 1
    ix = sl._index(sl._events(tmp_path))
    assert sl.record_status(recs[sl.shadow_id(s)], ix.get(sl.shadow_id(s)), t) == "ARCHIVED"


def test_missing_data_retries_without_deletion(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    t = D0 + timedelta(days=21)
    for k in range(3):
        sl.run_worker(t + timedelta(days=k), closes_fn(fail=True), live_fn(), tmp_path)
    ev = [e for e in sl._events(tmp_path) if e["horizon"] == 20]
    assert [e["kind"] for e in ev] == ["RETRY"] * 3 and ev[-1]["retry_count"] == 3
    assert ev[-1]["last_error"] and ev[-1]["next_attempt"]
    recs, ix = sl._records(tmp_path), sl._index(sl._events(tmp_path))
    assert sl.record_status(recs[sl.shadow_id(s)], ix.get(sl.shadow_id(s)), t) == "MATURED_RETRY_REQUIRED"
    assert sl.shadow_id(s) in recs                                             # Record bleibt


def test_permanent_missing_becomes_unavailable_not_invented(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    for k in range(sl.MAX_RETRIES + 2):
        sl.run_worker(D0 + timedelta(days=21 + 3 * k), closes_fn(fail=True), live_fn(), tmp_path)
    final = [e for e in sl._events(tmp_path) if e["horizon"] == 20 and e["kind"] == "UNAVAILABLE"]
    assert len(final) == 1 and "nicht verfügbar" in final[0]["reason"]
    assert "underlying_return" not in final[0] and "option_return" not in final[0]


def test_incomplete_snapshot_unavailable(tmp_path):
    s = snap(1)
    s["entry_date"] = None
    sl.register([s], "test", D0, tmp_path)
    sl.run_worker(D0, closes_fn(), live_fn(), tmp_path)
    assert {e["kind"] for e in sl._events(tmp_path)} == {"UNAVAILABLE"}


def test_worker_idempotent(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    t = D0 + timedelta(days=61)
    sl.run_worker(t, closes_fn(), live_fn(), tmp_path)
    n1 = len(sl._events(tmp_path))
    sl.run_worker(t, closes_fn(), live_fn(), tmp_path)
    sl.register([s], "test", D0, tmp_path)
    assert len(sl._events(tmp_path)) == n1 == 3 and len(sl._records(tmp_path)) == 1


def test_crash_during_evaluation_keeps_record(tmp_path):
    s = snap(1)
    sl.register([s], "test", D0, tmp_path)
    before = copy.deepcopy(sl._records(tmp_path))

    def boom(*_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        sl.run_worker(D0 + timedelta(days=61), boom, live_fn(), tmp_path)
    assert sl._records(tmp_path) == before and sl._events(tmp_path) == []
    ev = tmp_path / "events" / "2026-09.jsonl"                                # abgebrochene letzte Zeile
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text('{"shadow_id": "x", "kind": "EVAL')
    sl.run_worker(D0 + timedelta(days=61), closes_fn(), live_fn(), tmp_path)
    assert len([e for e in sl._events(tmp_path) if e.get("kind") == "EVALUATED"]) == 3


@pytest.mark.parametrize("fg,group", [({"Long-Term": "edge"}, "ROI_ONLY"), ({"Long-Term": "roi_initial", "Mid": "mc_pnl"}, "ROI_ONLY"),
                                      ({"Long-Term": "edge", "Mid": "open_interest"}, "MULTI_GATE"),
                                      ({"Long-Term": "spread"}, "LIQUIDITY_ONLY"),
                                      ({"Long-Term": "immediate_liquidation_loss"}, "EXECUTION_ONLY"),
                                      (None, "ROI_GATE_UNSPECIFIED")])
def test_gate_attribution_preserved(tmp_path, fg, group):
    s = snap(1, fail_gates=fg)
    sl.register([s], "test", D0, tmp_path)
    rec = sl._records(tmp_path)[sl.shadow_id(s)]
    assert rec["gate_group"] == group and rec["fail_gates"] == fg and rec["snapshot"]["fail_gates"] == fg


def test_recovery_of_archived_unresolved(tmp_path):
    archived = [snap(1), {**snap(2), "outcome": -0.4, "outcome_method": "option_quote", "close_date": "2026-09-15"}]
    arch = tmp_path / "archive.jsonl"
    arch.write_text("".join(json.dumps(a) + "\n" for a in archived))
    hist = {"shadow_trades": []}
    run_feedback(hist, D0 + timedelta(days=10), tmp_path)
    h = sl.health(D0 + timedelta(days=10), tmp_path / "ledger", arch, [])
    assert h["records"] == 2 and h["lost_before_evaluation"] == 0 and h["recovered_records"] == 2
    run_feedback(hist, D0 + timedelta(days=46), tmp_path)                    # 45 T fällig
    ev = {(e["shadow_id"], e["horizon"]): e for e in sl._events(tmp_path / "ledger") if e["kind"] == "EVALUATED"}
    e2 = ev[(sl.shadow_id(archived[1]), 45)]
    assert e2["option_return"] == -0.4 and e2["outcome_quality"] == "LEGACY_RECOVERED"   # Original übernommen
    assert e2["underlying_return"] is not None                                # zusätzlich Underlying (PIT)


def test_pit_snapshot_unchanged_and_no_lookahead(tmp_path):
    s = snap(1)
    orig = copy.deepcopy(s)
    sl.register([s], "test", D0, tmp_path)
    fn = closes_fn()                                                          # liefert 30 T über Fälligkeit hinaus
    sl.run_worker(D0 + timedelta(days=80), fn, live_fn(), tmp_path)
    assert sl._records(tmp_path)[sl.shadow_id(s)]["snapshot"] == orig
    e20 = next(e for e in sl._events(tmp_path) if e["horizon"] == 20)
    assert e20["last_close_date"] == (D0 + timedelta(days=20)).isoformat()   # nichts nach Fälligkeit
    assert e20["underlying_return"] == pytest.approx(0.20) and e20["mfe"] == pytest.approx(0.20)
    assert e20["option_return"] is None and e20["outcome_quality"] == "UNDERLYING_ONLY"   # zu spät für Live-Quote


def test_lost_before_evaluation_degrades_learning_health(tmp_path):
    from modules import learning_health as lh
    out = tmp_path / "outputs"
    (out / "intelligence").mkdir(parents=True)
    (out / "history.json").write_text(json.dumps({"closed_trades": [], "active_trades": [], "shadow_trades": []}))
    (out / "shadow_trades_archive.jsonl").write_text(json.dumps(snap(1)) + "\n")   # verdrängt, nicht im Ledger
    r = lh.assess(tmp_path, datetime(2026, 8, 10))
    assert r["paths"]["shadow_lifecycle"]["status"] == "DEGRADED" and r["overall"] == "DEGRADED"
    sl.register([snap(1)], "test", D0, out / "intelligence" / "shadow_ledger")
    r = lh.assess(tmp_path, datetime(2026, 8, 10))
    assert r["paths"]["shadow_lifecycle"]["status"] != "DEGRADED"


def test_gate_learning_need_more_data(tmp_path):
    sl.register([snap(i, fail_gates={"Long-Term": "edge"}) for i in range(5)], "test", D0, tmp_path)
    sl.run_worker(D0 + timedelta(days=46), closes_fn(), live_fn(), tmp_path)
    g = sl.gate_learning(D0 + timedelta(days=46), tmp_path)["groups"]["ROI_ONLY"]
    assert g == {"n": 5, "status": "NEED_MORE_DATA"}
