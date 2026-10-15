"""C1: News-Neuheit per Embedding – mit deterministischem Fake-Embedding (kein Modell, kein Netzwerk)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from modules import news_novelty as nn

NOW = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)


def fake_embed(texts):
    """Bag-of-words-Vektor über ein festes Vokabular, L2-normiert."""
    vocab = ["merger", "earnings", "beat", "guidance", "recall", "lawsuit", "ceo", "resigns", "fda", "approval"]
    out = []
    for t in texts:
        v = [float(t.lower().count(w)) for w in vocab]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_novelty_repeat_vs_new_story():
    past = ["Company announces merger", "Earnings beat guidance"]
    same = nn.novelty(["Merger announced by company"], past, embed_fn=fake_embed)
    new = nn.novelty(["FDA approval granted"], past, embed_fn=fake_embed)
    assert same["novelty_status"] == "ok" and same["novelty_max"] == 0.0
    assert new["novelty_max"] == 1.0 and new["n_history"] == 2


def test_novelty_none_without_history_headlines_or_model():
    assert nn.novelty(["x"], [], embed_fn=fake_embed)["novelty_max"] is None
    assert nn.novelty([], ["x"], embed_fn=fake_embed)["novelty_status"] == "no_headlines"
    r = nn.novelty(["x"], ["y"], embed_fn=lambda t: None)
    assert r["novelty_status"] == "model_unavailable" and r["novelty_mean"] is None          # nie 0


def test_memory_is_point_in_time_and_written_once(tmp_path):
    nn.remember("AAA", ["Old merger story"], datetime(2026, 10, 14, 14, tzinfo=timezone.utc), tmp_path)
    nn.remember("AAA", ["Today story"], NOW, tmp_path)
    nn.remember("AAA", ["Today story again"], NOW, tmp_path)                               # einmal je Tag
    nn.remember("BBB", ["Other ticker"], datetime(2026, 10, 14, tzinfo=timezone.utc), tmp_path)
    assert nn.history("AAA", NOW.date(), tmp_path) == ["Old merger story"]                  # heute nie Historie
    rows = [json.loads(x) for x in (tmp_path / "2026-10-15.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["headlines"] == ["Today story"]
    assert nn.history("AAA", NOW.date(), tmp_path, days=0) == []


def test_score_and_remember_with_root_and_test_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(nn, "embed", fake_embed)
    nn.remember("AAA", ["Company announces merger"], datetime(2026, 10, 13, tzinfo=timezone.utc), tmp_path)
    r = nn.score_and_remember("AAA", ["CEO resigns"], NOW, tmp_path)
    assert r["novelty_status"] == "ok" and r["novelty_max"] == 1.0
    assert (tmp_path / "2026-10-15.jsonl").exists()
    monkeypatch.chdir(tmp_path)
    assert nn.score_and_remember("AAA", ["x"])["novelty_status"] == "disabled_in_tests"
