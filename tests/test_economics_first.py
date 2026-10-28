"""Economics-first acquisition tests: no network/LLM dependencies."""
from modules import economics_first as ef


def _c(t, rv=1.0):
    return {"ticker": t, "current_price": 100.0, "rel_volume": rv,
            "dollar_volume": 1e8, "news": ["new catalyst"]}


class _Gates:
    def __init__(self, earnings=False):
        self.earnings = earnings

    def has_upcoming_earnings(self, ticker):
        return self.earnings


class _Designer:
    def __init__(self, option=None):
        self.option = option
        self.queries = []

    def _get_iv_rank(self, ticker):
        return 30

    def _select_strategy(self, ticker, direction, iv_rank):
        return "LONG_CALL"

    def _find_option_for_dte(self, ticker, strategy, spot, low, high):
        self.queries.append((low, high))
        return self.option


def test_probe_uses_real_long_dte_not_short_precheck():
    d = _Designer({"bid": 8.0, "ask": 9.0, "dte": 180})
    result = ef.probe_candidate(_c("ABC"), designer=d, gates=_Gates(), policy=ef.DEFAULTS)
    assert result["status"] == "pass"
    assert d.queries == [(150, 365)]


def test_earnings_rejected_before_chain():
    d = _Designer()
    verdict = ef.probe_candidate(_c("ABC"), designer=d, gates=_Gates(True), policy=ef.DEFAULTS)
    assert verdict["reason"] == "earnings_gate"
    assert not d.queries


def test_expensive_executable_quote_rejected():
    d = _Designer({"bid": 1.0, "ask": 4.0, "dte": 175})
    result = ef.probe_candidate(_c("ABC"), designer=d, gates=_Gates(), policy=ef.DEFAULTS)
    assert result["reason"] == "excessive_quote_friction"


def test_unknown_quote_not_marked_economically_bad():
    d = _Designer(None)
    result = ef.probe_candidate(_c("ABC"), designer=d, gates=_Gates(), policy=ef.DEFAULTS)
    assert result["status"] == "unknown"


def test_caps_and_ledger_decisions_for_all_candidates():
    candidates = [_c("T" + str(i), rv=1 + i / 20) for i in range(40)]
    p = {**ef.DEFAULTS, "max_quote_probes": 8, "audit_quote_probes": 2,
         "max_haiku_candidates": 5, "max_sonnet_candidates": 2}
    seen = []
    def probe(c):
        seen.append(c["ticker"])
        return {"status": "pass", "reason": "quoted_economics"}
    selected, decisions, summary = ef.select_before_haiku(candidates, day="2026-10-28", probe=probe, policy=p)
    assert len(seen) == 8 and len(set(seen)) == 8
    assert len(selected) == 5
    assert len(decisions) == len(candidates)
    assert summary["deferred"] == 35
    assert any(d["reason"] == "quote_probe_budget" for d in decisions.values())
    sonnet, skipped = ef.select_before_sonnet(selected, day="2026-10-28", policy=p)
    assert len(sonnet) == 2 and len(skipped) == 3
    assert not set(x["ticker"] for x in sonnet) & set(skipped)


def test_shadow_preserves_every_candidate():
    c = [_c("A"), _c("B")]
    p = {**ef.DEFAULTS, "mode": "shadow", "max_quote_probes": 1,
         "max_haiku_candidates": 0}
    selected, decisions, summary = ef.select_before_haiku(
        c, day="2026-10-28", probe=lambda _: {"status": "reject", "reason": "earnings_gate"}, policy=p)
    assert len(selected) == len(c) and summary["mode"] == "shadow"
    assert len(decisions) == len(c)


def test_deterministic_sampling_and_no_probes_at_zero_budget():
    c = [_c("T" + str(i)) for i in range(30)]
    p = {**ef.DEFAULTS, "max_quote_probes": 0, "max_haiku_candidates": 0}
    selected, decisions, summary = ef.select_before_haiku(
        c, day="2026-10-28", probe=lambda _: (_ for _ in ()).throw(AssertionError("called")), policy=p)
    assert selected == [] and len(decisions) == 30
    p.update(max_quote_probes=8, audit_quote_probes=2, max_haiku_candidates=5)
    a = ef.select_before_haiku(c, day="2026-10-28",
           probe=lambda _: {"status": "unknown", "reason": "no_data"}, policy=p)
    b = ef.select_before_haiku(list(reversed(c)), day="2026-10-28",
           probe=lambda _: {"status": "unknown", "reason": "no_data"}, policy=p)
    assert [x["ticker"] for x in a[0]] == [x["ticker"] for x in b[0]]
