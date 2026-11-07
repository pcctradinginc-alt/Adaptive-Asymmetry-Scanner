"""Audit 2026-11-07 (P7): werden Pflicht-Reviews für kritische Pfade tatsächlich durchgesetzt?

Live-Befund (GitHub-API 07.11.): main `protected: false`; #164 (config/option_learner_variants.yaml – nicht unter
CODEOWNERS), #165 (modules/research_lab.py), #166 (config/data_catalog.yaml) ohne Review gemergt. CODEOWNERS selbst:
"Wirksam erst mit Branch-Protection 'Require review from Code Owners' auf main". Neu: der tägliche Ops-Digest meldet
fehlenden Branch-Schutz und ungeprüfte Merges kritischer Pfade (nur lesend). Synthetische API-Antworten.
"""
from __future__ import annotations

from datetime import datetime, timezone

from modules import ops_report as rep
from modules import ops_workflows as ow
from tests.test_ops_workflows import API, REPO, FakeSession, Resp

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
PATTERNS = ow.codeowner_patterns("""# Kommentar
modules/research_lab.py       @owner
config/external_sources/      @owner
.github/                      @owner
/config.yaml                  @owner
""")


def _pr(n, merged_at="2026-11-07T01:00:00Z"):
    return {"number": n, "title": f"PR {n}", "merged_at": merged_at, "html_url": f"https://github.com/{REPO}/pull/{n}"}


def _routes(protected=False, reviews=None):
    reviews = reviews or {}
    files = {165: ["modules/research_lab.py", "tests/test_research_lab.py"], 164: ["modules/option_learner.py"],
             160: [".github/workflows/tests.yml"], 150: ["config.yaml"]}
    r = {f"{API}/branches/main": Resp({"name": "main", "protected": protected}),
         f"{API}/pulls": Resp([_pr(165), _pr(164), _pr(160), _pr(150, "2026-10-01T00:00:00Z"),
                               {"number": 149, "merged_at": None}])}
    for n, fs in files.items():
        r[f"{API}/pulls/{n}/files"] = Resp([{"filename": f} for f in fs])
        r[f"{API}/pulls/{n}/reviews"] = Resp(reviews.get(n, []))
    return r


def test_codeowner_matching():
    assert PATTERNS == ["modules/research_lab.py", "config/external_sources/", ".github/", "config.yaml"]
    assert ow.owned(".github/workflows/x.yml", PATTERNS) and ow.owned("config.yaml", PATTERNS)
    assert ow.owned("config/external_sources/maritime.yaml", PATTERNS)
    assert not ow.owned("config/data_catalog.yaml.bak", ["config/data_catalog.yaml"])
    assert not ow.owned("modules/option_learner.py", PATTERNS)


def test_unprotected_branch_and_unreviewed_critical_merges_are_reported():
    s = FakeSession(_routes(reviews={160: [{"state": "APPROVED"}]}))
    g = ow.governance(s, REPO, NOW, patterns=PATTERNS)
    assert g["status"] == "OK" and g["protected"] is False and g["merged_checked"] == 3   # #150 außerhalb des Fensters
    assert [u["number"] for u in g["unreviewed_codeowner_merges"]] == [165]               # #164 kein Owner-Pfad
    assert g["unreviewed_codeowner_merges"][0]["codeowner_paths"] == ["modules/research_lab.py"]
    assert all(params is None or "per_page" in params for _, params in s.calls)           # nur GET-Abfragen


def test_alert_rules_from_governance():
    g = ow.governance(FakeSession(_routes()), REPO, NOW, patterns=PATTERNS)
    alerts = {a[0]: a for a in rep._rules({"workflows": {"status": "OK", "failed": [], "governance": g}}, {}, None)}
    assert alerts["BRANCH_UNPROTECTED"][1] == "WARNING" and "nicht geschützt" in alerts["BRANCH_UNPROTECTED"][3]
    assert alerts["UNREVIEWED_CRITICAL_MERGE"][4] == (160, 165) and "#165" in alerts["UNREVIEWED_CRITICAL_MERGE"][3]
    assert rep._owner("BRANCH_UNPROTECTED") == "workflows"


def test_protected_and_reviewed_gives_no_alert():
    ok = {n: [{"state": "APPROVED"}] for n in (165, 160)}
    g = ow.governance(FakeSession(_routes(protected=True, reviews=ok)), REPO, NOW, patterns=PATTERNS)
    ids = {a[0] for a in rep._rules({"workflows": {"status": "OK", "failed": [], "governance": g}}, {}, None)}
    assert not ids & {"BRANCH_UNPROTECTED", "UNREVIEWED_CRITICAL_MERGE"}


def test_api_error_is_unavailable_and_never_raises_nor_alerts():
    g = ow.governance(FakeSession({}), REPO, NOW, patterns=PATTERNS)
    assert g["status"] == "UNAVAILABLE"
    ids = {a[0] for a in rep._rules({"workflows": {"status": "OK", "failed": [], "governance": g}}, {}, None)}
    assert "BRANCH_UNPROTECTED" not in ids
