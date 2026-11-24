from modules.production_intelligence_adapter import kill_switch_active


def test_kill_switch_fails_closed_for_missing_and_malformed_configuration(tmp_path):
    path = tmp_path / "switch.yaml"
    assert kill_switch_active(path)
    for content in ("{}", "active: null", "active: 0", "active: 'false'", "[]", "active: ["):
        path.write_text(content, encoding="utf-8")
        assert kill_switch_active(path), content
    path.write_text("active: false", encoding="utf-8")
    assert not kill_switch_active(path)
    path.write_text("active: true", encoding="utf-8")
    assert kill_switch_active(path)
