"""Stain x surface force table (configs/force_table.yaml) and the guardrail / rule policy that checks force against the table."""

import pytest

from cleanproof.agent import guardrails
from cleanproof.agent.force_table import ForceTable, decision_force_n, force_label, load_force_table
from cleanproof.agent.policy import RulePolicy
from cleanproof.agent.state import TaskState
from cleanproof.common.config import load_yaml
from cleanproof.common.schemas import Decision, TaskSpec, VerifierZoneResult
from cleanproof.common.zones import ZoneGrid


def test_table_values_used_in_the_video():
    t = load_force_table()
    assert t.min_n == 5.0 and t.hard_max_n == 15.0 and t.step_n == 5.0
    assert t.start("wet_spill", "sealed_wood") == 5.0          # chapter 1: wet coffee on a wooden board, 5 N
    assert t.start("ink_mark", "sealed_wood") == 10.0          # chapter 1: marker on a wooden board, 10 N
    assert t.start("ink_mark", "whiteboard") == 5.0            # real robot: marker on a whiteboard, 5 N
    assert [t.start("wet_spill", s) for s in ("sealed_wood", "rough_stone", "stainless_steel")] == [5.0, 8.0, 5.0]
    assert t.start("unknown", "sealed_wood") == t.start("wet_spill", "sealed_wood")
    assert (t.next(5.0, "sealed_wood"), t.next(10.0, "sealed_wood"), t.next(15.0, "sealed_wood")) == (10.0, 15.0, None)
    assert (t.next(5.0, "whiteboard"), t.next(10.0, "whiteboard")) == (10.0, None)
    assert t.next(8.0, "rough_stone") == 13.0 and t.next(13.0, "rough_stone") == 15.0


def test_table_rejects_out_of_range_cells():
    cfg = load_yaml("force_table.yaml")
    cfg["forces"]["dust"]["whiteboard"] = 3.0                  # below the control law lower limit of 5 N
    with pytest.raises(ValueError):
        ForceTable.from_cfg(cfg)
    with pytest.raises(ValueError):
        load_force_table().start("wet_spill", "carpet")


def _state(surface="sealed_wood", stain="ink_mark"):
    st = TaskState(TaskSpec(zones=["B2", "C3"], surface=surface), ZoneGrid())
    st.update_inspection({"B2": VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9, stain_type=stain),
                          "C3": VerifierZoneResult(zone_id="C3", status="clean", confidence=0.9)}, {})
    return st


def _wipe(f, z="B2", pattern="zigzag"):
    return Decision(action="wipe_zone", zone_ids=[z], force_n=f, pattern=pattern, reason="x")


def test_guard_first_wipe_follows_table_and_surface_limit():
    st = _state()
    assert guardrails.check(_wipe(10), st).ok                   # marker on a wooden board starts at 10 N
    assert guardrails.check(_wipe(5), st).ok                    # lighter is also fine
    g = guardrails.check(_wipe(15), st)
    assert not g.ok and "starts at 10" in g.reason             # may not use heavy force right away
    assert not guardrails.check(_wipe(4), st).ok                # below 5 N
    wb = _state("whiteboard")
    assert guardrails.check(_wipe(5), wb).ok and not guardrails.check(_wipe(10), wb).ok


def test_guard_escalation_is_exactly_one_step_then_ask_human():
    st = _state(stain="wet_spill")
    st.record_wipe(["B2"], 1, {}, 5.0, "zigzag")
    g = guardrails.check(_wipe(5), st)
    assert not g.ok and g.fallback.force_n == 10.0             # still dirty: must go 5 → 10
    assert not guardrails.check(_wipe(15), st).ok               # no skipping levels
    assert guardrails.check(_wipe(10), st).ok
    st.record_wipe(["B2"], 1, {}, 10.0, "zigzag")
    assert guardrails.check(_wipe(15), st).ok
    st.record_wipe(["B2"], 1, {}, 15.0, "zigzag")
    st.passes["B2"] = 0
    g = guardrails.check(_wipe(15, pattern="spiral"), st)
    assert not g.ok and g.fallback.action.value == "ask_human"  # wooden board limit of 15 N and still dirty → ask a person


def test_rule_policy_walks_the_table():
    st = _state(stain="wet_spill")
    pol = RulePolicy()
    assert pol.decide(st).decision.force_n == 5.0
    st.record_wipe(["B2"], 1, {}, 5.0, "zigzag")
    assert pol.decide(st).decision.force_n == 10.0
    st.record_wipe(["B2"], 1, {}, 15.0, "zigzag")
    st.passes["B2"] = 0
    assert pol.decide(st).decision.action.value == "ask_human"


def test_legacy_run_files_still_read_as_newtons():
    assert decision_force_n({"force_preset": "medium"}) == 10.0
    assert decision_force_n({"force_n": 8}) == 8.0
    assert force_label({"force_preset": "firm"}) == "15 N" and force_label({}) == ""


def test_rule_policy_cleans_other_zones_before_asking_staff():
    """When one cell cannot be wiped clean (still dirty at the surface limit), first wipe the other cells that can still be handled; only when no dirty cell can be handled, hand over to a person, listing them all together."""
    st = TaskState(TaskSpec(zones=["B2", "C3"], surface="sealed_wood"), ZoneGrid())
    st.update_inspection({"B2": VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9, stain_type="ink_mark"),
                          "C3": VerifierZoneResult(zone_id="C3", status="residue", confidence=0.9, stain_type="wet_spill")}, {})
    pol = RulePolicy()
    st.record_wipe(["B2"], 1, {}, 15.0, "zigzag")                # B2 was already wiped at the wooden board limit of 15 N and is still dirty
    d = pol.decide(st).decision
    assert d.action.value != "ask_human" and d.zone_ids == ["C3"] and d.force_n == 5.0
    st.record_wipe(["C3"], 1, {}, 15.0, "zigzag")                # C3 also hit the limit and is still dirty → both cells are handed to a person together
    d = pol.decide(st).decision
    assert d.action.value == "ask_human" and sorted(d.zone_ids) == ["B2", "C3"]


def test_site_override_is_clamped_with_warning():
    from cleanproof.agent.force_table import clamp_site_limits
    base = load_yaml("force_table.yaml")
    cfg = load_yaml("force_table.yaml")
    cfg["hard_max_n"] = 100.0
    cfg["surfaces"]["sealed_wood"]["max_n"] = 100.0           # above the global limit
    cfg["surfaces"]["glass"]["max_n"] = 1.0                   # below the largest starting force of that surface
    cfg["escalate_step_n"] = 50.0
    warns = clamp_site_limits(cfg, base)
    assert len(warns) == 4 and all("using" in w for w in warns)
    t = ForceTable.from_cfg(cfg)
    assert t.hard_max_n == 15.0 and t.cap("sealed_wood") == 15.0 and t.step_n == 5.0
    assert t.cap("glass") == max(t.forces[s]["glass"] for s in t.forces)
    assert clamp_site_limits(load_yaml("force_table.yaml"), base) == []     # defaults do not trigger it


def test_loader_clamps_hand_edited_site_file(tmp_path, monkeypatch, caplog):
    import logging
    from cleanproof.agent import force_table as ft
    from cleanproof.common import config
    d = tmp_path / "x"
    d.mkdir()
    (d / "force_table.yaml").write_text("surfaces:\n  sealed_wood: {max_n: 100}\n", encoding="utf-8")
    monkeypatch.setattr(config, "site_dir", lambda site=None: d if site == "x" else None)
    ft.reload_force_table()
    try:
        with caplog.at_level(logging.WARNING):
            t = ft.load_force_table("x")
        assert t.cap("sealed_wood") == 15.0
        assert "sealed_wood" in caplog.text and "100" in caplog.text
    finally:
        ft.reload_force_table()
