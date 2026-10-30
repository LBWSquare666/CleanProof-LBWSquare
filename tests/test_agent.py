"""Agent closed loop: simulated scenarios (rule policy + classical vision), baseline comparison, Nemotron policy (mock Token Factory), guardrails."""

import json

import httpx
import pytest

from cleanproof.agent import guardrails
from cleanproof.agent.policy import NemotronPolicy
from cleanproof.agent.state import TaskState
from cleanproof.common.config import load_yaml
from cleanproof.common.jsonl import read_jsonl
from cleanproof.common.llm_client import LLMClient, SpendLedger
from cleanproof.common.schemas import Decision, TaskSpec, VerifierZoneResult
from cleanproof.common.zones import ZoneGrid
from cleanproof.sim.run import run_sim_scenario
from cleanproof.sim.scenarios import scenario_names


def _generic_scenarios() -> list[str]:
    """Scenarios with a registered site (the Spot home dining table job used by the v12 video) are only recorded with the official pipeline and stay out of the rule-policy regression; for the site geometry see test_site_geometry.py."""
    from cleanproof.sim.scenarios import load_scenario
    return [n for n in scenario_names() if not load_scenario(n).get("site")]


@pytest.mark.parametrize("name", _generic_scenarios())
def test_scenarios_rules_cv(name, tmp_path):
    res = run_sim_scenario(name, "rules", "cv", "topdown", run_root=tmp_path)
    j = res["judge"]
    assert j["passed"], j
    assert j["false_clean_zones"] == []


def test_fixed_baseline_reports_done_but_leaves_dried_stain(tmp_path):
    j = run_sim_scenario("S2_dried", "fixed1", "cv", "topdown", run_root=tmp_path)["judge"]
    assert j["status"] == "done" and not j["targets_truly_clean"]


def _scripted_client(tmp_path, replies):
    queue = list(replies)
    seen = []

    def handler(req):
        body_in = json.loads(req.content)
        seen.append(body_in)
        text = queue.pop(0) if queue else json.dumps({"action": "ask_human", "zone_ids": [], "reason": "script end"})
        if text is None:
            return httpx.Response(503, json={"error": "unavailable"})
        body = {"id": f"req-{len(seen)}", "object": "chat.completion", "created": 0, "model": body_in["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1500, "completion_tokens": 100, "total_tokens": 1600}}
        return httpx.Response(200, json=body, headers={"x-request-id": f"tf-{len(seen)}"})

    cfg = load_yaml("models.yaml")
    cfg["timeouts"]["max_retries"] = 0
    client = LLMClient(cfg, log_path=tmp_path / "calls.jsonl", ledger=SpendLedger(tmp_path / "spend.json", 40),
                       http_client_factory=lambda b: httpx.Client(transport=httpx.MockTransport(handler)))
    return client, seen


def wipe(z, force_n=5.0, pattern="zigzag"):
    return json.dumps({"action": "wipe_zone", "zone_ids": [z], "force_n": force_n, "pattern": pattern,
                       "passes": 1, "reason": f"residue in {z}", "expected_outcome": f"{z} clean"})


def test_nemotron_policy_runs_scenario_with_guard_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    replies = [
        json.dumps({"action": "finish", "reason": "looks done"}),   # still dirty zones → guardrail rejects → ask again
        wipe("B2"),
        wipe("C3"),
        json.dumps({"action": "finish", "reason": "all clean"}),
    ]
    client, seen = _scripted_client(tmp_path, replies)
    res = run_sim_scenario("S1_light", "nemotron", "oracle", "topdown", run_root=tmp_path, client=client)
    s, j = res["summary"], res["judge"]
    assert s["status"] == "done" and j["targets_truly_clean"]
    assert all(r["model"] == "nvidia/nemotron-3-super-120b-a12b" for r in seen)
    assert "rejected" in seen[1]["messages"][1]["content"]            # the second request carries the guardrail feedback
    events = read_jsonl(f"{res['run_dir']}/events.jsonl")
    assert any(e["kind"] == "guard_reject" for e in events)
    decisions = [e for e in events if e["kind"] == "decision"]
    assert decisions[0]["meta"]["policy"][-1]["calls"][0]["request_id"] == "tf-2"
    assert s["decision_cost_usd"] == pytest.approx(len(seen) * (1500 * 0.30 + 100 * 0.90) / 1e6)
    assert len(seen) == 4                                                # rejected + ask again + two wipes + finish


def test_nemotron_unavailable_asks_human(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    client, _ = _scripted_client(tmp_path, [None, None, None])
    res = run_sim_scenario("S1_light", "nemotron", "oracle", "topdown", run_root=tmp_path, client=client)
    # model unavailable is its own outcome (not a Staff check): there is no human request, and the reason states what to do
    assert res["summary"]["status"] == "model_unavailable"
    assert res["summary"]["human_requests"] == []
    assert "Token Factory" in res["summary"]["reason"] and "Traceback" not in res["summary"]["reason"]


def _state(forbidden=()):
    grid = ZoneGrid()
    st = TaskState(TaskSpec(zones=["B2", "B3", "C2"], forbidden_zones=list(forbidden), max_passes_per_zone=2), grid)
    st.update_inspection({
        "B2": VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9),
        "B3": VerifierZoneResult(zone_id="B3", status="clean", confidence=0.9),
        "C2": VerifierZoneResult(zone_id="C2", status="uncertain", confidence=0.5),
    }, {})
    return st


def test_guardrails():
    st = _state()
    ok = guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=5, pattern="zigzag",
                                   reason="x"), st)
    assert ok.ok
    assert not guardrails.check(Decision(action="wipe_zone", zone_ids=["B3"], force_n=5,
                                         pattern="zigzag", reason="x"), st).ok          # already clean
    assert not guardrails.check(Decision(action="wipe_zone", zone_ids=["D5"], force_n=5,
                                         pattern="zigzag", reason="x"), st).ok          # not part of the task
    assert not guardrails.check(Decision(action="finish", reason="x"), st).ok           # dirty zones remain
    st.passes["B2"] = 2
    g = guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=15, pattern="zigzag",
                                  reason="x"), st)
    assert not g.ok and g.fallback.action.value == "ask_human"                         # attempt budget used up
    st.closeup_done.add("C2")
    assert not guardrails.check(Decision(action="inspect_closeup", zone_ids=["C2"], reason="x"), st).ok


def test_guardrail_blocks_forbidden_zone():
    grid = ZoneGrid()
    st = TaskState(TaskSpec(zones=["B2", "C3"], forbidden_zones=["C3"]), grid)
    assert "C3" not in st.targets
    g = guardrails.check(Decision(action="wipe_zone", zone_ids=["C3"], force_n=5, pattern="zigzag",
                                  reason="x"), st)
    assert not g.ok and "forbidden zone" in g.reason


def test_guardrail_requires_force_escalation_after_failed_wipe():
    st = _state()
    st.zone["B2"] = VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9,
                                       residue_bbox_norm=[0.1, 0.1, 0.3, 0.3])
    st.record_wipe(["B2"], 1, {}, 5.0, "zigzag")
    # S2 measured: after a light wipe there is still residue, and it proposes light spot_scrub again → reject, replace with medium zigzag
    g = guardrails.check(Decision(action="spot_scrub", zone_ids=["B2"], force_n=5,
                                  bbox_norm=[0.1, 0.1, 0.3, 0.3], reason="x"), st)
    assert not g.ok and g.fallback.action.value == "wipe_zone" and g.fallback.force_n == 10
    assert guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=10, pattern="zigzag",
                                     reason="x"), st).ok
    # with missed_coverage, a different method at the same level is allowed, but the same method at the same level is not
    st.zone["B2"] = VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9, likely_cause="missed_coverage",
                                       residue_bbox_norm=[0.1, 0.1, 0.3, 0.3])
    assert guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=5, pattern="edge_trace",
                                     reason="x"), st).ok
    assert not guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=5, pattern="zigzag",
                                         reason="x"), st).ok
    # still residue after firm → can only ask a person
    st.record_wipe(["B2"], 1, {}, 15.0, "zigzag")
    g = guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=15, pattern="spiral",
                                  reason="x"), st)
    assert not g.ok and g.fallback.action.value == "ask_human"
    # uncertain zones are not subject to this limit (per the prompt, wipe once more at the current force level)
    st.record_wipe(["C2"], 1, {}, 10.0, "zigzag")
    assert guardrails.check(Decision(action="wipe_zone", zone_ids=["C2"], force_n=10, pattern="zigzag",
                                     reason="x"), st).ok


def test_guardrail_large_residue_needs_force_not_pattern_or_spot():
    # S2 demo take3: dried stain covers about 80% of C3, judged missed_coverage after the light wipe → must not change method at the same level, nor use spot_scrub
    st = _state()
    st.zone["B2"] = VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9, likely_cause="missed_coverage",
                                       residue_bbox_norm=[0.05, 0.05, 0.95, 0.95])
    st.record_wipe(["B2"], 1, {}, 5.0, "zigzag")
    g = guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=5, pattern="spiral",
                                  reason="x"), st)
    assert not g.ok and g.fallback.force_n == 10
    assert not guardrails.check(Decision(action="spot_scrub", zone_ids=["B2"], force_n=10,
                                         bbox_norm=[0.05, 0.05, 0.95, 0.95], reason="x"), st).ok
    assert guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=10, pattern="zigzag",
                                     reason="x"), st).ok
    # small local residue: the missed-coverage exception still holds
    st.zone["B2"] = VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9, likely_cause="missed_coverage",
                                       residue_bbox_norm=[0.0, 0.0, 0.3, 0.3])
    assert guardrails.check(Decision(action="wipe_zone", zone_ids=["B2"], force_n=5, pattern="edge_trace",
                                     reason="x"), st).ok
