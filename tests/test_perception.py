import json
import os

import httpx
import numpy as np
import pytest

from cleanproof.common.config import load_yaml
from cleanproof.common.llm_client import LLMClient, SpendLedger
from cleanproof.common.schemas import VerifierZoneResult
from cleanproof.perception.residue_cv import ResidueDetector, ZoneResidue
from cleanproof.perception.sources import SimCameraSource
from cleanproof.perception.verifier import (
    CVVerifier,
    FusionThresholds,
    OracleVerifier,
    VLMVerifier,
    fuse,
)
from cleanproof.robot.backends.mujoco_backend import MujocoRobot


def topdown_setup():
    r = MujocoRobot(render=False)
    src = SimCameraSource(r, r.grid, mode="topdown")
    det = ResidueDetector(r.grid, reference_rgb=src.rectified())
    return r, src, det


def test_cv_detects_stain_in_right_zone_with_bbox():
    r, src, det = topdown_setup()
    z = r.grid["C3"]
    r.dirt.add_blob("light", (z.xmin + 0.012, z.center[1]), 0.008, seed=2)
    res = det.analyze(src.rectified())
    dirty = [k for k, v in res.items() if v.residue_fraction > 0.005]
    assert dirty == ["C3"]
    u0, v0, u1, v1 = res["C3"].bbox_norm
    assert u0 < 0.1 and u1 < 0.45          # near the minimum x edge (above the row)
    assert 0.25 < v0 < 0.5 < v1 < 0.75     # centered in the y direction


def test_cv_verifier_and_oracle_agree_on_sim():
    r, src, det = topdown_setup()
    r.dirt.add_blob("dried", r.grid["B2"].center, 0.015, seed=4)
    zones = ["B2", "B3", "D5"]
    cvv = CVVerifier(r.grid, det).inspect(zones, src.rectified())
    orc = OracleVerifier(r, r.grid).inspect(zones)
    for zid in zones:
        assert cvv.status(zid) == orc.status(zid)
    assert orc.results["B2"].likely_cause.value == "dried_stain"


def test_fusion_never_upgrades_conflict_to_clean():
    th = FusionThresholds()
    vlm_clean = VerifierZoneResult(zone_id="A1", status="clean", confidence=0.9, evidence="looks clean")
    strong = ZoneResidue("A1", 0.05, [0.1, 0.1, 0.3, 0.3], 20.0)
    out, tag = fuse(vlm_clean, strong, th)
    assert out.status.value == "uncertain" and tag == "fused_conflict" and out.residue_bbox_norm
    weak_res = VerifierZoneResult(zone_id="A1", status="residue", confidence=0.4, evidence="maybe")
    none = ZoneResidue("A1", 0.0, None, 1.0)
    assert fuse(weak_res, none, th)[0].status.value == "uncertain"
    force = {"target_n": 10.0, "mean_normal_n": 6.0, "within_1n_ratio": 0.3, "contact_loss_count": 0,
             "coverage_ratio": 1.0}
    res = VerifierZoneResult(zone_id="A1", status="residue", confidence=0.9, evidence="streak")
    assert fuse(res, strong, th, force)[0].likely_cause.value == "insufficient_force"


def _mock_client(tmp_path, reply, captured):
    def handler(req):
        captured.append(json.loads(req.content))
        if reply is None:
            return httpx.Response(503, json={"error": "down"})
        body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020}}
        return httpx.Response(200, json=body)

    cfg = load_yaml("models.yaml")
    cfg["timeouts"]["max_retries"] = 0
    return LLMClient(cfg, log_path=tmp_path / "calls.jsonl", ledger=SpendLedger(tmp_path / "s.json", 40),
                     http_client_factory=lambda b: httpx.Client(transport=httpx.MockTransport(handler)))


def test_vlm_verifier_messages_and_fusion(tmp_path):
    r, src, det = topdown_setup()
    r.dirt.add_blob("light", r.grid["C3"].center, 0.015, seed=1)
    after = src.rectified()
    reply = json.dumps({"zones": [
        {"zone_id": "C3", "status": "clean", "confidence": 0.8, "residue_bbox_norm": None,
         "likely_cause": "unknown", "evidence": "surface looks uniform"},
        {"zone_id": "C4", "status": "clean", "confidence": 0.9, "residue_bbox_norm": None,
         "likely_cause": "unknown", "evidence": "clean"}]})
    captured = []
    v = VLMVerifier(r.grid, det, _mock_client(tmp_path, reply, captured))
    out = v.inspect(["C3", "C4"], after, before=after, force={"C3": {
        "target_n": 5.0, "mean_normal_n": 5.0, "within_1n_ratio": 1.0, "contact_loss_count": 0, "coverage_ratio": 1.0}})
    content = captured[0]["messages"][1]["content"]
    kinds = [p["type"] for p in content]
    assert kinds == ["image_url"] * 6 + ["text"]          # media before text (Cosmos / Qwen3-VL convention); AFTER, BEFORE, REFERENCE for each cell
    assert content[-1]["text"].count("REFERENCE (clean surface)") == 2
    assert "C3" in content[-1]["text"] and "last_wipe" in content[-1]["text"]
    assert captured[0]["model"] == load_yaml("models.yaml")["roles"]["verifier"]["model"]
    assert out.status("C3") == "uncertain" and out.provenance["C3"] == "fused_conflict"   # VLM missed it, CV caught it
    assert out.status("C4") == "clean" and out.provenance["C4"] == "vlm"


def test_vlm_verifier_falls_back_to_cv_when_server_down(tmp_path):
    r, src, det = topdown_setup()
    r.dirt.add_blob("light", r.grid["A1"].center, 0.015, seed=1)
    v = VLMVerifier(r.grid, det, _mock_client(tmp_path, None, []))
    out = v.inspect(["A1", "A2"], src.rectified())
    assert out.provenance == {"A1": "cv_fallback", "A2": "cv_fallback"}
    assert out.status("A1") == "residue" and out.status("A2") == "clean"
    assert "error" in out.llm["calls"][0]


def _gl() -> bool:
    return bool(os.environ.get("MUJOCO_GL")) and (os.environ.get("MUJOCO_GL") != "glfw" or bool(os.environ.get("DISPLAY")))


@pytest.mark.render
@pytest.mark.skipif(not _gl(), reason="needs MUJOCO_GL")
def test_rectified_render_detects_stain_zone():
    r = MujocoRobot()
    r.move_j(r.grid.survey_q, 0.5, 1.0)
    src = SimCameraSource(r, r.grid, mode="render")
    det = ResidueDetector(r.grid, reference_rgb=src.rectified())
    r.dirt.add_blob("dried", r.grid["B4"].center, 0.015, seed=9)
    res = det.analyze(src.rectified())
    dirty = sorted(k for k, v in res.items() if v.residue_fraction > 0.005)
    assert dirty == ["B4"]
    r.close()
