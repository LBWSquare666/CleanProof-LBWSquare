import pytest
from pydantic import ValidationError

from cleanproof.common.schemas import Decision, VerifierZoneResult, response_format


def test_decision_requires_params_per_action():
    Decision(action="finish", reason="all clean")
    Decision(action="ask_human", reason="stubborn stain", zone_ids=["C3"])
    with pytest.raises(ValidationError):
        Decision(action="wipe_zone", zone_ids=["A1"], reason="x")  # missing force_preset / pattern
    with pytest.raises(ValidationError):
        Decision(action="spot_scrub", zone_ids=["A1", "A2"], force_n=15,
                 bbox_norm=[0.1, 0.1, 0.2, 0.2], reason="x")
    with pytest.raises(ValidationError):
        Decision(action="inspect_closeup", zone_ids=[], reason="x")


def test_bbox_and_ranges():
    VerifierZoneResult(zone_id="A1", status="residue", confidence=0.8, residue_bbox_norm=[0, 0, 0.5, 0.5])
    with pytest.raises(ValidationError):
        VerifierZoneResult(zone_id="A1", status="residue", confidence=1.2)
    with pytest.raises(ValidationError):
        VerifierZoneResult(zone_id="A1", status="residue", confidence=0.5, residue_bbox_norm=[0.5, 0, 0.4, 1])
    with pytest.raises(ValidationError):
        Decision(action="wipe_zone", zone_ids=["A1"], force_n=5, pattern="zigzag", passes=3, reason="x")


def test_extra_fields_forbidden_and_schema_export():
    with pytest.raises(ValidationError):
        Decision(action="finish", reason="x", velocity=0.5)  # cloud must not emit continuous motion quantities
    rf = response_format(Decision)
    assert rf["type"] == "json_schema" and "properties" in rf["json_schema"]["schema"]
