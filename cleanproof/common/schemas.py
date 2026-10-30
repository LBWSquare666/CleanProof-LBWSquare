"""Data contracts for model input/output and tasks.

The outputs of the verifier and the decision maker are both constrained and validated by the pydantic models here as JSON schema;
the schema is also used for Token Factory / vLLM structured output (response_format=json_schema).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ZoneStatus(str, Enum):
    clean = "clean"
    residue = "residue"
    uncertain = "uncertain"


class LikelyCause(str, Enum):
    insufficient_force = "insufficient_force"
    missed_coverage = "missed_coverage"
    dried_stain = "dried_stain"
    smear_from_cloth = "smear_from_cloth"
    lighting_artifact = "lighting_artifact"
    unknown = "unknown"


class StainType(str, Enum):
    """Stain type (used to look up the force table configs/force_table.yaml); likely_cause says "why it is not wiped off yet", and the two are kept separate."""

    dust = "dust"
    wet_spill = "wet_spill"
    dried_stain = "dried_stain"
    ink_mark = "ink_mark"
    grease = "grease"
    unknown = "unknown"


class Action(str, Enum):
    wipe_zone = "wipe_zone"
    spot_scrub = "spot_scrub"
    inspect_closeup = "inspect_closeup"
    ask_human = "ask_human"
    finish = "finish"


class ForcePreset(str, Enum):
    """Old three levels (run files before v8 and benchmark_v1); new decisions use Decision.force_n."""

    light = "light"
    medium = "medium"
    firm = "firm"


class Pattern(str, Enum):
    zigzag = "zigzag"
    spiral = "spiral"
    edge_trace = "edge_trace"


def _check_bbox(v: Optional[list[float]]) -> Optional[list[float]]:
    if v is None:
        return v
    if len(v) != 4:
        raise ValueError("bbox must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = v
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("bbox must be normalized coordinates inside the zone, with x0<x1 and y0<y1")
    return v


class VerifierZoneResult(BaseModel):
    """Verifier judgment for a single zone."""

    model_config = ConfigDict(extra="forbid")

    zone_id: str
    status: ZoneStatus
    confidence: float = Field(ge=0.0, le=1.0)
    residue_bbox_norm: Optional[list[float]] = None
    likely_cause: LikelyCause = LikelyCause.unknown
    stain_type: StainType = StainType.unknown
    evidence: str = Field(default="", max_length=400)

    @field_validator("residue_bbox_norm")
    @classmethod
    def _bbox(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        return _check_bbox(v)


class Decision(BaseModel):
    """Decision output: one whitelisted skill and its bounded parameters."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    zone_ids: list[str] = Field(default_factory=list, max_length=12)
    force_n: Optional[float] = Field(default=None, ge=0.0, le=50.0)   # target contact force; the valid range is checked by the guardrail and the skill server against the force table
    pattern: Optional[Pattern] = None
    passes: int = Field(default=1, ge=1, le=2)
    bbox_norm: Optional[list[float]] = None
    reason: str = Field(max_length=500)
    expected_outcome: str = Field(default="", max_length=300)

    @field_validator("bbox_norm")
    @classmethod
    def _bbox(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        return _check_bbox(v)

    @model_validator(mode="after")
    def _action_params(self) -> "Decision":
        a = self.action
        if a in (Action.wipe_zone, Action.spot_scrub, Action.inspect_closeup) and not self.zone_ids:
            raise ValueError(f"{a.value} needs zone_ids")
        if a == Action.wipe_zone and (self.force_n is None or self.pattern is None):
            raise ValueError("wipe_zone needs force_n and pattern")
        if a == Action.spot_scrub:
            if self.force_n is None or self.bbox_norm is None or len(self.zone_ids) != 1:
                raise ValueError("spot_scrub needs force_n and bbox_norm, and targets exactly one zone")
        if a == Action.inspect_closeup and len(self.zone_ids) != 1:
            raise ValueError("inspect_closeup targets exactly one zone")
        return self


class ForceSummary(BaseModel):
    """Force and coverage summary of one wipe on a zone (from the skill server)."""

    model_config = ConfigDict(extra="forbid")

    target_n: float
    mean_normal_n: float
    peak_normal_n: float
    within_1n_ratio: float = Field(ge=0.0, le=1.0)
    contact_loss_count: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    contact_duration_s: float = Field(ge=0.0)


class TaskSpec(BaseModel):
    """Structured task (parsed from a natural-language instruction, or given directly)."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = ""
    surface: str = Field(default="sealed_wood", description="surface material (key of configs/force_table.yaml surfaces), chosen by the operator at job start")
    zones: list[str] = Field(default_factory=list, description="zones to clean; empty means all")
    forbidden_zones: list[str] = Field(default_factory=list)
    max_passes_per_zone: int = Field(default=3, ge=1, le=6)
    max_steps: int = Field(default=30, ge=1, le=200)


def response_format(model: type[BaseModel], name: str | None = None, strict: bool = False) -> dict:
    """OpenAI-compatible json_schema structured-output parameters.

    vLLM-style backends (Token Factory, local vLLM) do constrained decoding from the schema; strict=True is an extra
    OpenAI requirement (all fields required) and is off by default.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or model.__name__,
            "schema": model.model_json_schema(),
            "strict": strict,
        },
    }


class VerifierBatch(BaseModel):
    """One verification request can return judgments for several zones at once."""

    model_config = ConfigDict(extra="forbid")

    zones: list[VerifierZoneResult]
