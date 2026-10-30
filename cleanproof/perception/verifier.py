"""Zone verifier: fused judgment from the VLM (Cosmos, local vLLM) + classical-vision residue + force summary.

Three implementations of the same interface inspect(zone_ids, ...) -> VerifierOutput:
- VLMVerifier: calls the verifier role of llm_client, fuses the structured output with classical vision;
- CVVerifier: classical vision only (the degraded mode without a GPU, also the evaluation baseline);
- OracleVerifier: simulation ground truth (reads the stain map), used only to test decision logic and generate simulation labels, clearly labeled in the UI and logs.

Fusion principle (against the common tendency of VLM detectors to say "success"): prefer uncertain, do not call clean lightly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..common.config import load_yaml
from ..common.llm_client import LLMClient, image_part
from ..common.schemas import LikelyCause, StainType, VerifierBatch, VerifierZoneResult, ZoneStatus
from ..common.zones import ZoneGrid
from .rectify import DEFAULT_PX_PER_M, zone_crop
from .residue_cv import ResidueDetector, ZoneResidue
from .sources import encode_jpeg

PROMPT_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"


@dataclass
class FusionThresholds:
    cv_residue: float = 0.005     # CV residue fraction above this: candidate residue
    cv_clean: float = 0.001       # below this: CV considers it clean
    cv_strong: float = 0.02       # VLM says clean but CV is above this: conflict → uncertain
    vlm_weak_conf: float = 0.6    # VLM says residue but CV≈0 and confidence below this → uncertain


@dataclass
class VerifierOutput:
    results: dict[str, VerifierZoneResult]
    provenance: dict[str, str] = field(default_factory=dict)   # vlm | fused_conflict | cv_fallback | cv | oracle
    cv: dict[str, ZoneResidue] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, dict[str, Any]] = field(default_factory=dict)   # raw judgments of the VLM before fusion (for evaluating misjudgment types)

    def status(self, zone_id: str) -> str:
        return self.results[zone_id].status.value


def _cause_from_force(force: Optional[dict[str, Any]]) -> Optional[LikelyCause]:
    if not force:
        return None
    if force.get("coverage_ratio", 1.0) < 0.9:
        return LikelyCause.missed_coverage
    if force.get("within_1n_ratio", 1.0) < 0.6 or force.get("mean_normal_n", 0) < force.get("target_n", 0) - 1.0:
        return LikelyCause.insufficient_force
    return None


def cv_judgement(zone_id: str, r: ZoneResidue, th: FusionThresholds, force: Optional[dict[str, Any]] = None) -> VerifierZoneResult:
    if r.residue_fraction > th.cv_residue:
        status, conf = ZoneStatus.residue, min(0.95, 0.6 + 10 * r.residue_fraction)
    elif r.residue_fraction < th.cv_clean:
        status, conf = ZoneStatus.clean, 0.7
    else:
        status, conf = ZoneStatus.uncertain, 0.5
    cause = LikelyCause.unknown
    if status != ZoneStatus.clean:
        cause = _cause_from_force(force) or LikelyCause.unknown
    return VerifierZoneResult(
        zone_id=zone_id, status=status, confidence=round(conf, 3),
        residue_bbox_norm=r.bbox_norm if status != ZoneStatus.clean else None, likely_cause=cause,
        evidence=f"classical vision: {100 * r.residue_fraction:.2f}% of zone differs from clean reference",
    )


def fuse(vlm: VerifierZoneResult, r: ZoneResidue, th: FusionThresholds,
         force: Optional[dict[str, Any]] = None) -> tuple[VerifierZoneResult, str]:
    out = vlm.model_copy()
    tag = "vlm"
    if vlm.status == ZoneStatus.clean and r.residue_fraction > th.cv_strong:
        out.status = ZoneStatus.uncertain
        out.confidence = min(vlm.confidence, 0.5)
        out.residue_bbox_norm = r.bbox_norm
        out.evidence = (f"VLM says clean but classical vision sees {100 * r.residue_fraction:.1f}% changed pixels; "
                        f"VLM: {vlm.evidence}")[:400]
        tag = "fused_conflict"
    elif vlm.status == ZoneStatus.residue and r.residue_fraction < th.cv_clean and vlm.confidence < th.vlm_weak_conf:
        out.status = ZoneStatus.uncertain
        out.evidence = f"VLM weakly reports residue but classical vision sees none; VLM: {vlm.evidence}"[:400]
        tag = "fused_conflict"
    if out.status != ZoneStatus.clean:
        fc = _cause_from_force(force)
        if fc is not None and out.likely_cause in (LikelyCause.unknown, LikelyCause.dried_stain):
            out.likely_cause = fc
        if out.residue_bbox_norm is None and r.bbox_norm is not None:
            out.residue_bbox_norm = r.bbox_norm
    else:
        out.residue_bbox_norm = None
        out.likely_cause = LikelyCause.unknown
        out.stain_type = StainType.unknown
    return out, tag


def _upscale(img: np.ndarray, side: int) -> np.ndarray:
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_CUBIC)


class CVVerifier:
    name = "cv"

    def __init__(self, grid: ZoneGrid, detector: ResidueDetector, thresholds: Optional[FusionThresholds] = None):
        self.grid, self.det, self.th = grid, detector, thresholds or FusionThresholds()

    def screen(self, after: np.ndarray, zone_ids: list[str]) -> list[str]:
        cv = self.det.analyze(after, zone_ids)
        return [z for z in zone_ids if cv[z].residue_fraction > self.th.cv_residue]

    def inspect(self, zone_ids: list[str], after: np.ndarray, before: Optional[np.ndarray] = None,
                force: Optional[dict[str, dict[str, Any]]] = None,
                closeups: Optional[dict[str, np.ndarray]] = None) -> VerifierOutput:
        cv = self.det.analyze(after, zone_ids)
        res = {z: cv_judgement(z, cv[z], self.th, (force or {}).get(z)) for z in zone_ids}
        return VerifierOutput(res, {z: "cv" for z in zone_ids}, cv)


class VLMVerifier:
    name = "vlm"

    def __init__(self, grid: ZoneGrid, detector: ResidueDetector, client: LLMClient,
                 thresholds: Optional[FusionThresholds] = None, crop_side: int = 336,
                 px_per_m: float = DEFAULT_PX_PER_M, max_zones_per_call: int = 6, overrides: Optional[dict[str, Any]] = None,
                 send_reference: Optional[bool] = None):
        self.grid, self.det, self.client = grid, detector, client
        self.overrides = dict(overrides or {})            # experiment: swap the verifier model / backend (e.g. a LoRA on 8002); empty online
        self.th = thresholds or FusionThresholds()
        self.side, self.px = crop_side, px_per_m
        self.max_zones = max_zones_per_call
        # v8: each cell comes with a clean reference crop, so texture on textured surfaces (wood grain, granite, brushed steel) does not count as residue; False = inputs of v7 and earlier
        # None = read configs/models.yaml roles.verifier.send_reference (online config, decided by E19; default True)
        if send_reference is None:
            send_reference = bool(load_yaml("models.yaml")["roles"]["verifier"].get("send_reference", True))
        self.send_reference = send_reference
        self.system_prompt = (PROMPT_DIR / "verifier_zones.md").read_text(encoding="utf-8")

    def _messages(self, zone_ids: list[str], after: np.ndarray, before: Optional[np.ndarray],
                  cv: dict[str, ZoneResidue], force: dict[str, dict[str, Any]],
                  closeups: Optional[dict[str, np.ndarray]] = None) -> list[dict[str, Any]]:
        parts, order, lines = [], [], []
        for z in zone_ids:
            if closeups and z in closeups:
                img = closeups[z]
                h, w = img.shape[:2]
                scale = 512 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                parts.append(image_part(encode_jpeg(img)))
                order.append(f"{len(order) + 1} = zone {z} CLOSEUP from the wrist camera (not rectified)")
            parts.append(image_part(encode_jpeg(_upscale(zone_crop(after, self.grid, z, self.px), self.side))))
            order.append(f"{len(order) + 1} = zone {z} AFTER (now)")
            if before is not None:
                parts.append(image_part(encode_jpeg(_upscale(zone_crop(before, self.grid, z, self.px), self.side))))
                order.append(f"{len(order) + 1} = zone {z} BEFORE (previous inspection)")
            ref = getattr(self.det, "ref_rgb", None)
            if self.send_reference and ref is not None:
                parts.append(image_part(encode_jpeg(_upscale(zone_crop(ref, self.grid, z, self.px), self.side))))
                order.append(f"{len(order) + 1} = zone {z} REFERENCE (clean surface)")
            r = cv[z]
            line = f"- {z}: classical_vision={100 * r.residue_fraction:.2f}% bbox={r.bbox_norm}"
            if z in force:
                f = force[z]
                line += (f"; last_wipe target={f['target_n']:.1f}N mean={f['mean_normal_n']:.1f}N "
                         f"within1N={100 * f['within_1n_ratio']:.0f}% losses={f['contact_loss_count']} "
                         f"coverage={100 * f['coverage_ratio']:.0f}%")
            lines.append(line)
        text = ("Image order: " + "; ".join(order) + "\n\nZones to judge:\n" + "\n".join(lines) +
                "\n\nReturn JSON with one entry per zone listed above.")
        parts.append({"type": "text", "text": text})
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": parts}]

    def screen(self, after: np.ndarray, zone_ids: list[str]) -> list[str]:
        cv = self.det.analyze(after, zone_ids)
        return [z for z in zone_ids if cv[z].residue_fraction > self.th.cv_residue]

    def inspect(self, zone_ids: list[str], after: np.ndarray, before: Optional[np.ndarray] = None,
                force: Optional[dict[str, dict[str, Any]]] = None,
                closeups: Optional[dict[str, np.ndarray]] = None) -> VerifierOutput:
        force = force or {}
        cv = self.det.analyze(after, zone_ids)
        results: dict[str, VerifierZoneResult] = {}
        prov: dict[str, str] = {}
        raw: dict[str, dict[str, Any]] = {}
        calls = []
        for k in range(0, len(zone_ids), self.max_zones):
            chunk = zone_ids[k : k + self.max_zones]
            try:
                res = self.client.chat("verifier", self._messages(chunk, after, before, cv, force, closeups),
                                       schema=VerifierBatch, tag="inspect", **self.overrides)
                calls.append({"request_id": res.request_id, "latency_s": round(res.latency_s, 3),
                              "schema_ok": res.schema_ok, "model": res.model, "backend": res.backend})
                got = {z.zone_id: z for z in res.parsed.zones} if res.schema_ok and res.parsed else {}
            except Exception as e:   # local vLLM unavailable etc.: degrade to CV and record it honestly
                calls.append({"error": f"{type(e).__name__}: {e}"[:300]})
                got = {}
            for z in chunk:
                if z in got:
                    raw[z] = got[z].model_dump(mode="json")
                    results[z], prov[z] = fuse(got[z], cv[z], self.th, force.get(z))
                else:
                    results[z], prov[z] = cv_judgement(z, cv[z], self.th, force.get(z)), "cv_fallback"
        return VerifierOutput(results, prov, cv, {"calls": calls}, raw)


# Simulated stain type (configs/sim.yaml dirt.types) → force-table stain type; used only by OracleVerifier
SIM_STAIN_TYPE = {"light": StainType.wet_spill, "dried": StainType.dried_stain, "marker": StainType.ink_mark,
                  "permanent": StainType.ink_mark, "wine": StainType.dried_stain, "grease": StainType.grease}


class OracleVerifier:
    """Simulation ground truth (reads the stain map). Used only for tests and generating simulation labels; must not be used to report verifier accuracy in demos or reports."""

    name = "oracle"

    def __init__(self, robot: Any, grid: ZoneGrid):
        self.robot, self.grid = robot, grid

    def screen(self, after: Any, zone_ids: list[str]) -> list[str]:
        return [z for z in zone_ids if not self.robot.dirt.zone_stats(z)["clean"]]

    def inspect(self, zone_ids: list[str], after: Any = None, before: Any = None,
                force: Optional[dict[str, dict[str, Any]]] = None, closeups: Any = None) -> VerifierOutput:
        res = {}
        for z in zone_ids:
            st = self.robot.dirt.zone_stats(z)
            if st["clean"]:
                res[z] = VerifierZoneResult(zone_id=z, status="clean", confidence=1.0, evidence="sim ground truth")
            else:
                kind = max(st["by_type"], key=st["by_type"].get)
                cause = {"dried": LikelyCause.dried_stain}.get(kind, LikelyCause.unknown)
                stain = SIM_STAIN_TYPE.get(kind, StainType.unknown)
                fc = _cause_from_force((force or {}).get(z))
                res[z] = VerifierZoneResult(zone_id=z, status="residue", confidence=1.0,
                                            residue_bbox_norm=st["bbox_norm"], likely_cause=fc or cause, stain_type=stain,
                                            evidence=f"sim ground truth: {kind} {100 * st['residue_fraction']:.1f}%")
        return VerifierOutput(res, {z: "oracle" for z in zone_ids})
