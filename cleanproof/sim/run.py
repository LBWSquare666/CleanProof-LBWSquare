"""Assemble and run one simulation scenario: scenario + camera source + verifier + decision policy → TaskRunner → judged against ground truth."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent.force_table import ForceTable, load_force_table
from ..agent.loop import Approver, LoopOptions, TaskRunner
from ..agent.policy import FixedPolicy, NemotronPolicy, RulePolicy
from ..agent.runlog import RunLogger
from ..common.llm_client import LLMClient
from ..perception.residue_cv import ResidueDetector
from ..perception.sources import SimCameraSource
from ..perception.verifier import CVVerifier, OracleVerifier, VLMVerifier
from .scenarios import build, judge, touched_zones

POLICIES = ("rules", "nemotron", "fixed1", "fixed2")
VERIFIERS = ("cv", "vlm", "oracle")


def make_policy(name: str, client: Optional[LLMClient], role: str = "decision",
                prompt_file: str = "decision_system.md", table: Optional[ForceTable] = None):
    """table: the force table of this job (read for this job's site); None = read for the site currently active in the process (old behavior)."""
    if name == "rules":
        return RulePolicy(table)
    if name == "fixed1":
        return FixedPolicy(1)
    if name == "fixed2":
        return FixedPolicy(2)
    if name == "nemotron":
        if client is None:
            raise ValueError("the nemotron policy needs an LLMClient (Token Factory)")
        return NemotronPolicy(client, table=table, role=role, prompt_file=prompt_file)        # role: a role in configs/models.yaml (E12 swaps the planning model, using fast / audit)
    raise ValueError(f"unknown policy {name!r}, choose from {POLICIES}")


def run_sim_scenario(name: str, policy: str = "rules", verifier: str = "cv", source: str = "topdown",
                     run_root: Optional[Path] = None, client: Optional[LLMClient] = None,
                     audit: bool = False, approve: Optional[Approver] = None,
                     on_runner: Optional[Callable[[TaskRunner, Any], None]] = None,
                     robot_backend: str = "mujoco", mode: str = "full", enforce_gate: bool = False,
                     gate: Any = None, spec: Optional[dict[str, Any]] = None, ablation: Optional[dict[str, Any]] = None,
                     platform: Optional[str] = None, recheck_all: bool = True,
                     force_site: Optional[str] = None) -> dict[str, Any]:
    """approve: operator approval hook (the console "needs approval" mode); on_runner(runner, scenario): callback before the loop starts,
    which the console uses to get the runner (request stop) and the simulated robot (E-stop).
    robot_backend: mujoco (skills drive MujocoRobot directly) or rtde_fake (skills → rtde_backend → fake RTDE (MuJoCo));
    mode: graded mode of the rtde backend, shadow / dry_run / full.
    spec: a scenario spec given directly (randomization baseline, cleanproof/sim/scenario_random.py); when None it is read from configs/scenarios by name.
    ablation (simulation experiments): {"guard": bool, "final_check": bool, "force_all": newtons (all three force levels set to the same value: fixed-force ablation),
    "recheck_all": False (old recheck logic: after a wipe only the wiped cells are rechecked, reproducing benchmark_v4)}; defaults to the official system.
    force_site: which site the force table is read for (a job in the software = the site the operator chose, not the site of the simulation scenario template); read once when the job starts and kept for the whole job.
    None = old behavior (the site registered by the scenario, the guardrail reads the process current site at every step), benchmarks and tests unchanged."""
    ablation = dict(ablation or {})
    sc = build(name, render=(source == "render"), robot_backend=robot_backend, mode=mode,
               enforce_gate=enforce_gate, gate=gate, spec=spec, platform=platform)
    src = SimCameraSource(sc.robot, sc.grid, mode=source)
    sc.executor.survey()
    reference = src.rectified()            # clean reference image before stains are placed (on the real robot the operator captures it after cleaning)
    det = ResidueDetector(sc.grid, reference_rgb=reference)
    sc.add_stains()
    if verifier == "cv":
        ver = CVVerifier(sc.grid, det)
    elif verifier == "oracle":
        ver = OracleVerifier(sc.robot, sc.grid)
    elif verifier == "vlm":
        if client is None:
            raise ValueError("the vlm verifier needs an LLMClient (Cosmos on a local vLLM)")
        ov = {k: ablation[f"verifier_{k}"] for k in ("model", "backend") if ablation.get(f"verifier_{k}")}
        # E17: swap the verifier for a LoRA model (ablation verifier_model / verifier_backend); the LoRA asks one zone per call, as in the training format (verifier_zones_per_call)
        ver = VLMVerifier(sc.grid, det, client, overrides=ov, max_zones_per_call=int(ablation.get("verifier_zones_per_call") or 6),
                          send_reference=ablation.get("verifier_reference"))   # None = production config (models.yaml); False = the input of v7 and earlier (for the E19 comparison)
    else:
        raise ValueError(f"unknown verifier {verifier!r}, choose from {VERIFIERS}")
    if ablation.get("force_all") is not None:                 # fixed-force ablation: execute the same force whatever the plan asks (the policy force-raising logic is unchanged)
        sc.executor.force_override_n = float(ablation["force_all"])
    table = load_force_table(force_site) if force_site is not None else None
    pol = make_policy(policy, client, role=str(ablation.get("planner_role", "decision")),
                      prompt_file=str(ablation.get("decision_prompt", "decision_system.md")), table=table)
    tag = "" if robot_backend == "mujoco" else f"-{robot_backend}" + ("" if mode == "full" else f"-{mode}")
    if sc.meta.get("platform", "ur3e") != "ur3e":
        tag += f"-{sc.meta['platform']}"
    run_id = f"sim-{name}-{policy}-{verifier}{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    logger = RunLogger(run_root, run_id=run_id)
    logger.write_json("reference_meta.json", {"source": src.name, "scenario": sc.spec, "robot": sc.meta})
    logger.image("reference", reference)
    logger.write_json("ablation.json", ablation) if ablation else None
    runner = TaskRunner(sc.task, sc.executor, src, ver, pol, logger,
                        LoopOptions(audit=audit, guard=bool(ablation.get("guard", True)), final_check=bool(ablation.get("final_check", True)),
                                    recheck_all=bool(ablation.get("recheck_all", recheck_all))),
                        approve=approve, audit_client=client if audit else None, injector=sc.injector(),
                        truth=sc.truth, force_table=table)
    if on_runner is not None:
        on_runner(runner, sc)
    try:
        summary = runner.run()
    finally:
        sc.close()
    summary["robot_mode"] = mode
    summary["robot_label"] = sc.meta.get("label", robot_backend)
    logger.write_json("summary.json", summary)      # add the backend label and graded mode (gate evidence must show whether it was fake or the real robot)
    j = judge(sc, summary, touched_zones(logger.events_path))
    j["robot"] = sc.meta
    if sc.fake is not None:
        j["rtde_calls"] = {k: sc.fake.count(k) for k in ("speedL", "speedStop", "moveL", "moveJ", "forceMode", "stopScript")}
    logger.write_json("scenario.json", j)
    return {"summary": summary, "judge": j, "run_dir": str(logger.dir)}
