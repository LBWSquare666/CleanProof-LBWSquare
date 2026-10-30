"""Build, run and ground-truth judging of simulation scenarios (configs/scenarios/*.yaml).

Ground truth comes from the stain map (available only in simulation): used to count "judged clean by mistake" (the verifier says clean, but it is actually still dirty)
and whether the task was truly completed. Judging results are written to runs/<run_id>/scenario.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..common.config import CONFIG_DIR, load_yaml
from ..common.schemas import TaskSpec
from ..common.zones import ZoneGrid
from ..robot.backends.mujoco_backend import MujocoRobot
from ..robot.io import RobotIO
from ..robot.skills import SkillExecutor
from .contact import Obstacle

ROBOT_BACKENDS = ("mujoco", "rtde_fake")   # rtde_fake: skills → rtde_backend → fake RTDE (MuJoCo) → the same MujocoRobot


def scenario_names() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "scenarios").glob("*.yaml"))


def load_scenario(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "scenarios" / f"{name}.yaml")


def _stain(robot: MujocoRobot, grid: ZoneGrid, s: dict[str, Any]) -> None:
    cx, cy = grid[s["zone"]].center
    ox, oy = s.get("offset", [0.0, 0.0])
    if s.get("shape", "blob") == "strokes":            # marker strokes (v8)
        robot.dirt.add_strokes(s["type"], (cx + ox, cy + oy), float(s.get("length", 0.034)), n=int(s.get("n", 3)),
                               spacing=float(s.get("spacing", 0.008)), width=float(s.get("width", 0.0035)),
                               angle_deg=float(s.get("angle_deg", 0.0)), level=float(s.get("level", 1.0)),
                               seed=int(s.get("seed", 0)))
        return
    if s.get("shape") == "hand":                       # greasy handprint (v12 triptych: stainless steel table)
        robot.dirt.add_hand(s["type"], (cx + ox, cy + oy), scale=float(s.get("scale", 1.0)), angle_deg=float(s.get("angle_deg", 0.0)),
                            level=float(s.get("level", 1.0)), seed=int(s.get("seed", 0)))
        return
    robot.dirt.add_blob(s["type"], (cx + ox, cy + oy), float(s["radius"]), level=float(s.get("level", 1.0)),
                        seed=int(s.get("seed", 0)))


@dataclass
class SimScenario:
    spec: dict[str, Any]
    grid: ZoneGrid
    robot: MujocoRobot          # physics and ground truth (stains, rendering) are always MujocoRobot
    executor: SkillExecutor
    task: TaskSpec
    io: Optional[RobotIO] = None      # the RobotIO skills actually use: robot itself for mujoco; RTDERobot for rtde_fake
    fake: Any = None                  # FakeRTDE for rtde_fake (fault injection, call records)
    robot_backend: str = "mujoco"
    mode: str = "full"
    meta: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        if self.io is not None and self.io is not self.robot:
            self.io.close()
        self.robot.close()

    def add_stains(self) -> None:
        for s in self.spec.get("stains", []):
            _stain(self.robot, self.grid, s)

    def injector(self):
        events = self.spec.get("events", [])

        def inject(step: int, state: Any) -> None:
            for ev in events:
                if int(ev["at_step"]) == step and "add_stain" in ev:
                    _stain(self.robot, self.grid, ev["add_stain"])

        return inject if events else None

    def truth(self) -> dict[str, bool]:
        return {z: self.robot.dirt.zone_stats(z)["clean"] for z in self.grid.ids}


def obstacles_of(spec: dict[str, Any], grid: ZoneGrid) -> list[Obstacle]:
    """obstacles of the scenario yaml (normalized in-zone [u0, v0, u1, v1] + height) → list of Obstacle in table coordinates."""
    out = []
    for o in spec.get("obstacles", []):
        z = grid[o["zone"]]
        u0, v0, u1, v1 = o["rel"]
        w, h = z.size
        out.append(Obstacle(z.xmin + u0 * w, z.ymin + v0 * h, z.xmin + u1 * w, z.ymin + v1 * h, float(o["height"])))
    return out


load_spec = load_scenario      # alias: story_video / submission_video fetch the spec by scenario name


def build(name: str, render: bool = False, robot_backend: str = "mujoco", mode: str = "full",
          enforce_gate: bool = False, gate: Any = None, spec: Optional[dict[str, Any]] = None, platform: Optional[str] = None) -> SimScenario:
    """With robot_backend=rtde_fake, skills drive the same MujocoRobot through rtde_backend → fake RTDE (MuJoCo);
    mode is the graded rollout mode (shadow / dry_run / full), and enforce_gate decides whether the gate is enforced for the fake as well.
    spec: a scenario spec given directly (same format as configs/scenarios/*.yaml, used by the randomized benchmark); when None the yaml is read by name."""
    if robot_backend not in ROBOT_BACKENDS:
        raise ValueError(f"unknown robot backend {robot_backend!r}, choose from {ROBOT_BACKENDS}")
    spec = dict(spec) if spec is not None else load_scenario(name)
    from ..common.config import use_site
    use_site(spec.get("site"))           # site geometry registered by the scenario (configs/site/<site>/); falls back to the default work area if none
    grid = ZoneGrid()
    obstacles = obstacles_of(spec, grid)
    task = TaskSpec(**spec["task"])
    # table appearance for perception rendering: a scenario can set render_surface on its own (for example the old flat-color table plain), by default the same as the task's table material
    # platform (configs/platforms.yaml): argument > scenario spec > environment variable CLEANPROOF_PLATFORM > ur3e
    import os
    platform = platform or spec.get("platform") or os.environ.get("CLEANPROOF_PLATFORM") or "ur3e"
    robot = MujocoRobot(grid=grid, obstacles=obstacles, render=render, surface=spec.get("render_surface", task.surface), platform=platform)
    io: RobotIO = robot
    fake = None
    meta: dict[str, Any] = {"robot_backend": robot_backend, "mode": mode, "platform": platform, "platform_label": robot.platform.label,
                            "platform_kind": robot.platform.kind}
    if robot_backend == "rtde_fake":
        from ..robot.backends.fake_rtde import FakeRTDE
        from ..robot.backends.rtde_backend import make_rtde_robot

        fake = FakeRTDE(robot)
        io = make_rtde_robot("rtde_fake", mode, rcfg=robot.rcfg, z_surface=grid.z_surface, fake=fake,
                             gate=gate, enforce_gate=enforce_gate)
        meta["label"] = fake.label          # "fake RTDE (MuJoCo)": logs and reports must make clear it is not the real robot
    ex = SkillExecutor(io, grid=grid, rcfg=robot.rcfg, forbidden_zones=task.forbidden_zones)
    return SimScenario(spec, grid, robot, ex, task, io=io, fake=fake, robot_backend=robot_backend, mode=mode, meta=meta)


def judge(sc: SimScenario, summary: dict[str, Any], touched: set[str]) -> dict[str, Any]:
    truth = sc.truth()
    final = summary["final_status"]
    false_clean = [z for z, s in final.items() if s == "clean" and not truth[z]]
    targets_clean = all(truth[z] for z in summary["targets"])
    exp = sc.spec.get("expect", {})
    checks = {}
    if "status" in exp:
        checks["status"] = summary["status"] == exp["status"]
    if exp.get("targets_truly_clean"):
        checks["targets_truly_clean"] = targets_clean
    if "max_false_clean" in exp:
        checks["false_clean"] = len(false_clean) <= int(exp["max_false_clean"])
    if "human_zones" in exp:
        asked = {z for r in summary["human_requests"] for z in r["zones"]}
        checks["human_zones"] = set(exp["human_zones"]) <= asked
    if "untouched" in exp:
        checks["untouched"] = not (set(exp["untouched"]) & touched)
    return {
        "scenario": sc.spec["name"], "passed": all(checks.values()), "checks": checks,
        "false_clean_zones": false_clean, "targets_truly_clean": targets_clean,
        "total_passes": int(sum(summary["passes"].values())), "robot_time_s": summary["robot_time_s"],
        "status": summary["status"],
    }


def touched_zones(events_path: Path) -> set[str]:
    from ..common.jsonl import read_jsonl

    out: set[str] = set()
    for e in read_jsonl(events_path):
        if e["kind"] == "skill" and e.get("skill") in ("wipe_zone", "spot_scrub") and e.get("status") != "rejected":
            out |= set(e.get("zone_ids", []))
    return out
