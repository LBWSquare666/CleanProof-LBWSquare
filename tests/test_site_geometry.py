"""Site geometry (configs/site/home_dining): whole tabletop workspace, Spot reachable, stowed-arm survey pose, a recording registers the site and rebuilds the same grid."""

import os

import numpy as np
import pytest

from cleanproof.common.config import ENV_SITE, use_site
from cleanproof.common.zones import ZoneGrid


@pytest.fixture
def home_site():
    use_site("home_dining")
    yield
    use_site(None)
    assert os.environ.get(ENV_SITE, "") == ""


def test_use_site_switches_grid(home_site):
    g = ZoneGrid()
    assert (g.rows, g.cols) == (3, 5) and len(g.ids) == 15
    assert np.allclose(g.size, [0.34, 0.60])
    assert g.table_half is not None and g.survey_arm_q.get("spot_arm")
    use_site(None)
    assert ZoneGrid().size.tolist() == [0.24, 0.30]
    use_site("home_dining")


def test_use_site_keeps_manual_site(monkeypatch):
    monkeypatch.setenv(ENV_SITE, "some_real_site")
    use_site(None)                                    # not a value set by this function: do not clear
    assert os.environ[ENV_SITE] == "some_real_site"


def test_spot_reaches_whole_work_area(home_site):
    from cleanproof.common.config import load_yaml
    from cleanproof.robot.platforms import PlatformKinematics, build_platform_scene, get_platform, workspace_points
    g = ZoneGrid()
    spec = get_platform("spot_arm")
    assert spec.reach_m == pytest.approx(1.20)       # the site overrides the protection radius
    rcfg = load_yaml("robot.yaml")
    sc = build_platform_scene(spec, g, load_yaml("sim.yaml"), tcp_len=float(rcfg["startup"]["expected_tcp_offset"][2]),
                              wrist_cam_pos=tuple(rcfg["wrist_camera"]["pos_in_flange"]))
    kin = PlatformKinematics(sc)
    kin.solve_home(rcfg, g)
    assert kin.survey_q is not None and kin.limit_margin_deg(kin.survey_q) >= 10.0
    for z in (0.0, 0.06):
        for T in workspace_points(g, z=z):
            q = kin.ik(T, kin.home_q)
            assert q is not None, f"no IK solution for work-area point {T[:3, 3]}"
            assert kin.limit_margin_deg(q) >= 5.0


def test_h1_survey_and_far_row_wipe(home_site, tmp_path):
    """Stowed-arm survey pose → wipe the farthest row of cells → back to survey pose: joint path stays within the protection radius, skills all succeed."""
    from cleanproof.sim.scenarios import build
    sc = build("H1_dining_dried")
    try:
        assert sc.meta["platform"] == "spot_arm" and len(sc.grid.ids) == 15
        sc.add_stains()
        assert sc.executor.survey().status == "ok"
        out = sc.executor.wipe(["C3"], 12.0, "zigzag", 1, None)
        assert out.status == "ok", out.reason
        assert sc.executor.survey().status == "ok"
        assert sc.robot.dirt.zone_stats("C3")["clean"]
    finally:
        sc.close()


def test_new_stain_types_visible_and_removable(home_site):
    from cleanproof.common.config import load_yaml
    from cleanproof.sim.dirt import DirtMap
    g = ZoneGrid()
    cfg = load_yaml("sim.yaml")["dirt"]
    for kind, surface, add in (("wine", "rough_stone", lambda d, c: d.add_blob("wine", c, 0.022, seed=1)),
                               ("grease", "stainless_steel", lambda d, c: d.add_hand("grease", c, scale=0.72, angle_deg=75, level=0.8, seed=2))):
        d = DirtMap(g, cfg, surface=surface)
        add(d, g["B3"].center)
        st = d.zone_stats("B3")
        assert not st["clean"] and st["by_type"][kind] > 0, (kind, st)


@pytest.mark.parametrize("site", [None] + sorted(p.name for p in __import__("pathlib").Path(__file__).resolve().parents[1].joinpath("configs", "site").iterdir() if p.is_dir()))
def test_table_legs_under_table_top(site):
    """Table legs must be under the tabletop ("the tabletop is floating, the legs do not support it at all"): each leg's 25 mm square post lies entirely within the tabletop projection."""
    from cleanproof.robot.platforms import table_footprint, table_leg_xy
    use_site(site)
    try:
        g = ZoneGrid()
        c, h = table_footprint(g)
        for x, y in table_leg_xy(g):
            assert abs(x - c[0]) + 0.025 <= h[0] + 1e-9 and abs(y - c[1]) + 0.025 <= h[1] + 1e-9, (site, x, y, c, h)
    finally:
        use_site(None)
