"""Shared fixture: the real run directory produced by one quick simulation (rule policy + classical vision + GL-free top-down source), reused by the report and console tests."""

from pathlib import Path

import pytest

from cleanproof.sim.run import run_sim_scenario


@pytest.fixture(scope="session")
def quick_run_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    res = run_sim_scenario("S1_light", "rules", "cv", "topdown", run_root=root)
    assert res["judge"]["passed"], res["judge"]
    return Path(res["run_dir"])
