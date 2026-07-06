"""Pipeline validation tests (plan's Verification section).

Run from the project root:  py -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdp import RustMDP
from estimate import estimate_nfxp


@pytest.fixture(scope="module")
def solution():
    return RustMDP().solve()


def test_value_iteration_converges(solution):
    assert solution.n_iterations < 10_000
    # V(x) must be decreasing in mileage: higher mileage is always worse.
    assert np.all(np.diff(solution.v_bar) <= 1e-10)


def test_ccp_monotone_in_mileage(solution):
    # Replacement probability rises with mileage under the optimal policy.
    assert np.all(np.diff(solution.ccp_replace) >= -1e-10)
    assert solution.ccp_replace[0] < 0.05
    assert solution.ccp_replace[-1] > solution.ccp_replace[0]


def test_simulation_matches_analytic_ccp(solution):
    panel = solution.simulate(n_buses=2000, n_periods=500,
                              rng=np.random.default_rng(42))
    emp_ccp, counts = panel.empirical_ccp(solution.mdp.n_states)
    well_visited = counts > 5000
    assert well_visited.sum() > 10, "simulation should visit many states often"
    gap = np.abs(emp_ccp[well_visited] - solution.ccp_replace[well_visited])
    assert np.nanmax(gap) < 0.02


def test_nfxp_recovers_true_parameters(solution):
    truth = solution.mdp
    panel = solution.simulate(n_buses=500, n_periods=200,
                              rng=np.random.default_rng(7))
    result = estimate_nfxp(panel, truth)
    assert result.converged
    assert abs(result.theta1 - truth.theta1) < 0.005
    assert abs(result.rc - truth.rc) < 1.0


def test_hotz_miller_recovers_true_parameters(solution):
    from estimate import estimate_hotz_miller

    truth = solution.mdp
    panel = solution.simulate(n_buses=500, n_periods=200,
                              rng=np.random.default_rng(7))
    result = estimate_hotz_miller(panel, truth)
    # CCP estimator: noisier than NFXP (nonparametric first stage), so
    # tolerances are looser but must still bracket the truth.
    assert abs(result.theta1 - truth.theta1) < 0.01
    assert abs(result.rc - truth.rc) < 1.5
