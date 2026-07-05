"""Fast checks for the 2-D environment (NFXP recovery is exercised in
experiment2d.py rather than here — too slow for the unit suite)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdp2d import RustMDP2D


@pytest.fixture(scope="module")
def solution():
    return RustMDP2D().solve()


def test_value_iteration_converges(solution):
    assert solution.n_iterations < 20_000


def test_transition_matrix_rows_sum_to_one(solution):
    p = solution.mdp.transition_matrix()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_ccp_increases_in_mileage_and_cost_regime(solution):
    grid = solution.ccp_grid()
    # worse regime -> replace more, at every mileage level
    assert np.all(np.diff(grid, axis=1) >= -1e-10)
    # more mileage -> replace more, in every regime
    assert np.all(np.diff(grid, axis=0) >= -1e-10)


def test_value_decreases_in_mileage_and_cost_regime(solution):
    v = solution.v_bar.reshape(solution.mdp.n_mileage, solution.mdp.n_cost)
    assert np.all(np.diff(v, axis=0) <= 1e-10)
    assert np.all(np.diff(v, axis=1) <= 1e-10)


def test_simulation_matches_analytic_ccp(solution):
    panel = solution.simulate(n_buses=3000, n_periods=300,
                              rng=np.random.default_rng(5))
    emp, counts = panel.empirical_ccp(solution.mdp.n_states)
    mask = counts > 3000
    assert mask.sum() > 20
    assert np.nanmax(np.abs(emp[mask] - solution.ccp_replace[mask])) < 0.03
