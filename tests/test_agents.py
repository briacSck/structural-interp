"""Agent-level checks (require torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from mdp import RustMDP
from train_agents import state_features, train_clone, train_heuristic


@pytest.fixture(scope="module")
def solution():
    return RustMDP().solve()


def _ccp(agent, n_states):
    feats = torch.tensor(state_features(np.arange(n_states), n_states),
                         dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(agent(feats)).numpy()


def test_clone_matches_optimal_policy(solution):
    agent = train_clone(solution)
    ccp = _ccp(agent, solution.mdp.n_states)
    assert np.max(np.abs(ccp - solution.ccp_replace)) < 0.02


def test_heuristic_trains_and_differs_from_optimal(solution):
    agent = train_heuristic(solution)
    ccp = _ccp(agent, solution.mdp.n_states)
    gap = np.abs(ccp - solution.ccp_replace)
    # Close on average (behavioral similarity is checked in experiment.py with
    # ergodic weights), but clearly not the same function everywhere.
    assert np.max(gap) > 0.05
    assert np.mean(gap) < 0.25
