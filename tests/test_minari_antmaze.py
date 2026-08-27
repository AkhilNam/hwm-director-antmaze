"""Tests for converting Minari-style episodes into Transitions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hwm_director.data.minari_antmaze import (
    transitions_from_minari_episode,
    transitions_from_minari_episodes,
)
from hwm_director.data.state import GOAL_DIM, OBSERVATION_DIM, STATE_DIM
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import ACTION_DIM
from hwm_director.data.worker_dataset import WorkerDataset


def _episode(episode_id: int, n_steps: int, *, x0: float = 0.0) -> SimpleNamespace:
    n_obs = n_steps + 1
    achieved = np.zeros((n_obs, 2), dtype=np.float64)
    achieved[:, 0] = x0 + np.arange(n_obs, dtype=np.float64)
    body = np.zeros((n_obs, OBSERVATION_DIM), dtype=np.float64)
    body[:, 0] = np.arange(n_obs, dtype=np.float64)
    desired = np.tile(np.array([10.0, 20.0], dtype=np.float64), (n_obs, 1))
    actions = np.full((n_steps, ACTION_DIM), float(episode_id), dtype=np.float64)
    actions[:, 0] = np.arange(n_steps, dtype=np.float64)
    return SimpleNamespace(
        id=episode_id,
        observations={
            "achieved_goal": achieved,
            "observation": body,
            "desired_goal": desired,
        },
        actions=actions,
        infos={},
    )


def test_minari_episode_conversion_shapes() -> None:
    transitions = transitions_from_minari_episode(_episode(7, 5), episode_id=7)
    assert len(transitions) == 5
    for t in transitions:
        t.validate()
        assert t.state.shape == (STATE_DIM,)
        assert t.action.shape == (ACTION_DIM,)
        assert t.next_state.shape == (STATE_DIM,)
        assert t.goal.shape == (GOAL_DIM,)
        assert t.episode_id == 7
        assert t.qpos is not None and t.qpos.shape == (15,)
        assert t.qvel is not None and t.qvel.shape == (14,)


def test_minari_episode_preserves_temporal_order_and_boundaries() -> None:
    ep0 = _episode(0, 4, x0=0.0)
    ep1 = _episode(1, 3, x0=100.0)
    transitions = transitions_from_minari_episodes([ep0, ep1])
    grouped = group_by_episode(transitions)
    assert set(grouped) == {0, 1}
    assert len(grouped[0]) == 4
    assert len(grouped[1]) == 3
    xs0 = [t.state[0] for t in grouped[0]]
    next_xs0 = [t.next_state[0] for t in grouped[0]]
    assert xs0 == [0.0, 1.0, 2.0, 3.0]
    assert next_xs0 == [1.0, 2.0, 3.0, 4.0]
    for t in grouped[0]:
        assert t.episode_id == 0
        assert t.next_state[0] < 100.0
    for t in grouped[1]:
        assert t.episode_id == 1
        assert t.state[0] >= 100.0


def test_no_transition_crosses_episode_boundary() -> None:
    transitions = transitions_from_minari_episodes(
        [_episode(0, 5, x0=0.0), _episode(1, 5, x0=50.0)]
    )
    grouped = group_by_episode(transitions)
    last_next = grouped[0][-1].next_state[0]
    first_next_ep = grouped[1][0].state[0]
    assert last_next != first_next_ep
    for t in transitions:
        if t.episode_id == 0:
            assert t.next_state[0] < 50.0
        else:
            assert t.state[0] >= 50.0


def test_worker_examples_from_converted_episodes_stay_in_episode() -> None:
    transitions = transitions_from_minari_episodes(
        [_episode(0, 6, x0=0.0), _episode(1, 6, x0=80.0)]
    )
    dataset = WorkerDataset(transitions, horizon_k=4)
    assert dataset.examples
    for example in dataset.examples:
        assert 1 <= example.offset <= 4
        # x increases by 1 per step within an episode; episode 1 starts at 80.
        if example.state[0] < 80.0:
            assert example.subgoal[0] < 80.0
        else:
            assert example.subgoal[0] >= 80.0


def test_rejects_episode_without_terminal_observation() -> None:
    episode = _episode(0, 3)
    episode.observations["achieved_goal"] = episode.observations["achieved_goal"][:-1]
    episode.observations["observation"] = episode.observations["observation"][:-1]
    episode.observations["desired_goal"] = episode.observations["desired_goal"][:-1]
    with pytest.raises(ValueError, match="one more observation"):
        transitions_from_minari_episode(episode, episode_id=0)


def test_prefers_infos_qpos_qvel_when_present() -> None:
    episode = _episode(0, 3)
    qpos = np.full((4, 15), 3.0, dtype=np.float64)
    qvel = np.full((4, 14), 4.0, dtype=np.float64)
    qpos[1, 0] = 9.0
    episode.infos = {"qpos": qpos, "qvel": qvel}
    transitions = transitions_from_minari_episode(episode, episode_id=0)
    np.testing.assert_array_equal(transitions[0].qpos, qpos[0])
    np.testing.assert_array_equal(transitions[1].qpos, qpos[1])
    np.testing.assert_array_equal(transitions[0].qvel, qvel[0])

