"""Director-Value ``pi_H``: pick a data-supported subgoal with ``Q_H``.

    pi_H^Value(h_tau, g*) -> g_tau

Procedure:

1. Retrieve recorded K-step futures from compatible offline source states.
2. Reject candidates farther than ``max_subgoal_distance`` (default 2 m).
3. Optionally reject by normalized source-state distance.
4. Optionally **hard**-reject by predicted ``(f_L, pi_L)^K`` subgoal error.
5. Score survivors with ``Q_H(h_tau, candidate, g*)``.
6. Optionally apply a **soft** reachability penalty (no hard cutoff):

       score = z(Q_H) - lambda * z(reach_error)

   z is a candidate-set z-score (see ``combined_reachability_scores``).
7. Return the highest-scoring candidate.
8. If none remain, fall back to the BC manager.

``Q_H`` is a scorer, not ``f_H``. Reachability calls
``controller.high_level_transition``: Director uses ``(f_L, pi_L)^K``;
HWM uses explicit ``f_H_phi``. Same scoring rule, different ``f_H``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import ACHIEVED_GOAL_DIM, STATE_DIM
from hwm_director.data.subgoal_candidates import (
    DEFAULT_CANDIDATE_RADIUS,
    DEFAULT_N_CANDIDATES,
    DEFAULT_STATE_DISTANCE_WEIGHT,
    DEFAULT_XY_DISTANCE_WEIGHT,
    SubgoalCandidateIndex,
)
from hwm_director.data.worker_dataset import normalize_subgoal
from hwm_director.models.director_manager import DEFAULT_MAX_SUBGOAL_DISTANCE
from hwm_director.models.high_level_value import HighLevelValueModel

BcFallback = Callable[[np.ndarray, np.ndarray], np.ndarray]
ReachabilityFn = Callable[[np.ndarray, np.ndarray], float]

SCORE_EPS = 1e-8
REACHABILITY_NORM_ZSCORE = "candidate_zscore"
REACHABILITY_NORM_RAW = "raw"
REACHABILITY_NORMALIZATIONS = (REACHABILITY_NORM_ZSCORE, REACHABILITY_NORM_RAW)


def candidate_set_zscore(
    values: np.ndarray, *, eps: float = SCORE_EPS
) -> np.ndarray:
    """Z-score over the current candidate set: ``(x - mean) / (std + eps)``.

    ``std`` is the population standard deviation (NumPy default). A
    constant set becomes all zeros (``eps`` in the denominator).
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 0.0
    return (arr - mean) / (std + float(eps))


def combined_reachability_scores(
    q_values: np.ndarray,
    reach_errors: np.ndarray,
    *,
    lambda_reach: float,
    normalization: str = REACHABILITY_NORM_ZSCORE,
    eps: float = SCORE_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``score = normalized_q - lambda * normalized_reach_error``.

    ``candidate_zscore`` (default):
        each term is z-scored over the **current** candidate set so raw
        ``Q_H`` units are not mixed with meters.
    ``raw``:
        uses ``Q_H`` and meters as-is (diagnostic only).
    """
    mode = str(normalization)
    if mode not in REACHABILITY_NORMALIZATIONS:
        raise ValueError(
            f"unknown reachability normalization {mode!r}; "
            f"expected one of {REACHABILITY_NORMALIZATIONS}"
        )
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    err = np.asarray(reach_errors, dtype=np.float64).reshape(-1)
    if q.shape != err.shape:
        raise ValueError("q_values and reach_errors must have the same length")
    if mode == REACHABILITY_NORM_ZSCORE:
        q_n = candidate_set_zscore(q, eps=eps)
        err_n = candidate_set_zscore(err, eps=eps)
    else:
        q_n = q
        err_n = err
    scores = q_n - float(lambda_reach) * err_n
    return scores, q_n, err_n


class ValueHighLevelPolicy:
    """``pi_H`` that selects among retrieved candidates using ``Q_H``."""

    def __init__(
        self,
        value_model: HighLevelValueModel,
        normalizer: StateNormalizer,
        candidate_index: SubgoalCandidateIndex,
        bc_fallback: BcFallback,
        *,
        candidate_state_radius: float = DEFAULT_CANDIDATE_RADIUS,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
        retrieval_mode: str = "xy",
        max_source_state_distance: float | None = None,
        state_distance_weight: float = DEFAULT_STATE_DISTANCE_WEIGHT,
        xy_distance_weight: float = DEFAULT_XY_DISTANCE_WEIGHT,
        reachability_fn: ReachabilityFn | None = None,
        max_predicted_subgoal_error: float | None = None,
        reachability_score_weight: float = 0.0,
        reachability_score_normalization: str = REACHABILITY_NORM_ZSCORE,
        use_soft_reachability: bool = False,
    ) -> None:
        self.value_model = value_model
        self.normalizer = normalizer
        self.candidate_index = candidate_index
        self.bc_fallback = bc_fallback
        self.candidate_state_radius = float(candidate_state_radius)
        self.n_candidates = int(n_candidates)
        self.max_subgoal_distance = float(max_subgoal_distance)
        self.retrieval_mode = str(retrieval_mode)
        self.max_source_state_distance = max_source_state_distance
        self.state_distance_weight = float(state_distance_weight)
        self.xy_distance_weight = float(xy_distance_weight)
        self.reachability_fn = reachability_fn
        self.max_predicted_subgoal_error = max_predicted_subgoal_error
        self.reachability_score_weight = float(reachability_score_weight)
        self.reachability_score_normalization = str(reachability_score_normalization)
        self.use_soft_reachability = bool(use_soft_reachability) or (
            self.reachability_score_weight != 0.0
        )
        self.n_queries = 0
        self.n_fallback = 0
        self.last_n_candidates = 0
        self.last_used_fallback = False
        self.last_diagnostics: dict[str, Any] = {}
        self.value_model.eval()

    def reset_stats(self) -> None:
        self.n_queries = 0
        self.n_fallback = 0
        self.last_n_candidates = 0
        self.last_used_fallback = False
        self.last_diagnostics = {}

    def retrieve_candidate_records(self, state: np.ndarray) -> list[dict]:
        return self.candidate_index.retrieve_records(
            np.asarray(state, dtype=np.float64),
            mode=self.retrieval_mode,
            radius=self.candidate_state_radius,
            n_candidates=self.n_candidates,
            max_subgoal_distance=self.max_subgoal_distance,
            max_source_state_distance=self.max_source_state_distance,
            state_distance_weight=self.state_distance_weight,
            xy_distance_weight=self.xy_distance_weight,
            normalizer=self.normalizer,
        )

    def retrieve_candidates(self, state: np.ndarray) -> np.ndarray:
        records = self.retrieve_candidate_records(state)
        if not records:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([r["future_xy"] for r in records], axis=0)

    def score_candidates(
        self, state: np.ndarray, candidates: np.ndarray, final_goal: np.ndarray
    ) -> np.ndarray:
        """``Q_H`` scores in raw-input / train-normalized coordinates."""
        state_b = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
        goal_b = np.asarray(final_goal, dtype=np.float64).reshape(ACHIEVED_GOAL_DIM)
        cand = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
        n = cand.shape[0]
        states = np.repeat(state_b[None, :], n, axis=0)
        goals = np.repeat(goal_b[None, :], n, axis=0)
        states_n = self.normalizer.normalize(states)
        cand_n = normalize_subgoal(cand, self.normalizer)
        goals_n = normalize_subgoal(goals, self.normalizer)
        with torch.no_grad():
            scores = self.value_model(
                torch.as_tensor(states_n, dtype=torch.float32),
                torch.as_tensor(cand_n, dtype=torch.float32),
                torch.as_tensor(goals_n, dtype=torch.float32),
            )
        return scores.cpu().numpy().reshape(-1)

    def _empty_score_diagnostics(self) -> dict[str, Any]:
        nan = float("nan")
        return {
            "qh_score_normalized": nan,
            "reach_error_normalized": nan,
            "combined_score": nan,
            "reachability_score_weight": float(self.reachability_score_weight),
            "qh_max": nan,
            "qh_min": nan,
            "qh_mean": nan,
            "reach_error_max": nan,
            "reach_error_min": nan,
            "reach_error_mean": nan,
        }

    def _fallback(self, state: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
        self.n_fallback += 1
        self.last_used_fallback = True
        self.last_n_candidates = 0
        self.last_diagnostics = {
            "used_fallback": True,
            "source_state_distance": float("nan"),
            "source_xy_distance": float("nan"),
            "predicted_subgoal_error": float("nan"),
            "qh_score": float("nan"),
            "n_candidates": 0,
            **self._empty_score_diagnostics(),
        }
        return np.asarray(self.bc_fallback(state, final_goal), dtype=np.float64)

    def _fill_predicted_errors(self, state: np.ndarray, records: list[dict]) -> None:
        if self.reachability_fn is None:
            for rec in records:
                rec["predicted_subgoal_error"] = float("nan")
            return
        for rec in records:
            rec["predicted_subgoal_error"] = float(
                self.reachability_fn(state, rec["future_xy"])
            )

    def select_subgoal(self, state: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
        """``pi_H(h_tau, g*) -> g_tau`` in raw meters."""
        self.n_queries += 1
        records = self.retrieve_candidate_records(state)
        hard_filter = (
            (not self.use_soft_reachability)
            and self.reachability_fn is not None
            and self.max_predicted_subgoal_error is not None
        )
        if hard_filter:
            kept: list[dict] = []
            for rec in records:
                pred_err = float(self.reachability_fn(state, rec["future_xy"]))
                rec["predicted_subgoal_error"] = pred_err
                if pred_err <= float(self.max_predicted_subgoal_error):
                    kept.append(rec)
            records = kept
        elif self.use_soft_reachability:
            self._fill_predicted_errors(state, records)
        else:
            for rec in records:
                rec["predicted_subgoal_error"] = float("nan")
        self.last_n_candidates = int(len(records))
        if not records:
            return self._fallback(state, final_goal)
        candidates = np.stack([r["future_xy"] for r in records], axis=0)
        q_raw = self.score_candidates(state, candidates, final_goal)
        extra = self._empty_score_diagnostics()
        extra["qh_max"] = float(np.max(q_raw))
        extra["qh_min"] = float(np.min(q_raw))
        extra["qh_mean"] = float(np.mean(q_raw))
        ranking = q_raw
        q_n = np.full(q_raw.shape, float("nan"))
        err_n = np.full(q_raw.shape, float("nan"))
        combined = q_raw
        if self.use_soft_reachability:
            errors = np.asarray(
                [float(r["predicted_subgoal_error"]) for r in records],
                dtype=np.float64,
            )
            extra["reach_error_max"] = float(np.max(errors))
            extra["reach_error_min"] = float(np.min(errors))
            extra["reach_error_mean"] = float(np.mean(errors))
            combined, q_n, err_n = combined_reachability_scores(
                q_raw,
                errors,
                lambda_reach=self.reachability_score_weight,
                normalization=self.reachability_score_normalization,
            )
            ranking = combined
        best_i = int(np.argmax(ranking))
        best = records[best_i]
        self.last_used_fallback = False
        extra["qh_score_normalized"] = float(q_n[best_i])
        extra["reach_error_normalized"] = float(err_n[best_i])
        extra["combined_score"] = float(combined[best_i])
        extra["reachability_score_weight"] = float(self.reachability_score_weight)
        self.last_diagnostics = {
            "used_fallback": False,
            "source_state_distance": float(
                best.get("source_state_distance", float("nan"))
            ),
            "source_xy_distance": float(best.get("source_xy_distance", float("nan"))),
            "predicted_subgoal_error": float(
                best.get("predicted_subgoal_error", float("nan"))
            ),
            "qh_score": float(q_raw[best_i]),
            "n_candidates": int(len(records)),
            "source_episode_id": int(best.get("source_episode_id", -1)),
            "source_t": int(best.get("source_t", -1)),
            **extra,
        }
        return np.asarray(best["future_xy"], dtype=np.float64)

    def __call__(self, state: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
        return self.select_subgoal(state, final_goal)

    def stats(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "n_fallback": self.n_fallback,
            "fallback_rate": (
                float(self.n_fallback / self.n_queries) if self.n_queries else float("nan")
            ),
            "last_n_candidates": self.last_n_candidates,
            "last_used_fallback": self.last_used_fallback,
            "retrieval_mode": self.retrieval_mode,
            "reachability_score_weight": self.reachability_score_weight,
            "use_soft_reachability": self.use_soft_reachability,
        }


def director_reachability_error(director: Any) -> ReachabilityFn:
    """Predicted ``||f_H(s, g)[:2] - g||`` using the controller's ``f_H``.

    Director: ``f_H = (f_L, pi_L)^K``. HWM: ``f_H = f_H_phi``.
    """

    def _error(state: np.ndarray, subgoal: np.ndarray) -> float:
        predicted = director.high_level_transition(state, subgoal)
        g = np.asarray(subgoal, dtype=np.float64).reshape(2)
        return float(np.linalg.norm(predicted[:2] - g))

    return _error
