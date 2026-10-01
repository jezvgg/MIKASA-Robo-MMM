"""Kinematic rollout of the pose servo (no arm, no obstacles) to score a stand pose before driving to it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (PREDICT_BACKWARD_WEIGHT, PREDICT_DT, PREDICT_FAIL_COST, PREDICT_MAX_STEPS, PREDICT_PATH_WEIGHT,
                     PREDICT_REVERSAL_COST, PREDICT_ROT_WEIGHT)
from .servo import ServoState, servo_command
from .turning import backward_distance, reversal_steps


@dataclass(frozen=True)
class Prediction:
    """What driving from `pose0` to `goal` would look like."""
    steps: int
    rotation: float        # rad, sum of |dyaw|
    reversals: int
    backward: float        # m driven against the heading
    path: float            # m
    arrived: bool


def rollout(pose0: np.ndarray, goal: np.ndarray) -> Prediction:
    """Integrate the servo law as an ideal unicycle at the planner rate; metrics at the 10 Hz of the dataset."""
    pose = np.array(pose0, dtype=float)
    state = ServoState()
    trace = [pose.copy()]
    arrived = False
    for _ in range(PREDICT_MAX_STEPS):
        v, w, arrived = servo_command(pose, goal, state)
        if arrived:
            break
        pose = pose + np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), w]) * PREDICT_DT
        trace.append(pose.copy())
    t = np.array(trace)[::2]
    if len(t) < 3:
        return Prediction(len(trace), 0.0, 0, 0.0, 0.0, arrived)
    yaw = np.unwrap(t[:, 2])
    return Prediction(len(trace), float(np.abs(np.diff(yaw)).sum()), len(reversal_steps(t)), backward_distance(t),
                      float(np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1).sum()), arrived)


def stand_cost(pose0: np.ndarray, goal: np.ndarray) -> float:
    """Cost of driving from `pose0` to the stand `goal`: rotation (turns), reversals, reverse driving, path."""
    p = rollout(pose0, goal)
    miss = 0.0 if p.arrived else PREDICT_FAIL_COST
    return (PREDICT_ROT_WEIGHT * p.rotation + PREDICT_REVERSAL_COST * p.reversals + PREDICT_BACKWARD_WEIGHT * p.backward
            + PREDICT_PATH_WEIGHT * p.path + miss)
