"""The strict definition of a base reversal, shared by the planner (prediction) and the analysis scripts.

The base (x, y, yaw) is sampled at 10 Hz. The yaw is unwrapped; a step is *turning* when its change exceeds
TURN_DEG_PER_STEP; a turning segment is a maximal run of turning steps of one sign and counts only if its net
rotation is at least MIN_SEGMENT_DEG. A REVERSAL is a change of sign between two consecutive counted segments.
BACKWARD motion is the distance travelled against the heading.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

TURN_DEG_PER_STEP = 0.5
MIN_SEGMENT_DEG = 5.0
BACKWARD_MIN_M = 0.05


def _yaw(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.unwrap(a[:, 2] if a.ndim == 2 else a)


def segments(trace: np.ndarray, turn_deg: float = TURN_DEG_PER_STEP, min_deg: float = MIN_SEGMENT_DEG) -> List[Tuple[float, int, int, float]]:
    """Counted turning segments (sign, first step, last step, net rad) of a yaw series or an (T, 3) trace."""
    d = np.diff(_yaw(trace))
    sgn = np.where(np.abs(d) > np.deg2rad(turn_deg), np.sign(d), 0.0)
    out: List[Tuple[float, int, int, float]] = []
    i, n = 0, len(d)
    while i < n:
        if sgn[i] == 0.0:
            i += 1
            continue
        j = i
        while j + 1 < n and sgn[j + 1] == sgn[i]:
            j += 1
        net = float(d[i:j + 1].sum())
        if abs(net) >= np.deg2rad(min_deg):
            out.append((float(sgn[i]), i, j, net))
        i = j + 1
    return out


def reversal_steps(trace: np.ndarray, **kw) -> List[int]:
    """10 Hz frame at which each reversal starts."""
    segs = segments(trace, **kw)
    return [b[1] for a, b in zip(segs, segs[1:]) if a[0] != b[0]]


def backward_distance(trace: np.ndarray) -> float:
    """Metres travelled against the heading, from an (T, 3) trace."""
    t = np.asarray(trace, dtype=float)
    yaw = _yaw(t)
    mid = 0.5 * (yaw[1:] + yaw[:-1])
    d = np.diff(t[:, :2], axis=0)
    fwd = d[:, 0] * np.cos(mid) + d[:, 1] * np.sin(mid)
    return float(-fwd[fwd < 0].sum())


def end_signs(trace: np.ndarray, **kw) -> Tuple[float, float]:
    """(sign of the first, sign of the last counted turning segment) of an (T, 3) trace; 0.0 for a trace without one."""
    segs = segments(trace, **kw)
    return (segs[0][0], segs[-1][0]) if segs else (0.0, 0.0)
