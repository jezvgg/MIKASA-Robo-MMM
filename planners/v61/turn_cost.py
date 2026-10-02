"""Cross-attempt turn memory: a retried approach should not start by turning back against the last turn of the previous one."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import TURN_FLIP_COST
from .turning import end_signs


@dataclass
class TurnMemory:
    """Direction (+1 left, -1 right, 0 none yet) of the last counted turn of the episode's base motion."""
    last_sign: float = 0.0


MEMORY = TurnMemory()   # ponytail: one episode per process; pass the memory through find_stand if episodes ever share a process


def reset() -> None:
    """Forget the turns of the previous episode."""
    MEMORY.last_sign = 0.0


def note_sign(sign: float) -> None:
    """Record the direction of the last turn (no-op for 0)."""
    if sign:
        MEMORY.last_sign = float(sign)


def note_log(servo_log: Sequence[Sequence[float]]) -> None:
    """Take the last turn from the servo log rows (attempt, step, x, y, yaw, ...) of the approaches so far."""
    if len(servo_log) > 4:
        note_sign(end_signs(np.array(servo_log, dtype=float)[::2, 2:5])[1])


def flip_cost(first_sign: float) -> float:
    """TURN_FLIP_COST when an approach starting with a turn of `first_sign` reverses the remembered turn, else 0."""
    return TURN_FLIP_COST if MEMORY.last_sign * first_sign < 0 else 0.0
