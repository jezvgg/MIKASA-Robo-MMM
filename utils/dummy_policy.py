"""Deterministic zero-action baseline for the 13D Fetch policy contract."""

from __future__ import annotations

import numpy as np


class DummyPolicy:
    """Return one fixed absolute ``pd_joint_pos`` action on every call."""

    def __init__(self, action=None, action_dim: int = 13):
        if action is None:
            action = np.zeros(action_dim, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (action_dim,):
            raise ValueError(f"expected action shape {(action_dim,)}, got {action.shape}")
        self.action = action.copy()

    def reset(self) -> None:
        """Reset policy state; this policy has no state."""

    def act(self, observation=None) -> np.ndarray:
        """Return a copy so callers can modify it before ``env.step``."""
        return self.action.copy()

    __call__ = act
    select_action = act


if __name__ == "__main__":
    policy = DummyPolicy()
    action = policy.act()
    assert action.shape == (13,)
    assert action.dtype == np.float32
    print(action)
