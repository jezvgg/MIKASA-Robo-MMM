"""Counting shakes of an inverted shaker (SeasonDish, owner 2026-10-27).

The owner's rule: the shaker is turned upside down over the bowl and then shaken, and "a tilt, back to zero,
a tilt the other way" is two shakes. So a shake is one excursion away from upside down, either side.

The counter works on the lean from straight down, the angle between the shaker's up-axis and world -Z
(0 = exactly upside down, 180 = upright), so it needs neither the wrist axis nor a direction:

- the counter is *armed* once the held shaker is over the bowl and leans at most `return_deg`;
- an armed shaker that leans `out_deg` or more over the bowl is one shake, and disarms the counter;
- the next shake counts only after the shaker has come back within `return_deg`.

Turning the shaker over the bowl for the first time arms the counter and is not a shake (it was never
armed on the way down). Excursions while the shaker is not held over the bowl neither count nor re-arm.
"""
import math

import torch


def lean_from_down_deg(tilt_rad: torch.Tensor) -> torch.Tensor:
    """Lean from straight down, degrees, from the tilt of the up-axis from world +Z (radians).

    Example:
        >>> [round(float(v), 1) for v in lean_from_down_deg(torch.tensor([math.pi, math.radians(150.), 0.]))]
        [0.0, 30.0, 180.0]
    """
    return 180.0 - torch.rad2deg(tilt_rad)


def step_shakes(count: torch.Tensor, armed: torch.Tensor, lean_deg: torch.Tensor, held_over_bowl: torch.Tensor,
                out_deg: float, return_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One control step of the counter: the new (count, armed). Call once per step.

    Example (one env; the shaker turned over, then +30, back, -30, back):
        >>> c, a = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.bool)
        >>> held = torch.ones(1, dtype=torch.bool)
        >>> for lean in (180., 90., 40., 5., 30., 15., 3., 30., 2.):
        ...     c, a = step_shakes(c, a, torch.tensor([lean]), held, 20., 10.)
        >>> int(c[0]), bool(a[0])
        (2, True)

        Not over the bowl, nothing counts:
        >>> c2, _ = step_shakes(c, a, torch.tensor([40.]), torch.zeros(1, dtype=torch.bool), 20., 10.)
        >>> int(c2[0])
        2
    """
    assert 0.0 <= return_deg < out_deg, (return_deg, out_deg)
    at_rest = held_over_bowl & (lean_deg <= return_deg)
    out = held_over_bowl & (lean_deg >= out_deg)
    count = count + (armed & out).to(count.dtype)
    armed = (armed & ~out) | at_rest
    return count, armed
