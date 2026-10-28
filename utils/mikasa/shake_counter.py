"""Counting shakes of an inverted shaker (SeasonDish, owner 2026-10-27).

The shaker is turned upside down over the bowl and shaken; since 2026-10-28 each shake swings it back toward
upright and returns (checklist D2 leaves the wrist no room past upside down), and after the last shake it is
turned upright again and held still.

The counter works on the lean from straight down, the angle between the shaker's up-axis and world -Z
(0 = exactly upside down, 180 = upright), so it needs neither the wrist axis nor a direction:

- the counter is *armed* once the held shaker has been over the bowl within `return_deg` of upside down;
- an armed shaker that then leans `out_deg` or more over the bowl has *swung*;
- a swung shaker that comes back within `return_deg` is one shake.

A shake counts when it comes back, so turning the shaker over (never armed on the way down) and turning it
upright again after the last shake (never comes back) are not shakes. Excursions while the shaker is not
held over the bowl neither count nor arm.
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


def step_shakes(count: torch.Tensor, armed: torch.Tensor, swung: torch.Tensor, lean_deg: torch.Tensor,
                held_over_bowl: torch.Tensor, out_deg: float, return_deg: float
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One control step of the counter: the new (count, armed, swung). Call once per step.

    Example (one env; turned over, two shakes, then turned upright again):
        >>> c, a, s = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
        >>> held = torch.ones(1, dtype=torch.bool)
        >>> for lean in (180., 90., 40., 5., 30., 15., 3., 30., 2., 60., 120., 180.):
        ...     c, a, s = step_shakes(c, a, s, torch.tensor([lean]), held, 20., 10.)
        >>> int(c[0]), bool(s[0])
        (2, True)

        Not over the bowl, nothing counts:
        >>> c2, _, _ = step_shakes(c, a, torch.ones(1, dtype=torch.bool), torch.tensor([2.]), torch.zeros(1, dtype=torch.bool), 20., 10.)
        >>> int(c2[0])
        2
    """
    assert 0.0 <= return_deg < out_deg, (return_deg, out_deg)
    at_rest = held_over_bowl & (lean_deg <= return_deg)
    out = held_over_bowl & (lean_deg >= out_deg)
    swung = swung | (armed & out)
    count = count + (swung & at_rest).to(count.dtype)
    return count, armed | at_rest, swung & ~at_rest
