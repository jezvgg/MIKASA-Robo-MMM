"""Search-rule counterexamples; physical replay is tested by the campaign."""

import math

import pytest
import torch

from my_scenes.cabinet_search import (
    CabinetSearchConfig, SearchLatches, at_home_predicate,
    compartment_layout, step_search_latches,
)


def fixture(target):
    cfg = CabinetSearchConfig()
    layout = compartment_layout(cfg.compartments)
    state = SearchLatches.zeros(1, 4, 0)
    state.cube_cab[:] = target
    step = 0

    def tick(opened=None, home=False, moved=0.):
        nonlocal step
        step += 1
        theta = torch.zeros(1, 4)
        if opened is not None:
            theta[0, opened] = 1.3
        info, _ = step_search_latches(
            state, step=torch.tensor([step]), theta=theta,
            hinge_still=torch.ones(1, 4, dtype=torch.bool),
            tcp_far=torch.ones(1, 4, dtype=torch.bool),
            at_home=torch.tensor([home]), foreign_hit=torch.tensor([False]),
            layout=layout, cfg=cfg, moved_m=torch.tensor([moved]),
        )
        return info

    return tick, state


@pytest.mark.parametrize("target", range(4))
@pytest.mark.parametrize("moved", [.02, .03])
def test_nudge_ends_at_can_without_final_return(target, moved):
    tick, state = fixture(target)
    tick(home=True)
    info = tick(target, moved=moved)
    assert info["found"].item() and info["success"].item()
    assert not info["at_home"].item()


@pytest.mark.parametrize("home_between", [False, True])
def test_completed_inspection_cannot_be_repeated(home_between):
    tick, _ = fixture(3)
    tick(home=True)
    tick(0)
    for _ in range(10):
        tick()
    if home_between:
        tick(home=True)
    info = tick(0)
    assert info["reopened"].item() and info["fail"].item()


def test_next_inspection_requires_home_even_after_empty_cabinet_closed():
    tick, _ = fixture(3)
    tick(home=True)
    tick(0)
    for _ in range(10):
        tick()
    info = tick(1)
    assert info["skipped_home"].item() and info["fail"].item()


def test_random_home_uses_recorded_marker_not_config_center():
    cfg = CabinetSearchConfig()
    home = torch.tensor([[cfg.home_xy[0] + .25, cfg.home_xy[1]]])
    yaw = torch.tensor([math.radians(cfg.home_yaw_deg)])
    speed = torch.zeros(1)
    assert at_home_predicate(home, yaw, speed, cfg, home).item()
    assert not at_home_predicate(home, yaw, speed, cfg).item()


@pytest.mark.parametrize("target", range(4))
@pytest.mark.parametrize("moved", [0., .019])
def test_reveal_or_insufficient_push_does_not_finish(target, moved):
    tick, _ = fixture(target)
    tick(home=True)
    assert not tick(target, moved=moved)["success"].item()


@pytest.mark.parametrize("target", range(4))
def test_push_without_initial_home_fails(target):
    tick, _ = fixture(target)
    info = tick(target, moved=.03)
    assert info["fail"].item() and not info["success"].item()
