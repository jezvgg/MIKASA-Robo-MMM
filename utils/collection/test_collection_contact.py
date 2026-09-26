"""Identically named links in different drawers must not share contact status."""

from types import SimpleNamespace

import torch

from planners.oracle.collection_solver import CollectionMotionPlanner


def test_articulation_contact_uses_the_selected_link_instance():
    selected = SimpleNamespace(name="inner_box")
    other = SimpleNamespace(name="inner_box")
    finger = object()
    queries = []

    def impulses(hand, target):
        queries.append(target)
        return torch.tensor([[0.1 if target is other else 0.0, 0.0, 0.0]])

    planner = object.__new__(CollectionMotionPlanner)
    planner.env_agent = SimpleNamespace(
        finger1_link=finger,
        finger2_link=object(),
        robot=SimpleNamespace(links_map={"gripper_link": object()}),
    )
    planner.base_env = SimpleNamespace(
        scene=SimpleNamespace(get_pairwise_contact_impulses=impulses)
    )
    assert not planner.gripper_touching(SimpleNamespace(get_links=lambda: [selected]))
    assert all(target is selected for target in queries)
    assert planner.gripper_touching(SimpleNamespace(get_links=lambda: [other]))
