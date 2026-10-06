"""Side carry owns its object-space goal; front-facing WP14 is fallback-only."""
import ast
from pathlib import Path

import numpy as np
import sapien
from planners.myrobocasa_takeitback_tray_planner import _side_skip_goal


def test_side_goal_uses_actual_grasp_without_forcing_cup_or_wrist_rotation():
    tcp = sapien.Pose([0.60, 0.0, 1.10])
    cup = sapien.Pose([0.62, 0.01, 1.08])
    tray = sapien.Pose([0.8, 0.4, 0.9])
    goal, cup_goal = _side_skip_goal(tcp, cup, tray, [0.2, 0.1, 0.01], [0.04, 0.04, 0.05])
    restored_cup = sapien.Pose(goal.p, goal.q) * (tcp.inv() * cup)
    np.testing.assert_allclose(restored_cup.p, [0.8, 0.4, 1.11], atol=1e-6)
    np.testing.assert_allclose(restored_cup.to_transformation_matrix(), cup_goal.to_transformation_matrix(), atol=1e-6)
    # Preserve actual grasp orientation even though tray sits diagonally to base.
    np.testing.assert_allclose(cup_goal.to_transformation_matrix()[:3, :3], cup.to_transformation_matrix()[:3, :3], atol=1e-6)
    np.testing.assert_allclose(goal.q, tcp.q, atol=1e-6)


def test_side_skip_does_not_recheck_attached_cup_path_in_stale_original_world():
    source = Path(__file__).parents[1] / 'planners/myrobocasa_takeitback_tray_planner.py'
    tree = ast.parse(source.read_text())
    side = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_side_skip_to_tray')
    calls = {n.func.id for n in ast.walk(side) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert 'plan_forward_rrt' in calls
    assert '_choose_shortest_safe_plan' not in calls


def test_front_facing_waypoint14_has_no_direct_skip_call():
    source = Path(__file__).parents[1] / 'planners/myrobocasa_takeitback_tray_planner.py'
    tree = ast.parse(source.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'attempt_waypoint14']
    assert len(calls) == 1 and not calls[0].args
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_side_skip_to_tray' for n in ast.walk(tree)), 'Skip must plan in object space, separately from WP14'
