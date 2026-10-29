"""Consolidated, behavior-preserving collection regression tests."""


# Source group: test_roll_paths.py
from types import SimpleNamespace

import numpy as np
import pytest

from planners.oracle.paths import RollPathPlanner


def make_guard():
    owner = SimpleNamespace(
        _roll_indices=[8, 10, 12], _roll_low=np.zeros(3), _roll_high=np.zeros(3)
    )
    native = SimpleNamespace(
        joint_limits=np.tile([-6.28, 6.28], (15, 1)),
        move_group_joint_indices=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12],
        fold_qpos=lambda q: q.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None),
    )
    return RollPathPlanner(native, owner), native, owner


@pytest.mark.parametrize(
    "method", ["plan_qpos_line", "plan_screw", "plan_qpos", "plan_pose"]
)
def test_rejects_intermediate_winding_with_valid_endpoint(method):
    guard, native, _ = make_guard()
    path = np.zeros((3, 11))
    path[1, 8] = 4.0
    native_result = {"status": "Success", "position": path}
    setattr(native, method, lambda *a, **kw: native_result)
    current = np.zeros(15)
    goal = [current] if method == "plan_qpos" else current
    saved_limits = native.joint_limits.copy()
    assert guard.__getattr__(method)(goal, current)["status"] != "Success"
    np.testing.assert_array_equal(native.joint_limits, saved_limits)
    assert path[1, 8] == 4.0


def test_history_and_both_ends_of_same_path_count_toward_span():
    guard, _, owner = make_guard()
    path = np.zeros((2, 11))
    path[:, 8] = [-1.7, 1.7]
    assert not guard.accepts(path, move_group=True)
    path[:, 8] = 1.7
    assert guard.accepts(path, move_group=True)
    owner._roll_low[1] = -1.7
    assert not guard.accepts(path, move_group=True)


def test_rrt_can_retry_after_bad_path_without_changing_start():
    guard, native, _ = make_guard()
    current = np.zeros(15)
    paths = [np.zeros((3, 11)), np.zeros((3, 11))]
    paths[0][1, 8] = 4
    seen = []

    def plan(goal, start):
        seen.append(start.copy())
        return {"status": "Success", "position": paths[len(seen) - 1]}

    native.plan_pose = plan
    result = guard.plan_pose(None, current)
    assert result["position"] is paths[1]
    np.testing.assert_array_equal(seen, [current, current])


def test_invalid_start_not_clipped_and_limits_restored_on_exception():
    guard, native, _ = make_guard()
    current = np.zeros(15)
    original_limits = native.joint_limits

    def fail(*args):
        assert native.joint_limits[10, 1] < np.pi
        raise RuntimeError("planner failure")

    native.plan_screw = fail
    with pytest.raises(RuntimeError, match="planner failure"):
        guard.plan_screw(None, current)
    assert native.joint_limits is original_limits
    current[10] = 4
    assert guard.plan_screw(None, current)["status"] != "Success"
    assert current[10] == 4


def test_empty_joint_line_preserves_no_motion_result():
    guard, native, owner = make_guard()
    path = np.empty((0, 11))
    native_result = {"status": "Success", "position": path}
    native.plan_qpos_line = lambda *a, **kw: native_result
    assert guard.plan_qpos_line(np.zeros(15), np.zeros(15)) is native_result
    # Empty motion must not erase a violation already present in the history.
    owner._roll_high[1] = 4.0
    assert not guard.accepts(path, move_group=True)


def test_grasp_branch_rejects_intermediate_flex_crossing_and_restores_limits():
    guard, native, owner = make_guard()
    owner._grasp_branch = {9: 1, 11: 1}
    current = np.zeros(15)
    current[[9, 11]] = 1
    path = np.tile(current[native.move_group_joint_indices], (3, 1))
    path[1, 9] = -0.2
    original = native.joint_limits

    def plan(*args, **kwargs):
        if owner._grasp_branch:
            assert native.joint_limits[9, 0] == guard.margin
            assert native.joint_limits[11, 0] == guard.margin
        return {"status": "Success", "position": path}

    native.plan_screw = plan
    assert guard.plan_screw(None, current)["status"] != "Success"
    assert native.joint_limits is original
    owner._grasp_branch = {}
    assert guard.plan_screw(None, current)["status"] == "Success"


@pytest.mark.parametrize("mask_style", ["positional", "keyword", "none", "empty"])
@pytest.mark.parametrize("closest", [False, True])
def test_ik_cannot_propose_uncommanded_head_or_finger_motion(mask_style, closest):
    guard, native, _ = make_guard()
    current = np.zeros(15)
    current[[4, 6, 13, 14]] = [.2, -.3, .05, .05]
    supplied = np.zeros(15, dtype=bool)
    supplied[3] = True
    saved = supplied.copy()

    def random_restart(goal, start, mask, **kwargs):
        # Model a random restart: only masked joints retain the measured start.
        candidate = np.full(15, .4)
        candidate[mask] = start[mask]
        return "Success", candidate if kwargs["return_closest"] else [candidate]

    native.IK = random_restart
    args, kwargs = (), {"return_closest": closest}
    if mask_style == "positional":
        args = (supplied,)
    elif mask_style == "keyword":
        kwargs["mask"] = supplied
    elif mask_style == "empty":
        kwargs["mask"] = []
    status, candidates = guard.IK(None, current, *args, **kwargs)
    assert status == "Success"
    q = candidates if closest else candidates[0]
    np.testing.assert_array_equal(q[[4, 6, 13, 14]], current[[4, 6, 13, 14]])
    if mask_style in {"positional", "keyword"}:
        assert q[3] == current[3]
    assert q[5] == .4  # The commanded arm remains free.
    np.testing.assert_array_equal(supplied, saved)


def test_joint_line_checks_real_open_fingers_after_hypothetical_closed_preview():
    guard, native, _ = make_guard()
    current = np.zeros(15)
    current[13:] = .05
    model = np.zeros(15)  # Previous preview left fingers closed.

    def set_model(q, full):
        assert full
        model[:] = q

    def joint_line(goal, start):
        model[native.move_group_joint_indices] = goal[native.move_group_joint_indices]
        # A narrow gap passes closed fingers but intersects the measured open jaw.
        blocked = np.any(model[13:] > .03)
        return {"status": "finger collision" if blocked else "Success",
                "position": np.zeros((2, 11))}

    native.robot.set_qpos = set_model
    native.plan_qpos_line = joint_line
    assert native.plan_qpos_line(current, current)["status"] == "Success"
    assert guard.plan_qpos_line(current, current)["status"] == "finger collision"
    np.testing.assert_array_equal(current[13:], [.05, .05])


def test_reference_starts_only_after_random_ik_fails_and_stay_bounded():
    guard, native, owner = make_guard()
    current = np.zeros(15)
    random_goal = np.zeros(15)
    random_goal[12] = 3.5  # outside the roll window: rejected
    near = np.zeros(15)
    near[12] = 1.0
    starts = []

    def ik(goal, start, mask=None, *, n_init_qpos=20):
        starts.append((start.copy(), n_init_qpos))
        if n_init_qpos == 1 and start[5] == -0.9:
            return "Success", [near]
        if n_init_qpos == 1:
            return "Success", [random_goal]
        return "Success", [random_goal]

    native.IK = ik
    # No references: the unchanged refusal.
    assert guard.IK(None, current, n_init_qpos=40)[0] != "Success"
    assert len(starts) == 1

    owner._ik_reference_arms = [("a", {5: -0.9}), ("b", {5: 0.6})]
    owner._ik_reference_used = []
    status, goals = guard.IK(None, current, n_init_qpos=40)
    assert status == "Success"
    np.testing.assert_array_equal(np.atleast_2d(goals), [near])
    assert owner._ik_reference_used == ["a"]
    assert [n for _, n in starts[1:]] == [40, 1, 1]
    assert starts[2][0][5] == -0.9 and starts[3][0][5] == 0.6
    np.testing.assert_array_equal(current, np.zeros(15))

    # A random-restart solution inside the window still wins; no reference call.
    starts.clear()
    native.IK = lambda goal, start, mask=None, *, n_init_qpos=20: (
        starts.append(n_init_qpos) or ("Success", [near]))
    assert guard.IK(None, current, n_init_qpos=40)[0] == "Success"
    assert starts == [40]


def make_branch_guard():
    guard, native, owner = make_guard()
    joints = {name: SimpleNamespace(active_index=np.array([index]))
              for name, index in (("shoulder_lift_joint", 7), ("elbow_flex_joint", 9),
                                  ("wrist_flex_joint", 11))}
    owner.env_agent = SimpleNamespace(robot=SimpleNamespace(active_joints_map=joints))
    owner._goal_branch = {9: -1, 11: 1}
    owner.reports = []
    owner._report = lambda stage, **fields: owner.reports.append((stage, fields))
    return guard, native, owner


def branch_path(elbow_end, *, elbow_start=0.9):
    # move-group columns: shoulder_lift 5, elbow_flex 7, wrist_flex 9
    path = np.zeros((3, 11))
    path[:, 7] = [elbow_start, 0.3, elbow_end]
    path[:, 9] = [2.0, 1.6, 1.3]
    return path


def test_goal_branch_checks_only_the_endpoint():
    guard, native, owner = make_branch_guard()
    native.plan_screw = lambda *a, **kw: {"status": "Success", "position": branch_path(-0.3)}
    result = guard.plan_screw(None, np.zeros(15), masked_joints=np.ones(15, dtype=bool))
    assert result["status"] == "Success" and result["position"][-1, 7] == -0.3
    assert owner.reports == []


def test_wrong_goal_branch_screw_retries_with_shoulder_lift_held():
    guard, native, owner = make_branch_guard()
    masks = []

    def plan(goal, start, **kw):
        masks.append(np.array(kw["masked_joints"]))
        return {"status": "Success", "position": branch_path(0.2 if len(masks) == 1 else -0.3)}

    native.plan_screw = plan
    mask = np.ones(15, dtype=bool)
    mask[:3] = False
    result = guard.plan_screw(None, np.zeros(15), masked_joints=mask)
    assert result["position"][-1, 7] == -0.3
    assert masks[0][7] and not masks[1][7]
    np.testing.assert_array_equal(np.delete(masks[1], 7), np.delete(mask, 7))
    assert owner.reports == [("goal_branch", {"held": "shoulder_lift_joint", "knots": 3})]


def test_wrong_goal_branch_is_refused_when_the_held_screw_does_not_fix_it():
    guard, native, _ = make_branch_guard()
    native.plan_screw = lambda *a, **kw: {"status": "Success", "position": branch_path(0.2)}
    result = guard.plan_screw(None, np.zeros(15), masked_joints=np.ones(15, dtype=bool))
    assert result["status"] != "Success"
    native.plan_pose = lambda *a, **kw: {"status": "Success", "position": branch_path(0.2)}
    assert guard.plan_pose(None, np.zeros(15))["status"] != "Success"


def test_goal_branch_filters_ik_candidates():
    guard, native, owner = make_branch_guard()
    wrong, right = np.zeros(15), np.zeros(15)
    wrong[[9, 11]] = [0.2, 1.3]
    right[[9, 11]] = [-0.3, 1.3]
    native.IK = lambda *a, **kw: ("Success", [wrong, right])
    status, goals = guard.IK(None, np.zeros(15))
    assert status == "Success" and len(goals) == 1 and goals[0][9] == -0.3
    owner._goal_branch = {}
    assert len(guard.IK(None, np.zeros(15))[1]) == 2


# Source group: test_search_budget.py
import time
import pytest
from planners.season_dish_planner import SearchBudget, PlanningBudgetExceeded

def test_nested_planning_counts_once_and_excludes_physics():
    now=[0.];b=SearchBudget(120,clock=lambda:now[0])
    with b.measure('outer'):
        now[0]+=2
        with b.measure('inner'):now[0]+=3
    now[0]+=70  # physics and loading outside planning scopes
    with b.measure('next'):now[0]+=4
    assert b.used==9 and not b.exhausted

def test_budget_interrupts_search_and_remains_exhausted():
    b=SearchBudget(.02)
    with pytest.raises(PlanningBudgetExceeded):
        with b.measure('slow'):time.sleep(.2)
    assert b.exhausted
    with pytest.raises(PlanningBudgetExceeded):
        with b.measure('retry'):pass

def test_ik_initializations_are_capped():
    b=SearchBudget()
    observed=[]
    fn=b.wrap(lambda **kw:observed.append(kw['n_init_qpos']),'IK',ik=True)
    fn(n_init_qpos=160);fn(n_init_qpos=8)
    assert observed==[32,8]


def test_keepout_propagates_budget_failure_without_second_yield():
    from types import SimpleNamespace
    from planners.oracle.oracle_season_dish import keepout
    planner=SimpleNamespace(planner=SimpleNamespace(planning_world=SimpleNamespace()))
    with pytest.raises(PlanningBudgetExceeded,match='spent'):
        with keepout(planner,[]):
            raise PlanningBudgetExceeded('spent')


# Source group: test_release_settling.py
import numpy as np
from planners.oracle.paths import ReleaseSettling


def test_continuous_motion_and_arm_motion_reset_stable_window():
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    check = ReleaseSettling(pose, .05)
    for _ in range(3):
        assert not check.update(pose, np.zeros(7))
    pose[0] += .001
    assert not check.update(pose, np.zeros(7))
    assert check.stable_steps == 0
    for _ in range(3):
        assert not check.update(pose, np.zeros(7))
    assert not check.update(pose, np.full(7, .06))
    for _ in range(3):
        assert not check.update(pose, np.zeros(7))
    assert check.update(pose, np.zeros(7))


def test_quaternion_sign_is_same_pose_but_rotation_is_motion():
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    check = ReleaseSettling(pose, .05)
    for _ in range(4):
        pose[3:] *= -1
        check.update(pose, np.zeros(7))
    assert check.ready
    pose[3:] = [np.cos(.03), 0., 0., np.sin(.03)]
    assert not check.update(pose, np.zeros(7))
    assert check.angular_speed > .5


def test_persistent_jitter_never_becomes_ready_by_timeout():
    pose = np.array([0., 0., 1., 1., 0., 0., 0.])
    check = ReleaseSettling(pose, .05)
    for i in range(20):
        pose[2] = 1. + .001 * (i % 2)
        assert not check.update(pose, np.zeros(7))


# Source group: test_path_clearance.py
from types import SimpleNamespace

import numpy as np
import pytest

from planners.oracle.paths import path_clear


def planning_world(collisions):
    q = np.zeros(2)
    world = SimpleNamespace(
        set_qpos_all=lambda value: np.copyto(q, value),
        is_state_colliding=lambda: bool(collisions(q)),
        check_collision=lambda: [SimpleNamespace(link_name1=a, link_name2=b)
                                 for a, b in collisions(q)],
    )
    native = SimpleNamespace(
        move_group_joint_indices=[0, 1], fold_qpos=lambda value: value.copy(),
        robot=SimpleNamespace(set_qpos=lambda *args: None), planning_world=world)
    return SimpleNamespace(planner=native)


def test_thin_upper_handle_between_clear_endpoints_blocks_path():
    planner = planning_world(lambda q: [('forearm', 'upper_cabinet_handle')]
                             if .48 <= q[0] <= .52 else [])
    assert not path_clear(planner, np.zeros(2), [[0, 0], [1, 0]])


def test_clear_detour_passes_with_every_fixture_present():
    planner = planning_world(lambda q: [('forearm', 'upper_cabinet_handle')]
                             if .48 <= q[0] <= .52 and q[1] < .2 else [])
    assert path_clear(planner, np.zeros(2), [[0, 0], [0, 1], [1, 1], [1, 0]])


@pytest.mark.parametrize('when', ['start', 'later', 'persistent'])
def test_support_exception_is_only_for_the_starting_state(when):
    pair = ('payload', 'counter')
    def collisions(q):
        hit = q[0] == 0 if when == 'start' else q[0] > .5 if when == 'later' else True
        return [pair] if hit else []
    planner = planning_world(collisions)
    assert path_clear(planner, np.zeros(2), [[0, 0], [1, 0]],
                      initial_contacts=[pair]) == (when == 'start')


def test_support_exception_does_not_allow_robot_upper_handle_contact():
    planner = planning_world(lambda q: [('payload', 'counter'), ('wrist', 'upper_handle')]
                             if q[0] == 0 else [])
    assert not path_clear(planner, np.zeros(2), [[0, 0], [1, 0]],
                          initial_contacts=[('payload', 'counter')])


# Source group: test_roll_goal_preference.py
from types import SimpleNamespace

import numpy as np

from planners.oracle.collection_solver import CollectionMotionPlanner
from robots.fetch.extand import FetchMotionPlanningSapienSolver


def test_roll_preference_tries_next_candidate_after_a_collision(monkeypatch):
    planner = object.__new__(CollectionMotionPlanner)
    planner._roll_indices = [8, 10, 12]
    planner._roll_low = np.array([0.68, -0.12, 0.0])
    planner._roll_high = np.array([0.70, -0.10, 0.02])
    planner.planner = SimpleNamespace(joint_limits=np.tile([-6.28, 6.28], (15, 1)))
    current = np.zeros(15)
    current[planner._roll_indices] = planner._roll_high
    goals = np.tile(current, (3, 1))
    # These are the measured handle probe's high-turn and two bounded solutions.
    goals[:, planner._roll_indices] = [
        [3.46, -2.97, 1.15],
        [-1.93, 2.47, 0.57],
        [0.195, 0.373, 1.15],
    ]
    attempted = []

    def inherited_line(self, pose, offered, *args, **kwargs):
        attempted.append(np.asarray(offered)[0])
        return None if len(attempted) == 1 else "executed"

    monkeypatch.setattr(
        FetchMotionPlanningSapienSolver, "_line_to_ik_goals", inherited_line
    )
    result = planner._line_to_ik_goals(
        None, goals, current, current, 0, 1, None, None, "test"
    )
    assert result == "executed"
    np.testing.assert_array_equal(attempted, goals[[2, 1]])


# Source group: test_rest_paths.py
import numpy as np
from planners.oracle.paths import monotone_knots


def test_curved_rest_stays_between_endpoints_without_joint_reversals():
    start = np.array([.2, -.4, 1.6, .3, -2.1])
    goal = np.array([.2, .8, -.2, .386, 0.])
    path = monotone_knots(start, goal, np.array([1., .5, 4., 2., 1.]))
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_allclose(path[-1], goal, atol=1e-15)
    assert np.all(path >= np.minimum(start, goal) - 1e-12)
    assert np.all(path <= np.maximum(start, goal) + 1e-12)
    assert np.all(np.diff(path, axis=0) * (goal-start) >= -1e-12)
    assert np.all(path[:, 0] == start[0])


# Source group: test_smooth_base.py
import numpy as np
from planners.oracle.paths import rounded_route


def test_rounding_preserves_endpoints_and_monotone_forward_distance():
    start=np.array([2.3,-1.16]);corner=np.array([.8,-2.1]);end=np.array([.5,-1.5])
    route=rounded_route(start,corner,end,.4)
    np.testing.assert_allclose(route['points'][[0,-1]],[start,end])
    assert np.all(np.diff(route['distance'])>0)
    assert np.max(np.abs(np.diff(route['yaw'])))<.05
    assert route['distance'][-1] < np.linalg.norm(corner-start)+np.linalg.norm(end-corner)
    assert np.min(np.linalg.norm(route['points']-corner,axis=1))>.03
    assert abs(route['curvature'][0])<1e-6
    assert abs(route['curvature'][-1])<1e-6


def test_tiny_corner_does_not_generate_an_unchecked_fallback():
    assert rounded_route([0,0],[0,.01],[1,1],.4) is None
