"""A valid endpoint does not establish that a full path obeys D2."""

from types import SimpleNamespace

import numpy as np
import pytest

from planners.oracle.roll_paths import RollPathPlanner


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
