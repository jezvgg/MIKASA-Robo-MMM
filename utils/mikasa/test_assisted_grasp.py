"""Assisted grasp: the rule without a simulator, then the weld inside MikasaSeasonDish-v0,
and the `AssistedGraspTask` wiring of SeasonDish, SameDrawer and CabinetSearch.

The simulator tests need mani_skill with the RoboCasa assets and run on CPU; they are
skipped where those are missing (the Mac host). Run them in the planner container.
"""
import numpy as np
import pytest

from utils.mikasa.assisted_grasp import GraspWindow, pick_candidate

DT = 0.01  # 100 Hz physics, the ManiSkill default


def run(window, steps, applying=True, candidate=0):
    events = []
    for _ in range(steps):
        ev = window.tick(applying, candidate)
        if ev is not None:
            events.append(ev)
    return events


def test_attaches_after_the_window_and_not_before():
    w = GraspWindow(DT)
    assert run(w, 30) == []                  # tick 1 arms the counter, 29 more: 0.29 s
    assert run(w, 1) == [("establish", 0)]   # 0.30 s
    assert w.attached and w.held == 0


def test_losing_the_candidate_restarts_the_window():
    w = GraspWindow(DT)
    run(w, 20)
    run(w, 1, candidate=None)
    assert run(w, 30) == []
    assert run(w, 1) == [("establish", 0)]


def test_switching_bodies_restarts_the_window():
    w = GraspWindow(DT)
    run(w, 20, candidate=0)
    # The switch tick resets, the next one arms, then 30 more are needed.
    assert run(w, 31, candidate=1) == []
    assert run(w, 1, candidate=1) == [("establish", 1)]


def test_an_open_command_never_attaches():
    w = GraspWindow(DT)
    assert run(w, 100, applying=False) == []
    assert w.held is None


def test_release_is_immediate_and_regrasp_waits_a_thirtieth():
    w = GraspWindow(DT)
    run(w, 31)
    assert run(w, 1, applying=False) == [("release", 0)]
    assert not w.attached
    # 1/30 s = 3.33 steps: still blocked for three more ticks, free on the fourth.
    run(w, 3, applying=True)
    assert w.held is not None
    run(w, 1, applying=True)
    assert w.held is None
    assert run(w, 31) == [("establish", 0)]


def test_checkpoint_state_round_trips():
    w = GraspWindow(DT)
    run(w, 12)
    copy = GraspWindow(DT)
    copy.load(w.state())
    assert run(copy, 19) == run(w, 19) == [("establish", 0)]


def test_push_from_outside_is_not_a_candidate():
    # Both fingers touch the prop, no pad ray passes through it: a closed hand pushing.
    assert pick_candidate(touch=[2], ray_hit=[False], dist=[0.05]) is None


def test_one_finger_is_not_enough():
    assert pick_candidate(touch=[1], ray_hit=[True], dist=[0.05]) is None
    assert pick_candidate(touch=[2], ray_hit=[True], dist=[0.05]) == 0


def test_nearest_one_finger_body_blocks_a_farther_grasp():
    # OmniGibson's order, kept on purpose: nearest eligible first, fingers counted after.
    assert pick_candidate(touch=[1, 2], ray_hit=[True, True], dist=[0.02, 0.06]) is None
    assert pick_candidate(touch=[0, 2], ray_hit=[True, True], dist=[0.02, 0.06]) == 1


# ---------------------------------------------------------------- simulator --

#: The objects each task welds, by kind: free bodies by name, and how many articulation
#: links (doors or drawers) come with them.
TASKS = {"MikasaSeasonDish-v0": (("shaker", "condiment_bottle"), 0),
         "MikasaSameDrawer-v0": (("apple",), 4),
         "MikasaCabinetSearch-v0": (("cube",), 4)}


#: As the planners and the collection profiles build them: the pinned kitchen (SameDrawer
#: draws one per env otherwise, and not every kitchen has its drawers) and 20/100 Hz.
MAKE_KW = dict(num_envs=1, robot_uids="ds_fetch", control_mode="pd_joint_pos", obs_mode="state",
               sim_backend="cpu", scene_idx=0, sim_config={"control_freq": 20, "sim_freq": 100})


def _make_env(assisted, env_id="MikasaSeasonDish-v0"):
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    try:
        import my_scenes  # noqa: F401  (registers the benchmark environments)
        env = gym.make(env_id, assisted_grasp=assisted, **MAKE_KW)
        env.reset(seed=0)
    except Exception as exc:  # no RoboCasa assets, no renderer, ...
        pytest.skip(f"{env_id} cannot be built here: {exc!r}")
    return env


@pytest.fixture(scope="module")
def ag_env():
    env = _make_env(True)
    yield env
    env.close()


def _action(env, gripper, torso_delta=0.0, yaw_rate=0.0):
    """ds_fetch pd_joint_pos: [arm 7, gripper 1, head pan, head tilt, torso, forward, yaw]."""
    ctrls = env.unwrapped.agent.controller.controllers
    arm = ctrls["arm"].qpos[0].cpu().numpy()
    body = ctrls["body"].qpos[0].cpu().numpy().copy()
    body[-1] += torso_delta
    return np.hstack([arm, [gripper], body, [0.0, yaw_rate]]).astype(np.float32)


def _torso_room(env):
    """+0.02 m per step if the torso can rise, -0.02 if it is near its top stop."""
    torso = float(env.unwrapped.agent.controller.controllers["body"].qpos[0, -1])
    return 0.02 if torso < 0.2 else -0.02


def _float_prop_at_hand(task, i, ahead=0.12):
    """Put prop i in free air ahead of the hand, where a weld cannot drag it into anything."""
    import sapien
    import torch

    hand = task._ag.hand.pose.sp
    p = hand.p + ahead * hand.to_transformation_matrix()[:3, 0]
    prop = task._ag.objects[i]
    prop.set_pose(sapien.Pose(p, [1, 0, 0, 0]))
    prop.set_linear_velocity(torch.zeros((1, 3)))
    prop.set_angular_velocity(torch.zeros((1, 3)))


def _rel(task, i):
    return task._ag.hand.pose.sp.inv() * task._ag.objects[i].pose.sp


@pytest.mark.parametrize("env_id", sorted(TASKS))
def test_off_by_default_adds_nothing(env_id):
    env = _make_env(False, env_id)
    task = env.unwrapped
    try:
        assert task._ag is None
        assert "ag_window" not in task.get_state_dict()
        assert "ag_held" not in task.get_info()
    finally:
        env.close()


@pytest.mark.parametrize("env_id", sorted(TASKS))
def test_each_task_welds_its_own_objects_and_checkpoints_it(env_id):
    """The wiring: the rule sees the task's graspable actors, reports in info, survives a
    checkpoint, and a reset releases it."""
    env = _make_env(True, env_id)
    task = env.unwrapped
    try:
        names, n_links = TASKS[env_id]
        from utils.mikasa.assisted_grasp import _is_link
        assert [o.name for o in task._ag.objects if not _is_link(o)] == list(names)
        assert sum(_is_link(o) for o in task._ag.objects) == n_links
        assert int(task.get_info()["ag_held"][0]) == -1
        free = next(i for i, o in enumerate(task._ag.objects) if not _is_link(o))
        env.step(_action(env, -1.0))
        _float_prop_at_hand(task, free)
        task._ag.attach(free)
        state = task.get_state_dict()
        assert int(task.get_info()["ag_held"][0]) == free
        env.reset(seed=0)
        assert task._ag.held_index() == -1 and _enabled(task) == []
        task.set_state_dict(state)
        assert task._ag.held_index() == free and _enabled(task) == [free]
    finally:
        env.close()


def test_weld_follows_the_hand_then_lets_go_on_open(ag_env):
    task = ag_env.unwrapped
    ag_env.reset(seed=0)
    ag_env.step(_action(ag_env, -1.0))               # gripper commanded closed: applying
    _float_prop_at_hand(task, 0)
    task._ag.attach(0)
    before = _rel(task, 0)
    z0, h0 = float(task._ag.objects[0].pose.p[0, 2]), float(task._ag.hand.pose.p[0, 2])
    delta = _torso_room(ag_env)
    for _ in range(15):
        ag_env.step(_action(ag_env, -1.0, torso_delta=delta))
    after = _rel(task, 0)
    moved = float(task._ag.objects[0].pose.p[0, 2]) - z0
    hand = float(task._ag.hand.pose.p[0, 2]) - h0
    assert abs(hand) > 0.03 and np.sign(hand) == np.sign(delta), f"the hand moved only {hand:+.3f} m"
    assert abs(moved - hand) < 3e-3, f"the hand moved {hand:+.3f} m and the prop {moved:+.3f} m"
    assert np.linalg.norm(after.p - before.p) < 2e-3, "the weld slipped"
    assert int(task.get_info()["ag_held"][0]) == 0
    ag_env.step(_action(ag_env, 1.0))                # fully open: released on the next step
    assert int(task.get_info()["ag_held"][0]) == -1
    assert task._ag.events[-1][1:] == ("release", task._ag.objects[0].name)
    z_release = float(task._ag.objects[0].pose.p[0, 2])
    for _ in range(20):
        ag_env.step(_action(ag_env, 1.0))
    assert float(task._ag.objects[0].pose.p[0, 2]) < z_release - 0.05, "released prop should fall"


def _angle_deg(a, b) -> float:
    """Rotation between two sapien poses' orientations, degrees."""
    w = abs(float((a.inv() * b).q[0]))
    return float(np.degrees(2 * np.arccos(min(1.0, w))))


def test_weld_holds_orientation_when_the_hand_turns(ag_env):
    """A weld locks rotation too. Turning the base swings the hand about the vertical;
    a joint that locked only translation would leave the prop facing the old way
    (the first version did: its cone limit did not lock the swing axes)."""
    task = ag_env.unwrapped
    ag_env.reset(seed=0)
    ag_env.step(_action(ag_env, -1.0))
    _float_prop_at_hand(task, 0)
    task._ag.attach(0)
    before, hand0 = _rel(task, 0), task._ag.hand.pose.sp
    for _ in range(15):
        ag_env.step(_action(ag_env, -1.0, yaw_rate=0.6))
    turned = _angle_deg(hand0, task._ag.hand.pose.sp)
    assert turned > 10, f"the hand turned only {turned:.1f} deg"
    drift = _angle_deg(before, _rel(task, 0))
    assert drift < 1.0, f"the hand turned {turned:.1f} deg, the prop lagged {drift:.1f} deg"
    assert np.linalg.norm(_rel(task, 0).p - before.p) < 2e-3, "the weld slipped"


def test_checkpoint_keeps_the_weld(ag_env):
    task = ag_env.unwrapped
    ag_env.reset(seed=0)
    ag_env.step(_action(ag_env, -1.0))
    _float_prop_at_hand(task, 1)
    task._ag.attach(1)
    state = task.get_state_dict()
    assert int(np.asarray(state["ag_window"]).reshape(-1)[0]) == 1
    ag_env.reset(seed=0)                             # episode reset releases the weld
    assert task._ag.held_index() == -1
    task.set_state_dict(state)
    assert task._ag.held_index() == 1
    assert np.allclose(task._ag.rel_pose, np.asarray(state["ag_rel_pose"]).reshape(-1), atol=1e-6)


def _enabled(task):
    return [i for i, c in enumerate(task._ag._drives) if c.is_enabled]


def test_a_state_without_the_rule_drops_the_weld(ag_env):
    """`env.set_state` restores actors and articulations only; the old weld must not
    survive it and drag the restored prop to the hand."""
    task = ag_env.unwrapped
    ag_env.reset(seed=0)
    state = {k: v for k, v in task.get_state_dict().items() if not k.startswith("ag_")}
    ag_env.step(_action(ag_env, -1.0))
    _float_prop_at_hand(task, 0)
    task._ag.attach(0)
    assert _enabled(task) == [0]
    task.set_state_dict(state)
    assert task._ag.held_index() == -1 and _enabled(task) == []


def test_attach_replaces_the_previous_weld(ag_env):
    task = ag_env.unwrapped
    ag_env.reset(seed=0)
    ag_env.step(_action(ag_env, -1.0))
    _float_prop_at_hand(task, 0)
    task._ag.attach(0)
    _float_prop_at_hand(task, 1, ahead=0.20)
    task._ag.attach(1)
    assert task._ag.held_index() == 1 and _enabled(task) == [1]


def test_closing_on_a_prop_attaches_by_the_rule(ag_env):
    """The whole rule, no attach(): contacts, pad rays and the 0.3 s window on ds_fetch.

    The first version never fired inside the planner — every ray started inside a finger
    and gave up before leaving it — and attach()-based tests could not see that.
    """
    import sapien
    import torch

    task = ag_env.unwrapped
    ag_env.reset(seed=0)
    prop = task._ag.objects[0]
    body = prop._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    body.disable_gravity = True                      # hang it between the open pads
    try:
        hand = task._ag.hand.pose.sp
        tcp = (np.asarray(task._ag.fingers[0].pose.sp.p) + np.asarray(task._ag.fingers[1].pose.sp.p)) / 2
        prop.set_pose(sapien.Pose(tcp, [1, 0, 0, 0]))   # upright, centred between the pads
        prop.set_linear_velocity(torch.zeros((1, 3)))
        prop.set_angular_velocity(torch.zeros((1, 3)))
        for _ in range(4):
            ag_env.step(_action(ag_env, 1.0))
        assert task._ag.held_index() == -1, "attached with the gripper open"
        for _ in range(20):                          # close; contact, then 0.3 s = 6 steps
            ag_env.step(_action(ag_env, -1.0))
        assert ("establish", prop.name) in [e[1:] for e in task._ag.events]
        assert task._ag.held_index() == 0
    finally:
        body.disable_gravity = False
