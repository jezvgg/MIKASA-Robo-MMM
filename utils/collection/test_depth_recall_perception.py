"""Perception mode's planning world follows the belief (`PropBelief.bind_world`).

Needs mplib, mani_skill, the RoboCasa assets and a renderer (the cameras); skipped elsewhere.
"""
import numpy as np
import pytest
import sapien


@pytest.fixture(scope="module")
def world():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    pytest.importorskip("mplib")
    try:
        import my_scenes  # noqa: F401
        from planners import depth_recall_v1_perception as perc
        from planners.oracle import oracle_common as common
        env = gym.make("MikasaDepthRecall-v1", num_envs=1, robot_uids="ds_fetch",
                       control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
        env.reset(seed=0)
        belief = perc.PropBelief(env.unwrapped)
        solver = common.default_planner_factory(env, False, False)
    except Exception as exc:
        pytest.skip(f"cannot build the perception planner here: {exc!r}")
    belief.bind_world(solver)
    yield env, belief, solver
    env.close()


def _pose_of(world, name):
    return world.get_object(name).pose


def test_unseen_props_are_parked_and_seen_ones_follow_the_belief(world):
    env, belief, solver = world
    w = solver.planner.planning_world
    names = belief.world_names
    solver.planner.update_from_simulation()
    for name in names:                      # nothing seen yet: every prop out of reach
        assert float(np.asarray(_pose_of(w, name).p)[2]) < -5.0
    true = np.asarray(env.unwrapped.props[0].pose.sp.p, dtype=np.float64)
    belief.est[0] = dict(x=float(true[0]) + 0.03, y=float(true[1]), z=float(true[2]), yaw=0.0)
    solver.planner.update_from_simulation()
    got = np.asarray(_pose_of(w, names[0]).p, dtype=np.float64)
    assert np.allclose(got, true + [0.03, 0.0, 0.0], atol=1e-6), "the world re-read the simulator"


def _gripper(env, g, steps=15):
    t = env.unwrapped
    ctrls = t.agent.controller.controllers
    for _ in range(steps):
        arm = ctrls["arm"].qpos[0].cpu().numpy()
        body = ctrls["body"].qpos[0].cpu().numpy()
        env.step(np.hstack([arm, [g], body, [0.0, 0.0]]).astype(np.float32))


def test_holding_is_read_from_the_command_and_the_finger_encoders(world):
    env, belief, _ = world
    assert not belief.fingers_holding()     # rest pose, 0.015 per finger, nothing commanded
    _gripper(env, -1.0)                     # shut on nothing: the fingers meet
    assert not belief.fingers_holding()
    _gripper(env, 1.0)


def test_closing_box_is_the_space_between_the_pads():
    """Pure geometry: two pads 10 cm apart along y, overlapping 4 cm along z."""
    pytest.importorskip("sapien")
    from planners.depth_recall_v1_perception import closing_box, count_between

    a = ((-0.01, -0.06, -0.03), (0.01, -0.05, 0.03))
    b = ((-0.01, 0.05, -0.01), (0.01, 0.06, 0.05))
    lo, hi, axis = closing_box(a, b)
    assert axis == 1
    assert np.allclose(lo, [-0.01, -0.05, -0.01]) and np.allclose(hi, [0.01, 0.05, 0.03])
    G = np.eye(4)
    G[:3, 3] = [1.0, 2.0, 3.0]
    pts = np.array([[1.0, 2.0, 3.01],      # between the pads
                    [1.0, 2.0, 3.04],      # past the shorter pad
                    [1.0, 2.055, 3.01]])   # inside a pad, not between
    n, centre = count_between(pts, G, lo, hi, inset=0.003)
    assert n == 1 and np.allclose(centre, [0.0, 0.0, 0.01])


def test_between_pads_refuses_a_hand_with_nothing_in_it(world):
    """The live check at the rest pose, far from every prop: no prop points, no close."""
    env, belief, solver = world
    for i in range(len(env.unwrapped.props)):
        chk = belief.between_pads(i)
        assert not chk["ok"] and chk["n"] == 0, chk
        assert chk["gap_mm"] > 0
