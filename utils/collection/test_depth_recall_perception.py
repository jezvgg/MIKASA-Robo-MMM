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
