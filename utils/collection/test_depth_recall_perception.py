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


def test_cross_rungs_are_ordered_by_how_far_the_pad_is():
    """Far pads start with the hand under the shoulder; near ones keep the first rung that
    planned there (measured, 68c5f27)."""
    pytest.importorskip("mplib")
    from planners import depth_recall_v1_planner as dr

    far, near = dr.extraction_cross_rungs(0.54), dr.extraction_cross_rungs(0.36)
    assert near == dr.CROSS_RUNGS and far[0] == (None, 0.10)
    assert sorted(far, key=str) == sorted(near, key=str)          # same rungs, other order
    z0 = 0.92
    far_r, near_r = dr.restore_cross_rungs(0.54, z0), dr.restore_cross_rungs(0.18, z0)
    assert far_r[0] == (None, z0 + dr.RESTORE_CROSS_Z[0]) and near_r[0] == (dr.CROSS_SAFE_TORSO[0], None)
    assert sorted(far_r, key=str) == sorted(near_r, key=str)


def test_hue_signatures_tell_the_four_props_apart():
    """The measured signatures name a pixel of each prop's measured colour, and nothing
    between two of them or grey. Pure numpy."""
    import colorsys
    from planners.depth_recall_v1_perception import PROP_HUES
    from utils.mikasa import point_cloud as pc

    def rgb(h, s=0.9, v=0.65):
        return [int(round(c * 255)) for c in colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)]

    centres = [sig.hue_deg for sig in PROP_HUES]
    gaps = [min(abs(a - b), 360 - abs(a - b)) for k, a in enumerate(centres) for b in centres[k + 1:]]
    assert min(gaps) > 2 * max(sig.max_gap_deg for sig in PROP_HUES), "bands overlap"
    px = np.array([rgb(h) for h in centres] + [rgb(h + 1.4) for h in centres]
                  + [rgb(27.0), rgb(0, s=0.1), rgb(120, v=0.05)], dtype=np.uint8)
    assert pc.hue_labels(px, PROP_HUES).tolist() == [0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1]


def test_belief_models_are_the_props_own_meshes(world):
    """The fit model, the perceived mesh and the rest height are each prop's own."""
    env, belief, _ = world
    task = env.unwrapped
    for i, prop in enumerate(task.props):
        mesh = prop.get_first_collision_mesh(to_world_frame=False)
        b = np.asarray(mesh.bounds)
        model = belief.models[i]
        assert model.shape[0] == belief.fit_cfg.model_points
        assert np.allclose(model.min(axis=0), b[0], atol=0.003)
        assert np.allclose(model.max(axis=0), b[1], atol=0.003)
        assert abs(belief.rest_lift[i] - float(task._rest_lift[i])) < 1e-6
        # with the belief at the truth, the perceived mesh is the prop's own, in place
        true = prop.pose.sp
        q = np.asarray(true.q)
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        belief.est[i] = dict(x=float(true.p[0]), y=float(true.p[1]), z=float(true.p[2]), yaw=yaw)
        got = np.asarray(belief.proxy(prop).get_first_collision_mesh().bounds)
        want = np.asarray(prop.get_first_collision_mesh(to_world_frame=True).bounds)
        assert np.allclose(got, want, atol=1e-3)
    belief.est.clear()


def test_grasp_lifts_keep_the_finger_pads_on_the_prop():
    """Short props get the ladder brought down; the bottles keep every rung."""
    pytest.importorskip("mplib")
    import dataclasses
    from types import SimpleNamespace

    from my_scenes.depth_recall_v1 import PROPS
    from planners import depth_recall_v1_planner as dr

    props = [object() for _ in PROPS]
    task = SimpleNamespace(cfg=SimpleNamespace(props=PROPS), props=props)
    for spec, prop in zip(PROPS, props):
        lifts = dr.grasp_lifts(task, prop)
        tcp = [spec.height / 2 + l for l in lifts]            # over the support
        assert len(set(lifts)) == len(lifts)
        assert all(t >= dr.WRIST_HANG + dr.WRIST_CLEAR - 1e-4 for t in tcp), (spec.label, tcp)
        assert all(t + dr.FINGER_HALF_H <= spec.height + 1e-4 for t in tcp), (spec.label, tcp)
        if spec.height >= 0.12:                    # the box the ladder was measured on, or taller
            assert lifts == dr.GRASP_LIFTS
    # the 12 cm box the ladder was measured on keeps it unchanged
    box = SimpleNamespace(cfg=SimpleNamespace(props=(dataclasses.replace(PROPS[0], height=0.12),)),
                          props=props[:1])
    assert dr.grasp_lifts(box, props[0]) == dr.GRASP_LIFTS
    assert dr.grasp_lifts(task, props[1]) == (0.0318, 0.034)
