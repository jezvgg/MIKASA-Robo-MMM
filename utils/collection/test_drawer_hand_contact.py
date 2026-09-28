"""A nested drawer-removal preview must not erase the caller's hand allowance.

`contact_stroke` removes the selected drawer model and adds it back. The installed
mplib drops that model's allowed-collision entries on re-add, so a withdrawal
planned afterwards inside `hand_contact` saw finger/drawer contact again, refused
the arm-only screw and fell back to a torso rise that pulled the drawer open.
"""

import gymnasium as gym
import numpy as np
import sapien


def test_moving_push_keeps_enclosing_hand_allowance():
    import my_scenes  # noqa: F401
    from mplib.pymp.collision_detection import AllowedCollision
    from my_scenes.same_drawer import DRAWER_ART_SUFFIX, DRAWER_FIXTURES
    from planners import same_drawer_approach as approach
    from planners.oracle import oracle_common as common

    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, sim_backend="cpu",
                   obs_mode="state", control_mode="pd_joint_pos",
                   sim_config={"control_freq": 20, "sim_freq": 100})
    try:
        env.reset(seed=8000000)
        task = env.unwrapped
        planner = common.collection_planner_factory(env, False, False)
        planner.planner.update_from_simulation()

        def entries():
            acm, pairs = approach._hand_pairs(planner, 0)
            assert pairs
            return {acm.get_entry(a, b) for a, b in pairs}

        before = entries()
        with approach.hand_contact(planner, task, 0):
            with common.contact_stroke(planner, [DRAWER_FIXTURES[0] + DRAWER_ART_SUFFIX]):
                pass
            # Documents the mplib behaviour the preview must compensate for.
            assert entries() == {None}
        assert entries() == before

        current = task.agent.robot.get_qpos()[0].cpu().numpy().astype(float)
        tcp = task.agent.tcp.pose[0].sp
        push = sapien.Pose(tcp.p + [0, .02, 0], tcp.q)
        with approach.hand_contact(planner, task, 0):
            approach.moving_drawer_push(planner, task, current, push, 0)
            assert entries() == {AllowedCollision.ALWAYS}
        assert entries() == before
        assert np.allclose(task.agent.robot.get_qpos()[0].cpu().numpy(), current)
    finally:
        env.close()
