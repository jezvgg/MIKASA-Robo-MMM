"""IK backends (`robots/fetch/ik_backend.py`) on the real Fetch planner.

The default must be mplib's IK unchanged; ``MIKASA_IK_STARTS=fixed`` must answer the same
request the same way and leave mplib's generator alone; the cuRobo client's plumbing
(frames, joint order, seeding) is checked against a stand-in server, so no GPU is needed.
Needs mplib, mani_skill and the RoboCasa assets (the planner container); skipped elsewhere.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
       "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")

STUB = r'''
import json, os, sys
print("CUROBO " + json.dumps({"id": 0, "ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["op"] == "names":
        out = {"names": json.loads(os.environ["STUB_NAMES"])}
    else:
        with open(os.environ["STUB_SEEN"], "w") as f:
            json.dump(req["args"], f)
        out = {"names": json.loads(os.environ["STUB_NAMES"]), "q": [json.loads(os.environ["STUB_Q"])],
               "ok": [True], "ms": 1.0}
    out["id"] = req["id"]
    print("CUROBO " + json.dumps(out), flush=True)
'''


@pytest.fixture(scope="module")
def planner():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    pytest.importorskip("mplib")
    try:
        import my_scenes  # noqa: F401  (registers the benchmark environments)
        env = gym.make("MikasaDepthRecall-v1", num_envs=1, robot_uids="ds_fetch",
                       control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
        env.reset(seed=0)
        from planners.oracle import oracle_common as common
        solver = common.default_planner_factory(env, False, False)
    except Exception as exc:  # no RoboCasa assets, no mplib build, ...
        pytest.skip(f"the Fetch planner cannot be built here: {exc!r}")
    # The simulator's posture, as the planners pass it. Not `planner.robot.get_qpos()`:
    # mplib leaves its model at the last configuration it checked, joints wrapped.
    sim = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0]
    solver.planner._test_q0 = np.asarray(solver.planner.fold_qpos(sim), dtype=np.float64)
    yield solver.planner
    env.close()


def _socket_path(tmp_path, request):
    """A Unix socket path under the 107-byte limit: pytest's tmp_path is often longer."""
    import os
    import shutil
    import tempfile

    path = tmp_path / "s.sock"
    if len(str(path)) < 100:
        return path
    short = tempfile.mkdtemp(prefix="sk", dir=os.path.expanduser("~"))
    request.addfinalizer(lambda: shutil.rmtree(short, ignore_errors=True))
    return Path(short) / "s.sock"


def _col(names, suffix):
    hits = [i for i, n in enumerate(names) if n == suffix or n.endswith("_" + suffix)]
    assert len(hits) == 1, (suffix, names)
    return hits[0]


def _problem(p, seed=0):
    """(goal pose, start qpos, mask holding all but the arm, target arm joints)."""
    names = list(p.user_joint_names)
    q0 = p._test_q0.copy()
    arm = [_col(names, j) for j in ARM]
    rng = np.random.default_rng(seed)
    move = p.move_group_joint_indices
    for _ in range(200):                 # a nearby arm posture that collides with nothing
        target = q0.copy()
        target[arm] += rng.uniform(-0.15, 0.15, len(arm))
        p.planning_world.set_qpos_all(target[move])
        if not p.planning_world.check_collision():
            break
    else:
        pytest.skip("no collision-free perturbed posture found")
    link = p.link_name_2_idx[p.move_group]
    p.pinocchio_model.compute_forward_kinematics(target)
    goal = p.pinocchio_model.get_link_pose(link)
    mask = np.ones(len(names), dtype=bool)
    mask[arm] = False
    return goal, q0, mask, target[arm]


def _same_angle(a, b, atol=1e-6):
    """Equal up to whole turns: mplib wraps revolute joints into [lower, lower + 2*pi)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.allclose((d + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=atol)


def _dist(p, goal, q):
    p.pinocchio_model.compute_forward_kinematics(np.asarray(q, dtype=np.float64))
    return goal.distance(p.pinocchio_model.get_link_pose(p.link_name_2_idx[p.move_group]))


def test_default_is_mplib_unchanged(planner, monkeypatch):
    import mplib

    monkeypatch.delenv("MIKASA_IK_STARTS", raising=False)
    monkeypatch.delenv("MIKASA_IK_BACKEND", raising=False)
    # Problem 0 is one mplib's 8 random restarts miss (its own posture is collision-free),
    # problem 1 one they solve: the default must match mplib on both.
    for seed in (0, 1):
        goal, q0, mask, _ = _problem(planner, seed)
        mplib.set_global_seed(5)
        ours = planner.IK(goal, q0, mask, n_init_qpos=8)
        mplib.set_global_seed(5)
        theirs = mplib.Planner.IK(planner, goal, q0, mask, n_init_qpos=8)
        assert ours[0] == theirs[0]
        assert (ours[1] is None) == (theirs[1] is None)
        if ours[1] is not None:
            assert len(ours[1]) == len(theirs[1])
            assert all(np.array_equal(a, b) for a, b in zip(ours[1], theirs[1]))


def test_fixed_starts_repeat_and_leave_mplib_rng_alone(planner, monkeypatch):
    import mplib

    monkeypatch.setenv("MIKASA_IK_STARTS", "fixed")
    goal, q0, mask, _ = _problem(planner, seed=1)
    mplib.set_global_seed(7)
    first = planner.IK(goal, q0, mask, n_init_qpos=12)
    after = planner.pinocchio_model.get_random_configuration()
    mplib.set_global_seed(7)
    untouched = planner.pinocchio_model.get_random_configuration()
    assert np.array_equal(after, untouched), "fixed starts drew from mplib's generator"
    mplib.set_global_seed(123)                       # a different generator state changes nothing
    planner.__dict__.pop("_mikasa_ik_requests", None)   # a new episode: the request is new again
    second = planner.IK(goal, q0, mask, n_init_qpos=12)
    assert first[0] == "Success" and second[0] == "Success"
    assert len(first[1]) == len(second[1])
    assert all(np.array_equal(a, b) for a, b in zip(first[1], second[1]))
    assert all(_dist(planner, goal, q) < 1e-3 for q in first[1])
    assert all(_same_angle(q[mask], q0[mask]) for q in first[1]), "a masked joint moved"


def test_curobo_client_frames_and_seed_on_a_stub(planner, monkeypatch, tmp_path):
    from robots.fetch import curobo_client

    goal, q0, mask, target_arm = _problem(planner, seed=2)
    stub, seen = tmp_path / "stub.py", tmp_path / "seen.json"
    stub.write_text(STUB)
    monkeypatch.setenv("MIKASA_IK_BACKEND", "curobo")
    monkeypatch.delenv("MIKASA_IK_STARTS", raising=False)
    monkeypatch.setenv("MIKASA_CUROBO_PY", sys.executable)
    monkeypatch.setenv("MIKASA_CUROBO_SERVER", str(stub))
    monkeypatch.setenv("STUB_NAMES", json.dumps(list(ARM)))
    monkeypatch.setenv("STUB_Q", json.dumps([float(v) for v in target_arm]))
    monkeypatch.setenv("STUB_SEEN", str(seen))
    curobo_client._SERVER.clear()
    try:
        status, goals = planner.IK(goal, q0, mask, n_init_qpos=4)
    finally:
        s = curobo_client._SERVER.pop("s", None)
        if s is not None:
            s.close()
        curobo_client._SERVER.clear()
    args = json.loads(seen.read_text())
    names = list(planner.user_joint_names)
    arm = [_col(names, j) for j in ARM]
    # torso masked -> the arm variant, goal in torso_lift_link's frame at the start qpos
    assert args["variant"] == "arm"
    links = list(planner.link_name_2_idx)
    torso = planner.link_name_2_idx[links[_col(links, "torso_lift_link")]]
    planner.pinocchio_model.compute_forward_kinematics(q0)
    T_torso = curobo_client._matrix(planner.pinocchio_model.get_link_pose(torso))
    expect = np.linalg.inv(T_torso) @ curobo_client._matrix(goal)
    assert np.allclose(np.asarray(args["goal"]), expect, atol=1e-9)
    assert np.allclose(args["seed"], q0[arm])
    # The proposal is an exact answer and passes FCL. The 7-joint arm is redundant: CLIK
    # from the start posture may land on a neighbouring solution first, and mplib's own
    # 0.1 rad uniqueness rule then keeps that one — so the answer is within 0.1 rad.
    assert status == "Success"

    def turn(d):
        return (np.asarray(d) + np.pi) % (2 * np.pi) - np.pi

    assert min(np.linalg.norm(turn(g[arm] - target_arm)) for g in goals) < 0.1


def test_fixed_starts_retry_takes_the_next_block(planner, monkeypatch):
    from robots.fetch import ik_backend

    monkeypatch.setenv("MIKASA_IK_STARTS", "fixed")
    planner.__dict__.pop("_mikasa_ik_requests", None)
    goal, q0, mask, _ = _problem(planner, seed=3)
    assert ik_backend.retry_index(planner, goal, q0, mask) == 0
    assert ik_backend.retry_index(planner, goal, q0, mask) == 1
    a, b = ik_backend.fixed_starts(planner, 5, 0), ik_backend.fixed_starts(planner, 5, 1)
    assert not np.allclose(a, b)
    assert np.array_equal(ik_backend.fixed_starts(planner, 5, 1), b)


def test_curobo_client_on_a_shared_socket(planner, monkeypatch, tmp_path, request):
    """The shared-server path: connect to a socket instead of starting a process."""
    import socket
    import threading

    from robots.fetch import curobo_client

    goal, q0, mask, target_arm = _problem(planner, seed=2)
    path = _socket_path(tmp_path, request)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen()
    asked = []

    def serve():
        conn, _ = srv.accept()
        with conn, conn.makefile("r") as rd, conn.makefile("w") as wr:
            wr.write("CUROBO " + json.dumps({"id": 0, "ready": True}) + "\n")
            wr.flush()
            for line in rd:
                req = json.loads(line)
                asked.append(req["op"])
                out = {"id": req["id"], "names": list(ARM)}
                if req["op"] == "propose":
                    out.update(q=[[float(v) for v in target_arm]], ok=[True], ms=1.0)
                wr.write("CUROBO " + json.dumps(out) + "\n")
                wr.flush()

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setenv("MIKASA_IK_BACKEND", "curobo")
    monkeypatch.delenv("MIKASA_IK_STARTS", raising=False)
    monkeypatch.delenv("MIKASA_CUROBO_PY", raising=False)
    monkeypatch.setenv("MIKASA_CUROBO_SOCKET", str(path))
    curobo_client._SERVER.clear()
    try:
        status, goals = planner.IK(goal, q0, mask, n_init_qpos=4)
    finally:
        s = curobo_client._SERVER.pop("s", None)
        if s is not None:
            s.close()
        curobo_client._SERVER.clear()
        srv.close()
    assert asked == ["names", "propose"]
    assert status == "Success" and goals


def test_scan_sends_goals_in_each_stands_frame_and_shapes_the_mask(monkeypatch, tmp_path, request):
    """No simulator: the scan's frames and mask layout against a stand-in socket server."""
    import socket
    import threading

    from robots.fetch import curobo_client

    path = _socket_path(tmp_path, request)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen()
    seen = {}

    def serve():
        conn, _ = srv.accept()
        with conn, conn.makefile("r") as rd, conn.makefile("w") as wr:
            wr.write("CUROBO " + json.dumps({"id": 0, "ready": True}) + "\n")
            wr.flush()
            for line in rd:
                req = json.loads(line)
                goals = np.asarray(req["args"]["goals"]).reshape(-1, 4, 4)
                seen["goals"] = goals
                ok = (goals[:, 0, 3] > 0).tolist()          # "reachable" = in front of the stand
                out = {"id": req["id"], "ok": ok, "q": [[0.0] * 8 for _ in ok], "names": [], "ms": 1.0}
                wr.write("CUROBO " + json.dumps(out) + "\n")
                wr.flush()

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setenv("MIKASA_CUROBO_SOCKET", str(path))
    curobo_client._SERVER.clear()
    goal = np.eye(4)
    goal[:3, 3] = [1.0, 0.0, 0.9]
    stands = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, -1.0, np.pi / 2)]
    try:
        ok, q = curobo_client.scan([goal], stands, verify_hits=False)   # the stand-in's q is not a solution
    finally:
        s = curobo_client._SERVER.pop("s", None)
        if s is not None:
            s.close()
        curobo_client._SERVER.clear()
        srv.close()
    assert ok.shape == (3, 1) and q.shape == (3, 1, 8)
    assert ok[:, 0].tolist() == [True, False, True]          # behind the second stand; ahead of the third
    assert np.allclose(seen["goals"][2][:3, 3], [1.0, 0.0, 0.9])   # stand 3 faces +y from (1, -1)


def test_near_first_drops_far_branches_and_orders_nearest():
    """BIMBO_EvoX's jump rule on cuRobo's proposals: no simulator, no server."""
    from robots.fetch.curobo_client import near_first

    seed = np.zeros(3)
    q = np.array([[0.5, 0.0, 0.0], [0.1, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.3, 0.3]])
    ok = np.array([True, True, True, False])
    assert near_first(q, ok, seed, 1.2) == ([1, 0], 1)           # 1.5 rad on one joint: another branch
    assert near_first(q, ok, seed, 0.0) == ([1, 0, 2], 0)        # 0 = off
    assert near_first(q, np.zeros(4, bool), seed, 1.2) == ([0, 1, 3], 0)   # nothing solved: plain starts


def test_scan_verify_keeps_true_hits_and_drops_false_ones():
    """The float64 re-check against the URDF: a true hit stays, a pose 2 cm off and a joint
    past its limit go, a miss stays a miss."""
    from robots.fetch import curobo_client

    c = curobo_client.chain()
    names = list(c.joint_names)
    assert names[0] == "torso_lift_joint" and len(names) == 8
    lo, hi = c.limits(names)
    rng = np.random.default_rng(0)
    Q = rng.uniform(np.maximum(lo, -3.0), np.minimum(hi, 3.0), size=(4, len(names)))
    goals = c.fk_many(Q, names)
    goals[1, 0, 3] += 0.02
    Q[2, 0] = hi[0] + 0.05
    goals[2] = c.fk_many(Q[2:3], names)[0]
    ok = np.array([True, True, True, False])
    before = curobo_client.STATS["verify_dropped"]
    out = curobo_client.verify(goals, Q, ok, names)
    assert out.tolist() == [True, False, False, False]
    assert curobo_client.STATS["verify_dropped"] - before == 2


def test_urdf_fk_matches_the_planners_model(planner):
    """`urdf_fk` against mplib's pinocchio model of the same URDF, base_link -> gripper_link."""
    from robots.fetch import curobo_client

    c = curobo_client.chain()
    names = list(planner.user_joint_names)
    links = list(planner.link_name_2_idx)
    grip = planner.link_name_2_idx[links[_col(links, "gripper_link")]]
    base = planner.link_name_2_idx[links[_col(links, "base_link")]]
    lo, hi = c.limits(c.joint_names)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = planner._test_q0.copy()
        vals = rng.uniform(np.maximum(lo, -3.0), np.minimum(hi, 3.0))
        for n, v in zip(c.joint_names, vals):
            q[_col(names, n)] = v
        planner.pinocchio_model.compute_forward_kinematics(q)
        T = (np.linalg.inv(curobo_client._matrix(planner.pinocchio_model.get_link_pose(base)))
             @ curobo_client._matrix(planner.pinocchio_model.get_link_pose(grip)))
        assert np.allclose(c.fk(dict(zip(c.joint_names, vals))), T, atol=1e-6)
