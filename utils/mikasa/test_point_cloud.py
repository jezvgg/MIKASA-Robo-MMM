"""Point-cloud perception pieces on synthetic clouds: no simulator, no renderer."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from utils.mikasa import point_cloud as pc  # noqa: E402

HALF = (0.0225, 0.0225, 0.06)     # the Depth Recall prop
SUPPORT = 1.42


def front_face(x, y, yaw, n=60, noise=0.001, seed=0):
    """What one camera south of the prop sees: its -y face (rotated by yaw), noisy."""
    rng = np.random.default_rng(seed)
    local = np.stack([rng.uniform(-HALF[0], HALF[0], n), np.full(n, -HALF[1]),
                      rng.uniform(-HALF[2], HALF[2], n)], axis=1)
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.stack([c * local[:, 0] - s * local[:, 1] + x,
                      s * local[:, 0] + c * local[:, 1] + y,
                      local[:, 2] + SUPPORT + HALF[2]], axis=1)
    return world + rng.normal(0, noise, world.shape)


def camera_south(x, y):
    """A pinhole view from 0.6 m south of the prop, looking north (+y), as world2cam."""
    eye = np.array([x, y - 0.6, SUPPORT + HALF[2]])
    # OpenCV: z forward (+y world), x right (+x world), y down (-z world)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    K = np.array([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]])
    return T, K


def render_depth(points, T, K, size=128):
    depth = np.zeros((size, size))
    cam = points @ T[:3, :3].T + T[:3, 3]
    u = np.floor(cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    r = np.floor(cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < size) & (r >= 0) & (r < size)
    for uu, rr, zz in zip(u[ok], r[ok], cam[ok, 2]):
        if depth[rr, uu] == 0 or zz < depth[rr, uu]:
            depth[rr, uu] = zz
    return depth


def test_box_model_samples_every_face_by_area():
    m = pc.box_model(HALF, n=1500)
    assert m.shape == (1500, 3)
    assert np.allclose(np.abs(m).max(axis=0), HALF, atol=1e-9)
    # the four 4.5 x 12 cm sides hold most of the points, the 4.5 x 4.5 caps few
    caps = np.isclose(np.abs(m[:, 2]), HALF[2]).sum()
    assert 100 < caps < 250


def test_fit_recovers_a_one_face_view_from_a_nearby_start():
    cloud = front_face(2.50, -0.30, np.deg2rad(5))
    fit = pc.fit_planar(cloud, pc.box_model(HALF), SUPPORT + HALF[2], init_xy=(2.51, -0.29))
    assert fit["accepted"], fit["reasons"]
    assert np.hypot(fit["x"] - 2.50, fit["y"] - (-0.30)) < 0.004
    assert abs(np.rad2deg(fit["yaw"]) - 5) < 3 or abs(np.rad2deg(fit["yaw"]) - 95) < 3


def test_too_few_points_are_refused():
    fit = pc.fit_planar(front_face(2.5, -0.3, 0, n=10), pc.box_model(HALF), SUPPORT + HALF[2])
    assert not fit["accepted"] and fit["reasons"] == ["too few points"]


def test_see_through_flags_a_model_standing_in_seen_empty_space():
    T, K = camera_south(2.50, -0.30)
    seen = front_face(2.50, -0.30, 0.0, n=4000, noise=0.0)
    view = dict(depth=render_depth(seen, T, K), K=K, world2cam=T)
    model = pc.box_model(HALF)
    true = pc._transform(model, 2.50, -0.30, SUPPORT + HALF[2], 0.0)
    slid = pc._transform(model, 2.50, -0.36, SUPPORT + HALF[2], 0.0)   # 6 cm toward the camera
    assert pc.see_through(true, [view], 0.04) < 0.1
    # Slid toward the camera the model hides nothing the camera saw; slid sideways it does.
    side = pc._transform(model, 2.56, -0.30, SUPPORT + HALF[2], 0.0)
    assert pc.see_through(side, [view], 0.04) > 0.3
    assert pc.see_through(slid, [view], 0.04) < pc.see_through(side, [view], 0.04)


def test_hue_labels_name_the_four_props_and_skip_grey():
    palette = [(1.0, 0.1, 0.1), (0.1, 0.8, 0.1), (0.15, 0.3, 1.0), (1.0, 0.85, 0.0)]
    rgb = (np.array([[0.9, 0.1, 0.1], [0.1, 0.7, 0.1], [0.12, 0.25, 0.85], [0.9, 0.75, 0.0],
                     [0.5, 0.5, 0.5], [0.05, 0.05, 0.05]]) * 255).astype(np.uint8)
    assert pc.hue_labels(rgb, palette).tolist() == [0, 1, 2, 3, -1, -1]


def test_voxel_groups_split_two_props_seven_cm_apart():
    # 2 cm cells: a sparse 50-point face (~1 cm spacing) stays whole, 7 cm neighbours stay apart.
    a = front_face(2.50, -0.30, 0.0, n=80)
    b = front_face(2.50, -0.23, 0.0, n=50, seed=1)
    groups = pc.voxel_groups(np.concatenate([a, b]), cell=0.02)
    assert [len(g) for g in groups] == [80, 50]


def test_band_keeps_the_support_height_only():
    pts = np.array([[2.5, -0.3, 1.43], [2.5, -0.3, 1.40], [2.5, -0.3, 1.60], [9.0, -0.3, 1.45]])
    keep = pc.band(pts, (2.0, -0.6), (3.0, 0.0), 1.43, 1.56)
    assert keep.tolist() == [True, False, False, False]
