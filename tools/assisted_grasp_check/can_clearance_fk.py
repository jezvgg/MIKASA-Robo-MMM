"""Hand clearance to the can during the touch preparation, by forward kinematics on saved
trajectories (qpos per 20 Hz step) - CPU only. The can: a vertical cylinder (r 0.033, h 0.123)
at the logged nudge target. Points: the collision-mesh vertices of wrist_roll, gripper and both
fingers, and forearm_roll and wrist_flex for the clearance only. Per phase (the planner events in\nthe run log): the minimum clearance, negative inside the can.
The trajectory is collection_episode.py's --trajectory-out, the log its stdout.

    python tools/assisted_grasp_check/can_clearance_fk.py run/p_7880257.npz run/7880257.log [...]
"""
import re, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from robots.fetch.urdf_fk import Chain
U = str(REPO / "robots" / "fetch" / "fetch.urdf")
R, H = 0.033, 0.123
root = ET.parse(U).getroot()
LINKS = ["wrist_roll_link", "gripper_link", "r_gripper_finger_link", "l_gripper_finger_link",
         "forearm_roll_link", "wrist_flex_link"]
pts = {}
for l in root.findall("link"):
    if l.get("name") in LINKS:
        c = l.find("collision"); o = c.find("origin"); g = c.find("geometry")[0]
        v = np.asarray(trimesh.load(U.rsplit("/", 1)[0] + "/" + g.get("filename"), force="mesh").vertices)
        if o is not None:
            v = v + np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
        pts[l.get("name")] = v[np.random.default_rng(0).choice(len(v), min(len(v), 400), replace=False)]
chains = {n: Chain(U, "root", n) for n in LINKS}
for npz, log in zip(sys.argv[1::2], sys.argv[2::2]):
    z = np.load(npz, allow_pickle=True); Q, names = z["qpos"], list(z["joints"])
    text = open(log).read()
    tgt = re.findall(r"nudge the cube target=\[([-\d., ]+)\]", text)
    if not tgt:
        print(npz, "no nudge target"); continue
    c = np.array([float(x) for x in tgt[0].split(",")])
    ev = [(int(s), p) for s, p in re.findall(r"\[event\] step=(\d+) phase (.+)", text)]
    marks = [(s, p) for s, p in ev if any(k in p for k in ("ready posture for the touch", "raise the torso for the touch",
             "level the hand for the touch", "ready arm and raise torso together", "pre-touch", "push start", "nudge the cube",
             "drive to the touch dock", "roll in to the touch dock"))]
    clear = np.full(len(Q), np.inf); who = [""] * len(Q); ymax = np.full(len(Q), -np.inf)
    for n in LINKS:
        T = chains[n].fk_many(Q, names)
        P = np.einsum("nij,kj->nki", T[:, :3, :3], pts[n]) + T[:, None, :3, 3]
        dz = np.abs(P[..., 2] - c[2]) - H / 2
        dxy = np.linalg.norm(P[..., :2] - c[:2], axis=-1) - R
        d = np.where(dz > 0, np.hypot(np.maximum(dz, 0), np.maximum(dxy, 0)), dxy).min(axis=1)
        better = d < clear
        clear[better] = d[better]
        for i in np.where(better)[0]:
            who[i] = n
        if n != "forearm_roll_link":
            ymax = np.maximum(ymax, P[..., 1].max(axis=1))
    print(f"== {npz.rsplit('/',1)[-1]} can={c.round(3).tolist()} steps={len(Q)}")
    bounds = [s for s, _ in marks] + [len(Q)]
    for (s, p), e in zip(marks, bounds[1:]):
        if e <= s:
            continue
        i = s + int(np.argmin(clear[s:e]))
        print(f"  {s:5d}-{e:5d} {p[:48]:48s} min_clear={clear[i]*100:6.1f} cm at {i} ({who[i]}) hand_ymax={ymax[s:e].max():.3f}")
