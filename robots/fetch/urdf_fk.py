"""Float64 forward kinematics of one URDF chain, in numpy (no simulator, no cuRobo).

`curobo_client.scan` uses it to re-check cuRobo's float32 answers against the URDF the
simulator loads, BIMBO_EvoX's ``_verify64`` (skills/gpu-ik/gpu_ik.py): a hit counts only
if the solution's own FK lands on the goal within the tolerance and every joint is inside
its URDF limits. cuRobo solves on a model RobotBuilder made from the URDF (spheres, a
YAML of its own, locked joints), so this is the one check that the two agree.

    >>> chain = Chain("robots/fetch/fetch.urdf", "base_link", "gripper_link")
    >>> T = chain.fk(dict(zip(chain.joint_names, q)))            # doctest: +SKIP
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class Chain:
    """The joints from ``base`` to ``tip``, in order, with their origins, axes and limits."""

    def __init__(self, urdf, base: str, tip: str):
        root = ET.parse(Path(urdf)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        joints, link = [], tip
        while link != base:
            if link not in by_child:
                raise KeyError(f"{tip!r} is not below {base!r} in {urdf}")
            j = by_child[link]
            joints.append(j)
            link = j.find("parent").get("link")
        self.joints = []
        for j in reversed(joints):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = _rpy(*(float(v) for v in o.get("rpy", "0 0 0").split()))
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            a = j.find("axis")
            axis = np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()])
            lim = j.find("limit")
            kind = j.get("type")
            lo, hi = -np.inf, np.inf
            if lim is not None and kind in ("revolute", "prismatic"):
                lo, hi = float(lim.get("lower", "-inf")), float(lim.get("upper", "inf"))
            self.joints.append((j.get("name"), kind, T, axis, lo, hi))
        self.joint_names = [n for n, kind, *_ in self.joints if kind != "fixed"]

    def fk(self, q: dict) -> np.ndarray:
        """``tip`` in the ``base`` frame for joint values by name (missing ones read 0)."""
        T = np.eye(4)
        for name, kind, origin, axis, _lo, _hi in self.joints:
            T = T @ origin
            if kind in ("revolute", "continuous"):
                M = np.eye(4)
                M[:3, :3] = _axis_angle(axis, float(q.get(name, 0.0)))
                T = T @ M
            elif kind == "prismatic":
                M = np.eye(4)
                M[:3, 3] = axis / np.linalg.norm(axis) * float(q.get(name, 0.0))
                T = T @ M
        return T

    def fk_many(self, Q: np.ndarray, names) -> np.ndarray:
        """`fk` for many configurations at once: ``Q`` (N, len(names)) -> (N, 4, 4)."""
        Q = np.asarray(Q, dtype=np.float64).reshape(-1, len(names))
        col = {n: i for i, n in enumerate(names)}
        T = np.broadcast_to(np.eye(4), (len(Q), 4, 4)).copy()
        for name, kind, origin, axis, _lo, _hi in self.joints:
            T = T @ origin
            if kind == "fixed" or name not in col:
                continue
            v = Q[:, col[name]]
            M = np.broadcast_to(np.eye(4), (len(Q), 4, 4)).copy()
            a = axis / np.linalg.norm(axis)
            if kind == "prismatic":
                M[:, :3, 3] = v[:, None] * a
            else:
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                M[:, :3, :3] = (np.eye(3) + np.sin(v)[:, None, None] * K
                                + (1 - np.cos(v))[:, None, None] * (K @ K))
            T = T @ M
        return T

    def within_limits(self, q: dict, eps: float = 1e-6) -> bool:
        return all(lo - eps <= float(q[name]) <= hi + eps
                   for name, kind, _T, _a, lo, hi in self.joints if name in q and kind != "fixed")

    def limits(self, names) -> tuple:
        """(lower, upper) arrays for ``names``; a joint outside the chain reads (-inf, inf)."""
        by = {n: (lo, hi) for n, _k, _T, _a, lo, hi in self.joints}
        lo = np.array([by.get(n, (-np.inf, np.inf))[0] for n in names])
        hi = np.array([by.get(n, (-np.inf, np.inf))[1] for n in names])
        return lo, hi


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple:
    """(position error m, rotation error rad) between two 4x4 poses; batched over leading axes."""
    dR = T_b[..., :3, :3] @ np.swapaxes(T_a[..., :3, :3], -1, -2)
    angle = np.arccos(np.clip((np.trace(dR, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0))
    pos = np.linalg.norm(T_a[..., :3, 3] - T_b[..., :3, 3], axis=-1)
    if np.ndim(pos) == 0:
        return float(pos), float(angle)
    return pos, angle
