"""MyRoboCasa_TakeItBackTray-v1 with a strict success checker.

`success` is True only if the cup was grasped and lifted, was carried over the tray, was released,
and now rests upright on the tray top. The previous looser check is kept as `legacy_success`.
All other `info` fields are diagnostics (see `evaluate`).
"""

import logging
import math

import numpy as np
import torch

from mani_skill.utils.registration import register_env

from .my_robocasa_takeitback import MyRoboCasaSceneTakeItBack

logger = logging.getLogger(__name__)


@register_env("MyRoboCasa_TakeItBackTray-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBackTray(MyRoboCasaSceneTakeItBack):
    """Place a randomly positioned cup onto a randomly positioned baking tray."""

    # Strict-checker thresholds (justification: see the commit message / report).
    LIFT_MIN_M = 0.03  # cup centre must rise this far above its resting height while grasped
    PLATE_MARGIN_M = 0.02  # cup centre must be this far inside the tray footprint edge
    BOTTOM_TOL_M = 0.015  # |cup bottom - tray top| for "stands on the tray"
    TILT_MAX_RAD = math.radians(15.0)  # cup axis vs world up
    REST_V_MAX = 0.02  # m/s
    REST_W_MAX = 0.10  # rad/s
    REST_STEPS = 5  # consecutive env steps below both speed limits

    _cup_bottom_off: float | None = None

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._cup_bottom_off = None  # cup actor is rebuilt on every reconfigure

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        n = self.num_envs
        if not hasattr(self, "_was_lifted") or self._was_lifted.shape[0] != n:
            self._was_lifted = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._held_over_tray = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._rest_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self._was_lifted[env_idx] = False
        self._held_over_tray[env_idx] = False
        self._rest_count[env_idx] = 0

    def _cup_bottom(self) -> float:
        """Distance from the cup origin down to the lowest point of its collision mesh (upright)."""
        if self._cup_bottom_off is None:
            try:
                mesh = self.cup.get_first_collision_mesh(to_world_frame=False)
                self._cup_bottom_off = float(-mesh.bounds[0, 2])
            except (AttributeError, RuntimeError, ValueError, IndexError):
                logger.warning("cup collision mesh unavailable, falling back to half height")
                self._cup_bottom_off = float(self.cup_half[2])
        return self._cup_bottom_off

    def _legacy_success(self) -> torch.Tensor:
        """The pre-strict check: cup xy within tray half extent + 5 cm, z within 10 cm, not grasped, slow."""
        cup_pos, tray_pos = self.cup.pose.p, self.tray.pose.p
        is_grasped = self.agent.is_grasping(self.cup)
        is_static = (torch.linalg.norm(self.cup.linear_velocity, dim=1) <= 0.1) & (
            torch.linalg.norm(self.cup.angular_velocity, dim=1) <= 0.2
        )
        xy_off = torch.abs(cup_pos[:, :2] - tray_pos[:, :2])
        tol = torch.as_tensor(self.tray_half[:2], device=self.device) + 0.05
        on_xy = (xy_off[:, 0] <= tol[0]) & (xy_off[:, 1] <= tol[1])
        tray_top = tray_pos[:, 2] + self.tray_half[2]
        on_z = torch.abs(cup_pos[:, 2] - tray_top - self.cup_half[2]) <= 0.10
        return on_xy & on_z & ~is_grasped & is_static

    def evaluate(self) -> dict:
        """Strict success plus diagnostics; called once per env step (and once at reset)."""
        cup_p, cup_q = self.cup.pose.p, self.cup.pose.q
        tray_p, tray_q = self.tray.pose.p, self.tray.pose.q
        grasped = self.agent.is_grasping(self.cup)
        bottom = self._cup_bottom()

        # lift: cup centre height over its resting height on the counter
        lift = cup_p[:, 2] - (self._counter_top() + bottom)
        self._was_lifted = self._was_lifted | (grasped & (lift >= self.LIFT_MIN_M))

        # cup offset in the tray frame (the tray is dynamic and may be rotated)
        yaw = torch.atan2(
            2 * (tray_q[:, 0] * tray_q[:, 3] + tray_q[:, 1] * tray_q[:, 2]),
            1 - 2 * (tray_q[:, 2] ** 2 + tray_q[:, 3] ** 2),
        )
        dx, dy = cup_p[:, 0] - tray_p[:, 0], cup_p[:, 1] - tray_p[:, 1]
        c, s = torch.cos(yaw), torch.sin(yaw)
        local = torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=1)
        half = torch.as_tensor(self.tray_half[:2], device=self.device, dtype=local.dtype)
        in_footprint = (local.abs() <= half).all(dim=1)
        in_plate = (local.abs() <= half - self.PLATE_MARGIN_M).all(dim=1)
        self._held_over_tray = self._held_over_tray | (grasped & self._was_lifted & in_footprint)

        # stands on the tray top, upright
        bottom_err = (cup_p[:, 2] - bottom) - (tray_p[:, 2] + self.tray_half[2])
        up_z = 1 - 2 * (cup_q[:, 1] ** 2 + cup_q[:, 2] ** 2)
        on_plate = in_plate & (bottom_err.abs() <= self.BOTTOM_TOL_M) & (up_z >= math.cos(self.TILT_MAX_RAD))

        # at rest for REST_STEPS consecutive steps
        slow = (torch.linalg.norm(self.cup.linear_velocity, dim=1) <= self.REST_V_MAX) & (
            torch.linalg.norm(self.cup.angular_velocity, dim=1) <= self.REST_W_MAX
        )
        self._rest_count = torch.where(slow, self._rest_count + 1, torch.zeros_like(self._rest_count))
        at_rest = self._rest_count >= self.REST_STEPS

        released = self._was_lifted & ~grasped
        strict = self._was_lifted & self._held_over_tray & released & on_plate & at_rest
        return dict(
            success=strict,
            strict_success=strict,
            legacy_success=self._legacy_success(),
            was_lifted=self._was_lifted.clone(),
            held_over_tray=self._held_over_tray.clone(),
            released=released,
            on_plate=on_plate,
            at_rest=at_rest,
            cup_lift=lift,
            plate_local_xy=local,
            bottom_minus_top=bottom_err,
            cup_up_z=up_z,
        )

    @property
    def _default_sensor_configs(self):
        # Trajectory collection records robot-mounted cameras only.
        return []
