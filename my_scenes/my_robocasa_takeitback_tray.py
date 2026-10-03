import numpy as np
import torch

from mani_skill.utils.registration import register_env

from .my_robocasa_takeitback import MyRoboCasaSceneTakeItBack


@register_env("MyRoboCasa_TakeItBackTray-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBackTray(MyRoboCasaSceneTakeItBack):
    """Place a randomly positioned cup onto a randomly positioned baking tray."""

    INITIAL_TORSO = 0.25
    INITIAL_TORSO_JITTER = 0.03
    INITIAL_ARM = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0])
    INITIAL_ARM_JITTER = 0.03
    TRAY_EDGE_MARGIN = 0.02
    TRAY_Z_TOL = 0.01
    TRAY_CONTACT_FORCE = 0.05

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        qpos = self.agent.robot.get_qpos()[env_idx].clone()
        for row in range(len(env_idx)):
            qpos[row, 3] = self.INITIAL_TORSO + self._main_rng.uniform(
                -self.INITIAL_TORSO_JITTER, self.INITIAL_TORSO_JITTER
            )
            qpos[row, 6:13] = torch.as_tensor(
                self.INITIAL_ARM
                + self._main_rng.uniform(
                    -self.INITIAL_ARM_JITTER, self.INITIAL_ARM_JITTER, size=7
                ),
                dtype=qpos.dtype,
                device=qpos.device,
            )
            qpos[row, 13:15] = 0.015
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_qvel(torch.zeros_like(qpos))

    def evaluate(self):
        cup_pos = self.cup.pose.p
        tray_pos = self.tray.pose.p
        is_grasped = self.agent.is_grasping(self.cup)
        is_static = (
            torch.linalg.norm(self.cup.linear_velocity, dim=1) <= 0.1
        ) & (torch.linalg.norm(self.cup.angular_velocity, dim=1) <= 0.2)
        xy_dist = torch.linalg.norm(cup_pos[:, :2] - tray_pos[:, :2], dim=1)
        accept_radius = (
            min(self.tray_half[:2]) - max(self.cup_half[:2]) - self.TRAY_EDGE_MARGIN
        )
        on_tray_xy = xy_dist <= accept_radius
        tray_top = tray_pos[:, 2] + self.tray_half[2]
        cup_bottom = cup_pos[:, 2] - self.cup_half[2]
        on_tray_z = torch.abs(cup_bottom - tray_top) <= self.TRAY_Z_TOL
        contact_force = self.scene.get_pairwise_contact_forces(self.cup, self.tray)
        has_tray_contact = (
            torch.linalg.norm(contact_force, dim=1) >= self.TRAY_CONTACT_FORCE
        )
        return dict(
            success=on_tray_xy & on_tray_z & has_tray_contact & ~is_grasped & is_static
        )

    @property
    def _default_sensor_configs(self):
        # Trajectory collection records robot-mounted cameras only.
        return []
