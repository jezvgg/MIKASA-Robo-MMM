import torch

from mani_skill.utils.registration import register_env

from .my_robocasa_takeitback import MyRoboCasaSceneTakeItBack


@register_env("MyRoboCasa_TakeItBackTray-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBackTray(MyRoboCasaSceneTakeItBack):
    """Place a randomly positioned cup onto a randomly positioned baking tray."""

    TRAY_EDGE_MARGIN = 0.02
    TRAY_Z_TOL = 0.01
    TRAY_CONTACT_FORCE = 0.05

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
