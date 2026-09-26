"""Render the actual salt/pepper assets for SeasonDish's fridge picture.

This is an asset preparation utility, not an extra policy camera. Run it once
when changing the fixed condiment models; the resulting PNGs travel with code.
"""
from pathlib import Path

import numpy as np
import sapien
from PIL import Image

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from utils.robocasa_utils import load_objaverse_actor, objaverse_mjcf


class CondimentPhotoEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, model_index, **kwargs):
        self.model_index = model_index
        super().__init__(robot_uids="none", reward_mode="none", **kwargs)

    def _load_scene(self, options):
        self.item = load_objaverse_actor(
            self, "shaker", "condiment", sapien.Pose(), index=self.model_index,
            body_type="kinematic",
        )
        bounds = self.item.get_first_collision_mesh(to_world_frame=False).bounds
        self.photo_target = bounds.mean(axis=0)
        self.photo_radius = float(np.max(bounds[1] - bounds[0]))
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[2, 2, 0.01],
            material=sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, float(bounds[0, 2]) - 0.011])
        builder.build_static(name="photo_background")

    @property
    def _default_sensor_configs(self):
        eye = self.photo_target + self.photo_radius * np.array([1.25, -2.4, 0.65])
        return [CameraConfig("photo", sapien_utils.look_at(eye, self.photo_target),
                             512, 512, 0.55, 0.01, 10)]

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info):
        return {}

    def _get_obs_agent(self):
        return {}

    def get_state_dict(self):
        return self.scene.get_sim_state()


def main():
    output = Path(__file__).resolve().parents[2] / "my_scenes/assets/season_dish"
    output.mkdir(parents=True, exist_ok=True)
    for name, index in [("salt", 1), ("pepper", 0)]:
        env = CondimentPhotoEnv(index, obs_mode="rgb", sim_backend="cpu",
                                render_backend="sapien_cuda", num_envs=1)
        try:
            obs, _ = env.reset(seed=0)
            rgb = obs["sensor_data"]["photo"]["rgb"][0].cpu().numpy()
            Image.fromarray(rgb).save(output / f"{name}.png")
            print(name, objaverse_mjcf("shaker", index), output / f"{name}.png", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
