"""Timed textured cues on a RoboCasa fridge, with poses in recorded sim state."""

from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.quaternions import mat2quat, quat2mat

from mani_skill.utils.structs import Actor, Pose


class FridgePictures:
    def __init__(self, env, images, width=0.3, front_distance=0.9):
        self.env = env
        homes, starts, fronts = [], [], []
        actors = [[] for _ in images]
        for i, data in enumerate(env.scene_builder.scene_data):
            fixtures = data["fixtures"]
            matches = [f for f in fixtures.values() if type(f).__name__ == "Fridge"]
            if len(matches) != 1:
                raise ValueError(f"Expected one fridge in scene {i}, found {len(matches)}")
            fridge = matches[0]
            front = quat2mat(fridge.quat) @ np.array([0., -1., 0.])
            center = np.asarray(fridge.pos) + front * (float(fridge.size[1]) / 2 + 0.006)
            door_center = center.copy()
            # At the door centre below the resting arm, both head cameras see it.
            width_dir = np.cross([0., 0., 1.], front)
            center[2] = float(fridge.pos[2]) + float(fridge.size[2]) * 0.20
            rot = np.column_stack([front, width_dir, [0., 0., 1.]])
            home = np.r_[center, mat2quat(rot)].astype(np.float32)
            yaw = float(np.arctan2(-front[1], -front[0]))
            start = np.r_[door_center[:2] + front[:2] * front_distance, 0.,
                          np.cos(yaw/2), 0., 0., np.sin(yaw/2)].astype(np.float32)
            homes.append(home)
            starts.append(start)
            fronts.append(front)
            for j, path in enumerate(images):
                path = Path(path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                mat = sapien.render.RenderMaterial()
                mat.base_color_texture = sapien.render.RenderTexture2D(
                    filename=str(path), mipmap_levels=1)
                builder = env.scene.create_actor_builder().set_scene_idxs([i])
                builder.add_visual_from_file(
                    filename=str(Path(__file__).parent / "assets/fridge_picture.obj"),
                    scale=[width, width, width], material=mat)
                hidden = home.copy()
                hidden[2] = 1000.
                builder.initial_pose = sapien.Pose(hidden[:3], hidden[3:])
                actors[j].append(builder.build_kinematic(name=f"fridge_picture_{j}_{i}"))
        self.home = torch.as_tensor(np.stack(homes), device=env.device)
        self.starts = np.stack(starts)
        self.fronts = np.stack(fronts)
        self.actors = [Actor.merge(group, name=f"fridge_picture_{j}")
                       for j, group in enumerate(actors)]
        # Store each merged actor once; those poses are sufficient for state replay.
        for group, merged in zip(actors, self.actors):
            for actor in group:
                env.remove_from_state_dict_registry(actor)
            env.add_to_state_dict_registry(merged)

    def update(self, selection, visible, env_idx=None):
        indices = slice(None) if env_idx is None else env_idx
        for j, actor in enumerate(self.actors):
            pose = self.home[indices].clone()
            shown = visible[indices] & (selection[indices] == j)
            pose[:, 2] = torch.where(shown, pose[:, 2], 1000.)
            actor.set_pose(Pose.create(pose))
