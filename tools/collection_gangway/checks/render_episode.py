"""A video of one recorded SeasonDish episode: its oracle actions replayed in the run's own profile with the
scene's render camera (the native 20 Hz replay's physics; rendering does not touch it).

    cd <the run's repo> && PYTHONPATH=. python render_episode.py RUN_DIR SEED OUT_DIR
"""
import json, sys
from pathlib import Path

import gymnasium as gym
import h5py

import my_scenes  # noqa: F401
from mani_skill.utils.wrappers import RecordEpisode
from utils.collection.profile import profile_env_kwargs

run_dir, seed, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
profile = json.loads((run_dir / "run.json").read_text())["signature"]["profile"]
env = gym.make(profile["env_id"], **profile_env_kwargs(
    profile, obs_mode="state", render_mode="rgb_array", human_render_camera_configs=dict(width=640, height=480)))
env = RecordEpisode(env, output_dir=str(out), save_video=True, save_trajectory=False, video_fps=20,
                    save_on_reset=False, max_steps_per_video=None)
with h5py.File(run_dir / "oracle" / str(seed) / "trajectory.h5") as f:
    actions = f["traj_0"]["actions"][:]
env.reset(seed=seed)
info = {}
for a in actions:
    *_, info = env.step(a)
print(f"seed {seed}: {len(actions)} steps, success {bool(info['success'][0])}, shakes "
      f"{int(info['shake_count'][0])}/{int(info['num_shakes'][0])}")
env.flush_video(name=f"season_dish_{seed}")
env.close()
