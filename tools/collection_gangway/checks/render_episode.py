"""A video of one recorded SeasonDish episode: its oracle actions replayed in the run's own profile (the native
20 Hz replay's physics; rendering does not touch it). The scene's render camera with the robot's cameras in a
column beside it, and on every frame the number on the card, the shakes counted so far, the held condiment's
lean from straight down and the verdict.

    cd <the run's repo> && PYTHONPATH=. python render_episode.py RUN_DIR SEED OUT_DIR
"""
import json, os, sys
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import my_scenes  # noqa: F401
from mani_skill.utils.visualization import misc
from utils.collection.profile import profile_env_kwargs

SIDE, BAR = 576, 64    # the render camera's frame and the two-line caption above it, pixels (multiples of 16 for ffmpeg)

run_dir, seed, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
profile = json.loads((run_dir / "run.json").read_text())["signature"]["profile"]
env = gym.make(profile["env_id"], **profile_env_kwargs(
    profile, obs_mode="rgb", render_mode="rgb_array", human_render_camera_configs=dict(width=SIDE, height=SIDE)))
task = env.unwrapped
font = ImageFont.truetype(os.path.join(os.path.dirname(misc.__file__), "UbuntuSansMono-Regular.ttf"), size=20)


def frame(obs, info, step):
    cams = [cam["rgb"][0].cpu().numpy().astype(np.uint8) for cam in obs["sensor_data"].values()]
    cell = SIDE // max(1, len(cams))
    sheet = Image.new("RGB", (SIDE + cell, SIDE + BAR))
    sheet.paste(Image.fromarray(env.render()[0].cpu().numpy()), (0, BAR))
    for i, cam in enumerate(cams):
        sheet.paste(Image.fromarray(cam).resize((cell, cell)), (SIDE, BAR + cell * i))
    target = "salt (white)" if bool(task.target_is_shaker.item()) else "pepper (black)"
    lines = [f"seed {seed}  step {step:4d}/{len(actions)}  {target}  card {int(task.num_shakes.item())}", ""]
    if info is not None:
        lines[1] = f"shakes {int(info['shake_count'][0])}  lean from straight down {float(info['lean_deg'][0]):5.1f} deg"
        lines[1] += "  SUCCESS" if bool(info["success"][0]) else "  FAIL" if bool(info["fail"][0]) else ""
    draw = ImageDraw.Draw(sheet)
    for row, text in enumerate(lines):
        draw.text((10, 6 + 28 * row), text, fill=(255, 255, 255), font=font)
    return np.asarray(sheet)


with h5py.File(run_dir / "oracle" / str(seed) / "trajectory.h5") as f:
    actions = f["traj_0"]["actions"][:]
obs, _ = env.reset(seed=seed)
frames, info = [frame(obs, None, 0)], {}
for step, a in enumerate(actions, 1):
    obs, _, _, _, info = env.step(a)
    frames.append(frame(obs, info, step))
print(f"seed {seed}: {len(actions)} steps, success {bool(info['success'][0])}, shakes "
      f"{int(info['shake_count'][0])}/{int(info['num_shakes'][0])}")
misc.images_to_video(frames, str(out), video_name=f"season_dish_{seed}", fps=20, verbose=False)
env.close()
