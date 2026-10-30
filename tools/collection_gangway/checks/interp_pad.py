"""Interpolated replay (MODE all / nogrip) with the last target held PAD more control steps: does the pour then
reach the task's hold? Prints pour_hold per step near the end and success."""
import json, sys
from pathlib import Path
import h5py, numpy as np
root, seed, sys_path, pad = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.path.insert(0, sys_path)
from interp_replay import interpolated
from utils.collection.client import as_numpy, scalar_bool
from utils.collection.pipeline import episode_path
from utils.collection.profile import make_env
from utils.mikasa.seeding import seed_everything
run = json.loads((root / "run.json").read_text())
a10 = h5py.File(episode_path(root, "oracle", seed))["traj_0"]["actions"][:][::2]
for name, acts in (("held", np.repeat(a10, 2, axis=0)), ("all", interpolated(a10, "all")), ("nogrip", interpolated(a10, "nogrip"))):
    acts = np.concatenate([acts, np.repeat(acts[-1:], pad, axis=0)])
    seed_everything(seed); env = make_env(run); env.reset(seed=seed)
    hold, first = [], None
    for i, a in enumerate(acts):
        _, _, _, trunc, info = env.step(a)
        hold.append(int(np.asarray(as_numpy(info["pour_hold"])).reshape(-1)[0]))
        if first is None and scalar_bool(info["success"]): first = i
    n = len(acts) - pad
    print(f"{seed} {name}: success at the recorded end {hold[n-1] >= 15} (pour_hold {hold[n-1]}), with +{pad} steps "
          f"{scalar_bool(info['success'])} (pour_hold {hold[-1]}); first success step {first} of recorded {n}; "
          f"pour starts at step {next((i for i, h in enumerate(hold) if h == 1), None)}")
    env.close()
