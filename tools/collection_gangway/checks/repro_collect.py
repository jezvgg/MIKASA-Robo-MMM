"""One-branch reproduction (owner 2026-10-26 «Да», step 4): re-collect 20 episodes of each published dataset on
repro/one-branch with the dataset's own collection profile and seeds, and compare them with the published
episodes bit for bit.

For each dataset: the episodes 5, 55, ..., 955 (their scene seeds from meta/episodes), a development campaign
through the held 10 Hz replay (`validated`), then per episode: the 10 Hz actions (oracle a20[::2]) against the
dataset's `action`, the replayed 12D state (validated qpos[2k, 3:]) against `observation.state`, the lengths, and
the instruction against the dataset's task. Writes ~/sd-collect/repro-check/<name>.json.

    python repro_collect.py [NAME ...]     (default: all four)
"""
import glob, json, os, subprocess, sys, time
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

H = Path.home()
REPO = H / "sd-collect/repo-vla-repro"
OUT = H / "sd-collect/repro-check"
TOK = H / "mikasa-pi05/openpi-cache/big_vision/paligemma_tokenizer.model"
DATASETS = {  # name -> (profile, dataset dir)
    "samedrawer-ag-1000ep": ("same_drawer", H / "datasets/samedrawer-ag-1000ep"),
    "seasondish-ag-1000ep": ("season_dish", H / "datasets/seasondish-ag-1000ep"),
    "depthrecall-ag-1000ep": ("depth_recall", H / "datasets/depthrecall-ag-1000ep"),
    "cabinetsearch5-ag-1000ep": ("cabinet_search5", H / "sd-collect/runs-ag/final/cabinetsearch5-ag-1000ep"),
}
EPISODES = list(range(5, 1000, 50))


def published(root):
    meta = pq.read_table(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")),
                         columns=["episode_index", "episode_seed"]).to_pandas().set_index("episode_index")
    data = pq.read_table(sorted(glob.glob(f"{root}/data/*/*.parquet")),
                         columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()
    tasks = pq.read_table(f"{root}/meta/tasks.parquet").to_pandas()
    task = list(tasks["task"]) if "task" in tasks.columns else list(tasks.index)
    return meta, data, task


def main(names):
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=".")
    for name in names:
        prof, root = DATASETS[name]
        meta, data, task = published(root)
        seeds = [int(meta.loc[e, "episode_seed"]) for e in EPISODES]
        manifest = OUT / f"{name}-seeds.json"
        manifest.write_text(json.dumps(dict(seeds=seeds, episodes=EPISODES, dataset=name,
                                            rule="episodes 5, 55, ..., 955 of the published dataset"), indent=1))
        run = OUT / name
        t0 = time.time()
        with open(OUT / f"{name}.log", "a") as log:
            code = subprocess.run([os.environ["SIM"], "-m", "utils.collection.campaign", "--profile", prof,
                                   "--tokenizer", str(TOK), "--output", str(run), "--seed-manifest", str(manifest),
                                   "--purpose", "development", "--jobs", os.environ.get("JOBS", "4"),
                                   "--through", "validated", "--timeout", "1800"],
                                  cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        lang = list(json.loads((run / "run.json").read_text())["language_validation"]["instruction_tokens"])
        rows = []
        for ep, seed in zip(EPISODES, seeds):
            rec = data[data.episode_index == ep].sort_values("frame_index")
            A, S = np.stack(rec["action"].to_numpy()), np.stack(rec["observation.state"].to_numpy())
            row = dict(episode=ep, seed=seed)
            try:
                with h5py.File(run / "oracle" / str(seed) / "trajectory.h5") as f:
                    a10 = f["traj_0"]["actions"][:][::2]
                with h5py.File(run / "validated" / str(seed) / "trajectory.h5") as f:
                    q = f["traj_0"]["qpos"][:][::2, 3:]
            except (OSError, KeyError) as exc:
                rows.append(dict(row, status=f"missing: {exc}")); continue
            n = min(len(a10), len(A))
            row.update(frames=(len(a10), len(A)), same_length=len(a10) == len(A),
                       actions_equal=bool(len(a10) == len(A) and np.array_equal(a10.astype(np.float32), A)),
                       max_action_diff=float(np.abs(a10[:n] - A[:n]).max()),
                       states_equal=bool(len(q) >= len(S) and np.array_equal(q[:len(S)].astype(np.float32), S)),
                       max_state_diff=float(np.abs(q[:min(len(q), len(S))] - S[:min(len(q), len(S))]).max()))
            row["status"] = "identical" if row["actions_equal"] and row["states_equal"] else "differs"
            rows.append(row)
        same = sum(r["status"] == "identical" for r in rows)
        out = dict(dataset=name, profile=prof, campaign_exit=code, minutes=round((time.time() - t0) / 60, 1),
                   instruction_matches=lang == task, instruction=lang, dataset_task=task,
                   identical=same, episodes=rows)
        (OUT / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"{name}: campaign exit {code}, {same}/{len(rows)} identical, instruction matches {lang == task}, "
              f"{out['minutes']} min", flush=True)
        for r in rows:
            if r["status"] != "identical":
                print("   ", {k: r.get(k) for k in ("episode", "seed", "status", "frames", "max_action_diff", "max_state_diff")}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
