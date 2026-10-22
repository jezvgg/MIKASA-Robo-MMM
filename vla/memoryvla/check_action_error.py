"""Open-loop check of a MemoryVLA checkpoint on held-out episodes of one task (LeRobot v3 copy).

Runs the authors' inference path (`vla.load_vla`, `predict_action` with the three cameras, their
deploy.py image preprocessing, DDIM 10 steps, CFG 1.5) once every `--every` frames, as their
MIKASA eval calls the model once per executed chunk, with the memory reset at each episode's
first frame. Compares the predicted chunk with the recorded actions:

  base turn (12) and forward speed (11): on steps where the recording moves vs. where it does
      not; the turn is the channel a q01/q99 bound used to erase in DepthRecall;
  arm (0-6): mean |error| against "hold still", on chunks where the arm moves and where it does
      not. With joint deltas (patch 0001, the default) both are compared as deltas from each
      step's joints and holding still is 0; with --absolute-joints as targets, holding still
      being the current joints (MemoryVLA sees no joint state, so it must read the pose from
      the images);
  gripper (7): sign agreement.

usage: check_action_error.py <run dir> <lerobot task dir> <unnorm key> [--episodes 990,991] [--every 4]
       [--out result.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party/MemoryVLA"))
from vla import load_vla  # noqa: E402

CAMERAS = ("observation.images.left_base_camera_link",    # primary
           "observation.images.fetch_hand",                # left_wrist slot
           "observation.images.right_base_camera_link")    # right_wrist slot
STATE_TO_ACTION_ORDER = (2, 4, 5, 6, 7, 8, 9, 10, 1, 3, 0, 11)   # mikasa_pi05.transforms
JOINTS = list(range(7)) + [8, 9, 10]                               # arm, head, torso


def resize_image(image: Image.Image, size=(224, 224)) -> Image.Image:
    """deploy.py's resize_image + scale_and_resize: centre square, 224, then the 0.9-area centre
    crop that stands in for training's random crop."""
    w, h = image.size
    left = min(max((w - h) // 2, 0), max(w - h, 0))
    image = image.crop((left, 0, left + min(w, h), min(w, h))).resize(size, resample=Image.LANCZOS)
    nw, nh = int(size[0] * math.sqrt(0.9)), int(size[1] * math.sqrt(0.9))
    mw, mh = (size[0] - nw) // 2, (size[1] - nh) // 2
    return image.crop((mw, mh, mw + nw, mh + nh)).resize(size, resample=Image.LANCZOS)


def episode_frames(root: Path, info: dict, row, key: str, n: int) -> list[Image.Image]:
    path = root / info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                            file_index=int(row[f"videos/{key}/file_index"]))
    start, fps, out = float(row[f"videos/{key}/from_timestamp"]), float(info["fps"]), []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(start - 1.0, 0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.time is None or f.time < start - 0.5 / fps:
                continue
            out.append(f.to_image())
            if len(out) == n:
                break
    assert len(out) == n, (key, len(out), n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("lerobot", type=Path)
    ap.add_argument("unnorm_key")
    ap.add_argument("--episodes", default="990,991,992,993,994")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--absolute-joints", dest="joint_deltas", action="store_false",
                    help="the run predicts absolute joint targets (before patch 0001's deltas)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ckpt = sorted((args.run_dir / "checkpoints").glob("*.pt"))[-1]
    cfg = json.loads((args.run_dir / "config.json").read_text())
    arch = ("action_dim", "future_action_window_size", "action_model_type", "use_ema", "mem_length",
            "load_wrist", "load_bi_wrist", "consolidate_type", "fusion_type", "per_token_size",
            "retrieval_layers", "use_timestep_pe", "update_fused", "dataloader_type", "group_size")
    vla = load_vla(str(ckpt), load_for_training=False, **{k: cfg[k] for k in arch if k in cfg})
    vla = vla.to("cuda").to(torch.float32).eval()

    root, info = args.lerobot, json.loads((args.lerobot / "meta/info.json").read_text())
    text = list(pq.read_table(root / "meta/tasks.parquet").to_pandas().index)[0]
    eps = pq.read_table(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))).to_pandas()
    data = pq.read_table(sorted(glob.glob(f"{root}/data/*/*.parquet")),
                         columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()
    rec = {k: [] for k in ("pred", "truth", "hold")}
    for e in [int(x) for x in args.episodes.split(",")]:
        d = data[data.episode_index == e].sort_values("frame_index")
        a = np.stack(d["action"].to_numpy()).astype(np.float64)
        s = np.stack(d["observation.state"].to_numpy()).astype(np.float64)[:, list(STATE_TO_ACTION_ORDER)]
        row = eps[eps.episode_index == e].iloc[0]
        cams = [episode_frames(root, info, row, k, len(a)) for k in CAMERAS]
        for t in range(0, len(a), args.every):
            pred, _ = vla.predict_action(
                image=resize_image(cams[0][t]), image_wrist=resize_image(cams[1][t]),
                image_wrist_2=resize_image(cams[2][t]), instruction=text, unnorm_key=args.unnorm_key,
                cfg_scale=1.5, use_ddim=True, num_ddim_steps=10,
                episode_first_frame="True" if t == 0 else "False")
            pred = np.asarray(pred, dtype=np.float64)
            idx = np.minimum(np.arange(t, t + len(pred)), len(a) - 1)
            truth = a[idx].copy()
            if args.joint_deltas:
                # The model predicts arm/head/torso deltas from each step's joints (patch 0001's
                # fetch_dataset_transform); compare in that space, where "hold still" is 0.
                truth[:, JOINTS] -= s[idx][:, JOINTS]
                hold = np.zeros((len(pred), 11))
            else:
                hold = np.repeat(s[t:t + 1, :11], len(pred), axis=0)
            rec["pred"].append(pred); rec["truth"].append(truth); rec["hold"].append(hold)
        print(f"episode {e}: {len(range(0, len(a), args.every))} calls", flush=True)

    p, g, h = (np.stack(rec[k]) for k in ("pred", "truth", "hold"))     # [calls, chunk, 13|11]
    res = {"calls": int(len(p))}
    for name, dim in (("turn", 12), ("forward", 11)):
        moving = np.abs(g[..., dim]) > 1e-6
        err = np.abs(p[..., dim] - g[..., dim])
        res[name] = {
            "moving_steps": int(moving.sum()),
            "moving_mean_abs_truth": float(np.abs(g[..., dim])[moving].mean()) if moving.any() else None,
            "moving_mean_abs_error": float(err[moving].mean()) if moving.any() else None,
            "moving_sign_agreement": float(np.mean((np.sign(p[..., dim]) == np.sign(g[..., dim]))[moving]
                                                   & (np.abs(p[..., dim]) > 0.01)[moving])) if moving.any() else None,
            "still_steps": int((~moving).sum()),
            "still_fraction_predicting_motion": float(np.mean(np.abs(p[..., dim])[~moving] > 0.02)),
            "still_mean_abs_pred": float(np.abs(p[..., dim])[~moving].mean()),
            "max_abs_pred": float(np.abs(p[..., dim]).max()), "max_abs_truth": float(np.abs(g[..., dim]).max()),
        }
    arm_move = np.abs(g[..., :7] - h[..., :7]).max(axis=(1, 2)) > 1e-3
    model = np.abs(p[..., :7] - g[..., :7]).mean(axis=(1, 2)); hold = np.abs(h[..., :7] - g[..., :7]).mean(axis=(1, 2))
    res["arm"] = {split: {"chunks": int(m.sum()), "model_mean_abs_rad": float(model[m].mean()),
                          "hold_mean_abs_rad": float(hold[m].mean()),
                          "ratio": float(model[m].mean() / max(hold[m].mean(), 1e-9))}
                  for split, m in (("moving", arm_move), ("still", ~arm_move)) if m.any()}
    res["gripper_agreement"] = float(np.mean(np.sign(p[..., 7]) == np.sign(g[..., 7])))
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))
        np.savez_compressed(args.out.with_suffix(".npz"), pred=p, truth=g, hold=h)


if __name__ == "__main__":
    main()
