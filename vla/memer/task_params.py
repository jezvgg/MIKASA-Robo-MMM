"""Read each episode's task parameters from a fresh environment reset with the episode's seed.

MemER's subtask labels name what the memory has to hold (which drawer, which condiment, which
prop goes where), so the labels need the episode's answer. It is read here, at reset, the way
the collection built each episode (a fresh env, `seed_everything(seed)`, `reset(seed=seed)`);
nothing is stepped. Our four tasks use their collection profile (`make_env`), jezv's two the
env kwargs his collection recorded and his planners' reconfiguring reset.

usage (simulator venv, repo root on PYTHONPATH):
    task_params.py <task> <dataset dir> <out.json> [--episodes 0,1,2] [--shard k/n]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import mani_skill  # noqa: F401
import my_scenes  # noqa: F401
from utils.collection.profile import make_env
from utils.mikasa.seeding import seed_everything

OURS = {"samedrawer", "seasondish", "cabinetsearch", "depthrecall"}
JEZV_ENV = {"takeitback": "MyRoboCasa_TakeItBack-v1", "veggie": "MyRoboCasa_FridgeVeggies-v1"}


def np1(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x).reshape(-1)


def params(task: str, base, seed: int) -> dict:
    if task == "samedrawer":
        z = []
        for art in base._drawer_arts:
            z.append(max(float(np1(link.pose.p)[2]) for link in art.get_links()))
        return {"target_drawer": int(np1(base.target_drawer)[0]), "drawer_top_z": z}
    if task == "seasondish":
        return {"target_is_shaker": bool(np1(base.target_is_shaker)[0]), "cue_steps": int(base.cfg.cue_steps)}
    if task == "cabinetsearch":
        from planners.cabinet_search_planner import search_plan
        _, order = search_plan(seed, len(base.cfg.compartments))   # the oracle's sighted route
        return {"cube_cab": int(np1(base.cube_cab)[0]), "home_radius": float(base.cfg.home_radius),
                "search_order": [int(v) for v in order],
                "compartments": [c.name + ":" + c.open_hinge for c in base.cfg.compartments],
                "home_xy": np1(base.home_marker.pose.p)[:2].tolist()}
    if task == "depthrecall":
        from my_scenes.depth_recall_v1 import PROPS
        return {"original_slots": np1(base.original_slots).tolist(), "pad_slots": np1(base.pad_slots).tolist(),
                "props": [spec.label for spec in PROPS]}
    if task == "veggie":
        return {"veggies": [v.name for v in base.veggies], "picture_target": int(base._picture_target)}
    if task == "takeitback":
        return {"cup_xy": np1(base.cup.pose.p)[:2].tolist()}
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=sorted(OURS | set(JEZV_ENV)))
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    meta = json.loads((args.dataset / "meta/source_rlds_metadata.json").read_text())
    seeds = meta["episode_seeds"]
    index = meta.get("episode_indices") or list(range(len(seeds)))
    pairs = list(zip(index, seeds))
    if args.episodes:
        keep = {int(e) for e in args.episodes.split(",")}
        pairs = [p for p in pairs if p[0] in keep]
    k, n = map(int, args.shard.split("/"))
    pairs = pairs[k::n]
    if args.task in OURS:
        run = {"signature": {"profile": meta["profile"]}, "purpose": "validation"}
    else:
        kwargs = meta["env_info"]["env_kwargs"]
    out = {}
    for e, seed in pairs:
        seed_everything(int(seed))
        env = make_env(run) if args.task in OURS else gym.make(JEZV_ENV[args.task], **kwargs)
        # jezv's planners rebuild the scene for every seed (myrobocasa_*_planner.py:
        # `env.reset(seed=seed, options={"reconfigure": True})`); without it the scene keeps
        # the one ManiSkill built at gym.make, e.g. the same three vegetables in every episode.
        env.reset(seed=int(seed), options={"reconfigure": True} if args.task in JEZV_ENV else None)
        out[int(e)] = {"seed": int(seed), **params(args.task, env.unwrapped, int(seed))}
        env.close()
        print(json.dumps({int(e): out[int(e)]}), flush=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
