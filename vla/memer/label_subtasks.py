"""MemER subtask labels for one task: a LeRobot v3 copy of the dataset with `subtask_index`.

MemER's high-level data (`third_party/memer/scripts/generate_sft_data.py`) needs every frame's
current subtask: `meta/subtasks.parquet` (the labels, indexed like `tasks.parquet`) and a
per-frame `subtask_index`. The authors' operator pressed a key at each subtask's end; here the
boundary is read off what the oracle recorded, and the label text says what the memory must
hold (the low-level policy has none):

* base drives: the base command (channels 11, 12) is nonzero; legs closer than DRIVE_GAP
  frames are one drive;
* grasp: the gripper command (channel 7) drops from open (+1) to closed; release: it is back
  at +1 (the frame a ramped opening ends);
* the answer (which drawer, condiment, prop, vegetable; the search order) comes from
  `task_params.py`, which resets each episode's environment with its seed.

The subtasks and keyframe rules are the ones in `~/Downloads/memer_subtasks_2026-10-20.txt`
(owner-approved 2026-10-20); SeasonDish says "pick up the salt"/"the pepper", since the pepper
is a bottle, not a shaker. If any episode's events do not fit its task's sequence, nothing
is written (LeRobot v3 indexes frames globally, so an episode cannot simply be dropped): the
report lists them and the rules get fixed first. `--check` writes only the report.

usage: label_subtasks.py <task> <dataset dir> <params.json> <out dir> [--check]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE_GAP = 5          # frames (0.5 s): the oracle's legs of one drive are 0.1-0.2 s apart
DRAWERS = {0: "bottom", 1: "third from the top", 2: "second from the top", 3: "top"}
ORDINAL = ["first", "second", "third", "fourth"]
PAD_COLOURS = ["red", "green", "blue", "yellow"]   # prop i's pad (depth_recall_v1.PAD_MAP)
KEYFRAME_RULES = {
    "samedrawer": [{"pattern": "^close the .* drawer$", "select": "first"}],
    "seasondish": [{"exact": "look at the picture on the fridge", "select": "last"}],
    "cabinetsearch": [{"pattern": "^open the .* cabinet from the left$", "select": "last"}],
    "depthrecall": [{"pattern": "^take the .* to the .* pad$", "select": "first"}],
    "veggie": [{"exact": "look at the picture on the fridge", "select": "last"}],
    "takeitback": [{"exact": "pick up the cup", "select": "first"}],
}


class Unfit(Exception):
    """The episode's events do not match its task's subtask sequence."""


def drives(a: np.ndarray) -> list[tuple[int, int]]:
    """[(first, last)] frames of each drive, legs DRIVE_GAP apart merged."""
    moving = np.flatnonzero((np.abs(a[:, 11]) > 1e-6) | (np.abs(a[:, 12]) > 1e-6))
    out = []
    for t in moving:
        if out and t - out[-1][1] <= DRIVE_GAP:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(int(s), int(e)) for s, e in out]


def grasps(a: np.ndarray) -> list[tuple[int, int]]:
    """[(close, open)]: the frame the gripper command leaves +1 downward, the frame it is +1 again."""
    g = a[:, 7]
    out, closed_at = [], None
    for t in range(1, len(g)):
        if closed_at is None and g[t - 1] >= 1 - 1e-6 and g[t] < g[t - 1] - 1e-6:
            closed_at = t
        elif closed_at is not None and g[t] >= 1 - 1e-6:
            out.append((closed_at, t))
            closed_at = None
    if closed_at is not None:
        out.append((closed_at, len(g) - 1))   # held to the end (SeasonDish pours and holds)
    return out


def first_after(items, t):
    for it in items:
        if it[0] > t:
            return it
    raise Unfit(f"nothing after frame {t}")


def segments(task: str, a: np.ndarray, g: np.ndarray, p: dict) -> list[tuple[int, str]]:
    """[(start frame, label)], starts increasing from 0."""
    n = len(a)
    D, G = drives(a), grasps(a)
    if task == "samedrawer":
        k = DRAWERS[p["target_drawer"]]
        d1 = D[0] if D else None
        if d1 is None:
            raise Unfit("no drive")
        g1 = first_after(G, d1[1])
        d2 = first_after(D, g1[1])
        return [(0, f"close the {k} drawer"), (d1[0], "go to the far counter"),
                (d1[1] + 1, "pick up the apple"), (g1[0] + 1, "place the apple on the plate"),
                (g1[1] + 1, "go back to the drawers"), (d2[1] + 1, f"open the {k} drawer")]
    if task == "seasondish":
        c = "salt" if p["target_is_shaker"] else "pepper"
        cue_end = p["cue_steps"] // 2        # control steps -> stored 10 Hz frames
        if not D or G == [] or len(G) != 1:
            raise Unfit(f"drives {len(D)}, grasps {len(G)}")
        g1 = G[0]
        d = [x for x in D if x[1] < g1[0]]
        if not d:
            raise Unfit("no drive before the grasp")
        # The pour: from here on the oracle moves wrist_roll only (season_dish_planner).
        arm = a[:, :6]
        pour = n - 1
        while pour - 1 > g1[0] and np.allclose(arm[pour - 1], arm[n - 1], atol=1e-6):
            pour -= 1
        if pour <= g1[0] + 1:
            raise Unfit("no transfer before the pour")
        return [(0, "look at the picture on the fridge"), (cue_end, "go to the condiment station"),
                (d[-1][1] + 1, f"pick up the {c}"), (g1[0] + 1, f"bring the {c} to the bowl"),
                (pour, f"tip the {c} over the bowl")]
    if task == "cabinetsearch":
        # Rounds are cut at the arrivals on the floor mark. In a round the first grasp is the
        # handle (open: grasp, pull, release); every later one is the fist closing the door,
        # re-pushes included (the oracle's "one re-push"); the last release ends the close.
        order, cube = p["search_order"], p["cube_cab"]
        rounds = order[:order.index(cube) + 1]
        home = np.asarray(p["home_xy"])
        inside = np.linalg.norm(g[:, :2] - home, axis=1) <= p["home_radius"]
        arrivals = [t for t in range(n) if inside[t] and (t == 0 or not inside[t - 1])]
        if len(arrivals) != len(rounds):
            raise Unfit(f"{len(arrivals)} arrivals at the floor mark for {len(rounds)} rounds")
        out, start = [], 0
        for r, cab in enumerate(rounds):
            name = f"{ORDINAL[cab]} cabinet from the left"
            last = r == len(rounds) - 1
            lo, hi = arrivals[r], (n if last else arrivals[r + 1])
            gr = [x for x in G if lo < x[0] < hi]
            if len(gr) < (1 if last else 2):
                raise Unfit(f"round {r}: {len(gr)} grasps")
            out += [(start, "go to the floor mark"), (lo + 1, f"open the {name}")]
            if last:
                out.append((gr[0][1] + 1, "push the cola can"))
            else:
                out.append((gr[0][1] + 1, f"close the {name}"))
                start = gr[-1][1] + 1
        return out
    if task == "depthrecall":
        slots, names = p["original_slots"], p["props"]
        deep = len(slots) - 1
        takes = sorted(range(len(slots)), key=lambda i: slots[i])            # front first
        puts = sorted((i for i in takes if slots[i] != deep), key=lambda i: -slots[i])  # deepest empty first
        if len(G) != len(takes) + len(puts):
            raise Unfit(f"{len(G)} grasps, expected {len(takes) + len(puts)}")
        labels = [f"take the {names[i]} to the {PAD_COLOURS[i]} pad" for i in takes]
        labels += [f"put the {names[i]} back in the {ORDINAL[slots[i]]} spot from the front" for i in puts]
        starts = [0] + [G[j][1] + 1 for j in range(len(G) - 1)]
        return list(zip(starts, labels))
    if task == "takeitback":
        if len(G) != 2:
            raise Unfit(f"{len(G)} grasps, expected 2")
        (c1, o1), (c2, o2) = G
        return [(0, "pick up the cup"), (c1 + 1, "place the cup on the tray"),
                (o1 + 1, "pick up the cup from the tray"), (c2 + 1, "put the cup back where it was")]
    if task == "veggie":
        v = " ".join(p["veggies"][p["picture_target"]].split("_")[1:-1])    # veg_sweet_potato_2 -> sweet potato
        if not D or len(G) != 1:
            raise Unfit(f"drives {len(D)}, grasps {len(G)}")
        return [(0, "look at the picture on the fridge"), (D[0][0], f"pick up the {v}"),
                (G[0][0] + 1, f"put the {v} on the plate")]
    raise ValueError(task)


def per_frame(starts: list[tuple[int, str]], n: int) -> list[str]:
    starts = sorted(starts)
    if starts[0][0] != 0 or any(b[0] <= a[0] for a, b in zip(starts, starts[1:])) or starts[-1][0] >= n:
        raise Unfit(f"segment starts not increasing inside [0, {n}): {[s for s, _ in starts]}")
    out = []
    for (s, lab), (e, _) in zip(starts, starts[1:] + [(n, None)]):
        out += [lab] * (e - s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=sorted(KEYFRAME_RULES))
    ap.add_argument("dataset", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", action="store_true", help="report only, write no dataset")
    args = ap.parse_args()
    src, dst = args.dataset, args.out
    params = {int(k): v for k, v in json.loads(args.params.read_text()).items()}
    files = sorted(glob.glob(f"{src}/data/*/*.parquet"))
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "global_state"])
                       for f in files)
    labels, report = {}, {"labelled": 0, "unfit": {}, "subtasks": {}}
    for e, d in frames.groupby("episode_index"):
        d = d.sort_values("frame_index")
        a = np.stack(d["action"].to_numpy()); g = np.stack(d["global_state"].to_numpy())
        if args.check and int(e) not in params:
            continue                         # a check runs on the episodes it has parameters for
        try:
            if int(e) not in params:
                raise Unfit("no task parameters")
            lab = per_frame(segments(args.task, a, g, params[int(e)]), len(a))
        except Unfit as exc:
            report["unfit"][int(e)] = str(exc)
            continue
        labels[int(e)] = lab
        report["labelled"] += 1
    vocab = sorted({x for lab in labels.values() for x in lab})
    index = {x: i for i, x in enumerate(vocab)}
    report["subtasks"] = {x: sum(lab.count(x) for lab in labels.values()) for x in vocab}
    summary = json.dumps({"labelled": report["labelled"], "unfit": len(report["unfit"]), "subtasks": len(vocab)})
    if args.check or report["unfit"]:
        print(summary, "| unfit examples:", list(report["unfit"].items())[:8])
        print(json.dumps(report["subtasks"], indent=1))
        raise SystemExit(0 if args.check else 1)
    # The copy: videos hardlinked, metadata copied, data parquet rewritten with subtask_index.
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src / "videos", dst / "videos", copy_function=os.link)
    shutil.copytree(src / "meta", dst / "meta")
    for f in files:
        t = pd.read_parquet(f)
        t["subtask_index"] = [index[labels[int(e)][int(i)]] for e, i in zip(t.episode_index, t.frame_index)]
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(out, index=False)
    pd.DataFrame({"subtask_index": range(len(vocab))}, index=pd.Index(vocab, name="subtask")) \
        .to_parquet(dst / "meta/subtasks.parquet")
    info = json.loads((dst / "meta/info.json").read_text())
    info["features"]["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    (dst / "keyframe_rules.json").write_text(json.dumps(
        {"default_select": "none", "rules": KEYFRAME_RULES[args.task]}, indent=2) + "\n")
    (dst / "subtask_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
