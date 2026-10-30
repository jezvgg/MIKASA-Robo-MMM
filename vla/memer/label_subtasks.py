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
(owner-approved 2026-10-20). Since 2026-10-22 (owner) SeasonDish and DepthRecall call salt and
pepper "the white salt shaker" and "the black pepper shaker": in both tasks they are one
RoboCasa shaker in two colours (shaker_1, shaker_0), and the colour is all that tells them
apart on camera. The source must carry each environment's one current instruction
(`CURRENT_TEXT`), as the HF datasets do since 2026-10-30; an older download is refused, to be
fetched again from HF, and the copy keeps the source's text. If any episode's events do
not fit its task's sequence, nothing is written (LeRobot v3 indexes frames globally, so an
episode cannot simply be dropped): the report lists them and the rules get fixed first.
`--check` writes only the report. CabinetSearch5 (since 2026-10-26) also reads the oracle's
event log (`--events`): its first compartment is opened and shut on one grip, so no gripper
edge marks where the close begins.

SeasonDish with shaking (profile 28, collected 2026-10-28; scheme A approved by the owner
2026-10-29) is labelled from its event log (`--events`), which also gives the episode's answer
(`target chosen`: the shaker and the number N on the card) and the cue's length: one subtask per
shake, "shake the <shaker> over the bowl (k of N)", each ending when the oracle counts shake k,
with its last frame a keyframe, the way the authors label repeated subtasks (App. E). The turn
upside down starts at the `pour` event and ends where the wrist roll first turns back; the turn
upright starts at the N-th `shake`. Without `--events`, SeasonDish is the profile-26 task (one tip).
`params` may be `-`: the episodes' seeds are then read from the dataset's
meta/source_rlds_metadata.json (enough for a task whose answer is in its event log).

usage: label_subtasks.py <task> <dataset dir> <params.json | -> <out dir> [--check] [--events PATTERN]
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
ORDINAL = ["first", "second", "third", "fourth", "fifth"]
PAD_COLOURS = ["red", "green", "blue", "yellow"]   # prop i's pad (depth_recall_v1.PAD_MAP)
SPOKEN = {"salt": "white salt shaker", "pepper": "black pepper shaker"}   # depth_recall_v1 PropSpec.spoken
SHAKE_TEXT = ("Remember whether the picture on the fridge shows the white salt shaker or the black pepper "
              "shaker, and the number next to it. After the picture disappears, take that shaker, turn it "
              "upside down over the bowl, shake it as many times as the number showed and turn it upright "
              "again, leaving the other shaker untouched.")
# The environments' one current instruction per task (owner 2026-10-30): INSTRUCTIONS[0] of
# same_drawer.py, season_dish.py and depth_recall_v1.py, INSTRUCTIONS_NUDGE[0] of cabinet_search.py.
# The HF datasets carry it and every policy trains on it; a source with any other text is an old
# download (`check_task_text`).
CURRENT_TEXT = {
    "samedrawer": "Remember which drawer is open and close it. Keep it closed while putting the apple on the plate "
                  "at the far counter, then return and open the same drawer. Keep the apple on the plate until "
                  "the end and leave every other drawer closed.",
    "seasondish": SHAKE_TEXT,
    "depthrecall": "Take the props out of the cabinet onto the counter pads (ketchup on the red pad, black pepper "
                   "shaker on the green pad, white salt shaker on the blue pad, mustard on the yellow pad), keep "
                   "the one from the back, and put the others back where they stood.",
    "cabinetsearch": "Find the cola can in the wall cabinets and gently push it. Start at the rug. Close empty "
                     "cabinets and return directly to the rug before searching another. Never inspect a cabinet "
                     "twice. Finish after pushing the can.",
}
KEYFRAME_RULES = {
    "samedrawer": [{"pattern": "^close the .* drawer$", "select": "first"}],
    "seasondish": [{"exact": "look at the picture on the fridge", "select": "last"},
                   {"pattern": r"^shake the .* over the bowl \(\d+ of \d+\)$", "select": "last"}],
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


def oracle_events(pattern: str, seed: int) -> list[tuple[int, dict]]:
    """[(stored frame, event)] of the oracle's phase log for one seed (20 Hz steps -> 10 Hz frames)."""
    hits = glob.glob(os.path.expanduser(pattern.format(seed=seed)))
    if len(hits) != 1:
        raise Unfit(f"{len(hits)} event logs for seed {seed}")
    out = []
    for line in Path(hits[0]).read_text().splitlines():
        ev = json.loads(line)
        if ev.get("event") == "phase":
            out.append((int(ev["step"]) // 2, ev))
    return out


def seasondish_shake(a: np.ndarray, D, G, events: list[tuple[int, dict]]) -> list[tuple[int, str]]:
    """Profile 28: cue, drive, grasp, carry, turn upside down, N shakes, turn upright and hold."""
    by: dict[str, list] = {}
    for f, ev in events:
        by.setdefault(ev.get("message"), []).append((f, ev))
    one = {}
    for m in ("target chosen", "fridge cue observation complete", "pour", "held"):
        if len(by.get(m, [])) != 1:
            raise Unfit(f"{len(by.get(m, []))} '{m}' events")
        one[m] = by[m][0]
    n_shakes = int(one["target chosen"][1]["shakes"])
    shakes = [f for f, _ in by.get("shake", [])]
    if [ev["index"] for _, ev in by.get("shake", [])] != list(range(1, n_shakes + 1)) \
            or one["held"][1].get("shake_count") != n_shakes or not one["held"][1].get("success"):
        raise Unfit(f"N = {n_shakes}, shake events {[ev.get('index') for _, ev in by.get('shake', [])]}, "
                    f"held {one['held'][1].get('shake_count')}")
    c = SPOKEN["salt" if one["target chosen"][1]["target_is_shaker"] else "pepper"]
    cue_end = int(one["fridge cue observation complete"][1]["control_steps"]) // 2
    if len(G) != 1:
        raise Unfit(f"{len(G)} grasps")
    g1 = G[0]
    d = [x for x in D if x[1] < g1[0]]
    if not d:
        raise Unfit("no drive before the grasp")
    pour, ev = one["pour"]
    if pour <= g1[0] + 1:
        raise Unfit("no transfer before the pour")
    # Upside down: the first frame after the pour where the wrist roll (channel 6) stops moving
    # towards the pour's goal, past half the turn.
    roll, sign = a[:, 6], np.sign(ev["wrist_goal"] - ev["wrist_start"])
    half = abs(ev["wrist_goal"] - ev["wrist_start"]) / 2
    down = next((f for f in range(pour, shakes[0]) if abs(roll[f] - roll[pour]) > half
                 and sign * (roll[f + 1] - roll[f]) <= 1e-4), None)
    if down is None:
        raise Unfit("the wrist never turns back before the first shake")
    out = [(0, "look at the picture on the fridge"), (cue_end, "go to the condiment station"),
           (d[-1][1] + 1, f"pick up the {c}"), (g1[0] + 1, f"bring the {c} to the bowl"),
           (pour, f"turn the {c} upside down over the bowl")]
    out += [(start, f"shake the {c} over the bowl ({k} of {n_shakes})")
            for k, start in enumerate([down] + shakes[:-1], 1)]
    return out + [(shakes[-1], f"turn the {c} upright and hold it")]


def segments(task: str, a: np.ndarray, g: np.ndarray, p: dict,
             events: list[tuple[int, dict]] | None = None) -> list[tuple[int, str]]:
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
    if task == "seasondish" and events is not None:
        return seasondish_shake(a, D, G, events)
    if task == "seasondish":
        c = SPOKEN["salt" if p["target_is_shaker"] else "pepper"]
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
        # CabinetSearch5's first compartment (cab_1, one door) is shut on the grip that opened
        # it: there the close starts at the oracle's "swing shut on the bar" (`--events`).
        order, cube = p["search_order"], p["cube_cab"]
        rounds = order[:order.index(cube) + 1]
        shut = []
        if events is not None:
            visited = [ev["compartment"] for _, ev in events if ev.get("message") == "round"]
            if visited != [p["compartments"][c] for c in rounds]:
                raise Unfit(f"oracle rounds {visited} != search order {rounds}")
            shut = [f for f, ev in events if ev.get("message") == "swing shut on the bar"]
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
            bar = [f for f in shut if lo < f < hi]
            if len(gr) < (1 if last or bar else 2):
                raise Unfit(f"round {r}: {len(gr)} grasps")
            out += [(start, "go to the floor mark"), (lo + 1, f"open the {name}")]
            if last:
                out.append((gr[0][1] + 1, "push the cola can"))
            else:
                if bar and not any(c < bar[0] <= o for c, o in gr):
                    raise Unfit(f"round {r}: shut on the bar at frame {bar[0]}, grasps {gr}")
                out.append((bar[0] if bar else gr[0][1] + 1, f"close the {name}"))
                start = gr[-1][1] + 1
        return out
    if task == "depthrecall":
        slots, names = p["original_slots"], [SPOKEN.get(x, x) for x in p["props"]]
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


def check_task_text(src: Path, task: str) -> None:
    """The source's task texts must be the environment's one current instruction (`CURRENT_TEXT`)."""
    if task not in CURRENT_TEXT:
        return
    texts = list(pd.read_parquet(src / "meta/tasks.parquet").index)
    if texts != [CURRENT_TEXT[task]]:
        raise SystemExit(f"{src}: task text {texts} is not the current instruction {CURRENT_TEXT[task]!r}; "
                         "download meta/ and data/ again from the HF dataset (one instruction per task since "
                         "2026-10-30)")

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
    ap.add_argument("params", type=Path, help="task_params.py output, or - (seeds from the dataset)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", action="store_true", help="report only, write no dataset")
    ap.add_argument("--events", default="", help="the oracle's events.jsonl, '{seed}' in the path "
                    "(CabinetSearch5: .../cabinet_search5-*/oracle/{seed}/events.jsonl)")
    args = ap.parse_args()
    src, dst = args.dataset, args.out
    check_task_text(src, args.task)
    if str(args.params) == "-":
        meta = json.loads((src / "meta/source_rlds_metadata.json").read_text())
        params = {int(e): {"seed": int(s)} for e, s in zip(meta["episode_indices"], meta["episode_seeds"])}
    else:
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
            p = params[int(e)]
            ev = oracle_events(args.events, p["seed"]) if args.events else None
            lab = per_frame(segments(args.task, a, g, p, ev), len(a))
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
