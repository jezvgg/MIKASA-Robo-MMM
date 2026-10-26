"""Append the five-compartment dataset's notes (E1 rug rule, the D5 exception) to the README card that
finalize_dataset.py wrote, with E1 and D5 measured on the selected episodes. Idempotent.

usage: readme_cabinet5_notes.py DATASET SELECTION_JSON
E1 per episode: <batch>-views-e1.json (batches gated before the rule, re-scored) or <batch>-views-combined.json
(batches gated with it). D5: the batch motion audit (D5_direction_pass, motion10 reverse_frames and
base_direction_changes).
"""
import gzip, json, os, re, statistics, sys
from pathlib import Path

root, selection = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
A = Path(os.path.expanduser("~/sd-collect/driver"))
seeds = set(selection["episode_seeds"])


def load(p):
    return json.loads(gzip.open(str(p) + ".gz").read()) if not p.exists() else json.loads(p.read_text())


e1, raw, d5_fail, reverse, changes = {}, {}, 0, [], []
for export in selection["inputs"]:
    batch = re.match(r"(cabinet_search5-c5v7-train-\d+)-lerobot", Path(export).name).group(1)
    rescored = A / f"{batch}-views-e1.json"
    views = load(rescored) if rescored.exists() else load(A / f"{batch}-views-combined.json")
    for e in views["episodes"]:
        if e["seed"] in seeds:
            e1[e["seed"]], raw[e["seed"]] = e["visible_fraction"], e["visible_fraction_raw"]
    for e in load(A / f"{batch}-motion-audit.json")["episodes"]:
        if e["seed"] in seeds:
            d5_fail += not e["D5_direction_pass"]
            reverse.append(e["motion10"]["reverse_frames"])
            changes.append(e["motion10"]["base_direction_changes"])
n = len(seeds)
assert len(e1) == n and len(reverse) == n, (len(e1), len(reverse), n)
v = list(e1.values())
card = (root / "README.md").read_text()
mark = "\n## Five compartments\n"
card = card.split(mark)[0].rstrip() + "\n" + mark + f"""
`MikasaCabinetSearch5-v0` adds cab_1, the west wall unit, to CabinetSearch's four compartments as compartment
index 0, so the can is hidden in one of five. cab_1's door is opened by its handle and held open during the
look; the other four are opened as in CabinetSearch.

### Checklist E1 (target in frame)

Per frame, the current target must be in the segmentation of at least one of the three cameras. The targets
follow the oracle's phases: the rug while driving home, the leaf of the cabinet being searched during a round,
the can once found.

The rug (radius 0.20 m) is smaller than the robot's base, so once the robot stands on it no camera can see
it. In the rug phases a frame with the base centre within `home_radius` (0.20 m) of the rug centre therefore
counts as the destination reached (owner decision 2026-10-25).

On these {n} episodes: E1 min {min(v):.3f}, median {statistics.median(v):.3f}, {sum(x >= 0.95 for x in v)}/{n} at or
above 0.95; camera visibility alone, median {statistics.median(raw.values()):.3f}. During collection the rug was
in some camera on 99-100 % of the frames at every base distance of 0.1 m or more; the remaining misses are the
first frames of a round, while the head turns from the rug to the cabinet.

### Documented exception: D5 (base direction changes)

The checklist's D5 asks the base not to drive back and forth. CabinetSearch does that by design: every round
drives from the rug to a cabinet, and after closing it the robot backs off and returns to the rug. D5 fails in
{d5_fail}/{n} episodes, with a median of {statistics.median(reverse):.0f} reversing frames and
{statistics.median(changes):.0f} changes of direction an episode. The owner chose on 2026-10-25 to keep the motion
and document it as a task exception, as for DepthRecall. D5 was measured for every episode and never used to
select episodes.
"""
(root / "README.md").write_text(card)
print("README notes appended:", root / "README.md", f"| E1 min {min(v):.3f} | D5 failed {d5_fail}/{n}")
