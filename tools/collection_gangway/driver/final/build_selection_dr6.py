"""DepthRecall AG exact-1000 selection, the SeasonDish rule (first 1000 accepted; D6 is a hard gate).

usage: build_selection_dr.py OUT.json
First 1000 accepted episodes in collection order (batch number, then episode order within the batch
export) from the depth_recall-ag6- ledgers (profile 6). D6 is a hard gate for DepthRecall.
"""
import datetime, json, os, re, sys
from pathlib import Path

A = Path(os.path.expanduser("~/sd-collect/driver"))
ledgers = sorted((A / "ledger").glob("depth_recall-ag6-train-*.json"),
                 key=lambda p: int(re.search(r"train-(\d+)", p.name).group(1)))
order, inputs, accepted_total = [], [], 0
for path in ledgers:
    L = json.loads(path.read_text())
    accepted_total += len(L["accepted_seeds"])
    order += L["accepted_seeds"]
    inputs += L["exports"]
order = order[:1000]
sel = dict(task="DepthRecall", profile=6, assisted_grasp=True,
           declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           rule="first 1000 accepted episodes in collection order (batch number, then episode order within the batch export)",
           accepted_total=accepted_total, episode_seeds=order, inputs=inputs)
assert len(order) == 1000 and len(set(order)) == 1000, (len(order), accepted_total)
Path(sys.argv[1]).write_text(json.dumps(sel, indent=1) + "\n")
print("selection", len(order), "accepted", accepted_total, "inputs", len(inputs))
