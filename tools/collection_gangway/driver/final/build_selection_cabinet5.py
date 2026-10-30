"""CabinetSearch5 AG exact-1000 selection, the cabinetsearch-ag-1000ep rule.

usage: build_selection_cabinet5.py OUT.json [N]
First N (default 1000) accepted episodes in collection order (batch number, then episode order within the
batch export) among episodes that pass D6, from the cabinet_search5-c5v7- ledgers (five compartments,
profile 7). N other than 1000 is a dry run: it prints and writes nothing.
"""
import collections, datetime, json, os, re, sys
from pathlib import Path

import numpy as np

A = Path(os.path.expanduser("~/sd-collect/driver"))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
ledgers = sorted((A / "ledger").glob("cabinet_search5-c5v7-train-*.json"),
                 key=lambda p: int(re.search(r"train-(\d+)", p.name).group(1)))
order, inputs, d6_failed, accepted_total = [], [], [], 0
for path in ledgers:
    L = json.loads(path.read_text())
    rows = {r["seed"]: r for r in L["rows"]}
    accepted_total += len(L["accepted_seeds"])
    for seed in L["accepted_seeds"]:
        if not rows[seed].get("D6"):
            d6_failed.append(seed); continue
        order.append(seed)
    inputs += L["exports"]
order = order[:N]
# the can's compartment is the ticket's first draw (RandomState(seed).randint(5)); 0 is cab_1, printed from 1
can = collections.Counter(int(np.random.RandomState(s).randint(5)) + 1 for s in order)
print("selection", len(order), "accepted", accepted_total, "D6 failed", len(d6_failed), "inputs", len(inputs),
      "| can in cabinet", dict(sorted(can.items())))
if N != 1000:
    raise SystemExit(0)
sel = dict(task="CabinetSearch5", profile=7, assisted_grasp=True, compartments=5,
           declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           rule=("first 1000 accepted episodes in collection order (batch number, then episode order within the batch "
                 "export) among episodes that pass D6 (<=5% identical adjacent 10 Hz actions, longest pause <=20 frames, "
                 "per the batch motion audit)"),
           d6_failed_accepted_seeds=d6_failed, accepted_total=accepted_total, can_cabinet_counts=dict(sorted(can.items())),
           episode_seeds=order, inputs=inputs)
assert len(order) == 1000 and len(set(order)) == 1000, (len(order), accepted_total)
Path(sys.argv[1]).write_text(json.dumps(sel, indent=1) + "\n")
