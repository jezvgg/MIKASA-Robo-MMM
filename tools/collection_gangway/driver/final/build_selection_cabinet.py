"""CabinetSearch AG exact-1000 selection, same rule as cabinetsearch-1000ep (pod cabinet_search-p15-selection.json).

usage: build_selection_cabinet.py OUT.json
First 1000 accepted episodes in collection order (batch number, then episode order within the batch export)
among episodes that pass D6, from the cabinet_search-ag- ledgers.
"""
import datetime, json, os, re, sys
from pathlib import Path

A = Path(os.path.expanduser("~/sd-collect/driver"))
ledgers = sorted((A / "ledger").glob("cabinet_search-ag-train-*.json"),
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
order = order[:1000]
sel = dict(task="CabinetSearch", profile=16, assisted_grasp=True,
           declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           rule=("first 1000 accepted episodes in collection order (batch number, then episode order within the batch "
                 "export) among episodes that pass D6 (<=5% identical adjacent 10 Hz actions, longest pause <=20 frames, "
                 "per the batch motion audit)"),
           d6_failed_accepted_seeds=d6_failed, accepted_total=accepted_total, episode_seeds=order, inputs=inputs)
assert len(order) == 1000 and len(set(order)) == 1000, (len(order), accepted_total)
Path(sys.argv[1]).write_text(json.dumps(sel, indent=1) + "\n")
print("selection", len(order), "accepted", accepted_total, "D6 failed", len(d6_failed), "inputs", len(inputs))
