"""SameDrawer AG exact-1000 selection, same rule as samedrawer-1000ep (pod selection v2).

usage: build_selection.py OUT.json
First 250 accepted episodes of each of the four drawers in collection order (batch number, then
episode order within the batch export) among episodes passing D6, from the ag- ledgers.
"""
import datetime, json, os, re, sys
from pathlib import Path

A = Path(os.path.expanduser("~/sd-collect/driver"))
bank = json.loads((A / "bank.json").read_text())["seed_to_drawer"]
ledgers = sorted((A / "ledger").glob("same_drawer-ag-train-*.json"),
                 key=lambda p: int(re.search(r"train-(\d+)", p.name).group(1)))
per = {str(d): [] for d in range(4)}
inputs, d6_failed, accepted_total = [], [], 0
for path in ledgers:
    L = json.loads(path.read_text())
    rows = {r["seed"]: r for r in L["rows"]}
    accepted_total += len(L["accepted_seeds"])
    used = False
    for seed in L["accepted_seeds"]:                     # export order within the batch
        if not rows[seed].get("D6"):
            d6_failed.append(seed); continue
        d = str(bank[str(seed)])
        if len(per[d]) < 250:
            per[d].append(seed); used = True
    inputs += L["exports"]
chosen = {s for v in per.values() for s in v}
order = [s for path in ledgers for s in json.loads(path.read_text())["accepted_seeds"] if s in chosen]
sel = dict(task="SameDrawer", profile=21, assisted_grasp=True,
           declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           rule=("first 250 accepted episodes of each of the four drawers in collection order (batch number, then episode "
                 "order within the batch export) among episodes that pass D6 (<=5% identical adjacent 10 Hz actions, "
                 "longest pause <=20 frames, per the batch motion audit)"),
           d6_failed_accepted_seeds=d6_failed, accepted_total=accepted_total,
           per_drawer={k: len(v) for k, v in per.items()}, episode_seeds=order, inputs=inputs)
assert len(order) == 1000 and all(len(v) == 250 for v in per.values()), sel["per_drawer"]
Path(sys.argv[1]).write_text(json.dumps(sel, indent=1) + "\n")
print("selection", sel["per_drawer"], "accepted", accepted_total, "D6 failed", len(d6_failed), "inputs", len(inputs))
