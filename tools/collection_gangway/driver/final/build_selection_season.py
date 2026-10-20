"""SeasonDish AG exact-1000 selection, same rule as seasondish-1000ep (pod season_dish-p25-selection.json).

usage: build_selection_season.py OUT.json
First 1000 accepted episodes in collection order (batch number, then episode order within the batch
export) from the season_dish-ag- ledgers. D6 is already a hard gate for SeasonDish.
"""
import datetime, json, os, re, sys
from pathlib import Path

A = Path(os.path.expanduser("~/sd-collect/driver"))
ledgers = sorted((A / "ledger").glob("season_dish-ag-train-*.json"),
                 key=lambda p: int(re.search(r"train-(\d+)", p.name).group(1)))
order, inputs, accepted_total = [], [], 0
for path in ledgers:
    L = json.loads(path.read_text())
    accepted_total += len(L["accepted_seeds"])
    order += L["accepted_seeds"]
    inputs += L["exports"]
order = order[:1000]
sel = dict(task="SeasonDish", profile=26, assisted_grasp=True,
           declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           rule="first 1000 accepted episodes in collection order (batch number, then episode order within the batch export)",
           accepted_total=accepted_total, episode_seeds=order, inputs=inputs)
assert len(order) == 1000 and len(set(order)) == 1000, (len(order), accepted_total)
Path(sys.argv[1]).write_text(json.dumps(sel, indent=1) + "\n")
print("selection", len(order), "accepted", accepted_total, "inputs", len(inputs))
