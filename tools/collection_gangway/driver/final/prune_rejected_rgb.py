"""Delete the RGB trajectory.h5 of rejected source successes (owner 2026-10-18: «да», after the dataset's final assembly).

usage: prune_rejected_rgb.py <task>      (task: same_drawer, season_dish, ...)
Retention already removed rgb/<seed>/trajectory.h5 of every exported episode; rejected ones were never exported, so
their RGB h5 stayed. It is re-renderable from the kept oracle/native/validated h5; the rejection reasons stay in the
ledger. Only rgb/<seed>/trajectory.h5 goes; json/log files stay. Writes driver/deletions/<task>-ag-rejected-rgb.json.
"""
import datetime, glob, json, os, sys
from pathlib import Path

task = sys.argv[1]
A = Path(os.path.expanduser("~/sd-collect/driver")); C = Path(os.path.expanduser("~/sd-collect/runs-ag")).resolve()
out = A / "deletions" / f"{task}-ag-rejected-rgb.json"
out.parent.mkdir(exist_ok=True)
rec = json.loads(out.read_text()) if out.exists() else dict(task=task, rule=__doc__.strip().splitlines()[0], files=[])
done = {f["path"] for f in rec["files"]}
before, freed = len(rec["files"]), 0
for lp in sorted(glob.glob(str(A / "ledger" / f"{task}-ag-train-*.json"))):
    L = json.loads(Path(lp).read_text())
    exported = set(L["exported_seeds"])
    run = Path(L["run"]).resolve()
    assert run.parent == C, run
    for r in L["rows"]:
        s = r["seed"]
        if r["source"] != "success" or r["accepted"] or s in exported:
            continue
        p = run / "rgb" / str(s) / "trajectory.h5"
        if not p.is_file() or str(p) in done:
            continue
        size = p.stat().st_size
        p.unlink()
        freed += size
        rec["files"].append(dict(path=str(p), bytes=size, seed=s, batch=L["batch"], reasons=r["reasons"],
                                 utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
out.write_text(json.dumps(rec, indent=1) + "\n")
print(f"{task}: deleted {len(rec['files']) - before} rejected RGB h5, {freed / 2**30:.2f} GiB; manifest {out}")
