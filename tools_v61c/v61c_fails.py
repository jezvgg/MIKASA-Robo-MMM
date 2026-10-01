"""Print one line per failed seed of a run: python fails.py <out_dir> (reads <out_dir>/metrics.json and the diag flags)."""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rows = json.load(open(out / "metrics.json"))["rows"]
print("success", sum(r["success"] for r in rows), "/", len(rows))
for x in rows:
    d = np.load(out / "diag" / f"diag_seed{x['seed']}.npz")
    f = d["flags"][-1].astype(int).tolist()
    if x["success"] and "-all" not in sys.argv:
        continue
    print(x["seed"], x["success"], "retries", x["retries"], "turn", round(x["base_turn_deg"]), "steps", x["env_steps"],
          "tilt", round(x.get("tilt_deg", -1), 1), "flags", f, "hand", x["hand_contact_steps"], "arm", x["arm_contact_steps"])
