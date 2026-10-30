"""Point the cabinet_search5 collection at a new profile version: driver + gate repo, launcher, SEED_PLAN.

    python3 cab5_bump.py VERSION COMMIT CLONE PREFIX FIRST_SEED "PREVIOUS (not in the dataset)" "CHANGE" "REASON"
"""
import datetime
import json
import re
import sys
from pathlib import Path

version, commit, clone, prefix, first, previous, change, reason = sys.argv[1:9]
H = Path.home() / "sd-collect"
A = H / "driver"
line = (f"cabinet_search5=os.path.expanduser('~/sd-collect/{clone}'),  # five compartments, profile {version}, "
        f"feat/cabinet-fifth-door {commit} (unpushed)")
for name in ("collection_driver.py", "postbatch.py"):
    f = A / name
    s = f.read_text()
    s2, n = re.subn(r"cabinet_search5=os\.path\.expanduser\('~/sd-collect/repo-cab5c\d*'\),[^\n]*", line, s)
    assert n == 1, (name, n)
    f.write_text(s2)
    print("patched", name)

f = H / "launch_cab5.sh"
s = f.read_text()
s, n1 = re.subn(r"\(cabinet_search5 profile \d+, ~/sd-collect/repo-cab5c\d* = \w+ on",
                f"(cabinet_search5 profile {version}, ~/sd-collect/{clone} = {commit} on", s)
s, n2 = re.subn(r"^# seeds from \d+, prefix c5\w*-", f"# seeds from {first}, prefix {prefix}", s, flags=re.M)
s, n3 = re.subn(r"\$\{BUDGET:-110\} c5\w*- \d+ >>", "${BUDGET:-110} " + prefix + " " + first + " >>", s)
assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
f.write_text(s)
print("patched launch_cab5.sh")

f = A / "SEED_PLAN.json"
p = json.loads(f.read_text())
tag = f"cabinet_search5 profile {version} "
assert not any(a["change"].startswith(tag) for a in p["amendments"]), "already declared"
p["amendments"].append(dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    change=(f"{tag}(commit {commit} on feat/cabinet-fifth-door, NOT public; pinned clone ~/sd-collect/{clone}) "
            f"replaces the previous profile: {change} Training: sequential seeds from {first}, batches "
            f"cabinet_search5-{prefix}train-NNN of 20, 8 jobs, target 1020 accepted, selection = first 1000 accepted "
            f"in collection order. {previous} is NOT part of the dataset; its runs stay on disk until the owner decides."),
    reason=reason, dataset_rule=p["amendments"][-1]["dataset_rule"]))
f.write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n")
print("amendments", len(p["amendments"]))
