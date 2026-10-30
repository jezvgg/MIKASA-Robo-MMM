"""SeasonDish planner verdicts and failure reasons of campaign run dirs, and paired comparisons of two runs.

Reads each <run>/oracle/<seed>/{result.json,events.jsonl} (small files; nothing is changed). Per run: the verdicts,
the stage the planner refused at (its last `FAILED:` line, or `missed:` for a physical miss), the step it refused at,
the sub-reasons of the two big refusals, and the successes' episode length against the horizon.

- "no checked loaded route with a complete bowl pour": after the grasp, lift and fold no dock offset gives an upright
  hover with a wrist-only pour; counted by the hover refusal keys of every tried offset;
- "bounded horizontal grasp/lift/continuation refused": no grasp candidate had a connected contact, lift and carry
  (each candidate's last journal stage), or one was accepted and its execution failed (re-planned contact stroke
  refused, or the fingers closed on nothing).

    python3 sdish_tally.py RUN [RUN ...]            # one table per run, pooled when several
    python3 sdish_tally.py --json OUT RUN [RUN ...] # also write the numbers
    python3 sdish_tally.py --compare BASE NEW       # paired: the seeds both ran, b = only BASE succeeds, c = only NEW
"""
import collections, glob, json, os, re, statistics, sys

HORIZON = 1100  # profile 26's task.horizon; each run's own is read from its run.json


def stage(msg):
    return re.sub(r"\s*\(planner returned -1\)", "", msg).replace("FAILED: ", "")[:90]


def episode(seed_dir):
    """(seed, verdict, stage or None, step, details) of one oracle attempt."""
    r = json.load(open(os.path.join(seed_dir, "result.json")))
    seed, verdict, steps = r.get("seed"), r.get("status"), r.get("steps")
    if verdict == "success":
        return seed, verdict, None, steps, {}
    events = []
    path = os.path.join(seed_dir, "events.jsonl")
    if os.path.exists(path):
        for line in open(path):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    fail = next((e for e in reversed(events) if str(e.get("message", "")).startswith("FAILED")), None)
    miss = next((e for e in reversed(events) if str(e.get("message", "")).startswith("missed")), None)
    if fail is not None:
        name, step = stage(fail["message"]), fail.get("step")
    elif miss is not None:
        name, step = miss["message"][:90], miss.get("step")
    else:
        # drive_base refused the loaded drive: the solver returns -1 without a FAILED line
        last = [e for e in events if e.get("event") == "phase" and e.get("message") not in ("planning budget", "waypoint noise")]
        name, step = f"refused after '{last[-1]['message'][:50]}'" if last else "(no FAILED line)", steps
    details = {}
    if fail is not None and "loaded route" in fail["message"]:
        keys = collections.Counter()
        for tried in fail.get("tried", []):
            for k, n in (tried.get("hover") or {}).items():
                keys[k.split(",")[0][:70]] += n
            if tried.get("stage") != "hover":
                keys[f"{tried.get('stage')} (offset {tried.get('offset_x_m')})"] += 1
        details = dict(route_keys=dict(keys), offsets_tried=len(fail.get("tried", [])))
    elif fail is not None and "bounded horizontal grasp" in fail["message"]:
        last = {}
        accepted = False
        for e in events:
            if e.get("message") == "grasp candidate":
                last[e.get("candidate", e.get("request"))] = f"{e.get('candidate_stage')}:{e.get('reason')}"
                accepted |= e.get("candidate_stage") == "accepted"
        missed = any(e.get("message") == "close missed" for e in events)
        details = dict(candidates=len(last), candidate_last=dict(collections.Counter(last.values())),
                       grasp_outcome=("accepted, fingers closed on nothing" if missed else
                                      "accepted, contact stroke re-plan refused" if accepted else "no candidate accepted"))
    return seed, verdict, name, step, details


def read(run):
    return {s: (v, n, st, d) for s, v, n, st, d in
            (episode(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(run, "oracle", "*", "result.json"))))}


def horizon(run):
    try:
        return int(json.load(open(os.path.join(run, "run.json")))["signature"]["profile"]["task"]["horizon"])
    except (OSError, KeyError, ValueError):
        return HORIZON


def summary(rows, limit=HORIZON):
    verdicts = collections.Counter(v for v, *_ in rows.values())
    stages = collections.Counter(n for v, n, *_ in rows.values() if v != "success")
    route, grasp, outcome = collections.Counter(), collections.Counter(), collections.Counter()
    steps = collections.defaultdict(list)
    for v, n, st, d in rows.values():
        steps["success" if v == "success" else n].append(st)
        route.update(d.get("route_keys", {}))
        grasp.update(d.get("candidate_last", {}))
        if "grasp_outcome" in d:
            outcome[d["grasp_outcome"]] += 1
    ok = steps.get("success", [])
    return dict(n=len(rows), verdicts=dict(verdicts), stages=dict(stages.most_common()),
                route_refusal_keys=dict(route.most_common(10)), grasp_candidate_last=dict(grasp.most_common(10)),
                grasp_outcome=dict(outcome),
                refusal_step_median={k: statistics.median(v) for k, v in steps.items() if k != "success" and v},
                success_steps=dict(median=statistics.median(ok), max=max(ok),
                                   horizon=limit, within_100_of_horizon=sum(s > limit - 100 for s in ok)) if ok else None)


def show(title, s):
    sr = s["verdicts"].get("success", 0)
    print(f"== {title}: SR {sr}/{s['n']} {s['verdicts']}")
    for k, n in s["stages"].items():
        print(f"   {n:4d}  {k}  (step ~{s['refusal_step_median'].get(k)})")
    if s["route_refusal_keys"]:
        print("   loaded-route refusals, hover keys:", s["route_refusal_keys"])
    if s["grasp_outcome"]:
        print("   grasp refusals:", s["grasp_outcome"], "| candidates' last stage:", s["grasp_candidate_last"])
    if s["success_steps"]:
        print(f"   successes: median {s['success_steps']['median']} steps, max {s['success_steps']['max']}, "
              f"{s['success_steps']['within_100_of_horizon']} within 100 steps of the {s['success_steps']['horizon']}-step horizon")


def compare(base, new):
    a, b = read(base), read(new)
    seeds = sorted(set(a) & set(b))
    ok = lambda rows, s: rows[s][0] == "success"
    only_base = [s for s in seeds if ok(a, s) and not ok(b, s)]
    only_new = [s for s in seeds if ok(b, s) and not ok(a, s)]
    print(f"paired on {len(seeds)} seeds: BASE {sum(ok(a, s) for s in seeds)}  NEW {sum(ok(b, s) for s in seeds)}  "
          f"b={len(only_base)} (only BASE)  c={len(only_new)} (only NEW)")
    for s in only_base:
        print(f"   lost   {s}: NEW {b[s][1]}")
    for s in only_new:
        print(f"   gained {s}: BASE {a[s][1]}")
    moved = collections.Counter((a[s][1], b[s][1]) for s in seeds if not ok(a, s) and not ok(b, s) and a[s][1] != b[s][1])
    for (x, y), n in moved.most_common():
        print(f"   both fail, stage moved ({n}): {x} -> {y}")
    return dict(seeds=len(seeds), b=only_base, c=only_new)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--compare"]:
        compare(args[1], args[2])
        raise SystemExit(0)
    out = None
    if args[:1] == ["--json"]:
        out, args = args[1], args[2:]
    result, pooled = {}, {}
    for run in args:
        rows = read(run)
        pooled.update({(run, s): r for s, r in rows.items()})
        result[run] = summary(rows, horizon(run))
        show(run, result[run])
    if len(args) > 1:
        result["pooled"] = summary(pooled, horizon(args[0]))
        show("pooled", result["pooled"])
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
