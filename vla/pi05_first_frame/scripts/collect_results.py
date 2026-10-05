#!/usr/bin/env python3
"""Gather one training run's numbers for the paper (standard library only).

    python3 collect_results.py EXP [--config pi05_sd_ff_4xh100] [--work $PI05_WORK]

Reads what the run left behind and writes $PI05_WORK/results/EXP-paper.json and EXP-paper.md:
  checkpoints/CONFIG/EXP/run_meta.json   config, commits, data, weights, hardware
  checkpoints/CONFIG/EXP/metrics.jsonl   loss curve, wall time -> step time, GPU-hours
  logs/sr-EXP.tsv                        progress SR on the dev seeds
  results/EXP-final-{reset,black}/       final SR on the validation seeds and the control
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def training(ckpt: Path) -> dict:
    meta = load_json(ckpt / "run_meta.json") or {}
    resumes = sorted(glob.glob(str(ckpt / "run_meta.resume-*.json")))
    rows = []
    if (ckpt / "metrics.jsonl").is_file():
        rows = [json.loads(line) for line in (ckpt / "metrics.jsonl").read_text().splitlines() if line.strip()]
    out = {"meta": meta, "resumed": len(resumes)}
    if rows:
        steps = [r["step"] for r in rows]
        times = [r["time"] for r in rows]
        gaps = [(t2 - t1) / (s2 - s1) for (s1, t1), (s2, t2) in zip(zip(steps, times), zip(steps[1:], times[1:]))
                if s2 > s1 and t2 > t1]
        hours = (times[-1] - times[0]) / 3600
        devices = (meta.get("derived") or {}).get("devices") or 1
        out.update({
            "logged_steps": len(rows),
            "last_step": steps[-1],
            "loss_first": rows[0].get("loss"),
            "loss_last": rows[-1].get("loss"),
            "median_seconds_per_step": round(statistics.median(gaps), 3) if gaps else None,
            "wall_hours_logged": round(hours, 2),
            "gpu_hours_logged": round(hours * devices, 1),
            "loss_curve": [[r["step"], r.get("loss")] for r in rows],
        })
    return out


def progress(tsv: Path) -> list[dict]:
    if not tsv.is_file():
        return []
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]


def evaluation(out: Path) -> dict | None:
    summary = load_json(out / "summary.json")
    if not summary:
        return None
    runs = [load_json(p) for p in sorted(glob.glob(str(out / "shard-*/run.json")))]
    first = runs[0] if runs else {}
    argv = first.get("argv", [])

    def flag(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    return {
        "summary": summary,
        "checkpoint": (first.get("server_metadata") or {}).get("checkpoint"),
        "protocol": {
            "seeds": sum(len(r.get("seeds", [])) for r in runs),
            "seed_source": flag("--seeds", "validation"),
            "replan_steps": int(flag("--replan-steps", 5)),
            "max_policy_steps": int(flag("--max-policy-steps", 800)),
            "first_frame": flag("--first-frame", "reset"),
            "code_sha256": first.get("code_sha256"),
            "engine_sha256": first.get("engine_sha256"),
            "environment": first.get("environment"),
        },
    }


def markdown(report: dict) -> str:
    t, meta = report["training"], report["training"].get("meta", {})
    cfg, derived, code, data = meta.get("train_config", {}), meta.get("derived", {}), meta.get("code", {}), meta.get("data", {})
    lr = cfg.get("lr_schedule", {})
    lines = [f"# {report['exp']}: pi0.5 + first frame, OpenSameDrawer", "", "## Setup", "",
             "| | |", "|---|---|",
             f"| Code | MIKASA-Robo-MMM `{(code.get('repo_commit') or '?')[:10]}`, openpi `{(code.get('openpi_commit') or '?')[:10]}` + patch `{(code.get('openpi_patch_sha256') or '?')[:10]}` |",
             f"| Data | `{data.get('hf_repo')}`@`{(data.get('hf_revision') or '?')[:7]}`, {data.get('total_episodes')} episodes, {data.get('total_frames')} frames, {data.get('fps')} Hz |",
             f"| Hardware | {derived.get('devices')} x {derived.get('device_kind')} |",
             f"| Compute | {t.get('last_step')} steps logged, {t.get('median_seconds_per_step')} s/step (median), {t.get('wall_hours_logged')} h wall, {t.get('gpu_hours_logged')} GPU-h |",
             f"| Model | pi0.5 from `{(meta.get('weights') or {}).get('base')}`; images {derived.get('image_keys')} (first frame: {derived.get('first_frame_camera')}); frozen: `{derived.get('frozen')}` |",
             f"| Batch, steps | {cfg.get('batch_size')} global ({derived.get('per_device_batch')}/device), {cfg.get('num_train_steps')} steps, {derived.get('epochs')} epochs |",
             f"| Optimizer | AdamW, LR warmup {lr.get('warmup_steps')} -> peak {lr.get('peak_lr')} -> cosine to {lr.get('decay_lr')} over {lr.get('decay_steps')}; EMA {cfg.get('ema_decay')} |",
             f"| Actions | horizon {derived.get('action_horizon')}; state in prompt: {derived.get('discrete_state_input')} |",
             f"| Loss | {t.get('loss_first')} -> {t.get('loss_last')} |", ""]
    for name, key in (("Final evaluation (validation seeds)", "final"), ("Control: black first frame", "control")):
        ev = report.get(key)
        if not ev:
            continue
        s, p = ev["summary"], ev["protocol"]
        lo, hi = s["wilson95"]
        lines += [f"## {name}", "",
                  f"SR **{s['success']}/{s['episodes']} = {s['success_rate']:.2f}** (95% Wilson CI {lo:.2f}-{hi:.2f}); "
                  f"checkpoint step {(ev.get('checkpoint') or {}).get('step')}; {p['seeds']} seeds ({p['seed_source']}), "
                  f"replan every {p['replan_steps']} of {derived.get('action_horizon')} actions, limit {p['max_policy_steps']} policy steps.", "",
                  "| Target drawer | Episodes | Success |", "|---|---|---|"]
        lines += [f"| {d} | {v['n']} | {v['success']} |" for d, v in s["by_target_drawer"].items()]
        lines += ["", "| Stage | Episodes |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["stages"].items()] + [""]
    if report["progress"]:
        lines += ["## Progress SR (dev seeds)", "", "| Step | SR | 95% CI |", "|---|---|---|"]
        lines += [f"| {r['step']} | {r['success']}/{r['episodes']} | {r['wilson_lo']}-{r['wilson_hi']} |" for r in report["progress"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp")
    parser.add_argument("--config", default="pi05_sd_ff_4xh100")
    parser.add_argument("--work", default=os.environ.get("PI05_WORK", os.path.expanduser("~/mikasa-pi05")))
    args = parser.parse_args()
    work = Path(args.work)
    report = {
        "exp": args.exp,
        "config": args.config,
        "training": training(work / "checkpoints" / args.config / args.exp),
        "progress": progress(work / "logs" / f"sr-{args.exp}.tsv"),
        "final": evaluation(work / "results" / f"{args.exp}-final-reset"),
        "control": evaluation(work / "results" / f"{args.exp}-final-black"),
    }
    out = work / "results" / f"{args.exp}-paper"
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    out.with_suffix(".md").write_text(markdown(report))
    print(out.with_suffix(".md").read_text())
    print(f"written: {out}.json, {out}.md")


if __name__ == "__main__":
    main()
