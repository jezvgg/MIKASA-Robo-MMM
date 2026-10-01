"""Metrics of takeitback-tray planner runs from the per-step diag dumps of `dev/v61c_diag.py` plus the planner events.

    python v61c_metrics.py <out_dir> [<out_dir> ...] [--json out.json]

<out_dir> holds diag/diag_seed<S>.{npz,json} and trace/seed_<S>/*_events.jsonl (the layout of dev/run_diag.sh).
Success is the env verdict from the events; `proxy_success` is the checker-independent one (cup lifted >= 3 cm and
resting on the tray). All per-step arrays are 20 Hz env steps; smoothness and reversals use the 10 Hz stream.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_metrics as rev  # noqa: E402
import smooth_metrics as sm  # noqa: E402

logger = logging.getLogger(__name__)
ARM_LINKS = ("shoulder", "upperarm", "elbow", "forearm", "wrist")
HAND_LINKS = ("gripper", "finger")
OBJECT_PARTS = ("cup", "baking_tray")
ROLL_QPOS = (8, 10, 12)        # upperarm, forearm, wrist roll in the 15-D active qpos
HEAD_PAN = 4
VISIBLE_HALF_ANGLE = np.deg2rad(30.0)   # assumed half field of view of the head camera for the visibility proxy
COUNTER_TOP = 0.92


def _events_file(out: Path, seed: int) -> Optional[Path]:
    files = sorted((out / "trace" / f"seed_{seed}").glob("*_events.jsonl"))
    return files[0] if files else None


def _contact_steps(contacts: List[List[str]]) -> Dict[str, int]:
    """Env steps with a robot-vs-furniture contact: any link, forearm-and-above links, and the hand (gripper, fingers)."""
    anyc = armc = handc = 0
    for step in contacts:
        furn = [c for c in step if not any(p in c.split("~")[1] for p in OBJECT_PARTS)]
        anyc += bool(furn)
        armc += any(any(k in c.split("~")[0] for k in ARM_LINKS) for c in furn)
        handc += any(any(k in c.split("~")[0] for k in HAND_LINKS) for c in furn)
    return {"furniture_contact_steps": anyc, "arm_contact_steps": armc, "hand_contact_steps": handc}


def _final_cup(d, meta: dict) -> dict:
    """Strict-checker quantities at the last step: tilt of the cup, offset in the tray frame, bottom-to-tray-top error."""
    cq, tq, cup, tray = d["cup_q"][-1], d["tray_q"][-1], d["cup"][-1], d["tray"][-1]
    up_z = 1 - 2 * (cq[1] ** 2 + cq[2] ** 2)
    yaw = np.arctan2(2 * (tq[0] * tq[3] + tq[1] * tq[2]), 1 - 2 * (tq[2] ** 2 + tq[3] ** 2))
    dx, dy = cup[0] - tray[0], cup[1] - tray[1]
    local = np.array([np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy])
    half = np.array(meta["tray_half"][:2])
    bottom = meta["cup_bottom"]
    return {"tilt_deg": float(np.degrees(np.arccos(np.clip(up_z, -1, 1)))), "plate_margin_cm": float(100 * np.min(half - np.abs(local))),
            "bottom_err_cm": float(100 * ((cup[2] - bottom) - (tray[2] + meta["tray_half"][2])))}


def episode(out: Path, seed: int) -> Optional[dict]:
    """All metrics of one seed, or None when the dump is missing."""
    npz, js = out / "diag" / f"diag_seed{seed}.npz", out / "diag" / f"diag_seed{seed}.json"
    if not npz.exists():
        return None
    d, meta = np.load(npz), json.loads(js.read_text())
    act, q, tcp, cup, tray, base = (d[k] for k in ("action", "qpos", "tcp", "cup", "tray", "base"))
    evf = _events_file(out, seed)
    ev = [json.loads(l) for l in evf.read_text().splitlines() if l.strip()] if evf else []
    verdict = next((bool(e["success"]) for e in ev if e["event"] == "verdict"), False)
    m = {"seed": seed, "success": verdict, "env_steps": len(act), "retries": max(0, sum(e["event"] == "waypoint" for e in ev) - 1)}
    lift = float(cup[:, 2].max() - meta["meta"]["cup_pos0"][2])
    rest = float(np.abs(cup[-10:] - cup[-1]).max()) < 1e-3
    on_tray = bool(np.hypot(*(cup[-1, :2] - tray[-1, :2])) < 0.12 and 0.0 < cup[-1, 2] - tray[-1, 2] < 0.15)
    m.update(cup_lift_cm=100 * lift, proxy_success=bool(lift >= 0.03 and on_tray and rest))
    if "cup_q" in d.files and "cup_bottom" in meta["meta"]:
        m.update(_final_cup(d, meta["meta"]))
    a10 = act[::2]
    phase = sm.phases_from_events(str(evf), len(a10)) if evf else None
    sms = sm.episode_smoothness(a10, phase)
    m.update({k: sms[k] for k in ("max_dq_arm", "max_jerk_arm", "p99_jerk_arm", "n_jumps", "n_branch_jumps", "roll_variation", "n_still_runs", "max_still_frames")})
    rm = rev.episode_metrics(base[::2]) if len(base) > 4 else {"reversals": 0, "total_rotation_deg": 0.0, "backward_m": 0.0}
    m.update(reversals=int(rm["reversals"]), base_turn_deg=float(rm["total_rotation_deg"]), backward_m=float(rm["backward_m"]))
    rolls = q[:, list(ROLL_QPOS)]
    m.update(roll_wrap_steps=int((np.abs(rolls) > np.pi).any(axis=1).sum()), roll_abs_max=float(np.abs(rolls).max()))
    m.update(_contact_steps(meta["contacts"]))
    closed = np.nonzero(act[:, 7] < 0)[0]
    pre = slice(0, int(closed[0]) if len(closed) else len(act))
    over = (tcp[pre, 1] > -0.65) & (tcp[pre, 0] > 0.0)
    m["tcp_min_above_counter_cm"] = float(100 * (tcp[pre][over, 2].min() - COUNTER_TOP)) if over.any() else float("nan")
    bearing = np.arctan2(cup[:, 1] - base[:, 1], cup[:, 0] - base[:, 0]) - (base[:, 2] + q[:, HEAD_PAN])
    m["cup_visible_frac"] = float(np.mean(np.abs((bearing + np.pi) % (2 * np.pi) - np.pi) < VISIBLE_HALF_ANGLE))
    return m


def summarize(rows: List[dict]) -> dict:
    """Aggregate over seeds; the reversal rate and base turn are over successes."""
    ok = [r for r in rows if r["success"]]
    med = lambda k, rs=rows: float(np.nanmedian([r[k] for r in rs])) if rs else float("nan")
    return {"n": len(rows), "success": len(ok), "proxy_success": sum(r["proxy_success"] for r in rows),
            "median_base_turn_deg": med("base_turn_deg", ok), "rev_rate_success": float(np.mean([r["reversals"] > 0 for r in ok])) if ok else float("nan"),
            "arm_contact_seeds": sum(r["arm_contact_steps"] > 0 for r in rows), "arm_contact_steps_total": sum(r["arm_contact_steps"] for r in rows),
            "hand_contact_seeds": sum(r["hand_contact_steps"] > 0 for r in rows), "hand_contact_steps_total": sum(r["hand_contact_steps"] for r in rows),
            "median_roll_variation": med("roll_variation"), "mean_roll_variation": float(np.mean([r["roll_variation"] for r in rows])),
            "roll_wrap_seeds": sum(r["roll_wrap_steps"] > 0 for r in rows), "seeds_with_retry": sum(r["retries"] > 0 for r in rows),
            "retries_total": sum(r["retries"] for r in rows), "seeds_stall_ge_1s": sum(r["max_still_frames"] >= 10 for r in rows),
            "median_env_steps": med("env_steps"), "median_cup_visible": med("cup_visible_frac"), "median_cup_lift_cm": med("cup_lift_cm", ok),
            "max_dq_arm": max(r["max_dq_arm"] for r in rows), "max_jerk_arm": max(r["max_jerk_arm"] for r in rows),
            "median_tcp_min_above_counter_cm": med("tcp_min_above_counter_cm"), "jumps": sum(r["n_jumps"] for r in rows)}


def main(argv: List[str]) -> None:
    js = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [Path(a) for a in argv if not a.startswith("--") and a != js]
    for out in dirs:
        seeds = sorted(int(p.stem.removeprefix("diag_seed")) for p in (out / "diag").glob("diag_seed*.npz"))
        rows = [r for r in (episode(out, s) for s in seeds) if r]
        res = {"dir": str(out), "summary": summarize(rows), "rows": rows}
        print(json.dumps(res["summary"]))
        if js:
            Path(js).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
