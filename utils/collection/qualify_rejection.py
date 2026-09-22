"""Negative control: an unsuccessful physical episode must not pass the data gate."""
from pathlib import Path
import argparse

import numpy as np

from .client import execute_actions
from .contract import read_json, recording_info, write_json
from .pipeline import add_contract, collection_contract, preserve_diagnostic, record_env
from .profile import make_env


def qualify(root, output):
    run = read_json(root / "run.json")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "trajectory.h5"
    env = record_env(make_env(run), path)
    try:
        env.reset(seed=351)
        ctrls = env.unwrapped.agent.controller.controllers
        hold = np.concatenate([ctrls["arm"].qpos[0].cpu().numpy(), [1.],
                               ctrls["body"].qpos[0].cpu().numpy(), [0., 0.]])
        result = execute_actions(env, np.tile(hold, (4, 1)))
        if result["success"]:
            raise AssertionError("The no-search negative control unexpectedly succeeded")
        env.flush_trajectory(save=True)
        contract = collection_contract(env.unwrapped, run=run, seed=351, stage="validated")
    finally:
        env.close()
    add_contract(path, contract)
    try:
        recording_info(path, stage="validated")
    except ValueError as exc:
        if "success" not in str(exc).lower():
            raise
        reason = str(exc)
    else:
        raise AssertionError("An unsuccessful episode passed the export gate")
    diagnostic = preserve_diagnostic(path)
    report = dict(status="check_passed", seed=351, task_success=False,
                  control_steps=result["control_steps"], rejection=reason,
                  diagnostic_h5=diagnostic, source_sha256=run["signature"]["code_sha256"])
    write_json(output / "result.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.run.resolve(), args.output.resolve())
