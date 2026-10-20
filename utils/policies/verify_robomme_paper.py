"""Reload the three paper RoboMME checkpoints and query real Fetch episodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/robomme_paper"))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    root = Path(os.environ["POLICY_WORKSPACE"])
    from utils.policies.workspace import contained, require_space
    output = contained(root, args.output)
    report_path = output / f"{args.exp_name}-reload-verification.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    record = {"recipe": "paper", "complete": False, "variants": {}}
    def save():
        report_path.write_text(json.dumps(record, indent=2))
    if args.wait_pid:
        deadline = time.monotonic() + 1800
        while Path(f"/proc/{args.wait_pid}/cmdline").exists():
            try:
                command = Path(f"/proc/{args.wait_pid}/cmdline").read_bytes()
            except FileNotFoundError:
                break
            if not command:  # exited zombie awaiting reaping
                break
            if b"utils.policies.workspace" not in command or str(root).encode() not in command:
                raise RuntimeError("wait PID no longer identifies our workspace wrapper")
            if time.monotonic() > deadline:
                raise TimeoutError("previous owned smoke did not exit within 30 minutes")
            time.sleep(5)
    running = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True).strip()
    if running:
        raise RuntimeError(f"GPU already occupied; refusing parallel inference: {running}")
    python = root / ".venv-robomme/bin/python"
    env = {**os.environ, "PYTHONPATH": f"{root}/third_party/robomme/src:{root}",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.60"}
    try:
        for variant in ("pi05", "framesamp", "tokendrop"):
            require_space(root, 10_000_000_000)
            evidence = output / f"{variant}-{args.exp_name}-verification.json"
            training = json.loads(evidence.read_text())
            if (training["optimizer_steps"] != 3 or not training["all_losses_gradients_finite"]
                    or not training["parameter_audit"]["frozen_unchanged"]
                    or not training["parameter_audit"]["trainable_groups_updated"]):
                raise RuntimeError(f"paper training gate failed: {evidence}")
            checkpoint = Path(training["checkpoint_dir"]) / "2"
            server_log = output / f"{variant}-{args.exp_name}-server.log"
            probe_log = output / f"{variant}-{args.exp_name}-probe.log"
            probe_json = output / f"{variant}-{args.exp_name}-probe.json"
            command = [str(python), "-u", "-m", "utils.policies.robomme", "serve", "--recipe", "paper",
                "--variant", variant, "--root", "dataset", "--features", "logs/features_bounded",
                "--output", str(output), "--base-params", str(args.base_params), "--fsdp-devices", "1",
                "--batch-size", "1", "--exp-name", args.exp_name, "--checkpoint", str(checkpoint), "--port", "8000"]
            with server_log.open("x") as log:
                server = subprocess.Popen(command, env=env, cwd=root, stdout=log,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                try:
                    deadline = time.monotonic() + 600
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"paper server failed; see {server_log}")
                        try:
                            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                                break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError(f"paper server not ready; see {server_log}")
                            time.sleep(1)
                    with probe_log.open("x") as log:
                        subprocess.run([str(python), "-u", "-m", "utils.policies.probe_policy",
                            "--backend", "robomme", "--root", "dataset", "--port", "8000",
                            "--output", str(probe_json)], check=True, env=env, cwd=root,
                            stdout=log, stderr=subprocess.STDOUT, timeout=600)
                    probe = json.loads(probe_json.read_text())
                    episodes = probe["real_dataset_probe"]
                    if ({item["task"] for item in episodes} != {"takeitback", "veggie"}
                            or not all(item["reset_confirmed"] for item in episodes)
                            or not all(shape == [20, 13] for item in episodes for shape in item["action_shapes"])):
                        raise RuntimeError(f"paper policy probe gate failed: {probe_json}")
                    record["variants"][variant] = {"checkpoint": str(checkpoint), "training_evidence": str(evidence),
                        "real_query_evidence": str(probe_json), "reload_and_action13": "PASS"}
                    save()
                finally:
                    if server.poll() is None:
                        server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
        record["complete"] = True
        save()
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
