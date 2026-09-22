"""Policy observations and the shared 10 Hz target executor.

Only qpos, RGB and the task's language instruction cross the policy boundary.
The simulation and task clocks always advance at 20 Hz.
"""

from __future__ import annotations

import numpy as np

from .contract import (
    ACTION_REPEAT,
    CAMERAS,
    CONTROL_FPS,
    CONTROL_MODE,
    POLICY_FPS,
    ROBOT,
    check_actions,
)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_bool(value) -> bool:
    return bool(as_numpy(value).item())


def policy_metadata() -> dict:
    return {
        "robot": ROBOT,
        "control_mode": CONTROL_MODE,
        "control_fps": CONTROL_FPS,
        "policy_fps": POLICY_FPS,
        "action_repeat": ACTION_REPEAT,
        "action_dim": 13,
        "state_dim": 12,
        "cameras": {name: list(shape) for name, shape in CAMERAS.items()},
    }


def policy_observation(obs: dict, instruction: str) -> dict:
    state = as_numpy(obs["agent"]["qpos"])
    if state.shape != (1, 15) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite qpos (1, 15), got {state.shape}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing task instruction")
    if set(obs["sensor_data"]) != set(CAMERAS):
        raise ValueError("Unexpected policy cameras")
    result = {"observation.state": state[0, 3:].astype(np.float32), "prompt": instruction}
    for name, shape in CAMERAS.items():
        image = as_numpy(obs["sensor_data"][name]["rgb"])
        if image.shape != (1, *shape) or image.dtype != np.uint8:
            raise ValueError(f"Unexpected RGB shape/dtype for {name}")
        result[f"observation.images.{name}"] = image[0]
    return result


def execute_actions(
    env, actions: np.ndarray, *, repeat=ACTION_REPEAT, stop_on_success=False
) -> dict:
    """Execute absolute targets, unchanged, at a fixed number of control ticks.

    Both replay validation and the evaluation client use this function. It never
    clips or averages the recorded targets. Callers decide whether completing a
    task should end a model rollout early or a recording should play to its end.
    """
    actions = np.asarray(actions, dtype=np.float32)
    check_actions(actions)
    if repeat not in (1, ACTION_REPEAT):
        raise ValueError("Only native 20 Hz and two-step 10 Hz targets are supported")
    steps = 0
    total = len(actions) * repeat
    for action in actions:
        for _ in range(repeat):
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            success = scalar_bool(info["success"])
            status = "completed"
            if scalar_bool(truncated):
                status = "truncated"
            elif scalar_bool(terminated) and (
                not success or scalar_bool(info.get("fail", False))
            ):
                status = "terminated"
            elif stop_on_success and success:
                status = "success"
            if status != "completed":
                return dict(
                    obs=obs,
                    info=info,
                    success=success,
                    status=status,
                    control_steps=steps,
                    completed=steps == total,
                )
    return dict(
        obs=obs,
        info=info,
        success=success,
        status=status,
        control_steps=steps,
        completed=True,
    )


def run_policy_episode(
    env,
    policy,
    obs: dict,
    instruction: str,
    *,
    max_policy_steps: int,
    execute_horizon: int = 10,
    stop_on_success: bool = True,
    clip_actions: bool = False,
) -> dict:
    """Run recorded or learned actions through the same client and executor."""
    if max_policy_steps < 1 or execute_horizon < 1:
        raise ValueError("Policy horizon and rollout length must be positive")
    if policy.get_server_metadata().get("mikasa_data") != policy_metadata():
        raise ValueError("Policy server uses a different robot or timing contract")
    requests = steps = clipped = 0
    while steps < max_policy_steps * ACTION_REPEAT:
        response = policy.infer(policy_observation(obs, instruction))
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 13 or not len(actions):
            raise ValueError(f"Expected policy actions (H, 13), got {actions.shape}")
        remaining = (max_policy_steps * ACTION_REPEAT - steps) // ACTION_REPEAT
        actions = actions[: min(remaining, execute_horizon)]
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite policy targets")
        if clip_actions:
            bounded = np.clip(actions, env.action_space.low, env.action_space.high)
            clipped += int(np.any(bounded != actions, axis=1).sum())
            actions = bounded
        result = execute_actions(env, actions, stop_on_success=stop_on_success)
        steps += result["control_steps"]
        requests += 1
        obs = result["obs"]
        if result["status"] != "completed":
            break
    return {
        "success": result["success"],
        "status": result["status"],
        "control_steps": steps,
        "policy_requests": requests,
        "clipped_targets": clipped,
    }
