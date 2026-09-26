"""Optional C3 arm-command noise, confined to named transfer phases."""

from contextlib import contextmanager, nullcontext

import numpy as np


class ExecutionNoise:
    def __init__(self, *, seed, action_noise, noise_hold, log):
        if not np.isfinite(action_noise) or action_noise <= 0:
            raise ValueError(
                "action_noise must be a positive finite standard deviation"
            )
        if noise_hold < 1 or int(noise_hold) != noise_hold:
            raise ValueError(
                "noise_hold must be a positive integer number of control steps"
            )
        self.seed = int(seed)
        self.action_noise = float(action_noise)
        self.noise_hold = int(noise_hold)
        self.rng = np.random.default_rng(self.seed)
        self.log = log
        self.stage = None
        self.offset = None
        self.remaining = 0
        self.log(
            "execution_noise_config",
            "Gaussian arm target noise",
            seed=self.seed,
            action_noise=self.action_noise,
            noise_hold=self.noise_hold,
            units="rad",
        )

    @contextmanager
    def phase(self, name):
        if self.stage is not None:
            raise RuntimeError("Execution-noise phases cannot be nested")
        self.stage, self.remaining = str(name), 0
        self.log("execution_noise_begin", self.stage)
        try:
            yield
        finally:
            self.log("execution_noise_end", self.stage)
            self.stage, self.offset, self.remaining = None, None, 0

    def start_path(self, clean_arm, measured_arm, step):
        self.log(
            "execution_noise_path",
            self.stage or "clean recovery/contact path",
            control_step=int(step),
            measured_arm=np.asarray(measured_arm).tolist(),
            first_clean_arm=np.asarray(clean_arm).tolist(),
            noisy=self.stage is not None,
        )

    def apply(self, clean_arm, step, *, hold_steps=1):
        clean = np.asarray(clean_arm, dtype=np.float64).copy()
        if self.stage is None:
            return clean
        if hold_steps < 1 or self.noise_hold % hold_steps:
            raise ValueError("Noise blocks must contain whole source action holds")
        if clean.shape != (7,) or not np.isfinite(clean).all():
            raise ValueError("Expected seven finite arm targets")
        if self.remaining == 0:
            self.offset = self.rng.normal(0, self.action_noise, size=7)
            self.remaining = self.noise_hold
        executed = clean + self.offset
        self.log(
            "execution_noise_step",
            self.stage,
            control_step=int(step),
            clean_arm=clean.tolist(),
            noise=self.offset.tolist(),
            executed_arm=executed.tolist(),
            block_remaining=self.remaining,
            action_hold_steps=hold_steps,
        )
        self.remaining -= hold_steps
        return executed


def configure_execution_noise(planner, env, seed, *, action_noise=0.003, noise_hold=10):
    if not getattr(planner, "supports_execution_noise", False):
        raise TypeError("Collection requires an execution-noise-capable motion planner")
    log = getattr(env, "log_event", lambda *args, **kwargs: None)
    planner.execution_noise = ExecutionNoise(
        seed=seed, action_noise=action_noise, noise_hold=noise_hold, log=log
    )


def transfer_phase(planner, name):
    noise = getattr(planner, "execution_noise", None)
    return nullcontext() if noise is None else noise.phase(name)
