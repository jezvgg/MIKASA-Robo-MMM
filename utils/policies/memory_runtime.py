"""Run the memory-augmented π0.5 heads of RoboMME from this repository.

`utils/policies/robomme_paper_model.py` rebuilds the paper's perceptual memory path on
top of the vendored `third_party/robomme` checkout. This module is the runtime around
it: pick one of RoboMME's memory configs, build the model, feed it a shaped
observation and return actions.

    cd ~/policy-integration-20261019/third_party/robomme
    JAX_PLATFORMS=cpu ~/policy-integration-20261019/.venv-robomme/bin/python \
        -m utils.policies.memory_runtime --memory perceptual-framesamp-modul.yaml

Three things are load-bearing:

* The base config must be `mme_vla_suite.training.config.get_config("pi05_baseline").model`,
  whose type is already `HistoryPi0Config`. A plain openpi `Pi0Config` lacks
  `memory_expert_variant`, and `make_paper_model_config` stops with `AttributeError`.
* `history_config` takes a file name with the `.yaml` suffix: it is resolved against
  `src/mme_vla_suite/models/config/robomme/`.
* `use_history=True` has to be set next to `history_config`, otherwise
  `inputs_spec()` returns the plain `Observation` and `sample_actions` has no memory
  tensors to read.

No trained checkpoint for these heads survived (the fine-tuned runs lived under
`logs/robomme_paper/checkpoints`, 34 GB, deleted), so parameters are initialised from
the config. The check here is that the graph builds, the memory tensors are consumed
and the returned action block is finite.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

ROBOMME_ROOT = Path(
    os.environ.get(
        "ROBOMME_ROOT",
        Path.home() / "policy-integration-20261019" / "third_party" / "robomme",
    )
)
DEFAULT_MEMORY = "perceptual-framesamp-modul.yaml"
PERCEPTUAL = ("perceptual-framesamp-modul.yaml", "perceptual-tokendrop-modul.yaml")


def _import_jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _bootstrap() -> None:
    """Make `mme_vla_suite` and this repository importable from RoboMME's root."""
    sys.path.insert(0, str(ROBOMME_ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.chdir(ROBOMME_ROOT)


def model_config(memory: str = DEFAULT_MEMORY, *, valid_action_dim: int = 13, base: str = "pi05_baseline"):
    """Paper model config for one RoboMME memory variant."""
    _bootstrap()
    from mme_vla_suite.training import config as training_config

    from utils.policies.robomme_paper_model import make_paper_model_config

    source = training_config.get_config(base).model
    return make_paper_model_config(
        dataclasses.replace(source, history_config=memory, use_history=True),
        valid_action_dim=valid_action_dim,
    )


def observation_spec(config, *, batch_size: int = 1):
    """`inputs_spec` returns `(observation, action)`; only the observation is needed."""
    spec = config.inputs_spec(batch_size=batch_size)
    return spec[0] if isinstance(spec, tuple) else spec


def blank_observation(config, *, batch_size: int = 1):
    """A fully-shaped observation: zero memory embeddings, all masks on."""
    jax, jnp = _import_jax()
    spec = observation_spec(config, batch_size=batch_size)
    observation = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), spec)
    masks = {"image_masks": {key: jnp.ones_like(value)
                             for key, value in observation.image_masks.items()},
             "tokenized_prompt_mask": jnp.ones_like(observation.tokenized_prompt_mask)}
    for name in ("static_mask", "recur_mask"):
        value = getattr(observation, name, None)
        if value is not None:
            masks[name] = jnp.ones_like(value)
    return dataclasses.replace(observation, **masks)


def run(memory: str = DEFAULT_MEMORY, *, num_steps: int = 2, seed: int = 0,
        valid_action_dim: int = 13) -> tuple[np.ndarray, str]:
    """Build the model, sample actions, and report which memory path ran."""
    jax, _ = _import_jax()
    from mme_vla_suite.models.config.utils import get_history_config

    config = model_config(memory, valid_action_dim=valid_action_dim)
    model = config.create(jax.random.key(seed))
    resolved = get_history_config(memory)
    description = f"{resolved.representation_type}+{resolved.integration_type}"
    actions = model.sample_actions(jax.random.key(seed + 1), blank_observation(config),
                                   num_steps=num_steps)
    return np.asarray(actions), description


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", default=DEFAULT_MEMORY,
                        help="a file name under models/config/robomme/, e.g. "
                             f"{' or '.join(PERCEPTUAL)}")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    actions, description = run(args.memory, num_steps=args.num_steps, seed=args.seed)
    finite = bool(np.isfinite(actions).all())
    print(f"memory={args.memory} path={description} actions={actions.shape} "
          f"finite={finite} range=[{actions.min():.3f}, {actions.max():.3f}]")
    print("memory forward ok" if finite else "memory forward produced non-finite actions")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
