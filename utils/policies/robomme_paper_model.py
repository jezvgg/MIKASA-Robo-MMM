"""Fetch-specific paper path for RoboMME, without editing its source checkout.

The flow-matching forward below is adapted from RoboMME ecf086c3, Apache-2.0,
models/integration/history_pi0.py:557-638. Its attention, corruption distribution
and gradient paths are retained. The only loss change masks the 19 padded,
non-physical channels while keeping the pretrained 32-wide input/output layers.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import threading

import flax.linen as linen
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.integration import history_gemma
from mme_vla_suite.models.integration.history_pi0 import (
    HistoryPi0,
    HistoryPi0Config,
    make_attn_mask,
    preprocess_observation,
)
from mme_vla_suite.models.config.utils import get_history_config

_INIT_LOCK = threading.RLock()


def masked_action_mse(prediction, target, valid_action_dim: int = 13):
    """Native (batch,horizon) MSE, normalized by actual valid action width."""
    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("prediction/target must have equal non-scalar shapes")
    if not 0 < valid_action_dim <= prediction.shape[-1]:
        raise ValueError("valid action width must fit the padded native width")
    return jnp.mean(jnp.square(prediction[..., :valid_action_dim]
                               - target[..., :valid_action_dim]), axis=-1)


class PaperMemoryRMSNorm(linen.Module):
    """Native parameter names/math, zero-initialized conditional affine map.

    This changes ONLY MemoryRMSNorm's conditional Dense initializer; memory
    attention's output projection retains the pinned native initialization.
    Zero modulation means ordinary RMS normalization, not x without RMSNorm.
    """
    @linen.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normalized = x * jax.lax.rsqrt(variance + 1e-6)
        if cond is None:
            scale = self.param("scale", linen.initializers.zeros_init(), (x.shape[-1],))
            return (normalized * (1 + scale)).astype(dtype)
        modulation = linen.Dense(
            x.shape[-1] * 2, kernel_init=linen.initializers.zeros_init(),
            bias_init=linen.initializers.zeros_init(), dtype=dtype,
        )(cond)
        scale, shift = jnp.split(modulation, 2, axis=-1)
        return (normalized * (1 + scale) + shift).astype(dtype)


@contextmanager
def paper_modulation_initialization():
    """A scoped initializer substitution; no disk/global permanent patch.

    Native apply uses identical scope names and mathematics. Its initializer is
    consulted only for parameter creation, so restoring the native class after
    model creation does not change the initialized values or subsequent math.
    """
    with _INIT_LOCK:
        original = history_gemma.MemoryRMSNorm
        history_gemma.MemoryRMSNorm = PaperMemoryRMSNorm
        try:
            yield
        finally:
            history_gemma.MemoryRMSNorm = original


class FetchPaperHistoryPi0(HistoryPi0):
    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        self.fetch_valid_action_dim = config.valid_action_dim
        if self.integration_type not in (None, "modulation"):
            raise ValueError(f"unsupported paper Fetch integration: {self.integration_type}")

    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        expanded = time[..., None, None]
        noisy_actions = expanded * noise + (1 - expanded) * actions
        velocity_target = noise - actions
        prefix, prefix_mask, prefix_ar, prefix_na, stats = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, suffix_na, time_condition = self.embed_suffix(
            observation, noisy_actions, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        if self.use_history and self.representation_type != "symbolic":
            na_mask = jnp.concatenate([prefix_na, suffix_na], axis=0)
            attention_mask = make_attn_mask(input_mask, ar_mask, na_mask)
        else:
            attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        if self.integration_type == "modulation":
            memory, memory_mask, _, _, stats = self.embed_memory(observation)
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix, suffix], mask=attention_mask, positions=positions,
                adarms_cond=[None, time_condition], mem_seq=[None, memory],
                mem_mask=[None, memory_mask])
        elif self.integration_type is None:
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix, suffix], mask=attention_mask, positions=positions,
                adarms_cond=[None, time_condition])
        else:
            raise ValueError(f"unsupported integration: {self.integration_type}")
        prediction = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return masked_action_mse(prediction, velocity_target, self.fetch_valid_action_dim), stats


@dataclasses.dataclass(frozen=True)
class FetchPaperHistoryPi0Config(HistoryPi0Config):
    valid_action_dim: int = 13

    def create(self, rng):
        loaded = self
        if self.history_config is not None:
            history = get_history_config(self.history_config)
            if history.representation_type != "perceptual" or history.integration_type != "modulation":
                raise ValueError("paper Fetch memory supports perceptual+Modul only")
            loaded = dataclasses.replace(self, history_config=history)
        with paper_modulation_initialization():
            return FetchPaperHistoryPi0(loaded, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        loaded = self
        if self.history_config is not None:
            loaded = dataclasses.replace(self, history_config=get_history_config(self.history_config))
        return HistoryPi0Config.inputs_spec(loaded, batch_size=batch_size)


def make_paper_model_config(base_model_config, *, valid_action_dim=13):
    if valid_action_dim != 13 or base_model_config.action_dim != 32:
        raise ValueError("Fetch paper path requires 13 valid channels in native width32")
    if "lora" in base_model_config.paligemma_variant or "lora" in base_model_config.action_expert_variant:
        raise ValueError("LoRA is a separate alternative, not the selected author dense recipe")
    fields = {field.name: getattr(base_model_config, field.name)
              for field in dataclasses.fields(HistoryPi0Config)}
    return FetchPaperHistoryPi0Config(**fields, valid_action_dim=valid_action_dim)
