"""Paper max-pooling memory buffer for the pinned RoboMME implementation.

Adapted from Apache-2.0 RoboMME ecf086c3 shared/mem_buffer.py:add_buffer.
The release stores pool_type but never passes it into pool_tokens_to_size; this
project-owned override passes it explicitly for both training and serving.
"""
from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.shared import image_tools
from mme_vla_suite.shared.data_utils import pool_tokens_to_size
from mme_vla_suite.shared.mem_buffer import MemoryBuffer


class PaperMemoryBuffer(MemoryBuffer):
    def __init__(self, *, pool_type="max", max_steps=4096, **kwargs):
        if pool_type != "max":
            raise ValueError("the selected paper memory path requires max pooling")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        super().__init__(pool_type=pool_type, max_steps=max_steps, **kwargs)
        self._paper_max_steps = max_steps

    def add_buffer(self, images, states, step_idx_list):
        if self.vision_enc is None:
            raise RuntimeError("memory vision encoder was not initialized")
        time, views = images.shape[:2]
        if views != self.num_views or len(step_idx_list) != time or len(states) != time:
            raise ValueError("memory batch/view/state lengths do not agree")
        if len(set(step_idx_list)) != len(step_idx_list):
            raise ValueError("duplicate memory indices within batch")
        for step in step_idx_list:
            if not 0 <= step < self._paper_max_steps:
                raise ValueError("episode memory capacity exceeded; reset the episode")
            if step in self._history_feats:
                raise ValueError(f"memory frame already ingested: {step}")
        pixels = jnp.asarray(images.astype("float32") / 255.0 * 2.0 - 1.0)
        pixels = einops.rearrange(pixels, "t v h w c -> (t v) h w c")
        pixels = image_tools.resize_with_pad(pixels, 224, 224)
        pixels = einops.rearrange(pixels, "(t v) h w c -> t v h w c", t=time, v=views)
        output = self.vision_enc(pixels)
        pooled = {
            label: jax.device_get(pool_tokens_to_size(output, count, pool_type="max"))
            for label, count in (("8x8", 64), ("4x4", 16), ("2x2", 4))
        }
        for row, step in enumerate(step_idx_list):
            features = {"image_pixels": images[row].copy(), "state_emb": states[row]}
            for label in pooled:
                features[f"image_emb_{label}"] = pooled[label][row]
                features[f"pos_emb_{label}"] = self.pos_emb_dict[label][
                    step * self.num_views:(step + 1) * self.num_views]
            self._history_feats[step] = features
            if self.compute_token_drop_score:
                self._process_token_drop_score(step)


class PaperServingPolicy:
    """Delegate native serving, reinstating the same max-pooling path on reset."""
    def __init__(self, policy):
        self._policy = policy
        install_policy_paper_buffer(policy)

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def reset(self):
        result = self._policy.reset()
        install_policy_paper_buffer(self._policy)
        return result


def install_policy_paper_buffer(policy, *, max_steps=4096):
    """Replace only a newly reset, empty native perceptual buffer."""
    if policy.config is None:
        return
    if policy.config.representation_type != "perceptual":
        raise ValueError("paper buffer supports perceptual memory only")
    if policy.mem_buffer._history_feats:
        raise ValueError("cannot replace live memory; reset first")
    config = policy.config
    policy.mem_buffer = PaperMemoryBuffer(
        num_views=config.num_views,
        img_emb_dim=config.memory_feature.img.input_dim,
        pos_emb_dim=config.memory_feature.pos.input_dim,
        state_emb_dim=config.memory_feature.state.input_dim,
        compute_token_drop_score=config.perceptual_memory.type == "token_dropping",
        token_drop_stride=config.streaming_obs_horizon // 2,
        prepare_buffer=True, vision_enc_fn=policy._vision_encode,
        max_steps=max_steps,
    )
