"""Causal, episode-bounded RoboMME memory. No per-frame files are written."""
from __future__ import annotations

import numpy as np


def frozen_vision(base_params):
    """Use the identical frozen pretrained SigLIP subtree, not random features."""
    import flax.nnx as nnx
    import flax.nnx.bridge as bridge
    import jax
    import jax.numpy as jnp
    from openpi.models import model, siglip

    params = model.restore_params(str(base_params), restore_type=np.ndarray, dtype=None)
    image_params = params["PaliGemma"]["img"]
    image = bridge.ToNNX(siglip.Module(num_classes=2048, variant="So400m/14",
                                     pool_type="none", scan=True, dtype_mm="bfloat16"))
    image.lazy_init(jnp.ones((1, 224, 224, 3)), train=False, rngs=nnx.Rngs(2))
    graph, state = nnx.split(image)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, image_params))
    image = nnx.merge(graph, state)
    del params, image_params

    @jax.jit
    def encode(images):
        t, v = images.shape[:2]
        tokens, _ = image(images.reshape(t * v, 224, 224, 3), train=False)
        return tokens.reshape(t, v, tokens.shape[-2], tokens.shape[-1])
    return encode


class CausalMemory:
    """One episode in RAM; reset on every task/episode transition.

    Inputs must be chronological, with every exported 10-Hz frame ingested once.
    Features and TokenDrop scores are generated only through the current frame.
    At most one episode is retained; disk feature-cache usage is zero.
    """
    def __init__(self, source, config, base_params, *, max_ram_bytes=8_000_000_000):
        from mme_vla_suite.shared.mem_buffer import MemoryBuffer

        max_steps = max(episode.length for episode in source.episodes)
        # Conservative estimate: BF16 features, FP32 positional arrays, uint8
        # base-camera pixels and a factor of two for transient encoding arrays.
        estimated = max_steps * (84 * 2048 * 2 + 84 * 768 * 4 + 256 * 256 * 3 + 48) * 2
        if estimated > max_ram_bytes:
            raise MemoryError(f"episode memory estimate {estimated} exceeds {max_ram_bytes}")
        buffer_type = MemoryBuffer
        if getattr(config, "pool_type", "mean") == "max":
            from utils.policies.robomme_paper_memory import PaperMemoryBuffer
            buffer_type = PaperMemoryBuffer
        self.buffer = buffer_type(
            num_views=1, img_emb_dim=2048, pos_emb_dim=768, state_emb_dim=12,
            max_steps=max_steps, prepare_buffer=True,
            vision_enc_fn=frozen_vision(base_params),
            compute_token_drop_score=config.perceptual_memory.type == "token_dropping",
            token_drop_stride=config.streaming_obs_horizon // 2,
        )
        self.episode = None
        self.frame = -1
        self.ingested = 0
        self.resets = 0
        self.estimated_max_ram_bytes = estimated

    def ingest(self, episode: int, frame: int, image: np.ndarray, state: np.ndarray):
        if episode != self.episode or (frame == 0 and self.frame != -1):
            self.buffer.clear()
            self.episode, self.frame = episode, -1
            self.resets += 1
        if frame != self.frame + 1:
            raise ValueError(f"non-causal memory access: episode={episode}, frame={frame}, previous={self.frame}")
        self.buffer.add_buffer(image[None, None], state[None], [frame])
        self.frame = frame
        self.ingested += 1

    def gather(self, indices, epis_idx):
        if epis_idx != self.episode:
            raise ValueError("memory crossed episode boundary")
        if any(index < 0 or index > self.frame for index in indices):
            raise ValueError("memory attempted future/past-episode access")
        return {index: self.buffer.get_history_feats(index) for index in indices}
