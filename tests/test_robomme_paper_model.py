"""Tiny-array checks; run only in the pinned backend environment on gangway."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mme_vla_suite")
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from mme_vla_suite.models.integration import history_gemma
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from utils.policies.robomme_paper_model import (
    FetchPaperHistoryPi0, FetchPaperHistoryPi0Config, PaperMemoryRMSNorm,
    make_paper_model_config, masked_action_mse, paper_modulation_initialization,
)
from utils.policies.robomme_paper_memory import PaperMemoryBuffer


def base_config(**overrides):
    options = dict(pi05=True, action_dim=32, action_horizon=20,
                   discrete_state_input=False, paligemma_variant="gemma_2b",
                   action_expert_variant="gemma_300m")
    options.update(overrides)
    return HistoryPi0Config(**options)


def test_valid_action_loss_ignores_phantom_channels():
    prediction = jnp.ones((1, 20, 32)).at[..., 13:].set(jnp.nan)
    loss = masked_action_mse(prediction, jnp.zeros((1, 20, 32)))
    np.testing.assert_allclose(loss, 1)
    assert loss.shape == (1, 20)


def test_valid_action_gradient_does_not_train_phantom_joints():
    target = jnp.zeros((1, 2, 32))
    gradient = jax.grad(lambda value: masked_action_mse(value, target).sum())(jnp.ones_like(target))
    np.testing.assert_allclose(gradient[..., :13], 2 / 13, rtol=1e-6)
    np.testing.assert_array_equal(gradient[..., 13:], 0)


@pytest.mark.parametrize("width", [0, -1, 33])
def test_invalid_loss_width(width):
    with pytest.raises(ValueError):
        masked_action_mse(jnp.zeros((1, 32)), jnp.zeros((1, 32)), width)


def test_initial_memory_affine_is_identity_on_rms_normalized_input():
    module = PaperMemoryRMSNorm()
    x = jnp.arange(1, 25, dtype=jnp.float32).reshape(1, 3, 8)
    condition = jnp.ones((1, 3, 12))
    variables = module.init(jax.random.PRNGKey(7), x, condition)
    np.testing.assert_array_equal(variables["params"]["Dense_0"]["kernel"], 0)
    np.testing.assert_array_equal(variables["params"]["Dense_0"]["bias"], 0)
    expected = x / jnp.sqrt(jnp.square(x).mean(axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(module.apply(variables, x, condition), expected, rtol=1e-6)


def test_initializer_scope_is_restored_even_on_error():
    original = history_gemma.MemoryRMSNorm
    with pytest.raises(RuntimeError):
        with paper_modulation_initialization():
            assert history_gemma.MemoryRMSNorm is PaperMemoryRMSNorm
            raise RuntimeError("intentional test failure")
    assert history_gemma.MemoryRMSNorm is original


def test_config_preserves_pretrained_width_and_history_yaml(tmp_path):
    original_path = Path(history_gemma.__file__).parent.parent / "config/robomme/perceptual-framesamp-modul.yaml"
    history = OmegaConf.load(original_path)
    history.memory_feature.state.input_dim = 12
    history.pool_type = "max"
    path = tmp_path / "history.yaml"
    OmegaConf.save(history, path)
    config = make_paper_model_config(base_config(use_history=True, history_config=str(path)))
    assert isinstance(config, FetchPaperHistoryPi0Config)
    assert config.action_dim == 32 and config.valid_action_dim == 13
    assert not config.discrete_state_input
    observation, action = config.inputs_spec(batch_size=1)
    assert action.shape == (1, 20, 32)
    assert observation.static_image_emb.shape == (1, 512, 2048)
    assert observation.static_state_emb.shape == (1, 512, 12)
    assert config.history_config == str(path)  # no mutable config side effect


def test_paper_config_rejects_accidental_lora():
    with pytest.raises(ValueError, match="LoRA"):
        make_paper_model_config(base_config(paligemma_variant="gemma_2b_lora"))


def stub_model(memory_weight=None):
    batch, horizon, width = 1, 3, 4
    seen = {}
    def prefix(observation):
        return jnp.zeros((batch, 2, width)), jnp.ones((batch, 2), bool), jnp.zeros(2, bool), jnp.zeros(2, bool), {}
    def suffix(observation, noisy, time):
        seen.update(noisy=noisy, time=time)
        return jnp.zeros((batch, horizon, width)), jnp.ones((batch, horizon), bool), jnp.zeros(horizon, bool), jnp.zeros(horizon, bool), time[:, None]
    def llm(tokens, **kwargs):
        seen["attention"] = kwargs
        output = tokens[1]
        if memory_weight is not None:
            output = output + kwargs["mem_seq"][1].mean()
        return (tokens[0], output), None
    model = SimpleNamespace(use_history=memory_weight is not None,
        representation_type="perceptual", integration_type="modulation" if memory_weight is not None else None,
        action_horizon=horizon, fetch_valid_action_dim=13, embed_prefix=prefix, embed_suffix=suffix,
        PaliGemma=SimpleNamespace(llm=llm),
        action_out_proj=lambda value: jnp.repeat(value[..., :1], 32, axis=-1),
        embed_memory=lambda observation: (jnp.ones((batch, 2, width)) * memory_weight,
            jnp.ones((batch, 2), bool), None, None, {}))
    return model, seen


def test_native_flow_corruption_distribution_is_preserved(monkeypatch):
    import utils.policies.robomme_paper_model as module
    monkeypatch.setattr(module, "preprocess_observation", lambda rng, observation, train: observation)
    model, seen = stub_model()
    rng = jax.random.PRNGKey(11)
    actions = jnp.ones((1, 3, 32)).at[..., 13:].set(0)
    loss, _ = FetchPaperHistoryPi0.compute_loss(model, rng, None, actions, train=True)
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, (1,)) * .999 + .001
    expected_noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    np.testing.assert_allclose(seen["noisy"], expected_noisy, rtol=1e-6)
    np.testing.assert_allclose(loss, jnp.square(noise[..., :13] - actions[..., :13]).mean(-1), rtol=1e-6)


def test_memory_gradient_reaches_trainable_encoder(monkeypatch):
    import utils.policies.robomme_paper_model as module
    monkeypatch.setattr(module, "preprocess_observation", lambda rng, observation, train: observation)
    def loss(weight):
        model, _ = stub_model(weight)
        values, _ = FetchPaperHistoryPi0.compute_loss(model, jax.random.PRNGKey(11),
            None, jnp.ones((1, 3, 32)), train=True)
        return values.mean()
    gradient = jax.grad(loss)(jnp.asarray(2.0))
    assert jnp.isfinite(gradient) and abs(float(gradient)) > 0


def test_max_pooling_is_actually_used_and_bounded():
    buffer = PaperMemoryBuffer(num_views=1, img_emb_dim=1, pos_emb_dim=1,
        state_emb_dim=12, max_steps=4, prepare_buffer=False)
    buffer.vision_enc = lambda images: jnp.arange(256, dtype=jnp.float32).reshape(1, 1, 256, 1)
    buffer.pos_emb_dict = {label: np.zeros((4, count, 1), np.float32)
        for label, count in (("8x8", 64), ("4x4", 16), ("2x2", 4))}
    images, states = np.zeros((1, 1, 8, 8, 3), np.uint8), np.zeros((1, 12), np.float32)
    buffer.add_buffer(images, states, [0])
    features = buffer._history_feats[0]
    for label, expected in (("8x8", 17), ("4x4", 51), ("2x2", 119)):
        assert features[f"image_emb_{label}"][0, 0, 0] == expected
    with pytest.raises(ValueError, match="already ingested"):
        buffer.add_buffer(images, states, [0])
    with pytest.raises(ValueError, match="capacity"):
        buffer.add_buffer(images, states, [4])
