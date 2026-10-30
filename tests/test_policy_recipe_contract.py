"""Contract tests only: these fixtures do not claim model training evidence."""

import pytest

from utils.policies.recipe_contract import (
    CONFIGURATIONS,
    author_recipe,
    load_contract,
    smoke_recipe,
)


def test_six_author_configurations_exclude_memer_and_gr00t_substitution():
    contract = load_contract()
    assert len(CONFIGURATIONS) == 6
    assert contract["configurations"]["hamlet_cogact"]["backbone_source"] == "cogact_base"
    assert "memer" not in contract["configurations"]
    assert not contract["rules"]["lora_in_author_recipes"]


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_author_source_is_pinned_and_smoke_keeps_recipe_unchanged(name):
    contract = load_contract()
    original = author_recipe(name)
    source = contract["sources"][original["source"]]
    assert len(source["revision"]) == 40
    for stage in original["stages"]:
        smoke = smoke_recipe(name, stage=stage, micro_batch=1, context_frames=32)
        assert smoke["smoke_overrides"]["optimizer_steps"] == 3
        assert smoke["author_recipe"] == original
        assert author_recipe(name) == original


def test_robomme_dense_recipe_and_frame_token_budget():
    recipe = author_recipe("robomme_framesamp_modul")
    assert recipe["freeze"] == ["vision encoder"]
    assert "VLM language expert" in recipe["train_dense"]
    assert recipe["schedule"]["warmup_steps"] == 4000
    assert recipe["optimizer"]["weight_decay"] == 0
    assert recipe["ema_decay"] == 0.999
    assert not recipe["paper_proprioception"]
    memory = recipe["memory"]
    assert memory["sampled_frames"] * memory["tokens_per_frame"] == memory["budget"]


def test_hamlet_stages_have_distinct_training_maps():
    stages = author_recipe("hamlet_cogact")["stages"]
    assert "pretrained VLM" in stages["tcl"]["freeze"]
    assert "new moment token embeddings" in stages["tcl"]["train_dense"]
    assert "VLM including moment embeddings" in stages["action_sft"]["train_dense"]
    assert stages["tcl"]["transfer_to_next_stage"] == ["moment token embeddings"]


def test_ponder_keeps_grounding_and_temporal_gradients():
    recipe = author_recipe("ponderpounce")
    assert recipe["loss"]["grounding_cross_entropy"] == 0.1
    assert not recipe["gradient_routing"]["temporal_detach"]
    assert recipe["gradient_routing"]["action_gradient_to_ponder_scale"] == 0.5
    assert recipe["ponder_backbone_freeze_map"] is None  # unresolved, not guessed
    assert recipe["required_labels"]


@pytest.mark.parametrize("kwargs", [
    {"stage": "unknown"}, {"stage": "tcl", "micro_batch": 0},
    {"stage": "tcl", "micro_batch": True}, {"stage": "tcl", "context_frames": 0},
])
def test_invalid_smoke_descriptor_is_rejected(kwargs):
    with pytest.raises(ValueError):
        smoke_recipe("hamlet_cogact", **kwargs)


def test_smoke_api_cannot_override_loss_or_freeze_map():
    with pytest.raises(TypeError):
        smoke_recipe("ponderpounce", stage="joint_episode_sft", grounding_weight=0)
    with pytest.raises(TypeError):
        smoke_recipe("robomme_pi05", stage="action_sft", lora=True)
