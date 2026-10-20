import numpy as np
import pytest

from utils.policies.parameter_audit import (
    compare_fingerprints,
    fingerprint_parameters,
    robomme_parameter_group,
)


def snap(tree):
    return fingerprint_parameters(tree, lambda path: path[0])


def test_exact_change_detected():
    before = snap({"train": {"w": np.array([1.0, 2.0])}})
    after = snap({"train": {"w": np.array([1.0, 3.0])}})
    result = compare_fingerprints(before, after, required_trainable_groups={"train"}, frozen_groups=set())
    assert result["changed_leaves"] == ["train/w"]
    assert result["groups"]["train"] == {"leaves": 1, "changed": 1, "preserved": 0}


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="non-finite.*train/w"):
        snap({"train": {"w": np.array([np.nan])}})


def test_frozen_mutation_rejected():
    before = snap({"frozen": {"w": np.array([1])}})
    after = snap({"frozen": {"w": np.array([2])}})
    with pytest.raises(ValueError, match="frozen group frozen changed at frozen/w"):
        compare_fingerprints(before, after, required_trainable_groups=set(), frozen_groups={"frozen"})


def test_missing_required_group_rejected():
    value = snap({"other": {"w": np.array([1])}})
    with pytest.raises(ValueError, match="required trainable groups absent"):
        compare_fingerprints(value, value, required_trainable_groups={"train"}, frozen_groups=set())


def test_shape_mismatch_rejected():
    before = snap({"train": {"w": np.array([1])}})
    after = snap({"train": {"w": np.array([1, 2])}})
    with pytest.raises(ValueError, match="shape mismatch at train/w"):
        compare_fingerprints(before, after, required_trainable_groups=set(), frozen_groups=set())


def test_zero_update_rejected():
    value = snap({"train": {"w": np.array([1])}})
    with pytest.raises(ValueError, match="zero changed leaves"):
        compare_fingerprints(value, value, required_trainable_groups={"train"}, frozen_groups=set())


def test_robomme_classifier_and_lora_rejection():
    assert robomme_parameter_group(("PaliGemma", "img", "kernel")) == "vision"
    assert robomme_parameter_group(("mem_encoder", "kernel")) == "memory_bridge"
    assert robomme_parameter_group(("PaliGemma", "llm", "layers_1", "w")) == "action_expert"
    assert robomme_parameter_group(("PaliGemma", "llm", "embedder", "w")) == "vlm"
    assert robomme_parameter_group(("PaliGemma", "llm", "mem_rms_norm_ffn_1", "kernel")) == "memory_modulation"
    assert robomme_parameter_group(("PaliGemma", "llm", "mem_attn_ffn_1", "kernel")) == "memory_attention"
    assert robomme_parameter_group(("action_out_proj", "kernel")) == "action_adapters"
    assert robomme_parameter_group(("time_mlp_in", "bias")) == "action_adapters"
    with pytest.raises(ValueError, match="LoRA path"):
        robomme_parameter_group(("llm", "lora_a"))


def test_bfloat16_nan_is_not_mistaken_for_opaque_void_data():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    with pytest.raises(ValueError, match="non-finite"):
        snap({"train": {"w": np.asarray([float("nan")], dtype=ml_dtypes.bfloat16)}})
    finite = snap({"train": {"w": np.asarray([1.0], dtype=ml_dtypes.bfloat16)}})
    assert finite["leaves"]["train/w"]["dtype"] == "bfloat16"


def test_absent_frozen_group_cannot_count_as_preserved():
    value = snap({"train": {"w": np.array([1])}})
    with pytest.raises(ValueError, match="required frozen groups absent"):
        compare_fingerprints(value, value, required_trainable_groups=set(), frozen_groups={"vision"})
