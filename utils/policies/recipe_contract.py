"""Read versioned author recipes without disguising a smoke run as reproduction.

This module does not build models or launch training. Unknown paper details stay
unknown, and the caller cannot override a freeze-map/loss under the smoke API.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

CONTRACT_PATH = Path(__file__).parent / "recipes" / "author_recipes.json"
CONFIGURATIONS = (
    "robomme_pi05", "robomme_framesamp_modul", "robomme_tokendrop_modul",
    "hamlet_cogact", "memoryvla", "ponderpounce",
)


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported author recipe schema")
    if set(contract["configurations"]) != set(CONFIGURATIONS):
        raise ValueError("author contract must contain exactly six configurations")
    if contract["rules"]["smoke_optimizer_steps_per_mandatory_stage"] != 3:
        raise ValueError("the approved smoke contract requires exactly three updates")
    if contract["rules"]["lora_in_author_recipes"]:
        raise ValueError("LoRA must not silently replace the selected dense recipes")
    return contract


def author_recipe(name: str, *, path: Path = CONTRACT_PATH) -> dict:
    contract = load_contract(path)
    result = copy.deepcopy(contract["configurations"][name])
    inherited = result.pop("inherits", None)
    if inherited:
        namespace, key = inherited.split(".", 1)
        if namespace != "shared":
            raise ValueError(f"unsupported recipe inheritance: {inherited}")
        result = {**copy.deepcopy(contract[namespace][key]), **result}
    result["configuration"] = name
    return result


def smoke_recipe(name: str, *, stage: str, micro_batch: int = 1,
                 context_frames: int | None = None) -> dict:
    """Describe allowed resource overrides; preserve the full author settings."""
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
        raise ValueError("micro_batch must be a positive integer")
    if context_frames is not None and (
        isinstance(context_frames, bool) or not isinstance(context_frames, int) or context_frames < 1
    ):
        raise ValueError("context_frames must be a positive integer or None")
    recipe = author_recipe(name)
    if stage not in recipe["stages"]:
        raise ValueError(f"{name} has no stage {stage}")
    return {
        "mode": "three_step_integration_smoke_not_author_result",
        "author_recipe": recipe,
        "smoke_overrides": {
            "stage": stage, "optimizer_steps": 3, "micro_batch": micro_batch,
            "context_frames": context_frames,
        },
        "must_check_before_execution": [
            "resolved critical source/recipe decisions and correct mandatory labels",
            "resources and workspace budget", "both dataset tasks",
            "stage-specific frozen/trainable groups and gradient routing",
        ],
    }
