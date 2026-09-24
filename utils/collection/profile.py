"""Pinned task configuration and simulator provenance."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import textwrap
from pathlib import Path

from .contract import ACTION_NAMES, CAMERAS, read_json

REPO = Path(__file__).resolve().parents[2]
PROFILE = Path(__file__).with_name("cabinet_search_profile.json")
TASKS = {
    "cabinet_search": ("MikasaCabinetSearch-v0", "my_scenes.cabinet_search", "CabinetSearchTask"),
    "season_dish": ("MikasaSeasonDish-v0", "my_scenes.season_dish", "SeasonDishTask"),
    "depth_recall": ("MikasaDepthRecall-v1", "my_scenes.depth_recall_v1", "DepthRecallV1Task"),
}


def profile_name(name):
    for key, (env_id, _, _) in TASKS.items():
        if name in (key, env_id):
            return key
    raise ValueError(f"Unknown collection profile: {name}")


def json_value(value):
    return json.loads(json.dumps(value, default=lambda v: sorted(v) if isinstance(v, (set, frozenset)) else v.tolist()))


def class_methods_sha(path):
    source = Path(path).read_text()
    lines = [line.rstrip() for line in source.splitlines()]
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSFetch")
    methods = []
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            methods.append(textwrap.dedent("\n".join(lines[start:node.end_lineno])))
    return hashlib.sha256("\n".join(methods).encode()).hexdigest()


def load_profile(name="cabinet_search"):
    profile = read_json(PROFILE.with_name(profile_name(name) + "_profile.json"))
    overrides = {k: v for k, v in os.environ.items() if k.startswith("MIKASA_")}
    if overrides:
        raise ValueError(f"Pinned collection forbids implicit MIKASA overrides: {sorted(overrides)}")
    for name, expected in profile["runtime_packages"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Runtime requires {name}=={expected}, found {actual}")
    reference = read_json(REPO / "robots/fetch/reference.json")
    if reference["commit"] != profile["robot_reference"]["commit"]:
        raise ValueError("Robot provenance does not match collection profile")
    for relative, expected in reference["assets_sha256"].items():
        path = REPO / "robots/fetch" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference robot asset changed: {relative}")
    source = REPO / "robots/fetch/ds_fetch.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference["source_file_sha256"]:
        raise ValueError("DSFetch source differs from the pinned master reference")
    if class_methods_sha(REPO / "robots/fetch/ds_fetch.py") != reference["methods_source_sha256"]:
        raise ValueError("DSFetch methods differ from the original reference")
    return profile


def task_config(profile):
    import my_scenes  # noqa: F401
    _, module, cls = TASKS[profile_name(profile["env_id"])]
    cfg = getattr(importlib.import_module(module), cls).cfg
    cfg.validate()
    for key, expected in profile["task"].items():
        actual = len(cfg.compartments) if key == "num_compartments" else getattr(cfg, key)
        if json_value(actual) != expected:
            raise ValueError(f"Task parameter {key}: {actual!r} != {expected!r}")
    return json_value(dataclasses.asdict(cfg))


def runtime_signature(profile=None):
    profile = load_profile() if profile is None else profile
    files = {}
    for folder in ("my_scenes", "planners", "robots/fetch", "utils"):
        for path in sorted((REPO / folder).rglob("*")):
            if path.name.startswith("test_") or path.suffix not in {".py", ".urdf", ".srdf", ".json"}:
                continue
            if "collection" in path.parts and path.name not in {
                "__init__.py", "pipeline.py", "contract.py", "client.py", "profile.py",
                "cabinet_search_profile.json", "season_dish_profile.json", "depth_recall_profile.json",
            }:
                continue
            files[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Pin installed engine contents as well as its version: an editable old fork
    # carrying the same distribution version is not the same simulator.
    import mani_skill
    engine = Path(mani_skill.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(engine.rglob("*.py")):
        digest.update(str(path.relative_to(engine)).encode())
        digest.update(path.read_bytes())
    return {
        "code_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "source_files": files,
        "engine_sha256": digest.hexdigest(),
        "packages": profile["runtime_packages"],
        "task_config": task_config(profile),
        "profile": profile,
    }


def verify_env(env, profile):
    from robots.fetch.ds_fetch import DSFetch
    from mani_skill.agents.registration import REGISTERED_AGENTS

    base = env.unwrapped
    if type(base.agent) is not DSFetch or REGISTERED_AGENTS["ds_fetch"].agent_cls is not DSFetch:
        raise ValueError("Another implementation occupies the ds_fetch registration")
    if base.control_mode != "pd_joint_pos" or env.action_space.shape != (13,):
        raise ValueError("Expected reference 13D pd_joint_pos controller")
    if abs(float(base.control_timestep) - 0.05) > 1e-9:
        raise ValueError("Simulation control must run at 20 Hz")
    joints = [j.name for j in base.agent.robot.get_active_joints()]
    if len(joints) != 15 or joints[:3] != ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]:
        raise ValueError("Proprio/debug split no longer matches joint order")
    controllers = base.agent.controller.controllers
    ordered = list(controllers["arm"].config.joint_names) + ["gripper"]
    ordered += list(controllers["body"].config.joint_names)
    ordered += ["base_forward_velocity", "base_yaw_velocity"]
    if ordered != ACTION_NAMES:
        raise ValueError("Action channel order changed")
    sensors = {cfg.uid: cfg for cfg in base.agent._sensor_configs}
    if set(sensors) != set(CAMERAS) or base._default_sensor_configs:
        raise ValueError("The policy must use exactly the three reference cameras")
    for name, expected in profile["cameras"].items():
        actual = sensors[name]
        for key, value in expected.items():
            if abs(float(getattr(actual, key)) - value) > 1e-6:
                raise ValueError(f"Camera {name} {key} differs from the reference")
    return joints



def validate_instructions(profile, tokenizer):
    """Check full language strings with the supplied PaliGemma tokenizer."""
    if tokenizer is None:
        if profile["env_id"] in {"MikasaSeasonDish-v0", "MikasaDepthRecall-v1"}:
            raise ValueError("This task requires --tokenizer (PaliGemma SentencePiece)")
        return None  # Existing CabinetSearch runs remain resumable.
    import sentencepiece as spm
    _, module, _ = TASKS[profile_name(profile["env_id"])]
    task = importlib.import_module(module)
    texts = set(task.INSTRUCTIONS) | set(getattr(task, "INSTRUCTIONS_NUDGE", ()))
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    counts = {text: len(sp.encode(text.strip() + "\n", add_bos=True)) for text in sorted(texts)}
    if max(counts.values()) > profile["data"]["max_instruction_tokens"]:
        raise ValueError(f"Instruction exceeds the PaliGemma budget: {counts}")
    return {"tokenizer_sha256": hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest(),
            "sentencepiece_version": spm.__version__,
            "encoding": "stripped instruction + newline, BOS included, no EOS",
            "instruction_tokens": counts}


def make_env(run, *, rgb=False, render_backend=None):
    import gymnasium as gym
    import my_scenes  # noqa: F401

    profile = run["signature"]["profile"]
    kwargs = dict(profile["env_kwargs"], obs_mode="rgb" if rgb else "state")
    if render_backend is not None and render_backend != kwargs["render_backend"]:
        raise ValueError("Rendering backend differs from the pinned profile")
    env = gym.make(profile["env_id"], **kwargs)
    try:
        verify_env(env, profile)
    except Exception:
        env.close()
        raise
    return env
