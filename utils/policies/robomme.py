"""Fetch training/serving adapter for pinned RoboMME ecf086c3.

Run in an isolated RoboMME environment with its src directory on PYTHONPATH.
All data adapters live here; upstream sources are not modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from utils.policies.data import ACTION_DIM, CAMERAS, FetchDataset, epoch_plan

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "third_party/robomme"
VARIANTS = ("pi05", "framesamp", "tokendrop")
ASSET_ID = "fetch_myorobocasa"


@dataclasses.dataclass(frozen=True)
class Inputs:
    def __call__(self, data):
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        keys = ("observation/image", "observation/hand_image", "observation/side_image")
        result = {"state": np.asarray(data["observation/state"], dtype=np.float32),
                  "image": {n: np.asarray(data[k], dtype=np.uint8) for n, k in zip(names, keys)},
                  "image_mask": {n: np.True_ for n in names}, "prompt": data["prompt"],
                  "simple_subgoal": data.get("simple_subgoal", ""),
                  "grounded_subgoal": data.get("grounded_subgoal", "")}
        if result["state"].shape != (12,):
            raise ValueError("Fetch requires 12D proprioception")
        if "actions" in data:
            result["actions"] = data["actions"]
        for key in ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class Outputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}


def api_config():
    from mme_vla_suite.training import config
    return config


def data_factory(assets_dir):
    cfg = api_config()
    from openpi import transforms

    @dataclasses.dataclass(frozen=True)
    class FetchDataConfig(cfg.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(inputs=[Inputs()], outputs=[Outputs()]),
                model_transforms=cfg.ModelTransformFactory()(model_config),
            )

    return FetchDataConfig(repo_id=ASSET_ID, assets=cfg.AssetsConfig(
        assets_dir=str(assets_dir), asset_id=ASSET_ID))


def configuration(args):
    cfg = api_config()
    from omegaconf import OmegaConf
    from openpi.training import weight_loaders

    recipe = getattr(args, "recipe", "legacy")
    paper = recipe == "paper"
    author = None
    if paper:
        from utils.policies.recipe_contract import author_recipe
        key = "robomme_pi05" if args.variant == "pi05" else f"robomme_{args.variant}_modul"
        author = author_recipe(key)
    base = cfg.get_config("mme_vla_suite")
    history_path = None
    if args.variant != "pi05":
        filename = f"perceptual-{'framesamp' if args.variant == 'framesamp' else 'tokendrop'}-modul.yaml"
        history = OmegaConf.load(BACKEND / "src/mme_vla_suite/models/config/robomme" / filename)
        history.memory_feature.state.input_dim = 12
        if paper:
            history.pool_type = "max"
            history.use_state_emb = False
        directory = args.output / "configs" / ("paper" if paper else "legacy")
        directory.mkdir(parents=True, exist_ok=True)
        history_path = str((directory / filename).resolve())
        OmegaConf.save(history, history_path)
    model = dataclasses.replace(base.model, action_dim=32, action_horizon=20, max_token_len=128,
                                use_history=history_path is not None, history_config=history_path,
                                discrete_state_input=not paper,
                                paligemma_variant="gemma_2b" if paper else "gemma_2b_lora",
                                action_expert_variant="gemma_300m" if paper else "gemma_300m_lora")
    if paper:
        from utils.policies.robomme_paper_model import make_paper_model_config
        model = make_paper_model_config(model, valid_action_dim=13)
    dataset = FetchDataset(args.root)
    plan = epoch_plan(len(dataset), args.batch_size, args.epochs)
    config = dataclasses.replace(
        base, name=f"fetch_{args.variant}" + ("_paper" if paper else ""),
        exp_name=args.exp_name, model=model,
        data=data_factory(args.output / "assets"), dataset_path=str(args.features),
        assets_base_dir=str(args.output / "assets"), checkpoint_base_dir=str(args.output / "checkpoints"),
        batch_size=args.batch_size, num_workers=args.num_workers,
        num_train_steps=args.diagnostic_steps or (author["schedule"]["total_steps"] if paper else plan["total_steps"]),
        fsdp_devices=(getattr(args, "fsdp_devices", None) or
                      (1 if args.diagnostic_steps is not None or not paper else author["fsdp_devices"])),
        freeze_filter=model.get_freeze_filter(), ema_decay=author["ema_decay"] if paper else None,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base_params)),
        wandb_enabled=False, log_interval=10, save_interval=args.save_interval,
        keep_period=None, resume=args.resume,
    )
    if paper:
        from openpi.training import optimizer
        opt = author["optimizer"]
        config = dataclasses.replace(config,
            lr_schedule=optimizer.CosineDecaySchedule(
                warmup_steps=author["schedule"]["warmup_steps"], peak_lr=opt["lr"],
                decay_steps=author["schedule"]["total_steps"], decay_lr=opt["lr"]),
            optimizer=optimizer.AdamW(b1=opt["betas"][0], b2=opt["betas"][1],
                weight_decay=opt["weight_decay"], clip_gradient_norm=opt["max_grad_norm"]))
        plan.update({"author_recipe": author, "recipe": "paper",
                     "run_is_epoch_based": False, "requested_steps": config.num_train_steps})
    return config, dataset, plan


class Samples:
    """The fork's SampleDataset API, backed by immutable v3 video/parquet."""
    def __init__(self, root, horizon=20):
        self.source = FetchDataset(root)
        self.horizon = horizon
        self.memory = None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source.sample(index, self.horizon)
        frames = sample["images"]
        if self.memory is not None:
            self.memory.ingest(sample["episode_index"], sample["frame_index"],
                               frames[CAMERAS[0]], sample["state"])
        return {"observation/state": sample["state"],
                "observation/image": frames[CAMERAS[0]],
                "observation/hand_image": frames[CAMERAS[1]],
                "observation/side_image": frames[CAMERAS[2]],
                "actions": sample["actions"], "prompt": sample["prompt"],
                "epis_idx": np.array([sample["episode_index"]], dtype=np.int32),
                "step_idx": np.array([sample["frame_index"]], dtype=np.int32),
                "exec_start_idx": np.array([0], dtype=np.int32),
                "simple_subgoal": "", "grounded_subgoal": "",
                "simple_subgoal_online": "", "grounded_subgoal_online": ""}


def make_history_dataset(root, features, data_config, history_config, horizon, base_params=None):
    import mme_vla_suite.training.dataset as upstream
    from omegaconf import OmegaConf

    class FetchHistoryDataset(upstream.RoboMMEDataset):
        def __init__(self):
            # Only the constructor needs the fork's pickle SampleDataset replaced.
            original = upstream.SampleDataset
            try:
                def source(_):
                    ds = Samples(root, horizon)
                    ds.dataset_path = str(features)
                    return ds
                upstream.SampleDataset = source
                super().__init__(str(features), data_config, history_config, horizon)
            finally:
                upstream.SampleDataset = original
            if history_config is not None:
                from utils.policies.memory import CausalMemory
                self.memory = CausalMemory(self.dataset.source, history_config, base_params)
                self.mem_buffer = self.memory.buffer
                self.dataset.memory = self.memory

        def _gather_history_feat(self, indices_to_load, epis_idx):
            return self.memory.gather(indices_to_load, epis_idx)

        def prepare_token_drop(self, epis_idx, step_idx):
            # Never use episode-final offline kept_indices: their selection can
            # depend on future frames. The live prefix heap matches inference.
            return self.mem_buffer.prepare_token_dropping(
                step_idx, self.history_config.budget, self._gather_history_feat,
                epis_idx=epis_idx)

    if history_config is not None and isinstance(history_config, str):
        history_config = OmegaConf.load(history_config)
    return FetchHistoryDataset()


def compute_stats(dataset, output):
    from openpi.shared import normalize

    stats = {name: normalize.RunningStats() for name in ("state", "actions")}
    for number, episode in enumerate(dataset.episodes):
        rows = dataset.episode_rows(episode.global_index)
        for name, key in (("state", "observation.state"), ("actions", "action")):
            stats[name].update(np.asarray(rows[key], dtype=np.float32))
        if number % 100 == 0:
            print(f"stats episodes={number + 1}/{len(dataset.episodes)}", flush=True)
    path = output / "assets" / ASSET_ID
    normalize.save(path, {name: value.get_statistics() for name, value in stats.items()})
    return path


def train(args):
    import jax
    import torch
    from openpi.training.data_loader import transform_dataset
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    config, source, plan = configuration(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / "assets" / ASSET_ID / "norm_stats.json").exists():
        compute_stats(source, args.output)
    manifest = {**source.manifest(), "epoch_plan": plan, "variant": args.variant,
                "recipe": getattr(args, "recipe", "legacy"),
                "lora": getattr(args, "recipe", "legacy") != "paper",
                "history_camera": CAMERAS[0], "action_horizon": 20,
                "feature_root": str(args.features.resolve()),
                "history_cache": "one episode in RAM, zero disk feature-cache bytes",
                "episode_order": ("bounded diagnostic: balanced task prefixes, chronological frames"
                                  if args.diagnostic_steps is not None
                                  else "seeded permutation; chronological frames; no dropped anchors"),
                "sampling_seed": config.seed,
                "diagnostic_only": args.diagnostic_steps is not None,
                "requested_optimizer_steps": config.num_train_steps}
    manifest_path = args.output / f"{args.variant}-{args.exp_name}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    module_spec = importlib.util.spec_from_file_location("fetch_robomme_training", BACKEND / "scripts/train.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    class Loader:
        def __init__(self, data_config, sharding, history_config, action_horizon):
            self._data_config = data_config
            from utils.policies.sampling import DiagnosticSampler, EpisodeSampler
            dataset = make_history_dataset(args.root, args.features, data_config, history_config,
                                           action_horizon, args.base_params)
            self.dataset = transform_dataset(dataset, data_config)
            self.sampler = (DiagnosticSampler(dataset.dataset.source,
                                args.diagnostic_steps * config.batch_size)
                            if args.diagnostic_steps is not None
                            else EpisodeSampler(dataset.dataset.source, config.seed))
            # No workers: dynamically generated upstream adapter classes are not
            # spawn-picklable, and forking a live JAX runtime is unsafe.
            self.torch_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=args.batch_size, sampler=self.sampler, drop_last=False,
                num_workers=0, collate_fn=collate,
                generator=torch.Generator().manual_seed(config.seed),
            )
            self.sharding = sharding

        def data_config(self):
            return self._data_config

        def __iter__(self):
            while True:
                for batch in self.torch_loader:
                    batch = jax.device_put(batch, self.sharding)
                    yield HistAugObservation.from_dict(batch), batch["actions"]

    def collate(items):
        return jax.tree.map(lambda *values: np.stack(values), *items)

    def create_loader(dataset_path, data_config, *, sharding=None, history_config=None,
                      action_horizon=20, **unused):
        return Loader(data_config, sharding, history_config, action_horizon)

    module._data_loader.create_data_loader = create_loader
    original_step = module.train_step
    verification = {"optimizer_steps": 0, "trained_anchors": 0,
                    "all_losses_gradients_finite": True, "weights_updated": False,
                    "memory_weights_updated": False if args.variant != "pi05" else None}
    paper = getattr(args, "recipe", "legacy") == "paper"
    if paper:
        from utils.policies.parameter_audit import (
            fingerprint_parameters, compare_fingerprints, robomme_parameter_group)
        initial = {}
        original_init = module.init_train_state
        original_save = module._checkpoints.save_state
        audit_prefix = args.output / f"{args.variant}-{args.exp_name}-parameters"

        def audited_init(*values, **kwargs):
            state, sharding = original_init(*values, **kwargs)
            jax.effects_barrier()
            initial["fingerprint"] = fingerprint_parameters(
                state.params.to_pure_dict(), robomme_parameter_group)
            Path(str(audit_prefix) + "-before.json").write_text(json.dumps(initial["fingerprint"], indent=2))
            return state, sharding

        def audited_save(manager, state, loader, step):
            if int(step) == config.num_train_steps - 1:
                jax.effects_barrier()
                after = fingerprint_parameters(state.params.to_pure_dict(), robomme_parameter_group)
                Path(str(audit_prefix) + "-after.json").write_text(json.dumps(after, indent=2))
                required = ["vlm", "action_expert", "action_adapters"]
                if args.variant != "pi05":
                    required += ["memory_bridge", "memory_attention", "memory_modulation"]
                evidence = compare_fingerprints(initial["fingerprint"], after,
                    required_trainable_groups=required, frozen_groups=["vision"])
                verification["parameter_audit"] = evidence
                verification["weights_updated"] = True
                if args.variant != "pi05":
                    verification["memory_weights_updated"] = True
            return original_save(manager, state, loader, step)

        module.init_train_state = audited_init
        module._checkpoints.save_state = audited_save
    metrics_path = args.output / f"{args.variant}-{args.exp_name}-metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite experiment metrics: {metrics_path}")

    def verify_step(step, loss, grad, update, anchors, memory_update):
        values = [float(loss), float(grad), float(update), float(memory_update)]
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite optimization metrics at step {step}: {values}")
        if int(step) != verification["optimizer_steps"]:
            raise RuntimeError("optimizer step counter is not contiguous")
        verification["optimizer_steps"] += 1
        verification["trained_anchors"] += int(anchors)
        verification["weights_updated"] |= float(update) > 0
        if args.variant != "pi05":
            verification["memory_weights_updated"] |= float(memory_update) > 0
        verification["last_loss"], verification["last_grad_norm"] = values[:2]
        if int(step) % 10 == 0 or int(step) == config.num_train_steps - 1:
            with metrics_path.open("a") as stream:
                record = {"step": int(step), "loss": values[0], "grad_norm": values[1],
                          "trained_anchors": verification["trained_anchors"]}
                if not paper:
                    record.update(update_norm=values[2], memory_update_norm=values[3])
                stream.write(json.dumps(record) + "\n")

    def guarded_step(cfg, rng, state, batch):
        import jax.numpy as jnp
        import optax
        new_state, info, stats = original_step(cfg, rng, state, batch)
        if paper:
            # Dense before/after proof is streamed at init/final save, not an
            # extra model-sized GPU delta allocation inside every train step.
            delta = memory_delta = jnp.asarray(0.0)
        else:
            old = state.params.filter(cfg.trainable_filter)
            new = new_state.params.filter(cfg.trainable_filter)
            changes = jax.tree.map(lambda a, b: b - a, old, new)
            delta = optax.global_norm(changes)
            from openpi.shared import nnx_utils
            memory_delta = optax.global_norm(changes.filter(nnx_utils.PathRegex(".*mem.*")))
            info["update_norm"] = delta
        jax.debug.callback(verify_step, state.step, info["loss"], info["grad_norm"],
                           delta, jnp.asarray(batch[1].shape[0]), memory_delta, ordered=True)
        return new_state, info, stats

    module.train_step = guarded_step
    module.main(config, tentative_run=False)
    jax.effects_barrier()
    if verification["optimizer_steps"] != config.num_train_steps:
        raise RuntimeError("incomplete optimizer-step count")
    if not verification["weights_updated"]:
        raise RuntimeError("no trainable parameter changed")
    if args.variant != "pi05" and not verification["memory_weights_updated"]:
        raise RuntimeError("memory parameters did not change")
    verification["device_memory_stats"] = jax.devices()[0].memory_stats()
    if (getattr(args, "recipe", "legacy") != "paper" and args.diagnostic_steps is None
            and verification["trained_anchors"] != len(source) * args.epochs):
        raise RuntimeError("full-epoch anchor coverage mismatch")
    verification.update({"diagnostic_only": args.diagnostic_steps is not None,
                         "checkpoint_dir": str(config.checkpoint_dir),
                         "epoch_plan": plan})
    (args.output / f"{args.variant}-{args.exp_name}-verification.json").write_text(
        json.dumps(verification, indent=2))


def serve(args):
    from mme_vla_suite.policies.policy_config import create_trained_policy
    from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer

    config, _, _ = configuration(args)
    policy = create_trained_policy(config, args.checkpoint)
    if getattr(args, "recipe", "legacy") == "paper":
        from utils.policies.robomme_paper_memory import PaperServingPolicy
        policy = PaperServingPolicy(policy)
    WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stats", "train", "serve"))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--recipe", choices=("paper", "legacy"), default="paper",
                        help="paper=dense/frozen vision; legacy=old non-paper LoRA adapter")
    parser.add_argument("--fsdp-devices", type=int)
    parser.add_argument("--allow-author-training", action="store_true",
                        help="explicit permission for the full 80000-step author schedule")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--batch-size", type=int,
                        help="smoke microbatch (default1); full paper defaults64/memory or128/released baseline")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--diagnostic-steps", type=int,
                        help="bounded diagnostic ONLY, never reported as a full epoch")
    parser.add_argument("--exp-name", default="one_epoch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = ((128 if args.variant == "pi05" else 64)
                           if args.recipe == "paper" and args.diagnostic_steps is None
                           else (1 if args.recipe == "paper" else 4))
    if not 1 <= args.epochs <= 3 or args.batch_size < 1:
        parser.error("epochs must be 1..3 and batch size positive")
    if args.diagnostic_steps is not None and args.diagnostic_steps < 1:
        parser.error("diagnostic steps must be positive")
    if args.diagnostic_steps is not None and args.exp_name == "one_epoch":
        parser.error("diagnostics require a separate --exp-name")
    if args.recipe == "paper" and args.diagnostic_steps not in (None, 3):
        parser.error("paper smoke requires exactly --diagnostic-steps 3")
    if (args.mode == "train" and args.recipe == "paper"
            and args.diagnostic_steps is None and not args.allow_author_training):
        parser.error("paper training requires --diagnostic-steps 3 or explicit --allow-author-training")
    if args.fsdp_devices is not None and args.fsdp_devices < 1:
        parser.error("fsdp devices must be positive")
    if args.resume:
        parser.error("upstream resume resets optimizer state; use a new experiment instead")
    if args.num_workers != 0:
        parser.error("this exact-pass adapter currently requires --num-workers 0")
    if args.mode == "serve" and args.checkpoint is None:
        parser.error("serve requires --checkpoint")
    args.output = args.output.resolve()
    import os
    if workspace := os.environ.get("POLICY_WORKSPACE"):
        from utils.policies.workspace import contained, require_space
        contained(Path(workspace), args.output)
        contained(Path(workspace), args.features)
        require_space(Path(workspace), 30_000_000_000)
    return args


def main():
    args = parse_args()
    if args.mode == "stats":
        compute_stats(FetchDataset(args.root), args.output)
    elif args.mode == "train":
        train(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
