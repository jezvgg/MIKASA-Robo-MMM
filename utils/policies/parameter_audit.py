"""Exact, bounded-memory parameter snapshots for paper-only audits."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Callable


def fingerprint_parameters(parameters: Mapping, classify: Callable[[tuple[str, ...]], str]) -> dict:
    """Fingerprint each parameter leaf in turn; parameter arrays are never retained."""
    import numpy as np

    leaves = {}
    groups = {}
    for path, value in _walk(parameters):
        group = classify(path)
        if not isinstance(group, str):
            raise TypeError(f"classifier returned non-string group at {_path_text(path)}")
        array = np.asarray(value)
        try:
            finite = bool(np.isfinite(array).all())
        except TypeError as error:
            raise TypeError(f"unsupported nonnumeric parameter at {_path_text(path)}: {array.dtype}") from error
        if not finite:
            raise ValueError(f"non-finite numeric parameter at {_path_text(path)} (group {group})")
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(repr((tuple(array.shape), str(array.dtype))).encode("utf-8"))
        digest.update(memoryview(contiguous.view(np.uint8)).cast("B"))
        key = _path_text(path)
        if key in leaves:
            raise ValueError(f"duplicate parameter path {key}")
        leaves[key] = {"group": group, "shape": tuple(array.shape), "dtype": str(array.dtype),
                       "numel": int(array.size), "sha256": digest.hexdigest()}
        groups.setdefault(group, []).append((key, digest.hexdigest()))
        del contiguous, array

    group_hashes = {}
    group_counts = {}
    for group in sorted(groups):
        h = hashlib.sha256()
        entries = sorted(groups[group])
        for key, leaf_hash in entries:
            h.update(key.encode("utf-8")); h.update(b"\0"); h.update(leaf_hash.encode("ascii")); h.update(b"\n")
        group_hashes[group] = h.hexdigest()
        group_counts[group] = {"leaves": len(entries), "numel": sum(leaves[k]["numel"] for k, _ in entries)}
    return {"leaves": leaves, "group_counts": group_counts, "group_hashes": group_hashes}


def compare_fingerprints(before: dict, after: dict, *, required_trainable_groups, frozen_groups) -> dict:
    old, new = before["leaves"], after["leaves"]
    if set(old) != set(new):
        missing = sorted(set(old) - set(new)); added = sorted(set(new) - set(old))
        raise ValueError(f"parameter paths differ; missing after={missing[:5]}, added after={added[:5]}")
    counts = {}
    changed = set()
    for path in sorted(old):
        a, b = old[path], new[path]
        for field in ("shape", "dtype", "group"):
            if a[field] != b[field]:
                raise ValueError(f"{field} mismatch at {path}: {a[field]!r} != {b[field]!r}")
        group = a["group"]
        entry = counts.setdefault(group, {"leaves": 0, "changed": 0, "preserved": 0})
        entry["leaves"] += 1
        if a["sha256"] == b["sha256"]:
            entry["preserved"] += 1
        else:
            entry["changed"] += 1
            changed.add(path)
    frozen_groups = set(frozen_groups)
    required_trainable_groups = set(required_trainable_groups)
    if required_trainable_groups & frozen_groups:
        raise ValueError("a parameter group cannot be both frozen and required trainable")
    missing_frozen = sorted(frozen_groups - set(counts))
    if missing_frozen:
        raise ValueError(f"required frozen groups absent: {missing_frozen}")
    for group in sorted(frozen_groups):
        if counts.get(group, {}).get("changed", 0):
            path = next(p for p in sorted(changed) if old[p]["group"] == group)
            raise ValueError(f"frozen group {group} changed at {path}")
    absent = sorted(required_trainable_groups - set(counts))
    if absent:
        raise ValueError(f"required trainable groups absent: {absent}")
    unchanged = sorted(g for g in required_trainable_groups if counts[g]["changed"] == 0)
    if unchanged:
        raise ValueError(f"required trainable groups had zero changed leaves: {unchanged}")
    return {"groups": counts, "changed_leaves": sorted(changed),
            "frozen_unchanged": True, "trainable_groups_updated": True}


def robomme_parameter_group(path: tuple[str, ...]) -> str:
    """Classify Robomme paper parameters; LoRA is intentionally unsupported."""
    parts = tuple(path)
    joined = "/".join(parts).lower()
    if "lora" in joined:
        raise ValueError(f"LoRA path is unsupported by paper classifier: {_path_text(parts)}")
    qualified = "/" + joined + "/"
    if "/paligemma/img/" in qualified:
        return "vision"
    if parts and parts[0] == "mem_encoder":
        return "memory_bridge"
    if "mem_rms_norm" in joined:
        return "memory_modulation"
    if "mem_attn" in joined or "memory_attn" in joined:
        return "memory_attention"
    if parts and (parts[0].startswith("action_") or parts[0].startswith("time_")):
        return "action_adapters"
    if "/paligemma/llm/" in qualified and any(p.endswith("_1") for p in parts):
        return "action_expert"
    if "/paligemma/llm/" in qualified:
        return "vlm"
    return "unclassified"


def _walk(mapping: Mapping, prefix: tuple[str, ...] = ()):
    for key in sorted(mapping):
        if not isinstance(key, str):
            raise TypeError(f"parameter mapping key must be string: {key!r}")
        value = mapping[key]
        path = prefix + (key,)
        if isinstance(value, Mapping):
            yield from _walk(value, path)
        else:
            yield path, value


def _path_text(path: tuple[str, ...]) -> str:
    return "/".join(path)
