"""Merging must retain failed attempts and reject incompatible/duplicated data."""

import copy
import hashlib
import json

import pytest

from utils.collection.merge_lerobot import merge_provenance, select_episodes, validate_release
from utils.collection.provenance import dataset_provenance_complete


def batch(root, candidates, selected):
    return dict(
        control_hz=20,
        policy_hz=10,
        action_repeat=2,
        tokenizer_sha256="tokenizer",
        source_root=root,
        source_run=dict(seeds=candidates, signature=dict(code_sha256="same-code")),
        source_episode_outcomes=[
            dict(
                scene_seed=s,
                success=s in selected,
                status="success" if s in selected else "missed",
            )
            for s in candidates
        ],
        episodes=[
            dict(scene_seed=s, episode_index=i, success=True, truncated=False)
            for i, s in enumerate(selected)
        ],
    )


def test_merge_keeps_real_campaigns_and_failed_attempts():
    a, b = batch("/first", [1, 2], [1]), batch("/second", [3, 4, 5], [3, 5])
    merged = merge_provenance([a, b])
    assert merged["source_run"]["seeds"] == [1, 2]
    assert [s["run"]["seeds"] for s in merged["source_runs"]] == [[1, 2], [3, 4, 5]]
    assert [e["scene_seed"] for e in merged["source_episode_outcomes"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert sum(e["success"] for e in merged["source_episode_outcomes"]) == 3
    assert [e["episode_index"] for e in merged["episodes"]] == [0, 1, 2]
    assert [e["source_root"] for e in merged["episodes"]] == [
        "/first",
        "/second",
        "/second",
    ]
    assert "source_runs" not in a


def test_two_shards_count_one_campaign_only_once():
    a = batch("/same", [1, 2], [1, 2])
    b = copy.deepcopy(a)
    a["episodes"] = a["episodes"][:1]
    b["episodes"] = b["episodes"][1:]
    merged = merge_provenance([a, b])
    assert len(merged["source_runs"]) == 1
    assert len(merged["source_episode_outcomes"]) == 2


@pytest.mark.parametrize(
    "kind", ["duplicate", "runtime", "attempt_overlap", "outcome", "truncated"]
)
def test_invalid_inputs_are_rejected(kind):
    a, b = batch("/one", [1, 2], [1]), batch("/two", [3, 4], [3])
    if kind == "duplicate":
        b["episodes"][0]["scene_seed"] = 1
    elif kind == "runtime":
        b["source_run"]["signature"]["code_sha256"] = "different-code"
    elif kind == "attempt_overlap":
        b["source_run"]["seeds"] = [2, 3, 4]
    elif kind == "outcome":
        b = copy.deepcopy(a)
        b["episodes"] = []
        b["source_episode_outcomes"][0]["success"] = False
    else:
        b["episodes"][0]["truncated"] = True
    with pytest.raises(ValueError):
        merge_provenance([a, b])


def qualified_batch(root, start, stop, converter):
    metadata = batch(root, list(range(start, stop)), list(range(start, stop)))
    files = {"scene.py": "f" * 64}
    signature = dict(
        source_files=files,
        code_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        profile=dict(robot_reference=dict(commit="a" * 40)),
    )
    metadata["source_run"]["signature"] = signature
    metadata["provenance"] = dict(
        source_signature_sha256=signature["code_sha256"],
        source_file_count=1,
        component_commits=dict(
            robot="a" * 40,
            environment="b" * 40,
            planner="b" * 40,
            converter=converter * 40,
        ),
        exporter_source_files={"converter.py": converter * 64},
    )
    return metadata


def test_nested_merge_keeps_each_original_converter():
    a = qualified_batch("/one", 1, 3, "c")
    b = qualified_batch("/two", 3, 5, "d")
    c = qualified_batch("/three", 5, 7, "e")
    ab = merge_provenance([a, b])
    merged = merge_provenance([ab, c])
    by_id = {c["id"]: c for c in merged["conversion_runs"]}
    original = [
        by_id[e["conversion_id"]]["provenance"]["component_commits"]["converter"]
        for e in merged["episodes"]
    ]
    assert original == ["c" * 40] * 2 + ["d" * 40] * 2 + ["e" * 40] * 2
    assert dataset_provenance_complete(merged)
    assert "conversion_id" not in a["episodes"][0]


@pytest.mark.parametrize(
    "missing",
    ["original_converter", "final_converter", "episode_binding", "source_manifest"],
)
def test_production_merge_cannot_bypass_commit_gate_with_small_batches(missing):
    a = qualified_batch("/one", 0, 500, "c")
    b = qualified_batch("/two", 500, 1000, "d")
    merged = merge_provenance([a, b])
    validate_release(merged, "release-1000ep")
    if missing == "original_converter":
        merged["conversion_runs"][1]["provenance"]["component_commits"][
            "converter"
        ] = None
    elif missing == "final_converter":
        merged["provenance"]["component_commits"]["converter"] = None
    elif missing == "episode_binding":
        merged["episodes"][-1]["conversion_id"] = "unknown"
    else:
        merged["source_run"]["signature"]["source_files"]["scene.py"] = "0" * 64
    assert not dataset_provenance_complete(merged)
    with pytest.raises(ValueError, match="A2"):
        validate_release(merged, "release-1000ep")


def test_legacy_pilot_is_not_a_production_release():
    merged = merge_provenance([batch("/one", [1, 2], [1])])
    validate_release(merged, "dataset-test-1ep")
    assert not dataset_provenance_complete(merged)
    with pytest.raises(ValueError, match="sub-1000"):
        validate_release(merged, "release")


def test_nested_merge_keeps_distinct_video_settings_for_same_converter():
    a, b = batch("/one", [1], [1]), batch("/two", [2], [2])
    a["video"] = {"profile": "reference", "gop": 2}
    b["video"] = {"profile": "compact", "gop": 12}
    a["episodes"][0]["video_encoding"] = {"fetch_hand": {"crf": 12}}
    b["episodes"][0]["video_encoding"] = {"fetch_hand": {"crf": 16}}
    merged = merge_provenance([merge_provenance([a]), b])
    by_id = {row["id"]: row for row in merged["conversion_runs"]}
    assert len(by_id) == 2
    for episode, expected in zip(merged["episodes"], [a, b]):
        assert by_id[episode["conversion_id"]]["video"] == expected["video"]
        assert episode["video_encoding"] == expected["episodes"][0]["video_encoding"]


def test_selection_keeps_every_attempt_and_renumbers_selected_episodes():
    merged = merge_provenance([batch("/first", [1, 2, 3], [1, 3]), batch("/second", [4, 5, 6], [4, 5, 6])])
    selected = select_episodes(merged, dict(rule="first three accepted", episode_seeds=[1, 3, 4]))
    assert [e["scene_seed"] for e in selected["episodes"]] == [1, 3, 4]
    assert [e["episode_index"] for e in selected["episodes"]] == [0, 1, 2]
    assert len(selected["source_episode_outcomes"]) == 6
    assert sum(e["success"] for e in selected["source_episode_outcomes"]) == 5
    assert selected["episode_selection"]["accepted_not_selected_seeds"] == [5, 6]
    assert len(merged["episodes"]) == 5


@pytest.mark.parametrize("seeds", [[1, 1], [2], [7]])
def test_selection_rejects_duplicates_and_unmerged_seeds(seeds):
    merged = merge_provenance([batch("/first", [1, 2, 3], [1, 3])])
    with pytest.raises(ValueError):
        select_episodes(merged, dict(rule="x", episode_seeds=seeds))
