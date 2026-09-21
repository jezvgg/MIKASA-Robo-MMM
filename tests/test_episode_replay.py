import json

from utils.episode_replay import EpisodeSpec, verdict_from_transition


def test_episode_spec_round_trip_is_json_safe():
    spec = EpisodeSpec(
        episode_id=2,
        env_id="MikasaDepthRecall-v1",
        env_kwargs={"scene_idx": 0, "control_mode": "pd_joint_pos"},
        reset_kwargs={"seed": 17},
        seed=17,
        planner="depth_recall_v1_planner",
    )
    encoded = json.dumps(spec.to_dict())
    restored = EpisodeSpec.from_dict(json.loads(encoded))
    assert restored == spec


def test_replay_verdict_prefers_success_over_truncation():
    assert verdict_from_transition({"success": [True]}, [False], [True]) == "success"
    assert verdict_from_transition({"success": [False]}, [False], [True]) == "truncated"
    assert verdict_from_transition({"success": [False]}, [False], [False]) == "missed"
