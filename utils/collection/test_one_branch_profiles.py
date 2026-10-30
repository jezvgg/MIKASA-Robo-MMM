"""Every collection profile rebuilds its dataset's physics and text on one branch (the owner's call
2026-10-26): the assisted-grasp drive and the instruction version are profile fields."""
import pytest

from utils.collection import profile

#: profile -> (assisted_grasp_hand_collides, instruction_version or None): the proxy for
#: CabinetSearch only; instruction_version 3 for SeasonDish's shaking task (profile 27, owner
#: 2026-10-27); 2, the one current instruction, for DepthRecall and CabinetSearch5, whose HF datasets
#: carry it since 2026-10-30, and for the four-compartment CabinetSearch (profile 19, never collected).
EXPECTED = {"same_drawer": (False, None), "season_dish": (False, 3), "depth_recall": (False, 2),
            "cabinet_search": (True, 2), "cabinet_search5": (True, 2)}

#: The texts the episodes were recorded with (seasondish-ag-1000ep, and depthrecall-ag-1000ep's
#: sources/ since its HF text became the current one), copied here so a change to either scene
#: breaks this test.
RECORDED = {
    "season_dish": "Remember the salt or pepper shown in the picture on the fridge. After the picture "
                   "disappears, take that condiment and tip it over the bowl to season the dish, leaving "
                   "the other condiment untouched.",
    "depth_recall": "Take the props out of the cabinet onto the counter pads (ketchup on the red pad, pepper "
                    "on the green pad, salt on the blue pad, mustard on the yellow pad), keep the one from "
                    "the back, and put the others back where they stood.",
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_profile_names_its_drive_and_text(name):
    kwargs = profile.read_json(profile.PROFILE.with_name(f"{name}_profile.json"))["env_kwargs"]
    hand_collides, version = EXPECTED[name]
    assert kwargs["assisted_grasp"] is True
    assert kwargs["assisted_grasp_hand_collides"] is hand_collides
    assert kwargs.get("instruction_version") == version


@pytest.mark.parametrize("name,module", [("season_dish", "my_scenes.season_dish"),
                                         ("depth_recall", "my_scenes.depth_recall_v1")])
def test_instruction_version_1_is_the_recorded_text(name, module):
    pytest.importorskip("mani_skill")
    scene = pytest.importorskip(module)
    assert scene.INSTRUCTION_VERSIONS[1][0] == RECORDED[name]
    latest = scene.INSTRUCTION_VERSIONS[max(scene.INSTRUCTION_VERSIONS)]
    assert latest == scene.INSTRUCTIONS != scene.INSTRUCTION_VERSIONS[1]


def test_the_campaign_checks_the_profiles_own_text():
    """`validate_instructions` must vouch for the text the profile's env gives, or every
    episode fails its collection contract ("Task instruction was not checked")."""
    pytest.importorskip("mani_skill")
    from my_scenes import depth_recall_v1
    pinned = profile.read_json(profile.PROFILE.with_name("depth_recall_profile.json"))
    assert profile.profile_instructions(pinned) == set(depth_recall_v1.INSTRUCTIONS)
    recorded = dict(pinned, env_kwargs=dict(pinned["env_kwargs"], instruction_version=1))
    assert profile.profile_instructions(recorded) == {RECORDED["depth_recall"]}


def test_season_dish_collects_the_shaking_task_text():
    """Profile 27 pins the shaking task's text (version 3), and the campaign vouches for it alone."""
    pytest.importorskip("mani_skill")
    from my_scenes import season_dish
    pinned = profile.read_json(profile.PROFILE.with_name("season_dish_profile.json"))
    texts = profile.profile_instructions(pinned)
    assert season_dish.INSTRUCTIONS[0] in texts and RECORDED["season_dish"] not in texts
    assert "number" in season_dish.INSTRUCTIONS[0] and "shake" in season_dish.INSTRUCTIONS[0]


@pytest.mark.parametrize("module", ["my_scenes.same_drawer", "my_scenes.season_dish",
                                    "my_scenes.depth_recall_v1", "my_scenes.cabinet_search"])
def test_one_instruction_per_task(module):
    """One instruction per task (owner 2026-10-30), as LIBERO, RoboCasa and MIKASA-Robo-VLA."""
    pytest.importorskip("mani_skill")
    scene = pytest.importorskip(module)
    assert len(scene.INSTRUCTIONS) == 1
    assert len(getattr(scene, "INSTRUCTIONS_NUDGE", ("",))) == 1


@pytest.mark.parametrize("name", ["cabinet_search", "cabinet_search5"])
def test_cabinet_profiles_vouch_for_the_one_instruction(name):
    """Both profiles give the one instruction of each terminal (CabinetSearch5's HF dataset carries it
    since 2026-10-30); version 1 vouches for the three drawn phrasings the episodes were recorded with."""
    pytest.importorskip("mani_skill")
    from my_scenes import cabinet_search
    pinned = profile.read_json(profile.PROFILE.with_name(f"{name}_profile.json"))
    assert profile.profile_instructions(pinned) == {cabinet_search.PHRASINGS[0], cabinet_search.PHRASINGS_NUDGE[0]}
    drawn = dict(pinned, env_kwargs=dict(pinned["env_kwargs"], instruction_version=1))
    assert profile.profile_instructions(drawn) == set(cabinet_search.PHRASINGS) | set(cabinet_search.PHRASINGS_NUDGE)


@pytest.mark.parametrize("version,expected", [(1, [0, 1, 2, 1]), (2, [0, 0, 0, 0])])
def test_cabinet_search_gives_the_drawn_phrasing_only_under_version_1(version, expected):
    pytest.importorskip("mani_skill")
    import torch
    from types import SimpleNamespace
    from my_scenes import cabinet_search
    env = SimpleNamespace(cfg=SimpleNamespace(home_marker="rug", terminal="nudge"), instruction_version=version,
                          instruction_idx=torch.tensor([0, 1, 2, 1], dtype=torch.int32))
    texts = cabinet_search.CabinetSearchTask.get_language_instruction(env)
    assert texts == [cabinet_search.PHRASINGS_NUDGE[i] for i in expected]
