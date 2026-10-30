"""Every collection profile rebuilds its dataset's physics and text on one branch (the owner's call
2026-10-26): the assisted-grasp drive and the instruction version are profile fields."""
import pytest

from utils.collection import profile

#: profile -> (assisted_grasp_hand_collides, instruction_version or None): the proxy for
#: CabinetSearch5 only (its four-compartment sibling is gone, owner 2026-10-26); instruction_version
#: 1 where the dataset predates the colour names, 3 for SeasonDish's shaking task (profile 28).
EXPECTED = {"same_drawer": (False, None), "season_dish": (False, 3), "depth_recall": (False, 1),
            "cabinet_search5": (True, None)}

#: The texts the published datasets carry (meta/tasks of seasondish-ag-1000ep and
#: depthrecall-ag-1000ep), copied here so a change to either scene breaks this test.
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


@pytest.mark.parametrize("name", ["depth_recall"])
def test_the_campaign_checks_the_profiles_own_text(name):
    """`validate_instructions` must vouch for the text the profile's env gives, or every
    episode fails its collection contract ("Task instruction was not checked")."""
    pytest.importorskip("mani_skill")
    pinned = profile.read_json(profile.PROFILE.with_name(f"{name}_profile.json"))
    assert RECORDED[name] in profile.profile_instructions(pinned)
    current = dict(pinned, env_kwargs={k: v for k, v in pinned["env_kwargs"].items() if k != "instruction_version"})
    assert RECORDED[name] not in profile.profile_instructions(current)


def test_season_dish_collects_the_shaking_task_text():
    """Profile 27 pins the shaking task's text (version 3), and the campaign vouches for it alone."""
    pytest.importorskip("mani_skill")
    from my_scenes import season_dish
    pinned = profile.read_json(profile.PROFILE.with_name("season_dish_profile.json"))
    texts = profile.profile_instructions(pinned)
    assert season_dish.INSTRUCTIONS[0] in texts and RECORDED["season_dish"] not in texts
    assert "number" in season_dish.INSTRUCTIONS[0] and "shake" in season_dish.INSTRUCTIONS[0]
