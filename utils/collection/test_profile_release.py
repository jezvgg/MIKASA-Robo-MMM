"""The runtime pin compares releases; the local build label is recorded, not compared."""
import pytest

from utils.collection import profile


@pytest.mark.parametrize("installed,expected,ok", [
    ("2.14.0", "2.14.0", True),
    ("2.14.0+cu126", "2.14.0", True),
    ("2.14.1+cu126", "2.14.0", False),
    ("3.0.0b22", "3.0.0b22", True),
    ("3.0.0b21", "3.0.0b22", False),
])
def test_release_ignores_only_the_local_label(installed, expected, ok):
    assert (profile.release(installed) == expected) is ok


def test_load_profile_accepts_a_local_build_of_the_pinned_release(monkeypatch):
    pins = profile.read_json(profile.PROFILE.with_name("same_drawer_profile.json"))["runtime_packages"]
    installed = {name: version + ("+cu126" if name == "torch" else "") for name, version in pins.items()}
    monkeypatch.setattr(profile.importlib.metadata, "version", lambda name: installed[name])
    for key in [k for k in profile.os.environ if k.startswith("MIKASA_")]:
        monkeypatch.delenv(key)
    assert profile.load_profile("same_drawer")["runtime_packages"] == pins
    installed["torch"] = "2.15.0"
    with pytest.raises(ValueError, match="Runtime requires torch"):
        profile.load_profile("same_drawer")
