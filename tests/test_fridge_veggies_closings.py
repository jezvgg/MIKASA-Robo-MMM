"""CPU-only check of the FridgeVeggies closing axis: uv run pytest tests/test_fridge_veggies_closings.py.

Later planners dropped the rank-closings-by-plan-length stage (the TakeIt era tried several
jaw directions and kept the shortest screw path); the closing is now one measurement of the
vegetable's own collision mesh, so that measurement is what this file guards.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from planners import myrobocasa_fridge_veggies_planner as planner_module

AXIS = planner_module._narrow_closing_axis


def slab(width_x: float, width_y: float) -> SimpleNamespace:
    """Eight vertices of a box centred on the origin, `width_x` by `width_y` across."""
    xs = (-width_x / 2.0, width_x / 2.0)
    ys = (-width_y / 2.0, width_y / 2.0)
    vertices = [
        [x, y, z] for x in xs for y in ys for z in (-0.01, 0.01)
    ]
    return SimpleNamespace(vertices=np.asarray(vertices, dtype=np.float64))


def test_closing_takes_the_narrow_horizontal_axis():
    # Jaws close across the narrow side, whatever the long side points at.
    np.testing.assert_allclose(AXIS(slab(0.10, 0.06), np.array([0.0, 1.0, 0.0])), [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(AXIS(slab(0.06, 0.10), np.array([1.0, 0.0, 0.0])), [1, 0, 0], atol=1e-9)


def test_closing_flips_toward_the_hand():
    # The jaws are symmetric, so the sign follows the current finger axis.
    np.testing.assert_allclose(AXIS(slab(0.10, 0.06), np.array([0.0, -1.0, 0.0])), [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(AXIS(slab(0.06, 0.10), np.array([-1.0, 0.0, 0.0])), [-1, 0, 0], atol=1e-9)


def test_closing_ignores_the_world_origin():
    # A vegetable standing away from the origin still reports the same direction.
    mesh = slab(0.10, 0.06)
    mesh.vertices = mesh.vertices + np.array([0.55, -0.30, 0.90])
    np.testing.assert_allclose(AXIS(mesh, np.array([0.0, 1.0, 0.0])), [0, 1, 0], atol=1e-9)


@pytest.mark.parametrize(
    "vertices, message",
    [
        (np.zeros((2, 3)), "needs finite 3D vertices"),
        (np.array([[0.0, 0.0, 0.0], [0.1, np.nan, 0.0], [0.2, 0.1, 0.0]]), "needs finite 3D vertices"),
        (np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.2, 0.2, 0.0]]), "degenerate horizontal"),
    ],
)
def test_closing_refuses_a_mesh_it_cannot_measure(vertices, message):
    with pytest.raises(ValueError, match=message):
        AXIS(SimpleNamespace(vertices=vertices), np.array([0.0, 1.0, 0.0]))
