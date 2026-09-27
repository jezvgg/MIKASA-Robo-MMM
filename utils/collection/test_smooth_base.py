import numpy as np
from planners.oracle.smooth_base import rounded_route


def test_rounding_preserves_endpoints_and_monotone_forward_distance():
    start=np.array([2.3,-1.16]);corner=np.array([.8,-2.1]);end=np.array([.5,-1.5])
    route=rounded_route(start,corner,end,.4)
    np.testing.assert_allclose(route['points'][[0,-1]],[start,end])
    assert np.all(np.diff(route['distance'])>0)
    assert np.max(np.abs(np.diff(route['yaw'])))<.05
    assert route['distance'][-1] < np.linalg.norm(corner-start)+np.linalg.norm(end-corner)
    assert np.min(np.linalg.norm(route['points']-corner,axis=1))>.03
    assert abs(route['curvature'][0])<1e-6
    assert abs(route['curvature'][-1])<1e-6


def test_tiny_corner_does_not_generate_an_unchecked_fallback():
    assert rounded_route([0,0],[0,.01],[1,1],.4) is None
