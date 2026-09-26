"""Floor paths stop inside a measured goal region, without filtering actions."""
from types import SimpleNamespace
import numpy as np
from planners.oracle.collection_solver import CollectionMotionPlanner
from robots.fetch.extand import FetchMotionPlanningSapienSolver


def make_planner(distance=.2,tolerance=.1):
    p=object.__new__(CollectionMotionPlanner)
    p.navigation_arrival_tolerance=tolerance
    p.env_agent=SimpleNamespace(base_link=SimpleNamespace(pose=SimpleNamespace(sp=SimpleNamespace(p=np.array([distance,0.,0.])))))
    p._guard=SimpleNamespace(truncated=False)
    p._navigation_leg=[np.zeros(2),0]
    p.idle_steps=lambda t:("settled",t)
    return p


def test_stop_before_another_action_pair_after_measured_arrival(monkeypatch):
    p=make_planner();executed=[]
    path={"position":np.arange(6)[:,None],"velocity":np.array([1,1,1,1,-1,-1])[:,None]}
    before=path["velocity"].copy()
    def follow(self,pair,refine_steps=0):
        executed.extend(pair["velocity"][:,0].tolist())
        self.env_agent.base_link.pose.sp.p[0]-=.075
        return "moving"
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",follow)
    assert p.follow_moving_forward(path)==("settled",2)
    assert executed==[1,1,1,1]
    np.testing.assert_array_equal(path["velocity"],before)


def test_reverse_path_outside_arrival_region_is_not_filtered(monkeypatch):
    p=make_planner(distance=1.);executed=[]
    path={"position":np.zeros((4,1)),"velocity":np.array([1,1,-1,-1])[:,None]}
    def follow(self,pair,refine_steps=0):
        executed.extend(pair["velocity"][:,0].tolist());return "moving"
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",follow)
    assert p.follow_moving_forward(path)=="moving"
    assert executed==[1,1,-1,-1]


def test_default_preserves_canonical_path(monkeypatch):
    p=make_planner(tolerance=0.)
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",lambda self,result,refine_steps:result)
    path={"unchanged":True}
    assert p.follow_moving_forward(path) is path


def test_already_at_goal_needs_no_movement(monkeypatch):
    p=make_planner(distance=.01)
    def unexpected(*args,**kwargs):raise AssertionError("Already inside goal region")
    monkeypatch.setattr(FetchMotionPlanningSapienSolver,"follow_moving_forward",unexpected)
    assert p.follow_moving_forward({"position":np.zeros((2,1)),"velocity":np.ones((2,1))})==("settled",2)


def test_arrived_drive_skips_opening_turn_but_keeps_checked_final_view(monkeypatch):
    p = make_planner(distance=.03)
    p.forward_navigation = True
    calls = []
    def canonical(self, **kwargs):
        calls.append(kwargs)
        return "checked view"
    monkeypatch.setattr(FetchMotionPlanningSapienSolver, "drive_base", canonical)
    assert p.drive_base(np.zeros(3), [1, 0, 0], freeze_arm=True) == "checked view"
    assert calls == [dict(target_pos=None, target_view_vec=[1, 0, 0],
                          freeze_arm=True, arrive_tol=None, reverse_ok=False)]
    assert p.drive_base(np.zeros(3)) == ("settled", 2)
    assert len(calls) == 1


def test_drive_outside_region_preserves_target_and_explicit_reverse_override(monkeypatch):
    p = make_planner(distance=.2)
    p.forward_navigation = True
    monkeypatch.setattr(FetchMotionPlanningSapienSolver, "drive_base", lambda self, **kw: kw)
    target = np.zeros(3)
    result = p.drive_base(target, reverse_ok=True)
    assert result["target_pos"] is target
    assert result["reverse_ok"] is True
