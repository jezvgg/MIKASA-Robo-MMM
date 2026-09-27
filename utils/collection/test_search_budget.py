import time
import pytest
from planners.oracle.search_budget import SearchBudget, PlanningBudgetExceeded

def test_nested_planning_counts_once_and_excludes_physics():
    now=[0.];b=SearchBudget(120,clock=lambda:now[0])
    with b.measure('outer'):
        now[0]+=2
        with b.measure('inner'):now[0]+=3
    now[0]+=70  # physics and loading outside planning scopes
    with b.measure('next'):now[0]+=4
    assert b.used==9 and not b.exhausted

def test_budget_interrupts_search_and_remains_exhausted():
    b=SearchBudget(.02)
    with pytest.raises(PlanningBudgetExceeded):
        with b.measure('slow'):time.sleep(.2)
    assert b.exhausted
    with pytest.raises(PlanningBudgetExceeded):
        with b.measure('retry'):pass

def test_ik_initializations_are_capped():
    b=SearchBudget()
    observed=[]
    fn=b.wrap(lambda **kw:observed.append(kw['n_init_qpos']),'IK',ik=True)
    fn(n_init_qpos=160);fn(n_init_qpos=8)
    assert observed==[32,8]


def test_keepout_propagates_budget_failure_without_second_yield():
    from types import SimpleNamespace
    from planners.oracle.oracle_common import keepout
    planner=SimpleNamespace(planner=SimpleNamespace(planning_world=SimpleNamespace()))
    with pytest.raises(PlanningBudgetExceeded,match='spent'):
        with keepout(planner,[]):
            raise PlanningBudgetExceeded('spent')
