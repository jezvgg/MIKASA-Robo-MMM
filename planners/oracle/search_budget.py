"""Cumulative planning wall time; nested calls count once, physics is excluded."""
from contextlib import contextmanager
from functools import wraps
import signal
import time

class PlanningBudgetExceeded(RuntimeError):
    pass

class SearchBudget:
    def __init__(self, seconds=120., clock=time.perf_counter):
        self.limit=float(seconds); self.used=0.; self.depth=0; self.clock=clock
        self.exhausted=False; self.calls={}

    @contextmanager
    def measure(self, name):
        if self.exhausted or self.used >= self.limit:
            raise PlanningBudgetExceeded(f'planning budget exhausted: {self.used:.3f}/{self.limit:.0f}s')
        self.calls[name]=self.calls.get(name,0)+1
        outer=self.depth==0
        if outer:
            start=self.clock()
            previous=signal.getsignal(signal.SIGALRM)
            def timeout(*_):
                self.exhausted=True
                raise PlanningBudgetExceeded(f'planning budget exhausted: {self.limit:.0f}s')
            signal.signal(signal.SIGALRM,timeout)
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.limit-self.used))
        self.depth+=1
        try:
            yield
        finally:
            self.depth-=1
            if outer:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,previous)
                self.used+=self.clock()-start
                self.exhausted |= self.used >= self.limit
        if self.exhausted:
            raise PlanningBudgetExceeded(f'planning budget exhausted: {self.used:.3f}/{self.limit:.0f}s')

    def wrap(self, function, name, *, ik=False):
        @wraps(function)
        def call(*args, **kwargs):
            if ik:
                kwargs['n_init_qpos']=min(32,int(kwargs.get('n_init_qpos',32)))
            with self.measure(name):
                return function(*args,**kwargs)
        return call

def install(planner, task, seconds=120.):
    budget=SearchBudget(seconds); planner.search_budget=budget; task.search_budget=budget
    for name in ('IK','plan_qpos_line','plan_screw','plan_pose','plan_qpos','TOPP'):
        method=getattr(planner.planner,name,None)
        if callable(method):
            setattr(planner.planner,name,budget.wrap(method,name,ik=name=='IK'))
    return budget

def measured(function):
    @wraps(function)
    def call(planner,*args,**kwargs):
        budget=getattr(planner,'search_budget',None)
        if budget is None:return function(planner,*args,**kwargs)
        with budget.measure(function.__name__):
            return function(planner,*args,**kwargs)
    return call
