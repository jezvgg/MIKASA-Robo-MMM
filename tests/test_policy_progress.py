from types import SimpleNamespace

import numpy as np

from utils.policies import evaluate as ev
from utils.policies.progress import STAGES, Progress


def run(task, frames, env=None):
    progress = Progress(task)
    for step, info in enumerate(frames, start=1):
        progress.update(env, info, step)
    return progress.result()


def drawer(closed=False, apple=False, reopened=False, failed=False, success=False):
    return {"closed_done": [closed], "apple_done": [apple], "reopened": [reopened], "failed": [failed],
            "success": [success]}


def test_progress_is_the_unbroken_run_of_stages_from_the_first():
    out = run("samedrawer", [drawer(reopened=True),  # the cued drawer starts open: not the third stage
                             drawer(closed=True, reopened=True), drawer(closed=True)])
    assert out["stages"] == ["closed the open drawer", "apple on the plate", "reopened the same drawer", "success"]
    assert (out["progress"], out["furthest_stage"]) == (0.25, "closed the open drawer")
    assert out["stages_reached"] == {"closed the open drawer": 2}
    out = run("samedrawer", [drawer(closed=True), drawer(closed=True, apple=True, reopened=True, success=True)])
    assert (out["progress"], out["furthest_stage"]) == (1.0, "success")


def test_rule_breaks_are_errors_with_their_first_step():
    out = run("samedrawer", [drawer(), drawer(failed=True), drawer(failed=True)])
    assert (out["progress"], out["errors"]) == (0.0, {"broke a rule (other drawer, order, apple)": 2})


def test_seasondish_counts_the_shakes_of_the_pictured_condiment():
    def frame(grasp=True, over=True, shakes=0, distractor=False):
        return {"grasp_target": [grasp], "over_bowl": [over], "height_ok": [True], "shake_count": [shakes],
                "num_shakes": [3], "grasp_distractor": [distractor], "fail": [False], "success": [False]}

    out = run("seasondish", [frame(over=False), frame(), frame(shakes=1), frame(shakes=2)])
    assert (out["progress"], out["furthest_stage"]) == (0.6, "shook it once")
    assert run("seasondish", [frame(grasp=False, distractor=True)])["errors"] == {"took the other condiment": 1}


def test_depthrecall_needs_every_prop():
    def frame(left, staged, restored=False):
        return {"left_slot": [left], "all_left": [all(left)], "staged": [staged], "all_restored": [restored],
                "wrong_pad": [False], "wrong_assign": [False], "target_returned": [False], "success": [False]}

    out = run("depthrecall", [frame([1, 0, 0, 0], [0, 0, 0, 0]), frame([1, 1, 1, 1], [1, 1, 1, 0])])
    assert out["furthest_stage"] == "took all props out"  # one prop never reached its pad


def test_cabinetsearch_stages():
    frame = {"n_decisions": [2], "revealed": [True], "nudged": [False], "reopened": [True], "two_open": [False],
             "skipped_home": [False], "success": [False]}
    out = run("cabinetsearch", [frame])
    assert out["furthest_stage"] == "opened the can's cabinet"
    assert out["errors"] == {"inspected a cabinet twice": 1}


class Agent:
    def __init__(self, holding):
        self.holding = holding

    def is_grasping(self, actor):
        return np.array([actor is self.holding])


def test_jezv_tasks_read_the_scene():
    cup = object()
    scene = SimpleNamespace(cup=cup, placed_on_tray=np.array([False]), agent=Agent(cup))
    env = SimpleNamespace(unwrapped=scene)
    progress = Progress("takeitback")
    progress.update(env, {"success": [False]}, 1)
    scene.placed_on_tray, scene.agent = np.array([True]), Agent(None)
    progress.update(env, {"success": [False]}, 2)
    scene.agent = Agent(cup)
    progress.update(env, {"success": [False]}, 3)
    assert progress.result()["stages_reached"] == {"grasped the cup": 1, "cup on the tray": 2,
                                                   "picked it up from the tray": 3}
    veggies = [object(), object(), object()]
    scene = SimpleNamespace(veggies=veggies, _picture_target=1, agent=Agent(veggies[0]))
    out = run("veggie", [{"dbg_on_plate": False, "success": [False]}], SimpleNamespace(unwrapped=scene))
    assert (out["progress"], out["errors"]) == (0.0, {"grasped another vegetable": 1})


def test_every_task_has_stages():
    assert set(STAGES) == set(ev.TASKS)


def test_episode_records_its_progress():
    class Env:
        def __init__(self):
            self.n = 0

        def step(self, action):
            self.n += 1
            info = drawer(closed=self.n >= 2, apple=self.n >= 3, reopened=True, success=self.n >= 5)
            return {"step": self.n}, 0.0, info["success"][0], False, info

    class Client:
        frames = 0
        observation = staticmethod(lambda obs, prompt: obs)

        def ingest(self, observation):
            self.frames += 1

        def infer(self, observation):
            return np.zeros((20, 13), np.float32)

    out = ev.run_episode(Env(), {"step": 0}, Client(), "go", max_steps=100, horizon=2, execution="hold",
                         progress=Progress("samedrawer"))
    assert (out["status"], out["progress"]) == ("success", 1.0)
    assert out["stages_reached"] == {"closed the open drawer": 2, "apple on the plate": 3,
                                     "reopened the same drawer": 3, "success": 5}
