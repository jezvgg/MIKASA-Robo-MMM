"""How far a closed-loop episode got: its task's stages, read from the simulator at every step.

Each task has its stages in task order, success last. A stage counts once it has been true at
some step (the first step is kept). An episode's progress is the share of stages reached in an
unbroken run from the first one, as CALVIN counts completed subtasks in a row; its furthest
stage is the last of that run. Errors are the memory mistakes the task itself rules against an
episode (a wrong drawer, the other condiment, a cabinet inspected twice, ...), kept beside it.

Our four tasks read the latches and intermediates their evaluate() returns in `info`. jezv's two
return only success there, so their stages read the scene: the grasp, his tray latch, the pot.
"""
from __future__ import annotations

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def every(value) -> bool:
    return bool(_array(value).astype(bool).all())


def some(value) -> bool:
    return bool(_array(value).astype(bool).any())


def number(value) -> float:
    return float(_array(value).reshape(-1)[0])


def _grasping(env, actor) -> bool:
    return some(env.unwrapped.agent.is_grasping(actor))


def _samedrawer(env, info):
    stages = {
        "closed the open drawer": every(info["closed_done"]),
        "apple on the plate": every(info["apple_done"]),
        "reopened the same drawer": every(info["apple_done"]) and every(info["reopened"]),
    }
    return stages, {"broke a rule (other drawer, order, apple)": every(info["failed"])}


def _seasondish(env, info):
    holding = every(info["grasp_target"])
    stages = {
        "took the pictured condiment": holding,
        "held it over the bowl": holding and every(info["over_bowl"]) and every(info["height_ok"]),
        "shook it once": number(info["shake_count"]) >= 1,
        "shook it the shown number of times": number(info["shake_count"]) >= number(info["num_shakes"]),
    }
    return stages, {"took the other condiment": every(info["grasp_distractor"]),
                    "shook too often or turned upright early": every(info["fail"])}


def _cabinetsearch(env, info):
    stages = {
        "opened a cabinet": number(info["n_decisions"]) >= 1,
        "opened the can's cabinet": every(info["revealed"]),
        "pushed the can": every(info["nudged"]),
    }
    return stages, {"inspected a cabinet twice": every(info["reopened"]),
                    "two cabinets open": every(info["two_open"]),
                    "skipped the rug": every(info["skipped_home"])}


def _depthrecall(env, info):
    stages = {
        "took a prop out": some(info["left_slot"]),
        "took all props out": every(info["all_left"]),
        "every prop on its pad": every(info["staged"]),
        "put the others back": every(info["all_restored"]),
    }
    return stages, {"a prop on another's pad": every(info["wrong_pad"]),
                    "a prop in another's place": every(info["wrong_assign"]),
                    "put the back prop back": every(info["target_returned"])}


def _takeitback(env, info):
    base = env.unwrapped
    holding = _grasping(env, base.cup)
    on_tray = some(base.placed_on_tray)
    return {"grasped the cup": holding, "cup on the tray": on_tray,
            "picked it up from the tray": on_tray and holding}, {}


def _veggie(env, info):
    base = env.unwrapped
    target = base.veggies[base._picture_target]
    others = [veggie for veggie in base.veggies if veggie is not target]
    return ({"grasped the pictured vegetable": _grasping(env, target),
             "pictured vegetable in the pot": bool(info["dbg_on_plate"])},
            {"grasped another vegetable": any(_grasping(env, veggie) for veggie in others)})


STAGES = {"samedrawer": _samedrawer, "seasondish": _seasondish, "cabinetsearch": _cabinetsearch,
          "depthrecall": _depthrecall, "takeitback": _takeitback, "veggie": _veggie}


class Progress:
    """The stages and errors of one episode; update() after every env.step."""

    def __init__(self, task):
        self.read = STAGES[task]
        self.reached, self.errors, self.names = {}, {}, None

    def update(self, env, info, step):
        stages, errors = self.read(env, info)
        stages["success"] = every(info["success"])
        self.names = self.names or list(stages)
        for group, flags in ((self.reached, stages), (self.errors, errors)):
            for name, value in flags.items():
                if value and name not in group:
                    group[name] = step

    def result(self):
        names = self.names or []
        run = 0
        while run < len(names) and names[run] in self.reached:
            run += 1
        return {"progress": run / len(names) if names else 0.0, "furthest_stage": names[run - 1] if run else None,
                "stages": names, "stages_reached": self.reached, "errors": self.errors}
