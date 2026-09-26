"""Counterexamples exercise the actual stateful task evaluator, without physics."""

from types import SimpleNamespace

import pytest
import torch

from my_scenes.same_drawer import SameDrawerConfig, SameDrawerTask


class Drawer:
    def __init__(self):
        self.q = torch.zeros(1, 1)
        self.v = torch.zeros(1, 1)

    def get_qpos(self):
        return self.q

    def get_qvel(self):
        return self.v

    def set_qpos(self, q):
        self.q = q

    def set_qvel(self, v):
        self.v = v


def task_at(target):
    task = SimpleNamespace(
        cfg=SameDrawerConfig(),
        elapsed_steps=torch.tensor([0]),
        _last_eval_step=torch.tensor([-1], dtype=torch.int32),
        target_drawer=torch.tensor([target]),
        _drawer_arts=[Drawer() for _ in range(4)],
        held_count=torch.tensor([0]),
    )
    for name in ("closed_done", "apple_done", "wrong_drawer_touched", "sequence_violated"):
        setattr(task, name, torch.tensor([False]))
    task.apple = SimpleNamespace(
        pose=SimpleNamespace(p=torch.tensor([[0.3, 0.0, 0.03]])),
        is_static=lambda **kwargs: torch.tensor([True]),
    )
    task.plate = SimpleNamespace(pose=SimpleNamespace(p=torch.zeros(1, 3)))
    task.grasped = False
    task.agent = SimpleNamespace(is_grasping=lambda obj: torch.tensor([task.grasped]))
    task.drawer_open_amounts = lambda: -torch.cat(
        [art.get_qpos() for art in task._drawer_arts], dim=1
    )
    task._drawer_arts[target].q[:] = -0.15

    def tick(*, opened=None, apple_on_plate=None, advance=True):
        if opened is not None:
            for index, amount in opened.items():
                task._drawer_arts[index].q[:] = -amount
        if apple_on_plate is not None:
            task.apple.pose.p[0, 0] = 0.0 if apple_on_plate else 0.3
        if advance:
            task.elapsed_steps += 1
        return SameDrawerTask.evaluate(task)

    tick()
    return task, tick


def finish(task, tick):
    target = int(task.target_drawer.item())
    for _ in range(task.cfg.hold_steps):
        info = tick(opened={target: 0.15})
    return info


@pytest.mark.parametrize("target", range(4))
def test_all_four_require_close_then_released_apple_then_reopen(target):
    task, tick = task_at(target)
    assert not task.closed_done.item()
    tick(opened={target: 0.0})
    task.grasped = True
    tick(apple_on_plate=True)
    assert not task.apple_done.item()
    task.grasped = False
    tick()
    assert task.apple_done.item()
    assert finish(task, tick)["success"].item()


@pytest.mark.parametrize("violation", ["early_apple", "early_reopen", "wrong_drawer", "lost_apple"])
def test_later_correct_moves_do_not_credit_invalid_sequence(violation):
    task, tick = task_at(2)
    if violation == "early_apple":
        tick(apple_on_plate=True)
    tick(opened={2: 0.0})
    if violation == "early_reopen":
        tick(opened={2: 0.06})
        tick(opened={2: 0.0})
    if violation == "wrong_drawer":
        tick(opened={0: 0.06})
        tick(opened={0: 0.0})
    tick(apple_on_plate=violation != "lost_apple")
    assert not finish(task, tick)["success"].item()


def test_repeated_evaluate_cannot_advance_terminal_hold():
    task, tick = task_at(0)
    tick(opened={0: 0.0})
    tick(apple_on_plate=True)
    tick(opened={0: 0.15})
    held = task.held_count.clone()
    for _ in range(20):
        assert not tick(advance=False)["success"].item()
    assert torch.equal(task.held_count, held)
    assert finish(task, tick)["success"].item()
