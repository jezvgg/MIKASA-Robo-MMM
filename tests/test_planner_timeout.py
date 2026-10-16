"""CPU-only deadline tests; no simulator/assets required."""

import signal
import time
from unittest.mock import Mock

import pytest

from utils.planner_timeout import planner_time_limit


def test_timeout_returns_false_and_records_reason():
    env = Mock()

    @planner_time_limit(0.02)
    def planning(env, seed):
        # A generic planner recovery must not swallow the deadline.
        try:
            time.sleep(1)
        except Exception:
            return True

    started = time.monotonic()
    assert planning(env, 31) is False
    assert time.monotonic() - started < 0.5
    env.log_event.assert_called_once()
    assert env.log_event.call_args.args == ("error", "Episode timed out")
    assert env.log_event.call_args.kwargs["timeout_seconds"] == 0.02
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_success_and_handler_restoration():
    previous = signal.getsignal(signal.SIGALRM)
    env = Mock()

    @planner_time_limit(300)
    def planning(env, seed):
        return seed == 31

    assert planning(env, 31) is True
    assert planning(env, 32) is False
    env.log_event.assert_not_called()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_exception_cancels_timer():
    @planner_time_limit(300)
    def planning(env, seed):
        raise ValueError("planner failed")

    with pytest.raises(ValueError, match="planner failed"):
        planning(Mock(), 0)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_each_episode_gets_a_fresh_budget():
    @planner_time_limit(0.05)
    def planning(env, seed):
        time.sleep(0.03)
        return True

    assert planning(Mock(), 0) is True
    assert planning(Mock(), 1) is True


def test_invalid_budget_is_rejected():
    for seconds in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            planner_time_limit(seconds)
