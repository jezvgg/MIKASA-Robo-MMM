"""Wall-clock budget for a planner call on the Linux simulation host.

SIGALRM interrupts Python loops, including loops which never call env.step.
Python can defer signal delivery while a native extension is executing; use an
external per-episode process watchdog as well when a hard deadline is required.
"""

import functools
import math
import signal
import threading
import time


class _PlannerDeadlineExceeded(BaseException):
    """Bypass generic Exception recovery inside motion-planning helpers."""


def planner_time_limit(seconds: float):
    """Return False and log a timeout if planning(env, ...) exceeds its budget.

    The budget starts on entry, including the planner's reset/reconfigure. It
    excludes environment construction and video finalization. Only main-thread
    Unix execution is supported. An existing real-time alarm is rejected rather
    than overwritten; use an external process watchdog for enclosing deadlines.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("planner timeout must be a positive finite number")

    def decorate(planning_fn):
        @functools.wraps(planning_fn)
        def wrapped(env, *args, **kwargs):
            if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
                raise RuntimeError("planner deadline requires the main thread on Unix")
            if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
                raise RuntimeError("planner deadline cannot replace an active SIGALRM timer")

            def expired(signum, frame):
                raise _PlannerDeadlineExceeded()

            previous_handler = signal.getsignal(signal.SIGALRM)
            started = time.monotonic()
            timed_out = False
            signal.signal(signal.SIGALRM, expired)
            try:
                signal.setitimer(signal.ITIMER_REAL, seconds)
                try:
                    result = planning_fn(env, *args, **kwargs)
                    # A native call can postpone delivery. Never count a late
                    # success as a hit even when the signal was deferred.
                    timed_out = time.monotonic() - started >= seconds
                except _PlannerDeadlineExceeded:
                    timed_out = True
                    result = False
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)

            if timed_out:
                elapsed = time.monotonic() - started
                print(f"[TIMEOUT] Episode exceeded {seconds:g}s; counted as unsuccessful", flush=True)
                log_event = getattr(env, "log_event", None)
                if callable(log_event):
                    log_event(
                        "error", "Episode timed out", timeout_seconds=seconds,
                        elapsed_seconds=elapsed, success=False,
                    )
                return False
            return result

        return wrapped

    return decorate
