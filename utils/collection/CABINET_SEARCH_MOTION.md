# CabinetSearch motion profile 13

After physically closing an empty compartment, the planner opens the gripper on
the next even 20 Hz control step, before withdrawing or folding the arm. The same
ordering applies to the optional correction by the handle. The robot returns
directly to the remembered home marker with open fingers. Closure still requires
all compartment leaves within 0.01 rad for ten steps after releasing and withdrawing;
the environment never resets hinges after scene initialization.

Before touching the can, the planner checks a joint path that raises the torso
while moving the arm to its first preparation pose. The wheel base remains
stationary. If this path is unavailable, the original sequential preparation is
used and logged as `touch preparation: sequential_fallback`. Success ends at the
can, with no final return.

The fixed 40-scene development comparison passed 40/40 physical executions,
40/40 native20 replays and 40/40 true held10 replays. All 71 panel-closing strokes
released on the required even step; every recorded home transit used open fingers.
There were 19 combined preparations and 21 explicitly logged sequential fallbacks.
Handle correction was not needed in this pool. These results establish the stated
motion/replay checks, not acceptance of every dataset requirement or human E5.

Recordings retain their profile, source hash and source snapshot. The robot model,
controller, action/observation formats and 20 Hz recording rate are unchanged.
