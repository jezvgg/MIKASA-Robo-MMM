# README notes: CabinetSearch, five compartments (profile 7)

To go into the dataset README at upload. Drafted 2026-10-25.

## E1 (target in frame), how it is counted

Per frame, the current target must be in the segmentation of at least one of the three cameras. The targets
follow the oracle's phases: the rug while driving home, the leaf of the cabinet being searched during a round,
the can once found.

The rug (radius 0.20 m) is smaller than the robot's base, so once the robot stands on it no camera can see
it. In the rug phases a frame with the base centre within `home_radius` (0.20 m) of the rug centre therefore
counts as the destination reached. This rule was approved by the owner on 2026-10-25 (ACCEPTANCE_RULES
amendment 6). The raw camera visibility is kept per episode as `visible_fraction_raw`.

Measured on 451 accepted episodes:
- with the rule: min 0.971, median 0.977, 451/451 at or above 0.95;
- raw: median 0.895;
- the rug is visible on 99-100 % of frames at every base distance of 0.1 m or more, and on 3 % below it;
- the remaining misses are the first frames of a round, while the head turns up from the rug to the cabinet.

## Documented exception: D5 (base direction changes)

The checklist's D5 asks the base not to drive back and forth. CabinetSearch does that by design: every round
drives from the rug to a cabinet, and after closing it the robot backs off and returns to the rug. In 405 of
412 accepted episodes D5 fails, with a median of 90 reversing frames and about 18 changes of direction an
episode. The owner chose on 2026-10-25 to keep the motion and document it as a task exception, as for
DepthRecall (2026-10-18). D5 is measured and reported for every episode and was never used to filter
(ACCEPTANCE_RULES amendments 2 and 7).

## Open items

- H4: the validation seeds (100 from candidates 7847000-7847179) are built after the training collection.

## Upload

Hugging Face: `nurtayev-d/cabinetsearch5-ag-1000ep` (name chosen 2026-10-25, the owner: «сам придумай»),
MIT and public like the other datasets, uploaded after the exact 1000, validation100 and the checklist.
