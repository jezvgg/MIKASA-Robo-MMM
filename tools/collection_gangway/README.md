# Assisted-grasp collection pipeline (lab H100 server, 2026-10-17 → 10-26)

The scripts that collected the `nurtayev-d/*-ag-1000ep` datasets, copied as they ran from
`~/sd-collect` on the server (last synced 2026-10-26). They are the pod's 2026-09-28 collection
pipeline, adapted to this server: a 12-CPU quota, NFS, a shared disk budget. The code under test
lives in the task branches. These are the runners around it.

| Dataset | Collected on | Profile | Accepted / attempts |
|---|---|---|---|
| samedrawer-ag-1000ep | collection/samedrawer-ag 9238467 | 21 | 1106 / 1462 |
| seasondish-ag-1000ep | collection/seasondish-ag 5dbf5a8 | 26 | 1044 / 1320 |
| cabinetsearch-ag-1000ep (four compartments, superseded) | collection/samedrawer-ag 4562981 | 16 | 1051 / 1360 |
| depthrecall-ag-1000ep | feat/depth-recall-mustard ddfbccf | 6 | 1057 / 1100 |
| cabinetsearch5-ag-1000ep | feat/cabinet-fifth-door d65cd4c | cabinet_search5 7 | 1062 / 1080 |

Since 2026-10-26 `train/vla` (2fa4560 and later) rebuilds all of them but the four-compartment one:
20 re-collected episodes per dataset are bit-identical to the published ones
(`tools/assisted_grasp_check/RESULTS.md`, "One branch"). Run the collection from a clone of that
branch; each dataset's profile carries its assisted-grasp drive and instruction version.

## Layout

- `driver/collection_driver.py` is the resumable per-task loop.
  - Each batch is declared before it runs; a batch's postbatch gate overlaps the next campaign; at most two gates are pending.
  - It pauses above a disk budget.
  - Usage: `collection_driver.py <task> <batch> <jobs> <target> <budget_gb> [prefix] [first_seed]`.
  - SameDrawer stops at its per-drawer quota of 262: run it with a target above 1048 so that it ends on "seed range exhausted".
  - Each task's clone and seed range are `REPOS` and `RANGES` at its top.
- `driver/postbatch.py` is the gate:
  - it runs the audits and the acceptance (rules in `driver/ACCEPTANCE_RULES.json`, v7: amendment 6 is
    CabinetSearch's E1 rug rule, amendment 7 its D5 exception);
  - then the compact LeRobot export, retention prepare/prune, and the ledger.
- The audits:
  - `driver/audit_*.py` are the pod's audits;
  - `audit_can_early.py` is CabinetSearch's can-before-the-push gate;
  - `audit_restored_views.py` counts a CabinetSearch rug frame with the base on the rug as reached (E1);
    `e1_recompute.py` re-scored the batches gated before that rule;
  - the `*_dr.py` audits are DepthRecall's.
- `driver/SEED_PLAN.json` is the declared seed plan and every amendment, each with the owner's reason;
  its `excluded` list is every seed range already used.
- `driver/README_NOTES_cabinet_search5.md`: the five-compartment dataset's E1 rule and D5 exception.
- `driver/final/` turns the ledgers into exactly 1000 episodes:
  - `build_selection*.py` selects them;
  - `finalize_dataset.py` writes the H4 check and the README; `readme_cabinet5_notes.py` adds the
    five-compartment notes, measured on the selection;
  - `vla_ready.py` is the pi0.5 readiness check.
- Top-level shell scripts:
  - `launch*.sh` start a driver (`launch_cab19.sh` is cancelled: CabinetSearch is fixed at five compartments);
  - `restart_*.sh` restart one between campaigns with a new budget or target;
  - `after_*.sh` chain the finalize;
  - `finalize_*_ag.sh` run validation100 → selection → linked merge → README → checklist → readiness.
- Uploads from the server, the owner's token in a `/dev/shm` file (`HF_TOKEN_PATH`), never in Git:
  - `hf_folder_upload.py` commits a folder in verified portions (`--prefix ""` for the repo root);
  - `upload_cab5.sh` (dataset, then each batch as `sources/<batch>/`), `sources_upload_dr.sh`;
  - `org_upload_all.sh` and `org_replace_cabinet.py`: the organisation repo's task folders;
  - `check_sources.py` checks every `sources/` file on the Hub (size, h5 sha256).
- `checks/`: the 2026-10-26 checks:
  - `repro_collect.py` re-collects 20 episodes of a dataset and compares them with the published ones;
  - `overlap_run.sh` runs `tools/assisted_grasp_check/hand_overlap_replay.py` on 100 episodes per dataset;
  - `avg_replay.py`, `interp_replay.py`, `interp_pad.py`: 10 Hz → 20 Hz with pair-averaged or
    interpolated targets instead of holding (`sdish_check*.sh` collected their seeds).
- `mac/` holds the tools that ran on the Mac before the server could upload:
  - `hf_upload_pipe.py` copies the server to the Hub in verified portions;
  - `hf_sources_loop.py` uploads each batch's source `.h5` to `sources/<batch>/` of the dataset repo.

## Lessons this collection paid for

- **Locks on NFS.** An exclusive `flock` on a read-only descriptor is EBADF, so open lock files with `"a"`.
- **The linked merge needs the source h5.** `merge_lerobot` → `export_lerobot.verify` re-reads the source h5 of every exported episode. Never delete the h5 before the final merge.
- **Total targets overshoot.** In-flight batches push a total target past the per-drawer quota, so stop on the quotas instead.
- **One uploader on the Mac at a time.** Each `hf_upload_pipe` keeps up to three portions on disk.
- **No Xet on the Mac.** Use `HF_HUB_DISABLE_XET=1` there: hub 1.8's Xet upload hung on the first commit.
  From the server it is the other way round: its proxy blocks the LFS S3 host and Xet works
  (`HF_HUB_DISABLE_XET=0`, about 1 GB a minute).
- **Never move a clone a campaign runs from.** Every worker checks the run's source signature; a
  clone moved to a new commit mid-campaign fails every later seed. Bundle a new commit into a fresh clone.
