# Assisted-grasp collection pipeline (lab H100 server, 2026-10-17 → 10-20)

The scripts that collected the `nurtayev-d/*-ag-1000ep` datasets, copied as they ran from
`~/sd-collect` on the server. They are the pod's 2026-09-28 collection pipeline, adapted to this
server: a 12-CPU quota, NFS, a shared disk budget, no Hugging Face token, uploads through a Mac.
The code under test lives in the task branches. These are the runners around it.

| Dataset | Branch / commit | Profile | Accepted / attempts |
|---|---|---|---|
| samedrawer-ag-1000ep | collection/samedrawer-ag 9238467 | 21 | 1106 / 1462 |
| seasondish-ag-1000ep | collection/seasondish-ag 5dbf5a8 | 26 | 1044 / 1320 |
| cabinetsearch-ag-1000ep | collection/samedrawer-ag 4562981 | 16 | 1051 / 1360 |
| depthrecall-ag-1000ep | feat/depth-recall-mustard ddfbccf | 6 | collecting |

## Layout

- `driver/collection_driver.py` is the resumable per-task loop.
  - Each batch is declared before it runs; a batch's postbatch gate overlaps the next campaign; at most two gates are pending.
  - It pauses above a disk budget.
  - Usage: `collection_driver.py <task> <batch> <jobs> <target> <budget_gb> [prefix] [first_seed]`.
  - SameDrawer stops at its per-drawer quota of 262: run it with a target above 1048 so that it ends on "seed range exhausted".
- `driver/postbatch.py` is the gate:
  - it runs the audits and the acceptance (rules in `driver/ACCEPTANCE_RULES.json`, v5);
  - then the compact LeRobot export, retention prepare/prune, and the ledger.
- The audits:
  - `driver/audit_*.py` are the pod's audits;
  - `audit_can_early.py` is CabinetSearch profile 16's can-before-the-push gate;
  - the `*_dr.py` audits are DepthRecall's.
- `driver/SEED_PLAN.json` is the declared seed plan and every amendment, each with the owner's reason.
- `driver/final/` turns the ledgers into exactly 1000 episodes:
  - `build_selection*.py` selects them;
  - `finalize_dataset.py` writes the H4 check and the README;
  - `vla_ready.py` is the pi0.5 readiness check.
- Top-level shell scripts:
  - `launch*.sh` start a driver;
  - `restart_*.sh` restart one between campaigns with a new budget or target;
  - `after_*.sh` chain the finalize;
  - `finalize_*_ag.sh` run validation100 → selection → linked merge → README → checklist → readiness.
- `mac/` holds the tools that run on the Mac:
  - `hf_upload_pipe.py` copies the server to the Hub in verified portions;
  - `hf_sources_loop.py` uploads each batch's source `.h5` to `sources/<batch>/` of the dataset repo.

## Lessons this collection paid for

- **Locks on NFS.** An exclusive `flock` on a read-only descriptor is EBADF, so open lock files with `"a"`.
- **The linked merge needs the source h5.** `merge_lerobot` → `export_lerobot.verify` re-reads the source h5 of every exported episode. Never delete the h5 before the final merge.
- **Total targets overshoot.** In-flight batches push a total target past the per-drawer quota, so stop on the quotas instead.
- **One uploader on the Mac at a time.** Each `hf_upload_pipe` keeps up to three portions on disk.
- **No Xet.** Use `HF_HUB_DISABLE_XET=1`: hub 1.8's Xet upload hung on the first commit.
