# Export batches within a storage budget

Use the pinned LeRobot export environment (currently0.4.3), from the task worktree with `PYTHONPATH=.`. These data tools do not alter simulation commands, robot assets, controllers or recorded states.

1. Export a completed candidate pool, keeping every source attempt in its metadata. Explicitly select only qualified episodes. Test batches below1000episodes must include `test` in their output directory name.
2. Run `python -m utils.collection.retention prepare DATASET`. This checks the genuine LeRobot reader against all numerical rows and every rendered RGB frame, copies every non-image H5 field exactly, and records a SHA256 digest of all decoded RGB pixels for each episode/camera. The numerical copies include full20Hz actions, qpos, environment states, rewards and terminal flags. Preparation has a pending manifest for recovery after interruption.
3. Run `python -m utils.collection.retention prune DATASET` only when the intermediate RGB copies are no longer needed by other exports. Prepare those exports too if they reference the same renders. The command verifies again, removes only the explicitly recorded successful `CAMPAIGN/rgb/SEED/trajectory.h5` files, and verifies the retained dataset. All JSON, original physical recordings, numerical copies and LeRobot videos remain. Subsequent verification compares every decoded pixel with the previously render-verified digest; it does not claim to render the scene again.
4. Merge completed, verified batches with `python -m utils.collection.merge_lerobot --inputs DATASET_A DATASET_B --output MERGED_DATASET --repo-id LOCAL_ID`. It preserves the actual `source_runs`, every attempted source seed and per-episode source roots. It rejects different runtime/profile signatures, duplicated episodes, overlapping candidate pools, inconsistent outcomes and truncated/failed episodes. Prepare retention for every input before mixing retained and unretained storage modes.

Use `uv run --no-project --python EXPORT_PYTHON` before the commands above; `python` examples identify modules, not a different package manager. The merge checks every numerical row and every decoded RGB frame, including after render copies have been pruned. It never uploads a dataset.

Use `export_lerobot --seeds SEED_A SEED_B` to select qualified episodes while retaining all campaign outcomes. Exports bind the recorded source manifest to actual Git blobs for the robot, environment, planner and converter. A merged dataset also preserves each episode's original `conversion_id` and `conversion_runs`, as well as the merger's own code binding. Production exports and merges with1000or more episodes require complete commit bindings for every contributing converter. Small test batches with missing bindings remain A2 failures.

Retention and successful format checks do not waive any of the41requirements. In particular, human E5, the future training revision/G1, 1000qualified training episodes and100independent solvable validation seeds still need their own evidence. Correctly named small test datasets report H3 as PARTIAL.

## Merge without duplicating video storage

Add `--link-videos` to the merge command when all inputs and the output are on the same filesystem. The merger creates a hardlink for each existing video file, preserves its episode timestamps, and rewrites only the numerical parquet files and metadata with explicit file mappings. It never concatenates, re-encodes, or appends to linked videos. Cross-filesystem linking fails without falling back to a large copy. Full LeRobot numerical and RGB verification still runs for every input and the output. The aggregation metadata records the linked files and their identities.

Treat linked videos as immutable: editing their bytes in place would also affect the input shard. Deleting one video pathname leaves its other hardlink intact. Do not remove entire input directories: retained numerical H5 sources may still live there and be referenced by the merged metadata. Keep those numerical sources, original recordings, and campaign metadata. This option removes the second video payload at merge time; reserve space for numerical parquet rewriting, a working collection batch, and final checks.

## Compact encoding with a measured quality limit

`export_lerobot --video-profile compact` uses H264/YUV444P with GOP12. It first tries CRF16 and compares every decoded frame with the lossless source PNGs. If the episode/camera mean pixel error exceeds2/255, it retries CRF12 and then CRF0. Invalid dimensions or missing/extra frames fail immediately. Source PNGs are removed only after an acceptable video has been encoded; final LeRobot verification still checks every RGB frame against the original render.

Each episode's `video_encoding` records the actual CRF/GOP, quality measurements, attempted settings, and SHA256 of its complete decoded RGB sequence. Final verification checks that concatenation, merging and retained storage preserve this sequence. `conversion_runs` preserves the requested video policy as well as the original converter commit, so shards encoded with different settings remain attributable. The default `reference` profile retains CRF12/GOP2 for compatibility with earlier exports; actual camera settings are recorded for it too.

A smaller file is not evidence of adequate quality. The compact profile enforces the same E4 limit on every actual export; synthetic codec checks and small real-data probes do not establish human E5, full-dataset qualification, or a guaranteed storage ratio.
