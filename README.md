# MIKASA-Robo-MMM

Benchmark code split from ManiSkill. ManiSkill stays in `../ManiSkill` and is installed as an editable `uv` dependency.

```bash
cd ~/Projects/MIKASA-Robo-MMM
uv sync
uv run python -c 'import mani_skill; import my_scenes'
uv run python -m planners.myrobocasa_takeitback_planner --help
```

Import `my_scenes` before `gym.make(...)`; it registers MIKASA environments and `ds_fetch`.
RoboCasa assets remain in ManiSkill's `MS_ASSET_DIR` (download with ManiSkill's asset tool).
