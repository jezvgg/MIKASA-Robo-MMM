"""Build one task's RLDS dataset: build_rlds.py <fetch_samedrawer|...> [--data-dir DIR].

TFDS resolves a builder's resources from its module, so the builders live in `fetch_rlds`
and are run from here, not as `__main__`.
"""
import argparse
from pathlib import Path

from fetch_rlds import BUILDERS

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("name", choices=sorted(BUILDERS))
ap.add_argument("--data-dir", type=Path, default=Path.home() / "datasets" / "rlds")
args = ap.parse_args()
builder = BUILDERS[args.name](data_dir=str(args.data_dir))
builder.download_and_prepare()
print(builder.info)
