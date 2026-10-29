"""Writes the CabinetSearch home rug: home_rug.obj (a flat unit disk in the xy plane, normal +z, UVs
from xy) with home_rug.mtl and its texture home_rug.png (a round rug: fringe, navy border with cream
dots, red field with a gold ring and petals, cream medallion with a navy star, a woven grain).
The scene scales the disk to `home_radius`. Deterministic: run it again to get the same files.

    python my_scenes/assets/home_rug/make_home_rug.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
N_TEX, N_SEG = 512, 96

CREAM, NAVY, RED, GOLD, DARK = (np.array(c, dtype=float) for c in (
    (226, 214, 188), (34, 46, 92), (150, 30, 36), (206, 160, 62), (90, 20, 24)))


def texture() -> np.ndarray:
    ax = (np.arange(N_TEX) + 0.5) / N_TEX * 2 - 1
    x, y = np.meshgrid(ax, -ax)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    img = np.empty((N_TEX, N_TEX, 3))
    img[:] = RED
    petals = (np.cos(8 * th) * 0.5 + 0.5) > 0.82
    img[(r > 0.30) & (r < 0.52) & petals] = DARK
    img[(r > 0.55) & (r < 0.60)] = GOLD
    img[(r > 0.62) & (r < 0.78) & ((np.cos(16 * th) * 0.5 + 0.5) > 0.9)] = GOLD
    img[(r >= 0.80) & (r < 0.95)] = NAVY
    dots = (np.cos(36 * th) * 0.5 + 0.5 > 0.8) & (np.abs(r - 0.875) < 0.025)
    img[dots] = CREAM
    img[r >= 0.95] = CREAM
    img[(r >= 0.97) & ((np.cos(140 * th) * 0.5 + 0.5) > 0.5)] = CREAM * 0.8
    img[r < 0.25] = CREAM
    star = r < 0.2 * (0.65 + 0.35 * np.cos(6 * th))
    img[star] = NAVY
    img[r < 0.05] = GOLD
    weave = 0.06 * np.sin(np.pi * N_TEX / 4 * x) * np.sin(np.pi * N_TEX / 4 * y)
    grain = np.random.default_rng(0).normal(0.0, 0.035, (N_TEX, N_TEX))
    img = img * (1.0 + weave + grain)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def mesh() -> str:
    lines = ["# CabinetSearch home rug: unit disk, normal +z (make_home_rug.py)", "mtllib home_rug.mtl", "o home_rug",
             "v 0 0 0", "vt 0.5 0.5"]
    for k in range(N_SEG):
        a = 2 * np.pi * k / N_SEG
        c, s = np.cos(a), np.sin(a)
        lines += [f"v {c:.6f} {s:.6f} 0", f"vt {0.5 + 0.5 * c:.6f} {0.5 + 0.5 * s:.6f}"]
    lines += ["vn 0 0 1", "usemtl home_rug"]
    for k in range(N_SEG):
        a, b = 2 + k, 2 + (k + 1) % N_SEG
        lines.append(f"f 1/1/1 {a}/{a}/1 {b}/{b}/1")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Image.fromarray(texture()).save(HERE / "home_rug.png")
    (HERE / "home_rug.obj").write_text(mesh())
    (HERE / "home_rug.mtl").write_text("newmtl home_rug\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 1\nd 1\nillum 1\n"
                                       "map_Kd home_rug.png\n")
    print("wrote", sorted(p.name for p in HERE.glob("home_rug.*")))
