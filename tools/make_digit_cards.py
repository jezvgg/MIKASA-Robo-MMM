"""The SeasonDish digit cards: my_scenes/assets/season_dish/digit_<n>.png for n = 1..9.

A white card with a thin dark border and one bold black digit, square like the condiment pictures, so the
fridge cue reads as a picture with its number. The font is matplotlib's DejaVu Sans Bold (shipped with every
matplotlib), so the cards can be redrawn identically anywhere.

    python tools/make_digit_cards.py
"""
import os
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

SIZE = 512
OUT = Path(__file__).resolve().parents[1] / "my_scenes/assets/season_dish"
FONT = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"


def card(digit: int) -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)
    border = SIZE // 32
    draw.rectangle([border // 2, border // 2, SIZE - 1 - border // 2, SIZE - 1 - border // 2],
                   outline=(40, 40, 40), width=border)
    font = ImageFont.truetype(str(FONT), int(SIZE * 0.78))
    left, top, right, bottom = draw.textbbox((0, 0), str(digit), font=font)
    x = (SIZE - (right - left)) / 2 - left
    y = (SIZE - (bottom - top)) / 2 - top
    draw.text((x, y), str(digit), fill="black", font=font)
    return image


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n in range(1, 10):
        path = OUT / f"digit_{n}.png"
        card(n).save(path, optimize=True)
        print(path.relative_to(Path.cwd()) if str(path).startswith(os.getcwd()) else path)
