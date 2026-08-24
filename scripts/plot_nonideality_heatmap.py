from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/crosssim-adc8-mixed-lumped-fulltest"
OUTPUT = RESULTS / "accuracy_heatmap.png"
ERRORS = (0.0, 0.01, 0.03)
NOISES = (0.0, 0.03, 0.05)
MIN_ACCURACY = 77.0
MAX_ACCURACY = 93.0


def label(value: float) -> str:
    return f"{value:g}"


def color(accuracy: float) -> tuple[int, int, int]:
    # Low accuracy is red; high accuracy is green.
    stops = (
        (77.0, (178, 24, 43)),
        (82.0, (241, 111, 45)),
        (87.0, (255, 215, 91)),
        (90.0, (146, 202, 92)),
        (93.0, (0, 104, 55)),
    )
    for (low, low_color), (high, high_color) in zip(stops, stops[1:]):
        if accuracy <= high:
            fraction = max(0.0, (accuracy - low) / (high - low))
            return tuple(int(a + (b - a) * fraction) for a, b in zip(low_color, high_color))
    return stops[-1][1]


def vertical_text(image: Image.Image, position: tuple[int, int], text: str, font: ImageFont.FreeTypeFont) -> None:
    box = font.getbbox(text)
    label = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((4, 4), text, fill="black", font=font)
    image.alpha_composite(label.rotate(90, expand=True), position)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")
        if bold else ("/System/Library/Fonts/Supplemental/Arial.ttf",)
    ) + ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    values: dict[tuple[float, float], float] = {}
    digital_accuracy = None
    for error in ERRORS:
        for noise in NOISES:
            path = RESULTS / f"error{label(error)}_noise{label(noise)}" / "crosssim_results.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            values[error, noise] = report["crosssim_accuracy_mean"]
            digital_accuracy = report["digital_accuracy"]

    regular = font(24)
    bold = font(28, bold=True)
    title = font(32, bold=True)
    cell = 225
    left, top = 270, 220
    image = Image.new("RGBA", (1180, 970), "white")
    draw = ImageDraw.Draw(image)

    title_text = "ACIM Accuracy: Programming Error x Lumped Read Noise"
    box = draw.textbbox((0, 0), title_text, font=title)
    draw.text(((1180 - (box[2] - box[0])) / 2, 35), title_text, fill="black", font=title)
    draw.text((left + 225, 125), "Lumped read noise", fill="black", font=bold)
    vertical_text(image, (32, top + 270), "Programming error", bold)

    for column, noise in enumerate(NOISES):
        text = label(noise)
        box = draw.textbbox((0, 0), text, font=bold)
        draw.text((left + column * cell + (cell - (box[2] - box[0])) / 2, top - 38), text, fill="black", font=bold)
    for row, error in enumerate(ERRORS):
        text = label(error)
        box = draw.textbbox((0, 0), text, font=bold)
        draw.text((left - 25 - (box[2] - box[0]), top + row * cell + (cell - (box[3] - box[1])) / 2), text, fill="black", font=bold)
        for column, noise in enumerate(NOISES):
            accuracy = values[error, noise]
            x, y = left + column * cell, top + row * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color(accuracy), outline="white", width=4)
            text = f"{accuracy:.2f}%"
            box = draw.textbbox((0, 0), text, font=bold)
            draw.text((x + (cell - (box[2] - box[0])) / 2, y + 96), text, fill="black", font=bold)

    # Standard heatmap colorbar: low accuracy at bottom, high at top.
    bar_x, bar_y, bar_width, bar_height = left + cell * 3 + 45, top, 35, cell * 3
    for offset in range(bar_height):
        accuracy = MAX_ACCURACY - (MAX_ACCURACY - MIN_ACCURACY) * offset / (bar_height - 1)
        draw.line((bar_x, bar_y + offset, bar_x + bar_width, bar_y + offset), fill=color(accuracy))
    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height), outline="#20242a", width=1)
    for accuracy in (77, 80, 83, 86, 89, 92):
        y = bar_y + (MAX_ACCURACY - accuracy) / (MAX_ACCURACY - MIN_ACCURACY) * bar_height
        draw.line((bar_x + bar_width, y, bar_x + bar_width + 8, y), fill="#20242a", width=1)
        draw.text((bar_x + bar_width + 14, y - 12), f"{accuracy}%", fill="#20242a", font=regular)
    vertical_text(image, (bar_x + 84, bar_y + 240), "Accuracy (%)", regular)

    image.convert("RGB").save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
