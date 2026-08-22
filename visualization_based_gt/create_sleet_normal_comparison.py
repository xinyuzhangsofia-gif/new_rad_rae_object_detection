"""Create the two-panel Sleet/Normal comparison image."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SLEET_PATH = (
    ROOT
    / "generated/best_weather_epoch15"
    / "sleet_group4_epoch015_seq50_frame00344_gt_pred_radar_no_text.png"
)
NORMAL_PATH = (
    ROOT
    / "generated/pictures/sequence_11/no_title_green_gt_red_prediction"
    / "sequence_11_frame_000000_00034_00001_gt_pred.png"
)
OUTPUT_PATH = (
    ROOT
    / "generated/best_weather_epoch15"
    / "sleet_normal_comparison_left_to_right.png"
)
FONT_PATH = Path(
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
)


def main():
    panels = [Image.open(path).convert("RGB") for path in (SLEET_PATH, NORMAL_PATH)]
    if panels[0].size != panels[1].size:
        raise ValueError(
            f"Panel dimensions differ: {panels[0].size} and {panels[1].size}"
        )

    panel_width, panel_height = panels[0].size
    footer_height = 100
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + footer_height),
        color="black",
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel_width, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT_PATH), 54)
    for index, label in enumerate(("Sleet", "Normal")):
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        center_x = index * panel_width + panel_width // 2
        x = center_x - text_width // 2
        y = panel_height + (footer_height - text_height) // 2 - bounds[1]
        draw.text((x, y), label, fill="white", font=font)

    canvas.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
