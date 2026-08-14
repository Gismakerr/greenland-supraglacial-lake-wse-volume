from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORK_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = WORK_ROOT / "result" / "08_area_wse_fits" / "representative_panel_rows"
INPUT_DIR_02_2 = Path(os.environ.get("LAKE_PANEL_02_2_INPUT_DIR", str(INPUT_ROOT / "02_2")))
INPUT_DIR_02_1 = Path(os.environ.get("LAKE_PANEL_02_1_INPUT_DIR", str(INPUT_ROOT / "02_1")))
OUTPUT_DIR = Path(
    os.environ.get(
        "LAKE_PANEL_STACK_OUTPUT_DIR",
        str(WORK_ROOT / "result" / "12_paper_figures" / "figures"),
    )
)

ROW_FILES = [
    ("02_2", "lake_1_branch_1_premerge_composed.png"),
    ("02_2", "lake_1_branch_2_premerge_composed.png"),
    ("02_2", "lake_1_merged_postmerge_composed.png"),
    ("02_2", "lake_2_branch_1_premerge_composed.png"),
    ("02_2", "lake_2_branch_2_premerge_composed.png"),
    ("02_2", "lake_2_merged_postmerge_composed.png"),
    ("02_1", "lake_38_original_composed.png"),
    ("02_1", "lake_4_branch_2_direct_composed.png"),
    ("02_1", "lake_70_branch_1_direct_composed.png"),
    ("02_1", "lake_30_branch_1_direct_composed.png"),
]

# Bottom labels are placed in the composed canvas coordinate system.  Keeping
# them here means all row image frames remain byte-for-byte the same size.
BOTTOM_LABEL_HEIGHT = 210
BOTTOM_LABEL_Y = 55
BOTTOM_LABEL_DATE_X = float(os.environ.get("LAKE_PANEL_DATE_LABEL_X", "0.31"))
BOTTOM_LABEL_AREA_X = float(os.environ.get("LAKE_PANEL_AREA_LABEL_X", "0.69"))
BOTTOM_LABEL_WATER_COVERAGE_X = float(os.environ.get("LAKE_PANEL_WATER_COVERAGE_LABEL_X", "0.90"))
BOTTOM_LABELS = [
    ("Date", BOTTOM_LABEL_DATE_X, "#111111"),
    ("Area", BOTTOM_LABEL_AREA_X, "#2F6DF6"),
    ("Water Coverage count", BOTTOM_LABEL_WATER_COVERAGE_X, "#111111"),
]
BOTTOM_LABEL_FONT_SIZE = 80
BOTTOM_LABEL_FONT = Path("C:/Windows/Fonts/arialbd.ttf")


def content_bbox(image: Image.Image, threshold: int = 248) -> tuple[int, int, int, int]:
    """Return bounding box of visible, non-white content."""

    rgb = np.asarray(image.convert("RGB"))
    non_white = np.any(rgb < threshold, axis=2)
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, image.width, image.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def union_bbox(boxes: list[tuple[int, int, int, int]], size: tuple[int, int], pad: int = 28) -> tuple[int, int, int, int]:
    """Use one shared crop box so all stacked rows keep identical column alignment."""

    width, height = size
    left = max(0, min(box[0] for box in boxes) - pad)
    top = max(0, min(box[1] for box in boxes) - pad)
    right = min(width, max(box[2] for box in boxes) + pad)
    bottom = min(height, max(box[3] for box in boxes) + pad)
    return (left, top, right, bottom)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_dirs = {"02_2": INPUT_DIR_02_2, "02_1": INPUT_DIR_02_1}
    paths = [base_dirs[group] / name for group, name in ROW_FILES]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing row images: " + ", ".join(str(path) for path in missing))

    images = [Image.open(path).convert("RGB") for path in paths]
    size = images[0].size
    if any(image.size != size for image in images):
        raise ValueError("All row images must have the same size before stacking.")

    crop_box = union_bbox([content_bbox(image) for image in images], size)
    cropped = [image.crop(crop_box) for image in images]
    row_width = cropped[0].width
    total_height = sum(image.height for image in cropped)

    stacked = Image.new("RGB", (row_width, total_height + BOTTOM_LABEL_HEIGHT), "white")
    y = 0
    for image in cropped:
        stacked.paste(image, (0, y))
        y += image.height

    font = ImageFont.truetype(str(BOTTOM_LABEL_FONT), BOTTOM_LABEL_FONT_SIZE)
    draw = ImageDraw.Draw(stacked)
    for label, x_fraction, color in BOTTOM_LABELS:
        draw.text(
            (round(row_width * x_fraction), total_height + BOTTOM_LABEL_Y),
            label,
            font=font,
            fill=color,
            anchor="mm",
        )

    out_png = OUTPUT_DIR / "area_wse_representative_stacked.png"
    out_tif = OUTPUT_DIR / "area_wse_representative_stacked.tif"
    stacked.save(out_png, dpi=(300, 300))
    stacked.save(out_tif, dpi=(300, 300), compression="tiff_lzw")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_tif}")
    print(f"Crop box: {crop_box}; output size: {stacked.size}")


if __name__ == "__main__":
    main()
