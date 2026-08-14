from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result" / "11_slc_phase_panels"
PANEL_DIR = RESULT_DIR / "selected_panels"
FIGURE_DIR = RESULT_DIR / "figures"
MANIFEST_CSV = RESULT_DIR / "selected_panel_manifest.csv"

EXPECTED_DATES = [
    "2024-06-28", "2024-07-02", "2024-07-08", "2024-07-13", "2024-07-16", "2024-07-23",
    "2024-07-26", "2024-07-27", "2024-07-28", "2024-08-03", "2024-08-09", "2024-08-14",
]
OUTPUTS = {
    1: (FIGURE_DIR / "lake_1_slc_phase_part1.png", FIGURE_DIR / "lake_1_slc_phase_part1.tif"),
    2: (FIGURE_DIR / "lake_1_slc_phase_part2.png", FIGURE_DIR / "lake_1_slc_phase_part2.tif"),
}

FIG_WIDTH_IN = 8.27
FIG_HEIGHT_IN = 14.02
FIG_DPI = 300
ROW_LEFT = 0.008
ROW_WIDTH = 0.984
ROW_BOTTOM = 0.122
ROW_TOP = 0.965
TITLE_Y = 0.987
TITLE_X = [0.145, 0.385, 0.625, 0.865]
TITLE_TEXT = ["S2+SWOT PIXC", "Interferogram Phase", "Sigma0", "Coherence"]

WSE_CMAP = LinearSegmentedColormap.from_list(
    "wse", ["#f4edc3", "#f6c35e", "#f16b3a", "#df1744", "#98005b"]
)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "phase", ["#ff0000", "#ff9b00", "#ffff00", "#00d96f", "#00e7ff", "#004dff", "#9b00ff", "#ff00c8"]
)
SIGMA0_CMAP = LinearSegmentedColormap.from_list("sigma0", ["#000000", "#ffffff"])
COHERENCE_CMAP = LinearSegmentedColormap.from_list(
    "coherence", ["#000000", "#002fff", "#00dfff", "#f1ff00", "#ff3800", "#970000"]
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_CSV, encoding="utf-8-sig")
    required = {"lake_id", "date", "part", "row_order", "panel_file", "sha256", "qc_status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    manifest["date"] = pd.to_datetime(manifest["date"], errors="raise").dt.strftime("%Y-%m-%d")
    manifest["part"] = pd.to_numeric(manifest["part"], errors="raise").astype(int)
    manifest["row_order"] = pd.to_numeric(manifest["row_order"], errors="raise").astype(int)
    return manifest.sort_values(["part", "row_order"]).reset_index(drop=True)


def add_colorbar(fig, rect, cmap, vmin, vmax, ticks, ticklabels, label):
    axis = fig.add_axes(rect)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        cax=axis,
        orientation="horizontal",
    )
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(ticklabels)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
    colorbar.outline.set_linewidth(0.65)
    colorbar.set_label(label, fontsize=8, fontweight="bold", labelpad=2)


def render_part(manifest: pd.DataFrame, part: int) -> None:
    rows = manifest.loc[manifest["part"] == part].sort_values("row_order")
    if len(rows) != 6:
        raise ValueError(f"Part {part} must contain six rows, found {len(rows)}")

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI, facecolor="white")
    for x, title in zip(TITLE_X, TITLE_TEXT):
        fig.text(x, TITLE_Y, title, ha="center", va="top", fontsize=11, fontweight="bold")

    row_height = (ROW_TOP - ROW_BOTTOM) / 6.0
    for index, row in enumerate(rows.itertuples(index=False)):
        panel_path = PANEL_DIR / str(row.panel_file)
        with Image.open(panel_path) as image:
            rgb = image.convert("RGB")
            panel = rgb.copy()
        y = ROW_TOP - (index + 1) * row_height
        axis = fig.add_axes([ROW_LEFT, y, ROW_WIDTH, row_height])
        axis.imshow(panel, interpolation="lanczos", aspect="auto")
        axis.set_axis_off()

    add_colorbar(
        fig, [0.040, 0.077, 0.440, 0.012], WSE_CMAP, 672.0, 682.0,
        [672.0, 677.0, 682.0], ["672", "677", "682"], "WSE (m)",
    )
    add_colorbar(
        fig, [0.520, 0.077, 0.440, 0.012], PHASE_CMAP, -3.1415926536, 3.1415926536,
        [-3.1415926536, -1.5707963268, 0.0, 1.5707963268, 3.1415926536],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        "Interferogram phase (rad)",
    )
    add_colorbar(
        fig, [0.040, 0.025, 0.440, 0.012], SIGMA0_CMAP, 40.0, 75.0,
        [40, 45, 50, 55, 60, 65, 70, 75], [str(x) for x in range(40, 80, 5)], "Sigma0 (dB)",
    )
    add_colorbar(
        fig, [0.520, 0.025, 0.440, 0.012], COHERENCE_CMAP, 0.0, 1.0,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f"{x:.1f}" for x in [0, .2, .4, .6, .8, 1]], "Coherence",
    )

    png_path, tif_path = OUTPUTS[part]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=FIG_DPI, facecolor="white")
    fig.savefig(tif_path, dpi=FIG_DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def validate() -> None:
    manifest = load_manifest()
    if len(manifest) != 12 or manifest["lake_id"].astype(int).unique().tolist() != [1]:
        raise ValueError("The public phase-panel cohort must be exactly 12 Lake 1 rows.")
    if manifest["date"].tolist() != EXPECTED_DATES:
        raise ValueError("Selected Lake 1 dates or ordering changed.")
    if manifest.duplicated(["date"]).any():
        raise ValueError("Duplicate selected dates found.")
    if set(manifest["qc_status"].astype(str)) != {"ok"}:
        raise ValueError("A selected panel is not QC status 'ok'.")
    for row in manifest.itertuples(index=False):
        panel_path = PANEL_DIR / str(row.panel_file)
        if not panel_path.exists():
            raise FileNotFoundError(panel_path)
        if sha256(panel_path) != str(row.sha256).upper():
            raise ValueError(f"Panel hash changed: {panel_path.name}")
        with Image.open(panel_path) as image:
            if image.width != 5166 or image.height != 1378:
                raise ValueError(f"Unexpected panel dimensions: {panel_path.name}")
    for png_path, tif_path in OUTPUTS.values():
        for output in [png_path, tif_path]:
            if not output.exists() or output.stat().st_size < 500_000:
                raise ValueError(f"Missing or incomplete figure: {output}")
            with Image.open(output) as image:
                if image.width != 2481 or image.height != 4206:
                    raise ValueError(f"Unexpected final figure dimensions: {output.name}")
    print("VALID: Lake 1 only; 12 frozen panels; two 6-row PNG/TIFF phase figures.")


def run() -> None:
    manifest = load_manifest()
    for part in [1, 2]:
        render_part(manifest, part)
    validate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose the two publication Lake 1 SLC phase image plates.")
    parser.add_argument("--run", action="store_true", help="Render both figures; otherwise validate outputs.")
    args = parser.parse_args()
    run() if args.run else validate()


if __name__ == "__main__":
    main()
