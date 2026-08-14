from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "result" / "_test_lake1_vertical_legends"
OUTPUT_PNG = OUTPUT_DIR / "lake1_slc_phase_part2_legends_vertical.png"
OUTPUT_TIF = OUTPUT_DIR / "lake1_slc_phase_part2_legends_vertical.tif"

FIG_WIDTH_IN = 5.0
FIG_HEIGHT_IN = 6.0
FIG_DPI = 300
LEGEND_TITLE_FONT_SIZE = 12
LEGEND_TICK_FONT_SIZE = 12

WSE_CMAP = LinearSegmentedColormap.from_list(
    "wse", ["#f4edc3", "#f6c35e", "#f16b3a", "#df1744", "#98005b"]
)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "phase",
    ["#ff0000", "#ff9b00", "#ffff00", "#00d96f", "#00e7ff", "#004dff", "#9b00ff", "#ff00c8"],
)
SIGMA0_CMAP = LinearSegmentedColormap.from_list("sigma0", ["#000000", "#ffffff"])
COHERENCE_CMAP = LinearSegmentedColormap.from_list(
    "coherence", ["#000000", "#002fff", "#00dfff", "#f1ff00", "#ff3800", "#970000"]
)

LEGENDS = [
    {
        "title": "WSE (m)",
        "cmap": WSE_CMAP,
        "vmin": 672.0,
        "vmax": 682.0,
        "ticks": [672.0, 674.5, 677.0, 679.5, 682.0],
        "ticklabels": ["672", "674.5", "677", "679.5", "682"],
    },
    {
        "title": "Interferogram phase (rad)",
        "cmap": PHASE_CMAP,
        "vmin": -3.1415926536,
        "vmax": 3.1415926536,
        "ticks": [-3.1415926536, -1.5707963268, 0.0, 1.5707963268, 3.1415926536],
        "ticklabels": [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    },
    {
        "title": "Sigma0 (dB)",
        "cmap": SIGMA0_CMAP,
        "vmin": 40.0,
        "vmax": 75.0,
        "ticks": [40, 45, 50, 55, 60, 65, 70, 75],
        "ticklabels": [str(value) for value in range(40, 80, 5)],
    },
    {
        "title": "Coherence",
        "cmap": COHERENCE_CMAP,
        "vmin": 0.0,
        "vmax": 1.0,
        "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "ticklabels": [f"{value:.1f}" for value in [0, 0.2, 0.4, 0.6, 0.8, 1.0]],
    },
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def render() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI, facecolor="white")

    x_positions = [0.10, 0.35, 0.60, 0.85]
    for x_position, legend in zip(x_positions, LEGENDS):
        axis = fig.add_axes([x_position, 0.08, 0.045, 0.84])
        colorbar = fig.colorbar(
            ScalarMappable(
                norm=Normalize(vmin=legend["vmin"], vmax=legend["vmax"]),
                cmap=legend["cmap"],
            ),
            cax=axis,
            orientation="vertical",
        )
        colorbar.set_ticks(legend["ticks"])
        colorbar.set_ticklabels(legend["ticklabels"])
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.tick_params(
            labelsize=LEGEND_TICK_FONT_SIZE,
            length=3,
            width=0.8,
            pad=2,
            direction="out",
        )
        colorbar.outline.set_linewidth(0.8)
        fig.text(
            x_position - 0.030,
            0.50,
            legend["title"],
            ha="center",
            va="center",
            rotation=90,
            rotation_mode="anchor",
            fontsize=LEGEND_TITLE_FONT_SIZE,
            fontweight="bold",
        )

    fig.savefig(OUTPUT_PNG, dpi=FIG_DPI, facecolor="white")
    fig.savefig(
        OUTPUT_TIF,
        dpi=FIG_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(OUTPUT_PNG)
    print(OUTPUT_TIF)


if __name__ == "__main__":
    render()
