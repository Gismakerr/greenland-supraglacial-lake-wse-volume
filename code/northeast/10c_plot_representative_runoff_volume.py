from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = ROOT
RESULT_ROOT = ROOT / "result" / "10_lake_volume_change"
INPUT_ROOT = RESULT_ROOT / "runoff_figure_inputs"
IN_CSV = INPUT_ROOT / "representative_8_volume_timeseries.csv"
OUT_DIR = RESULT_ROOT / "runoff_volume_combined_rate_labels"
STACKED_PNG = OUT_DIR / "representative_6_dual_axis_runoff_volume_stacked.png"
STACKED_TIF = OUT_DIR / "representative_6_dual_axis_runoff_volume_stacked.tif"
FIT_LABEL_JSON = INPUT_ROOT / "fit_label_layouts.json"
LEGACY_FIT_LABEL_JSON = INPUT_ROOT / "fit_label_layouts.json"
NO_BUFFER_BASIN_CSV = INPUT_ROOT / "six_lake_buffer_sensitivity_10m.csv"

DPI = 300
# Keep the 07_3 look, but slightly taller so the row is less flattened.
FIGSIZE = (17.0, 4.10)

FONT_FAMILY = "Arial"
LABEL_SIZE = 20
AXIS_LABEL_SIZE = 16
LEGEND_SIZE = 14
TICK_SIZE = 18
SPINE_WIDTH = 1.35
TICK_WIDTH = 1.35
TICK_LENGTH = 8

BRANCH1_FILL = "#bfc4c8"
BRANCH2_FILL = "#969ca1"
TOTAL_LINE = "#1f1f1f"
ERR_COLOR = "#404040"
DEFAULT_FIT_LINE_COLOR = "#111111"
RUNOFF_RED = "#df3f3f"
FIT_LINE_X_SHIFT_DAYS = {("lake30_swot_only", "drain"): 0.5}
FIT_ANNOTATION_SIZE = 18.0
FIT_ANNOTATION_HALF_SPACING = 0.85

RUNOFF_MM_PER_1E6M3_PER_KM2 = 1000.0
LEFT_LABEL_TEXT = r"$10^6$ m$^3$"
RIGHT_UNIT_TEXT = "mm"
LEFT_AXIS_LABEL = "Water volume change"
RIGHT_AXIS_LABEL = "Cumulative runoff"
RATE_LABEL_STYLE = "parallel"

X_MIN = pd.Timestamp("2024-06-22")
X_MAX = pd.Timestamp("2024-08-17")
X_TICKS = pd.to_datetime(
    ["2024-06-25", "2024-07-05", "2024-07-15", "2024-07-25", "2024-08-05", "2024-08-15"]
)

# Final display windows requested for the representative figure.  The cutoff
# date is retained; observations after it are neither drawn nor counted in N.
PLOT_END_DATES = {
    38: pd.Timestamp("2024-07-15"),  # displayed as Lake 3
    70: pd.Timestamp("2024-07-15"),  # displayed as Lake 5
}

LAKES = [
    {
        "lake_id": 1,
        "title": "Lake 1",
        "row_keys": ("lake1_1", "lake1_2"),
        "merge_date": pd.Timestamp("2024-07-12"),
        "out_name": "lake1_dual_axis",
        "fit_key": "lake1_swot_only",
        "plot_mode": "stacked",
        "legend_labels": ("Lake1_1", "Lake1_2", "Merged"),
    },
    {
        "lake_id": 2,
        "title": "Lake 2",
        "row_keys": ("lake2_1", "lake2_2"),
        "merge_date": pd.Timestamp("2024-07-31"),
        "out_name": "lake2_dual_axis",
        "fit_key": "lake2_swot_only",
        "plot_mode": "stacked",
        "legend_labels": ("Lake2_1", "Lake2_2", "Merged"),
    },
    {
        "lake_id": 38,
        "title": "Lake 3",
        "row_keys": ("lake38",),
        "out_name": "lake38_dual_axis",
        "fit_key": "lake38_swot_only",
        "plot_mode": "single",
    },
    {
        "lake_id": 4,
        "title": "Lake 4",
        "row_keys": ("lake4",),
        "out_name": "lake4_dual_axis",
        "fit_key": "lake4_swot_only",
        "plot_mode": "single",
    },
    {
        "lake_id": 70,
        "title": "Lake 5",
        "row_keys": ("lake70",),
        "out_name": "lake70_dual_axis",
        "fit_key": "lake70_swot_only",
        "plot_mode": "single",
    },
    {
        "lake_id": 30,
        "title": "Lake 6",
        "row_keys": ("lake30",),
        "out_name": "lake30_dual_axis",
        "fit_key": "lake30_swot_only",
        "plot_mode": "single",
    },
]

FIT_WINDOWS = {
    "lake1_swot_only": {"inflow": ("2024-06-26", "2024-07-28"), "drain": ("2024-07-28", "2024-08-14")},
    "lake2_swot_only": {
        "inflow_1": ("2024-06-26", "2024-07-07"),
        "inflow_2": ("2024-07-07", "2024-08-06"),
    },
    "lake38_swot_only": {"inflow": ("2024-06-23", "2024-07-07"), "drain": ("2024-07-07", "2024-07-13")},
    "lake4_swot_only": {"inflow": ("2024-06-28", "2024-08-10")},
    "lake70_swot_only": {"inflow": ("2024-07-03", "2024-07-12"), "drain": ("2024-07-12", "2024-07-15")},
    "lake30_swot_only": {"inflow": ("2024-07-05", "2024-07-20"), "drain": ("2024-07-20", "2024-07-21")},
    "lake1_dual_axis": {"inflow": ("2024-06-26", "2024-07-28"), "drain": ("2024-07-28", "2024-08-14")},
    "lake2_dual_axis": {
        "inflow_1": ("2024-06-26", "2024-07-07"),
        "inflow_2": ("2024-07-07", "2024-08-06"),
    },
    "lake38_dual_axis": {"inflow": ("2024-06-23", "2024-07-07"), "drain": ("2024-07-07", "2024-07-13")},
    "lake4_dual_axis": {"inflow": ("2024-06-28", "2024-08-10")},
    "lake70_dual_axis": {"inflow": ("2024-07-03", "2024-07-12"), "drain": ("2024-07-12", "2024-07-15")},
    "lake30_dual_axis": {"inflow": ("2024-07-05", "2024-07-20"), "drain": ("2024-07-20", "2024-07-21")},
}

FIT_LABEL_LAYOUTS_DEFAULT = {
    "lake1_dual_axis": {
        "inflow": {"anchor_frac": 0.52, "normal_px": 34.0, "tangent_px": 0.0, "fontsize": 20, "rotation_deg": None},
        "drain": {"anchor_frac": 0.58, "normal_px": 34.0, "tangent_px": 0.0, "fontsize": 20, "rotation_deg": None},
    },
    "lake2_dual_axis": {
        "inflow_1": {"anchor_frac": 0.48, "normal_px": 44.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
        "inflow_2": {"anchor_frac": 0.56, "normal_px": 44.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
    },
    "lake2_swot_only": {
        "inflow_1": {"anchor_frac": 0.48, "normal_px": 105.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
        "inflow_2": {"anchor_frac": 0.58, "normal_px": 115.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
    },
    "lake38_dual_axis": {
        "inflow": {"anchor_frac": 0.40, "normal_px": 28.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
        "drain": {"anchor_frac": 0.44, "normal_px": 26.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
    },
    "lake4_dual_axis": {
        "inflow": {"anchor_frac": 0.52, "normal_px": 28.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
    },
    "lake70_dual_axis": {
        "inflow": {"anchor_frac": 0.42, "normal_px": 24.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
        "drain": {"anchor_frac": 0.48, "normal_px": 24.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
    },
    "lake30_dual_axis": {
        "inflow": {"anchor_frac": 0.46, "normal_px": 24.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
        "drain": {"anchor_frac": 0.44, "normal_px": 24.0, "tangent_px": 0.0, "fontsize": 18, "rotation_deg": None},
    },
}


plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "axes.linewidth": SPINE_WIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
    }
)


def read_volume() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    df["date_dt"] = pd.to_datetime(df["date_dt"], errors="coerce")
    df["volume_1e6_m3"] = pd.to_numeric(df["volume_1e6_m3"], errors="coerce")
    df["volume_uncertainty_1e6_m3"] = pd.to_numeric(df["volume_uncertainty_1e6_m3"], errors="coerce")
    return df[df["series_type"].astype(str).str.upper().eq("SWOT")].copy()


def read_basin_areas() -> dict[int, float]:
    # Use the final, unbuffered lake polygons burned into the 10 m filled DEM.
    # These areas correspond exactly to the Basin boundaries used in the maps.
    basin = pd.read_csv(
        NO_BUFFER_BASIN_CSV,
        encoding="utf-8-sig",
        usecols=["lake_id", "no_buffer_area_km2"],
    )
    basin["lake_id"] = pd.to_numeric(basin["lake_id"], errors="coerce")
    basin["no_buffer_area_km2"] = pd.to_numeric(basin["no_buffer_area_km2"], errors="coerce")
    basin = basin.dropna(subset=["lake_id", "no_buffer_area_km2"])
    areas = {
        int(row["lake_id"]): float(row["no_buffer_area_km2"])
        for _, row in basin.iterrows()
    }

    missing = [lake["lake_id"] for lake in LAKES if int(lake["lake_id"]) not in areas]
    if missing:
        raise ValueError(f"Missing basin areas for lakes: {missing}")
    return areas


BASIN_AREAS_KM2 = read_basin_areas()


def nice_y_axis(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -0.05, 1.0, np.array([0.0, 0.5, 1.0])
    ymax = max(float(np.nanmax(finite)) * 1.06, 0.1)
    target = ymax / 3.0
    mag = 10 ** np.floor(np.log10(max(target, 1e-9)))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * mag - target)) * mag
    top = np.ceil(ymax / step) * step
    ticks = np.arange(0.0, top + step * 0.5, step)
    if ticks.size < 4:
        top = step * 3
        ticks = np.arange(0.0, top + step * 0.5, step)
    return float(-0.06 * top), float(top), ticks


def nice_runoff_axis(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -5.0, 100.0, np.array([0.0, 50.0, 100.0])
    ymax = max(float(np.nanmax(finite)) * 1.04, 1.0)
    target = ymax / 4.0
    mag = 10 ** np.floor(np.log10(max(target, 1e-9)))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(m * mag - target)) * mag
    top = np.ceil(ymax / step) * step
    ticks = np.arange(0.0, top + step * 0.5, step)
    if ticks.size < 4:
        top = step * 4
        ticks = np.arange(0.0, top + step * 0.5, step)
    return float(-0.04 * top), float(top), ticks


def volume_to_runoff_mm(volume_1e6_m3: np.ndarray | float, basin_area_km2: float) -> np.ndarray | float:
    return np.asarray(volume_1e6_m3, dtype=float) / float(basin_area_km2) * RUNOFF_MM_PER_1E6M3_PER_KM2


def fit_linear_window(series: pd.DataFrame, start: str, end: str) -> dict | None:
    work = series.copy()
    work["date_dt"] = pd.to_datetime(work["date_dt"])
    work = work[
        work["date_dt"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        & work["volume_1e6_m3"].notna()
    ].sort_values("date_dt")
    if work.shape[0] < 2:
        return None
    x = (work["date_dt"] - work["date_dt"].min()).dt.days.to_numpy(dtype=float)
    y = work["volume_1e6_m3"].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, 1)
    yhat = coeff[0] * x + coeff[1]
    if y.size == 2:
        r2 = 1.0
    else:
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "start_dt": work["date_dt"].min(),
        "end_dt": work["date_dt"].max(),
        "anchor_dt": work["date_dt"].min(),
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "r2": float(r2),
    }


def load_fit_label_layouts() -> dict:
    user_cfg = {}
    for cfg_path in (FIT_LABEL_JSON, LEGACY_FIT_LABEL_JSON):
        if cfg_path.exists():
            try:
                user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                break
            except Exception:
                user_cfg = {}
    merged = json.loads(json.dumps(FIT_LABEL_LAYOUTS_DEFAULT))
    for lake_name, lake_cfg in user_cfg.items():
        merged.setdefault(lake_name, {})
        for seg_name, seg_cfg in lake_cfg.items():
            merged[lake_name].setdefault(seg_name, {})
            merged[lake_name][seg_name].update(seg_cfg)
    return merged


FIT_LABEL_LAYOUTS = load_fit_label_layouts()


def add_fit_overlay(
    ax: plt.Axes,
    series: pd.DataFrame,
    fit_key: str,
    y_top: float,
    basin_area_km2: float,
    fit_line_color: str = DEFAULT_FIT_LINE_COLOR,
    fit_text_color: str = DEFAULT_FIT_LINE_COLOR,
) -> None:
    fit_cfg = FIT_WINDOWS.get(fit_key, {})
    for seg_name in fit_cfg:
        window = fit_cfg.get(seg_name)
        if not window:
            continue
        fit = fit_linear_window(series, window[0], window[1])
        if fit is None:
            continue
        xs = pd.date_range(fit["start_dt"], fit["end_dt"], freq="D")
        xday = (xs - fit["anchor_dt"]).days.to_numpy(dtype=float)
        ys = fit["slope"] * xday + fit["intercept"]
        shift_days = FIT_LINE_X_SHIFT_DAYS.get((fit_key, seg_name), 0.0)
        xs_draw = xs + pd.to_timedelta(shift_days, unit="D")
        ax.plot(xs_draw, ys, linestyle="--", color=fit_line_color, linewidth=1.5, zorder=5)
        x0_num = mdates.date2num(xs_draw[0])
        x1_num = mdates.date2num(xs_draw[-1])
        p0 = ax.transData.transform((x0_num, float(ys[0])))
        p1 = ax.transData.transform((x1_num, float(ys[-1])))
        rotation = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        tx = dx / norm
        ty = dy / norm
        nx = -dy / norm
        ny = dx / norm
        base_segment = "inflow" if seg_name.startswith("inflow") else seg_name
        layout = FIT_LABEL_LAYOUTS.get(fit_key, {}).get(
            seg_name,
            FIT_LABEL_LAYOUTS.get(fit_key, {}).get(base_segment, {}),
        )
        rotation_override = layout.get("rotation_deg")
        if rotation_override is not None:
            rotation = float(rotation_override)
        anchor_frac = float(layout.get("anchor_frac", 0.5))
        anchor_frac = float(np.clip(anchor_frac, 0.05, 0.95))
        mid_idx = int(round((len(xs) - 1) * anchor_frac))
        mid_x = xs_draw[mid_idx]
        mid_y = float(ys[mid_idx])
        pmid = ax.transData.transform((mdates.date2num(mid_x), mid_y))
        auto_label_px = (
            pmid[0]
            + float(layout.get("tangent_px", 0.0)) * tx
            + float(layout.get("normal_px", 24.0)) * nx,
            pmid[1]
            + float(layout.get("tangent_px", 0.0)) * ty
            + float(layout.get("normal_px", 24.0)) * ny,
        )
        auto_label_data = ax.transData.inverted().transform(auto_label_px)
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top_lim = ax.get_ylim()
        x_pad = (x_right - x_left) * 0.04
        y_pad = (y_top_lim - y_bottom) * 0.08
        x_lab_auto = float(np.clip(auto_label_data[0], x_left + x_pad, x_right - x_pad))
        y_lab_auto = float(np.clip(auto_label_data[1], y_bottom + y_pad, y_top_lim - y_pad))
        clipped_auto_px = ax.transData.transform((x_lab_auto, y_lab_auto))
        # Apply the interactive screen-space offset only after the automatic
        # anchor has been constrained. Otherwise a large historical offset is
        # swallowed by clipping and left/right controls appear unresponsive.
        label_px = (
            clipped_auto_px[0] + float(layout.get("manual_dx_px", 0.0)),
            clipped_auto_px[1] + float(layout.get("manual_dy_px", 0.0)),
        )
        label_data = ax.transData.inverted().transform(label_px)
        x_lab = float(label_data[0])
        y_lab = float(label_data[1])
        rate_mm_per_day = float(volume_to_runoff_mm(fit["slope"], basin_area_km2))
        display_rate = rate_mm_per_day
        wrap_combined = False
        if RATE_LABEL_STYLE == "combined":
            rate_name = "Infilling rate" if base_segment == "inflow" else "Outflow rate"
            show_rate_name = fit_key == "lake1_swot_only"
            rate_prefix = f"{rate_name} = " if show_rate_name else ""
            volume_annotation = (
                f"{rate_prefix}{fit['slope']:.2f} × "
                + r"$10^6$ m$^3$ d$^{-1}$"
            )
            runoff_annotation = f" ({display_rate:.1f} mm " + r"d$^{-1}$)"
            wrap_combined = fit_key in {
                "lake38_swot_only",
                "lake70_swot_only",
                "lake30_swot_only",
            }
            if wrap_combined and show_rate_name:
                volume_annotation = (
                    f"{rate_name} =\n{fit['slope']:.2f} × "
                    + r"$10^6$ m$^3$ d$^{-1}$"
                )
        elif fit_key in {"lake70_swot_only", "lake30_swot_only"} and base_segment == "drain":
            runoff_annotation = f"{display_rate:.1f} mm/day"
            volume_annotation = rf"${fit['slope']:.2f}\times10^6\ \mathrm{{m^3/day}}$"
        else:
            q_label = "Infill_rate" if base_segment == "inflow" else "Outflow_rate"
            runoff_annotation = f"{q_label} = {display_rate:.1f} mm/day"
            volume_annotation = rf"$k = {fit['slope']:.2f}\times10^6\ \mathrm{{m^3/day}}$"
        # Keep all six lake panels typographically consistent. Layout JSON
        # controls position and rotation, while the paper figure uses one size.
        fontsize = FIT_ANNOTATION_SIZE
        label_center_px = ax.transData.transform((x_lab, y_lab))
        font_px = fontsize * ax.figure.dpi / 72.0
        line_offset_px = FIT_ANNOTATION_HALF_SPACING * font_px
        # Near-vertical drainage labels have taller rotated math-text boxes.
        # Compensate their centre spacing so the visible black/red gap matches
        # the other panels without changing the common 18 pt font size.
        if fit_key in {"lake70_swot_only", "lake30_swot_only"} and base_segment == "drain":
            line_offset_px = 1.50 * font_px
        # Separate the two parallel text rows perpendicular to the final text
        # rotation (not the fitted-line angle). This keeps the visible spacing
        # consistent even when a manual rotation override is used.
        text_angle_rad = np.deg2rad(rotation)
        text_nx = -float(np.sin(text_angle_rad))
        text_ny = float(np.cos(text_angle_rad))
        volume_label_px = (
            label_center_px[0] + line_offset_px * text_nx,
            label_center_px[1] + line_offset_px * text_ny,
        )
        runoff_label_px = (
            label_center_px[0] - line_offset_px * text_nx,
            label_center_px[1] - line_offset_px * text_ny,
        )
        volume_label_data = ax.transData.inverted().transform(volume_label_px)
        runoff_label_data = ax.transData.inverted().transform(runoff_label_px)
        common_text = {
            "fontsize": fontsize,
            "rotation": rotation,
            "va": "center",
            "rotation_mode": "anchor",
            "zorder": 6,
        }
        if RATE_LABEL_STYLE == "combined" and not wrap_combined:
            # Join the black and red fragments at one rotated anchor so they
            # read as a single sentence while retaining independent colours.
            # Measure both fragments and move their join point so the complete
            # two-colour sentence, rather than the join itself, is centred on
            # the stored layout anchor.
            probe_black = ax.text(0, 0, volume_annotation, fontsize=fontsize, alpha=0.0)
            probe_red = ax.text(0, 0, runoff_annotation, fontsize=fontsize, alpha=0.0)
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            black_width = probe_black.get_window_extent(renderer=renderer).width
            red_width = probe_red.get_window_extent(renderer=renderer).width
            probe_black.remove()
            probe_red.remove()
            text_tx = float(np.cos(text_angle_rad))
            text_ty = float(np.sin(text_angle_rad))
            join_shift_px = 0.5 * (black_width - red_width)
            join_px = (
                label_center_px[0] + join_shift_px * text_tx,
                label_center_px[1] + join_shift_px * text_ty,
            )
            # Keep the complete rotated sentence inside the plotting box.
            group_start = np.asarray(join_px) - black_width * np.asarray((text_tx, text_ty))
            group_end = np.asarray(join_px) + red_width * np.asarray((text_tx, text_ty))
            group_min = np.minimum(group_start, group_end) - 0.55 * font_px
            group_max = np.maximum(group_start, group_end) + 0.55 * font_px
            axes_box = ax.get_window_extent(renderer=renderer)
            lower = np.asarray((axes_box.x0 + 8.0, axes_box.y0 + 8.0))
            upper = np.asarray((axes_box.x1 - 8.0, axes_box.y1 - 8.0))
            correction = np.zeros(2, dtype=float)
            correction += np.maximum(lower - group_min, 0.0)
            correction -= np.maximum(group_max + correction - upper, 0.0)
            join_px = tuple(np.asarray(join_px) + correction)
            join_data = ax.transData.inverted().transform(join_px)
            anchor_dt = mdates.num2date(join_data[0]).replace(tzinfo=None)
            ax.text(
                anchor_dt,
                join_data[1],
                volume_annotation,
                color="black",
                ha="right",
                **common_text,
            )
            ax.text(
                anchor_dt,
                join_data[1],
                runoff_annotation,
                color=fit_text_color,
                ha="left",
                **common_text,
            )
            continue
        ax.text(
            mdates.num2date(volume_label_data[0]).replace(tzinfo=None),
            volume_label_data[1],
            volume_annotation,
            color="black",
            ha="center",
            **common_text,
        )
        ax.text(
            mdates.num2date(runoff_label_data[0]).replace(tzinfo=None),
            runoff_label_data[1],
            runoff_annotation,
            color=fit_text_color,
            ha="center",
            **common_text,
        )


def content_bbox(image: Image.Image, threshold: int = 248) -> tuple[int, int, int, int]:
    rgb = np.asarray(image.convert("RGB"))
    non_white = np.any(rgb < threshold, axis=2)
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, image.width, image.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def union_bbox(
    boxes: list[tuple[int, int, int, int]],
    size: tuple[int, int],
    pad_left: int = 20,
    pad_top: int = 20,
    pad_right: int = 90,
    pad_bottom: int = 28,
) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, min(box[0] for box in boxes) - pad_left)
    top = max(0, min(box[1] for box in boxes) - pad_top)
    right = min(width, max(box[2] for box in boxes) + pad_right)
    bottom = min(height, max(box[3] for box in boxes) + pad_bottom)
    return (left, top, right, bottom)


def interpolate_branch(branch: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    out = (
        branch.set_index("date_dt")[["volume_1e6_m3", "volume_uncertainty_1e6_m3"]]
        .reindex(dates)
        .sort_index()
    )
    out["volume_1e6_m3"] = out["volume_1e6_m3"].interpolate(method="time", limit_direction="both")
    out["volume_uncertainty_1e6_m3"] = out["volume_uncertainty_1e6_m3"].interpolate(method="time", limit_direction="both")
    out = out.reset_index().rename(columns={"index": "date_dt"})
    return out


def prepare_lake(df: pd.DataFrame, row_keys: tuple[str, str], merge_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["series_type"].eq("SWOT")].copy()
    b1 = df[df["row_key"].eq(row_keys[0])].sort_values("date_dt").copy()
    b2 = df[df["row_key"].eq(row_keys[1])].sort_values("date_dt").copy()

    pre1 = b1[b1["date_dt"] < merge_date].copy()
    pre2 = b2[b2["date_dt"] < merge_date].copy()
    post = b1[b1["date_dt"] >= merge_date].copy()

    pre_start = min(pre1["date_dt"].min(), pre2["date_dt"].min())
    pre_end = merge_date - pd.Timedelta(days=1)
    pre1_daily = interpolate_branch(pre1, pre_start, pre_end)
    pre2_daily = interpolate_branch(pre2, pre_start, pre_end)

    pre = pre1_daily.rename(
        columns={
            "volume_1e6_m3": "branch1_volume",
            "volume_uncertainty_1e6_m3": "branch1_unc",
        }
    ).merge(
        pre2_daily.rename(
            columns={
                "volume_1e6_m3": "branch2_volume",
                "volume_uncertainty_1e6_m3": "branch2_unc",
            }
        ),
        on="date_dt",
        how="outer",
    )
    pre["total_volume"] = pre["branch1_volume"] + pre["branch2_volume"]
    pre["total_unc"] = np.sqrt(np.square(pre["branch1_unc"]) + np.square(pre["branch2_unc"]))

    post = post[["date_dt", "volume_1e6_m3", "volume_uncertainty_1e6_m3"]].rename(
        columns={
            "volume_1e6_m3": "total_volume",
            "volume_uncertainty_1e6_m3": "total_unc",
        }
    )
    return pre.sort_values("date_dt"), post.sort_values("date_dt")


def pad_post_fill(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """Add one visual anchor so the merged fill starts flush with the pre-merge block."""
    if pre.empty or post.empty:
        return post
    anchor = post.iloc[[0]].copy()
    anchor["date_dt"] = pd.to_datetime(pre["date_dt"].max())
    return pd.concat([anchor, post], ignore_index=True, sort=False).sort_values("date_dt")


def setup_dual_axes(
    ax_left: plt.Axes,
    basin_area_km2: float,
    volume_ymin: float,
    volume_ymax: float,
    volume_yticks: np.ndarray,
    runoff_color: str = TOTAL_LINE,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot volume on the left and equivalent water depth on the right."""
    ax_left.set_ylim(volume_ymin, volume_ymax)
    ax_left.set_yticks(volume_yticks)
    ax_left.yaxis.set_label_position("left")
    ax_left.yaxis.tick_left()
    ax_left.spines["right"].set_visible(False)
    ax_left.tick_params(axis="y", labelsize=TICK_SIZE, width=TICK_WIDTH, length=TICK_LENGTH, pad=5)
    ax_left.set_ylabel(LEFT_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=10)
    ax_left.text(-0.005, 1.015, LEFT_LABEL_TEXT, transform=ax_left.transAxes, ha="left", va="bottom", fontsize=TICK_SIZE)

    runoff_limits = volume_to_runoff_mm(np.array([volume_ymin, volume_ymax]), basin_area_km2)
    runoff_ymin, runoff_ymax, runoff_yticks = nice_runoff_axis(runoff_limits)
    ax_right = ax_left.twinx()
    ax_right.set_ylim(runoff_ymin, runoff_ymax)
    ax_right.set_yticks(runoff_yticks)
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color(runoff_color)
    ax_right.tick_params(
        axis="y",
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        pad=5,
        colors=runoff_color,
    )
    ax_right.set_ylabel(
        RIGHT_AXIS_LABEL,
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        color=runoff_color,
        labelpad=10,
    )
    ax_right.tick_params(axis="x", labelsize=TICK_SIZE, width=TICK_WIDTH, length=TICK_LENGTH, pad=5)
    ax_right.text(
        1.005,
        1.015,
        RIGHT_UNIT_TEXT,
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=TICK_SIZE,
        color=runoff_color,
    )
    return ax_left, ax_right


def mean_observation_interval_days(dates: pd.Series) -> float:
    """Return the mean spacing between unique plotted observation dates."""
    unique_dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna().unique()).sort_values()
    if len(unique_dates) < 2:
        return 0.0
    intervals_days = np.diff(unique_dates.asi8) / (24.0 * 60.0 * 60.0 * 1e9)
    return float(np.mean(intervals_days))


def draw_single_lake(lake: dict, df: pd.DataFrame) -> None:
    key = lake["row_keys"][0]
    basin_area_km2 = BASIN_AREAS_KM2[int(lake["lake_id"])]
    sub = (
        df[df["row_key"].eq(key)][["date_dt", "volume_1e6_m3", "volume_uncertainty_1e6_m3"]]
        .dropna(subset=["date_dt", "volume_1e6_m3"])
        .sort_values("date_dt")
        .copy()
    )
    plot_end = PLOT_END_DATES.get(int(lake["lake_id"]))
    if plot_end is not None:
        sub = sub[sub["date_dt"].le(plot_end)].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    volume_ymin, volume_ymax, volume_yticks = nice_y_axis(sub["volume_1e6_m3"].to_numpy(float))
    ax_volume, ax_runoff = setup_dual_axes(
        ax,
        basin_area_km2,
        volume_ymin,
        volume_ymax,
        volume_yticks,
        runoff_color=RUNOFF_RED,
    )

    unc = sub["volume_uncertainty_1e6_m3"].fillna(0.0)
    if (unc > 0).any():
        ax_volume.errorbar(
            sub["date_dt"],
            sub["volume_1e6_m3"],
            yerr=unc,
            fmt="none",
            ecolor=ERR_COLOR,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.42,
            zorder=3,
        )
    ax_volume.plot(sub["date_dt"], sub["volume_1e6_m3"], color=TOTAL_LINE, linewidth=1.7, zorder=4)
    ax_volume.scatter(sub["date_dt"], sub["volume_1e6_m3"], s=26, color=TOTAL_LINE, edgecolors="none", zorder=4.2)

    mean_interval_days = mean_observation_interval_days(sub["date_dt"])
    ax.text(
        0.98,
        0.06,
        f"Mean observation interval = {mean_interval_days:.1f} days",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
        color="black",
        zorder=10,
    )

    ax.text(0.02, 0.93, lake["title"], transform=ax.transAxes, ha="left", va="top", fontsize=LABEL_SIZE, fontweight="bold")

    add_fit_overlay(
        ax_volume,
        sub[["date_dt", "volume_1e6_m3"]].copy(),
        lake["fit_key"],
        volume_ymax,
        basin_area_km2,
        fit_line_color=DEFAULT_FIT_LINE_COLOR,
        fit_text_color=RUNOFF_RED,
    )
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=TICK_SIZE, width=TICK_WIDTH, length=TICK_LENGTH, pad=5)
    ax.grid(False)
    ax.set_xlabel(
        "Date" if int(lake["lake_id"]) == 30 else "",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        labelpad=8,
    )
    ax_runoff.grid(False)

    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.22, top=0.92)
    out_png = OUT_DIR / f"{lake['out_name']}.png"
    out_tif = OUT_DIR / f"{lake['out_name']}.tif"
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_tif, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_tif}")


def draw_stacked_lake(lake: dict, df: pd.DataFrame) -> None:
    pre, post = prepare_lake(df, lake["row_keys"], lake["merge_date"])
    basin_area_km2 = BASIN_AREAS_KM2[int(lake["lake_id"])]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    yvals = np.concatenate([pre["total_volume"].to_numpy(float), post["total_volume"].to_numpy(float)])
    volume_ymin, volume_ymax, volume_yticks = nice_y_axis(yvals)
    ax_volume, ax_runoff = setup_dual_axes(
        ax,
        basin_area_km2,
        volume_ymin,
        volume_ymax,
        volume_yticks,
        runoff_color=RUNOFF_RED,
    )

    ax_volume.fill_between(
        pre["date_dt"],
        0.0,
        pre["branch1_volume"],
        color=BRANCH1_FILL,
        alpha=0.8,
        linewidth=0,
        label=lake["legend_labels"][0],
        zorder=1,
    )
    ax_volume.fill_between(
        pre["date_dt"],
        pre["branch1_volume"],
        pre["total_volume"],
        color=BRANCH2_FILL,
        alpha=0.8,
        linewidth=0,
        label=lake["legend_labels"][1],
        zorder=1.1,
    )
    total_all = pd.concat(
        [
            pre[["date_dt", "total_volume", "total_unc"]],
            post[["date_dt", "total_volume", "total_unc"]],
        ],
        ignore_index=True,
    ).sort_values("date_dt")

    unc = total_all["total_unc"].fillna(0.0)
    if (unc > 0).any():
        ax_volume.errorbar(
            total_all["date_dt"],
            total_all["total_volume"],
            yerr=unc,
            fmt="none",
            ecolor=ERR_COLOR,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.42,
            zorder=3,
        )
    ax_volume.plot(total_all["date_dt"], total_all["total_volume"], color=TOTAL_LINE, linewidth=1.7, zorder=4)
    ax_volume.scatter(total_all["date_dt"], total_all["total_volume"], s=26, color=TOTAL_LINE, edgecolors="none", zorder=4.2)

    mean_interval_days = mean_observation_interval_days(total_all["date_dt"])
    ax.text(
        0.98,
        0.06,
        f"Mean observation interval = {mean_interval_days:.1f} days",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
        color="black",
        zorder=10,
    )

    ax.text(0.02, 0.93, lake["title"], transform=ax.transAxes, ha="left", va="top", fontsize=LABEL_SIZE, fontweight="bold")

    fit_series = total_all.rename(columns={"total_volume": "volume_1e6_m3"})[["date_dt", "volume_1e6_m3"]].copy()
    add_fit_overlay(
        ax_volume,
        fit_series,
        lake["fit_key"],
        volume_ymax,
        basin_area_km2,
        fit_line_color=DEFAULT_FIT_LINE_COLOR,
        fit_text_color=RUNOFF_RED,
    )
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=TICK_SIZE, width=TICK_WIDTH, length=TICK_LENGTH, pad=5)
    ax.grid(False)
    ax.set_xlabel("")
    ax_runoff.grid(False)

    legend = ax_volume.legend(loc="upper left", bbox_to_anchor=(0.03, 0.79), frameon=True, fontsize=LEGEND_SIZE, handlelength=2.2)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.86)

    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.22, top=0.92)
    out_png = OUT_DIR / f"{lake['out_name']}.png"
    out_tif = OUT_DIR / f"{lake['out_name']}.tif"
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_tif, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_tif}")


def render_one_lake_to_image(lake: dict, df: pd.DataFrame) -> Image.Image:
    tmp_png = OUT_DIR / f"__preview_{lake['out_name']}.png"
    if lake["plot_mode"] == "stacked":
        draw_stacked_lake(lake, df)
    else:
        draw_single_lake(lake, df)
    src = OUT_DIR / f"{lake['out_name']}.png"
    return Image.open(src).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the six representative runoff-volume panels.")
    parser.add_argument(
        "--combined-rate-labels",
        action="store_true",
        help="Write an alternate version with one combined two-colour rate annotation.",
    )
    args = parser.parse_args()
    if args.combined_rate_labels:
        global OUT_DIR, STACKED_PNG, STACKED_TIF, FIT_LABEL_JSON
        global FIT_LABEL_LAYOUTS, RATE_LABEL_STYLE, RIGHT_AXIS_LABEL
        OUT_DIR = RESULT_ROOT / "runoff_volume_combined_rate_labels"
        STACKED_PNG = OUT_DIR / "representative_6_dual_axis_runoff_volume_combined_rate_labels.png"
        STACKED_TIF = OUT_DIR / "representative_6_dual_axis_runoff_volume_combined_rate_labels.tif"
        FIT_LABEL_JSON = INPUT_ROOT / "fit_label_layouts.json"
        RATE_LABEL_STYLE = "combined"
        RIGHT_AXIS_LABEL = "Catchment-normalized\nvolume change"
        FIT_LABEL_LAYOUTS = load_fit_label_layouts()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_volume()
    row_pngs: list[Path] = []
    for lake in LAKES:
        if lake["plot_mode"] == "stacked":
            draw_stacked_lake(lake, df)
        else:
            draw_single_lake(lake, df)
        row_pngs.append(OUT_DIR / f"{lake['out_name']}.png")

    images = [Image.open(path).convert("RGB") for path in row_pngs]
    size = images[0].size
    if any(image.size != size for image in images):
        raise ValueError("All 07_4 row images must have the same size before stacking.")
    content_boxes = [content_bbox(image) for image in images]
    common_left = max(0, min(box[0] for box in content_boxes) - 20)
    common_right = min(size[0], max(box[2] for box in content_boxes) + 90)
    cropped = []
    for image, box in zip(images, content_boxes):
        crop_box = (
            common_left,
            max(0, box[1] - 10),
            common_right,
            min(size[1], box[3] + 10),
        )
        cropped.append(image.crop(crop_box))
    row_width = cropped[0].width
    row_gap = 12
    total_height = sum(image.height for image in cropped) + row_gap * (len(cropped) - 1)
    stacked = Image.new("RGB", (row_width, total_height), "white")
    y = 0
    for index, image in enumerate(cropped):
        stacked.paste(image, (0, y))
        y += image.height
        if index < len(cropped) - 1:
            y += row_gap
    stacked.save(STACKED_PNG, dpi=(300, 300))
    stacked.save(STACKED_TIF, dpi=(300, 300), compression="tiff_lzw")
    print(f"Wrote {STACKED_PNG}")
    print(f"Wrote {STACKED_TIF}")


if __name__ == "__main__":
    main()
