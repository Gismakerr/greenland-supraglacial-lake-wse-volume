from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(r"Z:\冰面湖水量测算_Local\冰面湖水位论文代码数据整理")
WORK_ROOT = ROOT / "result_721" / "lake_1_2_3"
IN_CSV = WORK_ROOT / "result" / "07_3_representative_8_unified_plot" / "representative_8_volume_timeseries.csv"
OUT_DIR = Path(r"X:\冰面湖水量\东北\result\10_lake_volume_change\lake12_dual_axis_gray_test")
STACKED_PNG = OUT_DIR / "lake12_dual_axis_gray_stacked.png"
STACKED_TIF = OUT_DIR / "lake12_dual_axis_gray_stacked.tif"
FIT_LABEL_JSON = OUT_DIR / "fit_label_layouts.json"
LEGACY_FIT_LABEL_JSON = WORK_ROOT / "result" / "07_4_representative_swot_volume_only" / "fit_label_layouts.json"
WATERSHED_CSV = WORK_ROOT / "data" / "watersheds" / "lake_1_2_38_watershed_inventory.csv"
AICC_RACMO_CSV = ROOT / "result_volume" / "11_regional_climate_for_aicc_lakes_liuyu_10m_20240622_20240814" / "racmo_aicc_lakes_20240625_20240725.csv"

DPI = 300
# Keep the 07_3 look, but slightly taller so the row is less flattened.
FIGSIZE = (17.0, 4.10)

FONT_FAMILY = "Arial"
LABEL_SIZE = 20
LEGEND_SIZE = 14
TICK_SIZE = 18
SPINE_WIDTH = 1.35
TICK_WIDTH = 1.35
TICK_LENGTH = 8

BRANCH1_FILL = "#bfc4c8"
BRANCH2_FILL = "#969ca1"
TOTAL_LINE = "#1f1f1f"
ERR_COLOR = "#404040"
RUNOFF_RED = "#df3f3f"
FIT_LINE_COLOR = RUNOFF_RED
FIT_LINE_X_SHIFT_DAYS = {("lake30_swot_only", "drain"): 0.5}

RUNOFF_MM_PER_1E6M3_PER_KM2 = 1000.0
LEFT_LABEL_TEXT = r"$10^6$ m$^3$"
RIGHT_UNIT_TEXT = "mm"

X_MIN = pd.Timestamp("2024-06-22")
X_MAX = pd.Timestamp("2024-08-17")
X_TICKS = pd.to_datetime(
    ["2024-06-25", "2024-07-05", "2024-07-15", "2024-07-25", "2024-08-05", "2024-08-15"]
)

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
    "lake2_swot_only": {"inflow": ("2024-06-26", "2024-08-06")},
    "lake38_swot_only": {"inflow": ("2024-06-23", "2024-07-07"), "drain": ("2024-07-07", "2024-07-13")},
    "lake4_swot_only": {"inflow": ("2024-06-28", "2024-08-10")},
    "lake70_swot_only": {"inflow": ("2024-07-03", "2024-07-12"), "drain": ("2024-07-12", "2024-07-15")},
    "lake30_swot_only": {"inflow": ("2024-07-05", "2024-07-20"), "drain": ("2024-07-20", "2024-07-21")},
    "lake1_dual_axis": {"inflow": ("2024-06-26", "2024-07-28"), "drain": ("2024-07-28", "2024-08-14")},
    "lake2_dual_axis": {"inflow": ("2024-06-26", "2024-08-06")},
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
        "inflow": {"anchor_frac": 0.48, "normal_px": 34.0, "tangent_px": 0.0, "fontsize": 20, "rotation_deg": None},
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
    ws = pd.read_csv(WATERSHED_CSV, encoding="utf-8-sig")
    ws["lake_id"] = pd.to_numeric(ws["lake_id"], errors="coerce")
    ws["area_km2"] = pd.to_numeric(ws["area_km2"], errors="coerce")
    ws["branch_index"] = pd.to_numeric(ws["branch_index"], errors="coerce")
    ws["phase"] = ws["phase"].astype(str).str.lower()

    areas: dict[int, float] = {}
    merged = ws[ws["phase"].eq("merged")].dropna(subset=["lake_id", "area_km2"])
    for _, row in merged.iterrows():
        areas[int(row["lake_id"])] = float(row["area_km2"])
    original = ws[ws["phase"].eq("original")].dropna(subset=["lake_id", "area_km2"])
    for _, row in original.iterrows():
        areas[int(row["lake_id"])] = float(row["area_km2"])

    racmo = pd.read_csv(AICC_RACMO_CSV, encoding="utf-8-sig", usecols=["basin_id", "basin_area_km2"])
    racmo["basin_id"] = pd.to_numeric(racmo["basin_id"], errors="coerce")
    racmo["basin_area_km2"] = pd.to_numeric(racmo["basin_area_km2"], errors="coerce")
    for lake_id in (4, 30, 70):
        sub = racmo[racmo["basin_id"].eq(lake_id)]["basin_area_km2"].dropna()
        if not sub.empty:
            areas[lake_id] = float(sub.iloc[0])

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
) -> None:
    fit_cfg = FIT_WINDOWS.get(fit_key, {})
    for seg_name in ("inflow", "drain"):
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
        ax.plot(xs_draw, ys, linestyle="--", color=FIT_LINE_COLOR, linewidth=1.5, zorder=5)
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
        layout = FIT_LABEL_LAYOUTS.get(fit_key, {}).get(seg_name, {})
        rotation_override = layout.get("rotation_deg")
        if rotation_override is not None:
            rotation = float(rotation_override)
        anchor_frac = float(layout.get("anchor_frac", 0.5))
        anchor_frac = float(np.clip(anchor_frac, 0.05, 0.95))
        mid_idx = int(round((len(xs) - 1) * anchor_frac))
        mid_x = xs_draw[mid_idx]
        mid_y = float(ys[mid_idx])
        pmid = ax.transData.transform((mdates.date2num(mid_x), mid_y))
        label_px = (
            pmid[0] + float(layout.get("tangent_px", 0.0)) * tx + float(layout.get("normal_px", 24.0)) * nx,
            pmid[1] + float(layout.get("tangent_px", 0.0)) * ty + float(layout.get("normal_px", 24.0)) * ny,
        )
        label_data = ax.transData.inverted().transform(label_px)
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top_lim = ax.get_ylim()
        x_pad = (x_right - x_left) * 0.04
        y_pad = (y_top_lim - y_bottom) * 0.08
        x_lab = float(np.clip(label_data[0], x_left + x_pad, x_right - x_pad))
        y_lab = float(np.clip(label_data[1], y_bottom + y_pad, y_top_lim - y_pad))
        rate_mm_per_day = float(volume_to_runoff_mm(fit["slope"], basin_area_km2))
        display_rate = rate_mm_per_day
        if fit_key in {"lake70_swot_only", "lake30_swot_only"} and seg_name == "drain":
            annotation = f"{display_rate:.1f} mm/day"
        else:
            q_label = "Infill_rate" if seg_name == "inflow" else "Outflow_rate"
            annotation = f"{q_label} = {display_rate:.1f} mm/day"
        ax.text(
            mdates.num2date(x_lab).replace(tzinfo=None),
            y_lab,
            annotation,
            fontsize=float(layout.get("fontsize", 18)),
            color=FIT_LINE_COLOR,
            rotation=rotation,
            ha="center",
            va="center",
            zorder=6,
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
) -> tuple[plt.Axes, plt.Axes]:
    """Use volume on the left axis and red equivalent water depth on the right."""
    ax_left.set_ylim(volume_ymin, volume_ymax)
    ax_left.set_yticks(volume_yticks)
    ax_left.yaxis.set_label_position("left")
    ax_left.yaxis.tick_left()
    ax_left.spines["right"].set_visible(False)
    ax_left.tick_params(axis="y", labelsize=TICK_SIZE, width=TICK_WIDTH, length=TICK_LENGTH, pad=5)
    ax_left.text(-0.005, 1.015, LEFT_LABEL_TEXT, transform=ax_left.transAxes, ha="left", va="bottom", fontsize=TICK_SIZE)

    runoff_limits = volume_to_runoff_mm(np.array([volume_ymin, volume_ymax]), basin_area_km2)
    runoff_ymin, runoff_ymax, runoff_yticks = nice_runoff_axis(runoff_limits)
    ax_right = ax_left.twinx()
    ax_right.set_ylim(runoff_ymin, runoff_ymax)
    ax_right.set_yticks(runoff_yticks)
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color(RUNOFF_RED)
    ax_right.tick_params(
        axis="y",
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        pad=5,
        colors=RUNOFF_RED,
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
        color=RUNOFF_RED,
    )
    return ax_left, ax_right


def draw_single_lake(lake: dict, df: pd.DataFrame) -> None:
    key = lake["row_keys"][0]
    basin_area_km2 = BASIN_AREAS_KM2[int(lake["lake_id"])]
    sub = (
        df[df["row_key"].eq(key)][["date_dt", "volume_1e6_m3", "volume_uncertainty_1e6_m3"]]
        .dropna(subset=["date_dt", "volume_1e6_m3"])
        .sort_values("date_dt")
        .copy()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    volume_ymin, volume_ymax, volume_yticks = nice_y_axis(sub["volume_1e6_m3"].to_numpy(float))
    ax_volume, ax_runoff = setup_dual_axes(ax, basin_area_km2, volume_ymin, volume_ymax, volume_yticks)

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

    ax.text(0.02, 0.93, lake["title"], transform=ax.transAxes, ha="left", va="top", fontsize=LABEL_SIZE, fontweight="bold")

    add_fit_overlay(
        ax_volume,
        sub[["date_dt", "volume_1e6_m3"]].copy(),
        lake["fit_key"],
        volume_ymax,
        basin_area_km2,
    )
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=TICK_SIZE, width=TICK_WIDTH, length=TICK_LENGTH, pad=5)
    ax.grid(False)
    ax.set_xlabel("")
    ax_runoff.grid(False)

    fig.subplots_adjust(left=0.085, right=0.93, bottom=0.22, top=0.92)
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
    ax_volume, ax_runoff = setup_dual_axes(ax, basin_area_km2, volume_ymin, volume_ymax, volume_yticks)

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

    ax.text(0.02, 0.93, lake["title"], transform=ax.transAxes, ha="left", va="top", fontsize=LABEL_SIZE, fontweight="bold")

    fit_series = total_all.rename(columns={"total_volume": "volume_1e6_m3"})[["date_dt", "volume_1e6_m3"]].copy()
    add_fit_overlay(ax_volume, fit_series, lake["fit_key"], volume_ymax, basin_area_km2)
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

    fig.subplots_adjust(left=0.085, right=0.93, bottom=0.22, top=0.92)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_volume()
    row_pngs: list[Path] = []
    selected_lakes = [lake for lake in LAKES if int(lake["lake_id"]) in (1, 2)]
    for lake in selected_lakes:
        if lake["plot_mode"] == "stacked":
            draw_stacked_lake(lake, df)
        else:
            draw_single_lake(lake, df)
        row_pngs.append(OUT_DIR / f"{lake['out_name']}.png")

    images = [Image.open(path).convert("RGB") for path in row_pngs]
    size = images[0].size
    if any(image.size != size for image in images):
        raise ValueError("All 07_4 row images must have the same size before stacking.")
    crop_box = union_bbox([content_bbox(image) for image in images], size)
    cropped = [image.crop(crop_box) for image in images]
    row_width = cropped[0].width
    total_height = sum(image.height for image in cropped)
    stacked = Image.new("RGB", (row_width, total_height), "white")
    y = 0
    for image in cropped:
        stacked.paste(image, (0, y))
        y += image.height
    stacked.save(STACKED_PNG, dpi=(300, 300))
    stacked.save(STACKED_TIF, dpi=(300, 300), compression="tiff_lzw")
    print(f"Wrote {STACKED_PNG}")
    print(f"Wrote {STACKED_TIF}")


if __name__ == "__main__":
    main()
