"""
Stage 07_4:
- Read denoised branch stage tables from Stage 06_3 outputs.
- For each branch curve, shift the minimum water level to 0.
- Plot all normalized curves into one group figure.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_ROOT_DIR = PROJECT_ROOT_DIR / "result1"

INPUT_06_3_DIR = OUTPUT_ROOT_DIR / "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"
OUTPUT_DIR = OUTPUT_ROOT_DIR / "07_4_group_plot_min0_anchor"

METHOD_NAMES = ["Otsu"]
PIXC_VERSION_SUBDIR = "vC"
BRANCH_TABLE_DIRNAME = "06_3_BranchTables"
BRANCH_TABLE_ROOT_CANDIDATES = ["original", "原版", ""]

DATE_RANGE_START = "2024-06-05"
DATE_RANGE_END = "2024-08-15"

CURVE_DPI = 220
FIG_SIZE = (20, 8)
CURVE_WIDTH = 1.3
POINT_SIZE = 10
EXCLUDED_BRANCH_KEYS = {(70, 1), (377, 1)}
ALPHA_MIN = 0.10
ALPHA_MAX = 0.95
SPLIT_START_DATE = "2024-06-29"


def progress(msg: str) -> None:
    print(msg, flush=True)


def resolve_branch_table_dir(method_name: str) -> Path:
    for root_name in BRANCH_TABLE_ROOT_CANDIDATES:
        if root_name:
            candidate = INPUT_06_3_DIR / method_name / root_name / BRANCH_TABLE_DIRNAME / PIXC_VERSION_SUBDIR
        else:
            candidate = INPUT_06_3_DIR / method_name / BRANCH_TABLE_DIRNAME / PIXC_VERSION_SUBDIR
        if candidate.is_dir():
            return candidate
    return INPUT_06_3_DIR / method_name / "original" / BRANCH_TABLE_DIRNAME / PIXC_VERSION_SUBDIR


def resolve_no_noise_dir(method_name: str) -> Path:
    method_root = INPUT_06_3_DIR / method_name
    # Prefer dynamic discovery first to avoid Chinese-folder encoding mismatches.
    for path in method_root.glob("**/vc/no_noise"):
        if path.is_dir():
            return path
    preferred = [
        method_root / "original" / "vc" / "no_noise",
        method_root / "原版" / "vc" / "no_noise",
    ]
    for path in preferred:
        if path.is_dir():
            return path
    return method_root / "vc" / "no_noise"


def parse_lake_branch(name: str) -> Tuple[int, int]:
    match = re.match(r"lake_(\d+)_branch_(\d+)_stage_vC\.csv$", str(name), re.IGNORECASE)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def load_no_noise_branch_keys(no_noise_dir: Path) -> set:
    keys = set()
    if not no_noise_dir.is_dir():
        return keys
    for name in os.listdir(no_noise_dir):
        match = re.match(r"Lake_(\d+)_branch_(\d+)_vC\.png$", str(name), re.IGNORECASE)
        if not match:
            continue
        keys.add((int(match.group(1)), int(match.group(2))))
    return keys


def load_branch_curves(
    branch_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    allowed_keys: set,
) -> Dict[Tuple[int, int], pd.DataFrame]:
    curves: Dict[Tuple[int, int], pd.DataFrame] = {}
    if not branch_dir.is_dir():
        return curves

    for name in sorted(os.listdir(branch_dir)):
        lake_id, branch_idx = parse_lake_branch(name)
        if lake_id < 0:
            continue
        if allowed_keys and (int(lake_id), int(branch_idx)) not in allowed_keys:
            continue
        if (int(lake_id), int(branch_idx)) in EXCLUDED_BRANCH_KEYS:
            continue

        path = branch_dir / name
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if df.empty or "date" not in df.columns or "wse_m" not in df.columns:
            continue

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work["wse_m"] = pd.to_numeric(work["wse_m"], errors="coerce")
        work = work.dropna(subset=["date", "wse_m"]).copy()
        work = work[(work["date"] >= start_ts) & (work["date"] <= end_ts)].copy()
        if work.empty:
            continue

        # Keep both red and gray points; only noise points are removed below.
        if "is_noise" in work.columns:
            is_noise = work["is_noise"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
            work = work[~is_noise].copy()
        if work.empty:
            continue

        curve = (
            work.assign(day=work["date"].dt.normalize())
            .groupby("day", as_index=False)["wse_m"]
            .mean()
            .rename(columns={"day": "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        if curve.empty:
            continue

        curve["wse_rel"] = curve["wse_m"] - float(curve["wse_m"].min())
        curve["lake_id"] = int(lake_id)
        curve["branch_index"] = int(branch_idx)
        curve["wse_span_m"] = float(curve["wse_m"].max() - curve["wse_m"].min())
        curves[(int(lake_id), int(branch_idx))] = curve[["lake_id", "branch_index", "date", "wse_m", "wse_rel", "wse_span_m"]].copy()

    return curves


def save_points_table(curves: Dict[Tuple[int, int], pd.DataFrame], out_csv: Path) -> None:
    rows: List[pd.DataFrame] = []
    for (lake_id, branch_idx), df in curves.items():
        if df.empty:
            continue
        rows.append(df[["lake_id", "branch_index", "date", "wse_m", "wse_rel", "wse_span_m"]].copy())

    if not rows:
        progress(f"skip csv: no rows -> {out_csv}")
        return

    out = pd.concat(rows, ignore_index=True).sort_values(["lake_id", "branch_index", "date"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    progress(f"saved csv: {out_csv} rows={len(out)}")


def plot_group(curves: Dict[Tuple[int, int], pd.DataFrame], out_png: Path, title: str) -> None:
    if not curves:
        progress(f"skip plot: no curves -> {out_png}")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ordered = sorted(
        curves.keys(),
        key=lambda key: (
            float(curves[key]["wse_span_m"].iloc[0]) if not curves[key].empty else -np.inf,
            int(key[0]),
            int(key[1]),
        ),
        reverse=True,
    )
    spans = np.array(
        [float(curves[key]["wse_span_m"].iloc[0]) for key in ordered if not curves[key].empty],
        dtype=float,
    )
    span_min = float(np.nanmin(spans)) if spans.size else 0.0
    span_max = float(np.nanmax(spans)) if spans.size else 1.0
    span_range = max(1e-9, span_max - span_min)
    highlight_cmap = plt.get_cmap("Reds")

    for key in ordered:
        df = curves[key]
        if df.empty:
            continue
        span = float(df["wse_span_m"].iloc[0])
        frac = (span - span_min) / span_range
        color = highlight_cmap(0.20 + 0.70 * frac)
        alpha = ALPHA_MIN + frac * (ALPHA_MAX - ALPHA_MIN)
        ax.plot(
            df["date"],
            df["wse_rel"],
            color=color,
            alpha=alpha,
            linewidth=CURVE_WIDTH,
            zorder=2.0 + frac * 0.5,
        )
        ax.scatter(
            df["date"],
            df["wse_rel"],
            s=POINT_SIZE,
            color=color,
            alpha=min(1.0, alpha + 0.12),
            edgecolors="none",
            zorder=2.1 + frac * 0.5,
        )

    ax.set_title(title, fontsize=20, pad=10)
    ax.set_xlabel("Date", fontsize=24, labelpad=10)
    ax.set_ylabel("Relative water level above min (m)", fontsize=24, labelpad=12)
    ax.grid(True, linestyle="--", alpha=0.22, linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="both", labelsize=20, width=1.2, length=6)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(bottom=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#222222")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=CURVE_DPI)
    plt.close(fig)
    progress(f"saved plot: {out_png}")


def split_curves_by_start_date(
    curves: Dict[Tuple[int, int], pd.DataFrame],
    split_date: pd.Timestamp,
) -> Tuple[Dict[Tuple[int, int], pd.DataFrame], Dict[Tuple[int, int], pd.DataFrame]]:
    on_date: Dict[Tuple[int, int], pd.DataFrame] = {}
    others: Dict[Tuple[int, int], pd.DataFrame] = {}
    for key, df in curves.items():
        if df is None or df.empty:
            continue
        start_date = pd.to_datetime(df["date"], errors="coerce").dropna().min()
        if pd.isna(start_date):
            continue
        if pd.Timestamp(start_date).normalize() == pd.Timestamp(split_date).normalize():
            on_date[key] = df
        else:
            others[key] = df
    return on_date, others


def main() -> None:
    start_ts = pd.to_datetime(DATE_RANGE_START)
    end_ts = pd.to_datetime(DATE_RANGE_END)
    split_date = pd.to_datetime(SPLIT_START_DATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for method_name in METHOD_NAMES:
        branch_dir = resolve_branch_table_dir(method_name)
        progress(f"method={method_name} branch_dir={branch_dir}")
        no_noise_dir = resolve_no_noise_dir(method_name)
        allowed_keys = load_no_noise_branch_keys(no_noise_dir)
        allowed_keys = {key for key in allowed_keys if key not in EXCLUDED_BRANCH_KEYS}
        progress(f"method={method_name} no_noise_keys={len(allowed_keys)} source={no_noise_dir}")
        curves = load_branch_curves(branch_dir, start_ts, end_ts, allowed_keys)
        progress(f"method={method_name} retained_branches={len(curves)}")

        method_out = OUTPUT_DIR / method_name
        save_points_table(curves, method_out / "07_4_min0_points_table.csv")
        plot_group(
            curves,
            method_out / "07_4_group_plot_min0_anchor.png",
            title=f"Group Plot - Min-Anchored Relative Water Level (all denoised branches) - {method_name}",
        )

        on_date_curves, other_curves = split_curves_by_start_date(curves, split_date)
        progress(
            f"method={method_name} split start={split_date.strftime('%Y-%m-%d')} "
            f"on_date={len(on_date_curves)} others={len(other_curves)}"
        )
        save_points_table(on_date_curves, method_out / "07_4_min0_points_table_start_0629.csv")
        save_points_table(other_curves, method_out / "07_4_min0_points_table_not_start_0629.csv")
        plot_group(
            on_date_curves,
            method_out / "07_4_group_plot_min0_anchor_start_0629.png",
            title=f"Group Plot - Curves starting on 2024-06-29 - {method_name}",
        )
        plot_group(
            other_curves,
            method_out / "07_4_group_plot_min0_anchor_not_start_0629.png",
            title=f"Group Plot - Curves not starting on 2024-06-29 - {method_name}",
        )

    progress(f"done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
