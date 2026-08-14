import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result1"
STAGE_DIR = (
    RESULT_ROOT
    / "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"
    / "Otsu"
    / "original"
    / "06_3_BranchTables"
    / "vC"
)
OUT_PNG = (
    RESULT_ROOT
    / "07_4_lake_elevation_curves_classification"
    / "Otsu"
    / "GroupPlot_RelWSE_SameColor_Denoised_Selected6.png"
)

DATE_START = pd.Timestamp("2024-06-05")
DATE_END = pd.Timestamp("2024-08-15")

# category: (label, color, [(lake_id, branch_index), ...])
GROUPS = [
    ("Slow Drainage", "#9E9E9E", [(20, 1), (36, 1)]),
    ("Keep Filling", "#52B976", [(33, 1), (61, 1)]),
    ("Repeated Filling–Drainage", "#000000", [(258, 1), (190, 1)]),
]


def load_rel_curve(lake_id: int, branch_index: int):
    csv_path = STAGE_DIR / f"lake_{lake_id}_branch_{branch_index}_stage_vC.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "date" not in df.columns or "wse_m" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wse_m"] = pd.to_numeric(df["wse_m"], errors="coerce")
    if "is_noise" in df.columns:
        noise = df["is_noise"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        df = df.loc[~noise].copy()
    # Keep all valid denoised points in this branch (including gray points),
    # instead of only red points, so point count matches single-lake plots.

    df = df.loc[df["date"].notna() & df["wse_m"].notna()].copy()
    df = df.loc[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].copy()
    if df.empty:
        return None

    if "wse_std" in df.columns:
        df["wse_std"] = pd.to_numeric(df["wse_std"], errors="coerce")
    else:
        df["wse_std"] = np.nan

    daily = (
        df.assign(date_day=df["date"].dt.normalize())
        .groupby("date_day", as_index=False)
        .agg(wse_m=("wse_m", "mean"), wse_std=("wse_std", "mean"))
        .sort_values("date_day")
        .reset_index(drop=True)
    )
    if daily.empty:
        return None

    base = float(daily["wse_m"].iloc[0])
    daily["wse_rel"] = daily["wse_m"] - base
    return daily


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 5.6), dpi=200)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    legend_handles = []

    for label, color, pairs in GROUPS:
        # legend uses one representative line per category
        legend_handles.append(Line2D([0], [0], color=color, lw=1.0, label=label))
        for idx, (lake_id, branch_index) in enumerate(pairs):
            daily = load_rel_curve(lake_id, branch_index)
            if daily is None:
                continue

            x = daily["date_day"]
            # Rebase each curve so its own minimum is exactly 0.
            y = daily["wse_rel"] - float(daily["wse_rel"].min())
            # first in pair darker, second lighter
            alpha_line = 0.95 if idx == 0 else 0.65
            ax.plot(x, y, color=color, linewidth=1.0, alpha=alpha_line)
            ax.scatter(x, y, color=color, s=8, alpha=min(1.0, alpha_line + 0.05), edgecolors="none")

            std = pd.to_numeric(daily["wse_std"], errors="coerce")
            if std.notna().any():
                lo = y - 2.0 * std.fillna(0.0)
                hi = y + 2.0 * std.fillna(0.0)
                lo = np.maximum(lo, 0.0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.06 if idx == 0 else 0.04, linewidth=0)

    ax.set_xlim(pd.Timestamp("2024-06-03"), pd.Timestamp("2024-08-17"))
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    tick_dates = pd.date_range("2024-06-05", "2024-08-14", freq="10D")
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Relative Water Level Change (m)", fontsize=16)
    ax.grid(True, linestyle="--", color="#E1E1E1", alpha=0.9, linewidth=0.4)

    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=16)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
