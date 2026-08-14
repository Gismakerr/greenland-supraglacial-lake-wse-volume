from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result" / "10_lake_volume_change"
MODELS_CSV = ROOT / "result" / "08_area_wse_fits" / "fit_models_all.csv"
CURVES_CSV = ROOT / "result" / "08_area_wse_fits" / "fitted_curves_all.csv"
INPUT_CSV = RESULT_DIR / "area_wse_timeseries_input.csv"
SERIES_CSV = RESULT_DIR / "volume_timeseries_all.csv"
SUMMARY_CSV = RESULT_DIR / "volume_summary_all.csv"
STORAGE_CURVES_CSV = RESULT_DIR / "storage_curves_all.csv"
FIGURE_DIR = RESULT_DIR / "figures"
REPRESENTATIVE_PNG = RESULT_DIR / "representative_6_volume_stacked.png"
REPRESENTATIVE_TIF = RESULT_DIR / "representative_6_volume_stacked.tif"

EXPECTED_LAKES = 79
EXPECTED_UNITS = 83
EXPECTED_INPUT_ROWS = 2219
REPRESENTATIVE_LAKES = [1, 2, 38, 4, 70, 30]
REPRESENTATIVE_DISPLAY_IDS = {1: 1, 2: 2, 38: 3, 4: 4, 70: 5, 30: 6}
MERGE_DATES = {1: pd.Timestamp("2024-07-12"), 2: pd.Timestamp("2024-07-31")}
FIT_WINDOWS = {
    1: [("inflow", "2024-06-26", "2024-07-28"), ("drain", "2024-07-28", "2024-08-14")],
    2: [("inflow", "2024-06-26", "2024-08-06")],
    38: [("inflow", "2024-06-23", "2024-07-07"), ("drain", "2024-07-07", "2024-07-13")],
    4: [("inflow", "2024-06-28", "2024-08-10")],
    70: [("inflow", "2024-07-03", "2024-07-12"), ("drain", "2024-07-12", "2024-07-15")],
    30: [("inflow", "2024-07-05", "2024-07-20"), ("drain", "2024-07-20", "2024-07-21")],
}

BLACK = "#1f1f1f"
ERROR = "#404040"
BRANCH_1 = "#8fb8ff"
BRANCH_2 = "#f4a6a6"
MERGED = "#d8d8d8"
X_MIN = pd.Timestamp("2024-06-22")
X_MAX = pd.Timestamp("2024-08-17")
X_TICKS = pd.to_datetime(
    ["2024-06-25", "2024-07-05", "2024-07-15", "2024-07-25", "2024-08-05", "2024-08-15"]
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def parse_coefficients(text: str) -> np.ndarray:
    values = [float(value) for value in str(text).split(";") if str(value).strip()]
    if len(values) not in {2, 3}:
        raise ValueError(f"Unsupported coefficient list: {text}")
    return np.asarray(values, dtype=float)


def reference_integral_volume(
    curve_wse_m: np.ndarray,
    curve_area_m2: np.ndarray,
    query_wse_m: np.ndarray,
    reference_wse_m: float,
) -> np.ndarray:
    """Numerically evaluate integral(reference_wse..query_wse) A(h) dh."""
    frame = pd.DataFrame(
        {
            "wse": np.asarray(curve_wse_m, dtype=float),
            "area": np.asarray(curve_area_m2, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.groupby("wse", as_index=False)["area"].mean().sort_values("wse")
    if len(frame) < 2:
        raise ValueError("At least two distinct WSE values are required for storage integration.")
    h = frame["wse"].to_numpy(float)
    area = frame["area"].to_numpy(float)
    cumulative = np.r_[0.0, np.cumsum(np.diff(h) * (area[:-1] + area[1:]) / 2.0)]
    query = np.asarray(query_wse_m, dtype=float)
    query_cumulative = np.interp(query, h, cumulative)
    reference_cumulative = float(np.interp(reference_wse_m, h, cumulative))
    return query_cumulative - reference_cumulative


def calculate_timeseries(
    models: pd.DataFrame, source: pd.DataFrame, curves: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    curve_groups = {str(key): value.copy() for key, value in curves.groupby("fit_unit_id")}

    for model in models.itertuples(index=False):
        fit_unit_id = str(model.fit_unit_id)
        group = source.loc[source["fit_unit_id"].astype(str) == fit_unit_id].copy()
        group["date_dt"] = pd.to_datetime(group["date"], errors="coerce")
        for column in ["area_km2", "source_wse_m", "wse_std_m"]:
            group[column] = pd.to_numeric(group[column], errors="coerce")
        group = (
            group.dropna(subset=["date_dt", "area_km2", "source_wse_m"])
            .sort_values("date_dt")
            .drop_duplicates("date_dt", keep="last")
            .reset_index(drop=True)
        )
        if group.empty:
            raise ValueError(f"No usable time-series rows for {fit_unit_id}")

        coefficients = parse_coefficients(model.selected_coefficients)
        group["fitted_wse_m"] = np.polyval(coefficients, group["area_km2"].to_numpy(float))
        group["wse_fit_residual_m"] = group["source_wse_m"] - group["fitted_wse_m"]
        predicted = as_bool(group["is_wse_predicted"])
        group["wse_for_volume_m"] = group["source_wse_m"]
        group.loc[predicted, "wse_for_volume_m"] = group.loc[predicted, "fitted_wse_m"]
        group["area_m2"] = group["area_km2"] * 1.0e6

        fit_curve = curve_groups[fit_unit_id]
        curve_h = np.r_[
            pd.to_numeric(fit_curve["fitted_wse_m"], errors="raise").to_numpy(float),
            group["wse_for_volume_m"].to_numpy(float),
        ]
        curve_area = np.r_[
            pd.to_numeric(fit_curve["area_km2"], errors="raise").to_numpy(float) * 1.0e6,
            group["area_m2"].to_numpy(float),
        ]
        reference_wse_m = float(group["wse_for_volume_m"].median())
        reference_index = int((group["wse_for_volume_m"] - reference_wse_m).abs().idxmin())
        reference_date = group.loc[reference_index, "date_dt"].strftime("%Y-%m-%d")
        reference_area_m2 = float(group.loc[reference_index, "area_m2"])
        volume_from_reference = reference_integral_volume(
            curve_h,
            curve_area,
            group["wse_for_volume_m"].to_numpy(float),
            reference_wse_m,
        )
        volume_m3 = volume_from_reference - float(volume_from_reference[0])

        sigma = (
            group["wse_std_m"]
            .fillna(float(model.selected_rmse))
            .clip(lower=0.0)
            .to_numpy(float)
        )
        first_area_m2 = float(group.loc[0, "area_m2"])
        uncertainty = np.sqrt(
            np.square(group["area_m2"].to_numpy(float) * sigma)
            + np.square(first_area_m2 * float(sigma[0]))
        )
        uncertainty[0] = 0.0

        group["selected_model"] = str(model.selected_model).lower()
        group["selected_r2"] = float(model.selected_r2)
        group["volume_formula"] = "reference_wse_integral_A(h)_dh"
        group["reference_date"] = reference_date
        group["reference_wse_m"] = reference_wse_m
        group["reference_area_km2"] = reference_area_m2 / 1.0e6
        group["first_observation_zero_date"] = group.loc[0, "date_dt"].strftime("%Y-%m-%d")
        group["volume_from_reference_m3"] = volume_from_reference
        group["volume_m3"] = volume_m3
        group["volume_1e6_m3"] = volume_m3 / 1.0e6
        group["volume_uncertainty_m3"] = uncertainty
        group["volume_uncertainty_1e6_m3"] = uncertainty / 1.0e6
        group["volume_lower_m3"] = volume_m3 - uncertainty
        group["volume_upper_m3"] = volume_m3 + uncertainty
        group["delta_days"] = group["date_dt"].diff().dt.days
        group["delta_area_km2"] = group["area_km2"].diff().fillna(0.0)
        group["delta_wse_m"] = group["wse_for_volume_m"].diff().fillna(0.0)
        group["delta_volume_m3"] = group["volume_m3"].diff().fillna(0.0)
        group["daily_volume_change_m3_day"] = (
            group["delta_volume_m3"] / group["delta_days"].replace(0, np.nan)
        ).fillna(0.0)
        group["date"] = group["date_dt"].dt.strftime("%Y-%m-%d")

        keep = [
            "fit_unit_id", "lake_id", "phase", "branch_label", "date", "area_km2", "area_m2",
            "source_wse_m", "fitted_wse_m", "wse_for_volume_m", "wse_fit_residual_m",
            "wse_std_m", "is_wse_predicted", "is_area_interpolated", "is_area_predicted",
            "selected_model", "selected_r2", "volume_formula", "reference_date",
            "reference_wse_m", "reference_area_km2", "first_observation_zero_date",
            "volume_from_reference_m3", "volume_m3", "volume_1e6_m3",
            "volume_uncertainty_m3", "volume_uncertainty_1e6_m3", "volume_lower_m3",
            "volume_upper_m3", "delta_days", "delta_area_km2", "delta_wse_m",
            "delta_volume_m3", "daily_volume_change_m3_day", "source_kind",
            "source_provenance",
        ]
        series_rows.append(group[keep])
        summary_rows.append(
            {
                "fit_unit_id": fit_unit_id,
                "lake_id": int(model.lake_id),
                "phase": str(model.phase),
                "branch_label": str(model.branch_label),
                "selected_model": str(model.selected_model).lower(),
                "selected_r2": float(model.selected_r2),
                "volume_formula": "reference_wse_integral_A(h)_dh",
                "n_dates": len(group),
                "first_date": group.loc[0, "date"],
                "last_date": group.loc[len(group) - 1, "date"],
                "reference_date": reference_date,
                "reference_wse_m": reference_wse_m,
                "reference_area_km2": reference_area_m2 / 1.0e6,
                "min_volume_1e6_m3": group["volume_1e6_m3"].min(),
                "max_volume_1e6_m3": group["volume_1e6_m3"].max(),
                "range_volume_1e6_m3": group["volume_1e6_m3"].max() - group["volume_1e6_m3"].min(),
                "max_fill_rate_1e6_m3_day": group["daily_volume_change_m3_day"].max() / 1.0e6,
                "max_drain_rate_1e6_m3_day": group["daily_volume_change_m3_day"].min() / 1.0e6,
                "source_kind": ";".join(sorted(set(group["source_kind"].astype(str)))),
            }
        )
    return pd.concat(series_rows, ignore_index=True), pd.DataFrame(summary_rows)


def calculate_storage_curves(models: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    model_index = models.set_index("fit_unit_id")
    rows: list[pd.DataFrame] = []
    for fit_unit_id, group in curves.groupby("fit_unit_id", sort=False):
        model = model_index.loc[str(fit_unit_id)]
        group = group.sort_values("sequence").copy()
        area_m2 = pd.to_numeric(group["area_km2"], errors="raise").to_numpy(float) * 1.0e6
        h = pd.to_numeric(group["fitted_wse_m"], errors="raise").to_numpy(float)
        storage = reference_integral_volume(h, area_m2, h, float(h[0]))
        group["area_m2"] = area_m2
        group["phase"] = str(model["phase"])
        group["branch_label"] = str(model["branch_label"])
        group["selected_r2"] = float(model["selected_r2"])
        group["volume_formula"] = "reference_wse_integral_A(h)_dh"
        group["reference_area_km2"] = float(group.iloc[0]["area_km2"])
        group["reference_wse_m"] = float(h[0])
        group["relative_storage_m3"] = storage
        group["relative_storage_1e6_m3"] = storage / 1.0e6
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def prepare_lake_series(series: pd.DataFrame, lake_id: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    lake = series.loc[series["lake_id"] == lake_id].copy()
    lake["date_dt"] = pd.to_datetime(lake["date"], errors="coerce")
    lake = lake.sort_values("date_dt")
    if lake_id not in MERGE_DATES:
        return lake.rename(columns={"volume_1e6_m3": "total", "volume_uncertainty_1e6_m3": "unc"}), None

    merge_date = MERGE_DATES[lake_id]
    pre = lake.loc[(lake["phase"] == "premerge") & (lake["date_dt"] < merge_date)].copy()
    value_pivot = pre.pivot_table(index="date_dt", columns="branch_label", values="volume_1e6_m3", aggfunc="first")
    unc_pivot = pre.pivot_table(
        index="date_dt", columns="branch_label", values="volume_uncertainty_1e6_m3", aggfunc="first"
    )
    combined = value_pivot.join(unc_pivot, lsuffix="_volume", rsuffix="_unc").reset_index()
    for column in ["B1_volume", "B2_volume", "B1_unc", "B2_unc"]:
        if column not in combined:
            combined[column] = np.nan
    combined["branch1"] = combined["B1_volume"]
    combined["branch2"] = combined["B2_volume"]
    combined["total"] = combined[["branch1", "branch2"]].sum(axis=1, min_count=1)
    combined["unc"] = np.sqrt(
        np.square(combined["B1_unc"].fillna(0.0)) + np.square(combined["B2_unc"].fillna(0.0))
    )

    post = lake.loc[(lake["phase"] == "postmerge") & (lake["date_dt"] >= merge_date)].copy()
    post = post.rename(columns={"volume_1e6_m3": "total", "volume_uncertainty_1e6_m3": "unc"})
    if not post.empty:
        anchor = float(combined["total"].dropna().iloc[-1]) if not combined.empty else 0.0
        post["total"] = post["total"] - float(post["total"].iloc[0]) + anchor
    total = pd.concat([combined[["date_dt", "total", "unc"]], post[["date_dt", "total", "unc"]]])
    return total.sort_values("date_dt"), combined


def fit_segments(data: pd.DataFrame, lake_id: int) -> list[dict]:
    segments: list[dict] = []
    for label, start, end in FIT_WINDOWS.get(lake_id, []):
        sub = data.loc[
            data["date_dt"].between(pd.Timestamp(start), pd.Timestamp(end))
        ].dropna(subset=["total"])
        if len(sub) < 2:
            continue
        x = (sub["date_dt"] - sub["date_dt"].min()).dt.total_seconds().to_numpy(float) / 86400.0
        y = sub["total"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        segments.append(
            {
                "label": label,
                "dates": sub["date_dt"],
                "fitted": fitted,
                "slope": slope,
                "r2": r2,
            }
        )
    return segments


def style_axis(ax: plt.Axes, lake_id: int, show_xlabel: bool = True) -> None:
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="both", labelsize=12, width=1.2, length=6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.2)
    display_id = REPRESENTATIVE_DISPLAY_IDS.get(lake_id, lake_id)
    ax.text(0.015, 0.94, f"Lake {display_id}", transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold")
    ax.text(-0.01, 1.02, r"$10^6\ \mathrm{m^3}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12)
    if show_xlabel:
        ax.set_xlabel("Date", fontsize=13)


def draw_lake(ax: plt.Axes, series: pd.DataFrame, lake_id: int, show_xlabel: bool = True) -> None:
    total, branches = prepare_lake_series(series, lake_id)
    if branches is not None and not branches.empty:
        x = branches["date_dt"].to_numpy()
        b1 = branches["branch1"].to_numpy(float)
        b2 = branches["branch2"].to_numpy(float)
        ax.fill_between(x, 0.0, b1, color=BRANCH_1, alpha=0.8, linewidth=0, label="Branch 1")
        ax.fill_between(x, b1, b1 + np.nan_to_num(b2), color=BRANCH_2, alpha=0.8,
                        linewidth=0, label="Branch 2")
        post = total.loc[total["date_dt"] >= MERGE_DATES[lake_id]]
        if not post.empty:
            bridge_x = np.r_[branches["date_dt"].iloc[-1].to_datetime64(), post["date_dt"].to_numpy()]
            bridge_y = np.r_[post["total"].iloc[0], post["total"].to_numpy(float)]
            ax.fill_between(bridge_x, 0.0, bridge_y, color=MERGED, alpha=0.55,
                            linewidth=0, label="Merged")

    x = total["date_dt"]
    y = pd.to_numeric(total["total"], errors="coerce").to_numpy(float)
    unc = pd.to_numeric(total["unc"], errors="coerce").fillna(0.0).to_numpy(float)
    ax.plot(x, y, color=BLACK, linewidth=1.8, zorder=4)
    ax.errorbar(x, y, yerr=unc, fmt="o", markersize=5.8, color=BLACK, ecolor=ERROR,
                elinewidth=1.0, capsize=2.4, markeredgewidth=0, zorder=5)

    for index, segment in enumerate(fit_segments(total, lake_id)):
        ax.plot(segment["dates"], segment["fitted"], "--", color=BLACK, linewidth=1.3, zorder=6)
        middle = len(segment["dates"]) // 2
        x_text = segment["dates"].iloc[middle]
        y_text = float(segment["fitted"][middle])
        offset = 11 if index % 2 == 0 else -18
        sign = "+" if segment["slope"] >= 0 else ""
        label = f"k={sign}{segment['slope']:.2f}, $R^2$={segment['r2']:.2f}"
        ax.annotate(label, (x_text, y_text), xytext=(0, offset), textcoords="offset points",
                    ha="center", va="bottom" if offset > 0 else "top", fontsize=10)

    style_axis(ax, lake_id, show_xlabel)
    finite = np.r_[y - unc, y + unc]
    finite = finite[np.isfinite(finite)]
    if finite.size:
        span = max(float(finite.max() - finite.min()), 0.1)
        ax.set_ylim(float(finite.min() - 0.10 * span), float(finite.max() + 0.20 * span))
    if branches is not None:
        ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=3)


def render_all_figures(series: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for lake_id in sorted(series["lake_id"].astype(int).unique()):
        filename = f"lake_{lake_id:03d}_volume.png"
        expected.add(filename)
        fig, ax = plt.subplots(figsize=(12.0, 4.1), dpi=300)
        draw_lake(ax, series, lake_id)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.93)
        fig.savefig(FIGURE_DIR / filename, dpi=300, facecolor="white")
        plt.close(fig)
    for old in FIGURE_DIR.glob("*.png"):
        if old.name not in expected:
            old.unlink()

    fig, axes = plt.subplots(len(REPRESENTATIVE_LAKES), 1, figsize=(12.0, 19.2), dpi=300)
    for index, (ax, lake_id) in enumerate(zip(axes, REPRESENTATIVE_LAKES)):
        draw_lake(ax, series, lake_id, show_xlabel=index == len(REPRESENTATIVE_LAKES) - 1)
        if index < len(REPRESENTATIVE_LAKES) - 1:
            ax.tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.055, top=0.99, hspace=0.16)
    fig.savefig(REPRESENTATIVE_PNG, dpi=300, facecolor="white")
    fig.savefig(REPRESENTATIVE_TIF, dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def validate() -> None:
    required = [MODELS_CSV, CURVES_CSV, INPUT_CSV, SERIES_CSV, SUMMARY_CSV,
                STORAGE_CURVES_CSV, REPRESENTATIVE_PNG, REPRESENTATIVE_TIF]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing stage-10 files:\n" + "\n".join(missing))
    models = pd.read_csv(MODELS_CSV, encoding="utf-8-sig")
    source = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    series = pd.read_csv(SERIES_CSV, encoding="utf-8-sig")
    summary = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    curves = pd.read_csv(STORAGE_CURVES_CSV, encoding="utf-8-sig")
    units = set(models["fit_unit_id"].astype(str))
    for name, table in [("input", source), ("series", series), ("summary", summary), ("curves", curves)]:
        actual = set(table["fit_unit_id"].astype(str))
        if actual != units:
            raise ValueError(f"{name} fit-unit cohort differs from stage 08.")
    if len(models) != EXPECTED_UNITS or models["lake_id"].nunique() != EXPECTED_LAKES:
        raise ValueError("Stage-08 model cohort changed.")
    if len(source) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"Input row count changed: {len(source)} != {EXPECTED_INPUT_ROWS}")
    if len(summary) != EXPECTED_UNITS:
        raise ValueError("Volume summary must contain one row per fit unit.")
    if not (pd.to_numeric(models["selected_r2"], errors="raise") > 0.8).all():
        raise ValueError("A final model no longer satisfies R2 > 0.8.")
    if series.duplicated(["fit_unit_id", "date"]).any():
        raise ValueError("Duplicate fit-unit/date rows found.")
    predicted = as_bool(series["is_wse_predicted"])
    if not np.allclose(series.loc[predicted, "wse_for_volume_m"], series.loc[predicted, "fitted_wse_m"]):
        raise ValueError("Predicted WSE rows do not use the final stage-08 model.")
    if not np.allclose(series.groupby("fit_unit_id")["volume_m3"].first(), 0.0, atol=1e-7):
        raise ValueError("A time series is not zeroed at its first observation.")
    if not np.allclose(curves.groupby("fit_unit_id")["relative_storage_m3"].first(), 0.0, atol=1e-7):
        raise ValueError("A storage curve is not zeroed at its first point.")
    if set(series["volume_formula"]) != {"reference_wse_integral_A(h)_dh"}:
        raise ValueError("Unexpected volume method.")
    figures = sorted(FIGURE_DIR.glob("*.png"))
    expected_names = {f"lake_{lake_id:03d}_volume.png" for lake_id in series["lake_id"].unique()}
    if {path.name for path in figures} != expected_names:
        raise ValueError("Lake-level figure cohort is incomplete.")
    if any(path.stat().st_size < 20_000 for path in figures + [REPRESENTATIVE_PNG, REPRESENTATIVE_TIF]):
        raise ValueError("A stage-10 figure appears incomplete.")
    print(
        f"VALID: {EXPECTED_UNITS} AICc fit units, {EXPECTED_LAKES} lake-level figures, "
        f"{len(series)} dated rows; volume = integral A(h) dh."
    )


def run() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    models = pd.read_csv(MODELS_CSV, encoding="utf-8-sig")
    curves = pd.read_csv(CURVES_CSV, encoding="utf-8-sig")
    source = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    series, summary = calculate_timeseries(models, source, curves)
    storage_curves = calculate_storage_curves(models, curves)
    series.to_csv(SERIES_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    storage_curves.to_csv(STORAGE_CURVES_CSV, index=False, encoding="utf-8-sig")
    render_all_figures(series)
    validate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate relative lake storage from the final AICc area-WSE curves."
    )
    parser.add_argument("--run", action="store_true", help="Recompute stage-10 outputs.")
    args = parser.parse_args()
    run() if args.run else validate()


if __name__ == "__main__":
    main()
