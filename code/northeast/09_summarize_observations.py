from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parent.parent
RESULT_DIR = PACKAGE_ROOT / "result" / "09_observation_statistics"
S2_RECORDS = RESULT_DIR / "s2_status_records.csv"
SWOT_RECORDS = RESULT_DIR / "swot_status_records.csv"
LAKE_SUMMARY = RESULT_DIR / "lake_observation_summary.csv"
BRANCH_SUMMARY = RESULT_DIR / "branch_observation_summary.csv"
FIT_SUMMARY = RESULT_DIR / "fit_unit_observation_summary.csv"
OVERVIEW = RESULT_DIR / "statistics_overview.csv"
WORKBOOK = RESULT_DIR / "observation_statistics.xlsx"

EFFECTIVE_INDEX = PACKAGE_ROOT / "result" / "03_effective_area" / "04_3_effective_boundary_index.csv"
FIT_MODELS = PACKAGE_ROOT / "result" / "08_area_wse_fits" / "fit_models_all.csv"
FIT_OBSERVATIONS = PACKAGE_ROOT / "result" / "08_area_wse_fits" / "area_wse_observations_all.csv"
FIT_ROUTING = PACKAGE_ROOT / "result" / "08_area_wse_fits" / "fit_routing_table.csv"

EXPECTED_S2_ROWS = 12750
EXPECTED_SWOT_ROWS = 7167
EXPECTED_LAKES = 150
EXPECTED_BRANCHES = 152
EXPECTED_FIT_LAKES = 79
EXPECTED_FIT_UNITS = 83


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def yyyymmdd(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def date_list(values) -> str:
    cleaned = sorted({pd.Timestamp(value).strftime("%Y-%m-%d") for value in values if pd.notna(value)})
    return ";".join(cleaned)


def daily_s2_status(group: pd.DataFrame) -> str:
    statuses = set(group["image_status"].fillna("").astype(str).str.lower())
    if "normal" in statuses:
        return "normal"
    if "cloud_covered" in statuses:
        return "cloud_covered"
    return "invalid"


def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s2 = pd.read_csv(S2_RECORDS, encoding="utf-8-sig")
    swot = pd.read_csv(SWOT_RECORDS, encoding="utf-8-sig")
    effective = pd.read_csv(EFFECTIVE_INDEX, encoding="utf-8-sig")

    s2["lake_id"] = pd.to_numeric(s2["lake_id"], errors="raise").astype(int)
    s2["date_dt"] = yyyymmdd(s2["image_date"])
    s2["image_status"] = s2["image_status"].fillna("").astype(str).str.lower()
    s2["boundary_status"] = s2["boundary_status"].fillna("").astype(str)
    s2["water_period_status"] = s2["water_period_status"].fillna("").astype(str)

    swot["lake_id"] = pd.to_numeric(swot["lake_id"], errors="raise").astype(int)
    swot["branch_index"] = pd.to_numeric(swot["branch_index"], errors="raise").astype(int)
    swot["date_dt"] = yyyymmdd(swot["swot_date"])
    swot["audit_status"] = swot["audit_status"].fillna("").astype(str).str.lower()
    swot["noise_status"] = swot["noise_status"].fillna("").astype(str)
    swot["invalid_reason"] = swot["invalid_reason"].fillna("").astype(str)
    swot["water_period_status"] = swot["water_period_status"].fillna("").astype(str)

    effective["lake_id"] = pd.to_numeric(effective["lake_id"], errors="raise").astype(int)
    effective["water_period_start"] = pd.to_datetime(effective["water_period_start"], errors="coerce")
    effective["water_period_end"] = pd.to_datetime(effective["water_period_end"], errors="coerce")
    return s2, swot, effective


def water_windows(effective: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    windows = (
        effective.groupby("lake_id", as_index=False)
        .agg(
            lake_type=("lake_type", "first"),
            water_period_start=("water_period_start", "min"),
            water_period_end=("water_period_end", "max"),
        )
    )
    areas = s2.groupby("lake_id", as_index=False).agg(lake_area_m2=("lake_area_m2", "first"))
    windows = areas.merge(windows, on="lake_id", how="left")
    windows["water_period_days"] = (
        windows["water_period_end"] - windows["water_period_start"]
    ).dt.days + 1
    return windows


def s2_metrics(s2: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[str, set]]]:
    data = s2.merge(windows[["lake_id", "water_period_start", "water_period_end"]], on="lake_id", how="left")
    in_window = (
        data["date_dt"].between(data["water_period_start"], data["water_period_end"], inclusive="both")
        & data["water_period_status"].eq("有水")
    )
    data = data.loc[in_window].copy()
    rows = []
    date_maps: dict[int, dict[str, set]] = {}
    for lake_id, group in data.groupby("lake_id", sort=True):
        daily = (
            group.groupby("date_dt", group_keys=False)
            .apply(daily_s2_status, include_groups=False)
            .rename("daily_status")
            .reset_index()
        )
        effective_dates = set(group.loc[group["boundary_status"].eq("有效边界"), "date_dt"].dropna())
        cloud_dates = set(group.loc[group["image_status"].eq("cloud_covered"), "date_dt"].dropna())
        date_maps[int(lake_id)] = {
            "effective": effective_dates,
            "cloud": cloud_dates,
            "all": set(daily["date_dt"].dropna()),
        }
        rows.append(
            {
                "lake_id": int(lake_id),
                "s2_scene_count": len(group),
                "s2_normal_scene_count": int(group["image_status"].eq("normal").sum()),
                "s2_cloud_scene_count": int(group["image_status"].eq("cloud_covered").sum()),
                "s2_invalid_scene_count": int(
                    (~group["image_status"].isin(["normal", "cloud_covered"])).sum()
                ),
                "s2_effective_boundary_scene_count": int(group["boundary_status"].eq("有效边界").sum()),
                "s2_observation_days": int(daily["date_dt"].nunique()),
                "s2_normal_days": int(daily["daily_status"].eq("normal").sum()),
                "s2_cloud_days": int(daily["daily_status"].eq("cloud_covered").sum()),
                "s2_invalid_days": int(daily["daily_status"].eq("invalid").sum()),
                "s2_effective_boundary_days": len(effective_dates),
            }
        )
    return pd.DataFrame(rows), date_maps


def swot_flags(group: pd.DataFrame) -> dict:
    valid_mask = group["audit_status"].eq("valid_wse")
    nonnoise = valid_mask & group["noise_status"].eq("非噪声")
    noise = valid_mask & group["noise_status"].eq("噪声")
    valid = bool(valid_mask.any())
    is_nonnoise = bool(nonnoise.any())
    is_noise = bool(valid and not is_nonnoise and noise.any())
    invalid = not valid
    return {
        "valid": valid,
        "nonnoise": is_nonnoise,
        "noise": is_noise,
        "invalid": invalid,
        "base_invalid": bool(invalid and group["invalid_reason"].str.contains("基础", na=False).any()),
        "clip_invalid": bool(invalid and group["invalid_reason"].str.contains("裁剪", na=False).any()),
        "date_dt": group["date_dt"].dropna().iloc[0] if group["date_dt"].notna().any() else pd.NaT,
    }


def aggregate_swot(
    swot: pd.DataFrame,
    keys: list[str],
    s2_dates: dict[int, dict[str, set]],
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple, dict[str, set]]]:
    data = swot.merge(
        windows[["lake_id", "water_period_start", "water_period_end"]],
        on="lake_id",
        how="left",
    )
    data = data.loc[
        data["date_dt"].between(data["water_period_start"], data["water_period_end"], inclusive="both")
        & data["audit_status"].isin(["valid_wse", "no_valid_wse"])
    ].copy()
    record_key = data["granule_name"].fillna("").astype(str)
    fallback = (
        data["swot_date"].astype(str) + "_" + data["swot_time"].astype(str) + "_" + data.index.astype(str)
    )
    data["record_key"] = record_key.where(record_key.ne(""), fallback)
    rows = []
    date_maps = {}
    for key, group in data.groupby(keys, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        lake_id = int(key_tuple[0])
        record_flags = []
        for _, record_group in group.groupby("record_key", sort=False):
            record_flags.append(swot_flags(record_group))
        records = pd.DataFrame(record_flags)
        if records.empty:
            continue
        all_dates = set(records["date_dt"].dropna())
        valid_dates = set(records.loc[records["valid"], "date_dt"].dropna())
        nonnoise_dates = set(records.loc[records["nonnoise"], "date_dt"].dropna())
        noise_dates = set(records.loc[records["noise"], "date_dt"].dropna())
        invalid_dates = set(records.loc[records["invalid"], "date_dt"].dropna())
        s2_effective = s2_dates[lake_id]["effective"]
        s2_cloud = s2_dates[lake_id]["cloud"]
        intersection = s2_effective & nonnoise_dates
        union = s2_effective | nonnoise_dates
        row = {
            "lake_id": lake_id,
            "swot_record_count": len(records),
            "swot_valid_wse_count": int(records["valid"].sum()),
            "swot_non_noise_count": int(records["nonnoise"].sum()),
            "swot_noise_count": int(records["noise"].sum()),
            "swot_invalid_wse_count": int(records["invalid"].sum()),
            "swot_base_filter_invalid_count": int(records["base_invalid"].sum()),
            "swot_clip_invalid_count": int(records["clip_invalid"].sum()),
            "swot_observation_days": len(all_dates),
            "swot_valid_days": len(valid_dates),
            "swot_non_noise_days": len(nonnoise_dates),
            "swot_noise_days": len(noise_dates),
            "swot_invalid_days": len(invalid_dates),
            "intersection_days": len(intersection),
            "union_days": len(union),
            "s2_only_days": len(s2_effective - nonnoise_dates),
            "swot_only_days": len(nonnoise_dates - s2_effective),
            "s2_cloud_swot_supplement_days": len(s2_cloud & nonnoise_dates),
            "s2_effective_date_list": date_list(s2_effective),
            "swot_non_noise_date_list": date_list(nonnoise_dates),
            "intersection_date_list": date_list(intersection),
            "s2_only_date_list": date_list(s2_effective - nonnoise_dates),
            "swot_only_date_list": date_list(nonnoise_dates - s2_effective),
            "s2_cloud_swot_supplement_date_list": date_list(s2_cloud & nonnoise_dates),
        }
        if len(keys) > 1:
            row["branch_index"] = int(key_tuple[1])
            row["branch_id"] = str(group["branch_id"].dropna().iloc[0]) if group["branch_id"].notna().any() else ""
        rows.append(row)
        date_maps[key_tuple] = {
            "all": all_dates,
            "valid": valid_dates,
            "nonnoise": nonnoise_dates,
            "noise": noise_dates,
            "invalid": invalid_dates,
        }
    return pd.DataFrame(rows), date_maps


def build_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s2, swot, effective = prepare_inputs()
    windows = water_windows(effective, s2)
    s2_table, s2_dates = s2_metrics(s2, windows)
    lake_swot, _ = aggregate_swot(swot, ["lake_id"], s2_dates, windows)
    branch_swot, _ = aggregate_swot(swot, ["lake_id", "branch_index"], s2_dates, windows)

    lake = windows.merge(s2_table, on="lake_id", how="inner").merge(lake_swot, on="lake_id", how="inner")
    branch = (
        branch_swot.merge(windows, on="lake_id", how="left")
        .merge(s2_table, on="lake_id", how="left")
        .sort_values(["lake_id", "branch_index"])
        .reset_index(drop=True)
    )

    models = pd.read_csv(FIT_MODELS, encoding="utf-8-sig")
    fit_obs = pd.read_csv(FIT_OBSERVATIONS, encoding="utf-8-sig")
    routing = pd.read_csv(FIT_ROUTING, encoding="utf-8-sig")
    aicc_lakes = set(pd.to_numeric(models["lake_id"], errors="raise").astype(int))
    lake["in_wse_curve_cohort"] = True
    lake["in_aicc_fit_cohort"] = lake["lake_id"].isin(aicc_lakes)

    fit_dates = fit_obs.copy()
    fit_dates["date_dt"] = yyyymmdd(fit_dates["date"])
    fit_range = (
        fit_dates.groupby("fit_unit_id", as_index=False)
        .agg(fit_observation_start=("date_dt", "min"), fit_observation_end=("date_dt", "max"))
    )
    fit = (
        models.merge(routing[["fit_unit_id", "processing_route"]], on="fit_unit_id", how="left")
        .merge(fit_range, on="fit_unit_id", how="left")
        .merge(lake, on="lake_id", how="left", suffixes=("", "_lake"))
    )
    fit["fit_observation_start"] = fit["fit_observation_start"].fillna(fit["water_period_start"])
    fit["fit_observation_end"] = fit["fit_observation_end"].fillna(fit["water_period_end"])
    fit["observation_statistics_scope"] = "lake water period; fit dates/model remain fit-unit specific"

    overview = build_overview(lake, branch, fit)
    return lake, branch, fit, overview


def summarize_rows(df: pd.DataFrame, cohort: str, unit_level: str, metrics: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for column, definition in metrics:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append(
            {
                "cohort": cohort,
                "unit_level": unit_level,
                "metric": column,
                "n": len(values),
                "total": float(values.sum()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "definition": definition,
            }
        )
    return rows


def build_overview(lake: pd.DataFrame, branch: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    observation_metrics = [
        ("water_period_days", "water-period duration"),
        ("s2_scene_count", "Sentinel-2 scenes within the water period"),
        ("s2_effective_boundary_scene_count", "Sentinel-2 scenes with effective lake boundaries"),
        ("s2_effective_boundary_days", "unique Sentinel-2 effective-boundary dates"),
        ("swot_record_count", "unique SWOT granules"),
        ("swot_valid_wse_count", "SWOT granules with valid WSE"),
        ("swot_non_noise_count", "valid non-noise SWOT granules"),
        ("swot_noise_count", "valid but noise-flagged SWOT granules"),
        ("swot_invalid_wse_count", "SWOT granules without valid WSE"),
        ("swot_non_noise_days", "unique non-noise SWOT dates"),
        ("intersection_days", "dates shared by effective S2 and non-noise SWOT"),
        ("union_days", "union of effective S2 and non-noise SWOT dates"),
        ("s2_only_days", "effective S2 dates without non-noise SWOT"),
        ("swot_only_days", "non-noise SWOT dates without effective S2"),
        ("s2_cloud_swot_supplement_days", "cloud-covered S2 dates supplemented by non-noise SWOT"),
    ]
    fit_metrics = [
        ("point_count", "points used in the final area-WSE fit"),
        ("selected_r2", "R2 of the AICc-selected model"),
        ("selected_rmse", "RMSE of the AICc-selected model, metres"),
    ]
    rows = []
    rows.extend(summarize_rows(lake, "wse_curve_cohort", "lake", observation_metrics))
    rows.extend(summarize_rows(branch, "wse_curve_cohort", "branch", observation_metrics))
    rows.extend(summarize_rows(fit, "aicc_fit_cohort", "fit_unit", fit_metrics))
    return pd.DataFrame(rows)


def write_outputs() -> None:
    lake, branch, fit, overview = build_summaries()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    lake.to_csv(LAKE_SUMMARY, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    branch.to_csv(BRANCH_SUMMARY, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    fit.to_csv(FIT_SUMMARY, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    overview.to_csv(OVERVIEW, index=False, encoding="utf-8-sig")
    validate(require_workbook=False)


def validate(require_workbook: bool = True) -> None:
    required = [
        S2_RECORDS,
        SWOT_RECORDS,
        LAKE_SUMMARY,
        BRANCH_SUMMARY,
        FIT_SUMMARY,
        OVERVIEW,
    ]
    if require_workbook:
        required.append(WORKBOOK)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing outputs:\n" + "\n".join(missing))

    s2 = pd.read_csv(S2_RECORDS, encoding="utf-8-sig")
    swot = pd.read_csv(SWOT_RECORDS, encoding="utf-8-sig")
    lake = pd.read_csv(LAKE_SUMMARY, encoding="utf-8-sig")
    branch = pd.read_csv(BRANCH_SUMMARY, encoding="utf-8-sig")
    fit = pd.read_csv(FIT_SUMMARY, encoding="utf-8-sig")
    errors = []
    checks = [
        (len(s2) == EXPECTED_S2_ROWS, f"S2 rows {len(s2)} != {EXPECTED_S2_ROWS}"),
        (len(swot) == EXPECTED_SWOT_ROWS, f"SWOT rows {len(swot)} != {EXPECTED_SWOT_ROWS}"),
        (lake["lake_id"].nunique() == EXPECTED_LAKES, f"lakes != {EXPECTED_LAKES}"),
        (len(branch) == EXPECTED_BRANCHES, f"branches {len(branch)} != {EXPECTED_BRANCHES}"),
        (len(fit) == EXPECTED_FIT_UNITS, f"fit units {len(fit)} != {EXPECTED_FIT_UNITS}"),
        (fit["lake_id"].nunique() == EXPECTED_FIT_LAKES, f"fit lakes != {EXPECTED_FIT_LAKES}"),
        ((pd.to_numeric(fit["selected_r2"], errors="coerce") > 0.8).all(), "selected R2 threshold failed"),
    ]
    for passed, message in checks:
        if not passed:
            errors.append(message)

    for frame_name, frame in (("lake", lake), ("branch", branch)):
        identity = (
            frame["intersection_days"] + frame["s2_only_days"] + frame["swot_only_days"]
            == frame["union_days"]
        )
        left = frame["intersection_days"] + frame["s2_only_days"] == frame["s2_effective_boundary_days"]
        right = frame["intersection_days"] + frame["swot_only_days"] == frame["swot_non_noise_days"]
        if not identity.all() or not left.all() or not right.all():
            errors.append(f"date-set identity failed: {frame_name}")

    branch_pairs = set(zip(branch["lake_id"].astype(int), branch["branch_index"].astype(int)))
    for expected in ((1, 1), (1, 2), (2, 1), (2, 2)):
        if expected not in branch_pairs:
            errors.append(f"missing branch pair {expected}")
    for lake_id in (1, 2):
        if int((fit["lake_id"].astype(int) == lake_id).sum()) != 3:
            errors.append(f"Lake {lake_id} does not have 3 final fit units")

    locked_branch_totals = {
        "s2_effective_boundary_scene_count": 4444,
        "swot_record_count": 5450,
        "swot_valid_wse_count": 5008,
        "swot_non_noise_count": 3442,
        "swot_noise_count": 1566,
        "swot_invalid_wse_count": 442,
    }
    for column, expected in locked_branch_totals.items():
        actual = int(pd.to_numeric(branch[column], errors="coerce").sum())
        if actual != expected:
            errors.append(f"locked total {column}: {actual} != {expected}")

    locked_date_totals = {
        "lake": {
            "s2_effective_boundary_days": 3234,
            "swot_non_noise_days": 2635,
            "intersection_days": 1941,
            "union_days": 3928,
            "s2_only_days": 1293,
            "swot_only_days": 694,
            "s2_cloud_swot_supplement_days": 633,
        },
        "branch": {
            "s2_effective_boundary_days": 3307,
            "swot_non_noise_days": 2679,
            "intersection_days": 1975,
            "union_days": 4011,
            "s2_only_days": 1332,
            "swot_only_days": 704,
            "s2_cloud_swot_supplement_days": 643,
        },
    }
    for frame_name, frame in (("lake", lake), ("branch", branch)):
        for column, expected in locked_date_totals[frame_name].items():
            actual = int(pd.to_numeric(frame[column], errors="coerce").sum())
            if actual != expected:
                errors.append(f"locked {frame_name} date total {column}: {actual} != {expected}")

    if errors:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(errors))
    print(
        f"VALID: {len(lake)} lakes, {len(branch)} branches, {len(fit)} AICc fit units; "
        f"S2={len(s2)}, SWOT={len(swot)} records."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate unified lake observation statistics.")
    parser.add_argument("--run", action="store_true", help="recompute summary CSV files")
    args = parser.parse_args()
    if args.run:
        write_outputs()
    else:
        validate(require_workbook=True)


if __name__ == "__main__":
    main()
