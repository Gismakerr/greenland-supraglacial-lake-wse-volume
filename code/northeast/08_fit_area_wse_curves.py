from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parent.parent
RESULT_DIR = PACKAGE_ROOT / "result" / "08_area_wse_fits"
OBS_CSV = RESULT_DIR / "area_wse_observations_all.csv"
MODEL_CSV = RESULT_DIR / "fit_models_all.csv"
CURVE_CSV = RESULT_DIR / "fitted_curves_all.csv"
ROUTING_CSV = RESULT_DIR / "fit_routing_table.csv"
FIGURE_DIR = RESULT_DIR / "figures"
MIN_R2 = 0.8
EXPECTED_UNITS = 83
EXPECTED_LAKES = 79
LAKE2_EXCLUDED_DATES = {"20240811", "20240812"}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def safe_aicc(n: int, rss: float, k: int) -> float:
    if n <= k + 1 or rss <= 0:
        return math.nan
    aic = n * math.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def fit_candidate(x: np.ndarray, y: np.ndarray, degree: int, count_residual_variance: bool) -> dict:
    minimum = 5 if degree == 2 else 4
    if len(x) < minimum:
        return {
            "ok": False,
            "coefficients": "",
            "r2": math.nan,
            "rmse": math.nan,
            "aicc": math.nan,
        }
    coefficients = np.polyfit(x, y, degree)
    predicted = np.polyval(coefficients, x)
    rss = float(np.sum((y - predicted) ** 2))
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = math.nan if tss <= 0 else 1.0 - rss / tss
    return {
        "ok": True,
        "coefficients": ";".join(f"{float(value):.12g}" for value in coefficients),
        "r2": r2,
        "rmse": float(np.sqrt(rss / len(x))),
        "aicc": safe_aicc(len(x), rss, degree + 2 if count_residual_variance else degree + 1),
    }


def choose_model(linear: dict, quadratic: dict) -> str | None:
    passing = []
    for name, fit in (("linear", linear), ("quadratic", quadratic)):
        if fit["ok"] and np.isfinite(fit["r2"]) and float(fit["r2"]) > MIN_R2:
            passing.append((name, fit))
    if not passing:
        return None
    return min(
        passing,
        key=lambda item: (
            math.inf if not np.isfinite(item[1]["aicc"]) else float(item[1]["aicc"]),
            -float(item[1]["r2"]),
        ),
    )[0]


def compute_models(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    used = observations.loc[as_bool(observations["fit_used"])].copy()
    model_rows: list[dict] = []
    curve_rows: list[dict] = []
    for unit_id, group in used.groupby("fit_unit_id", sort=True):
        group = group.dropna(subset=["area_km2", "wse_m"])
        x = group["area_km2"].to_numpy(dtype=float)
        y = group["wse_m"].to_numpy(dtype=float)
        source_kind = str(group.iloc[0]["source_kind"])
        count_residual_variance = source_kind == "ordinary_scatter_normal_points"
        linear = fit_candidate(x, y, 1, count_residual_variance)
        quadratic = fit_candidate(x, y, 2, count_residual_variance)
        selected = choose_model(linear, quadratic)
        if selected is None:
            raise RuntimeError(f"No AICc candidate passes strict R² > {MIN_R2}: {unit_id}")
        chosen = linear if selected == "linear" else quadratic
        meta = group.iloc[0]
        model_rows.append(
            {
                "fit_unit_id": unit_id,
                "lake_id": int(meta["lake_id"]),
                "phase": meta["phase"],
                "branch_label": meta["branch_label"],
                "point_count": len(group),
                "linear_r2": linear["r2"],
                "linear_rmse": linear["rmse"],
                "linear_aicc": linear["aicc"],
                "linear_coefficients": linear["coefficients"],
                "quadratic_r2": quadratic["r2"],
                "quadratic_rmse": quadratic["rmse"],
                "quadratic_aicc": quadratic["aicc"],
                "quadratic_coefficients": quadratic["coefficients"],
                "selected_model": selected,
                "selected_r2": chosen["r2"],
                "selected_rmse": chosen["rmse"],
                "selected_aicc": chosen["aicc"],
                "selected_coefficients": chosen["coefficients"],
                "selection_rule": "minimum finite AICc among candidates with R2 > 0.8",
                "aicc_parameterization": (
                    "regression coefficients + residual variance (locked ordinary workflow)"
                    if count_residual_variance
                    else "regression coefficients (locked Lake 1/2/38 workflow)"
                ),
            }
        )
        coefficients = np.array([float(v) for v in chosen["coefficients"].split(";")], dtype=float)
        area_grid = np.linspace(float(np.min(x)), float(np.max(x)), 101)
        for sequence, (area, wse) in enumerate(zip(area_grid, np.polyval(coefficients, area_grid))):
            curve_rows.append(
                {
                    "fit_unit_id": unit_id,
                    "lake_id": int(meta["lake_id"]),
                    "sequence": sequence,
                    "area_km2": float(area),
                    "fitted_wse_m": float(wse),
                    "selected_model": selected,
                }
            )
    return pd.DataFrame(model_rows), pd.DataFrame(curve_rows)


def validate() -> None:
    required = [OBS_CSV, MODEL_CSV, CURVE_CSV, ROUTING_CSV]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs:\n" + "\n".join(missing))
    observations = pd.read_csv(OBS_CSV, encoding="utf-8-sig")
    models = pd.read_csv(MODEL_CSV, encoding="utf-8-sig")
    routing = pd.read_csv(ROUTING_CSV, encoding="utf-8-sig")
    figures = sorted(FIGURE_DIR.glob("*.png"))
    errors = []
    if len(models) != EXPECTED_UNITS:
        errors.append(f"fit units: {len(models)} != {EXPECTED_UNITS}")
    if models["lake_id"].nunique() != EXPECTED_LAKES:
        errors.append(f"unique lakes: {models['lake_id'].nunique()} != {EXPECTED_LAKES}")
    if len(figures) != EXPECTED_UNITS:
        errors.append(f"figures: {len(figures)} != {EXPECTED_UNITS}")
    if len(routing) != EXPECTED_UNITS:
        errors.append(f"routing rows: {len(routing)} != {EXPECTED_UNITS}")
    if not (pd.to_numeric(models["selected_r2"], errors="coerce") > MIN_R2).all():
        errors.append("at least one selected model does not satisfy strict R² > 0.8")
    if models["fit_unit_id"].isin(["lake_099_original_b01", "lake_121_original_b01"]).any():
        errors.append("Lake 99 or Lake 121 was not excluded")
    lake2 = models.loc[models["fit_unit_id"] == "lake_2_merged_postmerge"]
    if len(lake2) != 1 or int(lake2.iloc[0]["point_count"]) != 5 or lake2.iloc[0]["selected_model"] != "linear":
        errors.append("Lake 2 postmerge is not the locked 5-point AICc-selected linear fit")
    excluded = observations.loc[
        (observations["fit_unit_id"] == "lake_2_merged_postmerge")
        & ~as_bool(observations["fit_used"])
    ]
    excluded_dates = set(excluded["date"].astype(str).str.replace(r"\.0$", "", regex=True))
    if excluded_dates != LAKE2_EXCLUDED_DATES:
        errors.append("Lake 2 excluded-date audit does not match 20240811/20240812")
    if errors:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(errors))
    print(
        f"VALID: {len(models)} fit units, {models['lake_id'].nunique()} lakes, "
        f"{len(figures)} figures; all selected R2 > {MIN_R2}."
    )
    print(f"Lake 2 postmerge: N=5, linear, R2={float(lake2.iloc[0]['selected_r2']):.6f}.")


def write_outputs() -> None:
    if not OBS_CSV.exists():
        raise FileNotFoundError(f"Missing bundled observation table: {OBS_CSV}")
    observations = pd.read_csv(OBS_CSV, encoding="utf-8-sig")
    models, curves = compute_models(observations)
    models.to_csv(MODEL_CSV, index=False, encoding="utf-8-sig")
    curves.to_csv(CURVE_CSV, index=False, encoding="utf-8-sig")
    validate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute and validate the unified AICc area–WSE fits."
    )
    parser.add_argument("--run", action="store_true", help="recompute model and fitted-curve tables")
    args = parser.parse_args()
    if args.run:
        write_outputs()
    else:
        validate()


if __name__ == "__main__":
    main()
