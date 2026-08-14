from pathlib import Path
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = (
    Path(str(os.environ.get("OUTPUT_ROOT_DIR", "")).strip())
    if str(os.environ.get("OUTPUT_ROOT_DIR", "")).strip()
    else SCRIPT_DIR.parent / "result2"
)


def resolve_classify_dir(result_root: Path) -> Path:
    base = result_root / "06_4_classify_lake_wse_behaviors_plot"
    preferred = base / "双轴图曲线_老ID"
    if preferred.exists():
        return preferred
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_dir() and "老ID" in p.name:
                return p
    return preferred


SOURCE_CLASSIFY_DIR = resolve_classify_dir(RESULT_ROOT)
CLASSIFY_CSV = SOURCE_CLASSIFY_DIR / "classification_manual_vD.csv"
BASE_SHP = RESULT_ROOT / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons.shp"
BRANCH_DIR = (
    RESULT_ROOT
    / "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"
    / "Otsu"
    / "original"
    / "06_3_BranchTables"
    / "vD"
)

OUT_DIR = RESULT_ROOT / "06_5_export_all_features_from_classify_plot" / SOURCE_CLASSIFY_DIR.name
OUT_SHP = OUT_DIR / "lakes_all_with_classified_wse_maxrange_newid_vd_Otsu.shp"
OUT_SHP_HAS1 = OUT_DIR / "lakes_has_curve_1_with_wse_maxrange_newid_vd_Otsu.shp"
OUT_SHP_HAS0 = OUT_DIR / "lakes_large_medium_has_curve_0_with_wse_maxrange_newid_vd_Otsu.shp"

PLOT_NAME_PATTERN = re.compile(r"^Lake_(\d+)_branch_(\d+)_vD\.png$", re.IGNORECASE)


def remove_existing_shapefile(path: Path):
    stem = path.with_suffix("")
    for p in stem.parent.glob(f"{stem.name}.*"):
        if p.suffix.lower() == ".lock":
            continue
        try:
            p.unlink()
        except Exception:
            pass


def detect_col(columns, candidates):
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def build_classify_csv_from_dirs():
    rows = []
    source_plot_root = (
        RESULT_ROOT
        / "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"
        / "Otsu"
        / "original"
        / "vd"
        / "dual_axis_area_wse"
    )
    for category_dir in sorted(SOURCE_CLASSIFY_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if category not in {
            "drainage_after_recharge",
            "slow_drainage",
            "stable_after_recharge",
            "sudden_drainage",
        }:
            continue
        for png in sorted(category_dir.glob("*.png")):
            match = PLOT_NAME_PATTERN.match(png.name)
            if match is None:
                continue
            lake_id_old = int(match.group(1))
            branch_index = int(match.group(2))
            rows.append(
                {
                    "lake_id_old": lake_id_old,
                    "branch_index": branch_index,
                    "category": category,
                    "file_name": png.name,
                    "source_plot": str(source_plot_root / png.name),
                }
            )

    if not rows:
        raise RuntimeError(f"No classified png found under: {SOURCE_CLASSIFY_DIR}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["lake_id_old", "branch_index"], keep="first")
    df = df.sort_values(["category", "lake_id_old", "branch_index"]).reset_index(drop=True)
    df.to_csv(CLASSIFY_CSV, index=False, encoding="utf-8-sig")
    return df


def main():
    if not SOURCE_CLASSIFY_DIR.exists():
        raise FileNotFoundError(f"Missing classify dir: {SOURCE_CLASSIFY_DIR}")
    if not BASE_SHP.exists():
        raise FileNotFoundError(f"Missing base shp: {BASE_SHP}")
    if not BRANCH_DIR.exists():
        raise FileNotFoundError(f"Missing branch dir: {BRANCH_DIR}")

    if CLASSIFY_CSV.exists():
        class_df = pd.read_csv(CLASSIFY_CSV, encoding="utf-8-sig")
    else:
        class_df = build_classify_csv_from_dirs()

    if class_df.empty:
        raise RuntimeError("classification_manual_vD.csv is empty")

    for col in ["lake_id_old", "branch_index", "category"]:
        if col not in class_df.columns:
            raise RuntimeError(f"Missing column in classify csv: {col}")

    class_df["lake_id_old"] = pd.to_numeric(class_df["lake_id_old"], errors="coerce")
    class_df["branch_index"] = pd.to_numeric(class_df["branch_index"], errors="coerce")
    class_df = class_df.dropna(subset=["lake_id_old", "branch_index"]).copy()
    class_df["lake_id_old"] = class_df["lake_id_old"].astype(int)
    class_df["branch_index"] = class_df["branch_index"].astype(int)

    pair_df = class_df[["lake_id_old", "branch_index", "category"]].drop_duplicates().reset_index(drop=True)

    range_rows = []
    missing_csv = 0
    for row in pair_df.itertuples(index=False):
        lake_id = int(row.lake_id_old)
        branch = int(row.branch_index)
        csv_path = BRANCH_DIR / f"lake_{lake_id}_branch_{branch}_stage_vD.csv"
        if not csv_path.exists():
            missing_csv += 1
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            missing_csv += 1
            continue
        if df.empty or "wse_m" not in df.columns:
            continue

        if "is_noise" in df.columns:
            is_noise = df["is_noise"].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])
            df_use = df.loc[~is_noise].copy()
            if df_use.empty:
                continue
        else:
            df_use = df

        wse = pd.to_numeric(df_use["wse_m"], errors="coerce").dropna()
        if wse.empty:
            continue
        max_wse = float(wse.max())
        min_wse = float(wse.min())
        range_rows.append(
            {
                "lake_id_old": lake_id,
                "branch_index": branch,
                "category": str(row.category),
                "max_wse_m": max_wse,
                "min_wse_m": min_wse,
                "max_rng_m": max_wse - min_wse,
                "obs_n": int(len(wse)),
            }
        )

    range_df = pd.DataFrame(range_rows)
    if range_df.empty:
        raise RuntimeError("No valid branch WSE stats computed.")

    lake_agg = (
        range_df.sort_values(["lake_id_old", "max_rng_m"], ascending=[True, False]).groupby("lake_id_old", as_index=False).first()
    )
    ranked = lake_agg.sort_values(["max_rng_m", "lake_id_old"], ascending=[False, True]).reset_index(drop=True)
    ranked["lake_id_new"] = np.arange(1, len(ranked) + 1, dtype=int)

    base = gpd.read_file(BASE_SHP)
    lake_col = detect_col(base.columns, ["lake_id", "old_lake_i"])
    if lake_col is None:
        raise RuntimeError(f"No lake id column in base shapefile: {list(base.columns)}")

    base["lake_id_old"] = pd.to_numeric(base[lake_col], errors="coerce")
    base = base.dropna(subset=["lake_id_old"]).copy()
    base["lake_id_old"] = base["lake_id_old"].astype(int)

    merge_cols = [
        "lake_id_old",
        "lake_id_new",
        "max_rng_m",
        "max_wse_m",
        "min_wse_m",
        "branch_index",
        "category",
        "obs_n",
    ]
    out = base.merge(ranked[merge_cols], on="lake_id_old", how="left")
    out["has_curve"] = out["lake_id_new"].notna().astype(int)

    for c in ["lake_id_new", "branch_index", "obs_n", "has_curve"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["max_rng_m", "max_wse_m", "min_wse_m"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["category"] = out["category"].fillna("").astype(str)
    if "lake_type" in out.columns:
        out["lake_type"] = out["lake_type"].fillna("").astype(str)
    else:
        out["lake_type"] = ""
    out["ice_flag"] = pd.to_numeric(out.get("ice_flag"), errors="coerce")

    behavior_code_map = {
        "drainage_after_recharge": 0,
        "slow_drainage": 1,
        "stable_after_recharge": 2,
        "sudden_drainage": 3,
    }
    out["bhv4_code"] = out["category"].str.strip().str.lower().map(behavior_code_map)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_shapefile(OUT_SHP)
    out.to_file(OUT_SHP, driver="ESRI Shapefile", encoding="utf-8")

    out_csv = OUT_SHP.with_suffix(".csv")
    out[
        [
            "lake_id_old",
            "lake_type",
            "lake_id_new",
            "has_curve",
            "max_rng_m",
            "max_wse_m",
            "min_wse_m",
            "branch_index",
            "category",
            "bhv4_code",
            "obs_n",
        ]
    ].to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_has1 = out[(out["has_curve"] == 1) & (out["ice_flag"] == 1)].copy()
    out_has0 = out[
        (out["has_curve"] == 0)
        & (out["ice_flag"] == 1)
        & (out["lake_type"].str.strip().str.lower().isin(["large", "medium"]))
    ].copy()

    remove_existing_shapefile(OUT_SHP_HAS1)
    remove_existing_shapefile(OUT_SHP_HAS0)
    if not out_has1.empty:
        out_has1.to_file(OUT_SHP_HAS1, driver="ESRI Shapefile", encoding="utf-8")
    if not out_has0.empty:
        out_has0.to_file(OUT_SHP_HAS0, driver="ESRI Shapefile", encoding="utf-8")

    out_has1.to_csv(OUT_SHP_HAS1.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    out_has0.to_csv(OUT_SHP_HAS0.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"[ok] result root: {RESULT_ROOT}")
    print(f"[ok] source classify dir: {SOURCE_CLASSIFY_DIR}")
    print(f"[ok] source classify csv: {CLASSIFY_CSV}")
    print(f"[ok] output shp: {OUT_SHP}")
    print(f"[ok] output csv: {out_csv}")
    print(f"[ok] output shp (has_curve=1, ice_flag=1): {OUT_SHP_HAS1}")
    print(f"[ok] output shp (large/medium has_curve=0, ice_flag=1): {OUT_SHP_HAS0}")
    print(f"[ok] total all features: {len(out)}")
    print(f"[ok] features with curves (ice_flag=1): {len(out_has1)}")
    print(f"[ok] large/medium features without curves (ice_flag=1): {len(out_has0)}")
    print(f"[ok] classified branches parsed: {len(range_df)} (missing csv: {missing_csv})")


if __name__ == "__main__":
    main()
