"""
Stage 04_3: export effective-water boundaries and JPG previews.

Effective water is defined directly from Stage 04_2 outputs:
1. rows classified as Water / WaterEdge
2. rows falling inside the water-period window already written by Stage 04_2

It then:
1. links retained dates back to Stage 04_1 mask outputs
2. reuses Stage 04_1 GDB mask polygons directly
3. exports one merged GDB layer
4. renders JPG previews using RGB imagery with retained boundaries overlaid
"""

import os
import re
import shutil

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from rasterio.errors import WindowError
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_ROOT_DIR = r"Z:\冰面湖水量测算_Local\冰面湖水位论文代码数据整理\result"
BUFFER_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons_Buffered.shp"
)
LAKE_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons.shp"
)
INPUT_04_1_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_1_segment_masks")
INPUT_04_2_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_2_filter_and_classify")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_3_export_effective_boundaries")

TARGET_METHODS = ["Otsu"]
TARGET_SIZE_CATEGORIES = []
TARGET_LAKE_IDS = []
MAX_LAKE_ID = 99999
TEST_FIRST_N_LAKES = 0
ENABLE_RESUME = True

TARGET_CRS = "EPSG:32627"
TARGET_RES = 10
PADDING_METERS = 200
GDB_LAYER_NAME = "effective_boundaries"

EFFECTIVE_FINAL_CLASSES = {"Water", "WaterEdge"}
DOMINANT_AREA_RATIO = 5.0
INVALID_BOUNDARY_LATE_SEASON_TOTAL_FEATURE_THRESHOLD = 2
LATE_SEASON_MONTH_DAY = "0815"
MIN_FEATURE_AREA_KM2 = 0.04
ALWAYS_VALID_FEATURE_AREA_KM2 = 0.08
MAX_VALID_FEATURE_COUNT = 2
BOUNDARY_VALID = "有效边界"
BOUNDARY_INVALID = "无效边界"


def progress(message: str) -> None:
    print(message, flush=True)


def is_target_lake_id(lake_id) -> bool:
    try:
        lid = int(lake_id)
    except Exception:
        return False
    if TARGET_LAKE_IDS and lid not in TARGET_LAKE_IDS:
        return False
    if MAX_LAKE_ID and MAX_LAKE_ID > 0 and lid > int(MAX_LAKE_ID):
        return False
    return True


def load_allowed_lake_ids():
    if not os.path.exists(LAKE_SHP):
        return set()
    gdf = gpd.read_file(LAKE_SHP)
    if gdf.empty or "lake_id" not in gdf.columns:
        return set()
    return set(pd.to_numeric(gdf["lake_id"], errors="coerce").dropna().astype(int).tolist())


def normalize_display(image_array: np.ndarray) -> np.ndarray:
    valid_mask = image_array > 0
    if not np.any(valid_mask):
        return image_array
    p2, p98 = np.percentile(image_array[valid_mask], (2, 98))
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        return image_array
    return np.clip((image_array - p2) / (p98 - p2), 0, 1)


def sanitize_layer_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not text:
        text = "effective_boundaries"
    if text[0].isdigit():
        text = f"layer_{text}"
    return text[:120]


def classify_polygon_validity(polygons_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = polygons_gdf.copy()
    work["geom_area_km2"] = work.geometry.area / 1e6
    work = work[work["geom_area_km2"] >= MIN_FEATURE_AREA_KM2].copy()
    work["is_invalid_small_feature"] = False
    work["feature_status"] = "valid"
    work["relative_to_max_area"] = np.nan
    work["max_feature_area_km2"] = np.nan

    if work.empty:
        return work

    work = work.sort_values("geom_area_km2", ascending=False).reset_index(drop=True)
    max_idx = work["geom_area_km2"].idxmax()
    max_area = float(work.loc[max_idx, "geom_area_km2"])
    if max_area <= 0:
        return work

    work["max_feature_area_km2"] = max_area
    work["relative_to_max_area"] = work["geom_area_km2"] / max_area

    small_mask = (
        (work.index != max_idx)
        & (work["geom_area_km2"] > 0)
        & (work["geom_area_km2"] <= ALWAYS_VALID_FEATURE_AREA_KM2)
        & (max_area >= DOMINANT_AREA_RATIO * work["geom_area_km2"])
    )
    work.loc[small_mask, "is_invalid_small_feature"] = True
    work.loc[small_mask, "feature_status"] = "invalid_small"

    overflow_mask = work.index >= MAX_VALID_FEATURE_COUNT
    work.loc[overflow_mask, "is_invalid_small_feature"] = True
    work.loc[overflow_mask, "feature_status"] = "invalid_overflow"
    return work


def make_empty_boundary_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)


def classify_boundary_scene_invalid(date_str: str, total_features: int) -> bool:
    if total_features <= 0:
        return True

    month_day = str(date_str)[4:8] if pd.notna(date_str) else ""
    if month_day > LATE_SEASON_MONTH_DAY and total_features > INVALID_BOUNDARY_LATE_SEASON_TOTAL_FEATURE_THRESHOLD:
        return True
    return False


def load_lake_geometries():
    buffer_gdf = gpd.read_file(BUFFER_SHP)
    lake_gdf = gpd.read_file(LAKE_SHP)
    if str(buffer_gdf.crs) != TARGET_CRS:
        buffer_gdf = buffer_gdf.to_crs(TARGET_CRS)
    if str(lake_gdf.crs) != TARGET_CRS:
        lake_gdf = lake_gdf.to_crs(TARGET_CRS)

    buffer_map = {}
    lake_map = {}
    size_map = {}

    for _, row in buffer_gdf.iterrows():
        if "lake_id" not in row:
            continue
        lake_id = int(row["lake_id"])
        if not is_target_lake_id(lake_id):
            continue
        buffer_map[lake_id] = row.geometry
        if "lake_type" in row and pd.notna(row["lake_type"]):
            size_map[lake_id] = str(row["lake_type"])

    for _, row in lake_gdf.iterrows():
        if "lake_id" not in row:
            continue
        lake_id = int(row["lake_id"])
        if not is_target_lake_id(lake_id):
            continue
        lake_map[lake_id] = row.geometry
        if lake_id not in size_map and "lake_type" in row and pd.notna(row["lake_type"]):
            size_map[lake_id] = str(row["lake_type"])

    return buffer_map, lake_map, size_map


def load_04_1_index(method_name: str) -> pd.DataFrame:
    method_dir = os.path.join(INPUT_04_1_DIR, method_name)
    csv_candidates = [
        os.path.join(method_dir, "04_1_mask_index.csv"),
        os.path.join(method_dir, "04_1_metrics_raw.csv"),
    ]
    csv_path = ""
    for candidate in csv_candidates:
        if os.path.exists(candidate):
            csv_path = candidate
            break
    if not csv_path:
        return pd.DataFrame()

    progress(f"[04_3] using 04_1 index source: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df

    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce")
    work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False)
    work["preview_name"] = work["preview_name"].fillna("").astype(str)
    work["gdb_path"] = work["gdb_path"].fillna("").astype(str)
    work["gdb_layer"] = work["gdb_layer"].fillna("").astype(str)
    work["rgb_path"] = work["rgb_path"].fillna("").astype(str)
    work = work.dropna(subset=["lake_id", "date"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work = work[work["lake_id"].map(is_target_lake_id)].copy()
    return work


def load_classified_csv(method_name: str) -> pd.DataFrame:
    csv_path = os.path.join(INPUT_04_2_DIR, method_name, "04_2_metrics_classified.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df
    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce")
    work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False)
    work = work.dropna(subset=["lake_id", "date"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work = work[work["lake_id"].map(is_target_lake_id)].copy()
    return work

def build_effective_rows_for_method(method_name: str) -> pd.DataFrame:
    classified = load_classified_csv(method_name)
    if classified.empty:
        return pd.DataFrame()
    work = classified.copy()
    work["water_period_start"] = pd.to_datetime(work["water_period_start"], errors="coerce")
    work["water_period_end"] = pd.to_datetime(work["water_period_end"], errors="coerce")
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    work = work[
        work["final_class"].isin(EFFECTIVE_FINAL_CLASSES)
        & work["date_dt"].notna()
        & work["water_period_start"].notna()
        & work["water_period_end"].notna()
        & (work["date_dt"] >= work["water_period_start"])
        & (work["date_dt"] <= work["water_period_end"])
    ].copy()

    if TARGET_LAKE_IDS:
        work = work[work["lake_id"].isin(TARGET_LAKE_IDS)].copy()
    allowed_lake_ids = load_allowed_lake_ids()
    if allowed_lake_ids:
        work = work[work["lake_id"].astype("Int64").isin(sorted(allowed_lake_ids))].copy()
    if MAX_LAKE_ID and MAX_LAKE_ID > 0:
        work = work[work["lake_id"] <= int(MAX_LAKE_ID)].copy()

    keep_cols = [
        "lake_id",
        "size_category",
        "date",
        "preview_name",
        "file_basename",
        "final_class",
        "export_class",
        "water_area_km2",
        "water_period_start",
        "water_period_end",
    ]
    work = work[keep_cols].sort_values(["size_category", "lake_id", "date", "preview_name"]).reset_index(drop=True)

    if TEST_FIRST_N_LAKES and TEST_FIRST_N_LAKES > 0:
        selected_lake_ids = sorted(work["lake_id"].dropna().astype(int).unique())[:TEST_FIRST_N_LAKES]
        work = work[work["lake_id"].isin(selected_lake_ids)].copy()
    return work


def _normalize_resume_index_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "lake_id" in work.columns:
        work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce")
    if "date" in work.columns:
        work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False)
    if "preview_name" in work.columns:
        work["preview_name"] = work["preview_name"].fillna("").astype(str)
    work = work.dropna(subset=["lake_id", "date"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work = work[work["lake_id"].map(is_target_lake_id)].copy()
    return work


def _make_scene_key_columns(df: pd.DataFrame) -> pd.Series:
    return (
        df["lake_id"].astype(int).astype(str)
        + "|"
        + df["date"].astype(str)
        + "|"
        + df["preview_name"].fillna("").astype(str)
    )


def load_polygons_from_gdb(gdb_path: str, gdb_layer: str):
    if not gdb_path or not gdb_layer:
        return make_empty_boundary_gdf()
    if gdb_layer.lower() == "nan" or not os.path.exists(gdb_path):
        return make_empty_boundary_gdf()
    try:
        gdf = gpd.read_file(gdb_path, layer=gdb_layer)
    except Exception:
        return make_empty_boundary_gdf()
    if gdf.empty:
        return make_empty_boundary_gdf()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    return gdf


def build_render_canvas(rgb_path: str, buffer_geom):
    if not rgb_path or not os.path.exists(rgb_path):
        return None

    minx, miny, maxx, maxy = buffer_geom.bounds
    minx -= PADDING_METERS
    miny -= PADDING_METERS
    maxx += PADDING_METERS
    maxy += PADDING_METERS

    dst_width = max(1, int((maxx - minx) / TARGET_RES))
    dst_height = max(1, int((maxy - miny) / TARGET_RES))
    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)

    with rasterio.open(rgb_path) as src:
        src_crs = src.crs
        if str(src_crs) != TARGET_CRS:
            dst_box = box(minx, miny, maxx, maxy)
            dst_gdf = gpd.GeoDataFrame({"geometry": [dst_box]}, crs=TARGET_CRS)
            src_bounds = dst_gdf.to_crs(src_crs).total_bounds
            src_window = windows.from_bounds(*src_bounds, transform=src.transform)
        else:
            src_window = windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        src_window = src_window.round_offsets(op="ceil", pixel_precision=0)
        try:
            src_window = src_window.intersection(windows.Window(0, 0, src.width, src.height))
        except WindowError:
            return None
        if src_window.width <= 0 or src_window.height <= 0:
            return None

        bands = [3, 2, 1] if src.count >= 3 else [1]
        reprojected = []
        for band_idx in bands:
            source_data = src.read(band_idx, window=src_window)
            src_transform = src.window_transform(src_window)
            dest = np.zeros((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=source_data,
                destination=dest,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
            )
            reprojected.append(dest)

    image = np.dstack(reprojected) if len(reprojected) == 3 else reprojected[0]
    image = normalize_display(image)
    return {"image": image, "bounds": (minx, miny, maxx, maxy)}


def export_preview_jpg(
    output_path: str,
    render_info,
    polygons_gdf: gpd.GeoDataFrame,
    lake_id: int,
    date_str: str,
    total_features: int,
    effective_features: int,
    invalid_features: int,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    minx, miny, maxx, maxy = render_info["bounds"]
    ax.imshow(render_info["image"], extent=[minx, maxx, miny, maxy], origin="upper")

    valid_gdf = polygons_gdf[~polygons_gdf["is_invalid_small_feature"]].copy()
    invalid_gdf = polygons_gdf[polygons_gdf["is_invalid_small_feature"]].copy()
    if not valid_gdf.empty:
        valid_gdf.boundary.plot(ax=ax, color="yellow", linewidth=1.6, alpha=0.95, zorder=10)
    if not invalid_gdf.empty:
        invalid_gdf.boundary.plot(ax=ax, color="deepskyblue", linewidth=1.6, alpha=0.95, zorder=11)

    max_area = float(polygons_gdf["max_feature_area_km2"].dropna().iloc[0]) if polygons_gdf["max_feature_area_km2"].notna().any() else np.nan
    for feature_idx, feature_row in polygons_gdf.iterrows():
        geom = feature_row.geometry
        if geom is None or geom.is_empty:
            continue
        label_point = geom.representative_point()
        rel_ratio = float(feature_row["relative_to_max_area"]) if pd.notna(feature_row["relative_to_max_area"]) else np.nan
        area_km2 = float(feature_row["geom_area_km2"]) if pd.notna(feature_row["geom_area_km2"]) else np.nan
        is_invalid = bool(feature_row["is_invalid_small_feature"])
        color = "deepskyblue" if is_invalid else "yellow"
        feature_label = (
            f"#{feature_idx + 1}\n"
            f"A={area_km2:.4f}\n"
            f"R={rel_ratio:.3f}"
        )
        ax.text(
            label_point.x,
            label_point.y,
            feature_label,
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            zorder=20,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    ax.set_title(f"Lake {lake_id} {date_str} | effective boundary", fontsize=13)
    ax.text(
        0.02,
        0.02,
        (
            f"max_area = {max_area:.4f} km2\n"
            f"ratio_rule = {DOMINANT_AREA_RATIO:.1f}x\n"
            f"total = {total_features}\n"
            f"valid = {effective_features}\n"
            f"invalid = {invalid_features}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="yellow",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    buffer_map, _lake_map, size_map = load_lake_geometries()

    for method_name in TARGET_METHODS:
        progress(f"[04_3] processing method={method_name}")
        effective_df = build_effective_rows_for_method(method_name)
        mask_index_df = load_04_1_index(method_name)
        if effective_df.empty:
            progress(f"[04_3] skip {method_name}: no effective rows found")
            continue
        if mask_index_df.empty:
            progress(f"[04_3] skip {method_name}: no 04_1 mask index found")
            continue

        merged_df = effective_df.merge(
            mask_index_df,
            on=["lake_id", "date", "preview_name"],
            how="left",
            suffixes=("", "_04_1"),
        )
        merged_df["size_category"] = merged_df["size_category"].fillna(merged_df.get("size_category_04_1", ""))
        merged_df = merged_df[merged_df["size_category"].isin(TARGET_SIZE_CATEGORIES)].copy()

        method_dir = os.path.join(OUTPUT_DIR, method_name)
        jpg_dir = os.path.join(method_dir, "04_3_jpg")
        gdb_path = os.path.join(method_dir, "04_3_effective_boundaries.gdb")
        summary_csv = os.path.join(method_dir, "04_3_effective_boundary_index.csv")

        os.makedirs(method_dir, exist_ok=True)
        if os.path.exists(gdb_path):
            progress(f"[04_3] rebuild {method_name}: existing GDB will be refreshed -> {gdb_path}")

        existing_index_df = pd.DataFrame()
        existing_gdb_df = gpd.GeoDataFrame()
        if ENABLE_RESUME and os.path.exists(summary_csv):
            try:
                existing_index_df = pd.read_csv(summary_csv, encoding="utf-8-sig")
                existing_index_df = _normalize_resume_index_df(existing_index_df)
            except Exception as exc:
                progress(f"[04_3] resume warning: failed to read existing index -> {exc}")
                existing_index_df = pd.DataFrame()

        if ENABLE_RESUME and os.path.exists(gdb_path):
            try:
                existing_gdb_df = gpd.read_file(gdb_path, layer=GDB_LAYER_NAME)
                if str(existing_gdb_df.crs) != TARGET_CRS:
                    existing_gdb_df = existing_gdb_df.to_crs(TARGET_CRS)
                if "lake_id" in existing_gdb_df.columns:
                    existing_gdb_df = existing_gdb_df[
                        pd.to_numeric(existing_gdb_df["lake_id"], errors="coerce").map(is_target_lake_id)
                    ].copy()
            except Exception as exc:
                progress(f"[04_3] resume warning: failed to read existing gdb -> {exc}")
                existing_gdb_df = gpd.GeoDataFrame()

        total_before_resume = len(merged_df)
        resume_skip_count = 0
        if ENABLE_RESUME and not existing_index_df.empty:
            existing_keys = set(_make_scene_key_columns(existing_index_df).tolist())
            merged_df = merged_df.copy()
            merged_df["_scene_key"] = _make_scene_key_columns(merged_df)
            resume_mask = merged_df["_scene_key"].isin(existing_keys)
            resume_skip_count = int(resume_mask.sum())
            merged_df = merged_df[~resume_mask].drop(columns=["_scene_key"]).copy()
            progress(
                f"[04_3] resume {method_name}: total={total_before_resume}, "
                f"skip_done={resume_skip_count}, pending={len(merged_df)}"
            )

        gdb_rows = []
        index_rows = []
        total_scene_count = len(merged_df)

        for idx, row in enumerate(merged_df.itertuples(index=False), 1):
            lake_id = int(row.lake_id)
            size_category = str(row.size_category)
            buffer_geom = buffer_map.get(lake_id)
            if buffer_geom is None:
                progress(f"[04_3] lake={lake_id} date={row.date}: missing buffer geometry")
                continue

            gdb_layer = "" if pd.isna(getattr(row, "gdb_layer", "")) else str(getattr(row, "gdb_layer", "")).strip()
            gdb_path_row = "" if pd.isna(getattr(row, "gdb_path", "")) else str(getattr(row, "gdb_path", "")).strip()
            rgb_path = "" if pd.isna(getattr(row, "rgb_path", "")) else str(getattr(row, "rgb_path", "")).strip()

            polygons_gdf = load_polygons_from_gdb(gdb_path_row, gdb_layer)
            if str(polygons_gdf.crs) != TARGET_CRS:
                polygons_gdf = polygons_gdf.to_crs(TARGET_CRS)
            polygons_gdf = classify_polygon_validity(polygons_gdf.reset_index(drop=True))
            total_features = len(polygons_gdf)
            invalid_features = int(polygons_gdf["is_invalid_small_feature"].sum())
            effective_features = total_features - invalid_features
            if total_features == 0:
                progress(f"[04_3] lake={lake_id} date={row.date}: no retained polygons after filtering")

            for feature_idx, geom_row in polygons_gdf.iterrows():
                gdb_rows.append(
                    {
                        "lake_id": lake_id,
                        "size_category": size_category,
                        "date": str(row.date),
                        "preview_name": str(row.preview_name),
                        "file_basename": str(getattr(row, "file_basename", "")),
                        "final_class": str(getattr(row, "final_class", "")),
                        "export_class": str(getattr(row, "export_class", "")),
                        "feature_idx": int(feature_idx + 1),
                        "total_features": int(total_features),
                        "effective_features": int(effective_features),
                        "invalid_features": int(invalid_features),
                        "is_invalid_small_feature": bool(geom_row["is_invalid_small_feature"]),
                        "feature_status": str(geom_row["feature_status"]),
                        "water_area_km2": float(getattr(row, "water_area_km2", np.nan)) if pd.notna(getattr(row, "water_area_km2", np.nan)) else np.nan,
                        "geom_area_km2": float(geom_row["geom_area_km2"]),
                        "max_feature_area_km2": float(geom_row["max_feature_area_km2"]) if pd.notna(geom_row["max_feature_area_km2"]) else np.nan,
                        "relative_to_max_area": float(geom_row["relative_to_max_area"]) if pd.notna(geom_row["relative_to_max_area"]) else np.nan,
                        "geometry": geom_row.geometry,
                    }
                )

            render_info = build_render_canvas(rgb_path, buffer_geom)
            jpg_path = ""
            scene_is_invalid = classify_boundary_scene_invalid(str(row.date), total_features)
            jpg_bucket = BOUNDARY_INVALID if scene_is_invalid else BOUNDARY_VALID
            if render_info is not None:
                lake_jpg_dir = os.path.join(jpg_dir, size_category, str(lake_id), jpg_bucket)
                jpg_path = os.path.join(
                    lake_jpg_dir,
                    f"effective_boundary_{lake_id}_{row.date}_{sanitize_layer_name(row.preview_name)}.jpg",
                )
                export_preview_jpg(
                    jpg_path,
                    render_info,
                    polygons_gdf,
                    lake_id,
                    str(row.date),
                    total_features,
                    effective_features,
                    invalid_features,
                )

            index_rows.append(
                {
                    "lake_id": lake_id,
                    "size_category": size_category,
                    "lake_type": size_map.get(lake_id, size_category),
                    "date": str(row.date),
                    "preview_name": str(row.preview_name),
                    "file_basename": str(getattr(row, "file_basename", "")),
                    "final_class": str(getattr(row, "final_class", "")),
                    "export_class": str(getattr(row, "export_class", "")),
                    "water_period_start": str(getattr(row, "water_period_start", "")),
                    "water_period_end": str(getattr(row, "water_period_end", "")),
                    "total_features": int(total_features),
                    "effective_features": int(effective_features),
                    "invalid_features": int(invalid_features),
                    "jpg_bucket": jpg_bucket,
                    "gdb_path": gdb_path_row,
                    "gdb_layer": gdb_layer,
                    "rgb_path": rgb_path,
                    "jpg_path": jpg_path,
                }
            )
            progress(
                f"[04_3] {method_name} {idx}/{total_scene_count} done | "
                f"lake={lake_id} | date={row.date} | total={total_features} | "
                f"valid={effective_features} | invalid={invalid_features}"
            )

        index_frames = []
        if not existing_index_df.empty:
            index_frames.append(existing_index_df.copy())
        if index_rows:
            index_frames.append(pd.DataFrame(index_rows))
        if index_frames:
            index_df = pd.concat(index_frames, ignore_index=True)
            index_df = _normalize_resume_index_df(index_df)
            index_df = index_df.drop_duplicates(subset=["lake_id", "date", "preview_name"], keep="last")
            index_df = index_df.sort_values(["size_category", "lake_id", "date"]).reset_index(drop=True)
        else:
            index_df = pd.DataFrame()
        index_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        gdb_frames = []
        if isinstance(existing_gdb_df, gpd.GeoDataFrame) and not existing_gdb_df.empty:
            gdb_frames.append(existing_gdb_df.copy())
        if gdb_rows:
            gdb_frames.append(gpd.GeoDataFrame(gdb_rows, geometry="geometry", crs=TARGET_CRS))

        if gdb_frames:
            out_gdf = gpd.GeoDataFrame(pd.concat(gdb_frames, ignore_index=True), geometry="geometry", crs=TARGET_CRS)
            dedup_keys = [c for c in ["lake_id", "date", "preview_name", "feature_idx"] if c in out_gdf.columns]
            if dedup_keys:
                out_gdf = out_gdf.drop_duplicates(subset=dedup_keys, keep="last").copy()
            out_gdf.to_file(gdb_path, layer=GDB_LAYER_NAME, driver="OpenFileGDB")
            progress(f"[04_3] gdb written: {gdb_path} | features={len(out_gdf)}")
        else:
            progress(f"[04_3] no boundary features written for method={method_name}")

        progress(f"[04_3] index csv: {summary_csv}")


if __name__ == "__main__":
    main()
