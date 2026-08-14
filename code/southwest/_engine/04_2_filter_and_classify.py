"""
Stage 04_2: draw single views, filter, classify, and export classified views.
"""

import os
import shutil
import threading

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as ndi
from rasterio.features import rasterize
from rasterio.mask import mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

PLOT_LOCK = threading.Lock()

OUTPUT_ROOT_DIR = str(os.environ.get("OUTPUT_ROOT_DIR", r"X:\2024_西南水位曲线\result1")).strip()
BUFFERED_LAKE_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons_Buffered.shp"
)
LAKE_AREA_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons.shp"
)
INPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_1_segment_masks")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_2_filter_and_classify")

METHOD_NAMES = ["Otsu"]
TARGET_SIZE_CATEGORIES = ["Large", "Medium"]
TARGET_LAKE_IDS = []
MAX_LAKE_ID = 0

SINGLE_VIEW_DPI = 120
SHOW_COMPONENT_MEAN_NDWI = True
COMPONENT_MEAN_DECIMALS = 3

CLASS_WATER = "Water"
CLASS_WATER1 = "Water1"
CLASS_WATER_EDGE = "WaterEdge"
CLASS_NONWATER = "NonWater"

OTSU_FLOOR = 0.05
RESCUE_NDWI_THRESHOLD = 0.25
RESCUE_PRE_WINDOW_DAYS = 5
RESCUE_COVERAGE_THRESHOLD = 0.80
AOI_ONLY_OUT_RATIO_MAX = 0.0
EDGE_WATER_OUT_RATIO_MIN = 0.03
EDGE_WATER_OUT_RATIO_MAX = 0.10
EDGE_WATER_IN_RATIO_MAX = 0.80
PEAK_SUPPORT_WINDOW_DAYS = 3
PEAK_SUPPORT_RATIO = 0.70
WATER1_MAX_NUM_FEATURES = 3
WATER1_TOP_COMPONENT_AOI_RATIO_MIN = 0.2
WATER1_TOP_COMPONENT_MEAN_NDWI_MIN = 0.1
DATE_END = pd.Timestamp("2024-08-20")


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


def get_method_paths(method_name):
    method_input_dir = os.path.join(INPUT_DIR, method_name)
    method_output_dir = os.path.join(OUTPUT_DIR, method_name)
    return {
        "method_name": method_name,
        "raw_metrics_csv": os.path.join(method_input_dir, "04_1_metrics_raw.csv"),
        "output_dir": method_output_dir,
        "single_view_dir": os.path.join(method_output_dir, "04_2_SingleViews"),
        "single_view_index_csv": os.path.join(method_output_dir, "04_2_single_view_index.csv"),
        "classified_csv": os.path.join(method_output_dir, "04_2_metrics_classified.csv"),
        "water_period_csv": os.path.join(method_output_dir, "04_2_water_period_summary.csv"),
        "classified_view_dir": os.path.join(method_output_dir, "04_2_Classified_Views"),
        "classified_view_all_dir": os.path.join(method_output_dir, "04_2_Classified_Views_All"),
    }


def load_lake_geometries():
    buffer_gdf = gpd.read_file(BUFFERED_LAKE_SHP)
    area_gdf = gpd.read_file(LAKE_AREA_SHP)

    buffer_map = {}
    original_map = {}
    for _, row in buffer_gdf.iterrows():
        if "lake_id" in row:
            lid = int(row["lake_id"])
            if is_target_lake_id(lid):
                buffer_map[lid] = row.geometry
    for _, row in area_gdf.iterrows():
        if "lake_id" in row:
            lid = int(row["lake_id"])
            if is_target_lake_id(lid):
                original_map[lid] = row.geometry
    return buffer_map, original_map


def transform_geom_to_pixel_coords(geom, transform):
    if geom is None or geom.is_empty:
        return []

    geoms_list = [geom] if geom.geom_type == "Polygon" else (list(geom.geoms) if hasattr(geom, "geoms") else [])
    inv_tf = ~transform
    results = []
    for g in geoms_list:
        if g.is_empty:
            continue
        xs, ys = g.exterior.xy
        pix_coords = [inv_tf * (x, y) for x, y in zip(xs, ys)]
        results.append(([p[0] for p in pix_coords], [p[1] for p in pix_coords]))
    return results


def load_mask_from_sources(record, buffer_geom):
    ndwi_path = str(record.get("ndwi_path", "")).strip()
    if not ndwi_path or not os.path.exists(ndwi_path):
        raise RuntimeError(f"missing ndwi_path: {ndwi_path}")

    with rasterio.open(ndwi_path) as src:
        out_image, out_transform = mask(src, [buffer_geom], crop=True, nodata=np.nan)
        data = out_image[0]
        out_crs = src.crs

    water_mask = np.zeros(data.shape, dtype=bool)
    gdb_path_raw = record.get("gdb_path", "")
    gdb_layer_raw = record.get("gdb_layer", "")
    gdb_path = "" if pd.isna(gdb_path_raw) else str(gdb_path_raw).strip()
    gdb_layer = "" if pd.isna(gdb_layer_raw) else str(gdb_layer_raw).strip()
    if gdb_layer.lower() == "nan":
        gdb_layer = ""
    if gdb_path and gdb_layer and os.path.exists(gdb_path):
        water_gdf = gpd.read_file(gdb_path, layer=gdb_layer)
        if not water_gdf.empty:
            if water_gdf.crs != out_crs:
                water_gdf = water_gdf.to_crs(out_crs)
            water_mask = rasterize(
                [(geom, 1) for geom in water_gdf.geometry if geom is not None and not geom.is_empty],
                out_shape=data.shape,
                transform=out_transform,
                fill=0,
                all_touched=False,
                dtype="uint8",
            ).astype(bool)

    return data, water_mask, out_transform


def save_single_view(record, buffer_geom, original_geom, single_view_root):
    data, water_mask, out_transform = load_mask_from_sources(record, buffer_geom)

    lake_dir = os.path.join(single_view_root, record["size_category"], str(int(record["lake_id"])))
    os.makedirs(lake_dir, exist_ok=True)
    out_path = os.path.join(lake_dir, record["preview_name"])

    bg_array = data
    bg_cmap = "RdBu"
    bg_clim_val = (-0.6, 0.6)

    rgb_path = str(record.get("rgb_path", "")).strip()
    if rgb_path and os.path.exists(rgb_path):
        try:
            with rasterio.open(rgb_path) as src_rgb:
                rgb_crop, _ = mask(src_rgb, [buffer_geom], crop=True, nodata=0)
                if rgb_crop.shape[0] >= 3:
                    rgb_float = rgb_crop[0:3].astype(float)
                    rgb_float[rgb_float <= 0] = np.nan
                    valid_pixels_rgb = rgb_float[~np.isnan(rgb_float)]
                    if valid_pixels_rgb.size > 0:
                        p2, p98 = np.percentile(valid_pixels_rgb, [2, 98])
                        if p98 - p2 < 1:
                            p98 = p2 + 1
                        rgb_float = (rgb_float - p2) / (p98 - p2)
                        rgb_float = np.clip(rgb_float, 0, 1)
                        bg_array = np.moveaxis(rgb_float, 0, -1)[..., ::-1]
                        bg_cmap = None
                        bg_clim_val = None
        except Exception:
            pass

    with PLOT_LOCK:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if bg_cmap:
            im = ax.imshow(bg_array, cmap=bg_cmap, vmin=bg_clim_val[0], vmax=bg_clim_val[1], zorder=1)
        else:
            im = ax.imshow(bg_array, zorder=1)

        ax.set_title(
            f"Lake {int(record['lake_id'])} {record['date']} | {record['method']} T: {float(record['thresh']):.3f}\n"
            "Yellow: Water | Red: Buffer | Gray: AOI",
            fontsize=13,
        )

        for xs, ys in transform_geom_to_pixel_coords(buffer_geom, out_transform):
            ax.plot(xs, ys, color="red", linestyle="--", linewidth=1.0, zorder=10)
        for xs, ys in transform_geom_to_pixel_coords(original_geom, out_transform):
            ax.plot(xs, ys, color="gray", linestyle="--", linewidth=1.0, zorder=10)

        if np.any(water_mask):
            ax.contour(water_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.5, zorder=12)
            if SHOW_COMPONENT_MEAN_NDWI:
                structure = ndi.generate_binary_structure(2, 2)
                labeled_components, component_count = ndi.label(water_mask, structure=structure)
                for comp_id in range(1, component_count + 1):
                    comp_mask = labeled_components == comp_id
                    if not np.any(comp_mask):
                        continue
                    comp_mean = float(np.nanmean(data[comp_mask]))
                    if not np.isfinite(comp_mean):
                        continue
                    cy, cx = ndi.center_of_mass(comp_mask)
                    if not (np.isfinite(cy) and np.isfinite(cx)):
                        continue
                    ax.text(
                        cx,
                        cy,
                        f"{comp_mean:.{COMPONENT_MEAN_DECIMALS}f}",
                        color="yellow",
                        fontsize=8,
                        ha="center",
                        va="center",
                        zorder=13,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, edgecolor="none"),
                    )

        ax.set_xticks([])
        ax.set_yticks([])
        if bg_cmap:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_path, dpi=SINGLE_VIEW_DPI, bbox_inches="tight")
        plt.close(fig)

    return out_path


def build_scene_key(record):
    lake_id = int(record.get("lake_id", -1))
    preview_name = str(record.get("preview_name", "")).strip()
    if preview_name:
        return f"{lake_id}|{preview_name}"
    date_text = str(record.get("date", "")).strip()
    basename = str(record.get("file_basename", "")).strip()
    return f"{lake_id}|{date_text}|{basename}"


def deduplicate_records(records):
    dedup = {}
    for row in records:
        dedup[build_scene_key(row)] = row
    return list(dedup.values())


def sort_records_for_export(records):
    if not records:
        return records
    df_tmp = pd.DataFrame(records)
    sort_cols = [col for col in ["lake_id", "date", "file_basename", "preview_name"] if col in df_tmp.columns]
    if sort_cols:
        df_tmp = df_tmp.sort_values(sort_cols)
    return df_tmp.to_dict(orient="records")


def load_existing_single_view_records(csv_path):
    if not os.path.exists(csv_path):
        return []
    try:
        df_exist = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[04_2] warning: failed to read existing single_view_index, will rebuild: {exc}")
        return []
    if df_exist.empty:
        return []
    if "lake_id" in df_exist.columns:
        df_exist = df_exist[pd.to_numeric(df_exist["lake_id"], errors="coerce").map(is_target_lake_id)]
    return df_exist.to_dict(orient="records")


def load_existing_classified_records(csv_path):
    if not os.path.exists(csv_path):
        return []
    try:
        df_exist = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[04_2] warning: failed to read existing classified csv, will rebuild: {exc}")
        return []
    if df_exist.empty:
        return []
    if "lake_id" in df_exist.columns:
        df_exist = df_exist[pd.to_numeric(df_exist["lake_id"], errors="coerce").map(is_target_lake_id)]
    return df_exist.to_dict(orient="records")


def load_existing_period_map(csv_path):
    if not os.path.exists(csv_path):
        return {}
    try:
        df_exist = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[04_2] warning: failed to read existing period csv, will rebuild: {exc}")
        return {}
    if df_exist.empty or "lake_id" not in df_exist.columns:
        return {}
    period_map = {}
    for _, row in df_exist.iterrows():
        try:
            lake_id = int(row["lake_id"])
        except Exception:
            continue
        if not is_target_lake_id(lake_id):
            continue
        period_map[lake_id] = row.to_dict()
    return period_map


def save_single_view_checkpoint(single_view_records, csv_path):
    dedup_sorted = sort_records_for_export(deduplicate_records(single_view_records))
    pd.DataFrame(dedup_sorted).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return dedup_sorted


def copy_classified_views(classified, method_paths):
    os.makedirs(method_paths["classified_view_dir"], exist_ok=True)
    os.makedirs(method_paths["classified_view_all_dir"], exist_ok=True)

    copied = 0
    skipped_missing = 0
    lake_groups = list(classified.groupby("lake_id", sort=True))
    total_lakes = len(lake_groups)

    for lake_idx, (lake_id, lake_df) in enumerate(lake_groups, start=1):
        lake_copied = 0
        for _, row in lake_df.iterrows():
            lake_id = int(row["lake_id"])
            size_category = str(row.get("size_category", "Unknown"))
            preview_name = str(row.get("preview_name", ""))
            if not preview_name:
                skipped_missing += 1
                continue
            if size_category not in ["Large", "Medium", "Small"]:
                continue

            export_class = str(row.get("export_class", ""))
            if export_class not in [CLASS_WATER, CLASS_WATER1, CLASS_NONWATER]:
                export_class = CLASS_WATER if row["final_class"] in [CLASS_WATER, CLASS_WATER_EDGE] else CLASS_NONWATER

            src_path = str(row.get("raw_image_path", "")).strip()
            if not src_path or not os.path.isfile(src_path):
                src_path = os.path.join(method_paths["single_view_dir"], size_category, str(lake_id), preview_name)
            if not os.path.isfile(src_path):
                skipped_missing += 1
                continue

            dst_dir = os.path.join(method_paths["classified_view_dir"], size_category, str(lake_id), export_class)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, preview_name)

            dst_all_dir = os.path.join(method_paths["classified_view_all_dir"], size_category, export_class)
            os.makedirs(dst_all_dir, exist_ok=True)
            dst_all_path = os.path.join(dst_all_dir, f"lake{lake_id}_{preview_name}")

            shutil.copy2(src_path, dst_path)
            shutil.copy2(src_path, dst_all_path)
            copied += 1
            lake_copied += 1

        print(f"[04_2] copy views {lake_idx}/{total_lakes} lakes done | lake={int(lake_id)} | copied={lake_copied}")

    print(f"[04_2] classified views copied: {copied}, missing source: {skipped_missing}")


def classify_group_otsu(group):
    group = group.sort_values(["date", "file_basename"]).copy()
    group["date_dt"] = pd.to_datetime(group["date"], format="%Y%m%d")

    boundary_mask = group["touches_boundary"].fillna(False).astype(bool)
    if bool(boundary_mask.all()):
        group["initial_reason"] = "touches boundary"
        group["initial_class"] = CLASS_NONWATER
        group["is_rescued"] = False
        group["final_class"] = CLASS_NONWATER
        group["class_reason"] = "touches boundary"
        return group

    empty_foreground = (group["num_features"] <= 0) | (group["water_area_km2"] <= 0)
    aoi_only_water = (
        (~boundary_mask)
        & (~empty_foreground)
        & (group["thresh"] < OTSU_FLOOR)
        & (group["out_ratio"] <= AOI_ONLY_OUT_RATIO_MAX)
        & (group["water_area_km2"] > 0)
    )
    group["initial_reason"] = np.where(
        boundary_mask,
        "touches boundary",
        np.where(
            empty_foreground,
            "empty foreground",
            np.where(
                aoi_only_water,
                "threshold below floor but foreground stays inside AOI",
                np.where(group["thresh"] < OTSU_FLOOR, "threshold below floor", "initial water"),
            ),
        ),
    )
    group["initial_class"] = np.where(
        (~boundary_mask) & (~empty_foreground) & (aoi_only_water | (group["thresh"] >= OTSU_FLOOR)),
        CLASS_WATER,
        CLASS_NONWATER,
    )
    group["is_rescued"] = False

    initial_water = group[group["initial_class"] == CLASS_WATER]
    if initial_water.empty:
        group["final_class"] = group["initial_class"]
        group["class_reason"] = group["initial_reason"]
        return group

    final_class = []
    class_reason = []
    is_rescued = []
    for _, row in group.iterrows():
        if row["initial_class"] == CLASS_WATER:
            final_class.append(CLASS_WATER)
            class_reason.append("initial water")
            is_rescued.append(False)
            continue

        can_rescue = pd.notna(row["mu_water"]) and row["mu_water"] > RESCUE_NDWI_THRESHOLD
        if not can_rescue:
            final_class.append(CLASS_NONWATER)
            class_reason.append("kept as non-water")
            is_rescued.append(False)
            continue

        if row["coverage_ratio"] > RESCUE_COVERAGE_THRESHOLD:
            final_class.append(CLASS_WATER)
            class_reason.append("rescued by high coverage")
        else:
            final_class.append(CLASS_WATER_EDGE)
            class_reason.append("rescued by low coverage")
        is_rescued.append(True)

    group["final_class"] = final_class
    group["class_reason"] = class_reason
    group["is_rescued"] = is_rescued
    return group


def audit_group(group):
    group = group.sort_values("date_dt").copy()
    group["is_outlier"] = False

    while True:
        water_df = group[group["final_class"].isin([CLASS_WATER, CLASS_WATER_EDGE])].sort_values("date_dt")
        if len(water_df) <= 1:
            break

        max_idx = water_df["water_area_km2"].idxmax()
        max_row = group.loc[max_idx]
        last_water_date = water_df["date_dt"].max()
        changed = False
        protect_as_water = pd.notna(max_row["mu_water"]) and max_row["mu_water"] > RESCUE_NDWI_THRESHOLD

        if max_row["date_dt"] == last_water_date and not protect_as_water:
            group.loc[max_idx, ["final_class", "is_outlier", "class_reason"]] = [
                CLASS_NONWATER,
                True,
                "outlier: terminal peak",
            ]
            changed = True
        elif not protect_as_water:
            window_start = max_row["date_dt"] - pd.Timedelta(days=PEAK_SUPPORT_WINDOW_DAYS)
            window_end = max_row["date_dt"] + pd.Timedelta(days=PEAK_SUPPORT_WINDOW_DAYS)
            neighbors = water_df[
                (water_df.index != max_idx)
                & (water_df["date_dt"] >= window_start)
                & (water_df["date_dt"] <= window_end)
            ]
            max_neighbor = neighbors["water_area_km2"].max() if not neighbors.empty else 0.0
            ratio = max_neighbor / max_row["water_area_km2"] if max_row["water_area_km2"] > 0 else 0.0
            if neighbors.empty or ratio < PEAK_SUPPORT_RATIO:
                group.loc[max_idx, ["final_class", "is_outlier", "class_reason"]] = [
                    CLASS_NONWATER,
                    True,
                    "outlier: unsupported peak",
                ]
                changed = True

        if not changed:
            break

    group["is_valid"] = group["final_class"].isin([CLASS_WATER, CLASS_WATER_EDGE])
    return group


def print_run_summary(method_paths):
    print(f"[04_2] Filter And Classify - {method_paths['method_name']}")
    print(f"  input_csv: {method_paths['raw_metrics_csv']}")
    print(f"  output_csv: {method_paths['classified_csv']}")
    print(f"  single_view_dir: {method_paths['single_view_dir']}")
    print(f"  classified_view_dir: {method_paths['classified_view_dir']}")


def print_completion_summary(method_paths, input_records, lake_count, classified):
    final_counts = classified["final_class"].value_counts()
    print(f"[04_2] Completed - {method_paths['method_name']}")
    print(f"  input_records: {input_records}")
    print(f"  lakes_processed: {lake_count}")
    print(f"  output_records: {len(classified)}")
    print(f"  water_records: {int(final_counts.get(CLASS_WATER, 0))}")
    print(f"  water_edge_records: {int(final_counts.get(CLASS_WATER_EDGE, 0))}")
    print(f"  nonwater_records: {int(final_counts.get(CLASS_NONWATER, 0))}")
    print(f"  output_csv: {method_paths['classified_csv']}")
    print(f"  water_period_csv: {method_paths['water_period_csv']}")


def prepare_lake_for_classification(lake_df):
    classified = lake_df.copy()
    if classified.empty:
        return classified

    if "touches_boundary" not in classified.columns:
        classified["touches_boundary"] = False
    classified["touches_boundary"] = classified["touches_boundary"].fillna(False).astype(bool)
    classified["date"] = classified["date"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(classified["date"].astype(str))
    if "has_component_inside_aoi" not in classified.columns:
        classified["has_component_inside_aoi"] = False
    if "component_inside_aoi_count" not in classified.columns:
        classified["component_inside_aoi_count"] = 0
    for rank in [1, 2, 3]:
        inside_col = f"top_3_is_fully_inside_aoi_{rank}"
        mean_col = f"top_3_mean_ndwi_{rank}"
        if inside_col not in classified.columns:
            classified[inside_col] = False
        if mean_col not in classified.columns:
            classified[mean_col] = np.nan
    classified["has_component_inside_aoi"] = classified["has_component_inside_aoi"].fillna(False).astype(bool)
    classified["component_inside_aoi_count"] = pd.to_numeric(classified["component_inside_aoi_count"], errors="coerce").fillna(0).astype(int)
    for rank in [1, 2, 3]:
        inside_col = f"top_3_is_fully_inside_aoi_{rank}"
        mean_col = f"top_3_mean_ndwi_{rank}"
        classified[inside_col] = classified[inside_col].fillna(False).astype(bool)
        classified[mean_col] = pd.to_numeric(classified[mean_col], errors="coerce")
    return classified


def apply_export_class_rules(classified):
    if classified.empty:
        return classified
    water1_component_ok = (
        (
            classified["top_3_is_fully_inside_aoi_1"]
            & (pd.to_numeric(classified["top_3_aoi_ratio_1"], errors="coerce") > WATER1_TOP_COMPONENT_AOI_RATIO_MIN)
            & (classified["top_3_mean_ndwi_1"] > WATER1_TOP_COMPONENT_MEAN_NDWI_MIN)
        )
        | (
            classified["top_3_is_fully_inside_aoi_2"]
            & (pd.to_numeric(classified["top_3_aoi_ratio_2"], errors="coerce") > WATER1_TOP_COMPONENT_AOI_RATIO_MIN)
            & (classified["top_3_mean_ndwi_2"] > WATER1_TOP_COMPONENT_MEAN_NDWI_MIN)
        )
        | (
            classified["top_3_is_fully_inside_aoi_3"]
            & (pd.to_numeric(classified["top_3_aoi_ratio_3"], errors="coerce") > WATER1_TOP_COMPONENT_AOI_RATIO_MIN)
            & (classified["top_3_mean_ndwi_3"] > WATER1_TOP_COMPONENT_MEAN_NDWI_MIN)
        )
    )
    classified["export_class"] = np.where(
        (classified["final_class"] == CLASS_NONWATER)
        & (classified["num_features"] <= WATER1_MAX_NUM_FEATURES)
        & (classified["has_component_inside_aoi"]),
        CLASS_WATER1,
        np.where(classified["final_class"].isin([CLASS_WATER, CLASS_WATER_EDGE]), CLASS_WATER, CLASS_NONWATER),
    )
    classified["export_class"] = np.where(
        (classified["export_class"] == CLASS_WATER1) & water1_component_ok,
        CLASS_WATER1,
        np.where(classified["final_class"].isin([CLASS_WATER, CLASS_WATER_EDGE]), CLASS_WATER, CLASS_NONWATER),
    )
    return classified


def annotate_water_period_for_lake(classified):
    if classified.empty:
        return classified, {"lake_id": -1, "first_water": "", "last_water": "", "peak_date": "", "max_area_km2": 0.0}
    lake_id = int(classified["lake_id"].iloc[0])
    water_df = classified[classified["final_class"].isin([CLASS_WATER, CLASS_WATER_EDGE])]
    if water_df.empty:
        classified.loc[:, ["water_period_start", "water_period_end", "peak_date", "max_area_km2"]] = ["", "", "", 0.0]
        return classified, {"lake_id": lake_id, "first_water": "", "last_water": "", "peak_date": "", "max_area_km2": 0.0}

    peak_idx = water_df["water_area_km2"].idxmax()
    first_water = water_df["date_dt"].min().strftime("%Y-%m-%d")
    last_water = water_df["date_dt"].max().strftime("%Y-%m-%d")
    peak_date = water_df.loc[peak_idx, "date_dt"].strftime("%Y-%m-%d")
    max_area = float(water_df["water_area_km2"].max())
    classified.loc[:, ["water_period_start", "water_period_end", "peak_date", "max_area_km2"]] = [first_water, last_water, peak_date, max_area]
    return classified, {"lake_id": lake_id, "first_water": first_water, "last_water": last_water, "peak_date": peak_date, "max_area_km2": max_area}


def save_classification_checkpoint(classified_records, period_map, method_paths):
    if classified_records:
        classified_df = pd.DataFrame(classified_records)
        sort_cols = [col for col in ["lake_id", "date", "file_basename", "preview_name"] if col in classified_df.columns]
        if sort_cols:
            classified_df = classified_df.sort_values(sort_cols)
    else:
        classified_df = pd.DataFrame()
    classified_df.to_csv(method_paths["classified_csv"], index=False, encoding="utf-8-sig")

    period_rows = list(period_map.values())
    period_df = pd.DataFrame(period_rows)
    if not period_df.empty and "lake_id" in period_df.columns:
        period_df = period_df.sort_values("lake_id")
    period_df.to_csv(method_paths["water_period_csv"], index=False, encoding="utf-8-sig")
    return classified_df


def process_method(method_paths, buffer_map, original_map):
    os.makedirs(method_paths["output_dir"], exist_ok=True)
    os.makedirs(method_paths["single_view_dir"], exist_ok=True)
    print_run_summary(method_paths)

    if not os.path.exists(method_paths["raw_metrics_csv"]):
        print(f"[04_2] skip {method_paths['method_name']}: input not found")
        return

    df = pd.read_csv(method_paths["raw_metrics_csv"], encoding="utf-8-sig")
    if "date" in df.columns:
        date_dt = pd.to_datetime(df["date"].astype(str).str.extract(r"(\d{8})", expand=False), format="%Y%m%d", errors="coerce")
        before_count = len(df)
        df = df[date_dt.notna() & (date_dt <= DATE_END)].copy()
        print(f"[04_2] applied date end filter <= {DATE_END.strftime('%Y-%m-%d')}: {len(df)}/{before_count} records remain")
    if "size_category" in df.columns and TARGET_SIZE_CATEGORIES:
        df = df[df["size_category"].astype(str).isin(TARGET_SIZE_CATEGORIES)].copy()
        print(f"[04_2] applied size filter {TARGET_SIZE_CATEGORIES}: {len(df)} records remain")
    if "lake_id" in df.columns:
        before_lake_filter = len(df)
        df["lake_id"] = pd.to_numeric(df["lake_id"], errors="coerce")
        df = df[df["lake_id"].map(is_target_lake_id)].copy()
        df["lake_id"] = df["lake_id"].astype(int)
        print(f"[04_2] applied lake filter (<= {MAX_LAKE_ID}): {len(df)}/{before_lake_filter} records remain")

    single_view_records = load_existing_single_view_records(method_paths["single_view_index_csv"])
    if single_view_records:
        print(f"[04_2] resume enabled: loaded {len(single_view_records)} existing scene records")
    classified_records = load_existing_classified_records(method_paths["classified_csv"])
    if classified_records:
        print(f"[04_2] resume enabled: loaded {len(classified_records)} existing classified records")
    period_map = load_existing_period_map(method_paths["water_period_csv"])
    if period_map:
        print(f"[04_2] resume enabled: loaded {len(period_map)} existing period rows")
    existing_classified_by_key = {build_scene_key(row): row for row in classified_records}
    existing_lake_ids = {int(row["lake_id"]) for row in classified_records if "lake_id" in row and str(row["lake_id"]).strip() != ""}
    grouped_lakes = [(int(lake_id), group.copy()) for lake_id, group in df.groupby("lake_id", sort=True)]
    total_lakes = len(grouped_lakes)
    completed_lakes = 0
    for lake_id, lake_df in grouped_lakes:
        if lake_id not in buffer_map or lake_id not in original_map:
            continue

        existing_by_key = {build_scene_key(row): row for row in single_view_records}
        task_records = [row.to_dict() for _, row in lake_df.iterrows()]
        lake_records = []
        reused_count = 0
        rendered_count = 0
        for record in task_records:
            scene_key = build_scene_key(record)
            preview_name = str(record.get("preview_name", "")).strip()
            size_category = str(record.get("size_category", "")).strip()
            fallback_path = os.path.join(method_paths["single_view_dir"], size_category, str(lake_id), preview_name)

            if scene_key in existing_by_key:
                reused_row = dict(existing_by_key[scene_key])
                reused_path = str(reused_row.get("raw_image_path", "")).strip()
                if reused_path and os.path.isfile(reused_path):
                    lake_records.append(reused_row)
                    reused_count += 1
                    continue
                if os.path.isfile(fallback_path):
                    reused_row["raw_image_path"] = fallback_path
                    lake_records.append(reused_row)
                    reused_count += 1
                    continue

            if preview_name and os.path.isfile(fallback_path):
                reused_row = dict(record)
                reused_row["raw_image_path"] = fallback_path
                lake_records.append(reused_row)
                reused_count += 1
                continue

            try:
                single_view_path = save_single_view(
                    record,
                    buffer_map[lake_id],
                    original_map[lake_id],
                    method_paths["single_view_dir"],
                )
                output_record = dict(record)
                output_record["raw_image_path"] = single_view_path
                lake_records.append(output_record)
                rendered_count += 1
            except Exception as exc:
                print(f"[04_2] skip failed row in lake {lake_id}: {exc}")
                continue

        single_view_records.extend(lake_records)
        single_view_records = save_single_view_checkpoint(single_view_records, method_paths["single_view_index_csv"])

        lake_scene_keys = [build_scene_key(row) for row in lake_records]
        lake_has_existing_classified = bool(lake_records) and all(key in existing_classified_by_key for key in lake_scene_keys)

        if lake_has_existing_classified:
            lake_classified_rows = [dict(existing_classified_by_key[key]) for key in lake_scene_keys]
            lake_classified = pd.DataFrame(lake_classified_rows)
            print(f"[04_2] reuse classified lake {lake_id}: {len(lake_classified_rows)} scenes already done")
        elif lake_records:
            lake_classified = prepare_lake_for_classification(pd.DataFrame(lake_records))
            lake_classified = classify_group_otsu(lake_classified)
            lake_classified = audit_group(lake_classified)
            lake_classified = apply_export_class_rules(lake_classified)
            lake_classified, period_row = annotate_water_period_for_lake(lake_classified)
            lake_classified_rows = lake_classified.to_dict(orient="records")
            classified_records = [row for row in classified_records if int(row.get("lake_id", -1)) != lake_id]
            classified_records.extend(lake_classified_rows)
            period_map[int(period_row["lake_id"])] = period_row
            existing_lake_ids.discard(lake_id)
            existing_classified_by_key.update({build_scene_key(row): row for row in lake_classified_rows})
            copy_classified_views(lake_classified, method_paths)
        else:
            lake_classified = pd.DataFrame()

        _ = save_classification_checkpoint(classified_records, period_map, method_paths)
        completed_lakes += 1
        print(
            f"[04_2] draw {completed_lakes}/{total_lakes} lakes done | lake={lake_id} | "
            f"scenes={len(lake_records)} | reused={reused_count} | rendered={rendered_count}"
        )
    single_view_records = save_single_view_checkpoint(single_view_records, method_paths["single_view_index_csv"])
    classified = save_classification_checkpoint(classified_records, period_map, method_paths)
    if classified.empty:
        print(f"[04_2] no single views generated for {method_paths['method_name']}")
        return
    print_completion_summary(
        method_paths=method_paths,
        input_records=len(df),
        lake_count=int(classified["lake_id"].nunique()) if not classified.empty else 0,
        classified=classified,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    buffer_map, original_map = load_lake_geometries()
    for method_name in METHOD_NAMES:
        process_method(get_method_paths(method_name), buffer_map, original_map)


if __name__ == "__main__":
    main()


