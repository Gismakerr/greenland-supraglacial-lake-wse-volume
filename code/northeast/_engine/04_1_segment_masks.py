"""
Stage 04_1: segment water masks and measure core metrics in one pass.

Outputs:
- mask GDB layers
- per-lake metric CSVs
- merged metric CSV for downstream 04_3 plotting
"""

import glob
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as ndi
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import geometry_mask, shapes
from rasterio.mask import mask
from shapely.geometry import shape
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("empty input for otsu")
        hist, bin_edges = np.histogram(arr, bins=256)
        hist = hist.astype(np.float64)
        prob = hist / np.sum(hist)
        omega = np.cumsum(prob)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        mu = np.cumsum(prob * bin_mids)
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
        return float(bin_mids[np.nanargmax(sigma_b2)])


BASE_LOCAL = r"Z:\冰面湖水量测算_Local"
PROJECT_ROOT_DIR = os.path.join(BASE_LOCAL, "冰面湖水位论文代码数据整理")
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, "result")
PREVIEW_BASE_DIR = os.path.join(OUTPUT_ROOT_DIR, "03_Classified_Previews")
NDWI_DIR = os.path.join(OUTPUT_ROOT_DIR, "01_Preprocessing_Results", "03_NDWI_ice")
S2_IMAGE_DIR = os.path.join(OUTPUT_ROOT_DIR, "01_Preprocessing_Results", "01_S2_Imgagery_Pro")
BUFFERED_LAKE_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons_Buffered.shp"
)
LAKE_AREA_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons.shp"
)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_1_segment_masks")

TARGET_SUBDIR = "02_Clean_Valid"
TARGET_SIZE_CATEGORIES = []
TARGET_LAKE_POSITIONS = ["冰面湖"]
TARGET_LAKE_IDS = []
MAX_LAKE_ID = 99999
TEST_MODE = False
TEST_LAKE_LIMIT = 50

METHOD_CONFIGS = [
    {"method_name": "Otsu", "threshold_mode": "otsu", "fixed_threshold": None},
]

USE_PARALLEL = True
MAX_WORKERS = 4
BATCH_SIZE = 12
HOLE_FILL_THRESHOLD_KM2 = 0.01
HARD_DELETE_THRESHOLD_KM2 = 0.01
MIN_VALID_PIXEL_COUNT = 10
TOUCH_BOUNDARY_MIN_PIXELS = 5
GDB_WRITE_RETRIES = 3
GDB_WRITE_RETRY_SLEEP_SECONDS = 1.0

WORKER_TASK_MAP = {}
WORKER_BUFFER_MAP = {}
WORKER_ORIGINAL_MAP = {}
WORKER_REF_AREA_MAP = {}

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def get_method_paths(method_name):
    method_dir = os.path.join(OUTPUT_DIR, method_name)
    return {
        "method_dir": method_dir,
        "mask_gdb_dir": os.path.join(method_dir, "01_Masks_GDB"),
        "raw_metrics_csv": os.path.join(method_dir, "04_1_metrics_raw.csv"),
        "per_lake_dir": os.path.join(method_dir, "04_1_metrics_by_lake"),
    }


def parse_preview_scene_info(filename):
    match = re.search(r"overlay_(\d+)_(\d{8})_(\d{6})_(S2A|S2B)\.png$", filename, re.IGNORECASE)
    if not match:
        return None
    return {
        "lake_id": int(match.group(1)),
        "date": match.group(2),
        "time": match.group(3),
        "satellite": match.group(4).upper(),
        "preview_name": filename,
    }


def sanitize_layer_name(preview_name):
    preview_stem = os.path.splitext(preview_name)[0]
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", preview_stem)
    return sanitized[:60]


def find_scene_file(directory, date_str, time_str, satellite, suffix, preferred_keywords=None):
    pattern = os.path.join(directory, f"*{date_str}*{satellite}*.{suffix}")
    candidates = glob.glob(pattern)
    if preferred_keywords:
        preferred = [
            path for path in candidates
            if any(key in os.path.basename(path) for key in preferred_keywords)
        ]
        if preferred:
            candidates = preferred
    exact_token = f"{date_str}T{time_str}"
    exact_matches = [path for path in candidates if exact_token in os.path.basename(path)]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return sorted(exact_matches)[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def get_lake_tasks():
    allowed_lake_ids = None
    if TARGET_LAKE_POSITIONS:
        try:
            buffer_gdf = gpd.read_file(BUFFERED_LAKE_SHP)
            if "lake_id" in buffer_gdf.columns and "lake_pos" in buffer_gdf.columns:
                allowed_lake_ids = set(
                    int(row["lake_id"])
                    for _, row in buffer_gdf[buffer_gdf["lake_pos"].isin(TARGET_LAKE_POSITIONS)].iterrows()
                )
        except Exception as exc:
            print(f"[04_1] warning: failed to apply lake_pos filter: {exc}")

    tasks = []
    for size_cat in os.listdir(PREVIEW_BASE_DIR):
        if TARGET_SIZE_CATEGORIES and size_cat not in TARGET_SIZE_CATEGORIES:
            continue
        size_dir = os.path.join(PREVIEW_BASE_DIR, size_cat)
        if not os.path.isdir(size_dir):
            continue
        for lid_str in os.listdir(size_dir):
            lake_dir = os.path.join(size_dir, lid_str)
            if not lid_str.isdigit() or not os.path.isdir(lake_dir):
                continue
            lake_id = int(lid_str)
            if TARGET_LAKE_IDS and lake_id not in TARGET_LAKE_IDS:
                continue
            if MAX_LAKE_ID and MAX_LAKE_ID > 0 and lake_id > int(MAX_LAKE_ID):
                continue
            if allowed_lake_ids is not None and lake_id not in allowed_lake_ids:
                continue
            tasks.append((size_cat, lake_id, lake_dir))

    tasks.sort(key=lambda item: (item[0], item[1]))
    if TEST_MODE:
        return tasks[:TEST_LAKE_LIMIT]
    return tasks


def save_mask_gdb(gdb_path, layer_name, water_mask, out_transform, crs, lake_id, date_str):
    os.makedirs(os.path.dirname(gdb_path), exist_ok=True)

    polygon_records = []
    for geom_json, value in shapes(
        water_mask.astype(np.uint8),
        mask=water_mask.astype(bool),
        transform=out_transform,
        connectivity=8,
    ):
        if int(value) != 1:
            continue
        polygon_records.append(
            {
                "lake_id": int(lake_id),
                "date": date_str,
                "mask_val": 1,
                "geometry": shape(geom_json),
            }
        )

    if not polygon_records:
        return False

    gdf = gpd.GeoDataFrame(polygon_records, crs=crs)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return False

    if gdf.crs and getattr(gdf.crs, "is_geographic", False):
        metric_gdf = gdf.to_crs(gdf.estimate_utm_crs())
        area_km2 = metric_gdf.geometry.area / 1e6
    else:
        area_km2 = gdf.geometry.area / 1e6
    gdf = gdf[area_km2 >= HARD_DELETE_THRESHOLD_KM2].copy()
    if gdf.empty:
        return False

    errors = []
    for attempt in range(1, GDB_WRITE_RETRIES + 1):
        for driver in ["OpenFileGDB", "FileGDB"]:
            try:
                gdf.to_file(gdb_path, layer=layer_name, driver=driver)
                return True
            except Exception as exc:
                errors.append(f"attempt={attempt}, driver={driver}, error={exc}")
        if attempt < GDB_WRITE_RETRIES:
            time.sleep(GDB_WRITE_RETRY_SLEEP_SECONDS)
    raise RuntimeError(
        f"failed to write GDB layer: {gdb_path} | {layer_name} | " + " || ".join(errors)
    )


def build_geometry_mask(geometries, out_shape, transform, invert=False, all_touched=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return geometry_mask(
            geometries,
            out_shape=out_shape,
            transform=transform,
            invert=invert,
            all_touched=all_touched,
        )


def compute_metrics_dict(base_record, data, valid_mask, water_mask, out_transform, original_aoi_geom, ref_area_km2):
    pixel_area = abs(out_transform[0] * out_transform[4])
    structure = ndi.generate_binary_structure(2, 2)
    labeled_final, num_final = ndi.label(water_mask, structure=structure)

    water_size = int(np.sum(water_mask))
    total_size = int(np.sum(valid_mask))
    bg_size = total_size - water_size
    water_area_km2 = (water_size * pixel_area) / 1e6
    water_ratio = water_size / total_size if total_size else 0.0
    background_ratio = bg_size / total_size if total_size else 0.0
    area_ratio_w_b = water_size / bg_size if bg_size > 0 else np.inf
    num_features_final = int(num_final)
    mu_water = float(np.nanmean(data[water_mask])) if np.any(water_mask) else np.nan
    mu_bg = float(np.nanmean(data[valid_mask & (~water_mask)])) if np.any(valid_mask & (~water_mask)) else np.nan

    touches_boundary = False
    edge_mask = np.zeros_like(water_mask, dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    if np.sum(water_mask & edge_mask) >= TOUCH_BOUNDARY_MIN_PIXELS:
        touches_boundary = True
    if not touches_boundary:
        filled_valid_mask = ndi.binary_fill_holes(valid_mask)
        dilated = ndi.binary_dilation(water_mask, structure=np.ones((3, 3)))
        buffer_coll_mask = dilated & (~filled_valid_mask)
        if np.any(buffer_coll_mask):
            coll_source = ndi.binary_dilation(buffer_coll_mask, structure=np.ones((3, 3))) & water_mask
            touches_boundary = np.sum(coll_source) >= TOUCH_BOUNDARY_MIN_PIXELS

    out_ratio = 0.0
    in_ratio = 0.0
    coverage_ratio = 0.0
    has_component_inside_aoi = False
    component_inside_aoi_count = 0
    top_3_areas = [0.0, 0.0, 0.0]
    top_3_aoi_ratios = [0.0, 0.0, 0.0]
    top_3_is_fully_inside_aoi = [False, False, False]
    top_3_mean_ndwi = [np.nan, np.nan, np.nan]
    component_stats = []

    if num_final > 0:
        final_sizes = np.bincount(labeled_final.ravel())[1:]
        sorted_component_ids = [int(idx) + 1 for idx in np.argsort(final_sizes)[::-1]]
        for rank_idx, component_id in enumerate(sorted_component_ids[:3]):
            component_mask = labeled_final == component_id
            component_size = int(np.sum(component_mask))
            top_3_areas[rank_idx] = (component_size * pixel_area) / 1e6
            if np.any(component_mask):
                top_3_mean_ndwi[rank_idx] = float(np.nanmean(data[component_mask]))
            component_stats.append({"component_mask": component_mask, "rank_idx": rank_idx})

    if original_aoi_geom is not None and np.any(water_mask):
        out_shp_mask = build_geometry_mask(
            [original_aoi_geom],
            out_shape=water_mask.shape,
            transform=out_transform,
            invert=False,
        )
        ring_buffer_pixels = valid_mask & out_shp_mask
        ring_buffer_size = np.sum(ring_buffer_pixels)
        if ring_buffer_size > 0:
            out_ratio = float(np.sum(water_mask & out_shp_mask) / ring_buffer_size)

        inside_mask = ~out_shp_mask
        max_lake_size = np.sum(valid_mask & inside_mask)
        if max_lake_size > 0:
            in_ratio = float(np.sum(water_mask & inside_mask) / max_lake_size)

        for component_id in range(1, num_final + 1):
            component_mask = labeled_final == component_id
            if np.any(component_mask) and not np.any(component_mask & out_shp_mask) and np.any(component_mask & inside_mask):
                has_component_inside_aoi = True
                component_inside_aoi_count += 1

        for stat in component_stats:
            component_mask = stat["component_mask"]
            rank_idx = stat["rank_idx"]
            top_3_is_fully_inside_aoi[rank_idx] = bool(
                np.any(component_mask & inside_mask) and not np.any(component_mask & out_shp_mask)
            )

        aoi_raster = ~build_geometry_mask(
            [original_aoi_geom],
            transform=out_transform,
            out_shape=water_mask.shape,
            all_touched=False,
        )
        aoi_pixels = np.sum(aoi_raster)
        if aoi_pixels > 0:
            coverage_ratio = float(np.sum(water_mask & aoi_raster) / aoi_pixels)
            component_aoi_ratios = []
            for component_id in range(1, num_final + 1):
                component_mask = labeled_final == component_id
                component_aoi_ratios.append(np.sum(component_mask & aoi_raster) / aoi_pixels)
            for idx, ratio in enumerate(sorted(component_aoi_ratios, reverse=True)[:3]):
                top_3_aoi_ratios[idx] = float(ratio)

    result = dict(base_record)
    result.update(
        {
            "num_features_final": num_features_final,
            "num_features": num_features_final,
            "water_area_km2": water_area_km2,
            "water_ratio": water_ratio,
            "background_ratio": background_ratio,
            "area_ratio_w_b": area_ratio_w_b,
            "mu_water": mu_water,
            "mu_bg": mu_bg,
            "out_ratio": out_ratio,
            "in_ratio": in_ratio,
            "coverage_ratio": coverage_ratio,
            "has_component_inside_aoi": bool(has_component_inside_aoi),
            "component_inside_aoi_count": int(component_inside_aoi_count),
            "touches_boundary": bool(touches_boundary),
            "top_3_area_1": top_3_areas[0],
            "top_3_area_2": top_3_areas[1],
            "top_3_area_3": top_3_areas[2],
            "top_3_aoi_ratio_1": top_3_aoi_ratios[0],
            "top_3_aoi_ratio_2": top_3_aoi_ratios[1],
            "top_3_aoi_ratio_3": top_3_aoi_ratios[2],
            "top_3_is_fully_inside_aoi_1": bool(top_3_is_fully_inside_aoi[0]),
            "top_3_is_fully_inside_aoi_2": bool(top_3_is_fully_inside_aoi[1]),
            "top_3_is_fully_inside_aoi_3": bool(top_3_is_fully_inside_aoi[2]),
            "top_3_mean_ndwi_1": top_3_mean_ndwi[0],
            "top_3_mean_ndwi_2": top_3_mean_ndwi[1],
            "top_3_mean_ndwi_3": top_3_mean_ndwi[2],
            "initial_hint": "NonWater" if touches_boundary else "Water",
            "ref_area_km2": ref_area_km2,
        }
    )
    return result


def init_worker(batch_tasks):
    global WORKER_TASK_MAP, WORKER_BUFFER_MAP, WORKER_ORIGINAL_MAP, WORKER_REF_AREA_MAP

    os.environ["GDAL_CACHEMAX"] = "128"
    WORKER_TASK_MAP = {int(lake_id): (size_cat, lake_dir) for size_cat, lake_id, lake_dir in batch_tasks}

    batch_lake_ids = {int(lake_id) for _, lake_id, _ in batch_tasks}
    buffer_gdf = gpd.read_file(BUFFERED_LAKE_SHP)
    area_gdf = gpd.read_file(LAKE_AREA_SHP)
    if "lake_id" in area_gdf.columns:
        allowed_lake_ids = set(pd.to_numeric(area_gdf["lake_id"], errors="coerce").dropna().astype(int).tolist())
        buffer_gdf = buffer_gdf[pd.to_numeric(buffer_gdf["lake_id"], errors="coerce").astype("Int64").isin(sorted(allowed_lake_ids))].copy()

    WORKER_BUFFER_MAP = {}
    WORKER_ORIGINAL_MAP = {}
    WORKER_REF_AREA_MAP = {}

    for _, row in buffer_gdf.iterrows():
        if "lake_id" in row:
            lid = int(row["lake_id"])
            if lid in batch_lake_ids:
                WORKER_BUFFER_MAP[lid] = row.geometry

    for _, row in area_gdf.iterrows():
        if "lake_id" in row:
            lid = int(row["lake_id"])
            if lid in batch_lake_ids:
                WORKER_ORIGINAL_MAP[lid] = row.geometry
                if "area" in row:
                    WORKER_REF_AREA_MAP[lid] = float(row["area"]) / 1e6


def process_single_lake(lake_id, method_cfg, method_paths):
    lake_task = WORKER_TASK_MAP.get(int(lake_id))
    buffered_aoi_geom = WORKER_BUFFER_MAP.get(int(lake_id))
    original_aoi_geom = WORKER_ORIGINAL_MAP.get(int(lake_id))
    ref_area_km2 = WORKER_REF_AREA_MAP.get(int(lake_id), 0.0)
    if lake_task is None or buffered_aoi_geom is None:
        return []

    size_category, lake_dir = lake_task
    target_dir = os.path.join(lake_dir, TARGET_SUBDIR)
    if not os.path.isdir(target_dir):
        return []

    preview_scene_infos = []
    for filename in sorted(os.listdir(target_dir)):
        if not filename.endswith(".png"):
            continue
        scene_info = parse_preview_scene_info(filename)
        if scene_info is not None:
            preview_scene_infos.append(scene_info)

    records = []
    for scene_info in preview_scene_infos:
        date = scene_info["date"]
        time_str = scene_info["time"]
        satellite = scene_info["satellite"]
        preview_name = scene_info["preview_name"]

        ndwi_path = find_scene_file(NDWI_DIR, date, time_str, satellite, "tif")
        rgb_path = find_scene_file(
            S2_IMAGE_DIR, date, time_str, satellite, "tif", preferred_keywords=["MSIL2A"]
        )
        if ndwi_path is None:
            continue

        file_basename = os.path.basename(ndwi_path)
        try:
            with rasterio.open(ndwi_path) as src:
                out_image, out_transform = mask(src, [buffered_aoi_geom], crop=True, nodata=np.nan)
                data = out_image[0]
                out_crs = src.crs
        except ValueError as exc:
            if "do not overlap raster" in str(exc):
                continue
            print(f"[04_1] lake {lake_id} date {date} failed: {exc}")
            continue
        except Exception as exc:
            print(f"[04_1] lake {lake_id} date {date} failed: {exc}")
            continue

        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            continue
        valid_pixels = data[valid_mask]
        if valid_pixels.size < MIN_VALID_PIXEL_COUNT:
            continue

        if method_cfg["threshold_mode"] == "fixed":
            thresh = float(method_cfg["fixed_threshold"])
        else:
            try:
                thresh = float(threshold_otsu(valid_pixels))
            except ValueError:
                thresh = 0.0

        pixel_area = abs(out_transform[0] * out_transform[4])
        min_hole_pixel_count = int((HOLE_FILL_THRESHOLD_KM2 * 1e6) / pixel_area)
        hard_delete_pixel_count = int((HARD_DELETE_THRESHOLD_KM2 * 1e6) / pixel_area)

        structure = ndi.generate_binary_structure(2, 2)
        water_mask_raw = (data > thresh) & valid_mask
        labeled_raw, num_raw = ndi.label(water_mask_raw, structure=structure)
        num_features_raw = int(num_raw)
        num_features_removed_by_area = 0

        if num_raw > 0:
            raw_sizes = np.bincount(labeled_raw.ravel())
            too_small_hard = raw_sizes < hard_delete_pixel_count
            too_small_hard[0] = False
            num_features_removed_by_area = int(np.sum(too_small_hard[1:]))
            water_mask_main = water_mask_raw & (~too_small_hard[labeled_raw])
        else:
            water_mask_main = np.zeros_like(water_mask_raw, dtype=bool)

        bg_mask = (~water_mask_main) & valid_mask
        labeled_bg, num_bg = ndi.label(bg_mask, structure=structure)
        if num_bg > 0:
            bg_sizes = np.bincount(labeled_bg.ravel())
            too_small_bg = bg_sizes < min_hole_pixel_count
            too_small_bg[0] = False
            water_mask_filtered = water_mask_main | too_small_bg[labeled_bg]
        else:
            water_mask_filtered = water_mask_main.copy()

        _, num_final = ndi.label(water_mask_filtered, structure=structure)
        lake_gdb_dir = os.path.join(method_paths["mask_gdb_dir"], size_category)
        os.makedirs(lake_gdb_dir, exist_ok=True)
        gdb_path = os.path.join(lake_gdb_dir, f"lake_{lake_id}.gdb")
        gdb_layer = sanitize_layer_name(preview_name)
        has_mask_features = save_mask_gdb(
            gdb_path,
            gdb_layer,
            water_mask_filtered,
            out_transform,
            out_crs,
            lake_id,
            date,
        )

        base_record = {
            "method": method_cfg["method_name"],
            "threshold_mode": method_cfg["threshold_mode"],
            "fixed_threshold": method_cfg["fixed_threshold"] if method_cfg["fixed_threshold"] is not None else "",
            "lake_id": int(lake_id),
            "size_category": size_category,
            "date": str(date),
            "preview_name": preview_name,
            "file_basename": file_basename,
            "ndwi_path": ndwi_path,
            "rgb_path": rgb_path or "",
            "gdb_path": gdb_path,
            "gdb_layer": gdb_layer if has_mask_features else "",
            "thresh": float(thresh),
            "num_features_raw": int(num_features_raw),
            "num_features_removed_by_area": int(num_features_removed_by_area),
            "num_features_final": int(num_final),
        }
        records.append(
            compute_metrics_dict(
                base_record,
                data,
                valid_mask,
                water_mask_filtered,
                out_transform,
                original_aoi_geom,
                ref_area_km2,
            )
        )

    return records


def print_run_summary(method_cfg, method_paths, task_count):
    mode_label = f"parallel({MAX_WORKERS})" if USE_PARALLEL and MAX_WORKERS > 1 else "serial"
    print(f"[04_1] {method_cfg['method_name']} start")
    print(f"  tasks: {task_count}")
    print(f"  mode: {mode_label}, batch_size={BATCH_SIZE}")
    print(f"  preview_dir: {PREVIEW_BASE_DIR}")
    print(f"  output_dir: {method_paths['method_dir']}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = get_lake_tasks()
    if not tasks:
        print("[04_1] no tasks found")
        return

    run_parallel = USE_PARALLEL and MAX_WORKERS > 1
    for method_cfg in METHOD_CONFIGS:
        method_paths = get_method_paths(method_cfg["method_name"])
        os.makedirs(method_paths["method_dir"], exist_ok=True)
        os.makedirs(method_paths["mask_gdb_dir"], exist_ok=True)
        os.makedirs(method_paths["per_lake_dir"], exist_ok=True)
        print_run_summary(method_cfg, method_paths, len(tasks))

        all_records = []
        processed_count = 0
        total_batches = int(np.ceil(len(tasks) / BATCH_SIZE))
        for batch_idx, batch_tasks in enumerate(chunked(tasks, BATCH_SIZE), start=1):
            print(f"[04_1] batch {batch_idx}/{total_batches}, lakes={len(batch_tasks)}")
            batch_records = []
            if run_parallel:
                with ProcessPoolExecutor(
                    max_workers=MAX_WORKERS,
                    initializer=init_worker,
                    initargs=(batch_tasks,),
                ) as executor:
                    futures = {
                        executor.submit(process_single_lake, lake_id, method_cfg, method_paths): lake_id
                        for _, lake_id, _ in batch_tasks
                    }
                    for future in as_completed(futures):
                        lake_id = futures[future]
                        try:
                            lake_records = future.result()
                        except Exception as exc:
                            print(f"[04_1] lake {lake_id} failed: {exc}")
                            lake_records = []
                        processed_count += 1
                        batch_records.extend(lake_records)
                        if processed_count % 10 == 0:
                            print(f"[04_1] progress {processed_count}/{len(tasks)}")
            else:
                init_worker(batch_tasks)
                for _, lake_id, _ in batch_tasks:
                    try:
                        lake_records = process_single_lake(lake_id, method_cfg, method_paths)
                    except Exception as exc:
                        print(f"[04_1] lake {lake_id} failed: {exc}")
                        lake_records = []
                    processed_count += 1
                    batch_records.extend(lake_records)
                    if processed_count % 10 == 0:
                        print(f"[04_1] progress {processed_count}/{len(tasks)}")

            if batch_records:
                batch_df = pd.DataFrame(batch_records).sort_values(["lake_id", "date", "file_basename"])
                for lake_id, lake_df in batch_df.groupby("lake_id", sort=True):
                    lake_csv = os.path.join(method_paths["per_lake_dir"], f"lake_{int(lake_id)}_metrics.csv")
                    if os.path.exists(lake_csv):
                        existing_df = pd.read_csv(lake_csv, encoding="utf-8-sig")
                        lake_df = pd.concat([existing_df, lake_df], ignore_index=True)
                        lake_df = lake_df.drop_duplicates(subset=["date", "preview_name"], keep="last")
                        lake_df = lake_df.sort_values(["date", "file_basename"])
                    lake_df.to_csv(lake_csv, index=False, encoding="utf-8-sig")
                all_records.extend(batch_df.to_dict("records"))

        final_df = pd.DataFrame(all_records)
        if not final_df.empty:
            final_df = final_df.sort_values(["lake_id", "date", "file_basename"]).reset_index(drop=True)
        final_df.to_csv(method_paths["raw_metrics_csv"], index=False, encoding="utf-8-sig")
        print(f"[04_1] done: {method_paths['raw_metrics_csv']} | records={len(final_df)}")


if __name__ == "__main__":
    main()

