"""
Stage 03: generate per-lake preview images and classify cloud-polluted scenes.

This refactored version is restricted to:
1) ice-surface lakes only (lake_pos == "鍐伴潰婀?)
2) large/medium lakes only (lake_type in ["Large", "Medium"])
"""

from __future__ import annotations

import gc
import glob
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from rasterio.errors import WindowError
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

matplotlib.use("Agg")


# --------------------- Config ---------------------
OUTPUT_ROOT_DIR = str(os.environ.get("OUTPUT_ROOT_DIR", r"X:\2024_西南水位曲线\result1")).strip()
IMAGE_DIR = r"X:\\2024_西南水位曲线\\data1\\S2_22WEV_2024-06-01_to_2024-08-30"
CLOUD_MASK_DIR = os.path.join(OUTPUT_ROOT_DIR, "01_Preprocessing_Results", "06_Cloud_Mask")

BUFFER_SHP = os.path.join(
    OUTPUT_ROOT_DIR,
    "02_Lake_Extraction",
    "04_Filtered_Lakes",
    "Water_Max_Filtered_Polygons_Buffered.shp",
)
LAKE_SHP = os.path.join(
    OUTPUT_ROOT_DIR,
    "02_Lake_Extraction",
    "04_Filtered_Lakes",
    "Water_Max_Filtered_Polygons.shp",
)

OUTPUT_BASE_DIR = os.path.join(OUTPUT_ROOT_DIR, "03_Classified_Previews")

CLOUD_RATIO_THRESHOLD = 0.1
NODATA_RATIO_THRESHOLD = 0.7

DIR_NAME_ORIGINAL = "01_Original_Base"
DIR_NAME_CLEAN = "02_Clean_Valid"
DIR_NAME_CLOUD = "03_Cloud_Polluted"
DIR_NAME_PLAIN = "04_Clean"
UNIFIED_CLOUD_MASK_DIR = "00_Unified_Cloud_Masks"

TARGET_CRS = "EPSG:32622"
USE_PARALLEL = True
MAX_WORKERS = 4
BATCH_SIZE = 12
OVERWRITE = False
END_DATE = None  # example: "20240820"

# required filter in this project
TARGET_SIZE_CATEGORIES = ["Large", "Medium"]
TARGET_LAKE_POSITIONS = []
TARGET_LAKE_IDS = []
MAX_LAKE_ID_EXCLUSIVE = 10**9
# --------------------------------------------------

WORKER_BUFFER_INDEX = {}
WORKER_LAKE_INDEX = {}
WORKER_IMAGE_MAP = {}


def get_id_column(gdf: gpd.GeoDataFrame) -> str:
    if "lake_id" in gdf.columns:
        return "lake_id"
    if "id" in gdf.columns:
        return "id"
    return gdf.columns[0]


def _is_allowed_lake_id(lake_id) -> bool:
    try:
        return int(lake_id) < int(MAX_LAKE_ID_EXCLUSIVE)
    except Exception:
        return False


def get_image_map(directory: str):
    image_map = defaultdict(list)
    for f in glob.glob(os.path.join(directory, "*.tif")):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if m:
            image_map[m.group(1)].append(f)
    return image_map


def get_mask_path(image_path: str, mask_dir: str, mask_type: str):
    basename = os.path.basename(image_path)
    stem = os.path.splitext(basename)[0]

    c1 = os.path.join(mask_dir, f"{stem}_{mask_type}.tif")
    if os.path.exists(c1):
        return c1
    c2 = os.path.join(mask_dir, f"{mask_type}_{basename}")
    if os.path.exists(c2):
        return c2

    dm = re.search(r"(\d{8})", basename)
    tm = re.search(r"T(\d{6})", basename)
    if dm:
        found = glob.glob(os.path.join(mask_dir, f"*{dm.group(1)}*{mask_type}*.tif"))
        if found:
            if tm:
                for p in found:
                    if tm.group(1) in os.path.basename(p):
                        return p
            return found[0]
    return None


def normalize(array: np.ndarray) -> np.ndarray:
    valid = array > 0
    if not valid.any():
        return array
    p2, p98 = np.percentile(array[valid], (2, 98))
    if p98 == p2:
        return array.astype(np.float32, copy=False)
    return np.clip((array - p2) / (p98 - p2), 0, 1).astype(np.float32, copy=False)


def init_daily_summary_row(lake_size_cat, lake_id, date_str):
    return {
        "lake_size": lake_size_cat,
        "lake_id": int(lake_id),
        "date": str(date_str),
        "total_images": 0,
        "valid_pixel_images": 0,
        "no_valid_pixel_images": 0,
        "cloud_polluted_images": 0,
        "clean_valid_images": 0,
    }


def finalize_daily_summary_rows(daily_summary_map):
    rows = []
    for date_str in sorted(daily_summary_map.keys()):
        row = daily_summary_map[date_str].copy()
        row["valid_pixel_images"] = int(row["cloud_polluted_images"]) + int(row["clean_valid_images"])
        row["classified_sum_check"] = int(row["no_valid_pixel_images"]) + int(row["valid_pixel_images"])
        row["matches_total"] = int(row["classified_sum_check"]) == int(row["total_images"])
        rows.append(row)
    return rows


def init_worker(lake_ids, image_map):
    global WORKER_BUFFER_INDEX, WORKER_LAKE_INDEX, WORKER_IMAGE_MAP
    gdf_buffer = gpd.read_file(BUFFER_SHP)
    gdf_lake = gpd.read_file(LAKE_SHP)

    if "lake_type" in gdf_buffer.columns:
        gdf_buffer = gdf_buffer[gdf_buffer["lake_type"].isin(TARGET_SIZE_CATEGORIES)]
    if "lake_type" in gdf_lake.columns:
        gdf_lake = gdf_lake[gdf_lake["lake_type"].isin(TARGET_SIZE_CATEGORIES)]
    if "ice_flag" in gdf_buffer.columns:
        gdf_buffer = gdf_buffer[gdf_buffer["ice_flag"].astype(int) == 1]
    elif "lake_pos" in gdf_buffer.columns:
        gdf_buffer = gdf_buffer[gdf_buffer["lake_pos"].isin(TARGET_LAKE_POSITIONS)]
    if "ice_flag" in gdf_lake.columns:
        gdf_lake = gdf_lake[gdf_lake["ice_flag"].astype(int) == 1]
    elif "lake_pos" in gdf_lake.columns:
        gdf_lake = gdf_lake[gdf_lake["lake_pos"].isin(TARGET_LAKE_POSITIONS)]

    bid = get_id_column(gdf_buffer)
    lid = get_id_column(gdf_lake)
    lake_id_set = {int(x) for x in lake_ids}
    gdf_buffer = gdf_buffer[gdf_buffer[bid].astype(int).isin(lake_id_set)].copy()
    gdf_lake = gdf_lake[gdf_lake[lid].astype(int).isin(lake_id_set)].copy()

    if gdf_buffer.crs != TARGET_CRS:
        gdf_buffer = gdf_buffer.to_crs(TARGET_CRS)
    if gdf_lake.crs != TARGET_CRS:
        gdf_lake = gdf_lake.to_crs(TARGET_CRS)

    WORKER_BUFFER_INDEX = {int(r[bid]): r for _, r in gdf_buffer.iterrows()}
    WORKER_LAKE_INDEX = {int(r[lid]): r for _, r in gdf_lake.iterrows()}
    WORKER_IMAGE_MAP = dict(image_map)


def process_single_lake(lake_id: int):
    buffer_row = WORKER_BUFFER_INDEX.get(int(lake_id))
    lake_row = WORKER_LAKE_INDEX.get(int(lake_id))
    if buffer_row is None or lake_row is None:
        return

    buffer_geom = buffer_row.geometry
    lake_geom = lake_row.geometry
    lake_size_cat = str(lake_row.get("lake_type", buffer_row.get("lake_type", "Unknown")))

    lake_out_dir = os.path.join(OUTPUT_BASE_DIR, lake_size_cat, str(lake_id))
    dir_orig = os.path.join(lake_out_dir, DIR_NAME_ORIGINAL)
    dir_clean = os.path.join(lake_out_dir, DIR_NAME_CLEAN)
    dir_cloud = os.path.join(lake_out_dir, DIR_NAME_CLOUD)
    dir_plain = os.path.join(lake_out_dir, DIR_NAME_PLAIN)
    for d in [dir_orig, dir_clean, dir_cloud, dir_plain]:
        os.makedirs(d, exist_ok=True)

    minx, miny, maxx, maxy = buffer_geom.bounds
    pad = 200
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    dst_res = 10
    dst_w = int((maxx - minx) / dst_res)
    dst_h = int((maxy - miny) / dst_res)
    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, dst_w, dst_h)

    daily_summary_map = {}
    polluted_records = []

    sorted_dates = sorted(WORKER_IMAGE_MAP.keys())
    if END_DATE:
        sorted_dates = [d for d in sorted_dates if d <= END_DATE]

    for date_str in sorted_dates:
        daily_summary_map.setdefault(date_str, init_daily_summary_row(lake_size_cat, lake_id, date_str))
        for img_path in WORKER_IMAGE_MAP[date_str]:
            fig = None
            daily_summary_map[date_str]["total_images"] += 1
            try:
                sat_id = "S2A" if "S2A" in os.path.basename(img_path) else ("S2B" if "S2B" in os.path.basename(img_path) else "S2X")
                tm = re.search(r"T(\d{6})", os.path.basename(img_path))
                time_str = tm.group(1) if tm else "000000"
                check_filename = f"overlay_{lake_id}_{date_str}_{time_str}_{sat_id}.png"

                cloud_mask_path = get_mask_path(img_path, CLOUD_MASK_DIR, "CloudMask") or get_mask_path(img_path, CLOUD_MASK_DIR, "Mask")
                out_cloud = os.path.join(dir_cloud, check_filename)
                out_clean = os.path.join(dir_clean, check_filename)
                out_orig = os.path.join(dir_orig, check_filename)
                out_plain = os.path.join(dir_plain, check_filename)

                if (not OVERWRITE) and (os.path.exists(out_cloud) or os.path.exists(out_clean)) and os.path.exists(out_plain):
                    if os.path.exists(out_cloud):
                        daily_summary_map[date_str]["cloud_polluted_images"] += 1
                    else:
                        daily_summary_map[date_str]["clean_valid_images"] += 1
                    continue

                with rasterio.open(img_path) as src:
                    src_crs = src.crs
                    if src_crs != TARGET_CRS:
                        src_bounds = gpd.GeoDataFrame({"geometry": [box(minx, miny, maxx, maxy)]}, crs=TARGET_CRS).to_crs(src_crs).total_bounds
                        src_window = windows.from_bounds(*src_bounds, transform=src.transform)
                    else:
                        src_window = windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                    src_window = src_window.round_offsets(op="ceil", pixel_precision=0)
                    try:
                        src_window = src_window.intersection(windows.Window(0, 0, src.width, src.height))
                    except WindowError:
                        daily_summary_map[date_str]["no_valid_pixel_images"] += 1
                        continue

                    if src_window.width <= 0 or src_window.height <= 0:
                        daily_summary_map[date_str]["no_valid_pixel_images"] += 1
                        continue

                    bands = [3, 2, 1] if src.count >= 3 else [1]
                    rep = []
                    for b in bands:
                        dat = src.read(b, window=src_window)
                        dest = np.zeros((dst_h, dst_w), dtype=np.float32)
                        reproject(
                            source=dat,
                            destination=dest,
                            src_transform=src.window_transform(src_window),
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=TARGET_CRS,
                            resampling=Resampling.bilinear,
                        )
                        rep.append(dest)

                aoi_mask = geometry_mask([buffer_geom], transform=dst_transform, invert=True, out_shape=(dst_h, dst_w))
                aoi_pixels = int(np.sum(aoi_mask))
                nodata_pixels = int(np.sum((rep[0] == 0) & aoi_mask))
                nodata_ratio = (nodata_pixels / aoi_pixels) if aoi_pixels > 0 else 1.0
                if nodata_ratio >= NODATA_RATIO_THRESHOLD:
                    daily_summary_map[date_str]["no_valid_pixel_images"] += 1
                    continue

                img_display = np.dstack(rep).astype(np.float32, copy=False) if len(rep) == 3 else rep[0].astype(np.float32, copy=False)
                img_display = normalize(img_display)
                if np.max(img_display) == 0:
                    daily_summary_map[date_str]["no_valid_pixel_images"] += 1
                    continue

                # plain image
                fig_plain, ax_plain = plt.subplots(figsize=(5, 5), dpi=100)
                if len(rep) == 3:
                    ax_plain.imshow(img_display, extent=(minx, maxx, miny, maxy), origin="upper")
                else:
                    ax_plain.imshow(img_display, extent=(minx, maxx, miny, maxy), origin="upper", cmap="gray")
                ax_plain.axis("off")
                plt.savefig(out_plain, dpi=120, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig_plain)

                cloud_display = np.zeros((dst_h, dst_w), dtype=np.uint8)
                if cloud_mask_path:
                    with rasterio.open(cloud_mask_path) as ms:
                        if ms.crs != TARGET_CRS:
                            mb = gpd.GeoDataFrame({"geometry": [box(minx, miny, maxx, maxy)]}, crs=TARGET_CRS).to_crs(ms.crs).total_bounds
                            mw = windows.from_bounds(*mb, transform=ms.transform)
                        else:
                            mw = windows.from_bounds(minx, miny, maxx, maxy, transform=ms.transform)
                        mw = mw.round_offsets(op="ceil", pixel_precision=0)
                        try:
                            mw = mw.intersection(windows.Window(0, 0, ms.width, ms.height))
                            mdat = ms.read(1, window=mw)
                            reproject(
                                source=mdat,
                                destination=cloud_display,
                                src_transform=ms.window_transform(mw),
                                src_crs=ms.crs,
                                dst_transform=dst_transform,
                                dst_crs=TARGET_CRS,
                                resampling=Resampling.nearest,
                            )
                        except WindowError:
                            pass

                cloud_ratio = float(np.sum(cloud_display[aoi_mask] > 0) / max(1, np.sum(aoi_mask)))
                polluted = cloud_ratio > CLOUD_RATIO_THRESHOLD
                out_path = out_cloud if polluted else out_clean
                status_str = "Polluted by Cloud" if polluted else "Clean Valid Data"
                text_color = "red" if polluted else "lime"

                fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
                if len(rep) == 3:
                    ax.imshow(img_display, extent=(minx, maxx, miny, maxy), origin="upper")
                else:
                    ax.imshow(img_display, extent=(minx, maxx, miny, maxy), origin="upper", cmap="gray")
                gpd.GeoSeries([lake_geom], crs=TARGET_CRS).plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=1.2)
                gpd.GeoSeries([buffer_geom], crs=TARGET_CRS).plot(ax=ax, facecolor="none", edgecolor="cyan", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.set_title(f"Lake {lake_id} - {date_str}_{time_str} ({sat_id}) [Size: {lake_size_cat}]", fontsize=9)
                ax.axis("off")
                plt.savefig(out_orig, dpi=120, bbox_inches="tight", pad_inches=0.1)

                if np.any(cloud_display > 0):
                    mc = np.ma.masked_where(cloud_display == 0, cloud_display)
                    ax.imshow(mc, extent=(minx, maxx, miny, maxy), origin="upper", cmap="Reds", alpha=0.5, vmin=0, vmax=1)
                ax.text(
                    0.98,
                    0.98,
                    f"Status: {status_str}\nCloud Cov: {cloud_ratio:.1%}",
                    transform=ax.transAxes,
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                    ha="right",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
                )
                plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

                if polluted:
                    daily_summary_map[date_str]["cloud_polluted_images"] += 1
                    polluted_records.append(
                        {
                            "lake_size": lake_size_cat,
                            "lake_id": lake_id,
                            "source_image": os.path.basename(img_path),
                            "preview_image": os.path.basename(out_path),
                            "date": date_str,
                            "time": time_str,
                            "satellite": sat_id,
                            "cloud_ratio": round(cloud_ratio, 6),
                        }
                    )
                    udir = os.path.join(OUTPUT_BASE_DIR, UNIFIED_CLOUD_MASK_DIR, lake_size_cat.capitalize())
                    os.makedirs(udir, exist_ok=True)
                    shutil.copy2(out_path, os.path.join(udir, os.path.basename(out_path)))
                else:
                    daily_summary_map[date_str]["clean_valid_images"] += 1

                del img_display, cloud_display, aoi_mask
                gc.collect()
            except Exception as exc:
                print(f"[lake={lake_id} date={date_str}] {exc}")
                daily_summary_map[date_str]["no_valid_pixel_images"] += 1
                if fig is not None:
                    plt.close(fig)
                plt.close("all")
                gc.collect()

    if polluted_records:
        pd.DataFrame(polluted_records).to_csv(
            os.path.join(lake_out_dir, "Polluted_Image_Details.csv"), index=False, encoding="utf-8-sig"
        )
    daily_rows = finalize_daily_summary_rows(daily_summary_map)
    if daily_rows:
        pd.DataFrame(daily_rows).to_csv(
            os.path.join(lake_out_dir, "Daily_Image_Classification_Summary.csv"), index=False, encoding="utf-8-sig"
        )


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    if not os.path.exists(BUFFER_SHP) or not os.path.exists(LAKE_SHP):
        print("missing shapefile inputs")
        return

    gdf_buffer = gpd.read_file(BUFFER_SHP)
    gdf_lake = gpd.read_file(LAKE_SHP)
    image_map = get_image_map(IMAGE_DIR)

    if "lake_type" in gdf_buffer.columns:
        gdf_buffer = gdf_buffer[gdf_buffer["lake_type"].isin(TARGET_SIZE_CATEGORIES)]
    if "lake_type" in gdf_lake.columns:
        gdf_lake = gdf_lake[gdf_lake["lake_type"].isin(TARGET_SIZE_CATEGORIES)]
    if "ice_flag" in gdf_buffer.columns:
        gdf_buffer = gdf_buffer[gdf_buffer["ice_flag"].astype(int) == 1]
    elif "lake_pos" in gdf_buffer.columns:
        gdf_buffer = gdf_buffer[gdf_buffer["lake_pos"].isin(TARGET_LAKE_POSITIONS)]
    if "ice_flag" in gdf_lake.columns:
        gdf_lake = gdf_lake[gdf_lake["ice_flag"].astype(int) == 1]
    elif "lake_pos" in gdf_lake.columns:
        gdf_lake = gdf_lake[gdf_lake["lake_pos"].isin(TARGET_LAKE_POSITIONS)]

    id_col = get_id_column(gdf_buffer)
    lake_ids = sorted(int(v) for v in gdf_buffer[id_col].unique() if _is_allowed_lake_id(v))
    if TARGET_LAKE_IDS:
        tset = set(int(x) for x in TARGET_LAKE_IDS)
        lake_ids = [x for x in lake_ids if x in tset]

    if USE_PARALLEL and MAX_WORKERS > 1:
        for batch in chunked(lake_ids, BATCH_SIZE):
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(batch, image_map)) as ex:
                futures = {ex.submit(process_single_lake, lid): lid for lid in batch}
                for fut in as_completed(futures):
                    _ = fut.result()
    else:
        init_worker(lake_ids, image_map)
        for lid in lake_ids:
            process_single_lake(lid)

    # collect summaries
    all_daily = []
    all_polluted = []
    for size_cat in TARGET_SIZE_CATEGORIES:
        size_dir = os.path.join(OUTPUT_BASE_DIR, size_cat)
        if not os.path.isdir(size_dir):
            continue
        for lake_id in os.listdir(size_dir):
            ldir = os.path.join(size_dir, lake_id)
            d1 = os.path.join(ldir, "Daily_Image_Classification_Summary.csv")
            d2 = os.path.join(ldir, "Polluted_Image_Details.csv")
            if os.path.exists(d1):
                all_daily.append(pd.read_csv(d1))
            if os.path.exists(d2):
                all_polluted.append(pd.read_csv(d2))
    if all_daily:
        pd.concat(all_daily, ignore_index=True).to_csv(
            os.path.join(OUTPUT_BASE_DIR, "Daily_Image_Classification_Summary_All.csv"), index=False, encoding="utf-8-sig"
        )
    if all_polluted:
        pd.concat(all_polluted, ignore_index=True).to_csv(
            os.path.join(OUTPUT_BASE_DIR, "Polluted_Image_Details_All.csv"), index=False, encoding="utf-8-sig"
        )

    print("done")


if __name__ == "__main__":
    main()



