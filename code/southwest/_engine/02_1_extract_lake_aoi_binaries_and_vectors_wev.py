"""
Stage 02_1: generate NDWI binary rasters and per-scene water vectors.

Default input:
X:/2024_西南水位曲线/result/01_Preprocessing_Results/03_NDWI_ice

Default output:
X:/2024_西南水位曲线/result/02_Lake_Extraction
  - 01_NDWI_binary_result
  - 02_Lake_vectors
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

os.environ["GDAL_SKIP"] = "DXF"
os.environ["CPL_LOG"] = "NUL"
os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"

INPUT_DIR = r"X:\2024_西南水位曲线\result1\01_Preprocessing_Results\03_NDWI_ice"
OUTPUT_BASE = r"X:\2024_西南水位曲线\result1\02_Lake_Extraction"

NDWI_THRESHOLD = 0.4
MAX_WORKERS = 4
USE_EXISTING_RESULTS = False


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 02_1 for WEV tile")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--ndwi-threshold", type=float, default=NDWI_THRESHOLD)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--use-existing-results", action="store_true")
    return parser.parse_args()


def save_raster(data, profile, output_path, dtype=rasterio.float32, nodata=None):
    profile.update(dtype=dtype, count=1, compress="lzw", nodata=nodata)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def vectorise_water(binary_mask, transform, crs, threshold):
    rows = (
        {"properties": {"raster_val": int(v), "threshold": float(threshold)}, "geometry": s}
        for s, v in shapes(binary_mask, mask=(binary_mask == 1), transform=transform)
    )
    rows = list(rows)
    if not rows:
        return None

    feats = []
    for r in rows:
        props = r["properties"]
        props["geometry"] = shape(r["geometry"])
        feats.append(props)
    return gpd.GeoDataFrame(feats, crs=crs)


def get_tile_id(_filename):
    return "Global"


def process_single_image(tif_path: str, output_dirs: Tuple[str, str], threshold: float):
    binary_out_dir, vector_out_dir = output_dirs
    filename = os.path.basename(tif_path)
    bin_path = os.path.join(binary_out_dir, filename.replace(".tif", "_binary.tif"))
    shp_path = os.path.join(vector_out_dir, filename.replace(".tif", "_lake.shp"))

    try:
        if os.path.exists(bin_path) and os.path.exists(shp_path):
            return get_tile_id(filename), bin_path, shp_path, filename

        with rasterio.open(tif_path) as src:
            ndwi = src.read(1).astype("float32")
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        if nodata is None:
            valid_mask = np.isfinite(ndwi)
        elif isinstance(nodata, float) and np.isnan(nodata):
            valid_mask = np.isfinite(ndwi)
        else:
            valid_mask = ndwi != nodata

        binary = np.zeros_like(ndwi, dtype=np.uint8)
        binary[valid_mask & (ndwi > threshold)] = 1
        save_raster(binary, profile.copy(), bin_path, dtype=rasterio.uint8, nodata=0)

        gdf = vectorise_water(binary, transform, crs, threshold)
        if gdf is not None and not gdf.empty:
            gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return get_tile_id(filename), bin_path, shp_path, filename
    except Exception as exc:
        print(f"  [error] Failed to process {filename}: {exc}")
        return None


def main():
    args = parse_args()
    binary_out_dir = os.path.join(args.output_base, "01_NDWI_binary_result")
    vector_out_dir = os.path.join(args.output_base, "02_Lake_vectors")
    os.makedirs(binary_out_dir, exist_ok=True)
    os.makedirs(vector_out_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    log_path = os.path.join(args.output_base, "processed_files_02_1.txt")

    processed_files = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            processed_files = {line.strip() for line in f if line.strip()}

    if args.use_existing_results:
        print("[02_1] USE_EXISTING_RESULTS=True, skip processing new tif files.")
        return

    tif_files = [f for f in tif_files if os.path.basename(f) not in processed_files]
    print(f"[02_1] pending tif files: {len(tif_files)}")

    def update_log(names):
        if not names:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            for n in names:
                f.write(n + "\n")

    current_batch = []
    output_dirs = (binary_out_dir, vector_out_dir)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_file = {
            ex.submit(process_single_image, tif_path, output_dirs, args.ndwi_threshold): tif_path
            for tif_path in tif_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            tif_path = future_to_file[future]
            filename = os.path.basename(tif_path)
            try:
                _ = future.result(timeout=600)
                current_batch.append(filename)
                if len(current_batch) >= 20:
                    update_log(current_batch)
                    current_batch = []
            except Exception as exc:
                print(f"  [error] failed future for {filename}: {exc}")

    if current_batch:
        update_log(current_batch)

    print(f"[02_1] done. outputs: {binary_out_dir} | {vector_out_dir}")


if __name__ == "__main__":
    main()
