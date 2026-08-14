"""
Stage 02_1: generate NDWI binary rasters and per-scene lake vectors.

Outputs:
- result/02_Lake_Extraction/01_NDWI_binary_result
- result/02_Lake_Extraction/02_Lake_vectors
"""

import concurrent.futures
import glob
import os
import re

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, shape

os.environ["GDAL_SKIP"] = "DXF"
os.environ["CPL_LOG"] = "NUL"
os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"

INPUT_DIR = r"Z:\冰面湖水量测算_Local\冰面湖水位论文代码数据整理\result\01_Preprocessing_Results\04_Daily_Max_NDWI"
OUTPUT_BASE = r"Z:\冰面湖水量测算_Local\冰面湖水位论文代码数据整理\result\02_Lake_Extraction"

NDWI_THRESHOLD = 0.4
MAX_WORKERS = 4
USE_EXISTING_RESULTS = False
START_MMDD = (6, 15)
END_MMDD = (8, 15)


def save_raster(data, profile, output_path, dtype=rasterio.float32, nodata=None):
    profile.update(dtype=dtype, count=1, compress="lzw", nodata=nodata)
    with rasterio.open(output_path, "w", **profile) as dst:
        if len(data.shape) == 2:
            dst.write(data.astype(dtype), 1)
        else:
            dst.write(data.astype(dtype))


def vectorise_water(binary_mask, transform, crs, threshold):
    results = (
        {"properties": {"raster_val": int(v), "threshold": float(threshold)}, "geometry": s}
        for s, v in shapes(binary_mask, mask=(binary_mask == 1), transform=transform)
    )
    geoms = list(results)
    if not geoms:
        return None

    data_list = []
    for item in geoms:
        geom = shape(item["geometry"])
        props = item["properties"]
        props["geometry"] = geom
        data_list.append(props)
    return gpd.GeoDataFrame(data_list, crs=crs)


def get_tile_id(_filename):
    return "Global"


def in_target_window(filename):
    match = re.search(r"(\d{8})", filename)
    if not match:
        return False
    date_str = match.group(1)
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    mmdd = (month, day)
    return START_MMDD <= mmdd <= END_MMDD


def process_single_image(tif_path, output_dirs, threshold):
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

        valid_mask = (ndwi != nodata) if nodata is not None else (ndwi != -9999)
        binary = np.zeros_like(ndwi, dtype=np.uint8)
        binary[valid_mask & (ndwi > threshold)] = 1
        save_raster(binary, profile.copy(), bin_path, dtype=rasterio.uint8, nodata=None)

        gdf = vectorise_water(binary, transform, crs, threshold)
        if gdf is not None and not gdf.empty:
            gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return get_tile_id(filename), bin_path, shp_path, filename
    except Exception as exc:
        print(f"  [error] Failed to process {filename}: {exc}")
        return None


def main():
    binary_out_dir = os.path.join(OUTPUT_BASE, "01_NDWI_binary_result")
    vector_out_dir = os.path.join(OUTPUT_BASE, "02_Lake_vectors")
    os.makedirs(binary_out_dir, exist_ok=True)
    os.makedirs(vector_out_dir, exist_ok=True)

    output_dirs = (binary_out_dir, vector_out_dir)
    tif_files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    tif_files = [f for f in tif_files if in_target_window(os.path.basename(f))]
    log_path = os.path.join(OUTPUT_BASE, "processed_files.txt")

    processed_files = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f if line.strip())

    if USE_EXISTING_RESULTS:
        print("[02_1] USE_EXISTING_RESULTS=True, skip processing new tif files.")
        return

    tif_files = [f for f in tif_files if os.path.basename(f) not in processed_files]
    print(f"[02_1] pending tif files: {len(tif_files)}")

    def update_log(file_list):
        if not file_list:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            for fname in file_list:
                f.write(fname + "\n")

    current_batch_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_image, tif_path, output_dirs, NDWI_THRESHOLD): tif_path
            for tif_path in tif_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            tif_path = future_to_file[future]
            filename = os.path.basename(tif_path)
            try:
                result = future.result(timeout=600)
                current_batch_files.append(filename)
                if result and len(current_batch_files) >= 20:
                    update_log(current_batch_files)
                    current_batch_files = []
            except Exception as exc:
                print(f"  [error] failed future for {filename}: {exc}")
    if current_batch_files:
        update_log(current_batch_files)

    print(f"[02_1] done. outputs: {binary_out_dir} | {vector_out_dir}")


if __name__ == "__main__":
    main()


