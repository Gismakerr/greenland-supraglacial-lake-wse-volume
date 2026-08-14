"""
01_data_preprocessing.py

Refactored preprocessing pipeline for local S2/HLS-like multi-band GeoTIFF stacks.
Default project paths:
- Input:  X:/2024_西南水位曲线/data1/S2_22WEV_2024-06-01_to_2024-08-30
- Output: X:/2024_西南水位曲线/result1/01_Preprocessing_Results

Outputs:
1) 01_Filtered_Imagery
2) 02_NDSI
3) 03_NDWI_ice
4) 04_Daily_Max_NDWI
5) 05_Daily_Max_NDSI
6) 06_Cloud_Mask
7) 07_Rock_Shadow_Mask
8) 08_Shadow_Mask
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from tqdm import tqdm


# Sentinel-2-like 5-band stack order used in this project:
# 1=B2(Blue), 2=B3(Green), 3=B4(Red), 4=B8(NIR), 5=B11(SWIR)
BAND_BLUE = 1
BAND_GREEN = 2
BAND_RED = 3
BAND_SWIR = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess S2/HLS GeoTIFF stacks")
    parser.add_argument(
        "--input-dir",
        default=r"X:\2024_西南水位曲线\data1\S2_22WEV_2024-06-01_to_2024-08-30",
        help="Input GeoTIFF directory",
    )
    parser.add_argument(
        "--result-base",
        default=r"X:\2024_西南水位曲线\result1\01_Preprocessing_Results",
        help="Result base directory",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--calculate-daily-max", action="store_true")

    parser.add_argument("--filter-enable", action="store_true")
    parser.add_argument("--filter-invalid-threshold", type=float, default=0.5)
    parser.add_argument("--filter-start-date", default=None, help="YYYYMMDD")
    parser.add_argument("--filter-end-date", default=None, help="YYYYMMDD")

    parser.add_argument("--ndsi-threshold", type=float, default=0.8)
    parser.add_argument("--swir-threshold", type=float, default=0.1)

    return parser.parse_args()


def calculate_ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    denom = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = (green - swir) / denom
    ndsi[(denom == 0) | (ndsi >= 1) | (ndsi <= -1)] = np.nan
    return ndsi.astype(np.float32)


def calculate_ndwi_ice(blue: np.ndarray, red: np.ndarray) -> np.ndarray:
    denom = blue + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (blue - red) / denom
    ndwi[(denom == 0) | (ndwi >= 1) | (ndwi <= -1)] = np.nan
    return ndwi.astype(np.float32)


def get_sensing_time(filename: str) -> Optional[str]:
    # Supports MSIL2A_YYYYMMDDTHHMMSS_... style
    m = re.search(r"MSIL2A_(\d{8}T\d{6})_", filename)
    if m:
        return m.group(1)
    # Fallback: first full timestamp in file name
    m2 = re.search(r"(\d{8}T\d{6})", filename)
    if m2:
        return m2.group(1)
    return None


def date_from_filename(filename: str) -> Optional[str]:
    m = re.search(r"(\d{8})", filename)
    return m.group(1) if m else None


def analyze_image_quality(tif_path: str) -> Tuple[str, float, int]:
    try:
        with rasterio.open(tif_path) as src:
            nodata = 0 if src.nodata is None else src.nodata
            arr = src.read(1)
            total = arr.size
            if total == 0:
                return tif_path, 1.0, 0

            if isinstance(nodata, float) and np.isnan(nodata):
                invalid = int(np.isnan(arr).sum())
                valid = int((~np.isnan(arr)).sum())
            else:
                invalid = int((arr == nodata).sum())
                valid = int((arr != nodata).sum())

            return tif_path, invalid / total, valid
    except Exception:
        return tif_path, 1.0, 0


def copy_file(src: str, dst_dir: str, overwrite: bool) -> Optional[str]:
    import shutil

    dst = os.path.join(dst_dir, os.path.basename(src))
    try:
        if overwrite or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def filter_images(
    tif_files: List[str],
    out_filtered_dir: str,
    max_workers: int,
    overwrite: bool,
    invalid_threshold: float,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[str]:
    if not tif_files:
        return []

    s_date = datetime.strptime(start_date, "%Y%m%d") if start_date else None
    e_date = datetime.strptime(end_date, "%Y%m%d") if end_date else None

    date_filtered: List[str] = []
    for f in tif_files:
        d = date_from_filename(os.path.basename(f))
        if not d:
            continue
        try:
            fd = datetime.strptime(d, "%Y%m%d")
            if s_date and fd < s_date:
                continue
            if e_date and fd > e_date:
                continue
            date_filtered.append(f)
        except Exception:
            continue

    if not date_filtered:
        return []

    global_max_pixels = 0
    for f in date_filtered[:10]:
        try:
            with rasterio.open(f) as src:
                global_max_pixels = max(global_max_pixels, src.width * src.height)
        except Exception:
            continue

    file_stats: Dict[str, Tuple[float, int]] = {}
    valid_ratio_files: List[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(analyze_image_quality, f) for f in date_filtered]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Quality Check"):
            f_path, local_ratio, valid_pixels = fut.result()
            ratio = local_ratio
            if global_max_pixels > 0:
                ratio = max(0.0, 1.0 - (valid_pixels / global_max_pixels))
            file_stats[f_path] = (ratio, valid_pixels)
            if ratio <= invalid_threshold:
                valid_ratio_files.append(f_path)

    grouped = defaultdict(list)
    no_time = []
    for f in valid_ratio_files:
        t = get_sensing_time(os.path.basename(f))
        if t:
            grouped[t].append(f)
        else:
            no_time.append(f)

    selected = list(no_time)
    for _, items in grouped.items():
        if len(items) == 1:
            selected.append(items[0])
            continue
        best = max(items, key=lambda p: file_stats.get(p, (1.0, 0))[1])
        selected.append(best)

    copied: List[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(copy_file, f, out_filtered_dir, overwrite) for f in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Copying Files"):
            r = fut.result()
            if r:
                copied.append(r)

    list_path = os.path.join(out_filtered_dir, "valid_images_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in copied:
            f.write(f"{p}\n")

    return copied


def process_single_image_indices(
    tif_path: str,
    output_dirs: Dict[str, str],
    overwrite: bool,
    ndsi_threshold: float,
    swir_threshold: float,
) -> Optional[Tuple[str, str, str]]:
    filename = os.path.basename(tif_path)

    out_ndsi = os.path.join(output_dirs["ndsi"], filename.replace(".tif", "_NDSI.tif"))
    out_ndwi = os.path.join(output_dirs["ndwi"], filename.replace(".tif", "_NDWI_ice.tif"))
    out_cloud = os.path.join(output_dirs["cloud"], filename.replace(".tif", "_CloudMask.tif"))
    out_rock = os.path.join(output_dirs["rock"], filename.replace(".tif", "_RockShadowMask.tif"))
    out_shadow = os.path.join(output_dirs["shadow"], filename.replace(".tif", "_ShadowMask.tif"))

    if not overwrite and all(os.path.exists(p) for p in [out_ndsi, out_ndwi, out_cloud, out_rock, out_shadow]):
        return date_from_filename(filename), out_ndsi, out_ndwi

    try:
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()

            profile_f32 = profile.copy()
            profile_f32.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)

            profile_u8 = profile.copy()
            profile_u8.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)

            blue = src.read(BAND_BLUE).astype(np.float32)
            green = src.read(BAND_GREEN).astype(np.float32)
            red = src.read(BAND_RED).astype(np.float32)
            swir = src.read(BAND_SWIR).astype(np.float32)

            ndsi = calculate_ndsi(green, swir)
            ndwi = calculate_ndwi_ice(blue, red)

            blue_ref = blue / 10000.0
            swir_ref = swir / 10000.0
            valid_data = (blue > 0) & (green > 0) & (red > 0) & (swir > 0)

            rock_shadow = np.zeros_like(ndsi, dtype=np.uint8)
            rock_shadow[(~np.isnan(ndsi)) & (~np.isnan(blue_ref)) & valid_data & (ndsi < 0.85) & (blue_ref < 0.4)] = 1

            shadow = np.zeros_like(ndsi, dtype=np.uint8)
            shadow[(~np.isnan(blue_ref)) & valid_data & (blue_ref < 0.4)] = 1

            cloud = np.zeros_like(ndsi, dtype=np.uint8)
            cloud[(~np.isnan(ndsi)) & (~np.isnan(swir_ref)) & valid_data & (ndsi < ndsi_threshold) & (swir_ref > swir_threshold)] = 1

            with rasterio.open(out_rock, "w", **profile_u8) as dst:
                dst.write(rock_shadow, 1)
            with rasterio.open(out_shadow, "w", **profile_u8) as dst:
                dst.write(shadow, 1)
            with rasterio.open(out_cloud, "w", **profile_u8) as dst:
                dst.write(cloud, 1)
            with rasterio.open(out_ndsi, "w", **profile_f32) as dst:
                dst.write(ndsi, 1)
            with rasterio.open(out_ndwi, "w", **profile_f32) as dst:
                dst.write(ndwi, 1)

            return date_from_filename(filename), out_ndsi, out_ndwi

    except Exception:
        return None


def process_daily_max(date_str: str, input_files: List[str], output_dir: str, suffix: str, overwrite: bool) -> Optional[str]:
    if not input_files:
        return None

    out_path = os.path.join(output_dir, f"{suffix}_MAX_{date_str}.tif")
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    src_files = []
    try:
        for f in input_files:
            src_files.append(rasterio.open(f))

        mosaic, out_trans = merge(src_files, method="max")
        meta = src_files[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_trans,
            dtype=rasterio.float32,
            compress="lzw",
        )
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
        return out_path
    except Exception:
        return None
    finally:
        for s in src_files:
            try:
                s.close()
            except Exception:
                pass


def ensure_dirs(result_base: str) -> Dict[str, str]:
    out = {
        "filtered": os.path.join(result_base, "01_Filtered_Imagery"),
        "ndsi": os.path.join(result_base, "02_NDSI"),
        "ndwi": os.path.join(result_base, "03_NDWI_ice"),
        "max_ndwi": os.path.join(result_base, "04_Daily_Max_NDWI"),
        "max_ndsi": os.path.join(result_base, "05_Daily_Max_NDSI"),
        "cloud": os.path.join(result_base, "06_Cloud_Mask"),
        "rock": os.path.join(result_base, "07_Rock_Shadow_Mask"),
        "shadow": os.path.join(result_base, "08_Shadow_Mask"),
    }
    for d in out.values():
        os.makedirs(d, exist_ok=True)
    return out


def write_run_summary(summary_csv: str, rows: List[Tuple[str, str]]) -> None:
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, v])


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    result_base = args.result_base
    out_dirs = ensure_dirs(result_base)

    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No tif found in: {input_dir}")

    source_files = tif_files
    if args.filter_enable:
        source_files = filter_images(
            tif_files=tif_files,
            out_filtered_dir=out_dirs["filtered"],
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            invalid_threshold=args.filter_invalid_threshold,
            start_date=args.filter_start_date,
            end_date=args.filter_end_date,
        )
        if not source_files:
            raise RuntimeError("No valid images after filtering.")

    ndsi_by_date: Dict[str, List[str]] = defaultdict(list)
    ndwi_by_date: Dict[str, List[str]] = defaultdict(list)

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                process_single_image_indices,
                f,
                out_dirs,
                args.overwrite,
                args.ndsi_threshold,
                args.swir_threshold,
            )
            for f in source_files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Index Processing"):
            res = fut.result()
            if not res:
                continue
            date_str, ndsi_path, ndwi_path = res
            if not date_str:
                continue
            ndsi_by_date[date_str].append(ndsi_path)
            ndwi_by_date[date_str].append(ndwi_path)

    if args.calculate_daily_max:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(process_daily_max, d, files, out_dirs["max_ndwi"], "NDWI_ice", args.overwrite)
                for d, files in ndwi_by_date.items()
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Daily Max NDWI"):
                pass

        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(process_daily_max, d, files, out_dirs["max_ndsi"], "NDSI", args.overwrite)
                for d, files in ndsi_by_date.items()
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Daily Max NDSI"):
                pass

    summary_rows = [
        ("input_dir", input_dir),
        ("result_base", result_base),
        ("input_tif_count", str(len(tif_files))),
        ("processed_tif_count", str(len(source_files))),
        ("unique_ndsi_days", str(len(ndsi_by_date))),
        ("unique_ndwi_days", str(len(ndwi_by_date))),
        ("calculate_daily_max", str(args.calculate_daily_max)),
    ]
    write_run_summary(os.path.join(result_base, "run_summary.csv"), summary_rows)

    print("Preprocessing finished.")
    print(f"Result base: {result_base}")


if __name__ == "__main__":
    main()
