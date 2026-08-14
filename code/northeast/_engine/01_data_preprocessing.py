"""
============================================================
脚本名称: 01_data_preprocessing.py
功能说明: Sentinel-2 预处理，计算 NDSI/NDWI-ice，并生成每日最大值合成。
============================================================
"""

import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

# --- 参数配置 ---
# 项目根目录
BASE_LOCAL = r"Z:\冰面湖水量测算_Local"

# 输入影像目录（整理后的原始数据）
INPUT_DIR = os.path.join(BASE_LOCAL, "冰面湖水位论文代码数据整理", "result", "01_Preprocessing_Results", "01_S2_Imgagery_Pro")

# 主输出目录
OUTPUT_BASE = os.path.join(BASE_LOCAL, "冰面湖水位论文代码数据整理", "result", "01_Preprocessing_Results")
OUTPUT_DIR_FILTERED = os.path.join(OUTPUT_BASE, "01_S2_Imgagery_Pro")
OUTPUT_DIR_NDSI = os.path.join(OUTPUT_BASE, "02_NDSI")
OUTPUT_DIR_NDWI = os.path.join(OUTPUT_BASE, "03_NDWI_ice")
OUTPUT_DIR_MAX_NDWI = os.path.join(OUTPUT_BASE, "04_Daily_Max_NDWI")
OUTPUT_DIR_MAX_NDSI = os.path.join(OUTPUT_BASE, "05_Daily_Max_NDSI")

# 掩膜输出目录（保持原流程）
OUTPUT_BASE_MASK = OUTPUT_BASE
OUTPUT_DIR_CLOUD_MASK = os.path.join(OUTPUT_BASE_MASK, "06_Cloud_Mask_Result")
OUTPUT_DIR_ROCK_SHADOW_MASK = os.path.join(OUTPUT_BASE_MASK, "07_Rock_Shadow_Mask")
OUTPUT_DIR_SHADOW_MASK = os.path.join(OUTPUT_BASE_MASK, "08_Shadow_Mask")

# 并行进程数
MAX_WORKERS = 4

# 处理开关
CALCULATE_INDICES = True
CALCULATE_DAILY_MAX = True
OVERWRITE = True

# 影像筛选参数（仅处理 6 月 15 日到 8 月 15 日）
FILTER_ENABLE = True
FILTER_INVALID_THRESHOLD = 0.5
FILTER_START_DATE = "20240615"  # YYYYMMDD
FILTER_END_DATE = "20240815"  # YYYYMMDD

# Sentinel-2 波段设置
# Blue: B1, Green: B2, Red: B3, SWIR: B5
BAND_BLUE = 1
BAND_GREEN = 2
BAND_RED = 3
BAND_SWIR = 5

# 云掩膜参数
CALCULATE_CLOUD_MASK = True
NDSI_THRESHOLD = 0.8
SWIR_THRESHOLD = 0.1


def calculate_ndsi(green, swir):
    """NDSI = (Green - SWIR) / (Green + SWIR)"""
    denominator = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = (green - swir) / denominator
        ndsi[denominator == 0] = np.nan
        ndsi[np.isnan(ndsi)] = np.nan
        ndsi[ndsi >= 1] = np.nan
        ndsi[ndsi <= -1] = np.nan
    return ndsi.astype(np.float32)


def calculate_ndwi_ice(blue, red):
    """NDWI-ice = (Blue - Red) / (Blue + Red)"""
    denominator = blue + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (blue - red) / denominator
        ndwi[denominator == 0] = np.nan
        ndwi[np.isnan(ndwi)] = np.nan
        ndwi[ndwi >= 1] = np.nan
        ndwi[ndwi <= -1] = np.nan
    return ndwi.astype(np.float32)


def get_sensing_time(filename):
    """从文件名提取感知时间，用于去重。"""
    match = re.search(r"MSIL2A_(\d{8}T\d{6})_", filename)
    if match:
        return match.group(1)
    return None


def analyze_image_quality(tif_path):
    """返回 (tif_path, invalid_ratio, valid_pixels)。"""
    try:
        with rasterio.open(tif_path) as src:
            nodata = src.nodata if src.nodata is not None else 0
            data = src.read(1)
            total_pixels = data.size
            if total_pixels == 0:
                return tif_path, 1.0, 0

            if np.isnan(nodata):
                invalid_pixels = np.sum(np.isnan(data))
                valid_pixels = np.sum(~np.isnan(data))
            else:
                invalid_pixels = np.sum(data == nodata)
                valid_pixels = np.sum(data != nodata)

            ratio = invalid_pixels / total_pixels
            return tif_path, ratio, valid_pixels
    except Exception as exc:
        print(f"[错误] 读取失败 {os.path.basename(tif_path)}: {exc}")
        return tif_path, 1.0, 0


def copy_file(src, dst_dir, overwrite=False):
    """复制文件到目标目录。"""
    import shutil

    try:
        fname = os.path.basename(src)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst) or overwrite:
            shutil.copy2(src, dst)
        return dst
    except Exception as exc:
        print(f"[错误] 复制失败 {src}: {exc}")
        return None


def filter_images(tif_files):
    """执行筛选：日期 -> 无效比例 -> 重复时相去重。"""
    print(f"\n>>> 开始影像筛选（总数: {len(tif_files)}）...")

    # 1) 日期筛选
    date_filtered = []
    s_date = datetime.strptime(FILTER_START_DATE, "%Y%m%d") if FILTER_START_DATE else None
    e_date = datetime.strptime(FILTER_END_DATE, "%Y%m%d") if FILTER_END_DATE else None

    if s_date or e_date:
        print(f"  应用日期筛选: {FILTER_START_DATE} - {FILTER_END_DATE}")
        for f in tif_files:
            d_str = date_from_filename(os.path.basename(f))
            if not d_str:
                continue
            try:
                f_date = datetime.strptime(d_str, "%Y%m%d")
                if s_date and f_date < s_date:
                    continue
                if e_date and f_date > e_date:
                    continue
                date_filtered.append(f)
            except Exception:
                continue
    else:
        date_filtered = tif_files

    print(f"  日期筛选后数量: {len(date_filtered)}")
    if not date_filtered:
        return []

    # 2) 无效比例筛选
    print("  正在计算无效像元比例...")
    global_max_pixels = 0
    sample_files = date_filtered[:10]
    print(f"  使用前 {len(sample_files)} 景估算全局最大像元数...")

    for f in sample_files:
        try:
            with rasterio.open(f) as src:
                pixels = src.width * src.height
                if pixels > global_max_pixels:
                    global_max_pixels = pixels
        except Exception as exc:
            print(f"  [警告] 样本读取失败 {os.path.basename(f)}: {exc}")

    if global_max_pixels > 0:
        print(f"  全局最大像元数: {global_max_pixels}")
    else:
        print("  [警告] 无法估算全局最大像元数，将回退到单景无效率。")

    valid_ratio_files = []
    file_stats = {}  # path -> (ratio, valid_pixels)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_image_quality, f): f for f in date_filtered}
        for future in tqdm(as_completed(futures), total=len(date_filtered), desc="Quality Check"):
            f_path, local_ratio, valid_pixels = future.result()

            if global_max_pixels > 0:
                valid_ratio_global = valid_pixels / global_max_pixels
                ratio = 1.0 - valid_ratio_global
                if ratio < 0:
                    ratio = 0.0
            else:
                ratio = local_ratio

            file_stats[f_path] = (ratio, valid_pixels)
            if ratio <= FILTER_INVALID_THRESHOLD:
                valid_ratio_files.append(f_path)

    print(f"  无效率筛选后数量: {len(valid_ratio_files)}")
    if not valid_ratio_files:
        return []

    # 3) 重复时相去重
    print("  正在进行重复时相去重...")
    final_files = []
    files_by_time = defaultdict(list)

    for f in valid_ratio_files:
        t = get_sensing_time(os.path.basename(f))
        if t:
            files_by_time[t].append(f)
        else:
            final_files.append(f)

    for _, files in files_by_time.items():
        if len(files) == 1:
            final_files.append(files[0])
        else:
            best_file = None
            max_valid = -1
            for f in files:
                _, valid_pixels = file_stats.get(f, (1.0, 0))
                if valid_pixels > max_valid:
                    max_valid = valid_pixels
                    best_file = f
            if best_file:
                final_files.append(best_file)

    print(f"  筛选完成，最终保留: {len(final_files)}\n")

    # 复制筛选后影像（若输入输出同目录则跳过复制）
    input_dir_norm = os.path.normcase(os.path.abspath(INPUT_DIR))
    output_dir_norm = os.path.normcase(os.path.abspath(OUTPUT_DIR_FILTERED))
    copied_files = []
    if input_dir_norm == output_dir_norm:
        print("  输入目录与输出目录相同，跳过复制步骤。")
        copied_files = list(final_files)
    else:
        print(f"  正在复制筛选后影像到: {OUTPUT_DIR_FILTERED} ...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(copy_file, f, OUTPUT_DIR_FILTERED, OVERWRITE): f for f in final_files}
            for future in tqdm(as_completed(futures), total=len(final_files), desc="Copying Files"):
                res = future.result()
                if res:
                    copied_files.append(res)

    list_path = os.path.join(OUTPUT_DIR_FILTERED, "valid_images_list.txt")
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for item in copied_files:
                f.write(f"{item}\n")
    except Exception:
        pass

    return copied_files


def process_single_image_indices(tif_path):
    """处理单景影像，计算 NDSI/NDWI-ice/CloudMask/RockShadow/Shadow。"""
    filename = os.path.basename(tif_path)

    out_path_ndsi = os.path.join(OUTPUT_DIR_NDSI, filename.replace(".tif", "_NDSI.tif"))
    out_path_ndwi = os.path.join(OUTPUT_DIR_NDWI, filename.replace(".tif", "_NDWI_ice.tif"))
    out_path_cloud = os.path.join(OUTPUT_DIR_CLOUD_MASK, filename.replace(".tif", "_CloudMask.tif"))
    out_path_rock = os.path.join(OUTPUT_DIR_ROCK_SHADOW_MASK, filename.replace(".tif", "_RockShadowMask.tif"))
    out_path_shadow = os.path.join(OUTPUT_DIR_SHADOW_MASK, filename.replace(".tif", "_ShadowMask.tif"))

    if (
        not OVERWRITE
        and os.path.exists(out_path_ndsi)
        and os.path.exists(out_path_ndwi)
        and os.path.exists(out_path_cloud)
        and os.path.exists(out_path_rock)
        and os.path.exists(out_path_shadow)
    ):
        return date_from_filename(filename), out_path_ndsi, out_path_ndwi

    try:
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            profile_float = profile.copy()
            profile_float.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)

            profile_byte = profile.copy()
            profile_byte.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)

            green = src.read(BAND_GREEN).astype("float32")
            swir = src.read(BAND_SWIR).astype("float32")
            blue = src.read(BAND_BLUE).astype("float32")
            red = src.read(BAND_RED).astype("float32")

            ndsi = calculate_ndsi(green, swir)
            ndwi = calculate_ndwi_ice(blue, red)

            blue_reflectance = blue / 10000.0
            swir_reflectance = swir / 10000.0
            valid_data_mask = (blue > 0) & (green > 0) & (red > 0) & (swir > 0)

            rock_shadow_mask = np.zeros_like(ndsi, dtype=np.uint8)
            valid_mask_rs = (~np.isnan(ndsi)) & (~np.isnan(blue_reflectance)) & valid_data_mask
            condition_rs = (ndsi < 0.85) & (blue_reflectance < 0.4)
            rock_shadow_mask[valid_mask_rs & condition_rs] = 1
            with rasterio.open(out_path_rock, "w", **profile_byte) as dst:
                dst.write(rock_shadow_mask, 1)

            shadow_mask = np.zeros_like(ndsi, dtype=np.uint8)
            valid_mask_s = (~np.isnan(blue_reflectance)) & valid_data_mask
            condition_s = blue_reflectance < 0.4
            shadow_mask[valid_mask_s & condition_s] = 1
            with rasterio.open(out_path_shadow, "w", **profile_byte) as dst:
                dst.write(shadow_mask, 1)

            cloud_mask = np.zeros_like(ndsi, dtype=np.uint8)
            if CALCULATE_CLOUD_MASK:
                valid_mask = (~np.isnan(ndsi)) & (~np.isnan(swir_reflectance)) & valid_data_mask
                condition = (ndsi < NDSI_THRESHOLD) & (swir_reflectance > SWIR_THRESHOLD)
                cloud_mask[valid_mask & condition] = 1
                with rasterio.open(out_path_cloud, "w", **profile_byte) as dst:
                    dst.write(cloud_mask, 1)

            if CALCULATE_INDICES:
                with rasterio.open(out_path_ndsi, "w", **profile_float) as dst:
                    dst.write(ndsi, 1)
                with rasterio.open(out_path_ndwi, "w", **profile_float) as dst:
                    dst.write(ndwi, 1)

            del green, swir, blue, red, ndsi, ndwi, cloud_mask, swir_reflectance
            return date_from_filename(filename), out_path_ndsi, out_path_ndwi

    except Exception as exc:
        print(f"[错误] 处理失败 {filename}: {exc}")
        return None


def date_from_filename(filename):
    match = re.search(r"(\d{8})", filename)
    if match:
        return match.group(1)
    return None


def process_daily_max(date_str, input_files, output_dir, suffix):
    """计算每日最大值合成（NDWI 或 NDSI）。"""
    if not input_files:
        return None

    out_filename = f"{suffix}_MAX_{date_str}.tif"
    out_path = os.path.join(output_dir, out_filename)

    if not OVERWRITE and os.path.exists(out_path):
        return out_path

    try:
        src_files = [rasterio.open(f) for f in input_files]
        mosaic, out_trans = merge(src_files, method="max")

        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "dtype": rasterio.float32,
                "compress": "lzw",
            }
        )

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files:
            src.close()

        return out_path
    except Exception as exc:
        print(f"[错误] 合成失败 {date_str} ({suffix}): {exc}")
        return None


def main():
    print("开始数据预处理流程（NDSI / NDWI-ice / Daily Max）...")

    for d in [
        OUTPUT_DIR_FILTERED,
        OUTPUT_DIR_NDSI,
        OUTPUT_DIR_NDWI,
        OUTPUT_DIR_MAX_NDWI,
        OUTPUT_DIR_MAX_NDSI,
        OUTPUT_DIR_CLOUD_MASK,
        OUTPUT_DIR_ROCK_SHADOW_MASK,
        OUTPUT_DIR_SHADOW_MASK,
    ]:
        os.makedirs(d, exist_ok=True)

    tif_files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    print(f"找到 {len(tif_files)} 个原始影像文件。")

    if FILTER_ENABLE:
        tif_files = filter_images(tif_files)
        if not tif_files:
            print("没有符合条件的影像，程序结束。")
            return

    ndwi_files_by_date = {}
    ndsi_files_by_date = {}

    if CALCULATE_INDICES:
        print(">>> 阶段 1：计算 NDSI / NDWI-ice / Cloud Mask ...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_image_indices, f): f for f in tif_files}
            for future in tqdm(as_completed(futures), total=len(tif_files), desc="Calculating Indices"):
                res = future.result()
                if not res:
                    continue
                date_str, ndsi_path, ndwi_path = res
                if not date_str:
                    continue

                if ndwi_path:
                    ndwi_files_by_date.setdefault(date_str, []).append(ndwi_path)
                if ndsi_path:
                    ndsi_files_by_date.setdefault(date_str, []).append(ndsi_path)
    else:
        print(">>> 跳过阶段 1，扫描已有指数结果...")
        existing_ndwi = glob.glob(os.path.join(OUTPUT_DIR_NDWI, "*_NDWI_ice.tif"))
        for f in existing_ndwi:
            date_str = date_from_filename(os.path.basename(f))
            if date_str:
                ndwi_files_by_date.setdefault(date_str, []).append(f)

        existing_ndsi = glob.glob(os.path.join(OUTPUT_DIR_NDSI, "*_NDSI.tif"))
        for f in existing_ndsi:
            date_str = date_from_filename(os.path.basename(f))
            if date_str:
                ndsi_files_by_date.setdefault(date_str, []).append(f)

    if CALCULATE_DAILY_MAX:
        print("\n>>> 阶段 2：计算每日最大值合成...")

        print(f"  正在计算 NDWI Max（{len(ndwi_files_by_date)} 天）...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_daily_max, date, files, OUTPUT_DIR_MAX_NDWI, "NDWI_ice"): date
                for date, files in ndwi_files_by_date.items()
            }
            for _ in tqdm(as_completed(futures), total=len(futures), desc="NDWI Max"):
                pass

        print(f"  正在计算 NDSI Max（{len(ndsi_files_by_date)} 天）...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_daily_max, date, files, OUTPUT_DIR_MAX_NDSI, "NDSI"): date
                for date, files in ndsi_files_by_date.items()
            }
            for _ in tqdm(as_completed(futures), total=len(futures), desc="NDSI Max"):
                pass

    print("\n预处理完成。")
    print(f"输出根目录: {OUTPUT_BASE}")
    print(f"  1. Filtered List: {OUTPUT_DIR_FILTERED}")
    print(f"  2. NDSI: {OUTPUT_DIR_NDSI}")
    print(f"  3. NDWI: {OUTPUT_DIR_NDWI}")
    print(f"  4. Max NDWI: {OUTPUT_DIR_MAX_NDWI}")
    print(f"  5. Max NDSI: {OUTPUT_DIR_MAX_NDSI}")
    print(f"  6. Cloud Mask: {OUTPUT_DIR_CLOUD_MASK}")
    print(f"  7. Rock/Shadow Mask: {OUTPUT_DIR_ROCK_SHADOW_MASK}")
    print(f"  8. Shadow Mask: {OUTPUT_DIR_SHADOW_MASK}")


if __name__ == "__main__":
    main()
