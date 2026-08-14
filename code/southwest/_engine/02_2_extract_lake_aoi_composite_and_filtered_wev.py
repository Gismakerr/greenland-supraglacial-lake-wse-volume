"""
Stage 02_2: build water composites and filtered lake AOIs from Stage 02_1 binaries.

Default input:
X:/2024_西南水位曲线/result/02_Lake_Extraction/01_NDWI_binary_result

Default output:
X:/2024_西南水位曲线/result/02_Lake_Extraction
  - 03_Water_Composite
  - 04_Filtered_Lakes
"""

from __future__ import annotations

import argparse
import glob
import os
import traceback

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.features import rasterize, shapes
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import box
from shapely.geometry import MultiPolygon, Polygon, shape

os.environ["GDAL_SKIP"] = "DXF"
os.environ["CPL_LOG"] = "NUL"
os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"

OUTPUT_BASE = r"X:\2024_西南水位曲线\result1\02_Lake_Extraction"

MIN_AREA_THRESHOLD = 10000
MEDIUM_LAKE_THRESHOLD = 62500
LARGE_LAKE_THRESHOLD = 625000

BUFFER_MODE = "dynamic"
BUFFER_DISTANCE = 300
BUFFER_DYNAMIC_MULTIPLIER = 1
BUFFER_DYNAMIC_MAX_RADIUS = 300

ICE_SURFACE_MASK_PATH = r"D:\Haoyu_space\03_data\Greenland_IceSheet_IceMap_Merge\Greenland_IceSheet_Main.shp"
DEM_TIF = r"D:\Haoyu_space\03_data\ArcticDEM\30m\DEM.tif"

SKIP_COMPOSITE = False
FINAL_OUT_SUBDIR = "04_Filtered_Lakes"
FILTER_ONLY_ICE_SURFACE_LAKES = False


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 02_2 for WEV tile")
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--ice-mask-path", default=ICE_SURFACE_MASK_PATH)
    parser.add_argument("--dem-tif", default=DEM_TIF)
    parser.add_argument("--skip-composite", action="store_true")
    parser.add_argument("--final-out-subdir", default=FINAL_OUT_SUBDIR)
    parser.add_argument("--filter-only-ice-surface-lakes", action="store_true")
    return parser.parse_args()


def fill_holes(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(part.exterior) for part in geom.geoms])
    return geom


def save_raster(data, profile, output_path, dtype=rasterio.float32, nodata=None):
    profile.update(dtype=dtype, count=1, compress="lzw", nodata=nodata)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def clip_ice_mask_to_tile_aoi(ice_gdf: gpd.GeoDataFrame, tile_bounds, target_crs):
    """Pre-clip large ice mask by tile AOI to speed up spatial predicates."""
    if ice_gdf is None or ice_gdf.empty:
        return ice_gdf
    if ice_gdf.crs != target_crs:
        ice_gdf = ice_gdf.to_crs(target_crs)
    tile_aoi = gpd.GeoDataFrame(geometry=[box(*tile_bounds)], crs=target_crs)
    clipped = gpd.overlay(ice_gdf, tile_aoi, how="intersection")
    return clipped


def add_lake_elevation_field(gdf, dem_tif, lake_id_col="lake_id", output_field="elev_m"):
    if gdf is None or gdf.empty or lake_id_col not in gdf.columns:
        return gdf
    if not os.path.exists(dem_tif):
        print(f"  [warning] DEM not found: {dem_tif}")
        return gdf

    gdf = gdf.copy()
    gdf[output_field] = np.nan
    try:
        with rasterio.open(dem_tif) as src_dem:
            work_gdf = gdf.to_crs(src_dem.crs) if gdf.crs != src_dem.crs else gdf.copy()
            for idx, row in work_gdf.iterrows():
                try:
                    out_image, _ = rasterio.mask.mask(src_dem, [row.geometry], crop=True)
                    vals = out_image.astype(float)
                    if src_dem.nodata is not None:
                        vals = vals[vals != src_dem.nodata]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        gdf.at[idx, output_field] = float(np.nanmax(vals))
                except Exception:
                    continue
    except Exception as exc:
        print(f"  [warning] DEM extraction failed: {exc}")
    return gdf


def get_tile_id(_filename):
    return "Global"


def merge_and_clean(file_list, output_name, composite_out_dir):
    if not file_list:
        return None
    try:
        src_files = [rasterio.open(fp) for fp in file_list]
        mosaic, out_trans = merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": 1,
                "dtype": rasterio.float32,
                "compress": "lzw",
            }
        )
        out_path = os.path.join(composite_out_dir, output_name)
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        for fp in file_list:
            if os.path.exists(fp):
                os.remove(fp)
        return out_path
    except Exception as exc:
        print(f"  [error] Merge failed: {exc}")
        traceback.print_exc()
        return None


def main():
    args = parse_args()
    binary_out_dir = os.path.join(args.output_base, "01_NDWI_binary_result")
    composite_out_dir = os.path.join(args.output_base, "03_Water_Composite")
    final_out_dir = os.path.join(args.output_base, args.final_out_subdir)
    os.makedirs(composite_out_dir, exist_ok=True)
    os.makedirs(final_out_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(binary_out_dir, "*_binary.tif")))
    if not bin_files:
        print("[02_2] no binary tif files found in 01_NDWI_binary_result.")
        return

    tile_binary_files = {}
    for bin_path in bin_files:
        tile_binary_files.setdefault(get_tile_id(os.path.basename(bin_path)), []).append(bin_path)

    max_files_to_mosaic = []
    sum_files_to_mosaic = []
    if not args.skip_composite:
        from rasterio.transform import from_bounds

        for tile_id, tile_bin_files in tile_binary_files.items():
            if not tile_bin_files:
                continue
            try:
                min_x, min_y = float("inf"), float("inf")
                max_x, max_y = float("-inf"), float("-inf")
                first_crs = None
                first_res = None
                for bf in tile_bin_files:
                    with rasterio.open(bf) as src:
                        b = src.bounds
                        min_x = min(min_x, b.left)
                        min_y = min(min_y, b.bottom)
                        max_x = max(max_x, b.right)
                        max_y = max(max_y, b.top)
                        if first_crs is None:
                            first_crs = src.crs
                            first_res = src.res

                width = int(np.ceil((max_x - min_x) / first_res[0]))
                height = int(np.ceil((max_y - min_y) / first_res[1]))
                unified_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
                max_arr = np.zeros((height, width), dtype=np.uint8)
                sum_arr = np.zeros((height, width), dtype=np.uint16)

                for bf in tile_bin_files:
                    with rasterio.open(bf) as src:
                        aligned_data = np.zeros((height, width), dtype=np.uint8)
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=aligned_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=unified_transform,
                            dst_crs=first_crs,
                            resampling=Resampling.nearest,
                        )
                        max_arr = np.maximum(max_arr, aligned_data)
                        sum_arr += aligned_data

                profile = {
                    "driver": "GTiff",
                    "dtype": rasterio.uint8,
                    "count": 1,
                    "crs": first_crs,
                    "transform": unified_transform,
                    "width": width,
                    "height": height,
                    "compress": "lzw",
                    "nodata": 0,
                }
                max_path = os.path.join(composite_out_dir, f"Temp_Max_{tile_id}.tif")
                sum_path = os.path.join(composite_out_dir, f"Temp_Sum_{tile_id}.tif")
                save_raster(max_arr, profile, max_path, dtype=rasterio.uint8, nodata=0)
                save_raster(sum_arr, profile, sum_path, dtype=rasterio.uint16, nodata=0)
                max_files_to_mosaic.append(max_path)
                sum_files_to_mosaic.append(sum_path)
            except Exception as exc:
                print(f"  [error] Composite failed for {tile_id}: {exc}")

    water_max_path = os.path.join(composite_out_dir, "Water_Max_Composite.tif")
    water_sum_path = os.path.join(composite_out_dir, "Water_Sum_Composite.tif")
    if not args.skip_composite:
        water_max_path = merge_and_clean(max_files_to_mosaic, "Water_Max_Composite.tif", composite_out_dir)
        water_sum_path = merge_and_clean(sum_files_to_mosaic, "Water_Sum_Composite.tif", composite_out_dir)

    if not water_max_path or not os.path.exists(water_max_path):
        print("[02_2] No Water_Max_Composite.tif found.")
        return

    try:
        from scipy.ndimage import binary_closing, binary_opening

        with rasterio.open(water_max_path) as src:
            max_data = src.read(1)
            max_transform = src.transform
            max_crs = src.crs

        struct = np.ones((3, 3), dtype=bool)
        max_data = binary_closing(binary_opening(max_data == 1, structure=struct), structure=struct).astype(np.uint8)

        geoms = list(
            {"properties": {"raster_val": int(v)}, "geometry": s}
            for s, v in shapes(max_data, mask=(max_data == 1), transform=max_transform)
        )
        if not geoms:
            print("[02_2] No valid polygons found in max composite.")
            return

        gdf_max = gpd.GeoDataFrame({"geometry": [shape(g["geometry"]) for g in geoms]}, crs=max_crs)
        gdf_max = gdf_max.explode(index_parts=True).reset_index(drop=True)
        gdf_max["geometry"] = gdf_max["geometry"].apply(fill_holes)

        if gdf_max.crs and gdf_max.crs.is_geographic:
            gdf_max_utm = gdf_max.to_crs(gdf_max.estimate_utm_crs())
            gdf_max["area"] = gdf_max_utm.area
        else:
            gdf_max["area"] = gdf_max.area

        gdf_max = gdf_max[gdf_max["area"] >= MIN_AREA_THRESHOLD].copy()
        if gdf_max.empty:
            print("[02_2] No polygons remain after area filtering.")
            return

        gdf_max["lake_pos"] = "冰前湖"
        gdf_max["ice_flag"] = 0
        if os.path.exists(args.ice_mask_path):
            try:
                ice = gpd.read_file(args.ice_mask_path)
                clipped_ice = clip_ice_mask_to_tile_aoi(ice, src.bounds, gdf_max.crs)
                if clipped_ice is None or clipped_ice.empty:
                    print("  [warning] Ice mask has no overlap with current tile AOI.")
                    clipped_ice = None

                if clipped_ice is not None:
                    unified = (
                        clipped_ice.union_all()
                        if hasattr(clipped_ice, "union_all")
                        else clipped_ice.unary_union
                    )
                else:
                    unified = None

                if unified is not None and not unified.is_empty:
                    inside_mask = gdf_max.geometry.apply(lambda geom: geom.covered_by(unified))
                    gdf_max.loc[inside_mask, "lake_pos"] = "冰面湖"
                    gdf_max.loc[inside_mask, "ice_flag"] = 1
                else:
                    print("  [warning] Unified clipped ice geometry is empty.")
            except Exception as exc:
                print(f"  [warning] Ice-mask classification failed: {exc}")
        else:
            print(f"  [warning] Ice-mask not found: {args.ice_mask_path}")

        if args.filter_only_ice_surface_lakes:
            gdf_max = gdf_max[gdf_max["lake_pos"] == "冰面湖"].copy()
            if gdf_max.empty:
                print("[02_2] No ice-surface lakes remain after mask filtering.")
                return

        gdf_max = gdf_max.sort_values(by="area", ascending=False).reset_index(drop=True)
        gdf_max["lake_id"] = range(1, len(gdf_max) + 1)
        gdf_max = add_lake_elevation_field(gdf_max, args.dem_tif, "lake_id", "elev_m")

        gdf_max.to_file(os.path.join(composite_out_dir, "Water_Max_Composite_Polygons.shp"), driver="ESRI Shapefile", encoding="utf-8")

        conditions = [
            gdf_max["area"] >= LARGE_LAKE_THRESHOLD,
            (gdf_max["area"] >= MEDIUM_LAKE_THRESHOLD) & (gdf_max["area"] < LARGE_LAKE_THRESHOLD),
        ]
        gdf_max["lake_type"] = np.select(conditions, ["Large", "Medium"], default="Small")
        gdf_max.to_file(os.path.join(final_out_dir, "Water_Max_Filtered_Polygons.shp"), driver="ESRI Shapefile", encoding="utf-8")

        if water_sum_path and os.path.exists(water_sum_path):
            min_geoms = []
            with rasterio.open(water_sum_path) as src_sum:
                shapes_with_freq = []
                for _, row in gdf_max.iterrows():
                    try:
                        out_image, out_transform = rasterio.mask.mask(src_sum, [row["geometry"]], crop=True, filled=True, nodata=0)
                        lake_sum_arr = out_image[0]
                        lake_max_freq = np.max(lake_sum_arr)
                        if lake_max_freq <= 0:
                            continue
                        shapes_with_freq.append((row["geometry"], float(lake_max_freq)))
                        lake_occ = lake_sum_arr.astype(float) / lake_max_freq
                        lake_min_bool = lake_occ >= 0.6
                        if not np.any(lake_min_bool):
                            continue
                        lake_min_arr = binary_closing(binary_opening(lake_min_bool, structure=np.ones((3, 3), dtype=bool)), structure=np.ones((3, 3), dtype=bool)).astype(np.uint8)
                        if np.max(lake_min_arr) != 1:
                            continue
                        results = (
                            {
                                "properties": {"raster_val": 1, "max_freq": float(lake_max_freq), "max_id": int(row.get("lake_id", -1))},
                                "geometry": s,
                            }
                            for s, v in shapes(lake_min_arr, mask=(lake_min_arr == 1), transform=out_transform)
                        )
                        for item in results:
                            min_geoms.append(item)
                    except ValueError:
                        continue

                if shapes_with_freq:
                    global_sum_array = src_sum.read(1)
                    max_freq_array = np.zeros_like(global_sum_array, dtype=np.float32)
                    rasterize(shapes_with_freq, out=max_freq_array, transform=src_sum.transform, fill=0, all_touched=False)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        occ = np.where(max_freq_array > 0, global_sum_array.astype(np.float32) / max_freq_array, 0.0)
                    occ_path = os.path.join(composite_out_dir, "Water_Occurrence_Composite.tif")
                    prof = src_sum.profile.copy()
                    prof.update(dtype=rasterio.float32, nodata=0.0)
                    with rasterio.open(occ_path, "w", **prof) as dst_occ:
                        dst_occ.write(occ, 1)

                if min_geoms:
                    polys = [shape(g["geometry"]) for g in min_geoms]
                    gdf_min = gpd.GeoDataFrame([g["properties"] for g in min_geoms], geometry=polys, crs=src_sum.crs)
                    gdf_min["geometry"] = gdf_min.geometry.buffer(0)
                    gdf_min = gdf_min.explode(index_parts=True).reset_index(drop=True)
                    gdf_min["geometry"] = gdf_min["geometry"].apply(fill_holes)
                    if gdf_min.crs and gdf_min.crs.is_geographic:
                        gdf_min_utm = gdf_min.to_crs(gdf_min.estimate_utm_crs())
                        gdf_min["area_min"] = gdf_min_utm.area
                    else:
                        gdf_min["area_min"] = gdf_min.area
                    gdf_min = gdf_min[gdf_min["area_min"] >= 10000].copy()
                    if not gdf_min.empty:
                        gdf_min = gdf_min.sort_values("area_min", ascending=False).reset_index(drop=True)
                        gdf_min["min_id"] = range(1, len(gdf_min) + 1)
                        gdf_min.to_file(os.path.join(final_out_dir, "Water_Min_Filtered_Polygons.shp"), driver="ESRI Shapefile", encoding="utf-8")

        buffer_resolution = 5
        if gdf_max.crs and gdf_max.crs.is_geographic:
            utm = gdf_max.estimate_utm_crs()
            gdf_utm = gdf_max.to_crs(utm)
            if BUFFER_MODE == "fixed":
                gdf_utm["geometry"] = gdf_utm.geometry.buffer(BUFFER_DISTANCE, resolution=buffer_resolution)
            else:
                radii = np.sqrt(gdf_utm["area"] / np.pi)
                dists = np.clip(radii * BUFFER_DYNAMIC_MULTIPLIER, 0, BUFFER_DYNAMIC_MAX_RADIUS)
                gdf_utm["geometry"] = gdf_utm.geometry.buffer(dists, resolution=buffer_resolution)
            gdf_buffered = gdf_utm.to_crs(gdf_max.crs)
        else:
            gdf_buffered = gdf_max.copy()
            if BUFFER_MODE == "fixed":
                gdf_buffered["geometry"] = gdf_max.geometry.buffer(BUFFER_DISTANCE, resolution=buffer_resolution)
            else:
                radii = np.sqrt(gdf_buffered["area"] / np.pi)
                dists = np.clip(radii * BUFFER_DYNAMIC_MULTIPLIER, 0, BUFFER_DYNAMIC_MAX_RADIUS)
                gdf_buffered["geometry"] = gdf_max.geometry.buffer(dists, resolution=buffer_resolution)

        gdf_buffered["geometry"] = gdf_buffered["geometry"].apply(fill_holes)
        gdf_buffered = gdf_buffered[gdf_buffered["geometry"].notnull() & ~gdf_buffered["geometry"].is_empty]
        gdf_buffered.to_file(
            os.path.join(final_out_dir, "Water_Max_Filtered_Polygons_Buffered.shp"),
            driver="ESRI Shapefile",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[error] Final AOI generation failed: {exc}")
        traceback.print_exc()
        return

    print(f"[02_2] done. outputs: {composite_out_dir} | {final_out_dir}")


if __name__ == "__main__":
    main()
