"""
Stage 02_2: build water composites and filtered lake AOIs from Stage 02_1 binaries.

Inputs:
- result/02_Lake_Extraction/01_NDWI_binary_result

Outputs:
- result/02_Lake_Extraction/03_Water_Composite
- result/02_Lake_Extraction/04_Filtered_Lakes
"""

import glob
import os
import re
import traceback

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import MultiPolygon, Polygon, shape

os.environ["GDAL_SKIP"] = "DXF"
os.environ["CPL_LOG"] = "NUL"
os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"

OUTPUT_BASE = r"Z:\冰面湖水量测算_Local\冰面湖水位论文代码数据整理\result\02_Lake_Extraction"

MIN_AREA_THRESHOLD = 10000
LARGE_LAKE_THRESHOLD = 62500

BUFFER_MODE = "dynamic"
BUFFER_DISTANCE = 300
BUFFER_DYNAMIC_MULTIPLIER = 1
BUFFER_DYNAMIC_MAX_RADIUS = 300

ICE_SURFACE_MASK_PATH = r"D:\Haoyu_space\03_data\Greenland_IceSheet_IceMap_Merge\27XVG_greenlandsheet.shp"
DEM_TIF = r"D:\Haoyu_space\03_data\ArcticDEM\30m\DEM.tif"

SKIP_COMPOSITE = False
FINAL_OUT_SUBDIR = "04_Filtered_Lakes"
START_MMDD = (6, 15)
END_MMDD = (8, 15)


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
        if len(data.shape) == 2:
            dst.write(data.astype(dtype), 1)
        else:
            dst.write(data.astype(dtype))


def add_lake_elevation_field(gdf, lake_id_col="lake_id", output_field="elev_m"):
    if gdf is None or gdf.empty or lake_id_col not in gdf.columns:
        return gdf
    if not os.path.exists(DEM_TIF):
        print(f"  [warning] DEM not found: {DEM_TIF}")
        return gdf

    gdf = gdf.copy()
    gdf[output_field] = np.nan
    try:
        with rasterio.open(DEM_TIF) as src_dem:
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


def in_target_window(filename):
    match = re.search(r"(\d{8})", filename)
    if not match:
        return False
    date_str = match.group(1)
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    mmdd = (month, day)
    return START_MMDD <= mmdd <= END_MMDD


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
    binary_out_dir = os.path.join(OUTPUT_BASE, "01_NDWI_binary_result")
    composite_out_dir = os.path.join(OUTPUT_BASE, "03_Water_Composite")
    final_out_dir = os.path.join(OUTPUT_BASE, FINAL_OUT_SUBDIR)
    os.makedirs(composite_out_dir, exist_ok=True)
    os.makedirs(final_out_dir, exist_ok=True)

    bin_files = glob.glob(os.path.join(binary_out_dir, "*_binary.tif"))
    bin_files = [f for f in bin_files if in_target_window(os.path.basename(f))]
    if not bin_files:
        print("[02_2] no binary tif files found in 01_NDWI_binary_result.")
        return

    tile_binary_files = {}
    for bin_path in bin_files:
        tile_binary_files.setdefault(get_tile_id(os.path.basename(bin_path)), []).append(bin_path)

    max_files_to_mosaic = []
    sum_files_to_mosaic = []
    if not SKIP_COMPOSITE:
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
                        bounds = src.bounds
                        min_x = min(min_x, bounds.left)
                        min_y = min(min_y, bounds.bottom)
                        max_x = max(max_x, bounds.right)
                        max_y = max(max_y, bounds.top)
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
    if not SKIP_COMPOSITE:
        water_max_path = merge_and_clean(max_files_to_mosaic, "Water_Max_Composite.tif", composite_out_dir)
        merge_and_clean(sum_files_to_mosaic, "Water_Sum_Composite.tif", composite_out_dir)

    if not water_max_path or not os.path.exists(water_max_path):
        print("[02_2] No Water_Max_Composite.tif found.")
        return

    try:
        from scipy.ndimage import binary_closing, binary_opening

        with rasterio.open(water_max_path) as src:
            max_data = src.read(1)
            max_transform = src.transform
            max_crs = src.crs

        struct_3x3 = np.ones((3, 3), dtype=bool)
        max_data = binary_closing(
            binary_opening(max_data == 1, structure=struct_3x3),
            structure=struct_3x3,
        ).astype(np.uint8)

        geoms = list(
            {"properties": {"raster_val": int(v)}, "geometry": s}
            for s, v in shapes(max_data, mask=(max_data == 1), transform=max_transform)
        )
        if not geoms:
            print("[02_2] No valid polygons found in max composite.")
            return

        polygons = [shape(g["geometry"]) for g in geoms]
        gdf_max = gpd.GeoDataFrame({"geometry": polygons}, crs=max_crs)
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

        gdf_max["lake_pos"] = "\u51b0\u524d\u6e56"
        if os.path.exists(ICE_SURFACE_MASK_PATH):
            try:
                ice_mask_gdf = gpd.read_file(ICE_SURFACE_MASK_PATH)
                if ice_mask_gdf.crs != gdf_max.crs:
                    ice_mask_gdf = ice_mask_gdf.to_crs(gdf_max.crs)
                unified_ice_geom = ice_mask_gdf.union_all() if hasattr(ice_mask_gdf, "union_all") else ice_mask_gdf.unary_union
                inside_mask = gdf_max.geometry.apply(lambda geom: geom.covered_by(unified_ice_geom))
                gdf_max.loc[inside_mask, "lake_pos"] = "\u51b0\u9762\u6e56"
            except Exception as exc:
                print(f"  [warning] Ice-mask classification failed: {exc}")
        else:
            print(f"  [warning] Ice-mask not found: {ICE_SURFACE_MASK_PATH}")

        gdf_max = gdf_max[gdf_max["lake_pos"] == "\u51b0\u9762\u6e56"].copy()
        if gdf_max.empty:
            print("[02_2] No ice-surface lakes remain after mask filtering.")
            return

        gdf_max = gdf_max.sort_values(by="area", ascending=False).reset_index(drop=True)
        gdf_max["lake_id"] = range(1, len(gdf_max) + 1)
        gdf_max = add_lake_elevation_field(gdf_max, "lake_id", "elev_m")

        max_shp_path = os.path.join(composite_out_dir, "Water_Max_Composite_Polygons.shp")
        gdf_max.to_file(max_shp_path, driver="ESRI Shapefile", encoding="utf-8")

        gdf_max["lake_type"] = np.where(gdf_max["area"] >= LARGE_LAKE_THRESHOLD, "Large", "Small")

        cat_shp_path = os.path.join(final_out_dir, "Water_Max_Filtered_Polygons.shp")
        gdf_max.to_file(cat_shp_path, driver="ESRI Shapefile", encoding="utf-8")

        # Removed minimum-water extraction by occurrence frequency (>= 0.6).

        buffer_resolution = 5
        if gdf_max.crs and gdf_max.crs.is_geographic:
            utm_crs = gdf_max.estimate_utm_crs()
            gdf_utm = gdf_max.to_crs(utm_crs)
            if BUFFER_MODE == "fixed":
                gdf_utm["geometry"] = gdf_utm.geometry.buffer(BUFFER_DISTANCE, resolution=buffer_resolution)
            else:
                radii = np.sqrt(gdf_utm["area"] / np.pi)
                buffer_dists = np.clip(radii * BUFFER_DYNAMIC_MULTIPLIER, 0, BUFFER_DYNAMIC_MAX_RADIUS)
                gdf_utm["geometry"] = gdf_utm.geometry.buffer(buffer_dists, resolution=buffer_resolution)
            gdf_buffered = gdf_utm.to_crs(gdf_max.crs)
        else:
            gdf_buffered = gdf_max.copy()
            if BUFFER_MODE == "fixed":
                gdf_buffered["geometry"] = gdf_max.geometry.buffer(BUFFER_DISTANCE, resolution=buffer_resolution)
            else:
                radii = np.sqrt(gdf_buffered["area"] / np.pi)
                buffer_dists = np.clip(radii * BUFFER_DYNAMIC_MULTIPLIER, 0, BUFFER_DYNAMIC_MAX_RADIUS)
                gdf_buffered["geometry"] = gdf_max.geometry.buffer(buffer_dists, resolution=buffer_resolution)
        gdf_buffered["geometry"] = gdf_buffered["geometry"].apply(fill_holes)
        gdf_buffered = gdf_buffered[gdf_buffered["geometry"].notnull() & ~gdf_buffered["geometry"].is_empty]
        cat_buf_path = os.path.join(final_out_dir, "Water_Max_Filtered_Polygons_Buffered.shp")
        gdf_buffered.to_file(cat_buf_path, driver="ESRI Shapefile", encoding="utf-8")
    except Exception as exc:
        print(f"[error] Final AOI generation failed: {exc}")
        traceback.print_exc()
        return

    print(f"[02_2] done. outputs: {composite_out_dir} | {final_out_dir}")


if __name__ == "__main__":
    main()


