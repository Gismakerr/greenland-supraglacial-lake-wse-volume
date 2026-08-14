"""
Stage 06_1: extract SWOT PIXC points into per-lake GDB using water-period windows.

Inputs:
- Stage 05_2 topology table for lake ids and water periods.
- Buffered lake polygons for extraction geometry.
- Greenland SWOT tiles and local PIXC directory.

Outputs:
- per-lake PIXC GDB: one layer per PIXC granule record id.
- extraction index table and lake summary table.
- corrupted-NC auto-repair log (resume-friendly).
"""

import os
import re
import threading
import time
import atexit
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from shapely.geometry import box

try:
    import earthaccess
except Exception:  # pragma: no cover
    earthaccess = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, "result")
PIXC_DIR_D = r"G:\SWOT\Data\PIXC"
TILES_SHP = r"D:\Haoyu_space\03_data\SWOT_Tiles\Greenland_tiles.shp"
LAKES_BUFFERED_SHP = os.path.join(
    OUTPUT_ROOT_DIR,
    "02_Lake_Extraction",
    "04_Filtered_Lakes",
    "Water_Max_Filtered_Polygons_Buffered.shp",
)
INPUT_05_2_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_2_plot_singlebranch_from_05_1_04_6_1")
LAKE_FILTER_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons.shp"
)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_1_extract_swot_pixc_to_gdb")

METHOD_NAMES = ["Otsu"]
TARGET_SIZE_CATEGORIES = []
MAX_WORKERS = 4
VERBOSE_PROGRESS = True
PROCESS_ALL_LAKES = False
DEFAULT_SIZE_CATEGORY_FOR_MISSING_LAKES = "Small"
MAX_LAKE_ID_TO_PROCESS = 10**9
UPDATE_CD_DIFF_EACH_LAKE = True
UPDATE_DAILY_COMPARE_EACH_LAKE = False
RESUME_FROM_INDEX = True
RESUME_SKIP_PRECHECK_WHEN_INDEX_EXISTS = True
LAKE_RESUME_FROM_SUMMARY = True
LAKE_RESUME_SKIP_STATUSES = {"ok"}
RESUME_TERMINAL_STATUSES = {
    "written",
    "existing_layer",
    "empty_after_lake_clip",
    "precheck_failed",
    "read_failed",
    "write_failed",
}

ENABLE_PIXC_AUTO_REPAIR = True
SKIP_RETRY_FOR_KNOWN_FAILED_REPAIR = True
PIXC_REPAIR_REMOTE_RETRIES = 2
PIXC_REPAIR_SHORT_NAMES = ["SWOT_L2_HR_PIXC_D", "SWOT_L2_HR_PIXC"]
NETCDF_OPEN_RETRIES = 2
NETCDF_OPEN_RETRY_SLEEP_SEC = 0.2
PRECHECK_NC_BEFORE_EXTRACT = True
RUN_PHASE1_PRECHECK = True
RUN_PHASE2_EXTRACT = True

INDEX_COLUMNS = [
    "lake_id", "size_category", "date", "granule_name", "layer_name", "status",
    "point_count", "gdb_path", "gdb_layer", "repair_status", "detail",
    "pixc_version", "gi_code", "granule_match_key",
]
SUMMARY_COLUMNS = [
    "lake_id", "size_category", "status", "candidate_granule_count", "written_layer_count",
    "existing_layer_count", "empty_clip_count", "failed_count",
    "candidate_granule_count_d",
    "written_layer_count_d",
    "existing_layer_count_d",
    "failed_count_d",
    "resume_skipped_count",
]
PIXC_REPAIR_LOG_COLUMNS = [
    "timestamp", "lake_id", "date", "granule_name", "status", "detail", "repaired_path",
]

GLOBAL_NETCDF_LOCK = threading.Lock()
GLOBAL_EARTHACCESS_LOCK = threading.Lock()
GLOBAL_EARTHACCESS_READY = False
GLOBAL_REPAIR_LOCK = threading.Lock()
GLOBAL_REPAIR_STATUS = {}
GLOBAL_REPAIR_ROWS = []
GLOBAL_NC_HEALTH_LOCK = threading.Lock()
GLOBAL_NC_HEALTH_CACHE = {}
GLOBAL_RUN_LOCK_PATH = None


def progress(message):
    print(message, flush=True)


def release_run_lock():
    global GLOBAL_RUN_LOCK_PATH
    lock_path = GLOBAL_RUN_LOCK_PATH
    GLOBAL_RUN_LOCK_PATH = None
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def acquire_run_lock(lock_path):
    global GLOBAL_RUN_LOCK_PATH
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder_text = ""
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                holder_text = f.read().strip()
        except Exception:
            holder_text = ""
        stale_pid = None
        m = re.search(r"pid=(\d+)", holder_text or "")
        if m:
            try:
                stale_pid = int(m.group(1))
            except Exception:
                stale_pid = None
        if stale_pid is not None:
            alive = True
            try:
                os.kill(stale_pid, 0)
            except OSError:
                alive = False
            except Exception:
                alive = True
            if not alive:
                try:
                    os.remove(lock_path)
                    progress(f"[06_1] stale lock removed for dead pid={stale_pid}")
                except Exception:
                    pass
                return acquire_run_lock(lock_path)
        progress(f"[06_1] lock exists, another instance is running: {holder_text or lock_path}")
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except Exception:
            pass
        raise
    GLOBAL_RUN_LOCK_PATH = lock_path
    atexit.register(release_run_lock)
    return True


def nc_file_signature(nc_path):
    try:
        stat = os.stat(nc_path)
        return (int(stat.st_size), int(stat.st_mtime))
    except Exception:
        return (-1, -1)


def get_method_paths(method_name):
    input_05_2_dir = os.path.join(INPUT_05_2_DIR, method_name)
    output_dir = os.path.join(OUTPUT_DIR, method_name)
    return {
        "method_name": method_name,
        "input_05_2_topology_csv": os.path.join(input_05_2_dir, "05_2_denoised_topology_records.csv"),
        "input_05_2_topology_csv_alt": os.path.join(input_05_2_dir, "05_2_denoised_curve_topology_relations.csv"),
        "output_dir": output_dir,
        # Keep legacy D path for compatibility with downstream scripts expecting this folder.
        "pixc_gdb_dir": os.path.join(output_dir, "06_1_swot_pixc_gdb_by_lake"),
        "index_csv": os.path.join(output_dir, "06_1_pixc_index.csv"),
        "summary_csv": os.path.join(output_dir, "06_1_pixc_extract_summary.csv"),
        "repair_log_csv": os.path.join(output_dir, "06_1_pixc_repair_log.csv"),
    }


def normalize_date_str(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    match = re.search(r"(\d{8})", text)
    if match:
        return match.group(1)
    dt = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(dt) else dt.strftime("%Y%m%d")


def to_datetime_ymd(value):
    text = normalize_date_str(value)
    if not text:
        return pd.NaT
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def parse_pixc_filename(filename):
    stem = os.path.splitext(str(filename))[0]
    parts = stem.split("_")
    if len(parts) >= 11:
        pg_code = str(parts[9])
        gi_code = pg_code[1].upper() if len(pg_code) >= 2 else ""
        pixc_version = pg_code[2].upper() if len(pg_code) >= 3 else ""
        return {
            "cycle": parts[4],
            "pass": parts[5],
            "tile": parts[6],
            "time": parts[7],
            "time_end": parts[8],
            "pg_code": pg_code,
            "product_rev": parts[10],
            "gi_code": gi_code,
            "pixc_version": pixc_version,
            "filename": str(filename),
        }
    if len(parts) >= 8:
        return {
            "cycle": parts[4],
            "pass": parts[5],
            "tile": parts[6],
            "time": parts[7],
            "time_end": "",
            "pg_code": "",
            "product_rev": parts[-1] if parts else "",
            "gi_code": "",
            "pixc_version": "",
            "filename": str(filename),
        }
    return None


def build_pixc_record_id(nc_path):
    filename = os.path.basename(str(nc_path))
    info = parse_pixc_filename(filename)
    stem = os.path.splitext(filename)[0]
    if not info:
        return stem
    pg_code = str(info.get("pg_code", "")).strip()
    product_rev = str(info.get("product_rev", "")).strip()
    if pg_code and product_rev:
        return f"pixc_{info['time']}_{info['pass']}_{info['tile']}_{pg_code}_{product_rev}"
    parts = stem.split("_")
    product_tag = parts[-1] if parts else "00"
    return f"pixc_{info['time']}_{info['pass']}_{info['tile']}_{product_tag}"


def build_legacy_pixc_record_id(nc_path):
    filename = os.path.basename(str(nc_path))
    info = parse_pixc_filename(filename)
    stem = os.path.splitext(filename)[0]
    if not info:
        return stem
    parts = stem.split("_")
    product_tag = parts[-1] if parts else "00"
    return f"pixc_{info['time']}_{info['pass']}_{info['tile']}_{product_tag}"


def build_granule_match_key(nc_path):
    info = parse_pixc_filename(os.path.basename(str(nc_path)))
    if not info:
        return os.path.splitext(os.path.basename(str(nc_path)))[0]
    return (
        f"{info.get('time','')}_{info.get('time_end','')}_{info.get('pass','')}_"
        f"{info.get('tile','')}_{info.get('gi_code','')}"
    )


def extract_pixc_date_str(nc_path):
    info = parse_pixc_filename(os.path.basename(str(nc_path)))
    ts = info["time"] if info else ""
    date_str = ts[:8] if len(ts) >= 8 else None
    return date_str if date_str and date_str.isdigit() else None


def classify_size_category(max_area_km2):
    value = pd.to_numeric(max_area_km2, errors="coerce")
    if pd.isna(value):
        return "Small"
    # Align with Stage-02 rule in this project:
    # area >= 0.0625 km2 -> Large; otherwise treated as non-Large.
    if float(value) >= 0.0625:
        return "Large"
    return "Small"


def load_topology_records(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df
    work = df.copy()
    # Compatibility: accept 05_2_denoised_curve_topology_relations.csv
    # (edge table with parent/child columns) and convert to node-style records.
    if "date" not in work.columns and {"parent_date", "child_date"}.issubset(set(work.columns)):
        node_rows = []
        for row in work.to_dict("records"):
            lake_id = row.get("lake_id", np.nan)
            size_category = row.get("size_category", "")
            p_date = normalize_date_str(row.get("parent_date", ""))
            c_date = normalize_date_str(row.get("child_date", ""))
            p_area = pd.to_numeric(row.get("parent_area_km2", np.nan), errors="coerce")
            c_area = pd.to_numeric(row.get("child_area_km2", np.nan), errors="coerce")
            if p_date:
                node_rows.append(
                    {"lake_id": lake_id, "date": p_date, "area_km2": p_area, "size_category": size_category}
                )
            if c_date:
                node_rows.append(
                    {"lake_id": lake_id, "date": c_date, "area_km2": c_area, "size_category": size_category}
                )
        work = pd.DataFrame(node_rows)
        if work.empty:
            return work
        work = work.drop_duplicates(subset=["lake_id", "date"], keep="last").reset_index(drop=True)
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["date"] = work["date"].astype(str).map(normalize_date_str)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    work["area_km2"] = pd.to_numeric(work.get("area_km2"), errors="coerce")
    work = work.dropna(subset=["date_dt"]).copy()
    return work.sort_values(["lake_id", "date_dt"]).reset_index(drop=True)


def build_lake_windows(topology_df):
    rows = []
    for lake_id, grp in topology_df.groupby("lake_id", sort=True):
        grp = grp.sort_values("date_dt")
        rows.append(
            {
                "lake_id": int(lake_id),
                "water_period_start": grp["date"].iloc[0],
                "water_period_end": grp["date"].iloc[-1],
                "size_category": classify_size_category(grp["area_km2"].max()),
            }
        )
    return pd.DataFrame(rows)


def to_date_text_ymd(value):
    if pd.isna(value):
        return ""
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y%m%d")


def expand_lake_windows_to_all_lakes(lake_windows_df, lake_geom_map, topology_df=None):
    if lake_windows_df is None or lake_windows_df.empty:
        base_df = pd.DataFrame(columns=["lake_id", "water_period_start", "water_period_end", "size_category"])
    else:
        base_df = lake_windows_df.copy()
    if not lake_geom_map:
        return base_df

    all_lake_ids = sorted([int(x) for x in lake_geom_map.keys()])
    if not all_lake_ids:
        return base_df

    existing_ids = set()
    if not base_df.empty and "lake_id" in base_df.columns:
        existing_ids = set(pd.to_numeric(base_df["lake_id"], errors="coerce").dropna().astype(int).tolist())
    missing_ids = [int(x) for x in all_lake_ids if int(x) not in existing_ids]
    if not missing_ids:
        return base_df

    start_text = ""
    end_text = ""
    if topology_df is not None and (not topology_df.empty) and ("date_dt" in topology_df.columns):
        start_text = to_date_text_ymd(topology_df["date_dt"].min())
        end_text = to_date_text_ymd(topology_df["date_dt"].max())
    if (not start_text) or (not end_text):
        start_text = to_date_text_ymd(base_df.get("water_period_start", pd.Series(dtype=object)).min())
        end_text = to_date_text_ymd(base_df.get("water_period_end", pd.Series(dtype=object)).max())
    if (not start_text) or (not end_text):
        # Keep a conservative default window when no date context is available.
        start_text = "20240101"
        end_text = "20241231"

    add_rows = [
        {
            "lake_id": int(lake_id),
            "water_period_start": str(start_text),
            "water_period_end": str(end_text),
            "size_category": str(DEFAULT_SIZE_CATEGORY_FOR_MISSING_LAKES),
        }
        for lake_id in missing_ids
    ]
    out_df = pd.concat([base_df, pd.DataFrame(add_rows)], ignore_index=True)
    out_df["lake_id"] = pd.to_numeric(out_df["lake_id"], errors="coerce")
    out_df = out_df.dropna(subset=["lake_id"]).copy()
    out_df["lake_id"] = out_df["lake_id"].astype(int)
    out_df = out_df.drop_duplicates(subset=["lake_id"], keep="first").sort_values("lake_id").reset_index(drop=True)
    return out_df


def load_tiles_gdf(tile_shp):
    if not os.path.exists(tile_shp):
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(tile_shp)
    if gdf.empty:
        return gdf
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load_lake_buffered_geom_map(shp_path):
    if not os.path.exists(shp_path):
        return {}
    gdf = gpd.read_file(shp_path)
    if gdf.empty or "lake_id" not in gdf.columns:
        return {}
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf["lake_id"] = pd.to_numeric(gdf["lake_id"], errors="coerce")
    gdf = gdf.dropna(subset=["lake_id"]).copy()
    gdf["lake_id"] = gdf["lake_id"].astype(int)
    out = {}
    for lake_id, grp in gdf.groupby("lake_id", sort=False):
        geom = grp.geometry.unary_union
        if geom is not None and not geom.is_empty:
            out[int(lake_id)] = geom
    return out


def load_allowed_lake_ids_from_shp(shp_path):
    if not os.path.exists(shp_path):
        return set()
    gdf = gpd.read_file(shp_path)
    if gdf.empty or "lake_id" not in gdf.columns:
        return set()
    return set(pd.to_numeric(gdf["lake_id"], errors="coerce").dropna().astype(int).tolist())


def build_pixc_file_map(pixc_dir):
    file_map = {}
    if not os.path.exists(pixc_dir):
        return {}
    for filename in os.listdir(pixc_dir):
        if not (filename.startswith("SWOT_L2_HR_PIXC") and filename.endswith(".nc")):
            continue
        info = parse_pixc_filename(filename)
        if not info:
            continue
        pass_id = str(info["pass"])
        tile_id = str(info["tile"])
        tile_pure = tile_id[:-1] if tile_id and tile_id[-1].isalpha() else tile_id
        pass_pure = pass_id.lstrip("0") or "0"
        file_path = os.path.join(pixc_dir, filename)
        for key in {
            f"{pass_id}_{tile_id}",
            f"{pass_id}_{tile_pure}",
            f"{pass_pure}_{tile_id}",
            f"{pass_pure}_{tile_pure}",
        }:
            file_map.setdefault(key, [])
            if file_path not in file_map[key]:
                file_map[key].append(file_path)
    return file_map


def gi_priority_key(nc_path):
    info = parse_pixc_filename(os.path.basename(str(nc_path)))
    gi = str(info.get("gi_code", "")).upper() if info else ""
    if gi == "G":
        return (0, str(nc_path))
    if gi == "I":
        return (1, str(nc_path))
    return (2, str(nc_path))


def get_tile_geom_by_pass_tile(tiles_gdf, pass_id, tile_id):
    if tiles_gdf is None or tiles_gdf.empty:
        return None
    tile_field = "Name" if "Name" in tiles_gdf.columns else ("NAME" if "NAME" in tiles_gdf.columns else None)
    if tile_field is None:
        return None
    target = f"{str(pass_id)}_{str(tile_id)}"
    sel = tiles_gdf[tiles_gdf[tile_field].astype(str) == target]
    if sel.empty:
        return None
    return sel.iloc[0].geometry


def granule_name_from_remote(granule):
    try:
        name = granule["umm"]["GranuleUR"]
        return f"{name}.nc" if not str(name).endswith(".nc") else str(name)
    except Exception:
        pass
    try:
        name = granule["meta"]["native-id"]
        return f"{name}.nc" if not str(name).endswith(".nc") else str(name)
    except Exception:
        return ""


def load_repair_state(log_csv_path):
    if not os.path.exists(log_csv_path):
        return {}
    try:
        df = pd.read_csv(log_csv_path, encoding="utf-8-sig")
    except Exception:
        return {}
    if df.empty or "granule_name" not in df.columns or "status" not in df.columns:
        return {}
    out = {}
    for row in df.to_dict("records"):
        name = str(row.get("granule_name", "")).strip()
        if name:
            out[name] = {
                "status": str(row.get("status", "")).strip(),
                "repaired_path": str(row.get("repaired_path", "")).strip(),
                "detail": str(row.get("detail", "")).strip(),
            }
    return out


def record_repair_row(lake_id, day_text, granule_name, status, detail, repaired_path):
    row = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lake_id": int(lake_id),
        "date": str(day_text),
        "granule_name": str(granule_name),
        "status": str(status),
        "detail": str(detail),
        "repaired_path": str(repaired_path or ""),
    }
    with GLOBAL_REPAIR_LOCK:
        GLOBAL_REPAIR_ROWS.append(row)
        GLOBAL_REPAIR_STATUS[str(granule_name)] = {
            "status": str(status),
            "repaired_path": str(repaired_path or ""),
            "detail": str(detail),
        }


def attempt_repair_corrupt_pixc(nc_path, tiles_gdf):
    if earthaccess is None:
        return False, "earthaccess_missing", ""
    granule_name = os.path.basename(str(nc_path))
    info = parse_pixc_filename(granule_name)
    if not info:
        return False, "filename_parse_failed", ""
    tile_geom = get_tile_geom_by_pass_tile(tiles_gdf, info.get("pass", ""), info.get("tile", ""))
    if tile_geom is None:
        return False, "tile_geometry_missing", ""
    date_text = str(info.get("time", ""))[:8]
    day_dt = to_datetime_ymd(date_text)
    if pd.isna(day_dt):
        return False, "invalid_date", ""
    start_text = day_dt.strftime("%Y-%m-%dT00:00:00")
    end_text = (day_dt + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59")
    minx, miny, maxx, maxy = tile_geom.bounds

    with GLOBAL_EARTHACCESS_LOCK:
        global GLOBAL_EARTHACCESS_READY
        if not GLOBAL_EARTHACCESS_READY:
            earthaccess.login(strategy="netrc")
            GLOBAL_EARTHACCESS_READY = True

        last_error = ""
        for short_name in PIXC_REPAIR_SHORT_NAMES:
            results = []
            for attempt in range(1, int(PIXC_REPAIR_REMOTE_RETRIES) + 1):
                try:
                    results = earthaccess.search_data(
                        short_name=short_name,
                        temporal=(start_text, end_text),
                        bounding_box=(float(minx), float(miny), float(maxx), float(maxy)),
                    )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < int(PIXC_REPAIR_REMOTE_RETRIES):
                        time.sleep(1.0)
            if not results:
                continue
            matched = [item for item in results if granule_name_from_remote(item) == granule_name]
            if not matched:
                continue

            backup_path = str(nc_path) + ".corrupt.bak"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            except Exception:
                pass
            if os.path.exists(nc_path):
                try:
                    os.replace(nc_path, backup_path)
                except Exception:
                    pass
            try:
                download_dir = os.path.dirname(str(nc_path)) or PIXC_DIR_D
                downloaded = earthaccess.download(matched, str(download_dir), threads=1)
                downloaded_paths = [str(path) for path in downloaded] if downloaded else []
                for path_text in downloaded_paths:
                    if os.path.basename(path_text) == granule_name and os.path.exists(path_text):
                        try:
                            if os.path.exists(backup_path):
                                os.remove(backup_path)
                        except Exception:
                            pass
                        return True, "repaired_downloaded", str(path_text)
                if os.path.exists(backup_path) and not os.path.exists(nc_path):
                    os.replace(backup_path, nc_path)
                return False, "download_failed", ""
            except Exception as exc:
                if os.path.exists(backup_path) and not os.path.exists(nc_path):
                    try:
                        os.replace(backup_path, nc_path)
                    except Exception:
                        pass
                return False, f"download_exception:{exc}", ""
        return False, f"remote_not_found:{last_error}", ""


def extract_data_from_nc(nc_path, target_geom):
    for attempt in range(int(NETCDF_OPEN_RETRIES) + 1):
        try:
            with GLOBAL_NETCDF_LOCK:
                with netCDF4.Dataset(nc_path, "r") as nc:
                    try:
                        nc_bbox = box(
                            nc.getncattr("geospatial_lon_min"),
                            nc.getncattr("geospatial_lat_min"),
                            nc.getncattr("geospatial_lon_max"),
                            nc.getncattr("geospatial_lat_max"),
                        )
                        if not target_geom.intersects(nc_bbox):
                            return None, "empty", ""
                    except Exception:
                        pass

                    group = nc.groups["pixel_cloud"]
                    lats = group.variables["latitude"][:]
                    lons = group.variables["longitude"][:]
                    minx, miny, maxx, maxy = target_geom.bounds
                    bbox_mask = (
                        (lons >= minx) & (lons <= maxx) &
                        (lats >= miny) & (lats <= maxy)
                    )
                    if not np.any(bbox_mask):
                        return None, "empty", ""

                    idx = np.where(bbox_mask)[0]
                    data = {"latitude": lats[idx], "longitude": lons[idx]}
                    vars_to_read = [
                        "classification", "classification_qual", "sig0", "sig0_qual", "coherent_power",
                        "power_plus_y", "power_minus_y", "water_frac", "water_frac_uncert", "cross_track",
                        "inc", "geolocation_qual", "height", "geoid", "solid_earth_tide", "load_tide_fes",
                        "load_tide_got", "pole_tide", "interferogram_qual",
                    ]
                    for var_name in vars_to_read:
                        if var_name in group.variables:
                            values = group.variables[var_name][idx]
                            if np.ma.is_masked(values):
                                values = values.filled(np.nan)
                            data[var_name] = values
                        else:
                            data[var_name] = np.nan

                    df = pd.DataFrame(data)
                    for col in ["height", "geoid", "solid_earth_tide", "pole_tide", "load_tide_fes", "load_tide_got"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    # WSE formula: height - geoid - solid_earth_tide - pole_tide - load_tide
                    # load_tide uses FES first, then GOT, finally 0 when both missing.
                    load_tide = df["load_tide_fes"].where(df["load_tide_fes"].notna(), df["load_tide_got"]).fillna(0.0)
                    df["load_tide_used"] = load_tide
                    df["wse"] = (
                        df["height"]
                        - df["geoid"].fillna(0.0)
                        - df["solid_earth_tide"].fillna(0.0)
                        - df["pole_tide"].fillna(0.0)
                        - load_tide
                    )
                    # Keep coherence metric for downstream 06_2 quality filters.
                    for col in ["coherent_power", "power_plus_y", "power_minus_y"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        coh_raw = df["coherent_power"] / np.sqrt(df["power_plus_y"] * df["power_minus_y"])
                    df["coh"] = np.clip(coh_raw, 0, 1)
                    # Ensure key fields always exist in exports.
                    for key_col in ["wse", "geolocation_qual", "classification", "coh", "sig0", "cross_track"]:
                        if key_col not in df.columns:
                            df[key_col] = np.nan
                    # Exclude low-confidence classes requested by workflow: remove class 1 and 2.
                    class_vals = pd.to_numeric(df["classification"], errors="coerce")
                    df = df[~class_vals.isin([1, 2])].copy()
                    if df.empty:
                        return None, "empty", ""
                    # Keep these fields for internal computation only; do not export them.
                    df = df.drop(
                        columns=[
                            "coherent_power",
                            "power_plus_y",
                            "power_minus_y",
                            "water_frac",
                            "water_frac_uncert",
                            "solid_earth_tide",
                            "load_tide_fes",
                            "load_tide_got",
                        ],
                        errors="ignore",
                    )
                    points = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                        crs="EPSG:4326",
                    )
                    clipped = points[points.within(target_geom)]
                    return (None, "empty", "") if clipped.empty else (clipped, "ok", "")
        except Exception as exc:
            err_text = str(exc)
            if attempt < int(NETCDF_OPEN_RETRIES):
                time.sleep(float(NETCDF_OPEN_RETRY_SLEEP_SEC))
                continue
            lowered = err_text.lower()
            if any(
                key in lowered
                for key in [
                    "truncated",
                    "superblock",
                    "unable to open file",
                    "h5fopen",
                    "read failed",
                    "netcdf: hdf error",
                    "hdf error",
                    "errno -101",
                ]
            ):
                return None, "corrupt", err_text
            return None, "error", err_text
    return None, "error", "unknown_error"


def check_nc_health(nc_path):
    path_key = str(nc_path)
    sig = nc_file_signature(path_key)
    with GLOBAL_NC_HEALTH_LOCK:
        cached = GLOBAL_NC_HEALTH_CACHE.get(path_key)
    if cached and cached.get("sig") == sig:
        return str(cached.get("status", "error")), str(cached.get("detail", "cached_unknown"))

    for attempt in range(int(NETCDF_OPEN_RETRIES) + 1):
        try:
            with GLOBAL_NETCDF_LOCK:
                with netCDF4.Dataset(nc_path, "r") as nc:
                    if "pixel_cloud" not in nc.groups:
                        status, detail = "error", "missing_group:pixel_cloud"
                        break
                    group = nc.groups["pixel_cloud"]
                    if "latitude" not in group.variables or "longitude" not in group.variables:
                        status, detail = "error", "missing_vars:latitude_or_longitude"
                        break
                    n = int(group.variables["latitude"].shape[0])
                    _ = int(group.variables["longitude"].shape[0])
                    # Deep-read probes: touch head/middle/tail values to catch truncated files
                    probe_vars = [name for name in ["latitude", "longitude", "classification", "height", "cross_track"] if name in group.variables]
                    if n > 0 and probe_vars:
                        probe_idx = sorted(set([0, n // 2, max(0, n - 1)]))
                        for var_name in probe_vars:
                            var = group.variables[var_name]
                            for i in probe_idx:
                                _ = var[i]
                            tail_start = max(0, n - 16)
                            _ = var[tail_start:n]
            status, detail = "ok", ""
            break
        except Exception as exc:
            err_text = str(exc)
            if attempt < int(NETCDF_OPEN_RETRIES):
                time.sleep(float(NETCDF_OPEN_RETRY_SLEEP_SEC))
                continue
            lowered = err_text.lower()
            if any(
                key in lowered
                for key in [
                    "truncated",
                    "superblock",
                    "unable to open file",
                    "h5fopen",
                    "read failed",
                    "netcdf: hdf error",
                    "hdf error",
                    "errno -101",
                ]
            ):
                status, detail = "corrupt", err_text
            else:
                status, detail = "error", err_text
            break
    else:
        status, detail = "error", "unknown_error"

    with GLOBAL_NC_HEALTH_LOCK:
        GLOBAL_NC_HEALTH_CACHE[path_key] = {"sig": sig, "status": status, "detail": detail}
    return status, detail


def build_candidate_pixc_files_by_day(lake_geom, window_start, window_end, tiles_gdf, pixc_map):
    if lake_geom is None or lake_geom.is_empty or tiles_gdf is None or tiles_gdf.empty or not pixc_map:
        return {}
    try:
        possible_tiles = tiles_gdf.iloc[list(tiles_gdf.sindex.intersection(lake_geom.bounds))].copy()
    except Exception:
        possible_tiles = tiles_gdf.copy()
    possible_tiles = possible_tiles[possible_tiles.intersects(lake_geom)].copy()
    if possible_tiles.empty:
        return {}
    tile_field = "Name" if "Name" in possible_tiles.columns else ("NAME" if "NAME" in possible_tiles.columns else None)
    if tile_field is None:
        return {}

    candidate_files = []
    for tile_name in possible_tiles[tile_field].dropna().astype(str).tolist():
        tile_text = tile_name.strip()
        if tile_text:
            candidate_files.extend(pixc_map.get(tile_text, []))
    candidate_files = sorted(set(candidate_files), key=gi_priority_key)
    if not candidate_files:
        return {}

    by_day = defaultdict(list)
    for nc_path in candidate_files:
        day_dt = to_datetime_ymd(extract_pixc_date_str(nc_path))
        if pd.isna(day_dt):
            continue
        day_dt = day_dt.normalize()
        if pd.notna(window_start) and day_dt < window_start.normalize():
            continue
        if pd.notna(window_end) and day_dt > window_end.normalize():
            continue
        by_day[day_dt].append(nc_path)
    return dict(by_day)


def export_pixc_layer_to_gdb(pixc_gdf, gdb_path, layer_name):
    if pixc_gdf is None or pixc_gdf.empty:
        return False
    os.makedirs(os.path.dirname(gdb_path), exist_ok=True)
    for driver_name in ("OpenFileGDB", "FileGDB"):
        try:
            pixc_gdf.to_file(gdb_path, layer=layer_name, driver=driver_name)
            return True
        except Exception:
            continue
    return False


def process_lake_extract(task):
    lake_id = int(task["lake_id"])
    size_category = str(task["size_category"])
    lake_geom = task["lake_geom"]
    window_start = task["window_start"]
    window_end = task["window_end"]
    tiles_gdf = task["tiles_gdf"]
    pixc_map_d = task["pixc_map_d"]
    pixc_gdb_path_d = task["pixc_gdb_path_d"]
    done_keys = task.get("done_keys", set())

    rows = []
    candidate_granule_count = 0
    written_layer_count = 0
    existing_layer_count = 0
    empty_clip_count = 0
    failed_count = 0
    resume_skipped_count = 0
    status = "ok"

    if lake_geom is None or lake_geom.is_empty:
        return {
            "lake_id": lake_id,
            "size_category": size_category,
            "status": "missing_lake_geom",
            "rows": rows,
            "candidate_granule_count": 0,
            "written_layer_count": 0,
            "existing_layer_count": 0,
            "empty_clip_count": 0,
            "failed_count": 0,
        }

    by_day_d = task.get("candidate_by_day_d")
    if by_day_d is None:
        by_day_d = build_candidate_pixc_files_by_day(lake_geom, window_start, window_end, tiles_gdf, pixc_map_d)
    if not by_day_d:
        return {
            "lake_id": lake_id,
            "size_category": size_category,
            "status": "no_pixc_candidates",
            "rows": rows,
            "candidate_granule_count": 0,
            "written_layer_count": 0,
            "existing_layer_count": 0,
            "empty_clip_count": 0,
            "failed_count": 0,
        }

    existing_layers = set()
    if os.path.exists(pixc_gdb_path_d):
        try:
            import fiona
            existing_layers = set(fiona.listlayers(pixc_gdb_path_d))
        except Exception:
            existing_layers = set()

    per_version_stats = {
        "D": {"candidate_granule_count": 0, "written_layer_count": 0, "existing_layer_count": 0, "failed_count": 0},
    }

    def iter_by_day(by_day):
        for day_dt in sorted(by_day.keys()):
            for nc_path in by_day[day_dt]:
                yield day_dt, nc_path

    for pixc_version, by_day, pixc_gdb_path in (("D", by_day_d, pixc_gdb_path_d),):
        for day_dt, nc_path in iter_by_day(by_day):
            day_text = day_dt.strftime("%Y%m%d")
            candidate_granule_count += 1
            per_version_stats[pixc_version]["candidate_granule_count"] += 1
            granule_name = os.path.basename(str(nc_path))
            layer_name = build_pixc_record_id(nc_path)
            layer_name_legacy = build_legacy_pixc_record_id(nc_path)
            granule_info = parse_pixc_filename(granule_name) or {}
            gi_code = str(granule_info.get("gi_code", "")).upper()
            granule_match_key = build_granule_match_key(nc_path)
            resume_key = (int(lake_id), str(pixc_version), str(granule_name))
            if resume_key in done_keys:
                resume_skipped_count += 1
                if VERBOSE_PROGRESS:
                    progress(f"[06_1] lake {lake_id} resume_skip {pixc_version} {granule_name}")
                continue
            if PRECHECK_NC_BEFORE_EXTRACT:
                health_status, health_detail = check_nc_health(nc_path)
                if health_status != "ok":
                    failed_count += 1
                    per_version_stats[pixc_version]["failed_count"] += 1
                    status = "partial_failed"
                    rows.append(
                        {
                            "lake_id": lake_id,
                            "size_category": size_category,
                            "date": day_text,
                            "granule_name": granule_name,
                            "layer_name": layer_name,
                            "status": "precheck_failed",
                            "point_count": pd.NA,
                            "gdb_path": pixc_gdb_path,
                            "gdb_layer": "",
                            "repair_status": str(GLOBAL_REPAIR_STATUS.get(granule_name, {}).get("status", "")),
                            "detail": health_detail,
                            "pixc_version": pixc_version,
                            "gi_code": gi_code,
                            "granule_match_key": granule_match_key,
                        }
                    )
                    continue
            existing_layer_name = ""
            if layer_name in existing_layers:
                existing_layer_name = layer_name
            elif layer_name_legacy in existing_layers:
                existing_layer_name = layer_name_legacy
            if existing_layer_name:
                existing_layer_count += 1
                per_version_stats[pixc_version]["existing_layer_count"] += 1
                rows.append(
                    {
                        "lake_id": lake_id,
                        "size_category": size_category,
                        "date": day_text,
                        "granule_name": granule_name,
                        "layer_name": layer_name,
                        "status": "existing_layer",
                        "point_count": pd.NA,
                        "gdb_path": pixc_gdb_path,
                        "gdb_layer": existing_layer_name,
                        "repair_status": "",
                        "detail": "",
                        "pixc_version": pixc_version,
                        "gi_code": gi_code,
                        "granule_match_key": granule_match_key,
                    }
                )
                continue

            parsed_gdf, read_status, read_detail = extract_data_from_nc(nc_path, lake_geom)
            repair_status = ""

            if read_status == "ok" and parsed_gdf is not None and not parsed_gdf.empty:
                saved = export_pixc_layer_to_gdb(parsed_gdf, pixc_gdb_path, layer_name)
                if saved:
                    existing_layers.add(layer_name)
                    written_layer_count += 1
                    per_version_stats[pixc_version]["written_layer_count"] += 1
                    if VERBOSE_PROGRESS:
                        progress(f"[06_1] lake {lake_id} export_pixc_gdb_{pixc_version} {layer_name} ok n={len(parsed_gdf)}")
                    rows.append(
                        {
                            "lake_id": lake_id,
                            "size_category": size_category,
                            "date": day_text,
                            "granule_name": granule_name,
                            "layer_name": layer_name,
                            "status": "written",
                            "point_count": int(len(parsed_gdf)),
                            "gdb_path": pixc_gdb_path,
                            "gdb_layer": layer_name,
                            "repair_status": repair_status,
                            "detail": "",
                            "pixc_version": pixc_version,
                            "gi_code": gi_code,
                            "granule_match_key": granule_match_key,
                        }
                    )
                else:
                    failed_count += 1
                    per_version_stats[pixc_version]["failed_count"] += 1
                    rows.append(
                        {
                            "lake_id": lake_id,
                            "size_category": size_category,
                            "date": day_text,
                            "granule_name": granule_name,
                            "layer_name": layer_name,
                            "status": "write_failed",
                            "point_count": pd.NA,
                            "gdb_path": pixc_gdb_path,
                            "gdb_layer": layer_name,
                            "repair_status": repair_status,
                            "detail": "gdb_write_failed",
                            "pixc_version": pixc_version,
                            "gi_code": gi_code,
                            "granule_match_key": granule_match_key,
                        }
                    )
            elif read_status == "empty":
                empty_clip_count += 1
                rows.append(
                    {
                        "lake_id": lake_id,
                        "size_category": size_category,
                        "date": day_text,
                        "granule_name": granule_name,
                        "layer_name": layer_name,
                        "status": "empty_after_lake_clip",
                        "point_count": 0,
                        "gdb_path": pixc_gdb_path,
                        "gdb_layer": "",
                        "repair_status": repair_status,
                        "detail": "",
                        "pixc_version": pixc_version,
                        "gi_code": gi_code,
                        "granule_match_key": granule_match_key,
                    }
                )
            else:
                failed_count += 1
                per_version_stats[pixc_version]["failed_count"] += 1
                status = "partial_failed"
                rows.append(
                    {
                        "lake_id": lake_id,
                        "size_category": size_category,
                        "date": day_text,
                        "granule_name": granule_name,
                        "layer_name": layer_name,
                        "status": "read_failed",
                        "point_count": pd.NA,
                        "gdb_path": pixc_gdb_path,
                        "gdb_layer": "",
                        "repair_status": repair_status,
                        "detail": read_detail,
                        "pixc_version": pixc_version,
                        "gi_code": gi_code,
                        "granule_match_key": granule_match_key,
                    }
                )

    return {
        "lake_id": lake_id,
        "size_category": size_category,
        "status": status,
        "rows": rows,
        "candidate_granule_count": int(candidate_granule_count),
        "written_layer_count": int(written_layer_count),
        "existing_layer_count": int(existing_layer_count),
        "empty_clip_count": int(empty_clip_count),
        "failed_count": int(failed_count),
        "candidate_granule_count_d": int(per_version_stats["D"]["candidate_granule_count"]),
        "written_layer_count_d": int(per_version_stats["D"]["written_layer_count"]),
        "existing_layer_count_d": int(per_version_stats["D"]["existing_layer_count"]),
        "failed_count_d": int(per_version_stats["D"]["failed_count"]),
        "resume_skipped_count": int(resume_skipped_count),
    }


def ensure_columns(df, columns):
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = pd.NA
    return work[columns].copy()


def load_existing_index_df(index_csv_path):
    if not RESUME_FROM_INDEX:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    if not os.path.exists(index_csv_path):
        return pd.DataFrame(columns=INDEX_COLUMNS)
    try:
        old_df = pd.read_csv(index_csv_path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    if old_df is None or old_df.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return ensure_columns(old_df, INDEX_COLUMNS)


def load_existing_summary_df(summary_csv_path):
    if not LAKE_RESUME_FROM_SUMMARY:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    if not os.path.exists(summary_csv_path):
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    try:
        old_df = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    if old_df is None or old_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return ensure_columns(old_df, SUMMARY_COLUMNS)


def build_completed_lake_id_set(summary_df):
    if summary_df is None or summary_df.empty:
        return set()
    work = summary_df.copy()
    work["lake_id"] = pd.to_numeric(work.get("lake_id"), errors="coerce").astype("Int64")
    work["status"] = work.get("status", "").astype(str)
    work = work[
        work["lake_id"].notna()
        & work["status"].isin(LAKE_RESUME_SKIP_STATUSES)
    ].copy()
    if work.empty:
        return set()
    return set(work["lake_id"].astype(int).tolist())


def build_resume_done_key_set(index_df):
    if index_df is None or index_df.empty:
        return set()
    work = index_df.copy()
    work["lake_id"] = pd.to_numeric(work.get("lake_id"), errors="coerce").astype("Int64")
    work["pixc_version"] = work.get("pixc_version", "").astype(str).str.upper()
    work["granule_name"] = work.get("granule_name", "").astype(str)
    work["status"] = work.get("status", "").astype(str)
    work = work[
        work["lake_id"].notna()
        & work["pixc_version"].str.len().gt(0)
        & work["granule_name"].str.len().gt(0)
        & work["status"].isin(RESUME_TERMINAL_STATUSES)
    ].copy()
    if work.empty:
        return set()
    return set(
        (int(row.lake_id), str(row.pixc_version), str(row.granule_name))
        for row in work.itertuples(index=False)
    )


def merge_index_rows(existing_index_df, new_index_df):
    if existing_index_df is None or existing_index_df.empty:
        return ensure_columns(new_index_df, INDEX_COLUMNS)
    if new_index_df is None or new_index_df.empty:
        return ensure_columns(existing_index_df, INDEX_COLUMNS)
    old_df = ensure_columns(existing_index_df, INDEX_COLUMNS)
    add_df = ensure_columns(new_index_df, INDEX_COLUMNS)
    merged = pd.concat([old_df, add_df], ignore_index=True)
    merged["lake_id"] = pd.to_numeric(merged.get("lake_id"), errors="coerce")
    merged["pixc_version"] = merged.get("pixc_version", "").astype(str).str.upper()
    merged["granule_name"] = merged.get("granule_name", "").astype(str)
    merged["layer_name"] = merged.get("layer_name", "").astype(str)
    merged["date"] = merged.get("date", "").astype(str)
    merged = merged.drop_duplicates(
        subset=["lake_id", "pixc_version", "granule_name", "layer_name", "date"],
        keep="last",
    ).copy()
    return ensure_columns(merged, INDEX_COLUMNS)


def merge_summary_rows(existing_summary_df, new_summary_df):
    if existing_summary_df is None or existing_summary_df.empty:
        return ensure_columns(new_summary_df, SUMMARY_COLUMNS)
    if new_summary_df is None or new_summary_df.empty:
        return ensure_columns(existing_summary_df, SUMMARY_COLUMNS)
    old_df = ensure_columns(existing_summary_df, SUMMARY_COLUMNS)
    add_df = ensure_columns(new_summary_df, SUMMARY_COLUMNS)
    merged = pd.concat([old_df, add_df], ignore_index=True)
    merged["lake_id"] = pd.to_numeric(merged.get("lake_id"), errors="coerce")
    merged = merged.dropna(subset=["lake_id"]).copy()
    merged["lake_id"] = merged["lake_id"].astype(int)
    merged = merged.drop_duplicates(subset=["lake_id"], keep="last").copy()
    return ensure_columns(merged.sort_values("lake_id").reset_index(drop=True), SUMMARY_COLUMNS)


def persist_method_outputs(method_paths, existing_index_df, existing_summary_df, all_index_rows, summary_rows, save_daily=False):
    new_index_df = ensure_columns(pd.DataFrame(all_index_rows), INDEX_COLUMNS)
    index_df = merge_index_rows(existing_index_df, new_index_df)
    new_summary_df = ensure_columns(pd.DataFrame(summary_rows), SUMMARY_COLUMNS)
    summary_df = merge_summary_rows(existing_summary_df, new_summary_df)
    index_df.to_csv(method_paths["index_csv"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(method_paths["summary_csv"], index=False, encoding="utf-8-sig")
    return index_df, summary_df


def build_cd_diff_tables(index_df):
    # D-only mode: keep function for compatibility, but no longer generate D/C diff tables.
    return pd.DataFrame(), pd.DataFrame()


def extract_date_from_layer_name(layer_name):
    text = str(layer_name)
    match = re.match(r"^pixc_(\d{8})T\d{6}_", text)
    return match.group(1) if match else ""


def list_lake_ids_from_gdb_dirs(gdb_dir_d):
    out = set()
    for folder in [gdb_dir_d]:
        if not os.path.exists(folder):
            continue
        for name in os.listdir(folder):
            lower = str(name).lower()
            if not lower.endswith(".gdb"):
                continue
            m = re.match(r"^swot_pixc_(\d+)\.gdb$", str(name), re.IGNORECASE)
            if m:
                out.add(int(m.group(1)))
    return sorted(out)


def list_layers_safe(gdb_path):
    if not os.path.exists(gdb_path):
        return []
    try:
        import fiona
        return list(fiona.listlayers(gdb_path))
    except Exception:
        return []


def build_daily_cd_compare_from_gdb(method_paths):
    gdb_dir_d = method_paths["pixc_gdb_dir"]
    lake_ids = list_lake_ids_from_gdb_dirs(gdb_dir_d)
    if not lake_ids:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for lake_id in lake_ids:
        path_d = os.path.join(gdb_dir_d, f"swot_pixc_{int(lake_id)}.gdb")
        layers_d = list_layers_safe(path_d)
        day_count_d = {}
        for lyr in layers_d:
            day = extract_date_from_layer_name(lyr)
            if day:
                day_count_d[day] = int(day_count_d.get(day, 0) + 1)
        all_days = sorted(set(day_count_d.keys()))
        for day in all_days:
            granules_d = int(day_count_d.get(day, 0))
            has_d = granules_d > 0
            coverage_class = "only_d" if has_d else "none"
            rows.append(
                {
                    "lake_id": int(lake_id),
                    "date": str(day),
                    "granules_d": granules_d,
                    "has_d": bool(has_d),
                    "coverage_class": coverage_class,
                }
            )
    daily_df = pd.DataFrame(rows)
    if daily_df.empty:
        return daily_df, pd.DataFrame()

    stats_df = (
        daily_df.groupby("lake_id", as_index=False)
        .agg(
            total_days=("date", "nunique"),
            only_d_days=("coverage_class", lambda s: int((s == "only_d").sum())),
            none_days=("coverage_class", lambda s: int((s == "none").sum())),
            total_granules_d=("granules_d", "sum"),
        )
        .sort_values("lake_id")
        .reset_index(drop=True)
    )
    daily_df = daily_df.sort_values(["lake_id", "date"]).reset_index(drop=True)
    return daily_df, stats_df


def build_lake_windows_from_existing_outputs(method_paths):
    # Fallback mode when 05_2 topology CSV is missing: infer lake windows from existing outputs.
    summary_map = {}
    if os.path.exists(method_paths["summary_csv"]):
        try:
            old_summary = pd.read_csv(method_paths["summary_csv"], encoding="utf-8-sig")
            if "lake_id" in old_summary.columns:
                old_summary["lake_id"] = pd.to_numeric(old_summary["lake_id"], errors="coerce")
                old_summary = old_summary.dropna(subset=["lake_id"]).copy()
                old_summary["lake_id"] = old_summary["lake_id"].astype(int)
                if "size_category" in old_summary.columns:
                    summary_map = {
                        int(row.lake_id): str(row.size_category)
                        for row in old_summary[["lake_id", "size_category"]].itertuples(index=False)
                    }
        except Exception:
            summary_map = {}

    index_df = load_existing_index_df(method_paths["index_csv"])
    lake_ids = list_lake_ids_from_gdb_dirs(method_paths["pixc_gdb_dir"])
    if not lake_ids and index_df is not None and not index_df.empty and "lake_id" in index_df.columns:
        vals = pd.to_numeric(index_df["lake_id"], errors="coerce").dropna().astype(int).tolist()
        lake_ids = sorted(set(vals))
    if not lake_ids:
        return pd.DataFrame(columns=["lake_id", "water_period_start", "water_period_end", "size_category"])

    rows = []
    for lake_id in lake_ids:
        day_list = []
        if index_df is not None and not index_df.empty:
            sub = index_df[pd.to_numeric(index_df.get("lake_id"), errors="coerce") == int(lake_id)].copy()
            if not sub.empty and "date" in sub.columns:
                day_vals = sub["date"].astype(str).str.extract(r"(\d{8})", expand=False).dropna().tolist()
                day_list.extend(day_vals)
        if not day_list:
            gdb_path = os.path.join(method_paths["pixc_gdb_dir"], f"swot_pixc_{int(lake_id)}.gdb")
            for layer_name in list_layers_safe(gdb_path):
                day_text = extract_date_from_layer_name(layer_name)
                if day_text:
                    day_list.append(day_text)
        day_list = sorted(set([str(x) for x in day_list if str(x)]))
        if not day_list:
            continue
        rows.append(
            {
                "lake_id": int(lake_id),
                "water_period_start": day_list[0],
                "water_period_end": day_list[-1],
                "size_category": summary_map.get(int(lake_id), "Unknown"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["lake_id", "water_period_start", "water_period_end", "size_category"])
    return pd.DataFrame(rows).sort_values("lake_id").reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_lock_path = os.path.join(OUTPUT_DIR, "06_1_extract_raw_wse.lock")
    if not acquire_run_lock(run_lock_path):
        return
    for method_name in METHOD_NAMES:
        method_paths = get_method_paths(method_name)
        os.makedirs(method_paths["output_dir"], exist_ok=True)
        os.makedirs(method_paths["pixc_gdb_dir"], exist_ok=True)
        existing_index_df = load_existing_index_df(method_paths["index_csv"])
        existing_summary_df = load_existing_summary_df(method_paths["summary_csv"])
        completed_lake_ids = build_completed_lake_id_set(existing_summary_df)
        if completed_lake_ids:
            progress(f"[06_1] {method_name}: lake-resume enabled, completed lakes={len(completed_lake_ids)}")
        resume_done_keys = build_resume_done_key_set(existing_index_df)
        if resume_done_keys:
            progress(f"[06_1] {method_name}: resume enabled, loaded {len(resume_done_keys)} done granules")

        topology_csv_path = ""
        if os.path.exists(method_paths["input_05_2_topology_csv"]):
            topology_csv_path = method_paths["input_05_2_topology_csv"]
        elif os.path.exists(method_paths.get("input_05_2_topology_csv_alt", "")):
            topology_csv_path = method_paths["input_05_2_topology_csv_alt"]
        topology_mode = bool(topology_csv_path)
        if topology_mode:
            topology_df = load_topology_records(topology_csv_path)
            if topology_df.empty:
                progress(f"[06_1] skip {method_name}: empty topology records")
                continue
            lake_windows_df = build_lake_windows(topology_df)
            if PROCESS_ALL_LAKES:
                before_count = int(len(lake_windows_df))
                lake_geom_map = load_lake_buffered_geom_map(LAKES_BUFFERED_SHP)
                lake_windows_df = expand_lake_windows_to_all_lakes(lake_windows_df, lake_geom_map, topology_df=topology_df)
                after_count = int(len(lake_windows_df))
                if after_count > before_count:
                    progress(
                        f"[06_1] {method_name}: expand to all lakes enabled, "
                        f"added={after_count - before_count}, total={after_count}"
                    )
            if TARGET_SIZE_CATEGORIES:
                lake_windows_df = lake_windows_df[
                    lake_windows_df["size_category"].astype(str).isin(TARGET_SIZE_CATEGORIES)
                ].copy()
        else:
            progress(f"[06_1] {method_name}: missing 05_2 topology csv, fallback to existing outputs")
            lake_windows_df = build_lake_windows_from_existing_outputs(method_paths)
            if lake_windows_df.empty:
                progress(f"[06_1] skip {method_name}: fallback windows empty")
                continue
            if PROCESS_ALL_LAKES:
                before_count = int(len(lake_windows_df))
                lake_geom_map = load_lake_buffered_geom_map(LAKES_BUFFERED_SHP)
                lake_windows_df = expand_lake_windows_to_all_lakes(lake_windows_df, lake_geom_map, topology_df=None)
                after_count = int(len(lake_windows_df))
                if after_count > before_count:
                    progress(
                        f"[06_1] {method_name}: expand to all lakes enabled, "
                        f"added={after_count - before_count}, total={after_count}"
                    )

        allowed_lake_ids = load_allowed_lake_ids_from_shp(LAKE_FILTER_SHP)
        if allowed_lake_ids:
            before_count = int(len(lake_windows_df))
            lake_windows_df = lake_windows_df[
                pd.to_numeric(lake_windows_df["lake_id"], errors="coerce").astype("Int64").isin(sorted(allowed_lake_ids))
            ].copy()
            progress(
                f"[06_1] {method_name}: apply lake-id filter from Water_Max_Filtered_Polygons.shp "
                f"{before_count}->{len(lake_windows_df)}"
            )

        tiles_gdf = load_tiles_gdf(TILES_SHP)
        lake_geom_map = load_lake_buffered_geom_map(LAKES_BUFFERED_SHP)
        pixc_map_d = build_pixc_file_map(PIXC_DIR_D)

        existing_state = load_repair_state(method_paths["repair_log_csv"])
        with GLOBAL_REPAIR_LOCK:
            GLOBAL_REPAIR_STATUS.clear()
            GLOBAL_REPAIR_STATUS.update(existing_state)
            GLOBAL_REPAIR_ROWS.clear()
        with GLOBAL_NC_HEALTH_LOCK:
            GLOBAL_NC_HEALTH_CACHE.clear()

        tasks = []
        skipped_lake_count = 0
        skipped_by_lake_id_cap = 0
        for row in lake_windows_df.to_dict("records"):
            lake_id = int(row["lake_id"])
            if int(lake_id) > int(MAX_LAKE_ID_TO_PROCESS):
                skipped_by_lake_id_cap += 1
                continue
            if lake_id in completed_lake_ids:
                skipped_lake_count += 1
                continue
            tasks.append(
                {
                    "lake_id": lake_id,
                    "size_category": str(row["size_category"]),
                    "lake_geom": lake_geom_map.get(lake_id),
                    "window_start": to_datetime_ymd(row["water_period_start"]),
                    "window_end": to_datetime_ymd(row["water_period_end"]),
                    "tiles_gdf": tiles_gdf,
                    "pixc_map_d": pixc_map_d,
                    # Keep D in legacy directory for downstream compatibility.
                    "pixc_gdb_path_d": os.path.join(method_paths["pixc_gdb_dir"], f"swot_pixc_{lake_id}.gdb"),
                    "done_keys": resume_done_keys,
                }
            )
        if skipped_lake_count > 0:
            progress(f"[06_1] {method_name}: skipped completed lakes={skipped_lake_count}")
        if skipped_by_lake_id_cap > 0:
            progress(
                f"[06_1] {method_name}: skipped by lake_id cap (> {MAX_LAKE_ID_TO_PROCESS})="
                f"{skipped_by_lake_id_cap}"
            )

        run_precheck = bool(RUN_PHASE1_PRECHECK and PRECHECK_NC_BEFORE_EXTRACT)
        if run_precheck and RESUME_SKIP_PRECHECK_WHEN_INDEX_EXISTS and len(resume_done_keys) > 0:
            run_precheck = False
            progress(f"[06_1] {method_name}: phase-1 precheck skipped (resume mode)")

        if run_precheck:
            progress(f"[06_1] {method_name}: phase-1 precheck start")
            all_candidates = set()
            for task in tasks:
                by_day_d = build_candidate_pixc_files_by_day(
                    task["lake_geom"],
                    task["window_start"],
                    task["window_end"],
                    task["tiles_gdf"],
                    task["pixc_map_d"],
                )
                task["candidate_by_day_d"] = by_day_d
                for day_dt in by_day_d:
                    for nc_path in by_day_d[day_dt]:
                        all_candidates.add(str(nc_path))

            precheck_total = len(all_candidates)
            progress(f"[06_1] {method_name}: phase-1 precheck candidates={precheck_total}")
            checked = 0
            repaired_ok_count = 0
            precheck_bad_count = 0
            for nc_path in sorted(all_candidates):
                checked += 1
                granule_name = os.path.basename(str(nc_path))
                day_text = extract_pixc_date_str(granule_name)
                health_status, health_detail = check_nc_health(nc_path)
                if health_status == "corrupt" and ENABLE_PIXC_AUTO_REPAIR:
                    known = GLOBAL_REPAIR_STATUS.get(granule_name, {})
                    known_status = str(known.get("status", "")).strip()
                    repair_status = known_status
                    if not (SKIP_RETRY_FOR_KNOWN_FAILED_REPAIR and known_status and known_status != "repaired_downloaded"):
                        if VERBOSE_PROGRESS:
                            progress(f"[06_1] precheck {checked}/{precheck_total} {granule_name} -> start_auto_repair")
                        ok, repair_status, repaired_path = attempt_repair_corrupt_pixc(nc_path, tiles_gdf)
                        record_repair_row(-1, day_text, granule_name, repair_status, health_detail, repaired_path)
                        if ok:
                            repaired_ok_count += 1
                            health_status, health_detail = check_nc_health(nc_path)
                    if VERBOSE_PROGRESS:
                        progress(f"[06_1] precheck {checked}/{precheck_total} {granule_name} -> {repair_status}")
                if health_status != "ok":
                    precheck_bad_count += 1
                    progress(
                        f"[06_1] precheck_bad {checked}/{precheck_total} {granule_name} "
                        f"status={health_status} detail={str(health_detail)[:160]}"
                    )
                elif VERBOSE_PROGRESS and checked % 200 == 0:
                    progress(f"[06_1] precheck progress {checked}/{precheck_total}")

            progress(
                f"[06_1] {method_name}: phase-1 precheck done checked={precheck_total} "
                f"bad={precheck_bad_count} repaired_ok={repaired_ok_count}"
            )

        if not RUN_PHASE2_EXTRACT:
            progress(f"[06_1] {method_name}: phase-2 extract disabled, precheck-only run completed")
            continue

        total = len(tasks)
        progress(f"[06_1] {method_name}: phase-2 extract start for {total} lakes with {MAX_WORKERS} workers")
        if total == 0:
            progress(f"[06_1] {method_name}: no pending lakes after lake-resume filter")

        all_index_rows = []
        summary_rows = []
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, int(MAX_WORKERS))) as executor:
            future_map = {executor.submit(process_lake_extract, task): task for task in tasks}
            for future in as_completed(future_map):
                done += 1
                task = future_map[future]
                lake_id = int(task["lake_id"])
                try:
                    result = future.result()
                except Exception as exc:
                    summary_rows.append(
                        {
                            "lake_id": lake_id,
                            "size_category": str(task["size_category"]),
                            "status": "error",
                            "candidate_granule_count": 0,
                            "written_layer_count": 0,
                            "existing_layer_count": 0,
                            "empty_clip_count": 0,
                            "failed_count": 1,
                        }
                    )
                    progress(f"[06_1] {method_name} lake {done}/{total}: {lake_id} error: {exc}")
                    if UPDATE_CD_DIFF_EACH_LAKE:
                        persist_method_outputs(
                            method_paths,
                            existing_index_df,
                            existing_summary_df,
                            all_index_rows,
                            summary_rows,
                            save_daily=bool(UPDATE_DAILY_COMPARE_EACH_LAKE),
                        )
                        progress(f"[06_1] {method_name} lake {done}/{total}: incremental cd-diff updated (error case)")
                    continue

                all_index_rows.extend(result["rows"])
                summary_rows.append(
                    {
                        "lake_id": int(result["lake_id"]),
                        "size_category": str(result["size_category"]),
                        "status": str(result["status"]),
                        "candidate_granule_count": int(result["candidate_granule_count"]),
                        "written_layer_count": int(result["written_layer_count"]),
                        "existing_layer_count": int(result["existing_layer_count"]),
                        "empty_clip_count": int(result["empty_clip_count"]),
                        "failed_count": int(result["failed_count"]),
                        "candidate_granule_count_d": int(result.get("candidate_granule_count_d", 0)),
                        "written_layer_count_d": int(result.get("written_layer_count_d", 0)),
                        "existing_layer_count_d": int(result.get("existing_layer_count_d", 0)),
                        "failed_count_d": int(result.get("failed_count_d", 0)),
                        "resume_skipped_count": int(result.get("resume_skipped_count", 0)),
                    }
                )
                progress(
                    f"[06_1] {method_name} lake {done}/{total}: {lake_id} "
                    f"status={result['status']} written={result['written_layer_count']} "
                    f"existing={result['existing_layer_count']} resume_skipped={result.get('resume_skipped_count', 0)}"
                )
                if UPDATE_CD_DIFF_EACH_LAKE:
                    persist_method_outputs(
                        method_paths,
                        existing_index_df,
                        existing_summary_df,
                        all_index_rows,
                        summary_rows,
                        save_daily=bool(UPDATE_DAILY_COMPARE_EACH_LAKE),
                    )
                    progress(f"[06_1] {method_name} lake {done}/{total}: incremental cd-diff updated")

        index_df, summary_df = persist_method_outputs(
            method_paths,
            existing_index_df,
            existing_summary_df,
            all_index_rows,
            summary_rows,
            save_daily=True,
        )

        with GLOBAL_REPAIR_LOCK:
            repair_rows = list(GLOBAL_REPAIR_ROWS)
        if repair_rows:
            repair_df = pd.DataFrame(repair_rows)
            if os.path.exists(method_paths["repair_log_csv"]):
                try:
                    old_df = pd.read_csv(method_paths["repair_log_csv"], encoding="utf-8-sig")
                    repair_df = pd.concat([old_df, repair_df], ignore_index=True)
                except Exception:
                    pass
            if "granule_name" in repair_df.columns and "status" in repair_df.columns:
                repair_df = repair_df.drop_duplicates(subset=["granule_name", "status"], keep="last").copy()
            repair_df = ensure_columns(repair_df, PIXC_REPAIR_LOG_COLUMNS)
            repair_df.to_csv(method_paths["repair_log_csv"], index=False, encoding="utf-8-sig")

        progress(f"[06_1] completed {method_name}")


if __name__ == "__main__":
    main()
