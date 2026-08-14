"""
Stage 06_2: Compute raw WSE from Stage 06_1 PIXC GDB outputs using Stage 05 branch seeds.
"""

import os
import re
import importlib.util
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import timedelta
from pathlib import Path

try:
    import fiona
except Exception:
    fiona = None
import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from shapely.geometry import box
try:
    import earthaccess
except Exception:  # pragma: no cover
    earthaccess = None


def list_gdb_layers(gdb_path):
    if fiona is not None:
        try:
            return list(fiona.listlayers(gdb_path))
        except Exception:
            pass
    try:
        import pyogrio

        layers_df = pyogrio.list_layers(gdb_path)
        if hasattr(layers_df, "columns") and "name" in layers_df.columns:
            return [str(x) for x in layers_df["name"].tolist()]
        return [str(x[0]) for x in layers_df]
    except Exception:
        return []


PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT_DIR = r"X:\\2024_西南水位曲线\\result1"
INPUT_06_1_ROOT = os.path.join(OUTPUT_ROOT_DIR, "06_1_extract_swot_pixc_to_gdb")
PIXC_DIR = r"G:\SWOT\Data\PIXC"
TILES_SHP = r"D:\Haoyu_space\03_data\SWOT_Tiles\Greenland_tiles.shp"
LAKES_BUFFERED_SHP = os.path.join(
    OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons_Buffered.shp"
)
INPUT_05_1_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_1_build_topology_metrics_from_04_6")
INPUT_05_2_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_2_plot_singlebranch_from_05_1_04_6_1")
INPUT_05_2_BRANCH_SEED_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_2_plot_singlebranch_from_05_1_04_6_1")
INPUT_04_6_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_3_export_effective_boundaries")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_2_extract_raw_wse")

METHOD_NAMES = ["Otsu"]
TARGET_SIZE_CATEGORIES = ["Large", "Medium"]
TARGET_LAKE_IDS = []
LAKE_WORKERS = 2
DAY_WORKERS = 2
DAY_BATCH_SIZE = 5
MAX_PIXC_CACHE_ITEMS = 8
MAX_LAYER_CACHE_ITEMS = 128
MAX_WORKERS = LAKE_WORKERS
RESUME_ENABLED = True

RAW_WSE_COLUMNS = [
    "lake_id", "pixc_version", "date", "date_dt", "branch_index", "branch_id", "branch_label",
    "wse_m", "wse_std", "count", "type", "mask_area_km2", "mask_date",
    "mask_feature_index", "source_layer", "sigma_iters", "type_rank",
    "wse_stage", "sigma_filtered", "pre_filter_count", "post_filter_count",
    "mask_role", "mask_pick_mode", "cross_track_mode", "cross_track_relaxed",
    "plot_force_noise_pre_0605",
]

AUDIT_COLUMNS = [
    "lake_id", "pixc_version", "branch_index", "branch_id", "date", "date_dt", "mask_role",
    "status", "reason_short", "reason_detail", "candidate_count", "source_layers",
    "cross_track_mode", "cross_track_relaxed", "plot_force_noise_pre_0605",
]

SUMMARY_COLUMNS = [
    "lake_id", "pixc_version", "size_category", "status", "record_count", "branch_mask_count",
]

PIXC_VERSION_SPECS = [
    {"pixc_version": "D", "input_subdir": "06_1_swot_pixc_gdb_by_lake", "output_suffix": "D"},
    {"pixc_version": "C", "input_subdir": "06_1_swot_pixc_gdb_by_lake_C", "output_suffix": "C"},
]

DATE_PAD_BEFORE_DAYS = 5
DATE_PAD_AFTER_DAYS = 0
CUSTOM_LAKE_WINDOW_START = "20240601"
CUSTOM_LAKE_WINDOW_END = "20240815"
PLOT_FORCE_NOISE_BEFORE_DATE = "20240605"
MIN_EXTRACTION_MASK_AREA_KM2 = 0.05

BASE_FILTER = {
    "classification": [3, 4, 5, 6],
    "coh >= 0.8": True,
    "sig0 > 10": True,
    "abs(cross_track) > 10000": True,
    "abs(cross_track) < 60000": True,
}
BASE_FILTER_NO_CROSSTRACK = {
    "classification": [3, 4, 5, 6],
    "coh >= 0.8": True,
    "sig0 > 10": True,
}
ALLOW_CROSSTRACK_RELAXATION = True
GEO_QUAL_HIGH_THRESH = 16777216
BAD_BITS_MASK = 5
SIGMA_TARGET_METERS = 0.5
SIGMA_MAX_ITERS = 10
DIRECT_PIXC_PREFERRED = False
VERBOSE_PIXC_PROGRESS = True
NETCDF_OPEN_RETRIES = 2
NETCDF_OPEN_RETRY_SLEEP_SEC = 0.2
ENABLE_PIXC_AUTO_REPAIR = False
PIXC_REPAIR_REMOTE_RETRIES = 2
PIXC_REPAIR_SHORT_NAMES = ["SWOT_L2_HR_PIXC_D", "SWOT_L2_HR_PIXC"]
PIXC_REPAIR_LOG_COLUMNS = [
    "timestamp", "lake_id", "date", "granule_name", "status", "detail", "repaired_path",
]

GLOBAL_LAYER_CACHE = {}
GLOBAL_LAYER_CACHE_ORDER = []
GLOBAL_LAYER_CACHE_LOCK = threading.Lock()
GLOBAL_NETCDF_LOCK = threading.Lock()
GLOBAL_EARTHACCESS_LOCK = threading.Lock()
GLOBAL_EARTHACCESS_READY = False
GLOBAL_PIXC_REPAIR_LOCK = threading.Lock()
GLOBAL_PIXC_REPAIR_STATUS = {}
GLOBAL_PIXC_REPAIR_ROWS = []

USE_LEGACY_417_WSE_PICK = False
LEGACY_417_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "code_3_31_417" / "06_1_extract_raw_wse.py"
LEGACY_417_MODULE = None

# Optional compatibility override: force the main-branch extraction mask date for specific lakes.
# This is useful when validating new results against older runs that used an earlier seed polygon.
FORCE_MAIN_MASK_DATE_BY_LAKE = {}


def _parse_env_list(name):
    text = str(os.environ.get(name, "")).strip()
    if not text:
        return []
    out = []
    for tok in text.split(","):
        tok = str(tok).strip()
        if tok:
            out.append(tok)
    return out


def _parse_env_int_list(name):
    out = []
    for tok in _parse_env_list(name):
        try:
            out.append(int(tok))
        except Exception:
            continue
    return out


_env_target_lake_ids = _parse_env_int_list("TARGET_LAKE_IDS")
if _env_target_lake_ids:
    TARGET_LAKE_IDS = _env_target_lake_ids

_env_output_root_dir = str(os.environ.get("OUTPUT_ROOT_DIR", "")).strip()
if _env_output_root_dir:
    OUTPUT_ROOT_DIR = _env_output_root_dir
    INPUT_06_1_ROOT = os.path.join(OUTPUT_ROOT_DIR, "06_1_extract_swot_pixc_to_gdb")
    LAKES_BUFFERED_SHP = os.path.join(
        OUTPUT_ROOT_DIR, "02_Lake_Extraction", "04_Filtered_Lakes", "Water_Max_Filtered_Polygons_Buffered.shp"
    )
    INPUT_05_1_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_1_build_topology_metrics_from_04_6")
    INPUT_05_2_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_2_plot_singlebranch_from_05_1_04_6_1")
    INPUT_05_2_BRANCH_SEED_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_2_plot_singlebranch_from_05_1_04_6_1")
    INPUT_04_6_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_3_export_effective_boundaries")
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_2_extract_raw_wse")

_env_pixc_versions = [v.upper() for v in _parse_env_list("PIXC_VERSIONS")]
if _env_pixc_versions:
    _allowed = {"C", "D"}
    PIXC_VERSION_SPECS = [spec for spec in PIXC_VERSION_SPECS if str(spec.get("pixc_version", "")).upper() in _allowed and str(spec.get("pixc_version", "")).upper() in _env_pixc_versions]

_env_disable_resume = str(os.environ.get("DISABLE_06_2_RESUME", "")).strip().lower()
if _env_disable_resume in {"1", "true", "yes", "y"}:
    RESUME_ENABLED = False


def progress(message):
    print(message, flush=True)


def load_legacy_417_module():
    if not USE_LEGACY_417_WSE_PICK:
        return None
    global LEGACY_417_MODULE
    if LEGACY_417_MODULE is not None:
        return LEGACY_417_MODULE
    if not LEGACY_417_SCRIPT_PATH.exists():
        progress(f"[06_1] legacy 417 script missing: {LEGACY_417_SCRIPT_PATH}")
        return None
    try:
        spec = importlib.util.spec_from_file_location("legacy06_1_extract_raw_wse_417", str(LEGACY_417_SCRIPT_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        LEGACY_417_MODULE = module
        progress(f"[06_1] legacy 417 WSE picker enabled: {LEGACY_417_SCRIPT_PATH}")
        return LEGACY_417_MODULE
    except Exception as exc:
        progress(f"[06_1] failed to load legacy 417 module: {exc}")
        return None


def get_method_paths(method_name):
    input_05_1_dir = os.path.join(INPUT_05_1_DIR, method_name)
    input_05_2_dir = os.path.join(INPUT_05_2_DIR, method_name)
    output_dir = os.path.join(OUTPUT_DIR, method_name)
    return {
        "method_name": method_name,
        "input_05_1_topology_gdb": os.path.join(input_05_1_dir, "05_1_topology.gdb"),
        "input_05_2_topology_csv": os.path.join(input_05_2_dir, "05_2_denoised_curve_topology_relations.csv"),
        "input_05_2_branch_start_gdb": os.path.join(INPUT_05_2_BRANCH_SEED_DIR, method_name, "05_2_branch_start_features.gdb"),
        "input_05_2_branch_start_csv": os.path.join(INPUT_05_2_BRANCH_SEED_DIR, method_name, "05_2_branch_start_features.csv"),
        "input_04_6_index_csv": os.path.join(INPUT_04_6_DIR, method_name, "04_3_effective_boundary_index.csv"),
        "output_dir": output_dir,
        "raw_wse_csv": os.path.join(output_dir, "06_1_raw_wse_table.csv"),
        "audit_csv": os.path.join(output_dir, "06_1_day_audit_table.csv"),
        "summary_csv": os.path.join(output_dir, "06_1_raw_wse_summary.csv"),
        "lake_summary_csv": os.path.join(output_dir, "06_1_mask_lake_summary.csv"),
        "branch_table_csv": os.path.join(output_dir, "06_1_mask_branch_table.csv"),
        "mask_dir": os.path.join(output_dir, "06_1_extraction_masks"),
        "mask_gdb": os.path.join(output_dir, "06_1_extraction_masks.gdb"),
        "mask_gdb_layer": "extraction_masks",
        "raw_wse_gdb": os.path.join(output_dir, "06_1_raw_wse_results.gdb"),
        "raw_wse_gdb_layer": "raw_wse_masks",
        "pixc_repair_log_csv": os.path.join(output_dir, "06_1_pixc_repair_log.csv"),
        "per_lake_raw_dir": os.path.join(output_dir, "06_1_raw_wse_by_lake"),
        "per_lake_audit_dir": os.path.join(output_dir, "06_1_day_audit_by_lake"),
        "per_lake_summary_dir": os.path.join(output_dir, "06_1_summary_by_lake"),
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
    try:
        numeric = int(float(text))
        if len(str(numeric)) == 8:
            return str(numeric)
    except Exception:
        pass
    dt = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(dt) else dt.strftime("%Y%m%d")


def to_datetime_ymd(value):
    text = normalize_date_str(value)
    if not text:
        return pd.NaT
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def resolve_custom_window_start():
    return to_datetime_ymd(CUSTOM_LAKE_WINDOW_START)


def resolve_custom_window_end():
    return to_datetime_ymd(CUSTOM_LAKE_WINDOW_END)


def resolve_plot_force_noise_before_date():
    return to_datetime_ymd(PLOT_FORCE_NOISE_BEFORE_DATE)


def parse_node_key_text(value):
    text = str(value or "").strip()
    if not text:
        return "", "", None
    parts = text.split("|")
    if len(parts) != 3:
        return "", "", None
    date_text = normalize_date_str(parts[0])
    preview_name = str(parts[1]).strip()
    try:
        feature_index = int(parts[2])
    except Exception:
        feature_index = None
    return date_text, preview_name, feature_index


def format_node_key(node_key):
    if not node_key:
        return ""
    return f"{node_key[0]}|{node_key[1]}|{int(node_key[2])}"


def parse_text_list(value, sep=";"):
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(sep) if str(item).strip()]


def parse_node_key_list(value):
    text = str(value or "").strip()
    if not text:
        return []
    out = []
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        node_key = parse_node_key_text(token)
        if node_key[0] and node_key[2] is not None:
            out.append(node_key)
    return out


def parse_pixc_filename(filename):
    parts = str(filename).split("_")
    if len(parts) >= 8:
        return {
            "cycle": parts[4],
            "pass": parts[5],
            "tile": parts[6],
            "time": parts[7],
            "filename": str(filename),
        }
    return None


def build_pixc_record_id(nc_path):
    filename = os.path.basename(str(nc_path))
    info = parse_pixc_filename(filename)
    stem = os.path.splitext(filename)[0]
    if not info:
        return stem
    parts = stem.split("_")
    product_tag = parts[-1] if parts else "00"
    return f"pixc_{info['time']}_{info['pass']}_{info['tile']}_{product_tag}"


def extract_pixc_date_str(nc_path):
    info = parse_pixc_filename(os.path.basename(str(nc_path)))
    ts = info["time"] if info else ""
    date_str = ts[:8] if len(ts) >= 8 else None
    return date_str if date_str and date_str.isdigit() else None


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


def extract_data_from_nc(nc_path, target_geom):
    for attempt in range(int(NETCDF_OPEN_RETRIES) + 1):
        try:
            # NetCDF/HDF5 access is serialized to avoid thread-level open conflicts.
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

                    indices = np.where(bbox_mask)[0]
                    vars_to_read = [
                        "classification", "classification_qual", "sig0", "sig0_qual", "coherent_power",
                        "power_plus_y", "power_minus_y", "water_frac", "water_frac_uncert", "cross_track",
                        "inc", "geolocation_qual", "height", "geoid", "solid_earth_tide", "load_tide_fes",
                        "load_tide_got", "pole_tide", "dheight_dphase", "dlatitude_dphase",
                        "dlongitude_dphase", "phase_unwrapping_region",
                        "ancillary_surface_classification_flag", "interferogram_qual", "looks_to_efflooks",
                        "prior_water_prob", "phase_noise_std",
                    ]
                    data = {"latitude": lats[indices], "longitude": lons[indices]}
                    for var_name in vars_to_read:
                        if var_name in group.variables:
                            try:
                                values = group.variables[var_name][indices]
                                if np.ma.is_masked(values):
                                    values = values.filled(np.nan)
                                data[var_name] = values
                            except Exception:
                                data[var_name] = np.nan
                        else:
                            data[var_name] = np.nan

                    df = pd.DataFrame(data)
                    interferogram_abs = None
                    if "interferogram" in group.variables:
                        try:
                            ifgram_vals = group.variables["interferogram"][indices]
                            if np.ma.is_masked(ifgram_vals):
                                ifgram_vals = ifgram_vals.filled(np.nan)
                            if ifgram_vals.ndim == 2 and ifgram_vals.shape[1] == 2:
                                complex_vals = ifgram_vals[:, 0] + 1j * ifgram_vals[:, 1]
                                interferogram_abs = np.asarray(np.abs(complex_vals))
                            else:
                                interferogram_abs = np.asarray(np.abs(ifgram_vals))
                        except Exception:
                            interferogram_abs = None

                    for col in ["height", "pole_tide", "solid_earth_tide", "load_tide_fes", "load_tide_got", "geoid"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    df["wse"] = (
                        df["height"]
                        - df["pole_tide"]
                        - df["solid_earth_tide"]
                        - df["load_tide_fes"]
                        - df["load_tide_got"]
                        - df["geoid"]
                    )
                    df["coherent_power"] = pd.to_numeric(df["coherent_power"], errors="coerce")
                    df["power_plus_y"] = pd.to_numeric(df["power_plus_y"], errors="coerce")
                    df["power_minus_y"] = pd.to_numeric(df["power_minus_y"], errors="coerce")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        gamma_raw = df["coherent_power"] / np.sqrt(df["power_plus_y"] * df["power_minus_y"])
                        df["gamma"] = np.clip(gamma_raw, 0, 1)
                        df["Pcoh_db"] = 10 * np.log10(df["coherent_power"] + 1e-12)
                    if interferogram_abs is not None:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            coh_raw = interferogram_abs / np.sqrt(df["power_plus_y"].values * df["power_minus_y"].values)
                            df["coh"] = np.clip(coh_raw, 0, 1)
                    else:
                        df["coh"] = np.nan

                    df = df.drop(
                        columns=[
                            "solid_earth_tide", "load_tide_fes", "load_tide_got", "pole_tide",
                            "coherent_power", "power_plus_y", "power_minus_y",
                        ],
                        errors="ignore",
                    )
                    points_gdf = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                        crs="EPSG:4326",
                    )
                    clipped = points_gdf[points_gdf.within(target_geom)]
                    return (None, "empty", "") if clipped.empty else (clipped, "ok", "")
        except Exception as exc:
            err_text = str(exc)
            if attempt < int(NETCDF_OPEN_RETRIES):
                time.sleep(float(NETCDF_OPEN_RETRY_SLEEP_SEC))
                continue
            lowered = err_text.lower()
            if any(key in lowered for key in ["truncated", "superblock", "unable to open file", "h5fopen", "read failed"]):
                return None, "corrupt", err_text
            return None, "error", err_text
    return None, "error", "unknown_error"


def parse_swot_date(layer_name):
    match_full = re.search(r"_(\d{8}T\d{6})_", str(layer_name))
    if match_full:
        try:
            return pd.to_datetime(match_full.group(1), format="%Y%m%dT%H%M%S")
        except Exception:
            pass
    match = re.search(r"_(\d{8})T", str(layer_name))
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    match_simple = re.search(r"(\d{8})", str(layer_name))
    if match_simple:
        return pd.to_datetime(match_simple.group(1), format="%Y%m%d")
    return None


def apply_quality_filters(gdf, conditions):
    df = gdf.copy()
    for key, value in conditions.items():
        if isinstance(value, list):
            if key in df.columns:
                df = df[df[key].isin(value)]
            continue

        abs_match = re.match(r"abs\((.+?)\)\s*(>=|<=|==|!=|>|<)\s*(.+)", key)
        if abs_match:
            field = abs_match.group(1).strip()
            operator = abs_match.group(2)
            val_str = abs_match.group(3).strip()
            if field in df.columns:
                try:
                    threshold = float(val_str)
                except Exception:
                    threshold = val_str
                vals = df[field].abs()
                if operator == ">":
                    df = df[vals > threshold]
                elif operator == ">=":
                    df = df[vals >= threshold]
                elif operator == "<":
                    df = df[vals < threshold]
                elif operator == "<=":
                    df = df[vals <= threshold]
                elif operator == "==":
                    df = df[vals == threshold]
                elif operator == "!=":
                    df = df[vals != threshold]
            continue

        cmp_match = re.match(r"(.+?)\s*(>=|<=|==|!=|>|<)\s*(.+)", key)
        if cmp_match:
            field = cmp_match.group(1).strip()
            operator = cmp_match.group(2)
            val_str = cmp_match.group(3).strip()
            if field not in df.columns:
                continue
            try:
                threshold = float(val_str)
            except Exception:
                threshold = val_str
            if operator == ">":
                df = df[df[field] > threshold]
            elif operator == ">=":
                df = df[df[field] >= threshold]
            elif operator == "<":
                df = df[df[field] < threshold]
            elif operator == "<=":
                df = df[df[field] <= threshold]
            elif operator == "==":
                df = df[df[field] == threshold]
            elif operator == "!=":
                df = df[df[field] != threshold]
    return df


def iterative_sigma_filter(series, sigma=2.0, std_target=0.5, max_iters=10):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return vals, 0.0, 0

    curr = vals.copy()
    n_iters = 0
    init_std = float(curr.std()) if len(curr) > 1 else 0.0
    if np.isnan(init_std):
        init_std = 0.0
    force_first_pass = (len(curr) > 1) and (init_std < std_target) and (init_std > 0.0)

    for i in range(max_iters):
        if len(curr) <= 1:
            break
        mu = float(curr.mean())
        std = float(curr.std())
        if np.isnan(std):
            std = 0.0
        if std == 0.0:
            break
        if std <= std_target and not (force_first_pass and i == 0):
            break

        low = mu - sigma * std
        high = mu + sigma * std
        nxt = curr[(curr >= low) & (curr <= high)]
        n_iters = i + 1
        if len(nxt) == len(curr) or nxt.empty:
            break
        curr = nxt

    final_std = float(curr.std()) if len(curr) > 1 else 0.0
    if np.isnan(final_std):
        final_std = 0.0
    return curr, final_std, n_iters


def get_cached_layer(swot_gdb_path, layer_name):
    cache_key = (str(swot_gdb_path), str(layer_name))
    with GLOBAL_LAYER_CACHE_LOCK:
        if cache_key in GLOBAL_LAYER_CACHE:
            try:
                GLOBAL_LAYER_CACHE_ORDER.remove(cache_key)
            except ValueError:
                pass
            GLOBAL_LAYER_CACHE_ORDER.append(cache_key)
            return GLOBAL_LAYER_CACHE[cache_key]
    return None


def set_cached_layer(swot_gdb_path, layer_name, layer_value):
    cache_key = (str(swot_gdb_path), str(layer_name))
    with GLOBAL_LAYER_CACHE_LOCK:
        GLOBAL_LAYER_CACHE[cache_key] = layer_value
        try:
            GLOBAL_LAYER_CACHE_ORDER.remove(cache_key)
        except ValueError:
            pass
        GLOBAL_LAYER_CACHE_ORDER.append(cache_key)
        while len(GLOBAL_LAYER_CACHE_ORDER) > int(MAX_LAYER_CACHE_ITEMS):
            old_key = GLOBAL_LAYER_CACHE_ORDER.pop(0)
            GLOBAL_LAYER_CACHE.pop(old_key, None)


def load_topology_records(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df
    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work["feature_index"] = pd.to_numeric(work["feature_index"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date", "preview_name", "feature_index"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["feature_index"] = work["feature_index"].astype(int)
    work["date"] = work["date"].astype(str).map(normalize_date_str)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    work["area_km2"] = pd.to_numeric(work["area_km2"], errors="coerce")
    work["is_main_lineage"] = work["is_main_lineage"].map(
        lambda value: False if pd.isna(value) else str(value).strip().lower() in {"1", "true", "yes"}
    )
    work["node_key"] = work.apply(
        lambda row: format_node_key((row["date"], row["preview_name"], int(row["feature_index"]))),
        axis=1,
    )
    return work.sort_values(["lake_id", "date_dt", "preview_name", "feature_index"]).reset_index(drop=True)


def load_topology_features_gdf(gdb_path):
    if not os.path.exists(gdb_path):
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(gdb_path, layer="topology_features")
    if gdf.empty:
        return gdf
    work = gdf.copy()
    if "date" not in work.columns and "date_" in work.columns:
        work = work.rename(columns={"date_": "date"})
    if "feature_index" not in work.columns and "feature_idx" in work.columns:
        work = work.rename(columns={"feature_idx": "feature_index"})
    if "feature_index" not in work.columns and "source_feature_idx" in work.columns:
        work = work.rename(columns={"source_feature_idx": "feature_index"})
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work["feature_index"] = pd.to_numeric(work["feature_index"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date", "preview_name", "feature_index"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["feature_index"] = work["feature_index"].astype(int)
    work["date"] = work["date"].astype(str).map(normalize_date_str)
    work["preview_name"] = work["preview_name"].fillna("").astype(str)
    work["area_km2"] = pd.to_numeric(work.get("area_km2", np.nan), errors="coerce")
    work["node_key"] = work.apply(
        lambda row: format_node_key((row["date"], row["preview_name"], int(row["feature_index"]))),
        axis=1,
    )
    return work.reset_index(drop=True)


def load_branch_start_features_gdf(gdb_path):
    if not os.path.exists(gdb_path):
        return gpd.GeoDataFrame()
    try:
        layers = list_gdb_layers(gdb_path)
    except Exception:
        layers = []
    if not layers:
        try:
            gdf = gpd.read_file(gdb_path, layer="branch_start_features")
            layers = ["branch_start_features"]
        except Exception:
            return gpd.GeoDataFrame()
    gdf_list = []
    for layer_name in layers:
        try:
            layer_gdf = gpd.read_file(gdb_path, layer=layer_name)
        except Exception:
            continue
        if layer_gdf is None or layer_gdf.empty:
            continue
        layer_gdf = layer_gdf.copy()
        layer_gdf["source_layer"] = str(layer_name)
        gdf_list.append(layer_gdf)
    if not gdf_list:
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(
        pd.concat(gdf_list, ignore_index=True),
        geometry="geometry",
        crs=gdf_list[0].crs,
    )
    if gdf.empty:
        return gdf
    work = gdf.copy()
    if "date" not in work.columns and "date_" in work.columns:
        work = work.rename(columns={"date_": "date"})
    if "lake_id" not in work.columns:
        work["lake_id"] = np.nan
    if "branch_id" not in work.columns:
        work["branch_id"] = ""
    if "branch_start_order" not in work.columns and "branch_order" in work.columns:
        work["branch_start_order"] = work["branch_order"]
    if "date" not in work.columns:
        work["date"] = ""
    if "feature_index" not in work.columns:
        work["feature_index"] = np.nan
    if "area_km2" not in work.columns:
        work["area_km2"] = np.nan
    work["lake_id"] = pd.to_numeric(work.get("lake_id"), errors="coerce")
    work["branch_id"] = work.get("branch_id", "").astype(str).str.strip()
    work["date"] = work["date"].astype(str).map(normalize_date_str)
    work["feature_index"] = pd.to_numeric(work.get("feature_index"), errors="coerce")
    work["area_km2"] = pd.to_numeric(work.get("area_km2"), errors="coerce")
    work = work.dropna(subset=["lake_id"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    return work.reset_index(drop=True)


def load_branch_start_records(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df
    work = df.copy()
    work["lake_id"] = pd.to_numeric(work.get("lake_id"), errors="coerce")
    work["branch_id"] = work.get("branch_id", "").astype(str).str.strip()
    work["date"] = work.get("date", "").astype(str).map(normalize_date_str)
    work["feature_index"] = pd.to_numeric(work.get("feature_index"), errors="coerce")
    work = work.dropna(subset=["lake_id", "feature_index"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["feature_index"] = work["feature_index"].astype(int)
    return work.reset_index(drop=True)


def load_effective_boundary_windows(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["lake_id", "effective_boundary_start", "effective_boundary_end"])

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame(columns=["lake_id", "effective_boundary_start", "effective_boundary_end"])

    work = df.copy()
    if "lake_id" not in work.columns or "date" not in work.columns:
        return pd.DataFrame(columns=["lake_id", "effective_boundary_start", "effective_boundary_end"])

    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce")
    work["date"] = work["date"].astype(str).map(normalize_date_str)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    work = work.dropna(subset=["lake_id", "date_dt"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["lake_id", "effective_boundary_start", "effective_boundary_end"])

    grouped = (
        work.groupby("lake_id", as_index=False)["date_dt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "effective_boundary_start_dt", "max": "effective_boundary_end_dt"})
    )
    grouped["lake_id"] = grouped["lake_id"].astype(int)
    grouped["effective_boundary_start"] = grouped["effective_boundary_start_dt"].dt.strftime("%Y%m%d")
    grouped["effective_boundary_end"] = grouped["effective_boundary_end_dt"].dt.strftime("%Y%m%d")
    return grouped[["lake_id", "effective_boundary_start", "effective_boundary_end"]]


def classify_final_curve_size_category(max_area_km2):
    value = pd.to_numeric(max_area_km2, errors="coerce")
    if pd.isna(value):
        return "Small"
    if float(value) > 0.625:
        return "Large"
    if float(value) > 0.0625:
        return "Medium"
    return "Small"


def build_lake_mask_metadata_from_branch_starts(branch_start_df):
    if branch_start_df is None or branch_start_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = branch_start_df.copy()
    work["lake_id"] = pd.to_numeric(work.get("lake_id"), errors="coerce")
    work["branch_id"] = work.get("branch_id", "").astype(str).str.strip()
    work["date"] = work.get("date", "").astype(str).map(normalize_date_str)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    work["area_km2"] = pd.to_numeric(work.get("area_km2"), errors="coerce")
    work["branch_order_num"] = pd.to_numeric(
        work.get("branch_start_order", work.get("branch_order", np.nan)),
        errors="coerce",
    )
    work["feature_index_num"] = pd.to_numeric(work.get("feature_index"), errors="coerce")
    work = work.dropna(subset=["lake_id", "branch_id", "date_dt"]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["lake_id"] = work["lake_id"].astype(int)

    lake_rows = []
    branch_rows = []
    for lake_id, lake_df in work.groupby("lake_id", sort=True):
        lake_df = lake_df.sort_values(["branch_order_num", "date_dt", "feature_index_num", "branch_id"]).copy()
        if lake_df.empty:
            continue
        size_vals = [str(v).strip() for v in lake_df.get("size_category", pd.Series(dtype=object)).tolist() if str(v).strip()]
        size_category = size_vals[0] if size_vals else classify_final_curve_size_category(lake_df["area_km2"].max())
        lake_rows.append(
            {
                "lake_id": int(lake_id),
                "size_category": size_category,
                "water_period_start": str(lake_df["date"].min()),
                "water_period_end": str(lake_df["date"].max()),
                "peak_date": "",
                "peak_area_km2": float(lake_df["area_km2"].max()) if lake_df["area_km2"].notna().any() else np.nan,
                "peak_branch_id": "",
                "premax_min_date": "",
                "premax_min_key": "",
                "prepeak_branch_class": "MaskOnly",
                "prepeak_branch_count": int(len(lake_df)),
                "source_mode": "05_2_branch_start_gdb",
            }
        )
        fallback_rank = 1
        for row in lake_df.itertuples(index=False):
            raw_order = pd.to_numeric(getattr(row, "branch_start_order", np.nan), errors="coerce")
            if pd.isna(raw_order):
                raw_order = pd.to_numeric(getattr(row, "branch_order", np.nan), errors="coerce")
            branch_index = int(raw_order) if pd.notna(raw_order) else int(fallback_rank)
            fallback_rank += 1
            feature_index = pd.to_numeric(getattr(row, "feature_index", np.nan), errors="coerce")
            branch_rows.append(
                {
                    "lake_id": int(lake_id),
                    "size_category": size_category,
                    "branch_index": int(branch_index),
                    "branch_id": str(getattr(row, "branch_id", "")).strip(),
                    "branch_start_date": str(getattr(row, "date", "")),
                    "branch_start_key": format_node_key(
                        (
                            str(getattr(row, "date", "")),
                            str(getattr(row, "preview_name", "")),
                            int(feature_index) if pd.notna(feature_index) else 0,
                        )
                    ),
                    "premerge_min_date": str(getattr(row, "date", "")),
                    "premerge_min_key": format_node_key(
                        (
                            str(getattr(row, "date", "")),
                            str(getattr(row, "preview_name", "")),
                            int(feature_index) if pd.notna(feature_index) else 0,
                        )
                    ),
                    "premerge_min_area_km2": float(pd.to_numeric(getattr(row, "area_km2", np.nan), errors="coerce")),
                    "merge_key": "",
                    "branch_node_count": 1,
                    "premerge_min_area_ratio_to_peak": np.nan,
                    "source_mode": "05_2_branch_start_gdb",
                }
            )
    return pd.DataFrame(lake_rows), pd.DataFrame(branch_rows)


def build_maps_from_topology_df(topology_df):
    node_rows = {}
    parent_map = defaultdict(list)
    children_map = defaultdict(list)
    for _, row in topology_df.iterrows():
        node_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        node_rows[node_key] = row
    for _, row in topology_df.iterrows():
        node_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        for parent_key in parse_node_key_list(row.get("parent_node_keys", "")):
            if parent_key not in node_rows:
                continue
            if parent_key not in parent_map[node_key]:
                parent_map[node_key].append(parent_key)
            if node_key not in children_map[parent_key]:
                children_map[parent_key].append(node_key)
    return node_rows, parent_map, children_map


def prune_oriented_children_from_leaves(child_lookup, root_key, node_rows):
    working_children = defaultdict(list)
    working_parent = defaultdict(list)
    all_nodes = {root_key}
    for parent_key, child_keys in child_lookup.items():
        unique_children = []
        for child_key in child_keys:
            if child_key not in unique_children:
                unique_children.append(child_key)
        working_children[parent_key] = unique_children
        all_nodes.add(parent_key)
        for child_key in unique_children:
            all_nodes.add(child_key)
            working_parent[child_key].append(parent_key)

    leaf_nodes = sorted(
        [node_key for node_key in all_nodes if node_key != root_key and len(working_children.get(node_key, [])) == 0],
        key=lambda key: (str(node_rows[key]["date"]), float(node_rows[key]["area_km2"]), key[2]),
    )
    visited = set()

    def walk_up(node_key):
        if node_key in visited:
            return
        visited.add(node_key)
        parent_keys = list(working_parent.get(node_key, []))
        if len(parent_keys) > 1:
            chosen_parent = parent_keys[0]
            for other_parent in parent_keys[1:]:
                working_children[other_parent] = [key for key in working_children.get(other_parent, []) if key != node_key]
                working_parent[node_key] = [key for key in working_parent[node_key] if key != other_parent]
            parent_keys = [chosen_parent]
        for parent_key in parent_keys:
            walk_up(parent_key)

    for leaf_key in leaf_nodes:
        walk_up(leaf_key)
    for node_key in sorted(all_nodes, key=lambda key: (str(node_rows[key]["date"]), float(node_rows[key]["area_km2"]), key[2])):
        walk_up(node_key)
    return working_children


def prune_small_side_branches(child_lookup, root_key, min_branch_nodes=4):
    working_children = defaultdict(list)
    for parent_key, child_keys in child_lookup.items():
        unique_children = []
        for child_key in child_keys:
            if child_key not in unique_children:
                unique_children.append(child_key)
        working_children[parent_key] = unique_children

    subtree_size_cache = {}

    def subtree_size(node_key):
        if node_key in subtree_size_cache:
            return subtree_size_cache[node_key]
        size = 1
        for child_key in working_children.get(node_key, []):
            size += subtree_size(child_key)
        subtree_size_cache[node_key] = size
        return size

    def prune_from(node_key):
        child_keys = list(working_children.get(node_key, []))
        kept_children = []
        for child_key in child_keys:
            child_branch_size = subtree_size(child_key)
            if len(child_keys) > 1 and child_branch_size <= min_branch_nodes:
                continue
            kept_children.append(child_key)
            prune_from(child_key)
        working_children[node_key] = kept_children

    prune_from(root_key)
    return working_children


def build_peak_root_time_layout(peak_key, children_map, parent_map, node_rows):
    pre_children = defaultdict(list)
    post_children = defaultdict(list)
    visited_pre = set()
    visited_post = set()

    def grow_pre(node_key):
        if node_key in visited_pre:
            return
        visited_pre.add(node_key)
        parent_keys = list(parent_map.get(node_key, []))
        parent_keys.sort(key=lambda key: (str(node_rows[key]["date"]), -float(node_rows[key]["area_km2"]), key[1]), reverse=True)
        for parent_key in parent_keys:
            pre_children[node_key].append(parent_key)
            grow_pre(parent_key)

    def grow_post(node_key):
        if node_key in visited_post:
            return
        visited_post.add(node_key)
        child_keys = list(children_map.get(node_key, []))
        child_keys.sort(key=lambda key: (str(node_rows[key]["date"]), -float(node_rows[key]["area_km2"]), key[1]))
        for child_key in child_keys:
            post_children[node_key].append(child_key)
            grow_post(child_key)

    grow_pre(peak_key)
    grow_post(peak_key)
    pre_children = prune_oriented_children_from_leaves(pre_children, peak_key, node_rows)
    post_children = prune_oriented_children_from_leaves(post_children, peak_key, node_rows)
    pre_children = prune_small_side_branches(pre_children, peak_key, 4)
    post_children = prune_small_side_branches(post_children, peak_key, 4)
    return pre_children, post_children


def analyze_prepeak_branches(pre_children, actual_root, node_rows, peak_key):
    peak_area_km2 = float(node_rows[peak_key]["area_km2"]) if peak_key in node_rows else np.nan
    pre_parent = {}
    pre_nodes = set()
    for parent_key, child_keys in pre_children.items():
        pre_nodes.add(parent_key)
        for child_key in child_keys:
            pre_nodes.add(child_key)
            pre_parent[child_key] = parent_key
    if actual_root in pre_nodes:
        pre_nodes.remove(actual_root)

    leaf_keys = sorted(
        [node_key for node_key in pre_nodes if len(pre_children.get(node_key, [])) == 0],
        key=lambda key: (str(node_rows[key]["date"]), float(node_rows[key]["area_km2"]), str(node_rows[key]["preview_name"]), int(node_rows[key]["feature_index"])),
    )

    branch_records = []
    for leaf_key in leaf_keys:
        branch_nodes = [leaf_key]
        current_key = leaf_key
        merge_key = None
        while current_key in pre_parent:
            parent_key = pre_parent[current_key]
            if len(pre_children.get(parent_key, [])) > 1:
                merge_key = parent_key
                break
            branch_nodes.append(parent_key)
            current_key = parent_key
        if not branch_nodes or merge_key is None or merge_key == actual_root:
            continue

        premerge_min_key = min(
            branch_nodes,
            key=lambda key: (
                float(node_rows[key]["area_km2"]),
                str(node_rows[key]["date"]),
                str(node_rows[key]["preview_name"]),
                int(node_rows[key]["feature_index"]),
            ),
        )
        premerge_min_area_km2 = float(node_rows[premerge_min_key]["area_km2"])
        area_ratio_to_peak = (
            premerge_min_area_km2 / peak_area_km2
            if pd.notna(peak_area_km2) and peak_area_km2 > 0
            else np.nan
        )
        branch_records.append(
            {
                "start_key": leaf_key,
                "start_date": str(node_rows[leaf_key]["date"]),
                "premerge_min_key": premerge_min_key,
                "premerge_min_date": str(node_rows[premerge_min_key]["date"]),
                "premerge_min_area_km2": premerge_min_area_km2,
                "premerge_min_area_ratio_to_peak": float(area_ratio_to_peak) if pd.notna(area_ratio_to_peak) else np.nan,
                "merge_key": merge_key,
                "node_count": len(branch_nodes),
            }
        )
    return branch_records


def build_lake_mask_metadata(topology_df):
    lake_rows = []
    branch_rows = []
    for lake_id, lake_df in topology_df.groupby("lake_id", sort=True):
        lake_df = lake_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        if lake_df.empty:
            continue

        peak_idx = lake_df["area_km2"].idxmax()
        peak_row = lake_df.loc[peak_idx]
        peak_key = (str(peak_row["date"]), str(peak_row["preview_name"]), int(peak_row["feature_index"]))
        water_period_start = str(lake_df["date"].iloc[0])
        water_period_end = str(lake_df["date"].iloc[-1])
        size_category = classify_final_curve_size_category(lake_df["area_km2"].max())

        main_branch_id = str(peak_row.get("branch_id", "")).strip()
        prepeak_df = lake_df[lake_df["date_dt"] <= peak_row["date_dt"]].copy()
        prepeak_main_df = prepeak_df[prepeak_df["branch_id"].astype(str) == main_branch_id].copy()
        if prepeak_main_df.empty:
            prepeak_main_df = prepeak_df[prepeak_df["is_main_lineage"].astype(str).str.lower() == "true"].copy()
        if prepeak_main_df.empty:
            prepeak_main_df = prepeak_df.copy()
        premax_min_row = prepeak_main_df.sort_values(
            ["area_km2", "date_dt", "preview_name", "feature_index"],
            ascending=[True, True, True, True],
        ).iloc[0]
        premax_min_key = (str(premax_min_row["date"]), str(premax_min_row["preview_name"]), int(premax_min_row["feature_index"]))

        lake_branch_rows = []
        branch_start_df = prepeak_df[
            prepeak_df["is_branch_start"].astype(str).str.lower() == "true"
        ].copy()
        branch_start_df = branch_start_df[branch_start_df["branch_id"].astype(str) != main_branch_id].copy()
        branch_start_df = branch_start_df.sort_values(
            ["branch_start_order", "date_dt", "preview_name", "feature_index"],
            ascending=[True, True, True, True],
        )
        for fallback_index, start_row in enumerate(branch_start_df.itertuples(index=False), 1):
            branch_id = str(start_row.branch_id).strip()
            if not branch_id:
                continue
            branch_nodes_df = prepeak_df[prepeak_df["branch_id"].astype(str) == branch_id].copy()
            if branch_nodes_df.empty:
                continue
            premerge_min_row = branch_nodes_df.sort_values(
                ["area_km2", "date_dt", "preview_name", "feature_index"],
                ascending=[True, True, True, True],
            ).iloc[0]
            premerge_min_key = (
                str(premerge_min_row["date"]),
                str(premerge_min_row["preview_name"]),
                int(premerge_min_row["feature_index"]),
            )

            merge_row = None
            merge_candidates = branch_nodes_df[
                branch_nodes_df["child_branch_ids"].astype(str).map(parse_text_list).map(lambda ids: main_branch_id in ids)
            ].copy()
            if merge_candidates.empty:
                merge_candidates = branch_nodes_df[
                    branch_nodes_df["parent_branch_ids"].astype(str).map(parse_text_list).map(lambda ids: main_branch_id in ids)
                ].copy()
            if not merge_candidates.empty:
                merge_row = merge_candidates.sort_values(
                    ["date_dt", "preview_name", "feature_index"],
                    ascending=[True, True, True],
                ).iloc[0]

            branch_index = pd.to_numeric(getattr(start_row, "branch_start_order", np.nan), errors="coerce")
            if pd.isna(branch_index):
                branch_index = fallback_index

            lake_branch_rows.append(
                {
                    "lake_id": int(lake_id),
                    "size_category": size_category,
                    "branch_index": int(branch_index),
                    "branch_id": branch_id,
                    "branch_start_date": str(start_row.date),
                    "branch_start_key": format_node_key((str(start_row.date), str(start_row.preview_name), int(start_row.feature_index))),
                    "premerge_min_date": str(premerge_min_row["date"]),
                    "premerge_min_key": format_node_key(premerge_min_key),
                    "premerge_min_area_km2": float(premerge_min_row["area_km2"]),
                    "merge_key": "" if merge_row is None else format_node_key((str(merge_row["date"]), str(merge_row["preview_name"]), int(merge_row["feature_index"]))),
                    "branch_node_count": int(len(branch_nodes_df)),
                    "premerge_min_area_ratio_to_peak": float(premerge_min_row["area_km2"] / peak_row["area_km2"]) if pd.notna(peak_row["area_km2"]) and float(peak_row["area_km2"]) > 0 else np.nan,
                    "source_mode": "05_2_branch_id",
                }
            )

        lake_rows.append(
            {
                "lake_id": int(lake_id),
                "size_category": size_category,
                "water_period_start": water_period_start,
                "water_period_end": water_period_end,
                "peak_date": str(peak_row["date"]),
                "peak_area_km2": float(peak_row["area_km2"]),
                "peak_branch_id": main_branch_id,
                "premax_min_date": str(premax_min_row["date"]),
                "premax_min_key": format_node_key(premax_min_key),
                "prepeak_branch_class": "WithBranch" if lake_branch_rows else "NoBranch",
                "prepeak_branch_count": int(len(lake_branch_rows)),
                "source_mode": "05_2_branch_id",
            }
        )
        branch_rows.extend(lake_branch_rows)
    return pd.DataFrame(lake_rows), pd.DataFrame(branch_rows)


def select_branch_start_mask(branch_start_df, topology_features_gdf, lake_id, branch_id):
    if branch_start_df is None or branch_start_df.empty:
        return gpd.GeoDataFrame(), "", None, np.nan, "missing_branch_start"
    branch_text = str(branch_id or "").strip()
    if not branch_text:
        return gpd.GeoDataFrame(), "", None, np.nan, "missing_branch_id"
    sel = branch_start_df[
        (branch_start_df["lake_id"] == int(lake_id))
        & (branch_start_df["branch_id"].astype(str) == branch_text)
    ].copy()
    if sel.empty:
        return sel, "", None, np.nan, "branch_start_not_found"
    order_col = "branch_start_order" if "branch_start_order" in sel.columns else ("branch_order" if "branch_order" in sel.columns else None)
    sort_cols = []
    if order_col is not None:
        sel["_order_num"] = pd.to_numeric(sel[order_col], errors="coerce")
        sort_cols.append("_order_num")
    sel = sel.sort_values(sort_cols + ["date", "feature_index"]).head(1).copy()
    date_text = str(sel.iloc[0]["date"])
    preview_name = str(sel.iloc[0].get("preview_name", "")).strip()
    feature_index = int(pd.to_numeric(sel.iloc[0]["feature_index"], errors="coerce"))
    geom_sel = sel[sel["geometry"].notna()].head(1).copy() if "geometry" in sel.columns else gpd.GeoDataFrame()
    if geom_sel.empty:
        geom_sel = topology_features_gdf[
            (topology_features_gdf["lake_id"] == int(lake_id))
            & (topology_features_gdf["date"].astype(str) == date_text)
            & (topology_features_gdf["preview_name"].astype(str) == preview_name)
            & (pd.to_numeric(topology_features_gdf["feature_index"], errors="coerce") == feature_index)
        ].copy()
        if geom_sel.empty:
            return gpd.GeoDataFrame(), date_text, feature_index, np.nan, "branch_start_geometry_not_found"
        geom_sel = geom_sel.head(1).copy()
    area_value = pd.to_numeric(sel.iloc[0].get("area_km2", np.nan), errors="coerce")
    return (
        geom_sel[["geometry"]].copy(),
        date_text,
        feature_index,
        (np.nan if pd.isna(area_value) else float(area_value)),
        "branch_start_feature",
    )


def select_forced_main_mask_by_date(topology_features_gdf, lake_id, forced_date_text):
    date_text = normalize_date_str(forced_date_text)
    if not date_text:
        return gpd.GeoDataFrame(), "", None, np.nan, "forced_mask_date_invalid"
    if topology_features_gdf is None or topology_features_gdf.empty:
        return gpd.GeoDataFrame(), date_text, None, np.nan, "forced_mask_date_missing_topology"

    sel = topology_features_gdf[
        (topology_features_gdf["lake_id"] == int(lake_id))
        & (topology_features_gdf["date"].astype(str) == date_text)
    ].copy()
    if sel.empty:
        return gpd.GeoDataFrame(), date_text, None, np.nan, "forced_mask_date_not_found"

    sel["area_num"] = pd.to_numeric(sel.get("area_km2", np.nan), errors="coerce")
    sel["feature_num"] = pd.to_numeric(sel.get("feature_index", np.nan), errors="coerce")
    sel = sel.sort_values(["area_num", "feature_num"], ascending=[False, True]).head(1).copy()

    feature_index = int(pd.to_numeric(sel.iloc[0].get("feature_index", np.nan), errors="coerce"))
    area_value = pd.to_numeric(sel.iloc[0].get("area_km2", np.nan), errors="coerce")
    return (
        sel[["geometry"]].copy(),
        date_text,
        feature_index,
        (np.nan if pd.isna(area_value) else float(area_value)),
        "forced_mask_date",
    )


def pick_forced_topology_feature_row(topology_features_gdf, lake_id, forced_date_text):
    date_text = normalize_date_str(forced_date_text)
    if not date_text or topology_features_gdf is None or topology_features_gdf.empty:
        return None
    sel = topology_features_gdf[
        (topology_features_gdf["lake_id"] == int(lake_id))
        & (topology_features_gdf["date"].astype(str) == date_text)
    ].copy()
    if sel.empty:
        return None
    sel["area_num"] = pd.to_numeric(sel.get("area_km2", np.nan), errors="coerce")
    sel["feature_num"] = pd.to_numeric(sel.get("feature_index", np.nan), errors="coerce")
    sel = sel.sort_values(["area_num", "feature_num"], ascending=[False, True]).head(1).copy()
    return sel.iloc[0]


def apply_forced_main_mask_to_branch_start_df(branch_start_df, topology_features_gdf, lake_id, peak_branch_id, forced_date_text):
    if branch_start_df is None or branch_start_df.empty:
        return branch_start_df
    row = pick_forced_topology_feature_row(topology_features_gdf, lake_id, forced_date_text)
    if row is None:
        progress(
            f"[06_1] lake {int(lake_id)} forced main mask date {forced_date_text} not found in topology features"
        )
        return branch_start_df

    out = branch_start_df.copy()
    mask = (
        (pd.to_numeric(out.get("lake_id", np.nan), errors="coerce") == int(lake_id))
        & (out.get("branch_id", "").astype(str).str.strip() == str(peak_branch_id).strip())
    )
    if not mask.any():
        progress(
            f"[06_1] lake {int(lake_id)} forced main mask date skipped: peak branch {peak_branch_id} not found in branch_start"
        )
        return out

    out.loc[mask, "date"] = str(row.get("date", ""))
    if "preview_name" in out.columns:
        out.loc[mask, "preview_name"] = str(row.get("preview_name", ""))
    if "feature_index" in out.columns:
        out.loc[mask, "feature_index"] = int(pd.to_numeric(row.get("feature_index", np.nan), errors="coerce"))
    if "area_km2" in out.columns:
        out.loc[mask, "area_km2"] = float(pd.to_numeric(row.get("area_km2", np.nan), errors="coerce"))

    progress(
        f"[06_1] lake {int(lake_id)} legacy main seed forced -> date={row.get('date','')}, "
        f"feature_index={int(pd.to_numeric(row.get('feature_index', np.nan), errors='coerce'))}, "
        f"area_km2={float(pd.to_numeric(row.get('area_km2', np.nan), errors='coerce')):.4f}"
    )
    return out


def resolve_lake_date_window(summary_row):
    window_start = resolve_custom_window_start()
    window_end = resolve_custom_window_end()
    if pd.isna(window_start):
        window_start = to_datetime_ymd(summary_row.get("effective_boundary_start", ""))
    if pd.isna(window_end):
        window_end = to_datetime_ymd(summary_row.get("effective_boundary_end", ""))
    if pd.isna(window_end):
        window_end = to_datetime_ymd(summary_row.get("water_period_end", ""))
    return window_start, window_end


def build_lake_union_mask(specs):
    mask_frames = [spec["mask_gdf"][["geometry"]].copy() for spec in specs if spec.get("mask_gdf") is not None and not spec["mask_gdf"].empty]
    if not mask_frames:
        return gpd.GeoDataFrame()
    merged = gpd.GeoDataFrame(pd.concat(mask_frames, ignore_index=True), geometry="geometry", crs=mask_frames[0].crs)
    union_geom = merged.geometry.unary_union
    return gpd.GeoDataFrame({"geometry": [union_geom]}, geometry="geometry", crs=merged.crs)


def load_tiles_gdf(tile_shp):
    if not os.path.exists(tile_shp):
        return gpd.GeoDataFrame()
    tiles_gdf = gpd.read_file(tile_shp)
    if tiles_gdf.empty:
        return tiles_gdf
    if tiles_gdf.crs is None or tiles_gdf.crs.to_epsg() != 4326:
        tiles_gdf = tiles_gdf.to_crs("EPSG:4326")
    return tiles_gdf


def load_lake_buffered_geom_map(shp_path):
    if not os.path.exists(shp_path):
        return {}
    lakes_gdf = gpd.read_file(shp_path)
    if lakes_gdf.empty or "lake_id" not in lakes_gdf.columns:
        return {}
    if lakes_gdf.crs is None or lakes_gdf.crs.to_epsg() != 4326:
        lakes_gdf = lakes_gdf.to_crs("EPSG:4326")
    lakes_gdf["lake_id"] = pd.to_numeric(lakes_gdf["lake_id"], errors="coerce")
    lakes_gdf = lakes_gdf.dropna(subset=["lake_id"]).copy()
    lakes_gdf["lake_id"] = lakes_gdf["lake_id"].astype(int)
    out = {}
    for lake_id, grp in lakes_gdf.groupby("lake_id", sort=False):
        geom = grp.geometry.unary_union
        if geom is not None and not geom.is_empty:
            out[int(lake_id)] = geom
    return out


def resolve_water_period_window(summary_row):
    window_start = resolve_custom_window_start()
    window_end = resolve_custom_window_end()
    if pd.isna(window_start):
        window_start = to_datetime_ymd(summary_row.get("water_period_start", ""))
    if pd.isna(window_end):
        window_end = to_datetime_ymd(summary_row.get("water_period_end", ""))
    return (
        window_start,
        window_end,
    )


def load_pixc_repair_state(log_csv_path):
    if not os.path.exists(log_csv_path):
        return {}
    try:
        df = pd.read_csv(log_csv_path, encoding="utf-8-sig")
    except Exception:
        return {}
    if df.empty or "granule_name" not in df.columns or "status" not in df.columns:
        return {}
    state = {}
    for row in df.to_dict("records"):
        name = str(row.get("granule_name", "")).strip()
        if not name:
            continue
        state[name] = {
            "status": str(row.get("status", "")).strip(),
            "repaired_path": str(row.get("repaired_path", "")).strip(),
            "detail": str(row.get("detail", "")).strip(),
        }
    return state


def record_pixc_repair(lake_id, day_text, granule_name, status, detail, repaired_path):
    row = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lake_id": int(lake_id),
        "date": str(day_text),
        "granule_name": str(granule_name),
        "status": str(status),
        "detail": str(detail),
        "repaired_path": str(repaired_path or ""),
    }
    with GLOBAL_PIXC_REPAIR_LOCK:
        GLOBAL_PIXC_REPAIR_ROWS.append(row)
        GLOBAL_PIXC_REPAIR_STATUS[str(granule_name)] = {
            "status": str(status),
            "repaired_path": str(repaired_path or ""),
            "detail": str(detail),
        }


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
                        continue
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
                downloaded = earthaccess.download(matched, str(PIXC_DIR), threads=1)
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


def build_candidate_pixc_files_by_day(lake_geom, summary_row, tiles_gdf, pixc_map):
    if tiles_gdf is None or tiles_gdf.empty or not pixc_map:
        return {}
    if lake_geom is None or lake_geom.is_empty:
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
        if not tile_text:
            continue
        candidate_files.extend(pixc_map.get(tile_text, []))
    candidate_files = sorted(set(candidate_files))
    if not candidate_files:
        return {}

    lake_start, lake_end = resolve_water_period_window(summary_row)
    by_day = defaultdict(list)
    for nc_path in candidate_files:
        date_text = extract_pixc_date_str(nc_path)
        day_dt = to_datetime_ymd(date_text)
        if pd.isna(day_dt):
            continue
        day_dt = day_dt.normalize()
        if pd.notna(lake_start) and day_dt < lake_start.normalize():
            continue
        if pd.notna(lake_end) and day_dt > lake_end.normalize():
            continue
        by_day[day_dt].append(nc_path)
    return dict(by_day)


def list_swot_layers_by_day(swot_gdb_path):
    if not os.path.exists(swot_gdb_path):
        return {}
    try:
        layer_names = list_gdb_layers(swot_gdb_path)
    except Exception:
        return {}
    by_day = defaultdict(list)
    # Only use version-explicit layers. We intentionally ignore legacy names
    # like ..._01 and only keep layers such as ..._PGD0_01 / ..._PIC0_01.
    layer_pattern = re.compile(r"^pixc_\d{8}T\d{6}_\d{3}_\d{3}[A-Z]_(?:PGD0|PIC0)_\d{2}$")
    for layer_name in layer_names:
        if not layer_pattern.match(str(layer_name)):
            continue
        dt = parse_swot_date(layer_name)
        if dt is not None and pd.notna(dt):
            by_day[pd.to_datetime(dt).normalize()].append(layer_name)
    return dict(by_day)


def build_quality_candidates_with_reason(clipped_gdf):
    gdf_base = apply_quality_filters(clipped_gdf, BASE_FILTER)
    cross_track_mode = "standard"
    cross_track_relaxed = False
    if gdf_base.empty:
        if not ALLOW_CROSSTRACK_RELAXATION:
            return [], "base_filter_empty", "standard", False
        gdf_base_relaxed = apply_quality_filters(clipped_gdf, BASE_FILTER_NO_CROSSTRACK)
        if gdf_base_relaxed.empty:
            return [], "base_filter_empty", "standard", False
        gdf_base = gdf_base_relaxed
        cross_track_mode = "relaxed_no_cross_track"
        cross_track_relaxed = True
    if "geolocation_qual" not in gdf_base.columns:
        return [], "missing_geolocation_qual", cross_track_mode, cross_track_relaxed

    gdf_tier = gdf_base[pd.to_numeric(gdf_base["geolocation_qual"], errors="coerce").fillna(np.inf) <= GEO_QUAL_HIGH_THRESH].copy()
    if gdf_tier.empty:
        return [], "geoqual_filtered_empty", cross_track_mode, cross_track_relaxed

    geoqual_vals = gdf_tier["geolocation_qual"].fillna(0).astype(np.int64)
    strict_df = gdf_tier[(geoqual_vals & BAD_BITS_MASK) == 0].copy()
    relaxed_df = gdf_tier[(geoqual_vals & BAD_BITS_MASK) != 0].copy()

    candidates = []
    sigma_failed = []
    for tier_rank, (tier_name, tier_df) in enumerate([("Strict", strict_df), ("Relaxed", relaxed_df)], 1):
        if tier_df.empty:
            sigma_failed.append(f"{tier_name.lower()}_empty")
            continue
        raw_std = float(pd.to_numeric(tier_df["wse"], errors="coerce").std()) if len(tier_df) > 1 else 0.0
        if np.isnan(raw_std):
            raw_std = 0.0
        raw_wse = float(tier_df["wse"].median() if tier_name == "Strict" else tier_df["wse"].mean())
        pre_count = int(len(tier_df))
        candidates.append(
            {
                "type": f"{tier_name}_raw",
                "type_rank": (tier_rank * 2),
                "wse_stage": "pre_sigma",
                "sigma_filtered": False,
                "pre_filter_count": pre_count,
                "post_filter_count": pre_count,
                "count": pre_count,
                "wse_m": raw_wse,
                "wse_std": raw_std,
                "sigma_iters": 0,
                "cross_track_mode": cross_track_mode,
                "cross_track_relaxed": cross_track_relaxed,
            }
        )
        kept_wse, std_value, sigma_iters = iterative_sigma_filter(
            tier_df["wse"], sigma=2.0, std_target=SIGMA_TARGET_METERS, max_iters=SIGMA_MAX_ITERS
        )
        if kept_wse.empty:
            sigma_failed.append(f"{tier_name.lower()}_sigma_empty")
            continue
        tier_df = tier_df.loc[kept_wse.index].copy()
        if std_value > SIGMA_TARGET_METERS:
            sigma_failed.append(f"{tier_name.lower()}_sigma_gt_target")
            continue
        candidates.append(
            {
                "type": f"{tier_name}_sigma",
                "type_rank": (tier_rank * 2) - 1,
                "wse_stage": "post_sigma",
                "sigma_filtered": True,
                "pre_filter_count": pre_count,
                "post_filter_count": int(len(tier_df)),
                "count": int(len(tier_df)),
                "wse_m": float(tier_df["wse"].median() if tier_name == "Strict" else tier_df["wse"].mean()),
                "wse_std": float(std_value),
                "sigma_iters": int(sigma_iters),
                "cross_track_mode": cross_track_mode,
                "cross_track_relaxed": cross_track_relaxed,
            }
        )
    return (
        candidates,
        ("|".join(sigma_failed) if sigma_failed else "no_quality_candidates") if not candidates else "",
        cross_track_mode,
        cross_track_relaxed,
    )


def extract_wse_candidates_for_day(swot_gdb_path, layer_names, mask_gdf, layer_cache):
    day_records = []
    failure_reasons = []
    for layer_name in layer_names:
        if layer_name not in layer_cache:
            cached_layer = get_cached_layer(swot_gdb_path, layer_name)
            if cached_layer is not None:
                layer_cache[layer_name] = cached_layer
            else:
                try:
                    loaded_layer = gpd.read_file(swot_gdb_path, layer=layer_name)
                except Exception:
                    loaded_layer = None
                layer_cache[layer_name] = loaded_layer
                if loaded_layer is not None:
                    set_cached_layer(swot_gdb_path, layer_name, loaded_layer)
        swot_gdf = layer_cache[layer_name]
        if swot_gdf is None or swot_gdf.empty:
            continue

        try:
            mask_geom = mask_gdf.to_crs(swot_gdf.crs)[["geometry"]] if mask_gdf.crs != swot_gdf.crs else mask_gdf[["geometry"]]
            clipped = gpd.clip(swot_gdf, mask_geom)
        except Exception:
            continue
        if clipped.empty:
            failure_reasons.append(f"{layer_name}:clip_empty")
            continue

        candidates, reason, cross_track_mode, cross_track_relaxed = build_quality_candidates_with_reason(clipped)
        if not candidates:
            failure_reasons.append(f"{layer_name}:{reason or 'no_qual'}")
            continue

        for candidate in candidates:
            candidate["source_layer"] = layer_name
            candidate["cross_track_mode"] = cross_track_mode
            candidate["cross_track_relaxed"] = cross_track_relaxed
            day_records.append(candidate)
    return day_records, failure_reasons


def export_pixc_to_gdb(pixc_gdf, pixc_gdb_path, layer_name):
    if pixc_gdf is None or pixc_gdf.empty:
        return False
    os.makedirs(os.path.dirname(pixc_gdb_path), exist_ok=True)
    for driver_name in ("OpenFileGDB", "FileGDB"):
        try:
            pixc_gdf.to_file(pixc_gdb_path, layer=layer_name, driver=driver_name)
            return True
        except Exception:
            continue
    return False


def extract_wse_candidates_for_day_from_pixc(
    nc_paths,
    lake_geom,
    mask_gdf,
    tiles_gdf,
    pixc_cache,
    lake_id,
    day_text,
    pixc_gdb_path,
    exported_pixc_layers,
):
    day_records = []
    failure_reasons = []
    if lake_geom is None or lake_geom.is_empty:
        return day_records, ["lake_geom_missing"]
    mask_4326 = mask_gdf.to_crs("EPSG:4326") if mask_gdf.crs != "EPSG:4326" else mask_gdf
    mask_geom = mask_4326[["geometry"]].copy()

    for nc_path in nc_paths:
        pixc_name = os.path.basename(str(nc_path))
        if nc_path not in pixc_cache:
            parsed_gdf, read_status, read_detail = extract_data_from_nc(nc_path, lake_geom)
            if read_status == "corrupt" and ENABLE_PIXC_AUTO_REPAIR:
                known = GLOBAL_PIXC_REPAIR_STATUS.get(pixc_name, {})
                known_status = str(known.get("status", "")).strip()
                if known_status == "repaired_downloaded":
                    if VERBOSE_PIXC_PROGRESS:
                        progress(f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> reuse_repaired_copy")
                    parsed_gdf, read_status, read_detail = extract_data_from_nc(nc_path, lake_geom)
                else:
                    if VERBOSE_PIXC_PROGRESS:
                        progress(f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> start_auto_repair")
                    repaired_ok, repaired_status, repaired_path = attempt_repair_corrupt_pixc(nc_path, tiles_gdf)
                    record_pixc_repair(
                        lake_id=lake_id,
                        day_text=day_text,
                        granule_name=pixc_name,
                        status=repaired_status,
                        detail=read_detail,
                        repaired_path=repaired_path,
                    )
                    if VERBOSE_PIXC_PROGRESS:
                        progress(
                            f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> "
                            f"repair_status={repaired_status}"
                        )
                    if repaired_ok:
                        parsed_gdf, read_status, read_detail = extract_data_from_nc(nc_path, lake_geom)
            pixc_cache[nc_path] = parsed_gdf if read_status in {"ok", "empty"} else None
            layer_name = build_pixc_record_id(nc_path)
            if (
                pixc_gdb_path
                and layer_name not in exported_pixc_layers
                and pixc_cache[nc_path] is not None
                and not pixc_cache[nc_path].empty
            ):
                if export_pixc_to_gdb(pixc_cache[nc_path], pixc_gdb_path, layer_name):
                    exported_pixc_layers.add(layer_name)
                    if VERBOSE_PIXC_PROGRESS:
                        progress(f"[06_1] lake {int(lake_id)} export_pixc_gdb {layer_name} ok")
        swot_gdf = pixc_cache[nc_path]
        if swot_gdf is None or swot_gdf.empty:
            repair_status = str(GLOBAL_PIXC_REPAIR_STATUS.get(pixc_name, {}).get("status", "")).strip()
            if repair_status and repair_status != "repaired_downloaded":
                failure_reasons.append(f"{pixc_name}:nc_corrupt_or_read_failed:{repair_status}")
            else:
                failure_reasons.append(f"{pixc_name}:clip_empty")
            if VERBOSE_PIXC_PROGRESS:
                progress(f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> empty_after_lake_clip_or_read_fail")
            continue
        try:
            clipped = gpd.clip(swot_gdf, mask_geom)
        except Exception:
            failure_reasons.append(f"{pixc_name}:mask_clip_failed")
            if VERBOSE_PIXC_PROGRESS:
                progress(f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> mask_clip_failed")
            continue
        if clipped.empty:
            failure_reasons.append(f"{pixc_name}:clip_empty")
            if VERBOSE_PIXC_PROGRESS:
                progress(f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> empty_after_seed_mask")
            continue

        candidates, reason, cross_track_mode, cross_track_relaxed = build_quality_candidates_with_reason(clipped)
        if not candidates:
            failure_reasons.append(f"{pixc_name}:{reason or 'no_qual'}")
            if VERBOSE_PIXC_PROGRESS:
                progress(f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> no_wse ({reason or 'no_qual'})")
            continue

        source_layer = build_pixc_record_id(nc_path)
        for candidate in candidates:
            candidate["source_layer"] = source_layer
            candidate["cross_track_mode"] = cross_track_mode
            candidate["cross_track_relaxed"] = cross_track_relaxed
            day_records.append(candidate)
        if candidates and VERBOSE_PIXC_PROGRESS:
            best_candidate = min(
                candidates,
                key=lambda item: (
                    int(item.get("type_rank", 99)),
                    -int(item.get("count", 0)),
                    float(item.get("wse_std", np.inf)),
                ),
            )
            progress(
                f"[06_1] lake {int(lake_id)} {day_text} pixc {pixc_name} -> "
                f"wse={float(best_candidate['wse_m']):.3f}m n={int(best_candidate['count'])} "
                f"type={best_candidate['type']} mode={best_candidate.get('cross_track_mode', 'standard')}"
            )
    return day_records, failure_reasons


def classify_failure_reasons(failure_reasons):
    if not failure_reasons:
        return "UNKNOWN", "Unknown"
    txt = "|".join(failure_reasons)
    if "no_pixc" in txt:
        return "NO_PIXC", "NoPIXC"
    if "base_filter" in txt:
        return "BASE_FILTER", "BaseFilter"
    if "geoqual" in txt:
        return "GEOQUAL", "GeoQual"
    if "nc_corrupt_or_read_failed" in txt:
        return "NC_CORRUPT", "NcCorrupt"
    if "clip_empty" in txt:
        return "CLIP_EMPTY", "ClipEmpty"
    if "sigma" in txt:
        return "SIGMA", "Sigma"
    return "OTHER", "Other"


def build_mask_specs(summary_row, branch_df, branch_start_df, topology_features_gdf, lake_id):
    period_start, period_end = resolve_water_period_window(summary_row)
    specs = []

    lake_starts = pd.DataFrame()
    if branch_start_df is not None and not branch_start_df.empty:
        lake_starts = branch_start_df[branch_start_df["lake_id"] == int(lake_id)].copy()
        lake_starts["branch_id"] = lake_starts.get("branch_id", "").astype(str).str.strip()
        lake_starts = lake_starts[lake_starts["branch_id"] != ""].copy()
        if not lake_starts.empty:
            lake_starts["branch_start_order_num"] = pd.to_numeric(
                lake_starts.get("branch_start_order", lake_starts.get("branch_order", np.nan)), errors="coerce"
            )
            lake_starts["date_num"] = pd.to_numeric(lake_starts.get("date", ""), errors="coerce")
            lake_starts["feature_index_num"] = pd.to_numeric(lake_starts.get("feature_index", np.nan), errors="coerce")
            lake_starts = lake_starts.sort_values(
                ["branch_start_order_num", "date_num", "feature_index_num", "branch_id"]
            ).copy()
            lake_starts = lake_starts.drop_duplicates(subset=["branch_id"], keep="first").copy()

    if not lake_starts.empty:
        fallback_rank = 1
        for row in lake_starts.itertuples(index=False):
            bid = str(getattr(row, "branch_id", "")).strip()
            if not bid:
                continue
            sel, d_str, f_idx, area_km2, mask_pick_mode = select_branch_start_mask(
                branch_start_df=branch_start_df,
                topology_features_gdf=topology_features_gdf,
                lake_id=lake_id,
                branch_id=bid,
            )
            if sel.empty or not d_str:
                continue
            raw_order = pd.to_numeric(getattr(row, "branch_start_order", np.nan), errors="coerce")
            if pd.isna(raw_order):
                raw_order = pd.to_numeric(getattr(row, "branch_order", np.nan), errors="coerce")
            branch_index = int(raw_order) if pd.notna(raw_order) else int(fallback_rank)
            if branch_index <= 0:
                branch_index = int(fallback_rank)
            fallback_rank += 1
            specs.append(
                {
                    "branch_index": int(branch_index),
                    "branch_id": bid,
                    "branch_label": f"Branch {int(branch_index)}",
                    "mask_area_km2": float(area_km2) if pd.notna(area_km2) else float(sel.iloc[0].get("area_km2", 0)),
                    "mask_date": d_str,
                    "mask_feature_index": int(f_idx) if f_idx is not None else 0,
                    "mask_gdf": sel[["geometry"]].copy(),
                    "active_start_dt": period_start,
                    "active_end_dt": period_end,
                    "branch_start_date": normalize_date_str(getattr(row, "date", "")),
                    "merge_date": "",
                    "branch_class": "MaskOnly",
                    "mask_role": "branch_start_feature",
                    "mask_pick_mode": mask_pick_mode,
                }
            )
    return specs


def process_lake(
    lake_id,
    pixc_version,
    summary_row,
    branch_df,
    branch_start_df,
    topology_features_gdf,
    swot_gdb_path,
    tiles_gdf,
    pixc_map,
    lake_geom,
):
    specs = build_mask_specs(summary_row, branch_df, branch_start_df, topology_features_gdf, lake_id)
    if not specs:
        return pd.DataFrame(), pd.DataFrame(), "no_mask", []

    layers_by_day = {}
    pixc_files_by_day = {}
    if DIRECT_PIXC_PREFERRED:
        pixc_files_by_day = build_candidate_pixc_files_by_day(lake_geom, summary_row, tiles_gdf, pixc_map)
    if not pixc_files_by_day:
        layers_by_day = list_swot_layers_by_day(swot_gdb_path)
    if not pixc_files_by_day and not layers_by_day:
        return pd.DataFrame(), pd.DataFrame(), "no_pixc_source", specs

    branch_rows = []
    audit_rows = []
    layer_cache = {}
    pixc_cache = {}
    p_start, p_end = resolve_water_period_window(summary_row)
    plot_force_noise_before_dt = resolve_plot_force_noise_before_date()

    for spec in specs:
        w_start = spec["active_start_dt"] if pd.notna(spec["active_start_dt"]) else p_start
        w_end = spec["active_end_dt"] if pd.notna(spec["active_end_dt"]) else p_end
        if pd.isna(w_start) or pd.isna(w_end):
            continue
        if w_end < w_start:
            continue

        for day_dt in pd.date_range(w_start.normalize(), w_end.normalize()):
            day_norm = day_dt.normalize()
            layer_names = layers_by_day.get(day_norm, [])
            nc_paths = pixc_files_by_day.get(day_norm, [])
            audit_base = {
                "lake_id": int(lake_id),
                "pixc_version": str(pixc_version),
                "branch_index": int(spec["branch_index"]),
                "branch_id": str(spec.get("branch_id", "")),
                "date": day_norm.strftime("%Y%m%d"),
                "date_dt": day_norm,
                "mask_role": spec["mask_role"],
                "plot_force_noise_pre_0605": bool(pd.notna(plot_force_noise_before_dt) and day_norm < plot_force_noise_before_dt),
            }
            if not layer_names and not nc_paths:
                audit_rows.append({**audit_base, "status": "no_pixc", "reason_short": "NoPIXC"})
                continue

            if nc_paths:
                day_candidates, fails = extract_wse_candidates_for_day_from_pixc(
                    nc_paths=nc_paths,
                    lake_geom=lake_geom,
                    mask_gdf=spec["mask_gdf"],
                    tiles_gdf=tiles_gdf,
                    pixc_cache=pixc_cache,
                    lake_id=lake_id,
                    day_text=day_norm.strftime("%Y%m%d"),
                    pixc_gdb_path=None,
                    exported_pixc_layers=None,
                )
            else:
                day_candidates, fails = extract_wse_candidates_for_day(swot_gdb_path, layer_names, spec["mask_gdf"], layer_cache)
            if not day_candidates:
                _code, reason_short = classify_failure_reasons(fails)
                audit_rows.append({**audit_base, "status": "no_valid_wse", "reason_short": reason_short, "reason_detail": "|".join(fails)})
                if VERBOSE_PIXC_PROGRESS:
                    progress(f"[06_1] lake {int(lake_id)} {day_norm.strftime('%Y%m%d')} -> no_valid_wse ({reason_short})")
                continue

            audit_rows.append(
                {
                    **audit_base,
                    "status": "valid_wse",
                    "candidate_count": int(len(day_candidates)),
                    "source_layers": "|".join(str(item["source_layer"]) for item in day_candidates),
                    "cross_track_mode": "|".join(sorted({str(item.get("cross_track_mode", "standard")) for item in day_candidates})),
                    "cross_track_relaxed": bool(any(bool(item.get("cross_track_relaxed", False)) for item in day_candidates)),
                }
            )
            if VERBOSE_PIXC_PROGRESS:
                best_day = min(
                    day_candidates,
                    key=lambda item: (
                        int(item.get("type_rank", 99)),
                        -int(item.get("count", 0)),
                        float(item.get("wse_std", np.inf)),
                    ),
                )
                progress(
                    f"[06_1] lake {int(lake_id)} {day_norm.strftime('%Y%m%d')} -> day_wse "
                    f"{float(best_day['wse_m']):.3f}m from {best_day['source_layer']}"
                )
            for candidate in day_candidates:
                branch_rows.append(
                    {
                        "lake_id": int(lake_id),
                        "pixc_version": str(pixc_version),
                        "date": day_norm.strftime("%Y%m%d"),
                        "date_dt": day_norm,
                        "branch_index": int(spec["branch_index"]),
                        "branch_id": str(spec.get("branch_id", "")),
                        "branch_label": spec["branch_label"],
                        "wse_m": candidate["wse_m"],
                        "wse_std": candidate["wse_std"],
                        "count": candidate["count"],
                        "type": candidate["type"],
                        "mask_area_km2": candidate.get("mask_area_km2", spec["mask_area_km2"]),
                        "mask_date": spec["mask_date"],
                        "mask_feature_index": candidate.get("mask_feature_index", spec["mask_feature_index"]),
                        "source_layer": candidate["source_layer"],
                        "sigma_iters": candidate["sigma_iters"],
                        "type_rank": candidate["type_rank"],
                        "wse_stage": candidate.get("wse_stage", ""),
                        "sigma_filtered": bool(candidate.get("sigma_filtered", False)),
                        "pre_filter_count": int(candidate.get("pre_filter_count", candidate["count"])),
                        "post_filter_count": int(candidate.get("post_filter_count", candidate["count"])),
                        "mask_role": spec["mask_role"],
                        "mask_pick_mode": spec["mask_pick_mode"],
                        "cross_track_mode": candidate.get("cross_track_mode", "standard"),
                        "cross_track_relaxed": bool(candidate.get("cross_track_relaxed", False)),
                        "plot_force_noise_pre_0605": bool(pd.notna(plot_force_noise_before_dt) and day_norm < plot_force_noise_before_dt),
                    }
                )

    return pd.DataFrame(branch_rows), pd.DataFrame(audit_rows), "ok", specs


def build_lake_masks_gdf(lake_id, pixc_version, specs):
    if not specs:
        return gpd.GeoDataFrame()
    lake_masks = []
    for spec in specs:
        mask_gdf = spec["mask_gdf"].copy()
        mask_gdf["lake_id"] = int(lake_id)
        mask_gdf["pixc_version"] = str(pixc_version)
        mask_gdf["branch_idx"] = int(spec["branch_index"])
        mask_gdf["branch_id"] = str(spec.get("branch_id", ""))
        mask_gdf["branch_label"] = str(spec.get("branch_label", ""))
        mask_gdf["mask_role"] = str(spec["mask_role"])
        mask_gdf["mask_date"] = str(spec["mask_date"])
        mask_gdf["area_km2"] = float(spec["mask_area_km2"])
        mask_gdf["pick_mode"] = str(spec.get("mask_pick_mode", ""))
        mask_gdf["Mask_ID"] = (
            f"L{int(lake_id)}_{str(pixc_version)}_B{int(spec['branch_index'])}"
            if int(spec["branch_index"]) != 0
            else f"L{int(lake_id)}_{str(pixc_version)}_T"
        )
        lake_masks.append(mask_gdf)
    if not lake_masks:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(lake_masks, ignore_index=True), geometry="geometry", crs=lake_masks[0].crs)


def export_lake_masks(mask_dir, lake_id, pixc_version, specs):
    combined = build_lake_masks_gdf(lake_id, pixc_version, specs)
    if combined.empty:
        return combined
    os.makedirs(mask_dir, exist_ok=True)
    combined.to_file(
        os.path.join(mask_dir, f"lake_{int(lake_id)}_{str(pixc_version)}.shp"),
        driver="ESRI Shapefile",
        encoding="utf-8",
    )
    return combined


def build_raw_wse_gdf(lake_id, branch_rows_df, specs):
    if branch_rows_df is None or branch_rows_df.empty or not specs:
        return gpd.GeoDataFrame()
    spec_map = {}
    for spec in specs:
        mask_gdf = spec.get("mask_gdf")
        if mask_gdf is None or mask_gdf.empty:
            continue
        spec_map[int(spec["branch_index"])] = {
            "geometry": mask_gdf.iloc[0].geometry,
            "mask_role": str(spec.get("mask_role", "")),
            "mask_pick_mode": str(spec.get("mask_pick_mode", "")),
            "mask_date": str(spec.get("mask_date", "")),
            "mask_feature_index": int(spec.get("mask_feature_index", 0) or 0),
        }
    rows = []
    for row in branch_rows_df.to_dict("records"):
        branch_index = int(pd.to_numeric(row.get("branch_index"), errors="coerce") or 0)
        spec_info = spec_map.get(branch_index)
        if spec_info is None:
            continue
        out_row = dict(row)
        out_row["lake_id"] = int(lake_id)
        out_row["geometry"] = spec_info["geometry"]
        rows.append(out_row)
    if not rows:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=specs[0]["mask_gdf"].crs)


def normalize_legacy_branch_rows_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_WSE_COLUMNS)
    work = df.copy()
    if "cross_track_mode" not in work.columns:
        work["cross_track_mode"] = "standard"
    if "cross_track_relaxed" not in work.columns:
        work["cross_track_relaxed"] = False
    return ensure_table_columns(work, RAW_WSE_COLUMNS)


def normalize_legacy_audit_rows_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    work = df.copy()
    if "cross_track_mode" not in work.columns:
        work["cross_track_mode"] = "standard"
    if "cross_track_relaxed" not in work.columns:
        work["cross_track_relaxed"] = False
    return ensure_table_columns(work, AUDIT_COLUMNS)


def run_lake_task(task):
    lake_id = int(task["lake_id"])
    pixc_version = str(task["pixc_version"])
    summary_row = task["summary_row"]
    branch_df = task["branch_df"]
    branch_start_df = task["branch_start_df"]
    topology_features_gdf = task["topology_features_gdf"]
    swot_gdb_path = task["swot_gdb_path"]
    tiles_gdf = task["tiles_gdf"]
    pixc_map = task["pixc_map"]
    lake_geom = task["lake_geom"]
    legacy_module = task.get("legacy_module")
    if legacy_module is not None:
        branch_start_df_eff = branch_start_df
        forced_main_date = FORCE_MAIN_MASK_DATE_BY_LAKE.get(int(lake_id), "")
        if forced_main_date:
            peak_branch_id = str(summary_row.get("peak_branch_id", "")).strip()
            if peak_branch_id:
                branch_start_df_eff = apply_forced_main_mask_to_branch_start_df(
                    branch_start_df=branch_start_df,
                    topology_features_gdf=topology_features_gdf,
                    lake_id=lake_id,
                    peak_branch_id=peak_branch_id,
                    forced_date_text=forced_main_date,
                )
        branch_rows_df, audit_rows_df, status, specs = legacy_module.process_lake(
            lake_id=lake_id,
            summary_row=summary_row,
            branch_df=branch_df,
            branch_start_df=branch_start_df_eff,
            topology_features_gdf=topology_features_gdf,
            swot_gdb_path=swot_gdb_path,
        )
        branch_rows_df = normalize_legacy_branch_rows_df(branch_rows_df)
        audit_rows_df = normalize_legacy_audit_rows_df(audit_rows_df)
        if not branch_rows_df.empty:
            branch_rows_df["pixc_version"] = str(pixc_version)
        if not audit_rows_df.empty:
            audit_rows_df["pixc_version"] = str(pixc_version)
    else:
        branch_rows_df, audit_rows_df, status, specs = process_lake(
            lake_id=lake_id,
            pixc_version=pixc_version,
            summary_row=summary_row,
            branch_df=branch_df,
            branch_start_df=branch_start_df,
            topology_features_gdf=topology_features_gdf,
            swot_gdb_path=swot_gdb_path,
            tiles_gdf=tiles_gdf,
            pixc_map=pixc_map,
            lake_geom=lake_geom,
        )
    return {
        "lake_id": lake_id,
        "pixc_version": pixc_version,
        "size_category": str(summary_row.get("size_category", "")),
        "branch_rows_df": branch_rows_df,
        "audit_rows_df": audit_rows_df,
        "status": status,
        "specs": specs,
    }


def ensure_table_columns(df, columns):
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = pd.NA
    return work[columns].copy()


def get_per_lake_output_paths(method_paths, lake_id, pixc_version):
    version_text = str(pixc_version)
    return {
        "raw_csv": os.path.join(method_paths["per_lake_raw_dir"], f"lake_{int(lake_id)}_raw_wse_{version_text}.csv"),
        "audit_csv": os.path.join(method_paths["per_lake_audit_dir"], f"lake_{int(lake_id)}_day_audit_{version_text}.csv"),
        "summary_csv": os.path.join(method_paths["per_lake_summary_dir"], f"lake_{int(lake_id)}_summary_{version_text}.csv"),
        "mask_shp": os.path.join(method_paths["mask_dir"], f"lake_{int(lake_id)}_{version_text}.shp"),
    }


def load_completed_task_outputs(method_paths, lake_id, pixc_version):
    out_paths = get_per_lake_output_paths(method_paths, lake_id, pixc_version)
    if not (
        os.path.exists(out_paths["raw_csv"])
        and os.path.exists(out_paths["audit_csv"])
        and os.path.exists(out_paths["summary_csv"])
    ):
        return None
    try:
        summary_df = pd.read_csv(out_paths["summary_csv"], encoding="utf-8-sig")
        raw_df = pd.read_csv(out_paths["raw_csv"], encoding="utf-8-sig")
        audit_df = pd.read_csv(out_paths["audit_csv"], encoding="utf-8-sig")
    except Exception:
        return None
    if summary_df.empty:
        return None
    summary_row = ensure_table_columns(summary_df.head(1), SUMMARY_COLUMNS).iloc[0].to_dict()
    mask_gdf = gpd.GeoDataFrame()
    if os.path.exists(out_paths["mask_shp"]):
        try:
            mask_gdf = gpd.read_file(out_paths["mask_shp"])
        except Exception:
            mask_gdf = gpd.GeoDataFrame()
    return {
        "summary_row": summary_row,
        "raw_df": ensure_table_columns(raw_df, RAW_WSE_COLUMNS),
        "audit_df": ensure_table_columns(audit_df, AUDIT_COLUMNS),
        "raw_wse_gdf": gpd.GeoDataFrame(),
        "mask_gdf": mask_gdf,
    }


def write_per_lake_outputs(method_paths, lake_id, pixc_version, branch_rows_df, audit_rows_df, summary_row, raw_wse_gdf=None):
    os.makedirs(method_paths["per_lake_raw_dir"], exist_ok=True)
    os.makedirs(method_paths["per_lake_audit_dir"], exist_ok=True)
    os.makedirs(method_paths["per_lake_summary_dir"], exist_ok=True)

    raw_df = ensure_table_columns(branch_rows_df, RAW_WSE_COLUMNS)
    audit_df = ensure_table_columns(audit_rows_df, AUDIT_COLUMNS)
    summary_df = ensure_table_columns(pd.DataFrame([summary_row]), SUMMARY_COLUMNS)
    out_paths = get_per_lake_output_paths(method_paths, lake_id, pixc_version)

    raw_df.to_csv(out_paths["raw_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(out_paths["audit_csv"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_paths["summary_csv"], index=False, encoding="utf-8-sig")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    legacy_module = load_legacy_417_module()
    for method_name in METHOD_NAMES:
        method_paths = get_method_paths(method_name)
        os.makedirs(method_paths["output_dir"], exist_ok=True)
        os.makedirs(method_paths["per_lake_raw_dir"], exist_ok=True)
        os.makedirs(method_paths["per_lake_audit_dir"], exist_ok=True)
        os.makedirs(method_paths["per_lake_summary_dir"], exist_ok=True)
        required_paths = [
            method_paths["input_05_1_topology_gdb"],
            method_paths["input_05_2_branch_start_gdb"],
        ]
        if not all(os.path.exists(path) for path in required_paths):
            progress(f"[06_1] skip {method_name}: missing required inputs")
            continue

        topology_features_gdf = load_topology_features_gdf(method_paths["input_05_1_topology_gdb"])
        branch_start_df = load_branch_start_features_gdf(method_paths["input_05_2_branch_start_gdb"])
        effective_window_df = load_effective_boundary_windows(method_paths["input_04_6_index_csv"])
        tiles_gdf = load_tiles_gdf(TILES_SHP)
        lake_geom_map = load_lake_buffered_geom_map(LAKES_BUFFERED_SHP)
        pixc_map = build_pixc_file_map(PIXC_DIR) if DIRECT_PIXC_PREFERRED else {}
        existing_repair_state = load_pixc_repair_state(method_paths["pixc_repair_log_csv"])
        with GLOBAL_PIXC_REPAIR_LOCK:
            GLOBAL_PIXC_REPAIR_STATUS.clear()
            GLOBAL_PIXC_REPAIR_STATUS.update(existing_repair_state)
            GLOBAL_PIXC_REPAIR_ROWS.clear()
        if topology_features_gdf.empty or branch_start_df.empty:
            progress(f"[06_1] skip {method_name}: empty topology inputs")
            continue
        if DIRECT_PIXC_PREFERRED and (tiles_gdf.empty or not pixc_map):
            progress(f"[06_1] {method_name}: direct PIXC unavailable, will fallback to per-lake GDB if present")

        lake_summary_df, branch_table_df = build_lake_mask_metadata_from_branch_starts(branch_start_df)
        if not effective_window_df.empty:
            lake_summary_df = lake_summary_df.merge(effective_window_df, on="lake_id", how="left")
            lake_summary_df["water_period_start"] = lake_summary_df["effective_boundary_start"].fillna(lake_summary_df["water_period_start"])
            lake_summary_df["water_period_end"] = lake_summary_df["effective_boundary_end"].fillna(lake_summary_df["water_period_end"])
        else:
            lake_summary_df["effective_boundary_start"] = ""
            lake_summary_df["effective_boundary_end"] = ""
        lake_summary_df.to_csv(method_paths["lake_summary_csv"], index=False, encoding="utf-8-sig")
        branch_table_df.to_csv(method_paths["branch_table_csv"], index=False, encoding="utf-8-sig")

        all_branch_rows = []
        all_audit_rows = []
        all_mask_gdfs = []
        all_raw_wse_gdfs = []
        summary_rows = []

        valid_lake_ids = set(pd.to_numeric(branch_start_df["lake_id"], errors="coerce").dropna().astype(int).tolist())
        scoped_summary = lake_summary_df[
            lake_summary_df["size_category"].astype(str).isin(TARGET_SIZE_CATEGORIES)
            & lake_summary_df["lake_id"].isin(valid_lake_ids)
        ].copy()
        if TARGET_LAKE_IDS:
            target_set = {int(x) for x in TARGET_LAKE_IDS}
            scoped_summary = scoped_summary[scoped_summary["lake_id"].astype(int).isin(target_set)].copy()
        total_lakes = len(scoped_summary)
        total_tasks = int(total_lakes) * int(len(PIXC_VERSION_SPECS))
        progress(
            f"[06_1] {method_name}: processing {total_lakes} lakes x {len(PIXC_VERSION_SPECS)} versions "
            f"= {total_tasks} tasks with {MAX_WORKERS} workers"
        )
        os.makedirs(method_paths["mask_dir"], exist_ok=True)

        tasks = []
        restored_count = 0
        for summary_row in scoped_summary.to_dict("records"):
            lake_id = int(summary_row["lake_id"])
            for version_spec in PIXC_VERSION_SPECS:
                pixc_version = str(version_spec["pixc_version"])
                if RESUME_ENABLED:
                    restored = load_completed_task_outputs(method_paths, lake_id, pixc_version)
                    if restored is not None:
                        restored_count += 1
                        if not restored["raw_df"].empty:
                            all_branch_rows.append(restored["raw_df"])
                        if not restored["audit_df"].empty:
                            all_audit_rows.append(restored["audit_df"])
                        if restored["mask_gdf"] is not None and not restored["mask_gdf"].empty:
                            all_mask_gdfs.append(restored["mask_gdf"])
                        if restored["raw_wse_gdf"] is not None and not restored["raw_wse_gdf"].empty:
                            all_raw_wse_gdfs.append(restored["raw_wse_gdf"])
                        summary_rows.append(restored["summary_row"])
                        progress(
                            f"[06_1] {method_name} resume: lake {lake_id} version={pixc_version} -> reuse existing outputs"
                        )
                        continue
                tasks.append(
                    {
                        "lake_id": lake_id,
                        "pixc_version": pixc_version,
                        "summary_row": summary_row,
                        "branch_df": branch_table_df,
                        "branch_start_df": branch_start_df,
                        "topology_features_gdf": topology_features_gdf,
                        "swot_gdb_path": os.path.join(
                            INPUT_06_1_ROOT,
                            method_name,
                            version_spec["input_subdir"],
                            f"swot_pixc_{lake_id}.gdb",
                        ),
                        "tiles_gdf": tiles_gdf,
                        "pixc_map": pixc_map,
                        "lake_geom": lake_geom_map.get(lake_id),
                        "legacy_module": legacy_module,
                    }
                )

        progress(f"[06_1] {method_name}: resume restored {restored_count} tasks; remaining {len(tasks)}")
        completed_count = restored_count
        with ThreadPoolExecutor(max_workers=max(1, int(MAX_WORKERS))) as executor:
            future_map = {executor.submit(run_lake_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                lake_id = int(task["lake_id"])
                pixc_version = str(task["pixc_version"])
                completed_count += 1
                try:
                    result = future.result()
                except Exception as exc:
                    progress(
                        f"[06_1] {method_name} task {completed_count}/{total_tasks}: "
                        f"lake {lake_id} version={pixc_version} error: {exc}"
                    )
                    summary_row = {
                        "lake_id": lake_id,
                        "pixc_version": pixc_version,
                        "size_category": str(task["summary_row"].get("size_category", "")),
                        "status": "error",
                        "record_count": 0,
                        "branch_mask_count": 0,
                    }
                    summary_rows.append(summary_row)
                    write_per_lake_outputs(
                        method_paths=method_paths,
                        lake_id=lake_id,
                        pixc_version=pixc_version,
                        branch_rows_df=pd.DataFrame(),
                        audit_rows_df=pd.DataFrame(),
                        summary_row=summary_row,
                        raw_wse_gdf=gpd.GeoDataFrame(),
                    )
                    continue

                branch_rows_df = result["branch_rows_df"]
                audit_rows_df = result["audit_rows_df"]
                status = result["status"]
                specs = result["specs"]
                pixc_version = str(result["pixc_version"])
                if not branch_rows_df.empty:
                    all_branch_rows.append(branch_rows_df)
                if not audit_rows_df.empty:
                    all_audit_rows.append(audit_rows_df)
                summary_row = {
                    "lake_id": lake_id,
                    "pixc_version": pixc_version,
                    "size_category": result["size_category"],
                    "status": status,
                    "record_count": int(len(branch_rows_df)),
                    "branch_mask_count": int(len(specs)),
                }
                summary_rows.append(summary_row)
                mask_gdf = export_lake_masks(method_paths["mask_dir"], lake_id, pixc_version, specs)
                if mask_gdf is not None and not mask_gdf.empty:
                    all_mask_gdfs.append(mask_gdf)
                raw_wse_gdf = build_raw_wse_gdf(lake_id, branch_rows_df, specs)
                if raw_wse_gdf is not None and not raw_wse_gdf.empty:
                    all_raw_wse_gdfs.append(raw_wse_gdf)
                write_per_lake_outputs(
                    method_paths=method_paths,
                    lake_id=lake_id,
                    pixc_version=pixc_version,
                    branch_rows_df=branch_rows_df,
                    audit_rows_df=audit_rows_df,
                    summary_row=summary_row,
                    raw_wse_gdf=raw_wse_gdf,
                )
                progress(
                    f"[06_1] {method_name} task {completed_count}/{total_tasks}: lake {lake_id} "
                    f"version={pixc_version} status={status} records={len(branch_rows_df)}"
                )

        pd.DataFrame(summary_rows).to_csv(method_paths["summary_csv"], index=False, encoding="utf-8-sig")
        if all_branch_rows:
            pd.concat(all_branch_rows, ignore_index=True).to_csv(method_paths["raw_wse_csv"], index=False, encoding="utf-8-sig")
        if all_audit_rows:
            pd.concat(all_audit_rows, ignore_index=True).to_csv(method_paths["audit_csv"], index=False, encoding="utf-8-sig")
        if os.path.exists(method_paths["mask_gdb"]):
            import shutil as _shutil
            _shutil.rmtree(method_paths["mask_gdb"], ignore_errors=True)
        if os.path.exists(method_paths["raw_wse_gdb"]):
            import shutil as _shutil
            _shutil.rmtree(method_paths["raw_wse_gdb"], ignore_errors=True)
        if all_mask_gdfs:
            mask_out = gpd.GeoDataFrame(pd.concat(all_mask_gdfs, ignore_index=True), geometry="geometry", crs=all_mask_gdfs[0].crs)
            try:
                mask_out.to_file(method_paths["mask_gdb"], layer=method_paths["mask_gdb_layer"], driver="OpenFileGDB")
            except Exception as exc:
                progress(f"[06_1] warn: skip combined mask_gdb export for {method_name}: {exc}")
        if all_raw_wse_gdfs:
            raw_wse_out = gpd.GeoDataFrame(pd.concat(all_raw_wse_gdfs, ignore_index=True), geometry="geometry", crs=all_raw_wse_gdfs[0].crs)
            try:
                raw_wse_out.to_file(method_paths["raw_wse_gdb"], layer=method_paths["raw_wse_gdb_layer"], driver="OpenFileGDB")
            except Exception as exc:
                progress(f"[06_1] warn: skip combined raw_wse_gdb export for {method_name}: {exc}")
        with GLOBAL_PIXC_REPAIR_LOCK:
            repair_rows = list(GLOBAL_PIXC_REPAIR_ROWS)
        if repair_rows:
            repair_df = pd.DataFrame(repair_rows)
            if os.path.exists(method_paths["pixc_repair_log_csv"]):
                try:
                    existing_df = pd.read_csv(method_paths["pixc_repair_log_csv"], encoding="utf-8-sig")
                    repair_df = pd.concat([existing_df, repair_df], ignore_index=True)
                except Exception:
                    pass
            if "granule_name" in repair_df.columns and "status" in repair_df.columns:
                repair_df = repair_df.drop_duplicates(subset=["granule_name", "status"], keep="last").copy()
            repair_df = ensure_table_columns(repair_df, PIXC_REPAIR_LOG_COLUMNS)
            repair_df.to_csv(method_paths["pixc_repair_log_csv"], index=False, encoding="utf-8-sig")
        progress(f"[06_1] completed {method_name}")


if __name__ == "__main__":
    main()





