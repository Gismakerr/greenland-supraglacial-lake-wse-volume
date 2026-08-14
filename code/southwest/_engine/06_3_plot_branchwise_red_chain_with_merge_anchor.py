"""
Stage 06_3 (refactored):
- Keep only currently used denoise/classification logic.
- Output plots directly under "鏈夊櫔澹? and "鏃犲櫔澹? (no extra subfolders).
- No-noise plot style follows the reference look from copy_code/06_2.
"""

import os
import sys
import importlib.util
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = SCRIPT_DIR / "copy_code" / "06_2_plot_single_red_chain_with_uncertainty.py"

TARGET_SIZE_CATEGORIES = ["Large", "Medium"]
EXCLUDED_LAKE_IDS = set()
# Hard exclusion at branch level: these branches are skipped entirely in 06_3 outputs.
EXCLUDED_LAKE_BRANCHES = set()
OUTPUT_ROOT_NAME = "06_3_plot_branchwise_red_chain_with_merge_anchor"
PIXC_VERSIONS = ["D"]
MAX_LAKES_TO_PROCESS = 0
PROCESS_ONLY_LAKE_IDS = []
ORIGINAL_GROUP_FOLDER_NAME = "original"
RANKED_GROUP_FOLDER_NAME = "ranked"

DISABLE_DEGRADED_POINTS_IN_CROSSTRACK_BAND = True
INITIAL_RED_GROUP_MIN_DAYS = 3
EDGE_GRAY_SHORT_CHAIN_MAX_LEN = 2
RED_POINTS_MIN_TO_SAVE = int(os.environ.get("RED_POINTS_MIN_TO_SAVE", "5"))
OUTPUT_ROOT_NAME = f"{OUTPUT_ROOT_NAME}_redmin{int(RED_POINTS_MIN_TO_SAVE)}"
RED_DAYS_LT5_FOLDER_NAME = "red_days_lt_5"
LOCAL_EXTREMA_NOISE_THRESHOLD_M = 0.5
HIGH_RATIO_NOISE_THRESHOLD = 0.3
HIGH_RATIO_NOISE_FOLDER_NAME = "high_ratio_noise_gt_0p3"

# Manual override rules (C-only in current workflow)
# version field is retained for compatibility, but rules are only applied to C.
MANUAL_RULES_TARGET_VERSIONS = {"C"}
MANUAL_RULE_BLOCKS = [
    {"lake_id": 1, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-08-01", "2024-08-02", "2024-08-03"], "reason": "manual_lake1_aug01_to_aug03"}]},
    {"lake_id": 2, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-08-12"], "reason": "manual_lake2_vc_aug12"}, {"days": ["2024-08-14"], "reason": "manual_lake2_vc_aug14"}]},
    {"lake_id": 2, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake2_branch1_vc_aug14_endpoint"}], "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 2, "branch_index": 2, "version": "C", "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 8, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-07-04"], "reason": "manual_lake8_vc_jul04"}]},
    {"lake_id": 8, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-07-04"], "reason": "manual_lake8_branch1_vc_jul04_day"}], "force_exact_point_noise": [{"day": "2024-07-04", "wse": 771.673096, "reason": "manual_lake8_branch1_vc_jul04"}], "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_local_peak_noise"}]},
    {"lake_id": 11, "branch_index": 1, "version": "C", "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 16, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-06-26"], "reason": "manual_lake16_vc_jun26"}]},
    {"lake_id": 16, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-06-26"], "reason": "manual_lake16_branch1_vc_jun26_day"}], "force_exact_point_noise": [{"day": "2024-06-26", "wse": 749.726746, "reason": "manual_lake16_branch1_vc_jun26"}], "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 16, "branch_index": 2, "version": "C", "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 22, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake22_vc_aug14"}]},
    {"lake_id": 22, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake22_branch1_vc_aug14_day"}], "force_exact_point_noise": [{"day": "2024-08-14", "wse": 499.324023, "reason": "manual_lake22_branch1_vc_aug14"}]},
    {"lake_id": 31, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-08-09"], "reason": "manual_lake31_aug09"}]},
    {"lake_id": 31, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-06-08", "2024-06-09"], "reason": "manual_lake31_pre0610_noise"}]},
    {"lake_id": 33, "branch_index": None, "version": "C", "force_not_noise": [{"days": ["2024-07-06", "2024-07-07"]}]},
    {"lake_id": 46, "branch_index": 1, "version": "C", "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 54, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake54_vc_aug14"}]},
    {"lake_id": 54, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake54_branch1_vc_aug14_day"}], "force_exact_point_noise": [{"day": "2024-08-14", "wse": 598.879395, "reason": "manual_lake54_branch1_vc_aug14"}]},
    {"lake_id": 60, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-07-05"], "reason": "manual_lake60_jul05"}]},
    {"lake_id": 74, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-07-25"], "reason": "manual_lake74_vc_jul25"}]},
    {"lake_id": 74, "branch_index": 2, "version": "C", "force_noise": [{"days": ["2024-07-25"], "reason": "manual_lake74_branch2_vc_jul25_day"}], "force_exact_point_noise": [{"day": "2024-07-25", "wse": 932.022180, "reason": "manual_lake74_branch2_vc_jul25"}]},
    {"lake_id": 117, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-07-26", "2024-07-28", "2024-07-29"], "reason": "manual_lake117_vc_jul26_28_29"}]},
    {"lake_id": 132, "branch_index": 1, "version": "C", "force_local_max_noise": [{"day": "2024-08-02", "reason": "manual_aug02_peak_noise"}]},
    {"lake_id": 165, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake165_vc_aug14"}]},
    {"lake_id": 165, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-14"], "reason": "manual_lake165_branch1_vc_aug14_day"}], "force_exact_point_noise": [{"day": "2024-08-14", "wse": 817.874084, "reason": "manual_lake165_branch1_vc_aug14"}]},
    {"lake_id": 196, "branch_index": None, "version": "C", "force_noise": [{"days": ["2024-07-05"], "reason": "manual_lake196_jul05"}]},
    {"lake_id": 11, "branch_index": 1, "version": "C", "force_exact_point_noise": [{"day": "2024-07-04", "wse": 705.052307, "reason": "manual_lake11_branch1_vc_jul04"}]},
    {"lake_id": 13, "branch_index": 1, "version": "C", "force_exact_point_noise": [{"day": "2024-07-02", "wse": 621.578064, "reason": "manual_lake13_branch1_vc_jul02"}]},
    {"lake_id": 16, "branch_index": 1, "version": "C", "force_exact_point_noise": [{"day": "2024-07-04", "wse": 755.162354, "reason": "manual_lake16_branch1_vc_jul04"}]},
    {"lake_id": 26, "branch_index": 2, "version": "C", "force_exact_point_noise": [{"day": "2024-07-29", "wse": 586.889526, "reason": "manual_lake26_branch2_vc_jul29"}]},
    {"lake_id": 28, "branch_index": 1, "version": "C", "force_exact_point_noise": [{"day": "2024-07-26", "wse": 439.530570, "reason": "manual_lake28_branch1_vc_jul26"}]},
    {"lake_id": 58, "branch_index": 1, "version": "C", "force_exact_point_noise": [{"day": "2024-07-28", "wse": 752.332031, "reason": "manual_lake58_branch1_vc_jul28"}]},
    {"lake_id": 130, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-01"], "reason": "manual_lake130_branch1_vc_aug01"}], "force_exact_point_noise": [{"day": "2024-08-04", "wse": 639.349365, "reason": "manual_lake130_branch1_vc_aug04_gray_drop"}]},
    {"lake_id": 15, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-07-24", "2024-07-28"], "reason": "manual_lake15_branch1_vc_jul24_jul28_box_noise"}]},
    {"lake_id": 24, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-01", "2024-08-04"], "reason": "manual_lake24_branch1_vc_aug01_aug04_box_noise"}]},
    {"lake_id": 101, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-07-02"], "reason": "manual_lake101_branch1_vc_jul02_box_noise"}]},
    {"lake_id": 126, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-07-09"], "reason": "manual_lake126_branch1_vc_jul09_box_noise"}]},
    {"lake_id": 155, "branch_index": 1, "version": "C", "force_noise": [{"days": ["2024-08-08", "2024-08-09", "2024-08-10"], "reason": "manual_lake155_branch1_vc_aug08_to_aug10_box_noise"}]},
]

# Plot style config
PLOT_FIG_WIDTH = 14
PLOT_FIG_HEIGHT = 6
PLOT_DPI = 170
PLOT_BG_COLOR = "white"
AX_BG_COLOR = "white"
PLOT_LINESTYLE = "--"
PLOT_LINEWIDTH = 1.35
PLOT_LINE_ALPHA = 0.9
PLOT_RED_POINT_SIZE = 34
PLOT_GRAY_POINT_SIZE = 16
PLOT_NOISE_POINT_SIZE = 45
PLOT_NOISE_LINEWIDTH = 1.2
PLOT_ERR_LINEWIDTH = 0.9
PLOT_ERR_CAPSIZE = 2.5
PLOT_ERR_ALPHA = 0.8
PLOT_TITLE_FONTSIZE = 30
PLOT_TITLE_SHOW_BRANCH = False
PLOT_TITLE_X = 0.02
PLOT_TITLE_Y = 0.965
PLOT_YLABEL_FONTSIZE = 18
PLOT_GRID_LINEWIDTH = 0.9
PLOT_GRID_ALPHA = 0.28
PLOT_GRID_COLOR = "#B5B5B5"
PLOT_XTICK_INTERVAL_DAYS = 8
PLOT_XTICK_ROTATION = 20
PLOT_XTICK_FONTSIZE = 30
PLOT_YTICK_FONTSIZE = 30
PLOT_XTICK_PAD = 24
PLOT_YTICK_PAD = 4
PLOT_SPINE_LINEWIDTH = 1.5
PLOT_TICK_LINEWIDTH = 1.5
PLOT_TICK_LENGTH = 6
PLOT_SPINE_COLOR = "#202020"
PLOT_LEGEND_FONTSIZE = 9
PLOT_Y_PAD_MIN = 0.20
PLOT_Y_PAD_RATIO = 0.12
PLOT_Y_PAD_FLAT = 0.50
PLOT_Y_TOP_EXTRA = 0.18
PLOT_X_GLOBAL_START = pd.Timestamp("2024-06-01")
PLOT_X_TICK_START = pd.Timestamp("2024-06-01")
PLOT_X_GLOBAL_END = pd.Timestamp("2024-08-15")
FORCE_NOISE_BEFORE_0605_CUTOFF = pd.Timestamp("2024-06-05")
PRE_JUNE13_NOISE_CUTOFF = pd.Timestamp("2024-06-13")
PLOT_MARGIN_LEFT = 0.09
PLOT_MARGIN_RIGHT = 0.985
PLOT_MARGIN_RIGHT_WITH_RIGHT_AXIS = 0.93
PLOT_MARGIN_BOTTOM = 0.26
PLOT_MARGIN_TOP = 0.90
PLOT_SHOW_X_LABEL = False
PLOT_SHOW_Y_LABEL = False
PLOT_XLABEL_TEXT = "Date"
PLOT_YLABEL_TEXT = "WSE (m)"

DUAL_AREA_FOLDER_NAME = "dual_axis_area_wse"
RED_WSE_RECORDS_FOLDER_NAME = "red_wse_records"
PROJECT_ROOT_DIR = SCRIPT_DIR.parent
RESULT_ROOT_DIR = Path(r"X:\\2024_西南水位曲线\\result1")
FILTERED_LAKE_SHP_PATH = RESULT_ROOT_DIR / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons.shp"
MAP_MAIN_DIR = RESULT_ROOT_DIR / "鐢诲浘" / "data" / "main"
AREA_COLOR = "#2F6DF6"
AREA_LINESTYLE = "-"
AREA_LINEWIDTH = 1.5
AREA_LINE_ALPHA = 0.9
AREA_POINT_SIZE = PLOT_RED_POINT_SIZE
AREA_TICK_FONTSIZE = PLOT_YTICK_FONTSIZE
PLOT_SHOW_AREA_Y_LABEL = False
PLOT_AREA_YLABEL_TEXT = "Area (km²)"
VC_COLOR = "#2F6DF6"
VD_COLOR = "#D62728"


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
    PROCESS_ONLY_LAKE_IDS = _env_target_lake_ids

_env_pixc_versions = [v.upper() for v in _parse_env_list("PIXC_VERSIONS")]
if _env_pixc_versions:
    PIXC_VERSIONS = [v for v in _env_pixc_versions if v in {"C", "D"}]

_env_output_root_dir = str(os.environ.get("OUTPUT_ROOT_DIR", "")).strip()
if _env_output_root_dir:
    RESULT_ROOT_DIR = Path(_env_output_root_dir)
    FILTERED_LAKE_SHP_PATH = RESULT_ROOT_DIR / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons.shp"
    MAP_MAIN_DIR = RESULT_ROOT_DIR / "画图" / "data" / "main"


def load_base_module():
    module_name = "stage06_2_base_module_for_06_3"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(BASE_SCRIPT_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base module: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # Force project-local paths to avoid garbled hard-coded Chinese paths in copied script.
    output_root_dir = str(RESULT_ROOT_DIR)
    module.OUTPUT_ROOT_DIR = output_root_dir
    module.INPUT_05_2_DIR = os.path.join(output_root_dir, "05_2_plot_singlebranch_from_05_1_04_6_1")
    module.INPUT_06_1_DIR = os.path.join(output_root_dir, "06_2_extract_raw_wse")
    module.OUTPUT_DIR = os.path.join(output_root_dir, OUTPUT_ROOT_NAME)
    return module


def normalize_size_filters(values):
    if not values:
        return []
    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text.lower())
    return sorted(set(normalized))


def normalize_pixc_version(value):
    text = str(value).strip().upper()
    if text in {"C", "D"}:
        return text
    return ""


def make_method_paths(base_module, method_name):
    input_05_2_dir = os.path.join(base_module.INPUT_05_2_DIR, method_name)
    input_06_1_dir = os.path.join(base_module.INPUT_06_1_DIR, method_name)
    output_dir = os.path.join(base_module.OUTPUT_ROOT_DIR, OUTPUT_ROOT_NAME, method_name)
    return {
        "method_name": method_name,
        "topology_csv": os.path.join(input_05_2_dir, "05_2_denoised_curve_topology_relations.csv"),
        "curve_point_csv": os.path.join(input_05_2_dir, "05_2_curve_point_jpg_index.csv"),
        "raw_wse_csv": os.path.join(input_06_1_dir, "06_1_raw_wse_table.csv"),
        "lake_summary_csv": os.path.join(input_06_1_dir, "06_1_mask_lake_summary.csv"),
        "output_dir": output_dir,
        "csv_dir": os.path.join(output_dir, ORIGINAL_GROUP_FOLDER_NAME, "06_3_BranchTables"),
        "plot_root_dir": os.path.join(output_dir, ORIGINAL_GROUP_FOLDER_NAME),
        "ranked_plot_root_dir": os.path.join(output_dir, RANKED_GROUP_FOLDER_NAME),
    }


def drop_degraded_points_in_cross_track_band(raw_wse_df):
    if raw_wse_df is None or raw_wse_df.empty or not DISABLE_DEGRADED_POINTS_IN_CROSSTRACK_BAND:
        return raw_wse_df
    if "cross_track_mode" not in raw_wse_df.columns:
        return raw_wse_df

    work = raw_wse_df.copy()
    mode_text = work["cross_track_mode"].astype(str).str.strip().str.lower()
    drop_mask = mode_text.eq("relaxed_no_cross_track")
    if not drop_mask.any():
        return work
    print(f"[06_3] drop relaxed_no_cross_track points: {int(drop_mask.sum())}", flush=True)
    return work.loc[~drop_mask].copy()


def keep_one_wse_per_scene(raw_wse_df):
    if raw_wse_df is None or raw_wse_df.empty:
        return raw_wse_df
    required = ["lake_id", "branch_index", "date", "source_layer"]
    if any(col not in raw_wse_df.columns for col in required):
        return raw_wse_df

    work = raw_wse_df.copy()
    source_text = work["source_layer"].fillna("").astype(str).str.strip()
    work["_scene_key"] = source_text.str.replace(r"_(PGD\d*|PIC\d*)_[^_]+$", "", regex=True)
    work["_prod_priority"] = np.where(source_text.str.contains(r"_PGD\d*_", case=False, regex=True), 0, 1)
    work["_type_rank_sort"] = pd.to_numeric(work.get("type_rank"), errors="coerce").fillna(999).astype(int)
    work["_wse_std_sort"] = pd.to_numeric(work.get("wse_std"), errors="coerce").fillna(1e9)
    work["_wse_sort"] = pd.to_numeric(work.get("wse_m"), errors="coerce").fillna(1e9)
    work = work.sort_values(
        ["lake_id", "branch_index", "date", "_scene_key", "_prod_priority", "_type_rank_sort", "_wse_std_sort", "_wse_sort"]
    ).copy()
    before = len(work)
    work = work.drop_duplicates(subset=["lake_id", "branch_index", "date", "_scene_key"], keep="first").copy()
    after = len(work)
    if after < before:
        print(f"[06_3] one-per-scene dedupe: {before} -> {after}", flush=True)
    return work.drop(columns=["_scene_key", "_prod_priority", "_type_rank_sort", "_wse_std_sort", "_wse_sort"], errors="ignore")


def load_branch_area_points(curve_point_csv_path):
    if not os.path.exists(curve_point_csv_path):
        return pd.DataFrame()
    work = pd.read_csv(curve_point_csv_path, encoding="utf-8-sig")
    if work.empty:
        return work
    work["lake_id"] = pd.to_numeric(work.get("lake_id"), errors="coerce")
    work["branch_order"] = pd.to_numeric(work.get("branch_order"), errors="coerce")
    work["area_km2"] = pd.to_numeric(work.get("area_km2"), errors="coerce")
    work["date"] = work.get("date", "").astype(str).str.extract(r"(\d{8})", expand=False)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    if "preview_name" not in work.columns:
        work["preview_name"] = ""
    if "feature_index" not in work.columns:
        work["feature_index"] = 0
    work["feature_index"] = pd.to_numeric(work["feature_index"], errors="coerce").fillna(0).astype(int)
    work = work.dropna(subset=["lake_id", "branch_order", "date_dt", "area_km2"]).copy()
    if work.empty:
        return work
    work["lake_id"] = work["lake_id"].astype(int)
    work["branch_order"] = work["branch_order"].astype(int)
    work = work.sort_values(["lake_id", "branch_order", "date_dt", "preview_name", "feature_index"]).reset_index(drop=True)
    work["x_plot_dt"] = work["date_dt"]
    for (_, _, day_dt), idx in work.groupby(["lake_id", "branch_order", work["date_dt"].dt.normalize()], sort=False).groups.items():
        day_df = work.loc[list(idx)].sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        if len(day_df) <= 1:
            continue
        center = (len(day_df) - 1) / 2.0
        shifted = [
            pd.Timestamp(base_dt) + pd.Timedelta(hours=(pos - center) * 3.0)
            for pos, base_dt in enumerate(day_df["date_dt"])
        ]
        work.loc[day_df.index, "x_plot_dt"] = shifted
    return work


def get_branch_area_series(area_points_df, lake_id, branch_index):
    if area_points_df is None or area_points_df.empty:
        return pd.DataFrame()
    out = area_points_df[
        (area_points_df["lake_id"] == int(lake_id)) & (area_points_df["branch_order"] == int(branch_index))
    ].copy()
    if out.empty:
        return out
    return out.sort_values(["date_dt", "preview_name", "feature_index"]).reset_index(drop=True)


def add_same_day_plot_offsets(records, offset_hours=3.0):
    if not records:
        return
    grouped = {}
    for idx, rec in enumerate(records):
        dt = pd.Timestamp(rec["date"])
        rec["x_plot_dt"] = dt
        grouped.setdefault(dt.normalize(), []).append(idx)
    for indices in grouped.values():
        if len(indices) <= 1:
            continue
        center = (len(indices) - 1) / 2.0
        for pos, idx in enumerate(indices):
            records[idx]["x_plot_dt"] = pd.Timestamp(records[idx]["date"]) + pd.Timedelta(hours=(pos - center) * offset_hours)


def summarize_red_wse_range(red_only_df, lake_id, branch_index, pixc_version):
    if red_only_df is None or red_only_df.empty or "wse_m" not in red_only_df.columns:
        return None
    work = red_only_df.copy()
    work["wse_m"] = pd.to_numeric(work["wse_m"], errors="coerce")
    work = work.dropna(subset=["wse_m"]).copy()
    if work.empty:
        return None
    wse_min = float(work["wse_m"].min())
    wse_max = float(work["wse_m"].max())
    return {
        "lake_id": int(lake_id),
        "branch_idx": int(branch_index),
        "pixc_ver": str(pixc_version).upper(),
        "pt_count": int(len(work)),
        "wse_min_m": wse_min,
        "wse_max_m": wse_max,
        "wse_rng_m": float(wse_max - wse_min),
    }


def compute_uncertainty_range_ratio(red_only_df):
    if red_only_df is None or red_only_df.empty:
        return None
    if "wse_m" not in red_only_df.columns or "wse_std" not in red_only_df.columns:
        return None
    wse = pd.to_numeric(red_only_df["wse_m"], errors="coerce").dropna()
    if len(wse) < 2:
        return None
    wse_range = float(wse.max() - wse.min())
    if wse_range <= 0:
        return None
    wse_std = pd.to_numeric(red_only_df["wse_std"], errors="coerce").dropna()
    if wse_std.empty:
        return None
    mean_std = float(wse_std.mean())
    return {
        "mean_std_m": mean_std,
        "wse_range_m": wse_range,
        "ratio": float(mean_std / wse_range),
    }


def load_filtered_lake_polygons():
    if not FILTERED_LAKE_SHP_PATH.exists():
        return gpd.GeoDataFrame()
    shp_gdf = gpd.read_file(FILTERED_LAKE_SHP_PATH)
    if shp_gdf.empty or "lake_id" not in shp_gdf.columns:
        return gpd.GeoDataFrame()
    shp_gdf = shp_gdf.copy()
    shp_gdf["lake_id"] = pd.to_numeric(shp_gdf["lake_id"], errors="coerce").astype("Int64")
    shp_gdf = shp_gdf.dropna(subset=["lake_id"]).copy()
    shp_gdf["lake_id"] = shp_gdf["lake_id"].astype(int)
    if EXCLUDED_LAKE_IDS:
        shp_gdf = shp_gdf.loc[~shp_gdf["lake_id"].isin(sorted(EXCLUDED_LAKE_IDS))].copy()
    return shp_gdf


def build_display_lake_id_map(shp_gdf, allowed_lake_ids=None):
    if shp_gdf is None or shp_gdf.empty or "area" not in shp_gdf.columns:
        return {}
    rank_df = shp_gdf[["lake_id", "area"]].copy()
    rank_df["area"] = pd.to_numeric(rank_df["area"], errors="coerce")
    rank_df = rank_df.dropna(subset=["area"]).copy()
    if allowed_lake_ids:
        allow_set = {int(x) for x in allowed_lake_ids}
        rank_df = rank_df.loc[rank_df["lake_id"].astype(int).isin(allow_set)].copy()
    rank_df = rank_df.sort_values(["area", "lake_id"], ascending=[False, True]).reset_index(drop=True)
    return {int(row["lake_id"]): int(idx + 1) for idx, row in rank_df.iterrows()}


def build_ranked_plot_filename(lake_id, branch_index, pixc_version=None, suffix=".png", display_lake_id=None, extra_tag=""):
    display_text = int(display_lake_id) if pd.notna(display_lake_id) else int(lake_id)
    name = f"Lake_{display_text}_orig_{int(lake_id)}_branch_{int(branch_index)}"
    if pixc_version:
        name += f"_v{str(pixc_version)}"
    if extra_tag:
        name += f"_{extra_tag}"
    return f"{name}{suffix}"


def export_lakes_with_wse_range(method_name, range_rows, shp_gdf=None, display_lake_id_map=None, out_tag=""):
    if not range_rows:
        print(f"[06_3] skip shapefile export {method_name}: no red wse range rows", flush=True)
        return
    if shp_gdf.empty or "lake_id" not in shp_gdf.columns:
        print(f"[06_3] skip shapefile export {method_name}: invalid filtered lake shp", flush=True)
        return

    range_df = pd.DataFrame(range_rows).copy()
    range_df["lake_id"] = pd.to_numeric(range_df["lake_id"], errors="coerce").astype("Int64")
    range_df = range_df.dropna(subset=["lake_id"]).copy()
    range_df["lake_id"] = range_df["lake_id"].astype(int)
    range_df = range_df.sort_values(["lake_id", "wse_rng_m", "pt_count"], ascending=[True, False, False]).reset_index(drop=True)

    lake_summary = (
        range_df.groupby("lake_id", as_index=False)
        .agg(
            max_rng_m=("wse_rng_m", "max"),
            max_wse_m=("wse_max_m", "max"),
            min_wse_m=("wse_min_m", "min"),
            curve_n=("lake_id", "size"),
            ver_cnt=("pixc_ver", lambda s: int(pd.Series(s).nunique())),
            br_cnt=("branch_idx", lambda s: int(pd.Series(s).nunique())),
            pt_sum=("pt_count", "sum"),
        )
    )
    top_curve = range_df.drop_duplicates(subset=["lake_id"], keep="first")[["lake_id", "branch_idx", "pixc_ver", "pt_count"]].rename(
        columns={
            "branch_idx": "top_br",
            "pixc_ver": "top_ver",
            "pt_count": "top_pt_n",
        }
    )
    lake_summary = lake_summary.merge(top_curve, on="lake_id", how="left")
    lake_summary["has_wse"] = 1

    shp_gdf["lake_id"] = pd.to_numeric(shp_gdf["lake_id"], errors="coerce").astype("Int64")
    out_gdf = shp_gdf.merge(lake_summary, on="lake_id", how="inner")
    if out_gdf.empty:
        print(f"[06_3] skip shapefile export {method_name}: no joined lakes", flush=True)
        return

    MAP_MAIN_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{out_tag}" if str(out_tag).strip() else ""
    out_shp = MAP_MAIN_DIR / f"lakes_with_wse_curve_maxrange{suffix}_{method_name}.shp"
    out_csv = MAP_MAIN_DIR / f"lakes_with_wse_curve_maxrange{suffix}_{method_name}.csv"
    out_gdf.to_file(out_shp, encoding="utf-8")
    pd.DataFrame(out_gdf.drop(columns="geometry")).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[06_3] wrote lake max-range shp: {out_shp}", flush=True)
    print(f"[06_3] wrote lake max-range csv: {out_csv}", flush=True)

    if display_lake_id_map:
        ranked_gdf = out_gdf.copy()
        ranked_gdf["old_lake_id"] = ranked_gdf["lake_id"].astype(int)
        local_rank_df = ranked_gdf[["old_lake_id", "area"]].rename(columns={"old_lake_id": "lake_id"}).copy()
        local_rank_map = build_display_lake_id_map(
            local_rank_df,
            allowed_lake_ids=local_rank_df["lake_id"].astype(int).tolist(),
        )
        ranked_gdf["lake_id_new"] = ranked_gdf["old_lake_id"].map(local_rank_map)
        ranked_gdf = ranked_gdf.dropna(subset=["lake_id_new"]).copy()
        ranked_gdf["lake_id_new"] = ranked_gdf["lake_id_new"].astype(int)
        ranked_gdf = ranked_gdf.sort_values(["lake_id_new", "old_lake_id"]).reset_index(drop=True)
        ranked_shp = MAP_MAIN_DIR / f"lakes_with_wse_curve_maxrange_ranked{suffix}_{method_name}.shp"
        ranked_csv = MAP_MAIN_DIR / f"lakes_with_wse_curve_maxrange_ranked{suffix}_{method_name}.csv"
        ranked_gdf.to_file(ranked_shp, encoding="utf-8")
        pd.DataFrame(ranked_gdf.drop(columns="geometry")).to_csv(ranked_csv, index=False, encoding="utf-8-sig")
        print(f"[06_3] wrote ranked lake max-range shp: {ranked_shp}", flush=True)
        print(f"[06_3] wrote ranked lake max-range csv: {ranked_csv}", flush=True)
        return ranked_gdf

    return out_gdf


def make_title_text(lake_id, branch_index, display_lake_id=None):
    title_lake_id = int(display_lake_id) if pd.notna(display_lake_id) else int(lake_id)
    title_text = f"Lake {title_lake_id}"
    if bool(PLOT_TITLE_SHOW_BRANCH):
        title_text = f"{title_text} | Branch {int(branch_index)}"
    return title_text


def get_initial_candidate_groups(base_module, sub_df, branch_index, params):
    work = sub_df.copy()
    if "branch_index" in work.columns:
        work = work.loc[pd.to_numeric(work["branch_index"], errors="coerce") == int(branch_index)].copy()
    work = work.sort_values("date_dt").copy()
    if work.empty:
        return [], [], []

    records = base_module.build_sub_records(work, branch_index, params)

    high_uncertainty_indices = set()
    for idx, rec in enumerate(records):
        wse_std = pd.to_numeric(rec.get("wse_std"), errors="coerce")
        if pd.notna(wse_std) and float(wse_std) > 1.0:
            high_uncertainty_indices.add(idx)
    if high_uncertainty_indices:
        base_module.set_noise(records, high_uncertainty_indices)
        for idx in high_uncertainty_indices:
            records[idx]["noise_reason"] = "uncertainty_gt_1m"

    base_module.apply_basic_red_rules(records, params)

    for rec in records:
        if bool(rec.get("from_bridge_rule", False)):
            rec["point_color"] = base_module.BACKGROUND_COLOR
            rec["from_bridge_rule"] = False

    segments = base_module.rebuild_segments(records, params, force_red=False)
    red_groups = base_module.build_groups(records, segments)
    for group in red_groups:
        day_len = len({pd.Timestamp(records[idx]["date"]).normalize() for idx in group})
        if day_len >= int(INITIAL_RED_GROUP_MIN_DAYS):
            continue
        for idx in group:
            if bool(records[idx].get("is_noise", False)):
                continue
            records[idx]["point_color"] = base_module.BACKGROUND_COLOR

    segments = base_module.rebuild_segments(records, params, force_red=False)
    groups = base_module.build_groups(records, segments)
    return records, segments, groups


def expand_red_groups_from_longest_once(base_module, records, params):
    threshold = float(params.red_point_max_jump_m)
    changed = False

    def group_day_len(group):
        return len({pd.Timestamp(records[idx]["date"]).normalize() for idx in group})

    def expand_one_side(anchor_idx, direction):
        nonlocal changed
        side_changed = False
        while True:
            valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
            if anchor_idx not in valid_indices:
                break
            pos = valid_indices.index(anchor_idx)
            next_pos = pos + direction
            if next_pos < 0 or next_pos >= len(valid_indices):
                break

            match_idx = None
            p = next_pos
            anchor_wse = float(records[anchor_idx]["wse"])
            while 0 <= p < len(valid_indices):
                cidx = valid_indices[p]
                if str(records[cidx].get("point_color", "")) != base_module.BACKGROUND_COLOR:
                    break
                cur_wse = float(records[cidx]["wse"])
                if abs(cur_wse - anchor_wse) < threshold:
                    match_idx = cidx
                    break
                p += direction

            if match_idx is None:
                break

            records[match_idx]["point_color"] = base_module.RED_COLOR
            anchor_idx = match_idx
            changed = True
            side_changed = True

        return side_changed

    segments = base_module.rebuild_segments(records, params, force_red=False)
    groups = base_module.build_groups(records, segments)
    if not groups:
        return False

    ordered_groups = sorted(groups, key=lambda g: (-group_day_len(g), min(g) if g else 10**9))
    for group in ordered_groups:
        if not group:
            continue
        expand_one_side(min(group), -1)
        expand_one_side(max(group), +1)

    return changed


def noise_remaining_gray_between_red_segments_once(base_module, records, params):
    threshold = float(params.red_point_max_jump_m)
    changed = False
    valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
    if not valid_indices:
        return False

    for pos, idx in enumerate(valid_indices):
        if str(records[idx].get("point_color", "")) != base_module.BACKGROUND_COLOR:
            continue

        left_red_idx = None
        p = pos - 1
        while p >= 0:
            cand = valid_indices[p]
            if str(records[cand].get("point_color", "")) == base_module.RED_COLOR:
                left_red_idx = cand
                break
            p -= 1

        right_red_idx = None
        p = pos + 1
        while p < len(valid_indices):
            cand = valid_indices[p]
            if str(records[cand].get("point_color", "")) == base_module.RED_COLOR:
                right_red_idx = cand
                break
            p += 1

        if left_red_idx is None or right_red_idx is None:
            continue

        cur = float(records[idx]["wse"])
        left = float(records[left_red_idx]["wse"])
        right = float(records[right_red_idx]["wse"])
        too_high_both = (cur > left + threshold) and (cur > right + threshold)
        too_low_both = (cur < left - threshold) and (cur < right - threshold)
        if too_high_both or too_low_both:
            changed = base_module.set_noise(records, {idx}) or changed
            records[idx]["noise_reason"] = "gray_between_red_segments_out_of_both_sides"
        else:
            records[idx]["point_color"] = base_module.RED_COLOR
            changed = True

    return changed


def noise_short_edge_gray_chain_if_above_nearest_red_once(base_module, records, max_chain_len=2):
    valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
    if not valid_indices:
        return False

    red_indices = [i for i in valid_indices if str(records[i].get("point_color", "")) == base_module.RED_COLOR]
    if not red_indices:
        return False

    first_red_idx = min(red_indices)
    last_red_idx = max(red_indices)
    first_red_wse = float(records[first_red_idx]["wse"])
    last_red_wse = float(records[last_red_idx]["wse"])
    global_red_min_wse = min(float(records[i]["wse"]) for i in red_indices)

    left_gray = [i for i in valid_indices if i < first_red_idx and str(records[i].get("point_color", "")) == base_module.BACKGROUND_COLOR]
    right_gray = [i for i in valid_indices if i > last_red_idx and str(records[i].get("point_color", "")) == base_module.BACKGROUND_COLOR]

    noise_indices = set()
    noise_reason_by_idx = {}

    if 1 <= len(left_gray) <= int(max_chain_len):
        for i in left_gray:
            if float(records[i]["wse"]) > first_red_wse:
                noise_indices.add(i)
                noise_reason_by_idx[i] = "edge_short_gray_above_nearest_red"

    if 1 <= len(right_gray) <= int(max_chain_len):
        for i in right_gray:
            if float(records[i]["wse"]) > last_red_wse:
                noise_indices.add(i)
                noise_reason_by_idx[i] = "edge_short_gray_above_nearest_red"

    for i in right_gray:
        if i in noise_indices:
            continue
        if float(records[i]["wse"]) > last_red_wse:
            noise_indices.add(i)
            noise_reason_by_idx[i] = "right_gray_above_nearest_red"

    for i in left_gray:
        if i in noise_indices:
            continue
        if abs(float(records[i]["wse"]) - first_red_wse) > 3.0:
            noise_indices.add(i)
            noise_reason_by_idx[i] = "left_gray_far_from_nearest_red_gt_3m"

    for i in right_gray:
        if i in noise_indices:
            continue
        if float(records[i]["wse"]) < (global_red_min_wse - 3.0):
            noise_indices.add(i)
            noise_reason_by_idx[i] = "right_gray_below_global_red_min_gt_3m"

    if not noise_indices:
        return False

    changed = base_module.set_noise(records, noise_indices)
    if changed:
        for idx in noise_indices:
            records[idx]["noise_reason"] = noise_reason_by_idx.get(idx, "edge_short_gray_rule")
    return changed


def noise_local_extrema_once(base_module, records, threshold_m=1.0):
    valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
    if len(valid_indices) < 3:
        return False

    threshold = float(threshold_m)
    global_max_wse = max(float(records[i]["wse"]) for i in valid_indices)
    global_max_indices = [i for i in valid_indices if abs(float(records[i]["wse"]) - global_max_wse) <= 1e-9]

    # Global-maximum protection is conditional:
    # keep it only when both immediate neighbors are also high-rank (top 4, inclusive).
    rank_lookup = {}
    sorted_pairs = sorted(((i, float(records[i]["wse"])) for i in valid_indices), key=lambda x: (-x[1], x[0]))
    for rank_pos, (idx, _) in enumerate(sorted_pairs, start=1):
        rank_lookup[idx] = rank_pos

    protected_global_max_indices = set()
    for idx in global_max_indices:
        pos = valid_indices.index(idx)
        if pos <= 0 or pos >= len(valid_indices) - 1:
            continue
        left_idx = valid_indices[pos - 1]
        right_idx = valid_indices[pos + 1]
        left_rank = rank_lookup.get(left_idx, 9999)
        right_rank = rank_lookup.get(right_idx, 9999)
        if left_rank <= 4 and right_rank <= 4:
            protected_global_max_indices.add(idx)

    noise_indices = set()
    reason_map = {}
    for pos in range(1, len(valid_indices) - 1):
        idx = valid_indices[pos]
        left_idx = valid_indices[pos - 1]
        right_idx = valid_indices[pos + 1]

        cur = float(records[idx]["wse"])
        left = float(records[left_idx]["wse"])
        right = float(records[right_idx]["wse"])

        is_local_peak = (cur >= left + threshold) and (cur >= right + threshold)
        is_local_valley = (cur <= left - threshold) and (cur <= right - threshold)

        if is_local_peak:
            # Keep the global maximum point(s), do not denoise them as local peaks.
            if idx in protected_global_max_indices:
                continue
            noise_indices.add(idx)
            if idx in global_max_indices:
                reason_map[idx] = "global_max_not_supported_by_neighbors"
            else:
                reason_map[idx] = "local_max_extreme_gt_1m"
            continue

        if is_local_valley:
            noise_indices.add(idx)
            reason_map[idx] = "local_min_extreme_gt_1m"

    if not noise_indices:
        return False

    changed = base_module.set_noise(records, noise_indices)
    if changed:
        for idx in noise_indices:
            records[idx]["noise_reason"] = reason_map.get(idx, "local_extrema_gt_1m")
    return changed


def _version_matches(rule_version, pixc_version):
    rv = str(rule_version).strip().upper()
    pv = str(pixc_version).strip().upper()
    return rv in {"", "*", "ALL"} or rv == pv


def _resolve_rule_day_set(rule):
    days = rule.get("days")
    if days:
        return {pd.Timestamp(d).normalize() for d in days}
    start_day = rule.get("start_day")
    end_day = rule.get("end_day", start_day)
    if not start_day:
        return set()
    sday = pd.Timestamp(start_day).normalize()
    eday = pd.Timestamp(end_day).normalize() if end_day else sday
    if eday < sday:
        sday, eday = eday, sday
    return {pd.Timestamp(d).normalize() for d in pd.date_range(sday, eday, freq="D")}


def apply_manual_noise_rule(base_module, records, day_set, reason="manual"):
    if not day_set:
        return False
    noise_indices = set()
    for idx, rec in enumerate(records):
        day = pd.Timestamp(rec["date"]).normalize()
        if day in day_set:
            noise_indices.add(idx)
    if not noise_indices:
        return False
    changed = base_module.set_noise(records, noise_indices)
    if changed:
        for idx in noise_indices:
            records[idx]["noise_reason"] = reason
    return changed


def apply_manual_not_noise_rule(base_module, records, day_set):
    if not day_set:
        return False
    changed = False
    for rec in records:
        day = pd.Timestamp(rec["date"]).normalize()
        if day not in day_set:
            continue
        if bool(rec.get("is_noise", False)):
            rec["is_noise"] = False
            changed = True
        rec["noise_reason"] = ""
        if bool(rec.get("force_noise_x", False)):
            rec["force_noise_x"] = False
            changed = True
        if rec.get("point_color") in (None, "", "None"):
            rec["point_color"] = base_module.BACKGROUND_COLOR
            changed = True
    return changed


def apply_manual_local_max_noise_rule(base_module, records, day, reason="manual_local_max_noise"):
    target_day = pd.Timestamp(day).normalize()
    valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
    if len(valid_indices) < 3:
        return False

    # Sort by date to evaluate local maxima against adjacent non-noise points.
    valid_indices = sorted(valid_indices, key=lambda i: pd.Timestamp(records[i]["date"]))
    noise_indices = set()
    eps = 1e-9
    for pos, i in enumerate(valid_indices):
        if pos == 0 or pos == len(valid_indices) - 1:
            continue
        rec = records[i]
        rec_day = pd.Timestamp(rec["date"]).normalize()
        if rec_day != target_day:
            continue
        w_curr = float(rec["wse"])
        w_prev = float(records[valid_indices[pos - 1]]["wse"])
        w_next = float(records[valid_indices[pos + 1]]["wse"])
        if (w_curr >= w_prev - eps and w_curr >= w_next - eps) and (w_curr > w_prev + eps or w_curr > w_next + eps):
            noise_indices.add(i)

    if not noise_indices:
        return False

    changed = base_module.set_noise(records, noise_indices)
    if changed:
        for idx in noise_indices:
            records[idx]["noise_reason"] = reason
    return changed


def apply_manual_exact_point_noise_rule(base_module, records, day, target_wse, reason="manual_exact_point_noise"):
    target_day = pd.Timestamp(day).normalize()
    target_wse = float(target_wse)
    matched = []
    for idx, rec in enumerate(records):
        rec_day = pd.Timestamp(rec["date"]).normalize()
        if rec_day != target_day:
            continue
        matched.append((abs(float(rec["wse"]) - target_wse), idx))

    if not matched:
        return False

    matched.sort(key=lambda x: (x[0], x[1]))
    noise_idx = matched[0][1]
    changed = base_module.set_noise(records, {noise_idx})
    if changed:
        records[noise_idx]["noise_reason"] = reason
    return changed


def apply_manual_rules(base_module, records, lake_id, branch_index, pixc_version):
    changed = False
    lake_id_int = int(lake_id)
    branch_idx_int = int(branch_index)
    version_text = str(pixc_version).strip().upper()
    if version_text not in MANUAL_RULES_TARGET_VERSIONS:
        return False

    for block in MANUAL_RULE_BLOCKS:
        if int(block.get("lake_id", -1)) != lake_id_int:
            continue
        if not _version_matches(block.get("version", "*"), version_text):
            continue
        block_branch_idx = block.get("branch_index", None)
        if block_branch_idx is not None and int(block_branch_idx) != branch_idx_int:
            continue

        for item in block.get("force_noise", []) or []:
            changed = apply_manual_noise_rule(
                base_module=base_module,
                records=records,
                day_set=_resolve_rule_day_set(item),
                reason=str(item.get("reason", "manual_force_noise")),
            ) or changed

        for item in block.get("force_not_noise", []) or []:
            changed = apply_manual_not_noise_rule(
                base_module=base_module,
                records=records,
                day_set=_resolve_rule_day_set(item),
            ) or changed

        for item in block.get("force_local_max_noise", []) or []:
            day = item.get("day")
            if not day:
                continue
            changed = apply_manual_local_max_noise_rule(
                base_module=base_module,
                records=records,
                day=day,
                reason=str(item.get("reason", "manual_local_max_noise")),
            ) or changed

        for item in block.get("force_exact_point_noise", []) or []:
            day = item.get("day")
            target_wse = item.get("wse")
            if not day or target_wse is None:
                continue
            changed = apply_manual_exact_point_noise_rule(
                base_module=base_module,
                records=records,
                day=day,
                target_wse=target_wse,
                reason=str(item.get("reason", "manual_exact_point_noise")),
            ) or changed

    return changed


def noise_pre_june13_if_higher_than_first_post_june13_once(base_module, records):
    valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
    if not valid_indices:
        return False

    dated_indices = []
    for idx in valid_indices:
        dt = pd.to_datetime(records[idx].get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        dated_indices.append((dt.normalize(), idx))
    if not dated_indices:
        return False

    dated_indices.sort(key=lambda item: item[0])
    ref_idx = next((idx for dt, idx in dated_indices if dt > PRE_JUNE13_NOISE_CUTOFF), None)
    if ref_idx is None:
        return False

    ref_wse = pd.to_numeric(records[ref_idx].get("wse"), errors="coerce")
    if pd.isna(ref_wse):
        return False

    noise_indices = set()
    for dt, idx in dated_indices:
        if dt >= PRE_JUNE13_NOISE_CUTOFF:
            continue
        wse_val = pd.to_numeric(records[idx].get("wse"), errors="coerce")
        if pd.notna(wse_val) and float(wse_val) > float(ref_wse):
            noise_indices.add(idx)

    if not noise_indices:
        return False

    changed = base_module.set_noise(records, noise_indices)
    if changed:
        for idx in noise_indices:
            records[idx]["noise_reason"] = "pre_0613_gt_first_post_0613"
    return changed


def force_noise_before_20240605_once(base_module, records):
    noise_indices = set()
    for idx, rec in enumerate(records):
        if bool(rec.get("is_noise", False)):
            continue
        dt = pd.to_datetime(rec.get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        if dt.normalize() < FORCE_NOISE_BEFORE_0605_CUTOFF:
            noise_indices.add(idx)
    if not noise_indices:
        return False
    changed = base_module.set_noise(records, noise_indices)
    if changed:
        for idx in noise_indices:
            records[idx]["noise_reason"] = "force_noise_pre_20240605"
    return changed


def get_stage_records(base_module, initial_records, params):
    records = deepcopy(initial_records)
    while True:
        changed = expand_red_groups_from_longest_once(base_module, records, params)
        if not changed:
            break
    while True:
        changed = noise_remaining_gray_between_red_segments_once(base_module, records, params)
        if not changed:
            break
    while True:
        changed = noise_short_edge_gray_chain_if_above_nearest_red_once(
            base_module, records, max_chain_len=EDGE_GRAY_SHORT_CHAIN_MAX_LEN
        )
        if not changed:
            break

    segments = base_module.rebuild_segments(records, params, force_red=False)
    groups = base_module.build_groups(records, segments)
    return records, segments, groups


def count_red_points(records, base_module):
    return sum(
        1
        for rec in records
        if str(rec.get("point_color", "")) == base_module.RED_COLOR and not bool(rec.get("is_noise", False))
    )


def render_branch_plot(base_module, lake_id, branch_index, records, segments, out_path, show_noise, fixed_xlim=None, display_lake_id=None):
    fig, ax = plt.subplots(figsize=(PLOT_FIG_WIDTH, PLOT_FIG_HEIGHT), dpi=PLOT_DPI)
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)
    add_same_day_plot_offsets(records)

    valid_indices = [i for i, rec in enumerate(records) if not bool(rec.get("is_noise", False))]
    for i0, i1 in zip(valid_indices, valid_indices[1:]):
        rec0 = records[i0]
        rec1 = records[i1]
        if show_noise:
            c0 = str(rec0.get("point_color", ""))
            c1 = str(rec1.get("point_color", ""))
            seg_color = base_module.RED_COLOR if (c0 == base_module.RED_COLOR and c1 == base_module.RED_COLOR) else base_module.BACKGROUND_COLOR
        else:
            seg_color = base_module.RED_COLOR
        ax.plot(
            [rec0["x_plot_dt"], rec1["x_plot_dt"]],
            [float(rec0["wse"]), float(rec1["wse"])],
            linestyle=PLOT_LINESTYLE,
            linewidth=PLOT_LINEWIDTH,
            color=seg_color,
            alpha=PLOT_LINE_ALPHA,
            zorder=2,
        )

    red_records = [rec for rec in records if str(rec.get("point_color", "")) == base_module.RED_COLOR and not bool(rec.get("is_noise", False))]
    gray_records = [rec for rec in records if str(rec.get("point_color", "")) == base_module.BACKGROUND_COLOR and not bool(rec.get("is_noise", False))]
    noise_records = [rec for rec in records if bool(rec.get("is_noise", False))]

    no_noise_red_records = red_records if show_noise else red_records + gray_records

    if no_noise_red_records:
        ax.scatter(
            [rec["x_plot_dt"] for rec in no_noise_red_records],
            [float(rec["wse"]) for rec in no_noise_red_records],
            s=PLOT_RED_POINT_SIZE,
            color=base_module.RED_COLOR,
            edgecolors=base_module.RED_COLOR,
            zorder=3,
            label="Red point",
        )
    if show_noise and noise_records:
        ax.scatter(
            [rec["x_plot_dt"] for rec in noise_records],
            [float(rec["wse"]) for rec in noise_records],
            marker="x",
            s=PLOT_NOISE_POINT_SIZE,
            linewidths=PLOT_NOISE_LINEWIDTH,
            color=base_module.NOISE_COLOR,
            zorder=3.2,
            label="Noise point",
        )
    if show_noise and gray_records:
        ax.scatter(
            [rec["x_plot_dt"] for rec in gray_records],
            [float(rec["wse"]) for rec in gray_records],
            s=PLOT_GRAY_POINT_SIZE,
            color=base_module.BACKGROUND_COLOR,
            edgecolors=base_module.BACKGROUND_COLOR,
            zorder=2.8,
            label="Gray point",
        )

    for rec in no_noise_red_records:
        wse_std = pd.to_numeric(rec.get("wse_std"), errors="coerce")
        if pd.notna(wse_std) and float(wse_std) > 0:
            ax.errorbar(
                [rec["x_plot_dt"]],
                [float(rec["wse"])],
                yerr=[[float(wse_std)], [float(wse_std)]],
                fmt="none",
                ecolor=base_module.RED_COLOR,
                elinewidth=PLOT_ERR_LINEWIDTH,
                capsize=PLOT_ERR_CAPSIZE,
                alpha=PLOT_ERR_ALPHA,
                zorder=2.6,
            )

    title_text = make_title_text(lake_id, branch_index, display_lake_id=display_lake_id)
    # Keep label inside frame at top-left.
    ax.text(
        PLOT_TITLE_X,
        PLOT_TITLE_Y,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    if PLOT_SHOW_Y_LABEL:
        ax.set_ylabel(PLOT_YLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="-", linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, color=PLOT_GRID_COLOR)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_SPINE_LINEWIDTH)
        spine.set_color(PLOT_SPINE_COLOR)

    visible_records = [rec for rec in records if show_noise or not bool(rec.get("is_noise", False))]
    all_dates = [rec["x_plot_dt"] for rec in visible_records]
    all_wse = [float(rec["wse"]) for rec in visible_records]
    if fixed_xlim is not None:
        x_start, x_end = fixed_xlim
        ax.set_xlim(x_start, x_end)
    elif all_dates:
        ax.set_xlim(min(all_dates) - pd.Timedelta(days=2), max(all_dates) + pd.Timedelta(days=1))
    # Use a global tick anchor so all lakes share the same x-tick start (2024-06-22).
    x_start_now, x_end_now = ax.get_xlim()
    x_end_ts = mdates.num2date(x_end_now).replace(tzinfo=None)
    tick_dates = pd.date_range(PLOT_X_TICK_START, pd.Timestamp(x_end_ts).normalize(), freq=f"{int(PLOT_XTICK_INTERVAL_DAYS)}D")
    if len(tick_dates) > 0:
        ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if PLOT_SHOW_X_LABEL:
        ax.set_xlabel(PLOT_XLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE)
    else:
        ax.set_xlabel("")
    plt.setp(
        ax.get_xticklabels(),
        rotation=PLOT_XTICK_ROTATION,
        ha="center",
        rotation_mode="anchor",
        fontsize=PLOT_XTICK_FONTSIZE,
        va="top",
    )
    plt.setp(ax.get_yticklabels(), fontsize=PLOT_YTICK_FONTSIZE)
    ax.tick_params(axis="x", pad=PLOT_XTICK_PAD, width=PLOT_TICK_LINEWIDTH, length=PLOT_TICK_LENGTH)
    ax.tick_params(axis="y", pad=PLOT_YTICK_PAD, width=PLOT_TICK_LINEWIDTH, length=PLOT_TICK_LENGTH)
    if all_wse:
        ymin = min(all_wse)
        ymax = max(all_wse)
        pad = max(PLOT_Y_PAD_MIN, (ymax - ymin) * PLOT_Y_PAD_RATIO if ymax > ymin else PLOT_Y_PAD_FLAT)
        # Add extra headroom so curve doesn't overlap with in-panel Lake label.
        ax.set_ylim(ymin - pad, ymax + pad + max(pad * PLOT_Y_TOP_EXTRA, 0.15))

    if show_noise:
        ax.legend(loc="upper right", fontsize=PLOT_LEGEND_FONTSIZE, frameon=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Use fixed subplot margins so every lake figure has the same frame position/size.
    fig.subplots_adjust(
        left=PLOT_MARGIN_LEFT,
        right=PLOT_MARGIN_RIGHT,
        bottom=PLOT_MARGIN_BOTTOM,
        top=PLOT_MARGIN_TOP,
    )
    fig.savefig(out_path)
    plt.close(fig)


def render_branch_dual_axis_plot(base_module, lake_id, branch_index, records, out_path, area_df, fixed_xlim=None, display_lake_id=None):
    fig, ax = plt.subplots(figsize=(PLOT_FIG_WIDTH, PLOT_FIG_HEIGHT), dpi=PLOT_DPI)
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)
    add_same_day_plot_offsets(records)

    visible_records = [rec for rec in records if not bool(rec.get("is_noise", False))]
    for rec0, rec1 in zip(visible_records, visible_records[1:]):
        ax.plot(
            [rec0["x_plot_dt"], rec1["x_plot_dt"]],
            [float(rec0["wse"]), float(rec1["wse"])],
            linestyle=PLOT_LINESTYLE,
            linewidth=PLOT_LINEWIDTH,
            color=base_module.RED_COLOR,
            alpha=PLOT_LINE_ALPHA,
            zorder=2.2,
        )

    if visible_records:
        ax.scatter(
            [rec["x_plot_dt"] for rec in visible_records],
            [float(rec["wse"]) for rec in visible_records],
            s=PLOT_RED_POINT_SIZE,
            color=base_module.RED_COLOR,
            edgecolors=base_module.RED_COLOR,
            zorder=3.2,
        )
        for rec in visible_records:
            wse_std = pd.to_numeric(rec.get("wse_std"), errors="coerce")
            if pd.notna(wse_std) and float(wse_std) > 0:
                ax.errorbar(
                    [rec["x_plot_dt"]],
                    [float(rec["wse"])],
                    yerr=[[float(wse_std)], [float(wse_std)]],
                    fmt="none",
                    ecolor=base_module.RED_COLOR,
                    elinewidth=PLOT_ERR_LINEWIDTH,
                    capsize=PLOT_ERR_CAPSIZE,
                    alpha=PLOT_ERR_ALPHA,
                    zorder=2.6,
                )

    ax_area = ax.twinx()
    if area_df is not None and not area_df.empty:
        ax_area.plot(
            area_df["x_plot_dt"],
            area_df["area_km2"],
            linestyle=AREA_LINESTYLE,
            linewidth=AREA_LINEWIDTH,
            color=AREA_COLOR,
            alpha=AREA_LINE_ALPHA,
            zorder=1.4,
        )
        ax_area.scatter(
            area_df["x_plot_dt"],
            area_df["area_km2"],
            s=AREA_POINT_SIZE,
            color=AREA_COLOR,
            edgecolors="none",
            alpha=0.95,
            zorder=1.6,
        )
        area_min = float(pd.to_numeric(area_df["area_km2"], errors="coerce").min())
        area_max = float(pd.to_numeric(area_df["area_km2"], errors="coerce").max())
        area_pad = max(0.02, (area_max - area_min) * 0.10 if area_max > area_min else 0.05)
        ax_area.set_ylim(area_min - area_pad, area_max + area_pad)

    title_text = make_title_text(lake_id, branch_index, display_lake_id=display_lake_id)
    ax.text(
        PLOT_TITLE_X,
        PLOT_TITLE_Y,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    if PLOT_SHOW_Y_LABEL:
        ax.set_ylabel(PLOT_YLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
    if PLOT_SHOW_AREA_Y_LABEL:
        ax_area.set_ylabel(PLOT_AREA_YLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE, color=AREA_COLOR)
    else:
        ax_area.set_ylabel("")

    ax.grid(True, linestyle="-", linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, color=PLOT_GRID_COLOR)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_area.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_SPINE_LINEWIDTH)
        spine.set_color(PLOT_SPINE_COLOR)
    ax_area.spines["right"].set_linewidth(PLOT_SPINE_LINEWIDTH)
    ax_area.spines["right"].set_color(AREA_COLOR)

    all_dates = [rec["x_plot_dt"] for rec in visible_records]
    all_wse = [float(rec["wse"]) for rec in visible_records]
    if fixed_xlim is not None:
        ax.set_xlim(*fixed_xlim)
    elif all_dates:
        ax.set_xlim(min(all_dates) - pd.Timedelta(days=2), max(all_dates) + pd.Timedelta(days=1))
    _, x_end_now = ax.get_xlim()
    x_end_ts = mdates.num2date(x_end_now).replace(tzinfo=None)
    tick_dates = pd.date_range(PLOT_X_TICK_START, pd.Timestamp(x_end_ts).normalize(), freq=f"{int(PLOT_XTICK_INTERVAL_DAYS)}D")
    if len(tick_dates) > 0:
        ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if PLOT_SHOW_X_LABEL:
        ax.set_xlabel(PLOT_XLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE)
    else:
        ax.set_xlabel("")
    plt.setp(
        ax.get_xticklabels(),
        rotation=PLOT_XTICK_ROTATION,
        ha="center",
        rotation_mode="anchor",
        fontsize=PLOT_XTICK_FONTSIZE,
        va="top",
    )
    plt.setp(ax.get_yticklabels(), fontsize=PLOT_YTICK_FONTSIZE)
    ax.tick_params(axis="x", pad=PLOT_XTICK_PAD, width=PLOT_TICK_LINEWIDTH, length=PLOT_TICK_LENGTH)
    ax.tick_params(axis="y", pad=PLOT_YTICK_PAD, width=PLOT_TICK_LINEWIDTH, length=PLOT_TICK_LENGTH)
    ax_area.tick_params(
        axis="y",
        pad=PLOT_YTICK_PAD,
        labelsize=AREA_TICK_FONTSIZE,
        colors=AREA_COLOR,
        width=PLOT_TICK_LINEWIDTH,
        length=PLOT_TICK_LENGTH,
    )
    ax_area.tick_params(axis="y", which="both", right=True, labelright=True)
    if all_wse:
        ymin = min(all_wse)
        ymax = max(all_wse)
        pad = max(PLOT_Y_PAD_MIN, (ymax - ymin) * PLOT_Y_PAD_RATIO if ymax > ymin else PLOT_Y_PAD_FLAT)
        ax.set_ylim(ymin - pad, ymax + pad + max(pad * PLOT_Y_TOP_EXTRA, 0.15))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(
        left=PLOT_MARGIN_LEFT,
        right=PLOT_MARGIN_RIGHT_WITH_RIGHT_AXIS,
        bottom=PLOT_MARGIN_BOTTOM,
        top=PLOT_MARGIN_TOP,
    )
    fig.savefig(out_path)
    plt.close(fig)


def render_branch_vc_d_dual_axis_plot(base_module, lake_id, branch_index, records_c, records_d, out_path, area_df, fixed_xlim=None, display_lake_id=None):
    fig, ax = plt.subplots(figsize=(PLOT_FIG_WIDTH, PLOT_FIG_HEIGHT), dpi=PLOT_DPI)
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)

    c_records = [deepcopy(rec) for rec in (records_c or []) if not bool(rec.get("is_noise", False))]
    d_records = [deepcopy(rec) for rec in (records_d or []) if not bool(rec.get("is_noise", False))]
    add_same_day_plot_offsets(c_records)
    add_same_day_plot_offsets(d_records)

    for series_records, color, label in [
        (c_records, VC_COLOR, "C WSE"),
        (d_records, VD_COLOR, "D WSE"),
    ]:
        for rec0, rec1 in zip(series_records, series_records[1:]):
            ax.plot(
                [rec0["x_plot_dt"], rec1["x_plot_dt"]],
                [float(rec0["wse"]), float(rec1["wse"])],
                linestyle=PLOT_LINESTYLE,
                linewidth=PLOT_LINEWIDTH,
                color=color,
                alpha=PLOT_LINE_ALPHA,
                zorder=2.2,
            )
        if series_records:
            ax.scatter(
                [rec["x_plot_dt"] for rec in series_records],
                [float(rec["wse"]) for rec in series_records],
                s=PLOT_RED_POINT_SIZE,
                color=color,
                edgecolors=color,
                zorder=3.2,
                label=label,
            )

    title_text = make_title_text(lake_id, branch_index, display_lake_id=display_lake_id)
    ax.text(
        PLOT_TITLE_X,
        PLOT_TITLE_Y,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
        color="black",
    )
    if PLOT_SHOW_Y_LABEL:
        ax.set_ylabel(PLOT_YLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="-", linewidth=PLOT_GRID_LINEWIDTH, alpha=PLOT_GRID_ALPHA, color=PLOT_GRID_COLOR)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_SPINE_LINEWIDTH)
        spine.set_color(PLOT_SPINE_COLOR)

    all_records = c_records + d_records
    all_dates = [rec["x_plot_dt"] for rec in all_records]
    all_wse = [float(rec["wse"]) for rec in all_records]
    if fixed_xlim is not None:
        ax.set_xlim(*fixed_xlim)
    elif all_dates:
        ax.set_xlim(min(all_dates) - pd.Timedelta(days=2), max(all_dates) + pd.Timedelta(days=1))
    _, x_end_now = ax.get_xlim()
    x_end_ts = mdates.num2date(x_end_now).replace(tzinfo=None)
    tick_dates = pd.date_range(PLOT_X_TICK_START, pd.Timestamp(x_end_ts).normalize(), freq=f"{int(PLOT_XTICK_INTERVAL_DAYS)}D")
    if len(tick_dates) > 0:
        ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if PLOT_SHOW_X_LABEL:
        ax.set_xlabel(PLOT_XLABEL_TEXT, fontsize=PLOT_YLABEL_FONTSIZE)
    else:
        ax.set_xlabel("")
    plt.setp(
        ax.get_xticklabels(),
        rotation=PLOT_XTICK_ROTATION,
        ha="center",
        rotation_mode="anchor",
        fontsize=PLOT_XTICK_FONTSIZE,
        va="top",
    )
    plt.setp(ax.get_yticklabels(), fontsize=PLOT_YTICK_FONTSIZE)
    ax.tick_params(axis="x", pad=PLOT_XTICK_PAD, width=PLOT_TICK_LINEWIDTH, length=PLOT_TICK_LENGTH)
    ax.tick_params(axis="y", pad=PLOT_YTICK_PAD, width=PLOT_TICK_LINEWIDTH, length=PLOT_TICK_LENGTH)
    if all_wse:
        ymin = min(all_wse)
        ymax = max(all_wse)
        pad = max(PLOT_Y_PAD_MIN, (ymax - ymin) * PLOT_Y_PAD_RATIO if ymax > ymin else PLOT_Y_PAD_FLAT)
        ax.set_ylim(ymin - pad, ymax + pad + max(pad * PLOT_Y_TOP_EXTRA, 0.15))

    h1, l1 = ax.get_legend_handles_labels()
    if h1:
        ax.legend(h1, l1, loc="upper right", fontsize=PLOT_LEGEND_FONTSIZE, frameon=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(
        left=PLOT_MARGIN_LEFT,
        right=PLOT_MARGIN_RIGHT,
        bottom=PLOT_MARGIN_BOTTOM,
        top=PLOT_MARGIN_TOP,
    )
    fig.savefig(out_path)
    plt.close(fig)


def prepare_branch_output_df(records, branch_index):
    rows = []
    for rec in records:
        rows.append(
            {
                "branch_index": int(branch_index),
                "date": pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"),
                "wse_m": float(rec["wse"]),
                "wse_std": pd.NA if pd.isna(rec.get("wse_std")) else float(rec["wse_std"]),
                "point_color": str(rec.get("point_color", "")),
                "is_noise": bool(rec.get("is_noise", False)),
                "noise_reason": str(rec.get("noise_reason", "")),
            }
        )
    return pd.DataFrame(rows)


def prepare_red_only_output_df(records, branch_index, lake_id, pixc_version, base_module):
    rows = []
    for rec in records:
        if bool(rec.get("is_noise", False)):
            continue
        if str(rec.get("point_color", "")) != str(base_module.RED_COLOR):
            continue
        rows.append(
            {
                "lake_id": int(lake_id),
                "branch_index": int(branch_index),
                "pixc_version": str(pixc_version).upper(),
                "date": pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"),
                "wse_m": float(rec["wse"]),
                "wse_std": pd.NA if pd.isna(rec.get("wse_std")) else float(rec["wse_std"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    base = load_base_module()
    size_filters = normalize_size_filters(TARGET_SIZE_CATEGORIES)

    vc_d_both_name = "CD_Both"
    vc_d_c_only_name = "C_Only"
    vc_d_d_only_name = "D_Only"
    with_noise_name = "with_noise"
    no_noise_name = "no_noise"

    selected_versions = [str(v).upper() for v in PIXC_VERSIONS if str(v).upper() in {"C", "D"}]
    enable_vc_d_compare = ("C" in selected_versions) and ("D" in selected_versions)
    selected_lake_ids = {int(x) for x in PROCESS_ONLY_LAKE_IDS} if PROCESS_ONLY_LAKE_IDS else set()

    for method_name in base.METHOD_NAMES:
        method_paths = make_method_paths(base, method_name)
        os.makedirs(method_paths["plot_root_dir"], exist_ok=True)
        os.makedirs(method_paths["csv_dir"], exist_ok=True)

        if enable_vc_d_compare:
            vc_d_dir = os.path.join(method_paths["plot_root_dir"], "VC_D")
            vc_d_both_dir = os.path.join(vc_d_dir, vc_d_both_name)
            vc_d_c_only_dir = os.path.join(vc_d_dir, vc_d_c_only_name)
            vc_d_d_only_dir = os.path.join(vc_d_dir, vc_d_d_only_name)
            for d in [vc_d_dir, vc_d_both_dir, vc_d_c_only_dir, vc_d_d_only_dir]:
                os.makedirs(d, exist_ok=True)

        required = [method_paths["raw_wse_csv"], method_paths["lake_summary_csv"]]
        missing = [path0 for path0 in required if not os.path.exists(path0)]
        if missing:
            print(f"[06_3] skip {method_name}: missing inputs", flush=True)
            continue

        lake_summary_df = base.load_lake_summary_df(method_paths["lake_summary_csv"])
        raw_wse_df = base.load_raw_wse_df(method_paths["raw_wse_csv"])
        area_points_df = load_branch_area_points(method_paths["curve_point_csv"])
        raw_wse_df = drop_degraded_points_in_cross_track_band(raw_wse_df)
        raw_wse_df = keep_one_wse_per_scene(raw_wse_df)
        if raw_wse_df.empty or lake_summary_df.empty:
            print(f"[06_3] skip {method_name}: empty inputs", flush=True)
            continue

        if "pixc_version" not in raw_wse_df.columns:
            raw_wse_df["pixc_version"] = ""
        raw_wse_df["_pixc_version_norm"] = raw_wse_df["pixc_version"].map(normalize_pixc_version)

        if size_filters:
            lake_summary_df = lake_summary_df[
                lake_summary_df["size_category"].astype(str).str.strip().str.lower().isin(size_filters)
            ].copy()
        if EXCLUDED_LAKE_IDS:
            lake_summary_df = lake_summary_df[
                ~lake_summary_df["lake_id"].astype(int).isin(sorted(EXCLUDED_LAKE_IDS))
            ].copy()
        if selected_lake_ids:
            lake_summary_df = lake_summary_df[
                lake_summary_df["lake_id"].astype(int).isin(sorted(selected_lake_ids))
            ].copy()
        if lake_summary_df.empty:
            print(f"[06_3] skip {method_name}: no matched lakes", flush=True)
            continue
        if int(MAX_LAKES_TO_PROCESS) > 0:
            lake_summary_df = lake_summary_df.sort_values("lake_id").head(int(MAX_LAKES_TO_PROCESS)).copy()

        total_plots = 0
        vc_d_map = defaultdict(dict) if enable_vc_d_compare else None

        for pixc_version in selected_versions:
            version_df = raw_wse_df[raw_wse_df["_pixc_version_norm"] == str(pixc_version)].copy()
            if version_df.empty:
                continue

            version_root_dir = os.path.join(method_paths["plot_root_dir"], f"v{str(pixc_version).lower()}")
            with_noise_dir = os.path.join(version_root_dir, with_noise_name)
            no_noise_dir = os.path.join(version_root_dir, no_noise_name)
            no_noise_dual_axis_dir = os.path.join(version_root_dir, DUAL_AREA_FOLDER_NAME)
            red_records_dir = os.path.join(version_root_dir, RED_WSE_RECORDS_FOLDER_NAME)
            red_days_lt5_dir = os.path.join(version_root_dir, RED_DAYS_LT5_FOLDER_NAME)
            high_ratio_noise_dir = os.path.join(version_root_dir, HIGH_RATIO_NOISE_FOLDER_NAME)
            version_csv_dir = os.path.join(method_paths["csv_dir"], f"v{pixc_version}")
            high_ratio_rows = []
            for d in [
                with_noise_dir,
                no_noise_dir,
                no_noise_dual_axis_dir,
                red_records_dir,
                red_days_lt5_dir,
                high_ratio_noise_dir,
                version_csv_dir,
            ]:
                os.makedirs(d, exist_ok=True)

            total_lakes = len(lake_summary_df)
            for lake_pos, summary_row in enumerate(lake_summary_df.to_dict("records"), start=1):
                lake_id = int(summary_row["lake_id"])
                lake_df = version_df[version_df["lake_id"] == lake_id].copy()
                if lake_df.empty:
                    continue
                lake_date_vals = pd.to_datetime(lake_df.get("date_dt"), errors="coerce").dropna()
                lake_fixed_xlim = None
                if not lake_date_vals.empty:
                    lake_fixed_xlim = (PLOT_X_GLOBAL_START, PLOT_X_GLOBAL_END)

                unique_branches = sorted(
                    pd.to_numeric(lake_df["branch_index"], errors="coerce").dropna().astype(int).unique().tolist()
                )
                branch_tables = []
                for branch_index in unique_branches:
                    if (int(lake_id), int(branch_index)) in EXCLUDED_LAKE_BRANCHES:
                        print(
                            f"[06_3] skip excluded branch: lake={int(lake_id)} branch={int(branch_index)}",
                            flush=True,
                        )
                        continue
                    sub_df = lake_df[lake_df["branch_index"] == branch_index].copy().sort_values("date_dt")
                    initial_records, _, _ = get_initial_candidate_groups(base, sub_df, branch_index, base.PARAMS)
                    stage_records, stage_segments, _ = get_stage_records(base, initial_records, base.PARAMS)
                    local_extrema_changed = noise_local_extrema_once(
                        base_module=base,
                        records=stage_records,
                        threshold_m=LOCAL_EXTREMA_NOISE_THRESHOLD_M,
                    )
                    manual_changed = apply_manual_rules(base, stage_records, lake_id, branch_index, pixc_version)
                    pre_0605_changed = force_noise_before_20240605_once(
                        base_module=base,
                        records=stage_records,
                    )
                    pre_june13_changed = noise_pre_june13_if_higher_than_first_post_june13_once(
                        base_module=base,
                        records=stage_records,
                    )
                    if local_extrema_changed or manual_changed or pre_0605_changed or pre_june13_changed:
                        stage_segments = base.rebuild_segments(stage_records, base.PARAMS, force_red=False)

                    red_point_count = count_red_points(stage_records, base)
                    if enable_vc_d_compare:
                        vc_d_map[(int(lake_id), int(branch_index))][str(pixc_version).upper()] = {
                            "records": deepcopy(stage_records),
                            "red_point_count": int(red_point_count),
                            "can_output": bool(red_point_count >= RED_POINTS_MIN_TO_SAVE),
                        }
                    branch_df = prepare_branch_output_df(stage_records, branch_index)
                    out_csv = os.path.join(version_csv_dir, f"lake_{lake_id}_branch_{branch_index}_stage_v{pixc_version}.csv")
                    branch_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

                    branch_area_df = get_branch_area_series(area_points_df, lake_id, branch_index)
                    if red_point_count < RED_POINTS_MIN_TO_SAVE:
                        red_days_lt5_png = os.path.join(
                            red_days_lt5_dir,
                            f"Lake_{lake_id}_branch_{branch_index}_v{pixc_version}.png",
                        )
                        render_branch_plot(
                            base,
                            lake_id,
                            branch_index,
                            stage_records,
                            stage_segments,
                            red_days_lt5_png,
                            show_noise=True,
                            fixed_xlim=lake_fixed_xlim,
                        )
                        total_plots += 1
                        continue

                    branch_tables.append(branch_df)
                    red_only_df = prepare_red_only_output_df(
                        stage_records,
                        branch_index=branch_index,
                        lake_id=lake_id,
                        pixc_version=pixc_version,
                        base_module=base,
                    )
                    red_only_csv = os.path.join(
                        red_records_dir,
                        f"lake_{lake_id}_branch_{branch_index}_red_wse_v{pixc_version}.csv",
                    )
                    red_only_df.to_csv(red_only_csv, index=False, encoding="utf-8-sig")
                    ratio_info = compute_uncertainty_range_ratio(red_only_df)
                    is_high_ratio_noise = bool(
                        ratio_info is not None and float(ratio_info["ratio"]) > float(HIGH_RATIO_NOISE_THRESHOLD)
                    )
                    if is_high_ratio_noise:
                        high_ratio_csv = os.path.join(
                            high_ratio_noise_dir,
                            f"lake_{lake_id}_branch_{branch_index}_red_wse_v{pixc_version}.csv",
                        )
                        red_only_df.to_csv(high_ratio_csv, index=False, encoding="utf-8-sig")
                        high_ratio_rows.append(
                            {
                                "lake_id": int(lake_id),
                                "branch_index": int(branch_index),
                                "pixc_version": str(pixc_version),
                                "mean_std_m": float(ratio_info["mean_std_m"]),
                                "wse_range_m": float(ratio_info["wse_range_m"]),
                                "ratio": float(ratio_info["ratio"]),
                                "source_csv": red_only_csv,
                                "noise_csv": high_ratio_csv,
                            }
                        )

                    no_noise_png = os.path.join(no_noise_dir, f"Lake_{lake_id}_branch_{branch_index}_v{pixc_version}.png")
                    with_noise_png = os.path.join(with_noise_dir, f"Lake_{lake_id}_branch_{branch_index}_v{pixc_version}.png")
                    dual_axis_png = os.path.join(
                        no_noise_dual_axis_dir,
                        f"Lake_{lake_id}_branch_{branch_index}_v{pixc_version}.png",
                    )
                    if not is_high_ratio_noise:
                        render_branch_plot(
                            base,
                            lake_id,
                            branch_index,
                            stage_records,
                            stage_segments,
                            no_noise_png,
                            show_noise=False,
                            fixed_xlim=lake_fixed_xlim,
                        )
                        render_branch_dual_axis_plot(
                            base,
                            lake_id,
                            branch_index,
                            stage_records,
                            dual_axis_png,
                            branch_area_df,
                            fixed_xlim=lake_fixed_xlim,
                        )
                    render_branch_plot(
                        base,
                        lake_id,
                        branch_index,
                        stage_records,
                        stage_segments,
                        with_noise_png,
                        show_noise=True,
                        fixed_xlim=lake_fixed_xlim,
                    )
                    total_plots += 1 if is_high_ratio_noise else 3

                combined_df = pd.concat(branch_tables, ignore_index=True) if branch_tables else pd.DataFrame()
                combined_csv = os.path.join(version_csv_dir, f"lake_{lake_id}_all_branches_stage_v{pixc_version}.csv")
                combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
                print(
                    f"[06_3] {method_name} v{pixc_version} {lake_pos}/{total_lakes} done | lake={lake_id} | branches={len(unique_branches)}",
                    flush=True,
                )
            if high_ratio_rows:
                high_ratio_df = pd.DataFrame(high_ratio_rows)
                high_ratio_df.to_csv(
                    os.path.join(high_ratio_noise_dir, f"high_ratio_noise_summary_v{pixc_version}.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                print(
                    f"[06_3] v{pixc_version}: high-ratio noise(>{HIGH_RATIO_NOISE_THRESHOLD})={len(high_ratio_df)} -> {high_ratio_noise_dir}",
                    flush=True,
                )

        if enable_vc_d_compare:
            for (lake_id, branch_index), versions in vc_d_map.items():
                c_info = versions.get("C")
                d_info = versions.get("D")
                if c_info is None and d_info is None:
                    continue
                c_ok = bool(c_info and c_info.get("can_output", False))
                d_ok = bool(d_info and d_info.get("can_output", False))
                if c_ok and d_ok:
                    out_dir = vc_d_both_dir
                elif c_ok and not d_ok:
                    out_dir = vc_d_c_only_dir
                elif d_ok and not c_ok:
                    out_dir = vc_d_d_only_dir
                else:
                    continue

                branch_area_df = get_branch_area_series(area_points_df, lake_id, branch_index)
                out_png = os.path.join(out_dir, f"Lake_{lake_id}_branch_{branch_index}_vC_D.png")
                render_branch_vc_d_dual_axis_plot(
                    base_module=base,
                    lake_id=lake_id,
                    branch_index=branch_index,
                    records_c=(c_info or {}).get("records", []),
                    records_d=(d_info or {}).get("records", []),
                    out_path=out_png,
                    area_df=branch_area_df,
                    fixed_xlim=(PLOT_X_GLOBAL_START, PLOT_X_GLOBAL_END),
                )
                total_plots += 1

        print(f"[06_3] done {method_name}, plots={total_plots}", flush=True)


if __name__ == "__main__":
    main()


