"""
Stage 07_4: build lake elevation-sorted WSE curve plots only.

Inputs:
1. Stage 06_2 cleaned lake membership from CleanWithArea plot folders.
2. Stage 06_2 per-branch stage CSV tables for WSE points.
3. Water_Max_Filtered_Polygons.shp for lake elevation and lake type.

Outputs:
- Group WSE curve plots for All lakes
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas.errors import EmptyDataError

# Match font family setup used in 09_plot_scatter.py
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_ROOT_DIR = str(PROJECT_ROOT_DIR / "result3")
INPUT_METHOD_DIR_CANDIDATES = [
    os.path.join(OUTPUT_ROOT_DIR, "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"),
    os.path.join(OUTPUT_ROOT_DIR, "06_2_plot_branch_main_candidate_stage"),
    os.path.join(OUTPUT_ROOT_DIR, "06_6_plot_branch_main_candidate_stage"),
]
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "07_4_lake_elevation_curves_classification")
# New classification source (active):
# result_3_31/06_4_classify_lake_wse_behaviors_plot
INPUT_06_4_CLASSIFY_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_4_classify_lake_wse_behaviors_plot")
# Legacy source kept only as fallback compatibility.
INPUT_06_4_CLASSIFY_DIR_BACKUP = os.path.join(OUTPUT_ROOT_DIR, "06_4_classify_lake_wse_behaviors")
INPUT_06_4_CLASSIFY_SUBDIR = "vd"
INPUT_06_4_RANKED_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_4_export_all_features_from_06_3")
INPUT_06_4_RANKED_SUBDIR = "vd"
INPUT_06_4_RANKED_NO_NOISE_DIRNAME = "no_noise"
INPUT_06_3_BRANCHWISE_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5")
INPUT_06_3_BRANCHWISE_GROUP_DIR = "ranked"
INPUT_06_3_BRANCHWISE_SUBDIR = "vd"
INPUT_06_3_NO_NOISE_DIRNAME = "no_noise"
INPUT_06_3_ORIGINAL_DIRNAME = "original"
INPUT_06_3_BRANCH_TABLES_DIRNAME = "06_3_BranchTables"
INPUT_06_3_PIXC_VERSION_SUBDIR = "vD"
INPUT_06_3_FEATURE_MAIN_DIR = os.path.join(OUTPUT_ROOT_DIR, "画图", "data", "main")
PIXC_VERSION_TEXT = "D"
PIXC_VERSION_SUBDIR = "vD"
PIXC_PLOT_SUFFIX = "vD"
LAKE_SHP = os.path.join(
    OUTPUT_ROOT_DIR,
    "02_lakes",
    "Water_Max_Filtered_Polygons.shp",
)
CURRENT_WSE_CURVES_DIR = os.path.join(OUTPUT_ROOT_DIR, "07_wse_curves")
CURRENT_MANUAL_CLASSIFY_CSV = os.path.join(
    OUTPUT_ROOT_DIR, "13_thematic_map_data", "tables", "manual_branch_classification.csv"
)

DATE_RANGE_START = "2024-06-05"
DATE_RANGE_END = "2024-08-15"
TARGET_SIZE_CATEGORIES = ["All"]

_env_output_root_dir = str(os.environ.get("OUTPUT_ROOT_DIR", "")).strip()
if _env_output_root_dir:
    OUTPUT_ROOT_DIR = _env_output_root_dir
    INPUT_METHOD_DIR_CANDIDATES = [
        os.path.join(OUTPUT_ROOT_DIR, "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"),
        os.path.join(OUTPUT_ROOT_DIR, "06_2_plot_branch_main_candidate_stage"),
        os.path.join(OUTPUT_ROOT_DIR, "06_6_plot_branch_main_candidate_stage"),
    ]
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "07_4_lake_elevation_curves_classification")
    INPUT_06_4_CLASSIFY_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_4_classify_lake_wse_behaviors_plot")
    INPUT_06_4_CLASSIFY_DIR_BACKUP = os.path.join(OUTPUT_ROOT_DIR, "06_4_classify_lake_wse_behaviors")
    INPUT_06_4_RANKED_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_4_export_all_features_from_06_3")
    INPUT_06_3_BRANCHWISE_DIR = os.path.join(OUTPUT_ROOT_DIR, "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5")
    INPUT_06_3_FEATURE_MAIN_DIR = os.path.join(OUTPUT_ROOT_DIR, "画图", "data", "main")
    LAKE_SHP = os.path.join(
        OUTPUT_ROOT_DIR,
        "02_lakes",
        "Water_Max_Filtered_Polygons.shp",
    )
    CURRENT_WSE_CURVES_DIR = os.path.join(OUTPUT_ROOT_DIR, "07_wse_curves")
    CURRENT_MANUAL_CLASSIFY_CSV = os.path.join(
        OUTPUT_ROOT_DIR, "13_thematic_map_data", "tables", "manual_branch_classification.csv"
    )

# 四类参数统一放在这里（优先修改这里）
# 规则:
# 1) manual_branch_pairs 用 lake_id_new: [(lake_id_new, branch_index), ...]
# 2) manual_branch_pairs 留空 [] 时，自动按该类 top_n 选最大起伏分支
# 3) top_n 仅在 manual_branch_pairs 为空或数量不足时补选
CATEGORY_PLOT_SETTINGS = {
    "drainage_after_recharge": {
        "label": "Repeated Filling-Drainage",
        "color": "#000000",
        "top_n": 2,
        "manual_branch_pairs": [(1, 2), (40, 1)],
    },
    "stable_after_recharge": {
        "label": "Continuous Filling",
        "color": "#5AC47A",
        "top_n": 2,
        "manual_branch_pairs": [(2, 2), (32, 1)],
    },
    "sudden_drainage": {
        "label": "Sudden Drainage",
        "color": "#FF0000",
        "top_n": 2,
        "manual_branch_pairs": [(79, 1), (123, 1)],
    },
    "slow_drainage": {
        "label": "Slow Drainage",
        "color": "#9E9E9E",
        "top_n": 2,
        "manual_branch_pairs": [(36, 1), (61, 1)],
    },
}

# 手动分支选择策略:
# - "prefer_manual": 先用手动，再按 top_n 自动补足（当前默认）
# - "manual_only": 只使用手动里有效的分支，不自动补
MANUAL_SELECTION_MODE = "prefer_manual"

# 是否允许“跨类别”手动分支:
# False: 手动分支必须属于该类别（推荐，和分类一致）
# True: 手动分支只要存在就可强制画入该类别
ALLOW_MANUAL_OUTSIDE_CATEGORY = False

# 下方逻辑统一读取 CATEGORY_PLOT_SETTINGS，不再分散设置
BEHAVIOR_HIGHLIGHT_CONFIG = [
    {
        "name": category_name,
        "label": category_cfg.get("label", category_name),
        "color": category_cfg.get("color", "#D62728"),
        "top_n": int(category_cfg.get("top_n", 3)),
        "manual_branch_pairs": category_cfg.get("manual_branch_pairs", []),
        "lake_ids": [],
    }
    for category_name, category_cfg in CATEGORY_PLOT_SETTINGS.items()
]
CURVE_HIGHLIGHT_LINEWIDTH = 1.8
CURVE_HIGHLIGHT_ALPHA = 0.95
# 12条主图透明度规则（每类3条）：
# - rank1: 该类起伏最大
# - rank2: 该类起伏第二
# - rank3: 该类起伏最小（4条最透明）
CURVE_HIGHLIGHT_ALPHA_BY_RANK = {
    1: 0.95,
    2: 0.48,
    3: 0.38,
}
CURVE_HIGHLIGHT_SHOW_MARKERS = False
CURVE_HIGHLIGHT_MARKER = "o"
CURVE_HIGHLIGHT_MARKERSIZE = 2.4
CURVE_HIGHLIGHT_MARKEVERY = 1
CURVE_SHOW_HIGHLIGHT_LABELS = True
CURVE_HIGHLIGHT_LABEL_FONT_SIZE = 11
CURVE_HIGHLIGHT_LABEL_X_OFFSET_PT = 6
CURVE_HIGHLIGHT_LABEL_Y_STEP_PT = 8
CURVE_NON_SELECTED_BY_CATEGORY_ALPHA = 0.1
CURVE_ALL_BY_CATEGORY_LINEWIDTH = 1.5
CURVE_ALL_BY_CATEGORY_ALPHA = 1.0
# Dynamic alpha for "AllCurves_ByCategoryColor" figure only:
# - more members in a category -> more transparent
# - larger WSE amplitude of a curve -> less transparent
CURVE_DYNAMIC_ALPHA_ENABLED = True
CURVE_DYNAMIC_ALPHA_MIN = 0.10
CURVE_DYNAMIC_ALPHA_MAX = 0.95
CURVE_DYNAMIC_ALPHA_WEIGHT_CATEGORY_COUNT = 0.55
CURVE_DYNAMIC_ALPHA_WEIGHT_AMPLITUDE = 0.45
# 3.1) 不确定性阴影参数（高亮分类曲线）
# 说明：
# - 仅对高亮曲线绘制阴影，避免全量曲线叠加后可读性下降
# - 阴影使用分支CSV中的 wse_std（按日聚合后作为 wse_rel_std）
CURVE_SHOW_UNCERTAINTY_SHADOW = True
CURVE_UNCERTAINTY_SIGMA_SCALE = 2
CURVE_UNCERTAINTY_SHADOW_ALPHA = 0.16

# Overlay 鍥句腑鐨?Raw 鑳屾櫙
OVERLAY_RAW_COLOR = "#D0D0D0"
OVERLAY_RAW_LINEWIDTH = 0.65
OVERLAY_RAW_ALPHA = 0.12

# 4) 鎬讳綋缁熻鏇茬嚎
CURVE_SHOW_MEDIAN = False
CURVE_SHOW_IQR_BAND = False
CURVE_MEDIAN_COLOR = "#000000"
CURVE_MEDIAN_LINEWIDTH = 2.4
CURVE_MEDIAN_ALPHA = 0.95
CURVE_IQR_COLOR = "#8F8F8F"
CURVE_IQR_ALPHA = 0.28

# 5) 鐢诲竷涓庤緭鍑?
CURVE_FIG_WIDTH_STANDARD = 20
CURVE_FIG_HEIGHT_STANDARD = 10
CURVE_FIG_WIDTH_DENSE = 20
CURVE_FIG_HEIGHT_DENSE = 8
CURVE_DPI = 200
CURVE_SORT_MODE = "amplitude_asc"
CURVE_SHOW_TITLE = False
CURVE_SHOW_AXIS_LABELS = True
CURVE_SHOW_GRID = True
CURVE_SHOW_TICKS = True
CURVE_SHOW_COUNT_BOX = True
CURVE_SHOW_COUNT_BOX_MAIN = False
CURVE_BASE_COLOR = "#C8C8C8"
CURVE_BASE_LINEWIDTH = 0.6
CURVE_BASE_ALPHA = 0.30
RED_COLOR = "#D62728"
CURVE_SMOOTH_ENABLED = False
CURVE_SMOOTH_WINDOW = 5

# 6) 鍧愭爣杞磋寖鍥翠笌鑷姩鐣欑櫧
CURVE_X_AXIS_MIN = "2024-06-03"
CURVE_X_AXIS_MAX = None
CURVE_USE_CUSTOM_Y_LIMITS = True
CURVE_Y_AXIS_MIN = -0.3
CURVE_Y_AXIS_MAX = 9.0
CURVE_USE_CUSTOM_Y_TICKS = True
CURVE_Y_TICKS = [0, 2, 4, 6, 8]
CURVE_Y_AXIS_BREAK_VALUE = 14.0
CURVE_Y_AXIS_COMPRESS_TO_SPAN = 2.0
CURVE_USE_BROKEN_Y_AXIS = False
CURVE_BROKEN_Y_LOWER_MAX = 14.0
CURVE_BROKEN_Y_UPPER_MIN = 16.0
CURVE_BROKEN_HEIGHT_RATIOS = (1, 6)
CURVE_Y_PAD_MIN = 0.15
CURVE_Y_PAD_RATIO = 0.05

# 7) 鏃ユ湡鍒诲害涓庢枃瀛?
CURVE_X_TICK_DAY_INTERVAL = 10
CURVE_X_TICK_START = "2024-06-05"
CURVE_X_TICK_DATE_FORMAT = "%m-%d"
CURVE_X_TICK_ROTATION = 0
CURVE_TITLE_FONT_SIZE = 14
CURVE_AXIS_LABEL_FONT_SIZE = 36
CURVE_TICK_FONT_SIZE = 36
CURVE_TICK_MARK_LENGTH = 8
CURVE_TICK_MARK_WIDTH = 1
CURVE_COUNT_TEXT_FONT_SIZE = 12
CURVE_COUNT_TEXT_X = 0.02
CURVE_COUNT_TEXT_Y = 0.98
CURVE_X_LABEL_TEXT = "Date"
CURVE_Y_LABEL_TEXT = "Relative Water Level Change (m)"
CURVE_TITLE_DENOISED = "All Lakes Group Plot"
CURVE_TITLE_RAW = "All Lakes Group Plot (Raw)"
CURVE_TITLE_OVERLAY = "All Lakes Group Plot (Overlay)"

# 8) 缃戞牸涓庤竟妗?
CURVE_GRID_COLOR = "#D9D9D9"
CURVE_GRID_STYLE = "--"
CURVE_GRID_ALPHA = 0.35
CURVE_SPINE_COLOR = "#202020"
CURVE_SPINE_LINEWIDTH = 1.2
CURVE_COUNT_BOX_ALPHA = 0.8
CURVE_SHOW_LEGEND = True
CURVE_EXPORT_STANDALONE_LEGEND = True
CURVE_LEGEND_FONT_SIZE = 24
CURVE_LEGEND_LOC = "upper left"
CURVE_LEGEND_BBOX_TO_ANCHOR = (0.04, 0.97)
CURVE_LEGEND_BBOX_TO_ANCHOR_BROKEN = (0.04, 1.0)
CURVE_LEGEND_FACE_COLOR = "none"
CURVE_LEGEND_EDGE_COLOR = "none"
CURVE_LEGEND_FRAME_ALPHA = 0.0
CURVE_STANDALONE_LEGEND_FILENAME = "Legend_Category.png"

ID_FIELD_CANDIDATES = [
    "lake_id",
    "lakeid",
    "lake_id_new",
    "id",
]
ELEVATION_FIELD_CANDIDATES = [
    "elev_m",
    "elevation",
    "elev",
    "height",
]
TYPE_FIELD_CANDIDATES = [
    "lake_type",
    "type",
]


def progress(message: str):
    print(message, flush=True)


def read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    lower_map = {str(col).lower(): col for col in columns}
    for name in candidates:
        match = lower_map.get(name.lower())
        if match is not None:
            return match
    return None


def find_method_dirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
    )


def resolve_input_06_2_dir() -> Optional[str]:
    for candidate in INPUT_METHOD_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_stage_csv_dir(method_dir_06_2: str) -> str:
    for subdir_name in ["06_2_BranchTables", "06_4_BranchTables"]:
        path = os.path.join(method_dir_06_2, subdir_name)
        if os.path.isdir(path):
            return path
    return os.path.join(method_dir_06_2, "06_2_BranchTables")


def build_method_paths(method_dir_06_2: str) -> Dict[str, str]:
    method_name = os.path.basename(method_dir_06_2)
    return {
        "method_name": method_name,
        "stage_csv_dir": resolve_stage_csv_dir(method_dir_06_2),
        "plot_dir": os.path.join(OUTPUT_DIR, method_name),
    }


def load_lake_metadata(shp_path: str) -> Tuple[Dict[int, float], Dict[int, str]]:
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Lake SHP not found: {shp_path}")

    gdf = gpd.read_file(shp_path)
    id_col = detect_column(list(gdf.columns), ID_FIELD_CANDIDATES)
    elev_col = detect_column(list(gdf.columns), ELEVATION_FIELD_CANDIDATES)
    type_col = detect_column(list(gdf.columns), TYPE_FIELD_CANDIDATES)
    if id_col is None or elev_col is None or type_col is None:
        raise ValueError(f"Required fields not found in {shp_path}. Columns: {list(gdf.columns)}")

    work = gdf[[id_col, elev_col, type_col]].copy()
    work[id_col] = pd.to_numeric(work[id_col], errors="coerce")
    work[elev_col] = pd.to_numeric(work[elev_col], errors="coerce")
    work[type_col] = work[type_col].astype(str).str.strip()
    work = work.dropna(subset=[id_col, elev_col]).copy()
    work[id_col] = work[id_col].astype(int)
    work = work.sort_values([id_col, elev_col]).drop_duplicates(subset=[id_col], keep="last")

    elev_map = dict(zip(work[id_col], work[elev_col]))
    type_map = dict(zip(work[id_col], work[type_col]))
    return elev_map, type_map


def parse_orig_lake_id_from_plot_name(filename: str) -> Optional[int]:
    suffix = re.escape(str(PIXC_PLOT_SUFFIX))
    m_legacy = re.match(rf"Lake_(\d+)_orig_(\d+)_branch_(\d+)_{suffix}\.png$", str(filename), re.IGNORECASE)
    if m_legacy:
        return int(m_legacy.group(2))
    m_new = re.match(rf"Lake_(\d+)_branch_(\d+)_{suffix}\.png$", str(filename), re.IGNORECASE)
    if m_new:
        return int(m_new.group(1))
    return None


def parse_lake_branch_from_plot_name(filename: str) -> Optional[Tuple[int, int]]:
    suffix = re.escape(str(PIXC_PLOT_SUFFIX))
    m_legacy = re.match(rf"Lake_(\d+)_orig_(\d+)_branch_(\d+)_{suffix}\.png$", str(filename), re.IGNORECASE)
    if m_legacy:
        return int(m_legacy.group(1)), int(m_legacy.group(3))
    m_new = re.match(rf"Lake_(\d+)_branch_(\d+)_{suffix}\.png$", str(filename), re.IGNORECASE)
    if m_new:
        return int(m_new.group(1)), int(m_new.group(2))
    return None


def parse_lake_branch_from_no_noise_name(filename: str) -> Optional[Tuple[int, int]]:
    suffix = re.escape(str(PIXC_PLOT_SUFFIX))
    match = re.match(rf"Lake_(\d+)_branch_(\d+)_{suffix}\.png$", str(filename), re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def make_branch_key(lake_id: int, branch_index: int) -> Tuple[int, int]:
    return int(lake_id), int(branch_index)


def format_branch_key(branch_key: Tuple[int, int]) -> str:
    lake_id, branch_index = branch_key
    return f"{int(lake_id)}:{int(branch_index)}"


def resolve_classify_single_plot_dir(method_name: str) -> Optional[str]:
    # New classification source (flat categories):
    # .../06_4_classify_lake_wse_behaviors_plot/双轴图曲线_老ID/<category>/*.png
    new_candidate = os.path.join(INPUT_06_4_CLASSIFY_DIR, "双轴图曲线_老ID")
    if os.path.isdir(new_candidate):
        return new_candidate

    # Legacy source:
    # .../06_4_classify_lake_wse_behaviors/<method>/vc/single_lake_plots
    candidate = os.path.join(
        INPUT_06_4_CLASSIFY_DIR,
        method_name,
        INPUT_06_4_CLASSIFY_SUBDIR,
        "single_lake_plots",
    )
    if os.path.isdir(candidate):
        return candidate

    # Legacy fallback root
    backup_candidate = os.path.join(
        INPUT_06_4_CLASSIFY_DIR_BACKUP,
        method_name,
        INPUT_06_4_CLASSIFY_SUBDIR,
        "single_lake_plots",
    )
    if os.path.isdir(backup_candidate):
        return backup_candidate
    return None


def resolve_06_3_no_noise_dir(method_name: str) -> Optional[str]:
    candidates = [
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_BRANCHWISE_GROUP_DIR,
            INPUT_06_3_BRANCHWISE_SUBDIR,
            INPUT_06_3_NO_NOISE_DIRNAME,
        ),
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_BRANCHWISE_SUBDIR,
            INPUT_06_3_NO_NOISE_DIRNAME,
        ),
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_ORIGINAL_DIRNAME,
            INPUT_06_3_BRANCHWISE_SUBDIR,
            INPUT_06_3_NO_NOISE_DIRNAME,
        ),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_06_3_branch_table_vc_dir(method_name: str) -> Optional[str]:
    candidates = [
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_ORIGINAL_DIRNAME,
            INPUT_06_3_BRANCH_TABLES_DIRNAME,
            INPUT_06_3_PIXC_VERSION_SUBDIR,
        ),
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_ORIGINAL_DIRNAME,
            INPUT_06_3_BRANCH_TABLES_DIRNAME,
            "vC",
        ),
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_BRANCH_TABLES_DIRNAME,
            INPUT_06_3_PIXC_VERSION_SUBDIR,
        ),
        os.path.join(
            INPUT_06_3_BRANCHWISE_DIR,
            method_name,
            INPUT_06_3_BRANCH_TABLES_DIRNAME,
            "vC",
        ),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_06_4_ranked_no_noise_dir(method_name: str) -> Optional[str]:
    candidates = [
        os.path.join(
            INPUT_06_4_RANKED_DIR,
            method_name,
            "ranked",
            INPUT_06_4_RANKED_SUBDIR,
            INPUT_06_4_RANKED_NO_NOISE_DIRNAME,
        ),
        os.path.join(
            INPUT_06_4_RANKED_DIR,
            method_name,
            "ranked",
            "vC",
            INPUT_06_4_RANKED_NO_NOISE_DIRNAME,
        ),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def load_classified_branch_entries(
    classify_single_plot_dir: str,
    no_noise_dir: Optional[str] = None,
    branch_csv_dir: Optional[str] = None,
    ranked_branch_keys: Optional[set] = None,
    manual_classify_csv: Optional[str] = None,
) -> Dict[str, List[Dict[str, object]]]:
    category_entries: Dict[str, List[Dict[str, object]]] = {}
    no_noise_branch_keys = set()
    if no_noise_dir is not None and os.path.isdir(no_noise_dir):
        for name in os.listdir(no_noise_dir):
            if not str(name).lower().endswith(".png"):
                continue
            parsed = parse_lake_branch_from_no_noise_name(name)
            if parsed is None:
                continue
            no_noise_branch_keys.add(make_branch_key(parsed[0], parsed[1]))

    for cfg in BEHAVIOR_HIGHLIGHT_CONFIG:
        category_name = str(cfg["name"])
        category_dir = os.path.join(classify_single_plot_dir, category_name)
        entries: List[Dict[str, object]] = []
        if os.path.isdir(category_dir):
            for filename in sorted(os.listdir(category_dir)):
                parsed = parse_lake_branch_from_plot_name(filename)
                if parsed is None:
                    continue
                lake_id, branch_index = parsed
                if ranked_branch_keys and make_branch_key(lake_id, branch_index) not in ranked_branch_keys:
                    continue
                if no_noise_branch_keys and make_branch_key(lake_id, branch_index) not in no_noise_branch_keys:
                    continue
                orig_lake_id = parse_orig_lake_id_from_plot_name(filename)
                csv_lake_id = int(lake_id)
                if branch_csv_dir is not None and os.path.isdir(branch_csv_dir):
                    orig_csv = (
                        os.path.join(
                            branch_csv_dir,
                            f"lake_{int(orig_lake_id)}_branch_{int(branch_index)}_stage_{PIXC_VERSION_SUBDIR}.csv",
                        )
                        if orig_lake_id is not None
                        else None
                    )
                    new_csv = os.path.join(
                        branch_csv_dir, f"lake_{int(lake_id)}_branch_{int(branch_index)}_stage_{PIXC_VERSION_SUBDIR}.csv"
                    )
                    # single_lake_plots files are generated from original lake_id data and then
                    # renamed as Lake_<lake_id_new>_orig_<orig_lake_id>_branch_<b>_vC.png.
                    # So we must prefer original lake_id to stay consistent with those figures.
                    if orig_csv is not None and os.path.exists(orig_csv):
                        csv_lake_id = int(orig_lake_id)
                    elif os.path.exists(new_csv):
                        csv_lake_id = int(lake_id)
                entries.append(
                    {
                        "lake_id": int(lake_id),
                        "lake_id_new": int(lake_id),
                        "orig_lake_id": int(orig_lake_id) if orig_lake_id is not None else None,
                        "csv_lake_id": int(csv_lake_id),
                        "main_lake_id": int(lake_id),
                        "branch_index": int(branch_index),
                        "branch_key": make_branch_key(int(lake_id), int(branch_index)),
                        "filename": str(filename),
                    }
                )
        category_entries[category_name] = entries

    # Current result layout stores the final manual classification directly in CSV,
    # rather than duplicating each branch as a category-specific PNG.
    if manual_classify_csv and os.path.exists(manual_classify_csv):
        manual_df = read_csv_if_exists(manual_classify_csv)
        required = {"lake_id_old", "branch_index", "category"}
        if not manual_df.empty and required.issubset(manual_df.columns):
            manual_df = manual_df.copy()
            manual_df["lake_id_old"] = pd.to_numeric(manual_df["lake_id_old"], errors="coerce")
            manual_df["branch_index"] = pd.to_numeric(manual_df["branch_index"], errors="coerce")
            manual_df["category"] = manual_df["category"].astype(str).str.strip()
            manual_df = manual_df.dropna(subset=["lake_id_old", "branch_index"])
            for category_name in category_entries:
                rows = manual_df[manual_df["category"] == category_name]
                if category_entries[category_name] or rows.empty:
                    continue
                category_entries[category_name] = [
                    {
                        "lake_id": int(row.lake_id_old),
                        "lake_id_new": int(row.lake_id_old),
                        "orig_lake_id": None,
                        "csv_lake_id": int(row.lake_id_old),
                        "main_lake_id": int(row.lake_id_old),
                        "branch_index": int(row.branch_index),
                        "branch_key": make_branch_key(int(row.lake_id_old), int(row.branch_index)),
                        "filename": f"Lake_{int(row.lake_id_old)}_branch_{int(row.branch_index)}_vD.png",
                    }
                    for row in rows.itertuples(index=False)
                ]
    return category_entries


def load_06_3_feature_rank_map(method_name: str) -> Dict[int, Dict[str, float]]:
    candidates = [
        os.path.join(
            INPUT_06_3_FEATURE_MAIN_DIR,
            f"lakes_with_wse_curve_maxrange_{str(PIXC_VERSION_TEXT).lower()}_{method_name}.csv",
        ),
        os.path.join(
            INPUT_06_3_FEATURE_MAIN_DIR,
            f"lakes_with_wse_curve_maxrange_ranked_{str(PIXC_VERSION_TEXT).lower()}_{method_name}.csv",
        ),
    ]
    path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        # Fallback for this project: 06_4 merged export table
        fallback = os.path.join(
            OUTPUT_ROOT_DIR,
            "06_4_export_all_features_from_06_3",
            method_name,
            f"lakes_large_medium_with_wse_stats_{str(PIXC_VERSION_TEXT).lower()}_{method_name}.csv",
        )
        if not os.path.exists(fallback):
            return {}
        df = read_csv_if_exists(fallback)
        if df.empty:
            return {}
        need_cols = ["lake_id_ol", "max_rng_m", "branch_ind", "has_curve"]
        if any(c not in df.columns for c in need_cols):
            return {}
        work = df.copy()
        work["lake_id_ol"] = pd.to_numeric(work["lake_id_ol"], errors="coerce")
        work["max_rng_m"] = pd.to_numeric(work["max_rng_m"], errors="coerce")
        work["branch_ind"] = pd.to_numeric(work["branch_ind"], errors="coerce")
        work["has_curve"] = pd.to_numeric(work["has_curve"], errors="coerce")
        work = work[(work["has_curve"] == 1) & work["lake_id_ol"].notna()].copy()
        if work.empty:
            return {}
        feature_map: Dict[int, Dict[str, float]] = {}
        for _, row in work.iterrows():
            lake_id = int(row["lake_id_ol"])
            feature_map[lake_id] = {
                "max_rng_m": float(row["max_rng_m"]) if pd.notna(row["max_rng_m"]) else np.nan,
                "top_br": int(row["branch_ind"]) if pd.notna(row["branch_ind"]) else -1,
            }
        return feature_map

    df = read_csv_if_exists(path)
    if df.empty:
        return {}
    for col in ["lake_id", "max_rng_m", "top_br"]:
        if col not in df.columns:
            return {}

    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce")
    work["max_rng_m"] = pd.to_numeric(work["max_rng_m"], errors="coerce")
    work["top_br"] = pd.to_numeric(work["top_br"], errors="coerce")
    work = work.dropna(subset=["lake_id"]).copy()
    if work.empty:
        return {}

    feature_map: Dict[int, Dict[str, float]] = {}
    for _, row in work.iterrows():
        lake_id = int(row["lake_id"])
        feature_map[lake_id] = {
            "max_rng_m": float(row["max_rng_m"]) if pd.notna(row["max_rng_m"]) else np.nan,
            "top_br": int(row["top_br"]) if pd.notna(row["top_br"]) else -1,
        }
    return feature_map


def load_specific_branch_curve(
    branch_csv_dir: str,
    lake_id: int,
    branch_index: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    use_denoised: bool = True,
) -> pd.DataFrame:
    filename_candidates = [
        f"lake_{int(lake_id)}_branch_{int(branch_index)}_stage_{PIXC_VERSION_SUBDIR}.csv",
        f"lake_{int(lake_id)}_branch_{int(branch_index)}_stage.csv",
    ]
    csv_path = None
    for fname in filename_candidates:
        p = os.path.join(branch_csv_dir, fname)
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        return pd.DataFrame()

    df = read_csv_if_exists(csv_path)
    if df.empty or "date" not in df.columns or "wse_m" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["wse_m"] = pd.to_numeric(work["wse_m"], errors="coerce")
    work["wse_std"] = pd.to_numeric(work.get("wse_std"), errors="coerce")
    work["is_noise"] = (
        work.get("is_noise", False)
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    work["point_color"] = work.get("point_color", "").fillna("").astype(str).str.strip().str.upper()
    work = work.dropna(subset=["date", "wse_m"])
    work = work[(work["date"] >= start_ts) & (work["date"] <= end_ts)]
    if work.empty:
        return pd.DataFrame()

    # Keep denoised branch curves consistent with 06_3/06_4 "no_noise":
    # use all non-noise points (not only red points), otherwise valid
    # post-drain segments that are non-red can be incorrectly dropped.
    denoised_signal = work[~work["is_noise"]].copy()
    if denoised_signal.empty:
        denoised_signal = work[work["point_color"] == RED_COLOR.upper()].copy()
    if denoised_signal.empty:
        return pd.DataFrame()

    def _rms_nan(series: pd.Series) -> float:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(np.square(arr))))

    denoised_curve = (
        denoised_signal.assign(date_day=denoised_signal["date"].dt.normalize())
        .groupby("date_day", as_index=False)
        .agg(
            wse_m=("wse_m", "mean"),
            wse_rel_std=("wse_std", _rms_nan),
        )
        .sort_values("date_day")
        .reset_index(drop=True)
        .rename(columns={"date_day": "date"})
    )
    if denoised_curve.empty:
        return pd.DataFrame()
    first_wse = float(denoised_curve.iloc[0]["wse_m"])

    if use_denoised:
        curve_df = denoised_curve.copy()
    else:
        raw_signal = work.copy()
        if raw_signal.empty:
            return pd.DataFrame()
        curve_df = (
            raw_signal.sort_values(["date", "wse_m"]).reset_index(drop=True)[["date", "wse_m", "wse_std"]].copy()
        )
        curve_df["wse_rel_std"] = pd.to_numeric(curve_df["wse_std"], errors="coerce")

    curve_df["wse_rel"] = curve_df["wse_m"] - first_wse
    if "wse_rel_std" not in curve_df.columns:
        curve_df["wse_rel_std"] = np.nan
    return curve_df[["date", "wse_m", "wse_rel", "wse_rel_std"]].copy()


def build_curve_y_for_plot(
    lake_id: int,
    df: pd.DataFrame,
    baseline_min_map: Optional[Dict[int, float]] = None,
) -> pd.Series:
    curve_y = pd.to_numeric(df["wse_rel"], errors="coerce").copy()

    vals = curve_y.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return pd.Series(dtype=float, index=df.index)

    if baseline_min_map is not None and lake_id in baseline_min_map and np.isfinite(baseline_min_map[lake_id]):
        shift_val = float(baseline_min_map[lake_id])
    else:
        shift_val = float(np.nanmin(vals))

    return curve_y - shift_val


def apply_curve_smoothing(values):
    series = pd.to_numeric(values, errors="coerce")
    if not CURVE_SMOOTH_ENABLED:
        return series
    window = max(1, int(CURVE_SMOOTH_WINDOW))
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def transform_y_value_for_display(value: float) -> float:
    if value is None or not np.isfinite(value):
        return np.nan
    v = float(value)
    if v <= CURVE_Y_AXIS_BREAK_VALUE:
        return v
    top_span = max(1e-12, float(CURVE_Y_AXIS_MAX) - CURVE_Y_AXIS_BREAK_VALUE)
    ratio = float(CURVE_Y_AXIS_COMPRESS_TO_SPAN) / top_span
    return CURVE_Y_AXIS_BREAK_VALUE + (v - CURVE_Y_AXIS_BREAK_VALUE) * ratio


def transform_y_values_for_display(values) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return out

    finite_vals = arr[finite_mask]
    below_mask = finite_vals <= CURVE_Y_AXIS_BREAK_VALUE
    transformed = np.empty_like(finite_vals, dtype=float)
    transformed[below_mask] = finite_vals[below_mask]

    top_span = max(1e-12, float(CURVE_Y_AXIS_MAX) - CURVE_Y_AXIS_BREAK_VALUE)
    ratio = float(CURVE_Y_AXIS_COMPRESS_TO_SPAN) / top_span
    transformed[~below_mask] = (
        CURVE_Y_AXIS_BREAK_VALUE
        + (finite_vals[~below_mask] - CURVE_Y_AXIS_BREAK_VALUE) * ratio
    )
    out[finite_mask] = transformed
    return out


def apply_custom_y_axis(ax, y_limits: Optional[Tuple[float, float]] = None):
    if CURVE_USE_CUSTOM_Y_LIMITS and (CURVE_Y_AXIS_MIN is not None or CURVE_Y_AXIS_MAX is not None):
        y0 = transform_y_value_for_display(CURVE_Y_AXIS_MIN) if CURVE_Y_AXIS_MIN is not None else None
        y1 = transform_y_value_for_display(CURVE_Y_AXIS_MAX) if CURVE_Y_AXIS_MAX is not None else None
        ax.set_ylim(y0, y1)
    elif y_limits is not None:
        y0, y1 = y_limits
        ax.set_ylim(transform_y_value_for_display(y0), transform_y_value_for_display(y1))
    else:
        if CURVE_USE_CUSTOM_Y_LIMITS and CURVE_Y_AXIS_MAX is not None:
            y0 = transform_y_value_for_display(CURVE_Y_AXIS_MIN) if CURVE_Y_AXIS_MIN is not None else None
            y1 = transform_y_value_for_display(CURVE_Y_AXIS_MAX)
            ax.set_ylim(y0, y1)
        else:
            ax.set_ylim(
                bottom=(
                    transform_y_value_for_display(CURVE_Y_AXIS_MIN)
                    if CURVE_Y_AXIS_MIN is not None
                    else None
                )
            )

    if CURVE_USE_CUSTOM_Y_TICKS and CURVE_Y_TICKS:
        tick_values = [float(v) for v in CURVE_Y_TICKS]
        tick_positions = [transform_y_value_for_display(v) for v in tick_values]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f"{int(v)}" if float(v).is_integer() else f"{v:g}" for v in tick_values])


def create_curve_axes(figsize: Tuple[float, float]):
    if not CURVE_USE_BROKEN_Y_AXIS:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, [ax], ax, None

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(CURVE_BROKEN_HEIGHT_RATIOS), "hspace": 0.05},
    )
    return fig, [ax_top, ax_bottom], ax_bottom, ax_top


def apply_broken_y_axis_format(ax_top, ax_bottom):
    y_min = float(CURVE_Y_AXIS_MIN) if CURVE_Y_AXIS_MIN is not None else 0.0
    y_max = float(CURVE_Y_AXIS_MAX) if CURVE_Y_AXIS_MAX is not None else 20.0
    ax_bottom.set_ylim(y_min, float(CURVE_BROKEN_Y_LOWER_MAX))
    ax_top.set_ylim(float(CURVE_BROKEN_Y_UPPER_MIN), y_max)

    # Hide touching spines; the axis gap alone indicates the truncation.
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    lower_ticks = [
        v
        for v in CURVE_Y_TICKS
        if float(v) < float(CURVE_BROKEN_Y_LOWER_MAX)
    ]
    upper_ticks = [v for v in CURVE_Y_TICKS if float(v) >= float(CURVE_BROKEN_Y_UPPER_MIN)]
    if lower_ticks:
        ax_bottom.set_yticks(lower_ticks)
    if upper_ticks:
        ax_top.set_yticks(upper_ticks)


def apply_custom_x_ticks(ax):
    tick_start = pd.to_datetime(CURVE_X_TICK_START)
    _, x1 = ax.get_xlim()
    tick_end = pd.Timestamp(mdates.num2date(x1)).tz_localize(None).normalize()
    tick_start = tick_start.normalize()
    if tick_end < tick_start:
        tick_end = tick_start
    ticks = pd.date_range(tick_start, tick_end, freq=f"{int(CURVE_X_TICK_DAY_INTERVAL)}D")
    if len(ticks) == 0:
        ticks = pd.DatetimeIndex([tick_start])
    ax.set_xticks(ticks.to_pydatetime())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(CURVE_X_TICK_DATE_FORMAT))


def build_category_legend_handles_labels(
    category_legend_rows: Optional[List[Dict[str, object]]],
    out_path: str,
    include_all_categories: bool = False,
) -> Tuple[List[Line2D], List[str]]:
    handles = []
    labels = []
    row_map = {}
    if category_legend_rows:
        row_map = {str(r.get("category_name")): r for r in category_legend_rows}

    if include_all_categories:
        iter_rows = []
        for cfg in reversed(BEHAVIOR_HIGHLIGHT_CONFIG):
            cat_name = str(cfg.get("name"))
            row = row_map.get(
                cat_name,
                {
                    "category_name": cat_name,
                    "label": str(cfg.get("label", cat_name)),
                    "color": str(cfg.get("color", "#999999")),
                    "selected_count": 0,
                    "show_count": False,
                },
            )
            iter_rows.append(row)
    else:
        if not category_legend_rows:
            return handles, labels
        iter_rows = list(reversed(category_legend_rows))

    for row in iter_rows:
        if (not include_all_categories) and int(row.get("selected_count", 0)) <= 0:
            continue
        handles.append(Line2D([0], [0], color=row["color"], linewidth=CURVE_HIGHLIGHT_LINEWIDTH))
        show_count = include_all_categories or bool(row.get("show_count", False)) or (
            os.path.basename(str(out_path)) == "GroupPlot_RelWSE_AllCurves_ByCategoryColor.png"
        )
        labels.append(f"{row['label']} ({row['selected_count']})" if show_count else f"{row['label']}")
    return handles, labels


def save_standalone_legend(category_legend_rows: Optional[List[Dict[str, object]]], out_path: str):
    if not CURVE_EXPORT_STANDALONE_LEGEND:
        return
    handles, labels = build_category_legend_handles_labels(
        category_legend_rows,
        out_path,
        include_all_categories=True,
    )
    if not handles:
        return
    fig_leg, ax_leg = plt.subplots(figsize=(8, 3.2))
    ax_leg.axis("off")
    ax_leg.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=CURVE_LEGEND_FONT_SIZE,
        frameon=False,
    )
    legend_path = os.path.join(os.path.dirname(out_path), CURVE_STANDALONE_LEGEND_FILENAME)
    fig_leg.savefig(legend_path, dpi=CURVE_DPI, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_leg)
    progress(f"Saved: {legend_path}")


def compute_curve_amplitude_map(
    ordered_ids: List[int],
    wse_map: Dict[int, pd.DataFrame],
    baseline_min_map: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    amp_map = {}
    for lake_id in ordered_ids:
        df = wse_map.get(lake_id)
        if df is None or df.empty:
            continue
        curve_y = build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
        vals = pd.to_numeric(curve_y, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        amp_map[lake_id] = float(np.nanmax(vals) - np.nanmin(vals))
    return amp_map


def _safe_norm01(v: float, vmin: float, vmax: float, fallback: float = 0.5) -> float:
    if not np.isfinite(v) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return float(fallback)
    return float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))


def sort_ids_for_plot(
    ordered_ids: List[int],
    wse_map: Dict[int, pd.DataFrame],
    baseline_min_map: Optional[Dict[int, float]] = None,
    mode: str = "amplitude_asc",
) -> List[int]:
    if mode == "keep":
        return list(ordered_ids)

    amp_map = compute_curve_amplitude_map(ordered_ids, wse_map, baseline_min_map=baseline_min_map)

    if mode == "amplitude_desc":
        return sorted(
            ordered_ids,
            key=lambda x: (amp_map.get(x, -np.inf), x),
            reverse=True,
        )

    return sorted(
        ordered_ids,
        key=lambda x: (amp_map.get(x, np.inf), x),
    )


def build_category_highlight_lookup(
    ordered_keys: List[Tuple[int, int]],
    wse_map: Dict[Tuple[int, int], pd.DataFrame],
    category_entries_map: Dict[str, List[Dict[str, object]]],
    feature_rank_map: Optional[Dict[int, Dict[str, float]]] = None,
    baseline_min_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> Tuple[Dict[Tuple[int, int], Dict[str, object]], List[Dict[str, object]], List[Tuple[int, int]]]:
    highlight_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    legend_rows: List[Dict[str, object]] = []
    selected_branch_keys: List[Tuple[int, int]] = []
    used_branch_keys = set()
    ordered_key_set = set(ordered_keys)
    feature_rank_map = feature_rank_map or {}

    for cfg in BEHAVIOR_HIGHLIGHT_CONFIG:
        category_name = str(cfg["name"])
        label = str(cfg["label"])
        color = str(cfg["color"])
        top_n = max(0, int(cfg.get("top_n", 0)))
        manual_ids = [int(x) for x in cfg.get("lake_ids", []) if pd.notna(x)]
        manual_branch_pairs_raw = cfg.get("manual_branch_pairs", [])
        manual_branch_pairs: List[Tuple[int, int]] = []
        for item in manual_branch_pairs_raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    manual_branch_pairs.append(make_branch_key(int(item[0]), int(item[1])))
                except Exception:
                    continue
                continue
            text = str(item).strip()
            if ":" in text:
                left, right = text.split(":", 1)
                try:
                    manual_branch_pairs.append(make_branch_key(int(left.strip()), int(right.strip())))
                except Exception:
                    continue
        category_entries = category_entries_map.get(category_name, [])
        entry_branch_map: Dict[Tuple[int, int], Dict[str, object]] = {}
        entry_lake_map: Dict[int, List[Tuple[int, int]]] = {}
        for entry in category_entries:
            lake_id = int(entry["lake_id"])
            branch_index = int(entry["branch_index"])
            branch_key = make_branch_key(lake_id, branch_index)
            entry_branch_map[branch_key] = entry
            entry_lake_map.setdefault(lake_id, []).append(branch_key)
        selected_keys: List[Tuple[int, int]] = []

        branch_score_map = compute_curve_amplitude_map(
            [key for key in entry_branch_map.keys() if key in ordered_key_set and key in wse_map],
            wse_map,
            baseline_min_map=baseline_min_map,
        )

        def get_branch_score(branch_key: Tuple[int, int]) -> float:
            lake_id, _ = branch_key
            branch_score = float(branch_score_map.get(branch_key, np.nan))
            if np.isfinite(branch_score):
                return branch_score
            return float(feature_rank_map.get(int(lake_id), {}).get("max_rng_m", -np.inf))

        # 0) Branch-level manual override first.
        for lake_id_manual, branch_manual in manual_branch_pairs:
            branch_key = make_branch_key(lake_id_manual, branch_manual)
            if branch_key not in ordered_key_set:
                continue
            if branch_key not in entry_branch_map:
                continue
            if branch_key not in wse_map or wse_map[branch_key].empty:
                continue
            if branch_key in selected_keys:
                continue
            selected_keys.append(branch_key)
            used_branch_keys.add(branch_key)
            if len(selected_keys) >= top_n:
                break

        use_manual_only = str(MANUAL_SELECTION_MODE).strip().lower() == "manual_only"
        if not use_manual_only:
            # 1) Manual IDs first (if configured)
            for lake_id in manual_ids:
                if lake_id not in entry_lake_map:
                    continue
                candidate_keys = [
                    key for key in entry_lake_map.get(lake_id, [])
                    if key in ordered_key_set and key in wse_map and not wse_map[key].empty and key not in used_branch_keys and key not in selected_keys
                ]
                if not candidate_keys:
                    continue
                best_key = max(
                    candidate_keys,
                    key=lambda key: (
                        get_branch_score(key),
                        key[1],
                        key[0],
                    ),
                )
                selected_keys.append(best_key)
                used_branch_keys.add(best_key)
                if len(selected_keys) >= top_n:
                    break

            # 2) Fill remainder by largest feature-reported range in this category
            if len(selected_keys) < top_n:
                category_pool = [
                    key
                    for key in entry_branch_map.keys()
                    if key in ordered_key_set
                    and key in wse_map
                    and not wse_map[key].empty
                    and key not in used_branch_keys
                    and key not in selected_keys
                ]
                ranked_keys = sorted(
                    category_pool,
                    key=lambda key: (
                        get_branch_score(key),
                        key[0],
                        key[1],
                    ),
                    reverse=True,
                )
                for branch_key in ranked_keys:
                    selected_keys.append(branch_key)
                    used_branch_keys.add(branch_key)
                    if len(selected_keys) >= top_n:
                        break

        ranked_selected_keys = sorted(
            selected_keys,
            key=lambda key: (
                get_branch_score(key),
                key[0],
                key[1],
            ),
            reverse=True,
        )
        alpha_rank_map: Dict[Tuple[int, int], int] = {}
        for rank_pos, branch_key in enumerate(ranked_selected_keys, start=1):
            alpha_rank_map[branch_key] = rank_pos

        for branch_key in selected_keys:
            rank_pos = int(alpha_rank_map.get(branch_key, 1))
            alpha_val = float(CURVE_HIGHLIGHT_ALPHA_BY_RANK.get(rank_pos, CURVE_HIGHLIGHT_ALPHA))
            highlight_lookup[branch_key] = {
                "category_name": category_name,
                "label": label,
                "color": color,
                "alpha": alpha_val,
                "alpha_rank": rank_pos,
            }

        legend_rows.append(
            {
                "category_name": category_name,
                "label": label,
                "color": color,
                "selected_ids": selected_keys,
                "selected_count": len(selected_keys),
            }
        )

        for branch_key in selected_keys:
            if branch_key not in selected_branch_keys:
                selected_branch_keys.append(branch_key)

    return highlight_lookup, legend_rows, selected_branch_keys


def build_selected_branch_curve_maps(
    branch_csv_dir: str,
    branch_source_map: Dict[Tuple[int, int], Tuple[int, int]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[Dict[Tuple[int, int], pd.DataFrame], Dict[Tuple[int, int], pd.DataFrame]]:
    denoised_map: Dict[Tuple[int, int], pd.DataFrame] = {}
    raw_map: Dict[Tuple[int, int], pd.DataFrame] = {}
    for branch_key, source_pair in branch_source_map.items():
        lake_id, branch_index = int(source_pair[0]), int(source_pair[1])
        den_df = load_specific_branch_curve(
            branch_csv_dir=branch_csv_dir,
            lake_id=lake_id,
            branch_index=branch_index,
            start_ts=start_ts,
            end_ts=end_ts,
            use_denoised=True,
        )
        if not den_df.empty:
            denoised_map[branch_key] = den_df

        raw_df = load_specific_branch_curve(
            branch_csv_dir=branch_csv_dir,
            lake_id=lake_id,
            branch_index=branch_index,
            start_ts=start_ts,
            end_ts=end_ts,
            use_denoised=False,
        )
        if not raw_df.empty:
            raw_map[branch_key] = raw_df
    return denoised_map, raw_map


def sanitize_name_for_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")


def format_branch_display_label(branch_key) -> str:
    if isinstance(branch_key, (tuple, list)) and len(branch_key) >= 2:
        return f"Lake {int(branch_key[0])} | B{int(branch_key[1])}"
    return f"Lake {branch_key}"


def annotate_highlight_curve_label(ax, branch_key, df: pd.DataFrame, curve_y_display, color: str, idx: int):
    if df is None or df.empty:
        return
    dates = pd.to_datetime(df["date"], errors="coerce").to_numpy()
    vals = pd.to_numeric(curve_y_display, errors="coerce")
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return
    valid_idx = np.where(np.isfinite(vals))[0]
    if valid_idx.size == 0:
        return
    last_i = int(valid_idx[-1])
    if last_i >= len(dates) or pd.isna(dates[last_i]):
        return

    label = format_branch_display_label(branch_key)
    y_offset = (idx % 5 - 2) * CURVE_HIGHLIGHT_LABEL_Y_STEP_PT
    ax.annotate(
        label,
        xy=(dates[last_i], vals[last_i]),
        xytext=(CURVE_HIGHLIGHT_LABEL_X_OFFSET_PT, y_offset),
        textcoords="offset points",
        fontsize=CURVE_HIGHLIGHT_LABEL_FONT_SIZE,
        color=color,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.2),
        zorder=4.2 + idx * 0.01,
    )


def export_selected_category_curve_files(
    out_root_dir: str,
    classify_single_plot_dir: str,
    category_entries_map: Dict[str, List[Dict[str, object]]],
    selected_keys_by_category: Dict[str, List[Tuple[int, int]]],
):
    if os.path.isdir(out_root_dir):
        shutil.rmtree(out_root_dir, ignore_errors=True)
    os.makedirs(out_root_dir, exist_ok=True)

    for category_name, selected_keys in selected_keys_by_category.items():
        cat_dir = os.path.join(out_root_dir, sanitize_name_for_filename(category_name))
        os.makedirs(cat_dir, exist_ok=True)

        entries = category_entries_map.get(category_name, [])
        for branch_key in selected_keys:
            matched = None
            for entry in entries:
                entry_key = tuple(
                    entry.get(
                        "branch_key",
                        make_branch_key(int(entry.get("lake_id", -1)), int(entry.get("branch_index", -1))),
                    )
                )
                if entry_key == branch_key:
                    matched = entry
                    break
            if matched is None:
                continue

            filename = str(matched.get("filename", "")).strip()
            if not filename:
                continue
            src = os.path.join(classify_single_plot_dir, sanitize_name_for_filename(category_name), filename)
            if not os.path.exists(src):
                continue
            dst = os.path.join(cat_dir, filename)
            shutil.copy2(src, dst)


def build_group_quantile_series(
    ordered_ids: List[int],
    wse_map: Dict[int, pd.DataFrame],
    baseline_min_map: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    rows = []
    for lake_id in ordered_ids:
        df = wse_map.get(lake_id)
        if df is None or df.empty:
            continue

        curve_y = build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
        curve_y = apply_curve_smoothing(curve_y)
        tmp = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
                "value": pd.to_numeric(curve_y, errors="coerce"),
            }
        ).dropna(subset=["date", "value"])

        if not tmp.empty:
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["date", "q25", "median", "q75"])

    all_df = pd.concat(rows, ignore_index=True)
    summary = (
        all_df.groupby("date")["value"]
        .agg(
            q25=lambda s: s.quantile(0.25),
            median=lambda s: s.quantile(0.50),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return summary


def compute_group_curve_ylim(
    ordered_ids: List[int],
    wse_map: Dict[int, pd.DataFrame],
    baseline_min_map: Optional[Dict[int, float]] = None,
) -> Optional[Tuple[float, float]]:
    values = []
    for lake_id in ordered_ids:
        df = wse_map.get(lake_id)
        if df is None or df.empty:
            continue
        curve_y = build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
        curve_y = apply_curve_smoothing(curve_y)
        vals = pd.to_numeric(curve_y, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            values.extend(vals.tolist())

    if not values:
        return None

    ymax = max(values)
    pad = max(CURVE_Y_PAD_MIN, ymax * CURVE_Y_PAD_RATIO)
    return (CURVE_Y_AXIS_MIN, ymax + pad)


def draw_uncertainty_shadow(
    ax,
    df: pd.DataFrame,
    curve_y: pd.Series,
    color: str,
    zorder: float,
):
    if not CURVE_SHOW_UNCERTAINTY_SHADOW:
        return
    if df is None or df.empty or "wse_rel_std" not in df.columns:
        return
    sigma = pd.to_numeric(df["wse_rel_std"], errors="coerce") * float(CURVE_UNCERTAINTY_SIGMA_SCALE)
    y_mid = pd.to_numeric(curve_y, errors="coerce")
    y_low = y_mid - sigma
    y_high = y_mid + sigma
    x = pd.to_datetime(df["date"], errors="coerce")

    if CURVE_USE_BROKEN_Y_AXIS:
        low_disp = pd.to_numeric(y_low, errors="coerce").to_numpy(dtype=float)
        high_disp = pd.to_numeric(y_high, errors="coerce").to_numpy(dtype=float)
    else:
        low_disp = transform_y_values_for_display(y_low)
        high_disp = transform_y_values_for_display(y_high)
    x_arr = x.to_numpy()
    mask = np.isfinite(low_disp) & np.isfinite(high_disp) & pd.notna(x_arr)
    if int(np.count_nonzero(mask)) < 2:
        return
    ax.fill_between(
        x_arr[mask],
        low_disp[mask],
        high_disp[mask],
        color=color,
        alpha=float(CURVE_UNCERTAINTY_SHADOW_ALPHA),
        linewidth=0.0,
        zorder=zorder,
    )


def plot_group_wse_curves_same_color(
    ordered_ids: List[int],
    wse_map: Dict[int, pd.DataFrame],
    elev_map: Dict[int, float],
    title: str,
    out_path: str,
    figsize: Tuple[float, float] = (20, 8),
    y_limits: Optional[Tuple[float, float]] = None,
    baseline_min_map: Optional[Dict[int, float]] = None,
    category_highlight_lookup: Optional[Dict[int, Dict[str, object]]] = None,
    category_legend_rows: Optional[List[Dict[str, object]]] = None,
    highlight_curve_map: Optional[Dict[int, pd.DataFrame]] = None,
    non_selected_category_lookup: Optional[Dict[int, Dict[str, object]]] = None,
    show_non_selected_base: bool = True,
    force_base_linewidth: Optional[float] = None,
    force_base_alpha: Optional[float] = None,
    skip_unmapped_base_curves: bool = False,
    show_highlight_labels_override: Optional[bool] = None,
    show_count_box_override: Optional[bool] = None,
    standalone_legend_rows: Optional[List[Dict[str, object]]] = None,
    show_curve_points: bool = False,
):
    if not ordered_ids:
        return

    plot_ids = sort_ids_for_plot(
        ordered_ids,
        wse_map,
        baseline_min_map=baseline_min_map,
        mode=CURVE_SORT_MODE,
    )
    category_highlight_lookup = category_highlight_lookup or {}
    non_selected_category_lookup = non_selected_category_lookup or {}
    highlight_ids = set(category_highlight_lookup.keys())
    show_highlight_labels = CURVE_SHOW_HIGHLIGHT_LABELS if show_highlight_labels_override is None else bool(show_highlight_labels_override)
    show_count_box = CURVE_SHOW_COUNT_BOX if show_count_box_override is None else bool(show_count_box_override)

    fig, plot_axes, ax, ax_top = create_curve_axes(figsize)

    # 1) 普通曲线（可选显示）
    if show_non_selected_base:
        for lake_id in plot_ids:
            if lake_id in highlight_ids:
                continue
            df = wse_map.get(lake_id)
            if df is None or df.empty:
                continue

            curve_y = build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
            curve_y = apply_curve_smoothing(curve_y)
            curve_y_display = curve_y if CURVE_USE_BROKEN_Y_AXIS else transform_y_values_for_display(curve_y)
            base_info = non_selected_category_lookup.get(lake_id, {})
            if skip_unmapped_base_curves and ("color" not in base_info):
                continue
            base_color = str(base_info.get("color", CURVE_BASE_COLOR))
            base_alpha = float(base_info.get("alpha", CURVE_BASE_ALPHA))
            line_w = float(force_base_linewidth) if force_base_linewidth is not None else float(CURVE_BASE_LINEWIDTH)
            line_a = float(force_base_alpha) if force_base_alpha is not None else float(base_alpha)
            for cur_ax in plot_axes:
                cur_ax.plot(
                    df["date"],
                    curve_y_display,
                    linewidth=line_w,
                    alpha=line_a,
                    color=base_color,
                    linestyle="-",
                    zorder=1.5,
                )
                if show_curve_points:
                    cur_ax.scatter(
                        df["date"],
                        curve_y_display,
                        s=16,
                        color=base_color,
                        alpha=min(1.0, max(0.15, line_a)),
                        edgecolors="none",
                        zorder=1.6,
                    )

    # 2) 楂樹寒婀栵細褰╄壊
    highlight_amp_map = compute_curve_amplitude_map(
        [lake_id for lake_id in plot_ids if lake_id in highlight_ids],
        wse_map,
        baseline_min_map=baseline_min_map,
    )
    highlight_list = sorted(
        [lake_id for lake_id in plot_ids if lake_id in highlight_ids],
        key=lambda lake_id: (highlight_amp_map.get(lake_id, -np.inf), lake_id),
    )
    for idx, lake_id in enumerate(highlight_list):
        if highlight_curve_map is not None and lake_id in highlight_curve_map:
            df = highlight_curve_map.get(lake_id)
        else:
            df = wse_map.get(lake_id)
        if df is None or df.empty:
            continue

        curve_y = build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
        curve_y = apply_curve_smoothing(curve_y)
        curve_y_display = curve_y if CURVE_USE_BROKEN_Y_AXIS else transform_y_values_for_display(curve_y)
        color = category_highlight_lookup.get(lake_id, {}).get("color", "#E66101")
        alpha_val = float(category_highlight_lookup.get(lake_id, {}).get("alpha", CURVE_HIGHLIGHT_ALPHA))
        for cur_ax in plot_axes:
            draw_uncertainty_shadow(cur_ax, df, curve_y, color=str(color), zorder=2.85 + idx * 0.01)
            cur_ax.plot(
                df["date"],
                curve_y_display,
                linewidth=CURVE_HIGHLIGHT_LINEWIDTH,
                alpha=alpha_val,
                color=color,
                marker=CURVE_HIGHLIGHT_MARKER if CURVE_HIGHLIGHT_SHOW_MARKERS else None,
                markersize=CURVE_HIGHLIGHT_MARKERSIZE if CURVE_HIGHLIGHT_SHOW_MARKERS else None,
                markevery=CURVE_HIGHLIGHT_MARKEVERY if CURVE_HIGHLIGHT_SHOW_MARKERS else None,
                zorder=3.0 + idx * 0.01,
            )
            if show_curve_points:
                cur_ax.scatter(
                    df["date"],
                    curve_y_display,
                    s=20,
                    color=color,
                    alpha=min(1.0, max(0.25, alpha_val)),
                    edgecolors="none",
                    zorder=3.05 + idx * 0.01,
                )
        if show_highlight_labels:
            annotate_highlight_curve_label(ax, lake_id, df, curve_y_display, color, idx)

    # 3) 鎬讳綋缁熻锛欼QR + 涓綅鏁?
    quant_df = build_group_quantile_series(plot_ids, wse_map, baseline_min_map=baseline_min_map)
    if not quant_df.empty:
        if CURVE_SHOW_IQR_BAND:
            q25 = quant_df["q25"] if CURVE_USE_BROKEN_Y_AXIS else transform_y_values_for_display(quant_df["q25"])
            q75 = quant_df["q75"] if CURVE_USE_BROKEN_Y_AXIS else transform_y_values_for_display(quant_df["q75"])
            for cur_ax in plot_axes:
                cur_ax.fill_between(
                    quant_df["date"],
                    q25,
                    q75,
                    color=CURVE_IQR_COLOR,
                    alpha=CURVE_IQR_ALPHA,
                    zorder=2.2,
                )
        if CURVE_SHOW_MEDIAN:
            q50 = quant_df["median"] if CURVE_USE_BROKEN_Y_AXIS else transform_y_values_for_display(quant_df["median"])
            for cur_ax in plot_axes:
                cur_ax.plot(
                    quant_df["date"],
                    q50,
                    color=CURVE_MEDIAN_COLOR,
                    linewidth=CURVE_MEDIAN_LINEWIDTH,
                    alpha=CURVE_MEDIAN_ALPHA,
                    zorder=3.8,
                )

    if CURVE_SHOW_TITLE:
        ax.set_title(title, fontsize=CURVE_TITLE_FONT_SIZE)

    if CURVE_SHOW_AXIS_LABELS:
        ax.set_xlabel(CURVE_X_LABEL_TEXT, fontsize=CURVE_AXIS_LABEL_FONT_SIZE)
        if CURVE_USE_BROKEN_Y_AXIS and ax_top is not None:
            ax.set_ylabel("")
            ax_top.set_ylabel("")
            fig.supylabel(CURVE_Y_LABEL_TEXT, fontsize=CURVE_AXIS_LABEL_FONT_SIZE, x=0.03)
        else:
            ax.set_ylabel(CURVE_Y_LABEL_TEXT, fontsize=CURVE_AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        if CURVE_USE_BROKEN_Y_AXIS and ax_top is not None:
            ax_top.set_ylabel("")

    if CURVE_USE_BROKEN_Y_AXIS and ax_top is not None:
        apply_broken_y_axis_format(ax_top, ax)
    else:
        apply_custom_y_axis(ax, y_limits=y_limits)

    if CURVE_X_AXIS_MIN is not None or CURVE_X_AXIS_MAX is not None:
        for cur_ax in plot_axes:
            cur_ax.set_xlim(
                pd.to_datetime(CURVE_X_AXIS_MIN) if CURVE_X_AXIS_MIN is not None else None,
                pd.to_datetime(CURVE_X_AXIS_MAX) if CURVE_X_AXIS_MAX is not None else None,
            )

    if CURVE_SHOW_GRID:
        for cur_ax in plot_axes:
            cur_ax.grid(True, linestyle=CURVE_GRID_STYLE, color=CURVE_GRID_COLOR, alpha=CURVE_GRID_ALPHA)

    apply_custom_x_ticks(ax)

    if CURVE_SHOW_TICKS:
        for cur_ax in plot_axes:
            cur_ax.tick_params(
                axis="both",
                labelsize=CURVE_TICK_FONT_SIZE,
                length=CURVE_TICK_MARK_LENGTH,
                width=CURVE_TICK_MARK_WIDTH,
            )
        plt.setp(ax.get_xticklabels(), rotation=CURVE_X_TICK_ROTATION, ha="center")

    if show_count_box:
        legend_lines = []
        if category_legend_rows:
            for row in category_legend_rows:
                selected_ids = row.get("selected_ids", [])
                legend_lines.append(f"{row['label']}: {len(selected_ids)} | Branches: {selected_ids}")
        ax.text(
            CURVE_COUNT_TEXT_X,
            CURVE_COUNT_TEXT_Y,
            "\n".join(
                [f"Total Branches: {len(plot_ids)} | Highlighted: {len(highlight_list)}"] + legend_lines
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=CURVE_COUNT_TEXT_FONT_SIZE,
            bbox=dict(facecolor="white", alpha=CURVE_COUNT_BOX_ALPHA),
        )

    if CURVE_SHOW_LEGEND and category_legend_rows:
        handles, labels = build_category_legend_handles_labels(category_legend_rows, out_path)
        if handles:
            if CURVE_USE_BROKEN_Y_AXIS and ax_top is not None:
                # Put legend inside the main (bottom) axes to avoid clipping on broken top panel.
                ax.legend(
                    handles,
                    labels,
                    loc=CURVE_LEGEND_LOC,
                    bbox_to_anchor=CURVE_LEGEND_BBOX_TO_ANCHOR_BROKEN,
                    fontsize=CURVE_LEGEND_FONT_SIZE,
                    frameon=True,
                    facecolor=CURVE_LEGEND_FACE_COLOR,
                    edgecolor=CURVE_LEGEND_EDGE_COLOR,
                    framealpha=CURVE_LEGEND_FRAME_ALPHA,
                )
            else:
                ax.legend(
                    handles,
                    labels,
                    loc=CURVE_LEGEND_LOC,
                    bbox_to_anchor=CURVE_LEGEND_BBOX_TO_ANCHOR,
                    fontsize=CURVE_LEGEND_FONT_SIZE,
                    frameon=True,
                    facecolor=CURVE_LEGEND_FACE_COLOR,
                    edgecolor=CURVE_LEGEND_EDGE_COLOR,
                    framealpha=CURVE_LEGEND_FRAME_ALPHA,
                )

    for cur_ax in plot_axes:
        for spine in cur_ax.spines.values():
            spine.set_linewidth(CURVE_SPINE_LINEWIDTH)
            spine.set_color(CURVE_SPINE_COLOR)

    if CURVE_USE_BROKEN_Y_AXIS:
        # Keep enough room for the top-axis legend and avoid tight_layout clipping.
        fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.11, hspace=0.04)
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=CURVE_DPI)
    plt.close(fig)
    progress(f"Saved: {out_path}")
    save_standalone_legend(standalone_legend_rows or category_legend_rows, out_path)


def plot_group_wse_curves_overlay(
    ordered_ids: List[int],
    denoised_map: Dict[int, pd.DataFrame],
    raw_map: Dict[int, pd.DataFrame],
    title: str,
    out_path: str,
    figsize: Tuple[float, float] = (20, 8),
    y_limits: Optional[Tuple[float, float]] = None,
    baseline_min_map: Optional[Dict[int, float]] = None,
    category_highlight_lookup: Optional[Dict[int, Dict[str, object]]] = None,
    category_legend_rows: Optional[List[Dict[str, object]]] = None,
    highlight_denoised_curve_map: Optional[Dict[int, pd.DataFrame]] = None,
    standalone_legend_rows: Optional[List[Dict[str, object]]] = None,
):
    if not ordered_ids:
        return

    plot_ids = sort_ids_for_plot(
        ordered_ids,
        denoised_map,
        baseline_min_map=baseline_min_map,
        mode=CURVE_SORT_MODE,
    )
    category_highlight_lookup = category_highlight_lookup or {}
    highlight_ids = set(category_highlight_lookup.keys())

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Raw锛氭瀬娣＄伆鑳屾櫙
    for lake_id in plot_ids:
        raw_df = raw_map.get(lake_id)
        if raw_df is None or raw_df.empty:
            continue

        raw_y = build_curve_y_for_plot(lake_id, raw_df, baseline_min_map=baseline_min_map)
        raw_y = apply_curve_smoothing(raw_y)
        raw_y_display = transform_y_values_for_display(raw_y)
        ax.plot(
            raw_df["date"],
            raw_y_display,
            linewidth=OVERLAY_RAW_LINEWIDTH,
            alpha=OVERLAY_RAW_ALPHA,
            color=OVERLAY_RAW_COLOR,
            zorder=1.0,
        )

    # 2) Denoised 鏅€氭箹锛氭祬鐏?
    for lake_id in plot_ids:
        if lake_id in highlight_ids:
            continue
        den_df = denoised_map.get(lake_id)
        if den_df is None or den_df.empty:
            continue

        den_y = build_curve_y_for_plot(lake_id, den_df, baseline_min_map=baseline_min_map)
        den_y = apply_curve_smoothing(den_y)
        den_y_display = transform_y_values_for_display(den_y)
        ax.plot(
            den_df["date"],
            den_y_display,
            linewidth=CURVE_BASE_LINEWIDTH,
            alpha=CURVE_BASE_ALPHA,
            color=CURVE_BASE_COLOR,
            zorder=2.0,
        )

    # 3) Denoised 楂樹寒婀栵細褰╄壊
    highlight_amp_map = compute_curve_amplitude_map(
        [lake_id for lake_id in plot_ids if lake_id in highlight_ids],
        denoised_map,
        baseline_min_map=baseline_min_map,
    )
    highlight_list = sorted(
        [lake_id for lake_id in plot_ids if lake_id in highlight_ids],
        key=lambda lake_id: (highlight_amp_map.get(lake_id, -np.inf), lake_id),
    )
    for idx, lake_id in enumerate(highlight_list):
        if highlight_denoised_curve_map is not None and lake_id in highlight_denoised_curve_map:
            den_df = highlight_denoised_curve_map.get(lake_id)
        else:
            den_df = denoised_map.get(lake_id)
        if den_df is None or den_df.empty:
            continue

        den_y = build_curve_y_for_plot(lake_id, den_df, baseline_min_map=baseline_min_map)
        den_y = apply_curve_smoothing(den_y)
        den_y_display = transform_y_values_for_display(den_y)
        color = category_highlight_lookup.get(lake_id, {}).get("color", "#E66101")
        alpha_val = float(category_highlight_lookup.get(lake_id, {}).get("alpha", CURVE_HIGHLIGHT_ALPHA))
        draw_uncertainty_shadow(ax, den_df, den_y, color=str(color), zorder=2.85 + idx * 0.01)
        ax.plot(
            den_df["date"],
            den_y_display,
            linewidth=CURVE_HIGHLIGHT_LINEWIDTH,
            alpha=alpha_val,
            color=color,
            marker=CURVE_HIGHLIGHT_MARKER if CURVE_HIGHLIGHT_SHOW_MARKERS else None,
            markersize=CURVE_HIGHLIGHT_MARKERSIZE if CURVE_HIGHLIGHT_SHOW_MARKERS else None,
            markevery=CURVE_HIGHLIGHT_MARKEVERY if CURVE_HIGHLIGHT_SHOW_MARKERS else None,
            zorder=3.0 + idx * 0.01,
        )
        if CURVE_SHOW_HIGHLIGHT_LABELS:
            annotate_highlight_curve_label(ax, lake_id, den_df, den_y_display, color, idx)

    # 4) Denoised 鎬讳綋缁熻
    quant_df = build_group_quantile_series(plot_ids, denoised_map, baseline_min_map=baseline_min_map)
    if not quant_df.empty:
        if CURVE_SHOW_IQR_BAND:
            ax.fill_between(
                quant_df["date"],
                transform_y_values_for_display(quant_df["q25"]),
                transform_y_values_for_display(quant_df["q75"]),
                color=CURVE_IQR_COLOR,
                alpha=CURVE_IQR_ALPHA,
                zorder=2.4,
            )
        if CURVE_SHOW_MEDIAN:
            ax.plot(
                quant_df["date"],
                transform_y_values_for_display(quant_df["median"]),
                color=CURVE_MEDIAN_COLOR,
                linewidth=CURVE_MEDIAN_LINEWIDTH,
                alpha=CURVE_MEDIAN_ALPHA,
                zorder=3.8,
            )

    if CURVE_SHOW_TITLE:
        ax.set_title(title, fontsize=CURVE_TITLE_FONT_SIZE)

    if CURVE_SHOW_AXIS_LABELS:
        ax.set_xlabel(CURVE_X_LABEL_TEXT, fontsize=CURVE_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(CURVE_Y_LABEL_TEXT, fontsize=CURVE_AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    apply_custom_y_axis(ax, y_limits=y_limits)

    if CURVE_X_AXIS_MIN is not None or CURVE_X_AXIS_MAX is not None:
        ax.set_xlim(
            pd.to_datetime(CURVE_X_AXIS_MIN) if CURVE_X_AXIS_MIN is not None else None,
            pd.to_datetime(CURVE_X_AXIS_MAX) if CURVE_X_AXIS_MAX is not None else None,
        )

    if CURVE_SHOW_GRID:
        ax.grid(True, linestyle=CURVE_GRID_STYLE, color=CURVE_GRID_COLOR, alpha=CURVE_GRID_ALPHA)

    apply_custom_x_ticks(ax)

    if CURVE_SHOW_TICKS:
        ax.tick_params(
            axis="both",
            labelsize=CURVE_TICK_FONT_SIZE,
            length=CURVE_TICK_MARK_LENGTH,
            width=CURVE_TICK_MARK_WIDTH,
        )
        plt.setp(ax.get_xticklabels(), rotation=CURVE_X_TICK_ROTATION, ha="center")

    if CURVE_SHOW_COUNT_BOX:
        legend_lines = []
        if category_legend_rows:
            for row in category_legend_rows:
                selected_ids = row.get("selected_ids", [])
                legend_lines.append(f"{row['label']}: {len(selected_ids)} | Branches: {selected_ids}")
        ax.text(
            CURVE_COUNT_TEXT_X,
            CURVE_COUNT_TEXT_Y,
            "\n".join(
                [f"Total Branches: {len(plot_ids)} | Highlighted: {len(highlight_list)}"] + legend_lines
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=CURVE_COUNT_TEXT_FONT_SIZE,
            bbox=dict(facecolor="white", alpha=CURVE_COUNT_BOX_ALPHA),
        )

    if CURVE_SHOW_LEGEND and category_legend_rows:
        handles, labels = build_category_legend_handles_labels(category_legend_rows, out_path)
        if handles:
            ax.legend(handles, labels, loc=CURVE_LEGEND_LOC, fontsize=CURVE_LEGEND_FONT_SIZE, frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(CURVE_SPINE_LINEWIDTH)
        spine.set_color(CURVE_SPINE_COLOR)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=CURVE_DPI)
    plt.close(fig)
    progress(f"Saved: {out_path}")
    save_standalone_legend(standalone_legend_rows or category_legend_rows, out_path)


def run_for_method(
    method_paths: Dict[str, str],
    elev_map: Dict[int, float],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
):
    classify_single_plot_dir = resolve_classify_single_plot_dir(method_paths["method_name"])
    if classify_single_plot_dir is None and os.path.isdir(CURRENT_WSE_CURVES_DIR):
        classify_single_plot_dir = CURRENT_WSE_CURVES_DIR
    if classify_single_plot_dir is None:
        progress(
            f"[07_4_classification] skip {method_paths['method_name']}: missing classify single_lake_plots dir"
        )
        return

    branch_table_vc_dir = resolve_06_3_branch_table_vc_dir(method_paths["method_name"])
    current_branch_dir = os.path.join(CURRENT_WSE_CURVES_DIR, "06_3_BranchTables", "vD")
    if branch_table_vc_dir is None and os.path.isdir(current_branch_dir):
        branch_table_vc_dir = current_branch_dir
    if branch_table_vc_dir is None:
        progress(
            f"[07_4_classification] skip {method_paths['method_name']}: missing 06_3 branch table vC dir"
        )
        return

    feature_rank_map = load_06_3_feature_rank_map(method_paths["method_name"])
    progress(f"[07_4_classification] classify single_lake_plots dir: {classify_single_plot_dir}")
    progress(f"[07_4_classification] branch table dir(vC): {branch_table_vc_dir}")
    progress(f"[07_4_classification] feature rank map lakes: {len(feature_rank_map)}")

    category_entries_map = load_classified_branch_entries(
        classify_single_plot_dir,
        no_noise_dir=None,
        branch_csv_dir=branch_table_vc_dir,
        ranked_branch_keys=None,
        manual_classify_csv=CURRENT_MANUAL_CLASSIFY_CSV,
    )
    full_category_branch_sources: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for cfg in BEHAVIOR_HIGHLIGHT_CONFIG:
        cat_name = str(cfg["name"])
        for entry in category_entries_map.get(cat_name, []):
            branch_key = tuple(entry.get("branch_key", make_branch_key(int(entry["lake_id"]), int(entry["branch_index"]))))  # type: ignore[arg-type]
            if branch_key in full_category_branch_sources:
                continue
            csv_lake_id = int(entry.get("csv_lake_id", int(entry["lake_id"])))
            branch_index = int(entry.get("branch_index", int(branch_key[1])))
            full_category_branch_sources[branch_key] = (csv_lake_id, branch_index)

    if not full_category_branch_sources:
        progress(f"[07_4_classification] skip {method_paths['method_name']}: no classified branch curves")
        return

    full_category_denoised_map, full_category_raw_map = build_selected_branch_curve_maps(
        branch_csv_dir=branch_table_vc_dir,
        branch_source_map=full_category_branch_sources,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    if not full_category_denoised_map and not full_category_raw_map:
        progress(f"[07_4_classification] skip {method_paths['method_name']}: no valid branch WSE series")
        return

    full_branch_map = full_category_denoised_map if full_category_denoised_map else full_category_raw_map
    ordered_branch_keys = [
        branch_key
        for branch_key in sorted(
            full_branch_map.keys(),
            key=lambda key: (
                elev_map.get(int(key[0]), -np.inf),
                int(key[0]),
                int(key[1]),
            ),
            reverse=True,
        )
    ]

    branch_baseline_min_map: Dict[Tuple[int, int], float] = {}
    for branch_key, df in full_category_denoised_map.items():
        if df is None or df.empty:
            continue
        vals = pd.to_numeric(df["wse_rel"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            branch_baseline_min_map[branch_key] = float(np.nanmin(vals))

    shared_y_limits = compute_group_curve_ylim(
        ordered_branch_keys,
        full_branch_map,
        baseline_min_map=branch_baseline_min_map,
    )
    branch_count = len(ordered_branch_keys)
    method_plot_dir = method_paths["plot_dir"]

    for target_type in TARGET_SIZE_CATEGORIES:
        fig_size = (
            (CURVE_FIG_WIDTH_DENSE, CURVE_FIG_HEIGHT_DENSE)
            if target_type in ["Medium", "Small", "All"]
            else (CURVE_FIG_WIDTH_STANDARD, CURVE_FIG_HEIGHT_STANDARD)
        )

        highlight_lookup, legend_rows, selected_branch_keys = build_category_highlight_lookup(
            ordered_branch_keys,
            full_branch_map,
            category_entries_map,
            feature_rank_map=feature_rank_map,
            baseline_min_map=branch_baseline_min_map,
        )
        selected_keys_by_category: Dict[str, List[Tuple[int, int]]] = {}

        for row in legend_rows:
            cat_name = str(row.get("category_name", ""))
            selected_keys = [tuple(x) for x in row.get("selected_ids", [])]
            selected_keys_by_category[cat_name] = list(selected_keys)
            selected_branch_range_map = compute_curve_amplitude_map(
                selected_keys,
                full_branch_map,
                baseline_min_map=branch_baseline_min_map,
            )
            selected_file_meta = []
            category_entries = category_entries_map.get(cat_name, [])
            for branch_key in selected_keys:
                matched = None
                for entry in category_entries:
                    entry_key = tuple(entry.get("branch_key", make_branch_key(int(entry["lake_id"]), int(entry["branch_index"]))))  # type: ignore[arg-type]
                    if entry_key == branch_key:
                        matched = entry
                        break
                if matched is not None:
                    lake_id = int(matched.get("lake_id", -1))
                    selected_file_meta.append(
                        {
                            "lake_id_new": int(matched.get("lake_id_new", lake_id)),
                            "main_lake_id": int(matched.get("main_lake_id", lake_id)),
                            "orig_lake_id": matched.get("orig_lake_id"),
                            "csv_lake_id": int(matched.get("csv_lake_id", lake_id)),
                            "branch": int(branch_key[1]),
                            "branch_key": format_branch_key(branch_key),
                            "max_rng_m": feature_rank_map.get(lake_id, {}).get("max_rng_m", np.nan),
                            "branch_rng_m": selected_branch_range_map.get(branch_key, np.nan),
                            "file": str(matched.get("filename", "")),
                        }
                    )
            progress(
                f"[07_4_classification] category={cat_name} selected_branch_keys={[format_branch_key(k) for k in selected_keys]}"
            )
            progress(f"[07_4_classification] category={cat_name} selected_file_meta={selected_file_meta}")

        selected_curve_out_root = os.path.join(method_plot_dir, "Selected_Category_Curves")
        export_selected_category_curve_files(
            out_root_dir=selected_curve_out_root,
            classify_single_plot_dir=classify_single_plot_dir,
            category_entries_map=category_entries_map,
            selected_keys_by_category=selected_keys_by_category,
        )
        progress(f"[07_4_classification] exported selected curves: {selected_curve_out_root}")

        category_color_by_name = {
            str(cfg["name"]): str(cfg["color"])
            for cfg in BEHAVIOR_HIGHLIGHT_CONFIG
        }
        all_category_legend_rows = [
            {
                "category_name": str(cfg["name"]),
                "label": str(cfg["label"]),
                "color": category_color_by_name.get(str(cfg["name"]), CURVE_BASE_COLOR),
                "selected_count": int(len(category_entries_map.get(str(cfg["name"]), []))),
                "show_count": True,
            }
            for cfg in BEHAVIOR_HIGHLIGHT_CONFIG
        ]
        # Per-category counts and per-curve amplitude are used to derive dynamic alpha.
        category_count_map: Dict[str, int] = {
            str(cat_name): int(len(entries))
            for cat_name, entries in category_entries_map.items()
        }
        count_values = np.array([v for v in category_count_map.values() if v > 0], dtype=float)
        count_min = float(np.nanmin(count_values)) if count_values.size > 0 else 1.0
        count_max = float(np.nanmax(count_values)) if count_values.size > 0 else 1.0
        amp_map_all = compute_curve_amplitude_map(
            ordered_branch_keys,
            full_branch_map,
            baseline_min_map=branch_baseline_min_map,
        )
        amp_values = np.array([v for v in amp_map_all.values() if np.isfinite(v)], dtype=float)
        amp_min = float(np.nanmin(amp_values)) if amp_values.size > 0 else 0.0
        amp_max = float(np.nanmax(amp_values)) if amp_values.size > 0 else 1.0

        all_category_curve_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
        for cat_name, entries in category_entries_map.items():
            cat_color = category_color_by_name.get(str(cat_name), CURVE_BASE_COLOR)
            cat_count = float(category_count_map.get(str(cat_name), 1))
            # More curves in category -> lower alpha.
            count_norm = _safe_norm01(cat_count, count_min, count_max, fallback=0.5)
            count_component = 1.0 - count_norm
            for entry in entries:
                key = tuple(entry.get("branch_key", make_branch_key(int(entry["lake_id"]), int(entry["branch_index"]))))  # type: ignore[arg-type]
                if key in all_category_curve_lookup:
                    continue
                amp_val = float(amp_map_all.get(key, np.nan))
                # Larger amplitude -> higher alpha.
                amp_component = _safe_norm01(amp_val, amp_min, amp_max, fallback=0.5)
                if CURVE_DYNAMIC_ALPHA_ENABLED:
                    w_count = float(CURVE_DYNAMIC_ALPHA_WEIGHT_CATEGORY_COUNT)
                    w_amp = float(CURVE_DYNAMIC_ALPHA_WEIGHT_AMPLITUDE)
                    w_sum = w_count + w_amp
                    if w_sum <= 0:
                        w_count, w_amp, w_sum = 0.5, 0.5, 1.0
                    score = (w_count * count_component + w_amp * amp_component) / w_sum
                    alpha_val = float(
                        CURVE_DYNAMIC_ALPHA_MIN
                        + np.clip(score, 0.0, 1.0) * (CURVE_DYNAMIC_ALPHA_MAX - CURVE_DYNAMIC_ALPHA_MIN)
                    )
                else:
                    alpha_val = float(CURVE_NON_SELECTED_BY_CATEGORY_ALPHA)
                all_category_curve_lookup[key] = {
                    "category_name": str(cat_name),
                    "color": cat_color,
                    "alpha": alpha_val,
                }

        selected_denoised_map = {
            branch_key: full_category_denoised_map[branch_key]
            for branch_key in selected_branch_keys
            if branch_key in full_category_denoised_map
        }
        selected_raw_map = {
            branch_key: full_category_raw_map[branch_key]
            for branch_key in selected_branch_keys
            if branch_key in full_category_raw_map
        }
        selected_branch_count = len(selected_branch_keys)
        mode_suffix = "ByCategory"
        mode_title_suffix = " | Highlight: By Category Branches (06_4 classify single_lake_plots)"

        if full_category_denoised_map:
            plot_group_wse_curves_same_color(
                ordered_ids=ordered_branch_keys,
                wse_map=full_category_denoised_map,
                elev_map=elev_map,
                title=f"{CURVE_TITLE_DENOISED}{mode_title_suffix} | Total Branches: {len(ordered_branch_keys)}",
                out_path=os.path.join(method_plot_dir, f"GroupPlot_RelWSE_SameColor_Denoised_{mode_suffix}.png"),
                figsize=fig_size,
                y_limits=shared_y_limits,
                baseline_min_map=branch_baseline_min_map,
                category_highlight_lookup=highlight_lookup,
                category_legend_rows=legend_rows,
                highlight_curve_map=selected_denoised_map,
                non_selected_category_lookup=all_category_curve_lookup,
                show_non_selected_base=False,
                show_highlight_labels_override=False,
                show_count_box_override=CURVE_SHOW_COUNT_BOX_MAIN,
                standalone_legend_rows=all_category_legend_rows,
                show_curve_points=True,
            )

        if full_category_raw_map:
            plot_group_wse_curves_same_color(
                ordered_ids=ordered_branch_keys,
                wse_map=full_category_raw_map,
                elev_map=elev_map,
                title=f"{CURVE_TITLE_RAW}{mode_title_suffix} | Total Branches: {len(ordered_branch_keys)}",
                out_path=os.path.join(method_plot_dir, f"GroupPlot_RelWSE_SameColor_Raw_{mode_suffix}.png"),
                figsize=fig_size,
                y_limits=shared_y_limits,
                baseline_min_map=branch_baseline_min_map,
                category_highlight_lookup=highlight_lookup,
                category_legend_rows=legend_rows,
                highlight_curve_map=selected_denoised_map if selected_denoised_map else selected_raw_map,
                show_non_selected_base=False,
                standalone_legend_rows=all_category_legend_rows,
            )

        # New figure: all curves shown, colored by behavior category (uniform linewidth/alpha, no uncertainty shadow).
        if full_category_denoised_map:
            plot_group_wse_curves_same_color(
                ordered_ids=ordered_branch_keys,
                wse_map=full_category_denoised_map,
                elev_map=elev_map,
                title=f"{CURVE_TITLE_DENOISED} | All Curves Colored By Category | Total Branches: {len(ordered_branch_keys)}",
                out_path=os.path.join(method_plot_dir, "GroupPlot_RelWSE_AllCurves_ByCategoryColor.png"),
                figsize=fig_size,
                y_limits=shared_y_limits,
                baseline_min_map=branch_baseline_min_map,
                category_highlight_lookup={},  # no highlighted branch -> no uncertainty shadow
                category_legend_rows=all_category_legend_rows,
                highlight_curve_map=None,
                non_selected_category_lookup={
                    k: {
                        "category_name": v.get("category_name"),
                        "color": v.get("color", CURVE_BASE_COLOR),
                        "alpha": float(v.get("alpha", CURVE_ALL_BY_CATEGORY_ALPHA)),
                    }
                    for k, v in all_category_curve_lookup.items()
                },
                show_non_selected_base=True,
                force_base_linewidth=CURVE_ALL_BY_CATEGORY_LINEWIDTH,
                force_base_alpha=None,
                skip_unmapped_base_curves=True,
                show_highlight_labels_override=False,
                show_count_box_override=False,
                standalone_legend_rows=all_category_legend_rows,
            )

        if full_category_denoised_map and full_category_raw_map:
            overlay_keys = [
                branch_key
                for branch_key in ordered_branch_keys
                if branch_key in full_category_denoised_map and branch_key in full_category_raw_map
            ]
            if overlay_keys:
                plot_group_wse_curves_overlay(
                    ordered_ids=overlay_keys,
                    denoised_map=full_category_denoised_map,
                    raw_map=full_category_raw_map,
                    title=f"{CURVE_TITLE_OVERLAY}{mode_title_suffix} | Total Branches: {len(overlay_keys)}",
                    out_path=os.path.join(method_plot_dir, f"GroupPlot_RelWSE_SameColor_Overlay_{mode_suffix}.png"),
                    figsize=fig_size,
                    y_limits=shared_y_limits,
                    baseline_min_map=branch_baseline_min_map,
                    category_highlight_lookup=highlight_lookup,
                    category_legend_rows=legend_rows,
                    highlight_denoised_curve_map=selected_denoised_map,
                    standalone_legend_rows=all_category_legend_rows,
                )

        for cfg in BEHAVIOR_HIGHLIGHT_CONFIG:
            cat_name = str(cfg["name"])
            cat_label = str(cfg["label"])
            cat_lookup = {
                branch_key: info
                for branch_key, info in highlight_lookup.items()
                if str(info.get("category_name")) == cat_name
            }
            cat_legend_rows = [
                row for row in legend_rows if str(row.get("category_name")) == cat_name
            ]
            if not cat_lookup:
                continue

            category_branch_keys = [
                tuple(entry.get("branch_key", make_branch_key(int(entry["lake_id"]), int(entry["branch_index"]))))  # type: ignore[arg-type]
                for entry in category_entries_map.get(cat_name, [])
            ]
            category_branch_keys = [
                branch_key
                for branch_key in ordered_branch_keys
                if branch_key in category_branch_keys and branch_key in full_category_denoised_map
            ]
            if not category_branch_keys:
                continue

            category_denoised_map = {
                branch_key: full_category_denoised_map[branch_key]
                for branch_key in category_branch_keys
                if branch_key in full_category_denoised_map
            }
            safe_cat = sanitize_name_for_filename(cat_name)
            plot_group_wse_curves_same_color(
                ordered_ids=category_branch_keys,
                wse_map=category_denoised_map,
                elev_map=elev_map,
                title=(
                    f"{CURVE_TITLE_DENOISED} | Category: {cat_label} "
                    f"| Selected: {len(cat_lookup)} | Total Branches: {len(category_branch_keys)}"
                ),
                out_path=os.path.join(
                    method_plot_dir,
                    f"GroupPlot_RelWSE_SameColor_Denoised_Category_{safe_cat}.png",
                ),
                figsize=fig_size,
                y_limits=shared_y_limits,
                baseline_min_map=branch_baseline_min_map,
                category_highlight_lookup=cat_lookup,
                category_legend_rows=cat_legend_rows,
                highlight_curve_map=selected_denoised_map,
                standalone_legend_rows=all_category_legend_rows,
            )

        progress(f"[07_4] done {method_paths['method_name']} {target_type}")
        progress(
            f"[07_4] summary {method_paths['method_name']} {target_type}: "
            f"branches={branch_count}, selected_category_branches={selected_branch_count}"
        )


def main():
    start_ts = pd.to_datetime(DATE_RANGE_START)
    end_ts = pd.to_datetime(DATE_RANGE_END)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    elev_map, _ = load_lake_metadata(LAKE_SHP)
    progress(f"[07_4] loaded elevation for {len(elev_map)} lakes from SHP")

    # Current packaged result: run once from its direct WSE tables.
    if os.path.isdir(os.path.join(CURRENT_WSE_CURVES_DIR, "06_3_BranchTables", "vD")):
        run_for_method(
            {"method_name": "current", "stage_csv_dir": os.path.join(CURRENT_WSE_CURVES_DIR, "06_3_BranchTables", "vD"), "plot_dir": OUTPUT_DIR},
            elev_map,
            start_ts,
            end_ts,
        )
        progress(f"[07_4] done. Output: {OUTPUT_DIR}")
        return

    input_06_2_dir = resolve_input_06_2_dir()
    if input_06_2_dir is None:
        progress(
            "[07_4] no method directories found: "
            + " | ".join(INPUT_METHOD_DIR_CANDIDATES)
        )
        return
    progress(f"[07_4] using input dir: {input_06_2_dir}")

    method_dirs = find_method_dirs(input_06_2_dir)
    if not method_dirs:
        progress("[07_4] no method directories found")
        return

    for method_dir in method_dirs:
        run_for_method(build_method_paths(method_dir), elev_map, start_ts, end_ts)

    progress(f"[07_4] done. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
