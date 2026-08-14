from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import _result2_observation_common as obs_common


PROJECT_ROOT = Path(r"X:\2024_西南水位曲线")
RESULT3_ROOT = PROJECT_ROOT / "result3"
CODE_DIR = PROJECT_ROOT / "code1"


def patch_result3_common() -> None:
    obs_common.RESULT2_ROOT = RESULT3_ROOT
    obs_common.OUTPUT_DIR = RESULT3_ROOT / "09_1_lake_observation_status_records" / "no_noise_lakes"
    obs_common.S2_PREVIEW_ROOT = RESULT3_ROOT / "03_Classified_Previews"
    obs_common.S2_NDSI_DIR = RESULT3_ROOT / "01_Preprocessing_Results" / "02_NDSI"
    obs_common.S2_CLOUD_DIR = RESULT3_ROOT / "01_Preprocessing_Results" / "06_Cloud_Mask"
    obs_common.S2_DAILY_ALL_CSV = obs_common.S2_PREVIEW_ROOT / "Daily_Image_Classification_Summary_All.csv"
    obs_common.S2_POLLUTED_ALL_CSV = obs_common.S2_PREVIEW_ROOT / "Polluted_Image_Details_All.csv"
    obs_common.S2_EFFECTIVE_INDEX_CSV = RESULT3_ROOT / "04_3_export_effective_boundaries" / "Otsu" / "04_3_effective_boundary_index.csv"
    obs_common.LAKE_SHAPEFILE = RESULT3_ROOT / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons.shp"
    obs_common.S2_STATUS_WORKBOOK = obs_common.OUTPUT_DIR / "S2统计_无噪声固定水期.xlsx"
    obs_common.SWOT_STATUS_WORKBOOK = obs_common.OUTPUT_DIR / "SWOT统计_无噪声固定水期.xlsx"
    obs_common.LEGACY_DAILY_WORKBOOK = obs_common.OUTPUT_DIR / "SWOT按日统计_无噪声固定水期.xlsx"
    obs_common.SUMMARY_WORKBOOK = obs_common.OUTPUT_DIR / "无噪声固定水期汇总.xlsx"
    obs_common.RESUMMARY_WORKBOOK = obs_common.OUTPUT_DIR / "无噪声固定水期再汇总.xlsx"
    obs_common.STATUS_WORKBOOK = obs_common.SWOT_STATUS_WORKBOOK

    obs_common.SWOT_AUDIT_CSV = RESULT3_ROOT / "06_2_extract_raw_wse" / "Otsu" / "06_1_day_audit_table.csv"
    obs_common.SWOT_RAW_CSV = RESULT3_ROOT / "06_2_extract_raw_wse" / "Otsu" / "06_1_raw_wse_table.csv"
    obs_common.SWOT_BRANCH_DIR = (
        RESULT3_ROOT
        / "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"
        / "Otsu"
        / "original"
        / "06_3_BranchTables"
        / "vD"
    )
    obs_common.SWOT_NO_NOISE_DIR = (
        RESULT3_ROOT
        / "06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5"
        / "Otsu"
        / "original"
        / "vd"
        / "no_noise"
    )
    obs_common.SWOT_PIXC_GDB_DIR = RESULT3_ROOT / "06_1_extract_swot_pixc_to_gdb" / "Otsu" / "06_1_swot_pixc_gdb_by_lake"
    obs_common._SWOT_PIXC_NC_INDEX = None


def load_script_module(filename: str, module_name: str) -> ModuleType:
    path = CODE_DIR / filename
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
