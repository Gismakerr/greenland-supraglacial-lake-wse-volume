from __future__ import annotations

from pathlib import Path

import pandas as pd

import _result2_observation_common as obs_common


RESULT3_ROOT = Path(r"X:\2024_西南水位曲线\result3")


def _drop_unwanted_s2_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if "鍗槦骞冲彴" in str(col)]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _patch_result3_paths() -> None:
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


def main() -> None:
    _patch_result3_paths()
    obs_common.ensure_dir(obs_common.OUTPUT_DIR)
    obs_common.clean_csv_outputs()
    if obs_common.LEGACY_DAILY_WORKBOOK.exists():
        obs_common.LEGACY_DAILY_WORKBOOK.unlink()

    selected_lake_ids = obs_common.load_selected_lake_ids()
    if not selected_lake_ids:
        raise FileNotFoundError(f"没有在 {obs_common.OUTPUT_DIR.parent} 下找到 no_noise 湖列表")

    lake_meta = obs_common.load_lake_metadata(selected_lake_ids)
    s2_status_df = obs_common.build_s2_reference_status_records(selected_lake_ids, lake_meta)
    swot_status_df = obs_common.build_swot_reference_status_records(selected_lake_ids, lake_meta)

    if s2_status_df.empty:
        raise ValueError("没有生成 S2 状态记录，请检查 result1 的原始输入")
    if swot_status_df.empty:
        raise ValueError("没有生成 SWOT 状态记录，请检查 result3 的原始输入")

    s2_daily_df = obs_common.build_s2_reference_daily_records(s2_status_df)
    swot_daily_df = obs_common.build_swot_reference_daily_records(swot_status_df)
    s2_status_df = _drop_unwanted_s2_columns(s2_status_df)
    s2_daily_df = _drop_unwanted_s2_columns(s2_daily_df)

    with pd.ExcelWriter(obs_common.S2_STATUS_WORKBOOK, engine="openpyxl") as writer:
        s2_status_df.to_excel(writer, sheet_name="状态记录", index=False)
        s2_daily_df.to_excel(writer, sheet_name="按日统计", index=False)

    with pd.ExcelWriter(obs_common.SWOT_STATUS_WORKBOOK, engine="openpyxl") as writer:
        swot_status_df.to_excel(writer, sheet_name="状态记录", index=False)
        swot_daily_df.to_excel(writer, sheet_name="按日统计", index=False)

    obs_common.format_excel_workbook(obs_common.S2_STATUS_WORKBOOK)
    obs_common.format_excel_workbook(obs_common.SWOT_STATUS_WORKBOOK)

    print(f"[09_1_result3] selected lakes: {len(selected_lake_ids)}")
    print(f"[09_1_result3] S2 status rows: {len(s2_status_df)}")
    print(f"[09_1_result3] S2 daily rows: {len(s2_daily_df)}")
    print(f"[09_1_result3] SWOT status rows: {len(swot_status_df)}")
    print(f"[09_1_result3] SWOT daily rows: {len(swot_daily_df)}")
    print(f"[09_1_result3] output -> {obs_common.S2_STATUS_WORKBOOK}")
    print(f"[09_1_result3] output -> {obs_common.SWOT_STATUS_WORKBOOK}")


if __name__ == "__main__":
    main()
