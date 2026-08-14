from __future__ import annotations

from _result3_observation_runtime import load_script_module, patch_result3_common
import _result2_observation_common as obs_common


def main() -> None:
    patch_result3_common()
    mod = load_script_module("09_4_plot_lake_swot_stacked_bars.py", "obs09_4_result3")
    mod.OUTPUT_DIR = obs_common.OUTPUT_DIR
    mod.SUMMARY_WORKBOOK = obs_common.SUMMARY_WORKBOOK
    mod.LAKE_SHAPEFILE = obs_common.LAKE_SHAPEFILE
    mod.OUTPUT_PNG = obs_common.OUTPUT_DIR / "09_4_SWOT观测构成_按湖堆叠柱状图_仅水期内.png"
    mod.OUTPUT_CSV = obs_common.OUTPUT_DIR / "09_4_SWOT观测构成_按湖堆叠柱状图_仅水期内.csv"
    mod.main()


if __name__ == "__main__":
    main()
