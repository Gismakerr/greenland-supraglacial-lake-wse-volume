from __future__ import annotations

from pathlib import Path

from _result3_observation_runtime import load_script_module, patch_result3_common
import _result2_observation_common as obs_common


def main() -> None:
    patch_result3_common()
    mod = load_script_module("09_4_plot_lake_swot_stacked_bars_xz_binned_0p2deg.py", "obs09_4_xz_bin_result3")
    mod.X_CSV = obs_common.OUTPUT_DIR / "09_4_SWOT观测构成_按湖堆叠柱状图_仅水期内.csv"
    mod.OUT_DIR = obs_common.OUTPUT_DIR
    mod.OUT_PNG = obs_common.OUTPUT_DIR / "09_4_X_Z_SWOT观测构成_按0p2度纬度带堆叠柱状图_同轴对比.png"
    mod.OUT_CSV = obs_common.OUTPUT_DIR / "09_4_X_Z_SWOT观测构成_按0p2度纬度带堆叠柱状图_同轴对比.csv"
    mod.main()


if __name__ == "__main__":
    main()
