from __future__ import annotations

from _result3_observation_runtime import load_script_module, patch_result3_common
import _result2_observation_common as obs_common


def main() -> None:
    patch_result3_common()
    mod = load_script_module("09_2_build_lake_observation_summary.py", "obs09_2_result3")
    mod.OUTPUT_DIR = obs_common.OUTPUT_DIR
    mod.SUMMARY_WORKBOOK = obs_common.SUMMARY_WORKBOOK
    mod.main()


if __name__ == "__main__":
    main()
