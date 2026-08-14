from __future__ import annotations

import pandas as pd

from _result3_observation_runtime import load_script_module, patch_result3_common
import _result2_observation_common as obs_common


def main() -> None:
    patch_result3_common()
    mod = load_script_module("09_3_build_lake_observation_resummary.py", "obs09_3_result3")
    with pd.ExcelWriter(obs_common.RESUMMARY_WORKBOOK, engine="openpyxl") as writer:
        for sheet_name in mod.SHEETS:
            sheet_df = pd.read_excel(obs_common.SUMMARY_WORKBOOK, sheet_name=sheet_name)
            out_df = mod.build_resummary(sheet_df)
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

    obs_common.format_excel_workbook(obs_common.RESUMMARY_WORKBOOK)
    print(f"[09_3_result3] source -> {obs_common.SUMMARY_WORKBOOK}")
    print(f"[09_3_result3] output -> {obs_common.RESUMMARY_WORKBOOK}")


if __name__ == "__main__":
    main()
