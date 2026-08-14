"""06. Compute quality-controlled raw lake water-surface elevations."""

import argparse

from _engine._loader import RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "06_lake_wse"
    if not report("06_compute_lake_wse", out, ns.run):
        return

    m = load_engine("06_2_compute_raw_wse_from_06_1.py")
    m.OUTPUT_ROOT_DIR = str(RESULT_DIR)
    m.INPUT_06_1_ROOT = str(RESULT_DIR / "05_swot_pixc" / "06_1_extract_swot_pixc_to_gdb")
    m.LAKES_BUFFERED_SHP = str(
        RESULT_DIR / "02_lakes" / "Water_Max_Filtered_Polygons_Buffered.shp"
    )
    m.LAKE_FILTER_SHP = str(RESULT_DIR / "02_lakes" / "Water_Max_Filtered_Polygons.shp")
    m.INPUT_05_1_DIR = str(RESULT_DIR / "04_lake_topology" / "05_1_build_topology")
    m.INPUT_05_2_DIR = str(RESULT_DIR / "04_lake_topology" / "05_2_denoised_branches")
    m.INPUT_05_2_BRANCH_SEED_DIR = m.INPUT_05_2_DIR
    m.INPUT_04_6_DIR = str(RESULT_DIR / "03_effective_area" / "04_3_export_effective_boundaries")
    m.OUTPUT_DIR = str(out / "06_2_extract_raw_wse")
    m.main()


if __name__ == "__main__":
    main()
