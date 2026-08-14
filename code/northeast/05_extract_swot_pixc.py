"""05. Match, screen and extract SWOT PIXC observations for each lake."""

import argparse

from _engine._loader import RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "05_swot_pixc"
    if not report("05_extract_swot_pixc", out, ns.run):
        return

    m = load_engine("06_1_core.py")
    m.OUTPUT_ROOT_DIR = str(RESULT_DIR)
    m.LAKES_BUFFERED_SHP = str(
        RESULT_DIR / "02_lakes" / "Water_Max_Filtered_Polygons_Buffered.shp"
    )
    m.LAKE_FILTER_SHP = str(RESULT_DIR / "02_lakes" / "Water_Max_Filtered_Polygons.shp")
    m.INPUT_05_2_DIR = str(RESULT_DIR / "04_lake_topology" / "05_2_denoised_branches")
    m.OUTPUT_DIR = str(out / "06_1_extract_swot_pixc_to_gdb")
    m.main()


if __name__ == "__main__":
    main()
