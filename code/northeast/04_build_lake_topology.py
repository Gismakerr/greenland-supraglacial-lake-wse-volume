"""04. Build stable lake branches and export denoised topology relations."""

import argparse
import os

from _engine._loader import RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "04_lake_topology"
    if not report("04_build_lake_topology", out, ns.run):
        return

    effective = RESULT_DIR / "03_effective_area" / "04_3_export_effective_boundaries"
    topology = load_engine("05_1_build_topology_metrics_from_04_6.py")
    topology.INPUT_04_6_DIR = str(effective)
    topology.OUTPUT_DIR = str(out / "05_1_build_topology")
    topology.INPUT_GDB_PATH_TEMPLATE = os.path.join(
        str(effective), "{method_name}", "04_3_effective_boundaries.gdb"
    )
    topology.INPUT_AREA_SUMMARY_TEMPLATE = os.path.join(
        str(effective), "{method_name}", "04_3_effective_boundary_index.csv"
    )
    topology.main()

    curves = load_engine("05_2_plot_singlebranch_from_05_1_04_6_1.py")
    curves.INPUT_05_1_DIR = str(out / "05_1_build_topology")
    curves.INPUT_04_3_DIR = str(effective)
    curves.OUTPUT_DIR = str(out / "05_2_denoised_branches")
    curves.LAKE_FILTER_SHP = str(RESULT_DIR / "02_lakes" / "Water_Max_Filtered_Polygons.shp")
    curves.main()


if __name__ == "__main__":
    main()
