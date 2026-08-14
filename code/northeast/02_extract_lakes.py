"""02. Extract lake candidates, composites and the filtered lake inventory."""

import argparse

from _engine._loader import RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "02_lakes"
    if not report("02_extract_lakes", out, ns.run):
        return

    first = load_engine("02_1_extract_lake_aoi_binaries_and_vectors.py")
    first.INPUT_DIR = str(RESULT_DIR / "01_s2_preprocessed" / "04_Daily_Max_NDWI")
    first.OUTPUT_BASE = str(out)
    first.main()

    second = load_engine("02_2_extract_lake_aoi_composite_and_filtered.py")
    second.OUTPUT_BASE = str(out)
    second.main()


if __name__ == "__main__":
    main()
