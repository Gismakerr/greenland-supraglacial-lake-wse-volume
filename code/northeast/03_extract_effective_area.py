"""03. Segment, classify and export effective lake-area boundaries."""

import argparse

from _engine._loader import RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "03_effective_area"
    if not report("03_extract_effective_area", out, ns.run):
        return

    lakes = RESULT_DIR / "02_lakes"
    buffered = lakes / "Water_Max_Filtered_Polygons_Buffered.shp"
    polygons = lakes / "Water_Max_Filtered_Polygons.shp"

    segment = load_engine("04_1_segment_masks.py")
    segment.PREVIEW_BASE_DIR = str(out / "_preview")
    segment.NDWI_DIR = str(RESULT_DIR / "01_s2_preprocessed" / "03_NDWI_ice")
    segment.S2_IMAGE_DIR = str(RESULT_DIR / "01_s2_preprocessed" / "01_S2_Imgagery_Pro")
    segment.BUFFERED_LAKE_SHP = str(buffered)
    segment.LAKE_AREA_SHP = str(polygons)
    segment.OUTPUT_DIR = str(out / "04_1_segment_masks")
    segment.main()

    classify = load_engine("04_2_filter_and_classify.py")
    classify.BUFFERED_LAKE_SHP = str(buffered)
    classify.LAKE_AREA_SHP = str(polygons)
    classify.INPUT_DIR = str(out / "04_1_segment_masks")
    classify.OUTPUT_DIR = str(out / "04_2_filter_and_classify")
    classify.main()

    export = load_engine("04_3_export_effective_boundaries.py")
    export.BUFFER_SHP = str(buffered)
    export.LAKE_SHP = str(polygons)
    export.INPUT_04_1_DIR = str(out / "04_1_segment_masks")
    export.INPUT_04_2_DIR = str(out / "04_2_filter_and_classify")
    export.OUTPUT_DIR = str(out / "04_3_export_effective_boundaries")
    export.main()


if __name__ == "__main__":
    main()
