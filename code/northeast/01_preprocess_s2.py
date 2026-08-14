"""01. Sentinel-2 filtering, NDWI/NDSI, cloud-shadow masks and daily maxima."""

import argparse

from _engine._loader import RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "01_s2_preprocessed"
    if not report("01_preprocess_s2", out, ns.run):
        return

    m = load_engine("01_data_preprocessing.py")
    m.OUTPUT_BASE = str(out)
    m.OUTPUT_DIR_FILTERED = str(out / "01_S2_Imgagery_Pro")
    m.OUTPUT_DIR_NDSI = str(out / "02_NDSI")
    m.OUTPUT_DIR_NDWI = str(out / "03_NDWI_ice")
    m.OUTPUT_DIR_MAX_NDWI = str(out / "04_Daily_Max_NDWI")
    m.OUTPUT_DIR_MAX_NDSI = str(out / "05_Daily_Max_NDSI")
    m.OUTPUT_BASE_MASK = str(out)
    m.OUTPUT_DIR_CLOUD_MASK = str(out / "06_Cloud_Mask_Result")
    m.OUTPUT_DIR_ROCK_SHADOW_MASK = str(out / "07_Rock_Shadow_Mask")
    m.OUTPUT_DIR_SHADOW_MASK = str(out / "08_Shadow_Mask")
    m.main()


if __name__ == "__main__":
    main()
