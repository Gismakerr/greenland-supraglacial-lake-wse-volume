from _support import stage_cli

if __name__ == "__main__":
    stage_cli("02_lakes", (
        "02_1_extract_lake_aoi_binaries_and_vectors_wev.py",
        "02_2_extract_lake_aoi_composite_and_filtered_wev.py",
        "02_3_label_lake_type_arcpy_wev.py",
        "02_4_compute_lake_aspect_ratio.py",
    ))
