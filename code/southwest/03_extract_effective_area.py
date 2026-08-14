from _support import stage_cli

if __name__ == "__main__":
    stage_cli("03_effective_area", (
        "03_generate_and_classify_previews.py",
        "04_1_segment_masks.py",
        "04_2_filter_and_classify.py",
        "04_3_export_effective_boundaries.py",
    ))
