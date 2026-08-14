from __future__ import annotations

from collections import Counter

from _support import PUBLIC_STAGES, RESULT_DIR, read_csv, validate_stage


def main() -> None:
    for stage in PUBLIC_STAGES:
        validate_stage(stage)

    lakes = read_csv(RESULT_DIR / "02_lakes" / "Water_Max_Filtered_Polygons_WithAspect.csv")
    if len(lakes) != 2407 or len({row["lake_id"] for row in lakes}) != 2407:
        raise ValueError("Expected 2,407 unique southwest lakes.")

    curve_dir = RESULT_DIR / "07_wse_curves" / "06_3_BranchTables" / "vD"
    branch_tables = list(curve_dir.glob("lake_*_branch_*_stage_vD.csv"))
    lake_tables = list(curve_dir.glob("lake_*_all_branches_stage_vD.csv"))
    if len(branch_tables) != 306 or len(lake_tables) != 304:
        raise ValueError("Expected 306 branch tables and 304 lake tables.")

    manual = read_csv(RESULT_DIR / "13_thematic_map_data" / "tables" / "manual_branch_classification.csv")
    expected_categories = Counter(
        {"drainage_after_recharge": 8, "slow_drainage": 195, "stable_after_recharge": 5}
    )
    if len(manual) != 208 or Counter(row["category"] for row in manual) != expected_categories:
        raise ValueError("Manual classification cohort changed.")

    mapped = read_csv(RESULT_DIR / "13_thematic_map_data" / "tables" / "lake_map_attributes.csv")
    if len(mapped) != 2407 or Counter(row["has_curve"] for row in mapped) != Counter({"0": 2199, "1": 208}):
        raise ValueError("Thematic-map lake cohort changed.")

    print(
        "VALID PACKAGE: 2,407 lakes; 304 curve lakes; 306 branches; "
        "208 manually classified lakes; all stage hashes unchanged."
    )


if __name__ == "__main__":
    main()

