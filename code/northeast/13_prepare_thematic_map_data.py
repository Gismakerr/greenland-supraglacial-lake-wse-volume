from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result" / "13_thematic_map_data"
SPATIAL_DIR = RESULT_DIR / "spatial"
TABLE_DIR = RESULT_DIR / "tables"
MANIFEST_PATH = RESULT_DIR / "FILE_MANIFEST.csv"

MAP_TABLE = TABLE_DIR / "lake_map_attributes.csv"
MANUAL_TABLE = TABLE_DIR / "manual_branch_classification.csv"
SUMMARY_TABLE = TABLE_DIR / "classification_summary.csv"

EXPECTED_SPATIAL = {
    "lakes_thematic_map.shp",
    "lakes_thematic_map.shx",
    "lakes_thematic_map.dbf",
    "lakes_thematic_map.prj",
    "lakes_thematic_map.cpg",
}

BRANCH_COUNTS = {
    "drainage_after_recharge": 11,
    "slow_drainage": 115,
    "stable_after_recharge": 18,
    "sudden_drainage": 8,
}

MAP_COUNTS = {
    "no_valid_wse_time_series": 348,
    "drainage_after_recharge": 9,
    "slow_drainage": 115,
    "stable_after_recharge": 18,
    "sudden_drainage": 8,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def map_category(row: dict[str, str]) -> str:
    category = row["category"].strip()
    return category if category else "no_valid_wse_time_series"


def write_summary() -> None:
    map_rows = read_csv(MAP_TABLE)
    manual_rows = read_csv(MANUAL_TABLE)
    branch_counts = Counter(row["category"].strip() for row in manual_rows)
    map_counts = Counter(map_category(row) for row in map_rows)
    output_rows: list[dict[str, object]] = []
    for category in sorted(set(BRANCH_COUNTS) | set(MAP_COUNTS)):
        output_rows.append(
            {
                "category": category,
                "manual_branch_count": branch_counts.get(category, 0),
                "thematic_map_lake_count": map_counts.get(category, 0),
            }
        )
    with SUMMARY_TABLE.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["category", "manual_branch_count", "thematic_map_lake_count"],
        )
        writer.writeheader()
        writer.writerows(output_rows)


def validate_manifest() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    manifest = read_csv(MANIFEST_PATH)
    for row in manifest:
        path = RESULT_DIR / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Size changed: {row['relative_path']}")
        if sha256(path) != row["sha256"].upper():
            raise ValueError(f"SHA-256 changed: {row['relative_path']}")


def validate() -> None:
    spatial_names = {path.name for path in SPATIAL_DIR.iterdir() if path.is_file()}
    if spatial_names != EXPECTED_SPATIAL:
        raise ValueError(
            f"Thematic shapefile components differ: missing={EXPECTED_SPATIAL-spatial_names}, "
            f"extra={spatial_names-EXPECTED_SPATIAL}"
        )

    map_rows = read_csv(MAP_TABLE)
    manual_rows = read_csv(MANUAL_TABLE)
    if len(map_rows) != 498 or len({row["lake_id_old"] for row in map_rows}) != 498:
        raise ValueError("The lake-level thematic table must contain 498 unique lakes.")
    if Counter(row["has_curve"].strip() for row in map_rows) != Counter({"0": 348, "1": 150}):
        raise ValueError("The has_curve distribution changed.")
    if Counter(map_category(row) for row in map_rows) != Counter(MAP_COUNTS):
        raise ValueError("The lake-level map category distribution changed.")

    pair_keys = {(row["lake_id_old"], row["branch_index"]) for row in manual_rows}
    if len(manual_rows) != 152 or len(pair_keys) != 152:
        raise ValueError("The manual classification table must contain 152 unique branch pairs.")
    if len({row["lake_id_old"] for row in manual_rows}) != 150:
        raise ValueError("Manual branch classifications must represent 150 lakes.")
    if Counter(row["category"].strip() for row in manual_rows) != Counter(BRANCH_COUNTS):
        raise ValueError("The manual branch category distribution changed.")
    if sum(row["selected_for_thematic_map"].strip().lower() == "true" for row in manual_rows) != 150:
        raise ValueError("Exactly one branch per classified lake must be selected for the map.")

    selected = {
        row["lake_id_old"]: row
        for row in manual_rows
        if row["selected_for_thematic_map"].strip().lower() == "true"
    }
    if selected["1"]["branch_index"] != "1" or selected["2"]["branch_index"] != "2":
        raise ValueError("Lake 1/2 thematic-map branch selections changed.")

    validate_manifest()
    print(
        "VALID: one complete shapefile, 498 mapped lakes, "
        "152 manually classified branches from 150 lakes."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen thematic-map shapefile and manual behavior classifications."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Regenerate the classification summary from frozen X-drive tables, then validate.",
    )
    args = parser.parse_args()
    if args.run:
        write_summary()
    validate()


if __name__ == "__main__":
    main()
