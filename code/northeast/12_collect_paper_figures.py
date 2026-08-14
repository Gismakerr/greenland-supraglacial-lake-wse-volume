from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result" / "12_paper_figures"
FIGURE_DIR = RESULT_DIR / "figures"
FILE_MANIFEST = RESULT_DIR / "FILE_MANIFEST.csv"
FIGURE_INDEX = RESULT_DIR / "FIGURE_INDEX.csv"

# Only stage-10/11 products need copying here.  The area-WSE composite is
# written directly to FIGURE_DIR by 08b_compose_representative_area_wse.py.
UPSTREAM_FILES = {
    "volume_representative_6.png": ROOT
    / "result/10_lake_volume_change/runoff_volume_combined_rate_labels"
    / "representative_6_dual_axis_runoff_volume_combined_rate_labels.png",
    "volume_representative_6.tif": ROOT
    / "result/10_lake_volume_change/runoff_volume_combined_rate_labels"
    / "representative_6_dual_axis_runoff_volume_combined_rate_labels.tif",
    "lake1_slc_phase_part1.png": ROOT
    / "result/11_slc_phase_panels/figures/lake_1_slc_phase_part1.png",
    "lake1_slc_phase_part1.tif": ROOT
    / "result/11_slc_phase_panels/figures/lake_1_slc_phase_part1.tif",
    "lake1_slc_phase_part2.png": ROOT
    / "result/11_slc_phase_panels/figures/lake_1_slc_phase_part2.png",
    "lake1_slc_phase_part2.tif": ROOT
    / "result/11_slc_phase_panels/figures/lake_1_slc_phase_part2.tif",
}

# Stable manuscript bundle.  Extra working previews or backups in figures/
# are deliberately ignored rather than deleted.
OFFICIAL_FILES = {
    "wse_classification_summary.png": ("wse_classification_summary", "frozen_result"),
    "wse_area_hovmoller.png": ("wse_area_hovmoller", "frozen_result"),
    "area_wse_representative_stacked.png": (
        "area_wse_representative_stacked",
        "generated_by_08b",
    ),
    "area_wse_representative_stacked.tif": (
        "area_wse_representative_stacked",
        "generated_by_08b",
    ),
    "volume_representative_6.png": ("volume_representative_6", "generated_by_10c"),
    "volume_representative_6.tif": ("volume_representative_6", "generated_by_10c"),
    "lake1_slc_phase_part1.png": ("lake1_slc_phase_part1", "stage11_copy"),
    "lake1_slc_phase_part1.tif": ("lake1_slc_phase_part1", "stage11_copy"),
    "lake1_slc_phase_part2.png": ("lake1_slc_phase_part2", "stage11_copy"),
    "lake1_slc_phase_part2.tif": ("lake1_slc_phase_part2", "stage11_copy"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def refresh_manifest() -> None:
    rows: list[dict[str, object]] = []
    for file_name, (figure_key, source_type) in OFFICIAL_FILES.items():
        path = FIGURE_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
            image_format = "TIFF" if path.suffix.lower() in {".tif", ".tiff"} else "PNG"
        rows.append(
            {
                "figure_key": figure_key,
                "file_name": file_name,
                "format": image_format,
                "width_px": width,
                "height_px": height,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_type": source_type,
            }
        )
    pd.DataFrame(rows).to_csv(FILE_MANIFEST, index=False, encoding="utf-8-sig")


def validate() -> None:
    if not FILE_MANIFEST.exists() or not FIGURE_INDEX.exists():
        raise FileNotFoundError("Paper-figure index files are missing.")
    files = pd.read_csv(FILE_MANIFEST, encoding="utf-8-sig")
    figures = pd.read_csv(FIGURE_INDEX, encoding="utf-8-sig")
    expected_names = set(OFFICIAL_FILES)
    manifest_names = set(files["file_name"].astype(str))
    if manifest_names != expected_names or files["file_name"].nunique() != len(expected_names):
        raise ValueError(
            f"Manifest mismatch: missing={expected_names-manifest_names}, "
            f"extra={manifest_names-expected_names}"
        )
    if len(figures) != 6 or figures["figure_key"].nunique() != 6:
        raise ValueError("The paper index must contain exactly six conceptual figures.")
    actual_names = {path.name for path in FIGURE_DIR.iterdir() if path.is_file()}
    missing = expected_names - actual_names
    if missing:
        raise ValueError(f"Figure bundle is missing: {missing}")
    for row in files.itertuples(index=False):
        path = FIGURE_DIR / str(row.file_name)
        if path.stat().st_size != int(row.size_bytes):
            raise ValueError(f"Size changed: {path.name}")
        if sha256(path) != str(row.sha256).upper():
            raise ValueError(f"SHA-256 changed: {path.name}")
        with Image.open(path) as image:
            if image.width != int(row.width_px) or image.height != int(row.height_px):
                raise ValueError(f"Dimensions changed: {path.name}")
    print(
        "VALID: six selected manuscript figures, ten official PNG/TIFF files; "
        "dimensions and hashes verified."
    )


def run() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for output_name, source_path in UPSTREAM_FILES.items():
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, FIGURE_DIR / output_name)
    refresh_manifest()
    validate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect, inventory, and validate the selected manuscript figures."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Refresh stage-10/11 copies and rebuild the official manifest.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild FILE_MANIFEST.csv from current official files without copying.",
    )
    args = parser.parse_args()
    if args.run:
        run()
    elif args.refresh_manifest:
        refresh_manifest()
        validate()
    else:
        validate()


if __name__ == "__main__":
    main()
