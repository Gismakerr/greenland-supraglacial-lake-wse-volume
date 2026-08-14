from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
PUBLIC_STAGES = (
    "01_s2_preprocessed",
    "02_lakes",
    "03_effective_area",
    "04_lake_topology",
    "05_swot_pixc",
    "06_lake_wse",
    "07_wse_curves",
    "09_observation_statistics",
    "12_paper_figures",
    "13_thematic_map_data",
)
IGNORED = {"FILE_MANIFEST.csv", "README.md", "VALIDATION.md"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_stage(stage: str) -> tuple[int, int]:
    stage_dir = RESULT_DIR / stage
    manifest_path = stage_dir / "FILE_MANIFEST.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = read_csv(manifest_path)
    expected = {row["relative_path"] for row in rows}
    actual = {
        path.relative_to(stage_dir).as_posix()
        for path in stage_dir.rglob("*")
        if path.is_file()
        and path.name not in IGNORED
        and not path.name.endswith(".sr.lock")
    }
    if expected != actual:
        raise ValueError(
            f"{stage}: file set changed; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    total = 0
    for row in rows:
        path = stage_dir / row["relative_path"]
        size = int(row["size_bytes"])
        if path.stat().st_size != size:
            raise ValueError(f"{stage}: size changed: {row['relative_path']}")
        if sha256(path) != row["sha256"].upper():
            raise ValueError(f"{stage}: SHA-256 changed: {row['relative_path']}")
        total += size
    return len(rows), total


def stage_cli(stage: str, engine_files: tuple[str, ...]) -> None:
    parser = argparse.ArgumentParser(description=f"Validate frozen southwest stage {stage}.")
    parser.add_argument(
        "--show-engine",
        action="store_true",
        help="Print the preserved original algorithm files for this stage.",
    )
    args = parser.parse_args()
    files, size = validate_stage(stage)
    print(f"VALID: {stage}: files={files}, bytes={size}")
    if args.show_engine:
        for name in engine_files:
            path = ROOT / "code" / "_engine" / name
            print(path)
