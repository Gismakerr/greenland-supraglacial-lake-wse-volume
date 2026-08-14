"""Internal loader for the frozen pre-06_3 source modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
RESULT_DIR = PROJECT_DIR / "result"
ENGINE_DIR = CODE_DIR / "_engine"


def load_engine(filename: str):
    path = ENGINE_DIR / filename
    module_name = "_frozen_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(stage: str, output_dir: Path, run: bool) -> bool:
    print(f"[{stage}] output: {output_dir}")
    if not run:
        print("Check mode only. Add --run to execute the frozen algorithm.")
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    return True
