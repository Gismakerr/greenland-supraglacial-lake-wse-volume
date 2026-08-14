"""Compatibility entrypoint for Stage 06_1.

Primary logic now lives in `06_1_core.py`.
"""

import importlib.util
from pathlib import Path


def load_stage_module():
    script_path = Path(__file__).resolve().parent / "06_1_core.py"
    spec = importlib.util.spec_from_file_location("stage06_1_core", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = load_stage_module()
    mod.main()


if __name__ == "__main__":
    main()

