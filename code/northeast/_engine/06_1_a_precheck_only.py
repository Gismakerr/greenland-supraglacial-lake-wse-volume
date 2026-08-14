"""
Stage 06_1_a: run phase-1 PIXC precheck only (no extraction).
"""

import importlib.util
from pathlib import Path


def load_stage_module():
    script_path = Path(__file__).resolve().parent / "06_1_core.py"
    spec = importlib.util.spec_from_file_location("stage06_1_extract_raw_wse", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = load_stage_module()
    mod.RUN_PHASE1_PRECHECK = True
    mod.RUN_PHASE2_EXTRACT = False
    mod.PRECHECK_NC_BEFORE_EXTRACT = True
    mod.main()


if __name__ == "__main__":
    main()


