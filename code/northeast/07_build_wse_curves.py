"""07. Build the locked branch-wise red-chain WSE curves (original 06_3)."""

import argparse

from _engine._loader import CODE_DIR, RESULT_DIR, load_engine, report


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--run", action="store_true")
    ns = args.parse_args()
    out = RESULT_DIR / "07_wse_curves"
    if not report("07_build_wse_curves", out, ns.run):
        return

    m = load_engine("06_3_plot_branchwise_red_chain_with_merge_anchor.py")
    m.SCRIPT_DIR = CODE_DIR
    m.BASE_SCRIPT_PATH = (
        CODE_DIR / "_engine" / "copy_code" / "06_2_plot_single_red_chain_with_uncertainty.py"
    )
    m.OUTPUT_ROOT_NAME = "07_wse_curves"
    original_loader = m.load_base_module

    def configured_base():
        base = original_loader()
        base.OUTPUT_ROOT_DIR = str(RESULT_DIR)
        base.INPUT_05_2_DIR = str(
            RESULT_DIR / "04_lake_topology" / "05_2_denoised_branches"
        )
        base.INPUT_06_1_DIR = str(RESULT_DIR / "06_lake_wse" / "06_2_extract_raw_wse")
        base.OUTPUT_DIR = str(out)
        return base

    m.load_base_module = configured_base
    m.main()


if __name__ == "__main__":
    main()
