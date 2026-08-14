from _support import stage_cli

if __name__ == "__main__":
    stage_cli("12_paper_figures", (
        "07_4_plot_classification.py",
        "07_4_plot_classification_combined.py",
        "lake_heatmap_project_v3/run_heatmaps_result3_fixed_wp_plotpoints.py",
        "lake_heatmap_project_v3/run_heatmaps_result3_interp_fixed_wp_plotpoints.py",
    ))
