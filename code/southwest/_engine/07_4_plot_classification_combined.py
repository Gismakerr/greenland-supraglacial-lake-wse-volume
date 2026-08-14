"""
Stage 07_4 combined figure:
在一个 figure 中上下组合两类分类曲线图。

上图:
- 与 GroupPlot_RelWSE_SameColor_Denoised_ByCategory.png 同风格
- 不显示日期 tick 标签

下图:
- 与 GroupPlot_RelWSE_AllCurves_ByCategoryColor.png 同风格
- 保留日期 tick 标签
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = SCRIPT_DIR / "07_4_plot_classification.py"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("plot_classification_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base script: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()
FIXED_X_TICK_DATES = [
    "2024-06-05",
    "2024-06-15",
    "2024-06-25",
    "2024-07-05",
    "2024-07-15",
    "2024-07-25",
    "2024-08-05",
    "2024-08-15",
]


def create_panel_axes(fig: plt.Figure, subspec):
    """在外部 GridSpec 的单个单元中创建一组断轴坐标。"""
    if not base.CURVE_USE_BROKEN_Y_AXIS:
        ax = fig.add_subplot(subspec)
        return [ax], ax, None

    inner = subspec.subgridspec(
        2,
        1,
        height_ratios=list(base.CURVE_BROKEN_HEIGHT_RATIOS),
        hspace=0.05,
    )
    ax_top = fig.add_subplot(inner[0])
    ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)
    return [ax_top, ax_bottom], ax_bottom, ax_top


def style_curve_axes(
    fig: plt.Figure,
    plot_axes: List[plt.Axes],
    ax: plt.Axes,
    ax_top: Optional[plt.Axes],
    y_limits: Optional[Tuple[float, float]],
    hide_bottom_xticklabels: bool,
):
    """统一应用坐标轴、网格、刻度和边框样式。"""
    if base.CURVE_SHOW_AXIS_LABELS:
        if hide_bottom_xticklabels:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(base.CURVE_X_LABEL_TEXT, fontsize=base.CURVE_AXIS_LABEL_FONT_SIZE)
        if base.CURVE_USE_BROKEN_Y_AXIS and ax_top is not None:
            ax.set_ylabel("")
            ax_top.set_ylabel("")
            fig.supylabel("Relative WSE time series", fontsize=base.CURVE_AXIS_LABEL_FONT_SIZE, x=0.04)
        else:
            ax.set_ylabel("Relative WSE time series", fontsize=base.CURVE_AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ax_top is not None:
            ax_top.set_ylabel("")

    if base.CURVE_USE_BROKEN_Y_AXIS and ax_top is not None:
        base.apply_broken_y_axis_format(ax_top, ax)
    else:
        base.apply_custom_y_axis(ax, y_limits=y_limits)

    if base.CURVE_X_AXIS_MIN is not None or base.CURVE_X_AXIS_MAX is not None:
        for cur_ax in plot_axes:
            cur_ax.set_xlim(
                pd.to_datetime(base.CURVE_X_AXIS_MIN) if base.CURVE_X_AXIS_MIN is not None else None,
                pd.to_datetime(base.CURVE_X_AXIS_MAX) if base.CURVE_X_AXIS_MAX is not None else None,
            )

    if base.CURVE_SHOW_GRID:
        for cur_ax in plot_axes:
            cur_ax.grid(True, linestyle=base.CURVE_GRID_STYLE, color=base.CURVE_GRID_COLOR, alpha=base.CURVE_GRID_ALPHA)

    tick_dates = pd.to_datetime(FIXED_X_TICK_DATES)
    ax.set_xticks(tick_dates.to_pydatetime())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(base.CURVE_X_TICK_DATE_FORMAT))

    if base.CURVE_SHOW_TICKS:
        for cur_ax in plot_axes:
            cur_ax.tick_params(
                axis="both",
                labelsize=base.CURVE_TICK_FONT_SIZE,
                length=base.CURVE_TICK_MARK_LENGTH,
                width=base.CURVE_TICK_MARK_WIDTH,
            )
        plt.setp(ax.get_xticklabels(), rotation=base.CURVE_X_TICK_ROTATION, ha="center")

    if hide_bottom_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")

    for cur_ax in plot_axes:
        for spine in cur_ax.spines.values():
            spine.set_linewidth(base.CURVE_SPINE_LINEWIDTH)
            spine.set_color(base.CURVE_SPINE_COLOR)


def plot_same_color_panel(
    fig: plt.Figure,
    plot_axes: List[plt.Axes],
    ax: plt.Axes,
    ax_top: Optional[plt.Axes],
    *,
    ordered_ids,
    wse_map,
    y_limits,
    baseline_min_map,
    category_highlight_lookup,
    category_legend_rows,
    highlight_curve_map,
    non_selected_category_lookup,
    show_non_selected_base,
    force_base_linewidth,
    force_base_alpha,
    skip_unmapped_base_curves,
    show_highlight_labels_override,
    show_count_box_override,
    standalone_legend_rows,
    show_curve_points,
    out_path_for_labels: str,
    hide_bottom_xticklabels: bool,
    show_legend: bool,
    panel_label: str,
):
    """复用 07_4 样式，在给定 Axes 上绘制单个面板。"""
    plot_ids = base.sort_ids_for_plot(
        ordered_ids,
        wse_map,
        baseline_min_map=baseline_min_map,
        mode=base.CURVE_SORT_MODE,
    )
    category_highlight_lookup = category_highlight_lookup or {}
    non_selected_category_lookup = non_selected_category_lookup or {}
    highlight_ids = set(category_highlight_lookup.keys())
    show_highlight_labels = (
        base.CURVE_SHOW_HIGHLIGHT_LABELS
        if show_highlight_labels_override is None
        else bool(show_highlight_labels_override)
    )
    show_count_box = (
        base.CURVE_SHOW_COUNT_BOX
        if show_count_box_override is None
        else bool(show_count_box_override)
    )

    if show_non_selected_base:
        for lake_id in plot_ids:
            if lake_id in highlight_ids:
                continue
            df = wse_map.get(lake_id)
            if df is None or df.empty:
                continue
            curve_y = base.build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
            curve_y = base.apply_curve_smoothing(curve_y)
            curve_y_display = curve_y if base.CURVE_USE_BROKEN_Y_AXIS else base.transform_y_values_for_display(curve_y)
            base_info = non_selected_category_lookup.get(lake_id, {})
            if skip_unmapped_base_curves and ("color" not in base_info):
                continue
            base_color = str(base_info.get("color", base.CURVE_BASE_COLOR))
            base_alpha = float(base_info.get("alpha", base.CURVE_BASE_ALPHA))
            line_w = float(force_base_linewidth) if force_base_linewidth is not None else float(base.CURVE_BASE_LINEWIDTH)
            line_a = float(force_base_alpha) if force_base_alpha is not None else float(base_alpha)
            for cur_ax in plot_axes:
                cur_ax.plot(
                    df["date"],
                    curve_y_display,
                    linewidth=line_w,
                    alpha=line_a,
                    color=base_color,
                    linestyle="-",
                    zorder=1.5,
                )
                if show_curve_points:
                    cur_ax.scatter(
                        df["date"],
                        curve_y_display,
                        s=16,
                        color=base_color,
                        alpha=min(1.0, max(0.15, line_a)),
                        edgecolors="none",
                        zorder=1.6,
                    )

    highlight_amp_map = base.compute_curve_amplitude_map(
        [lake_id for lake_id in plot_ids if lake_id in highlight_ids],
        wse_map,
        baseline_min_map=baseline_min_map,
    )
    highlight_list = sorted(
        [lake_id for lake_id in plot_ids if lake_id in highlight_ids],
        key=lambda lake_id: (highlight_amp_map.get(lake_id, -np.inf), lake_id),
    )
    for idx, lake_id in enumerate(highlight_list):
        if highlight_curve_map is not None and lake_id in highlight_curve_map:
            df = highlight_curve_map.get(lake_id)
        else:
            df = wse_map.get(lake_id)
        if df is None or df.empty:
            continue
        curve_y = base.build_curve_y_for_plot(lake_id, df, baseline_min_map=baseline_min_map)
        curve_y = base.apply_curve_smoothing(curve_y)
        curve_y_display = curve_y if base.CURVE_USE_BROKEN_Y_AXIS else base.transform_y_values_for_display(curve_y)
        color = category_highlight_lookup.get(lake_id, {}).get("color", "#E66101")
        alpha_val = float(category_highlight_lookup.get(lake_id, {}).get("alpha", base.CURVE_HIGHLIGHT_ALPHA))
        for cur_ax in plot_axes:
            base.draw_uncertainty_shadow(cur_ax, df, curve_y, color=str(color), zorder=2.85 + idx * 0.01)
            cur_ax.plot(
                df["date"],
                curve_y_display,
                linewidth=base.CURVE_HIGHLIGHT_LINEWIDTH,
                alpha=alpha_val,
                color=color,
                marker=base.CURVE_HIGHLIGHT_MARKER if base.CURVE_HIGHLIGHT_SHOW_MARKERS else None,
                markersize=base.CURVE_HIGHLIGHT_MARKERSIZE if base.CURVE_HIGHLIGHT_SHOW_MARKERS else None,
                markevery=base.CURVE_HIGHLIGHT_MARKEVERY if base.CURVE_HIGHLIGHT_SHOW_MARKERS else None,
                zorder=3.0 + idx * 0.01,
            )
            if show_curve_points:
                cur_ax.scatter(
                    df["date"],
                    curve_y_display,
                    s=20,
                    color=color,
                    alpha=min(1.0, max(0.25, alpha_val)),
                    edgecolors="none",
                    zorder=3.05 + idx * 0.01,
                )
        if show_highlight_labels:
            base.annotate_highlight_curve_label(ax, lake_id, df, curve_y_display, color, idx)

    quant_df = base.build_group_quantile_series(plot_ids, wse_map, baseline_min_map=baseline_min_map)
    if not quant_df.empty:
        if base.CURVE_SHOW_IQR_BAND:
            q25 = quant_df["q25"] if base.CURVE_USE_BROKEN_Y_AXIS else base.transform_y_values_for_display(quant_df["q25"])
            q75 = quant_df["q75"] if base.CURVE_USE_BROKEN_Y_AXIS else base.transform_y_values_for_display(quant_df["q75"])
            for cur_ax in plot_axes:
                cur_ax.fill_between(quant_df["date"], q25, q75, color=base.CURVE_IQR_COLOR, alpha=base.CURVE_IQR_ALPHA, zorder=2.2)
        if base.CURVE_SHOW_MEDIAN:
            q50 = quant_df["median"] if base.CURVE_USE_BROKEN_Y_AXIS else base.transform_y_values_for_display(quant_df["median"])
            for cur_ax in plot_axes:
                cur_ax.plot(quant_df["date"], q50, color=base.CURVE_MEDIAN_COLOR, linewidth=base.CURVE_MEDIAN_LINEWIDTH, alpha=base.CURVE_MEDIAN_ALPHA, zorder=3.8)

    style_curve_axes(fig, plot_axes, ax, ax_top, y_limits, hide_bottom_xticklabels)

    # Panel tag stays clear of the shifted category legend.
    label_ax = ax_top if ax_top is not None else ax
    label_ax.text(
        0.01,
        0.97,
        panel_label,
        transform=label_ax.transAxes,
        ha="left",
        va="top",
        fontsize=base.CURVE_LEGEND_FONT_SIZE * 1.3,
    )

    if show_count_box:
        legend_lines = []
        if category_legend_rows:
            for row in category_legend_rows:
                selected_ids = row.get("selected_ids", [])
                legend_lines.append(f"{row['label']}: {len(selected_ids)} | Branches: {selected_ids}")
        ax.text(
            base.CURVE_COUNT_TEXT_X,
            base.CURVE_COUNT_TEXT_Y,
            "\n".join([f"Total Branches: {len(plot_ids)} | Highlighted: {len(highlight_list)}"] + legend_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=base.CURVE_COUNT_TEXT_FONT_SIZE,
            bbox=dict(facecolor="white", alpha=base.CURVE_COUNT_BOX_ALPHA),
        )

    legend_source = standalone_legend_rows or category_legend_rows
    if show_legend and base.CURVE_SHOW_LEGEND and legend_source:
        handles, labels = base.build_category_legend_handles_labels(legend_source, out_path_for_labels)
        if handles:
            target_ax = ax
            bbox_anchor = base.CURVE_LEGEND_BBOX_TO_ANCHOR_BROKEN if ax_top is not None else base.CURVE_LEGEND_BBOX_TO_ANCHOR
            target_ax.legend(
                handles,
                labels,
                loc=base.CURVE_LEGEND_LOC,
                bbox_to_anchor=(0.04, bbox_anchor[1]),
                fontsize=base.CURVE_LEGEND_FONT_SIZE,
                frameon=True,
                facecolor=base.CURVE_LEGEND_FACE_COLOR,
                edgecolor=base.CURVE_LEGEND_EDGE_COLOR,
                framealpha=base.CURVE_LEGEND_FRAME_ALPHA,
            )


def run_for_method_combined(method_paths: Dict[str, str], elev_map: Dict[int, float], start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    classify_single_plot_dir = base.resolve_classify_single_plot_dir(method_paths["method_name"])
    if classify_single_plot_dir is None:
        classify_single_plot_dir = getattr(base, "CURRENT_WSE_CURVES_DIR", None)
    if classify_single_plot_dir is None:
        base.progress(f"[07_4_combined] skip {method_paths['method_name']}: missing classify dir")
        return

    branch_table_vc_dir = base.resolve_06_3_branch_table_vc_dir(method_paths["method_name"])
    if branch_table_vc_dir is None:
        branch_table_vc_dir = os.path.join(getattr(base, "CURRENT_WSE_CURVES_DIR", ""), "06_3_BranchTables", "vD")
    if branch_table_vc_dir is None:
        base.progress(f"[07_4_combined] skip {method_paths['method_name']}: missing branch table dir")
        return

    feature_rank_map = base.load_06_3_feature_rank_map(method_paths["method_name"]) or {}
    category_entries_map = base.load_classified_branch_entries(
        classify_single_plot_dir,
        no_noise_dir=None,
        branch_csv_dir=branch_table_vc_dir,
        ranked_branch_keys=None,
        manual_classify_csv=getattr(base, "CURRENT_MANUAL_CLASSIFY_CSV", None),
    )

    full_category_branch_sources: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for cfg in base.BEHAVIOR_HIGHLIGHT_CONFIG:
        cat_name = str(cfg["name"])
        for entry in category_entries_map.get(cat_name, []):
            branch_key = tuple(entry.get("branch_key", base.make_branch_key(int(entry["lake_id"]), int(entry["branch_index"]))))
            if branch_key in full_category_branch_sources:
                continue
            csv_lake_id = int(entry.get("csv_lake_id", int(entry["lake_id"])))
            branch_index = int(entry.get("branch_index", int(branch_key[1])))
            full_category_branch_sources[branch_key] = (csv_lake_id, branch_index)

    if not full_category_branch_sources:
        base.progress(f"[07_4_combined] skip {method_paths['method_name']}: no classified branches")
        return

    full_category_denoised_map, _ = base.build_selected_branch_curve_maps(
        branch_csv_dir=branch_table_vc_dir,
        branch_source_map=full_category_branch_sources,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    if not full_category_denoised_map:
        base.progress(f"[07_4_combined] skip {method_paths['method_name']}: no valid denoised curves")
        return

    ordered_branch_keys = [
        branch_key
        for branch_key in sorted(
            full_category_denoised_map.keys(),
            key=lambda key: (elev_map.get(int(key[0]), -np.inf), int(key[0]), int(key[1])),
            reverse=True,
        )
    ]

    branch_baseline_min_map: Dict[Tuple[int, int], float] = {}
    for branch_key, df in full_category_denoised_map.items():
        if df is None or df.empty:
            continue
        vals = pd.to_numeric(df["wse_rel"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            branch_baseline_min_map[branch_key] = float(np.nanmin(vals))

    shared_y_limits = base.compute_group_curve_ylim(
        ordered_branch_keys,
        full_category_denoised_map,
        baseline_min_map=branch_baseline_min_map,
    )

    highlight_lookup, legend_rows, selected_branch_keys = base.build_category_highlight_lookup(
        ordered_branch_keys,
        full_category_denoised_map,
        category_entries_map,
        feature_rank_map=feature_rank_map,
        baseline_min_map=branch_baseline_min_map,
    )

    category_color_by_name = {str(cfg["name"]): str(cfg["color"]) for cfg in base.BEHAVIOR_HIGHLIGHT_CONFIG}
    all_category_legend_rows = [
        {
            "category_name": str(cfg["name"]),
            "label": str(cfg["label"]),
            "color": category_color_by_name.get(str(cfg["name"]), base.CURVE_BASE_COLOR),
            "selected_count": int(len(category_entries_map.get(str(cfg["name"]), []))),
            "show_count": True,
        }
        for cfg in base.BEHAVIOR_HIGHLIGHT_CONFIG
    ]
    category_count_map: Dict[str, int] = {str(cat_name): int(len(entries)) for cat_name, entries in category_entries_map.items()}
    count_values = np.array([v for v in category_count_map.values() if v > 0], dtype=float)
    count_min = float(np.nanmin(count_values)) if count_values.size > 0 else 1.0
    count_max = float(np.nanmax(count_values)) if count_values.size > 0 else 1.0
    amp_map_all = base.compute_curve_amplitude_map(
        ordered_branch_keys,
        full_category_denoised_map,
        baseline_min_map=branch_baseline_min_map,
    )
    amp_values = np.array([v for v in amp_map_all.values() if np.isfinite(v)], dtype=float)
    amp_min = float(np.nanmin(amp_values)) if amp_values.size > 0 else 0.0
    amp_max = float(np.nanmax(amp_values)) if amp_values.size > 0 else 1.0

    all_category_curve_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for cat_name, entries in category_entries_map.items():
        cat_color = category_color_by_name.get(str(cat_name), base.CURVE_BASE_COLOR)
        cat_count = float(category_count_map.get(str(cat_name), 1))
        count_norm = base._safe_norm01(cat_count, count_min, count_max, fallback=0.5)
        count_component = 1.0 - count_norm
        for entry in entries:
            key = tuple(entry.get("branch_key", base.make_branch_key(int(entry["lake_id"]), int(entry["branch_index"]))))
            if key in all_category_curve_lookup:
                continue
            amp_val = float(amp_map_all.get(key, np.nan))
            amp_component = base._safe_norm01(amp_val, amp_min, amp_max, fallback=0.5)
            if base.CURVE_DYNAMIC_ALPHA_ENABLED:
                w_count = float(base.CURVE_DYNAMIC_ALPHA_WEIGHT_CATEGORY_COUNT)
                w_amp = float(base.CURVE_DYNAMIC_ALPHA_WEIGHT_AMPLITUDE)
                w_sum = w_count + w_amp if (w_count + w_amp) > 0 else 1.0
                score = (w_count * count_component + w_amp * amp_component) / w_sum
                alpha_val = float(base.CURVE_DYNAMIC_ALPHA_MIN + np.clip(score, 0.0, 1.0) * (base.CURVE_DYNAMIC_ALPHA_MAX - base.CURVE_DYNAMIC_ALPHA_MIN))
            else:
                alpha_val = float(base.CURVE_NON_SELECTED_BY_CATEGORY_ALPHA)
            all_category_curve_lookup[key] = {
                "category_name": str(cat_name),
                "color": cat_color,
                "alpha": alpha_val,
            }

    selected_denoised_map = {
        branch_key: full_category_denoised_map[branch_key]
        for branch_key in selected_branch_keys
        if branch_key in full_category_denoised_map
    }

    fig = plt.figure(figsize=(base.CURVE_FIG_WIDTH_DENSE, 15.5))
    outer = fig.add_gridspec(2, 1, hspace=0.08)

    plot_axes1, ax1, ax1_top = create_panel_axes(fig, outer[0])
    plot_same_color_panel(
        fig,
        plot_axes1,
        ax1,
        ax1_top,
        ordered_ids=ordered_branch_keys,
        wse_map=full_category_denoised_map,
        y_limits=shared_y_limits,
        baseline_min_map=branch_baseline_min_map,
        category_highlight_lookup={},
        category_legend_rows=all_category_legend_rows,
        highlight_curve_map=None,
        non_selected_category_lookup={
            k: {
                "category_name": v.get("category_name"),
                "color": v.get("color", base.CURVE_BASE_COLOR),
                "alpha": float(v.get("alpha", base.CURVE_ALL_BY_CATEGORY_ALPHA)),
            }
            for k, v in all_category_curve_lookup.items()
        },
        show_non_selected_base=True,
        force_base_linewidth=base.CURVE_ALL_BY_CATEGORY_LINEWIDTH,
        force_base_alpha=None,
        skip_unmapped_base_curves=True,
        show_highlight_labels_override=False,
        show_count_box_override=base.CURVE_SHOW_COUNT_BOX_MAIN,
        standalone_legend_rows=all_category_legend_rows,
        show_curve_points=False,
        out_path_for_labels="GroupPlot_RelWSE_AllCurves_ByCategoryColor.png",
        hide_bottom_xticklabels=True,
        show_legend=True,
        panel_label="(a)",
    )

    plot_axes2, ax2, ax2_top = create_panel_axes(fig, outer[1])
    plot_same_color_panel(
        fig,
        plot_axes2,
        ax2,
        ax2_top,
        ordered_ids=ordered_branch_keys,
        wse_map=full_category_denoised_map,
        y_limits=shared_y_limits,
        baseline_min_map=branch_baseline_min_map,
        category_highlight_lookup=highlight_lookup,
        category_legend_rows=legend_rows,
        highlight_curve_map=selected_denoised_map,
        non_selected_category_lookup={
            k: {
                "category_name": v.get("category_name"),
                "color": v.get("color", base.CURVE_BASE_COLOR),
                "alpha": float(v.get("alpha", base.CURVE_ALL_BY_CATEGORY_ALPHA)),
            }
            for k, v in all_category_curve_lookup.items()
        },
        show_non_selected_base=False,
        force_base_linewidth=None,
        force_base_alpha=None,
        skip_unmapped_base_curves=False,
        show_highlight_labels_override=False,
        show_count_box_override=False,
        standalone_legend_rows=all_category_legend_rows,
        show_curve_points=True,
        out_path_for_labels="GroupPlot_RelWSE_SameColor_Denoised_ByCategory.png",
        hide_bottom_xticklabels=False,
        show_legend=False,
        panel_label="(b)",
    )

    fig.subplots_adjust(left=0.09, right=0.99, top=0.985, bottom=0.07)
    out_path = Path(method_paths["plot_dir"]) / "GroupPlot_RelWSE_Combined_TopSummary_BottomAllCurves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=base.CURVE_DPI)
    plt.close(fig)
    base.progress(f"Saved: {out_path}")


def main():
    start_ts = pd.to_datetime(base.DATE_RANGE_START)
    end_ts = pd.to_datetime(base.DATE_RANGE_END)
    Path(base.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    elev_map, _ = base.load_lake_metadata(base.LAKE_SHP)
    current_branch_dir = Path(getattr(base, "CURRENT_WSE_CURVES_DIR", "")) / "06_3_BranchTables" / "vD"
    if current_branch_dir.exists():
        run_for_method_combined({"method_name": "current", "plot_dir": base.OUTPUT_DIR}, elev_map, start_ts, end_ts)
        base.progress(f"[07_4_combined] done. Output: {base.OUTPUT_DIR}")
        return

    input_06_2_dir = base.resolve_input_06_2_dir()
    if input_06_2_dir is None:
        base.progress("[07_4_combined] no method directories found")
        return

    method_dirs = base.find_method_dirs(input_06_2_dir)
    if not method_dirs:
        base.progress("[07_4_combined] no method directories found")
        return

    for method_dir in method_dirs:
        run_for_method_combined(base.build_method_paths(method_dir), elev_map, start_ts, end_ts)

    base.progress(f"[07_4_combined] done. Output: {base.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
