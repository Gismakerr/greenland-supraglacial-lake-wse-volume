"""
Simplified Stage 05_2 (from 05_1_04_6_1 single-branch denoised results).

Inputs:
1) 05_1 raw plot features (denoised status included)
2) 05_1 topology GDB (for branch-start geometry export)
3) 04_3 effective boundary index CSV (for scene JPG paths)

Outputs (only):
1) 05_2_EffectiveAreaPlots_PureSWOTBlue_SingleBranch
2) 05_2_BranchStartJpgs
3) 05_2_branch_start_features.gdb
4) 05_2_CurvePointJpgs
5) 05_2_denoised_curve_topology_relations.csv
6) 05_2_CurvePointFeatureGDBs
"""

import os
import re
import shutil
from collections import defaultdict

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    import geopandas as gpd
except ImportError:
    gpd = None

matplotlib.use("Agg")


PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT_DIR = str(os.environ.get("OUTPUT_ROOT_DIR", r"X:\2024_西南水位曲线\result1")).strip()
INPUT_05_1_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_1_build_topology_metrics_from_04_6")
INPUT_04_3_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_3_export_effective_boundaries")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_2_plot_singlebranch_from_05_1_04_6_1")

# Manually flagged area-noise points removed in Stage 05_2.
# Format: lake_id + exact node key (date, preview_name, feature_index).
MANUAL_AREA_NOISE_RULES = [
    {"lake_id": 16, "date": "20240814", "preview_name": "overlay_16_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 16, "date": "20240812", "preview_name": "overlay_16_20240812_145941_S2A.png", "feature_index": 0},
    {"lake_id": 23, "date": "20240813", "preview_name": "overlay_23_20240813_142741_S2A.png", "feature_index": 0},
    {"lake_id": 23, "date": "20240813", "preview_name": "overlay_23_20240813_151809_S2B.png", "feature_index": 0},
    {"lake_id": 23, "date": "20240814", "preview_name": "overlay_23_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 23, "date": "20240811", "preview_name": "overlay_23_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 23, "date": "20240811", "preview_name": "overlay_23_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 23, "date": "20240812", "preview_name": "overlay_23_20240812_145941_S2A.png", "feature_index": 0},
    {"lake_id": 32, "date": "20240810", "preview_name": "overlay_32_20240810_150809_S2B.png", "feature_index": 0},
    {"lake_id": 32, "date": "20240811", "preview_name": "overlay_32_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 32, "date": "20240811", "preview_name": "overlay_32_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 32, "date": "20240812", "preview_name": "overlay_32_20240812_145941_S2A.png", "feature_index": 0},
    {"lake_id": 32, "date": "20240814", "preview_name": "overlay_32_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 32, "date": "20240813", "preview_name": "overlay_32_20240813_142741_S2A.png", "feature_index": 0},
    {"lake_id": 46, "date": "20240814", "preview_name": "overlay_46_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 46, "date": "20240811", "preview_name": "overlay_46_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 46, "date": "20240811", "preview_name": "overlay_46_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 46, "date": "20240812", "preview_name": "overlay_46_20240812_145941_S2A.png", "feature_index": 0},
    {"lake_id": 69, "date": "20240811", "preview_name": "overlay_69_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 69, "date": "20240811", "preview_name": "overlay_69_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 69, "date": "20240812", "preview_name": "overlay_69_20240812_145941_S2A.png", "feature_index": 0},
    {"lake_id": 98, "date": "20240814", "preview_name": "overlay_98_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 98, "date": "20240811", "preview_name": "overlay_98_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 98, "date": "20240811", "preview_name": "overlay_98_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 99, "date": "20240814", "preview_name": "overlay_99_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 99, "date": "20240811", "preview_name": "overlay_99_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 99, "date": "20240811", "preview_name": "overlay_99_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 99, "date": "20240812", "preview_name": "overlay_99_20240812_145941_S2A.png", "feature_index": 0},
    {"lake_id": 101, "date": "20240814", "preview_name": "overlay_101_20240814_144759_S2B.png", "feature_index": 0},
    {"lake_id": 101, "date": "20240811", "preview_name": "overlay_101_20240811_143749_S2B.png", "feature_index": 0},
    {"lake_id": 101, "date": "20240811", "preview_name": "overlay_101_20240811_152941_S2A.png", "feature_index": 0},
    {"lake_id": 101, "date": "20240812", "preview_name": "overlay_101_20240812_145941_S2A.png", "feature_index": 0},
]

MANUAL_AREA_NOISE_LOOKUP = defaultdict(set)
for _rule in MANUAL_AREA_NOISE_RULES:
    try:
        _lake_id = int(_rule.get("lake_id"))
        _date = str(_rule.get("date", "")).strip()
        _preview = str(_rule.get("preview_name", "")).strip()
        _feat_idx = int(_rule.get("feature_index"))
    except Exception:
        continue
    if not _date or not _preview:
        continue
    MANUAL_AREA_NOISE_LOOKUP[_lake_id].add((_date, _preview, _feat_idx))

METHOD_NAMES = ["Otsu"]
TARGET_SIZE_CATEGORIES = ["Large", "Medium"]
TARGET_LAKE_IDS = []
MAX_LAKE_ID = 0
BRANCH_START_MIN_AREA_KM2 = 0.04
MAX_INSTANT_SPLIT_MERGE_RUN_LEN = 3
INCLUDE_SHARED_TRUNK = False

PURE_BLUE = "#2F6DF6"
BOUNDARY_VALID = "鏈夋晥杈圭晫"
PURE_SWOT_X_END = pd.Timestamp("2024-08-15")
PURE_SWOT_X_START = pd.Timestamp("2024-06-22")


def get_method_paths(method_name):
    input_dir = os.path.join(INPUT_05_1_DIR, method_name)
    output_dir = os.path.join(OUTPUT_DIR, method_name)
    return {
        "method_name": method_name,
        "input_features_csv": os.path.join(input_dir, "05_1_raw_plot_features.csv"),
        "input_topology_gdb": os.path.join(input_dir, "05_1_topology.gdb"),
        "input_04_3_index_csv": os.path.join(INPUT_04_3_DIR, method_name, "04_3_effective_boundary_index.csv"),
        "output_dir": output_dir,
        "single_branch_plot_dir": os.path.join(output_dir, "05_2_EffectiveAreaPlots_PureSWOTBlue_SingleBranch"),
        "single_branch_plot_noise_x_dir": os.path.join(output_dir, "05_2_EffectiveAreaPlots_PureSWOTBlue_SingleBranch_NoiseX"),
        "branch_start_jpg_dir": os.path.join(output_dir, "05_2_BranchStartJpgs"),
        "curve_point_jpg_dir": os.path.join(output_dir, "05_2_CurvePointJpgs"),
        "branch_start_gdb": os.path.join(output_dir, "05_2_branch_start_features.gdb"),
        "branch_start_csv": os.path.join(output_dir, "05_2_branch_start_features.csv"),
        "curve_point_csv": os.path.join(output_dir, "05_2_curve_point_jpg_index.csv"),
        "curve_topology_csv": os.path.join(output_dir, "05_2_denoised_curve_topology_relations.csv"),
        "curve_point_feature_gdb_dir": os.path.join(output_dir, "05_2_CurvePointFeatureGDBs"),
    }


def parse_semicolon_ints(value):
    if pd.isna(value):
        return []
    out = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token or token.lower() == "nan":
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def parse_semicolon_node_keys(value):
    if pd.isna(value):
        return []
    out = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token or token.lower() == "nan":
            continue
        parts = token.split("|")
        if len(parts) != 3:
            continue
        date_text = str(parts[0]).strip()
        preview_text = str(parts[1]).strip()
        try:
            feat_idx = int(parts[2])
        except Exception:
            continue
        out.append((date_text, preview_text, feat_idx))
    return out


def sanitize_filename(text):
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", str(text)).strip("_")


def extract_scene_time_text(preview_name):
    text = str(preview_name).strip()
    match = re.search(r"_(\d{8})_(\d{6})_", text)
    if match:
        return str(match.group(2))
    match = re.search(r"_(\d{6})(?:\D|$)", text)
    if match:
        return str(match.group(1))
    return ""


def build_branch_start_layer_name(row):
    lake_id_val = pd.to_numeric(row.get("lake_id"), errors="coerce")
    branch_order_val = pd.to_numeric(row.get("branch_order"), errors="coerce")
    lake_id = int(lake_id_val) if pd.notna(lake_id_val) else 0
    branch_order = int(branch_order_val) if pd.notna(branch_order_val) else 0
    date_text = str(row.get("date", "")).strip()
    time_text = extract_scene_time_text(row.get("preview_name", ""))
    base = f"lake_{lake_id}_{branch_order}_{date_text}"
    if time_text:
        base = f"{base}_{time_text}"
    return sanitize_filename(base)[:60]


def build_curve_point_layer_name(row):
    lake_id_val = pd.to_numeric(row.get("lake_id"), errors="coerce")
    branch_order_val = pd.to_numeric(row.get("branch_order"), errors="coerce")
    feature_index_val = pd.to_numeric(row.get("feature_index"), errors="coerce")
    lake_id = int(lake_id_val) if pd.notna(lake_id_val) else 0
    branch_order = int(branch_order_val) if pd.notna(branch_order_val) else 0
    feature_index = int(feature_index_val) if pd.notna(feature_index_val) else 0
    date_text = str(row.get("date", "")).strip()
    time_text = extract_scene_time_text(row.get("preview_name", ""))
    base = f"lake_{lake_id}_{branch_order}_{date_text}"
    if time_text:
        base = f"{base}_{time_text}"
    base = f"{base}_f{feature_index}"
    return sanitize_filename(base)[:60]


def get_node_key(row):
    return (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))


def apply_manual_area_noise_filter(branch_df, lake_id):
    if branch_df.empty:
        return branch_df.copy(), branch_df.iloc[0:0].copy()
    lake_rules = MANUAL_AREA_NOISE_LOOKUP.get(int(lake_id), set())
    if not lake_rules:
        return branch_df.copy(), branch_df.iloc[0:0].copy()
    drop_mask = branch_df.apply(lambda row: get_node_key(row) in lake_rules, axis=1)
    if not bool(drop_mask.any()):
        return branch_df.copy(), branch_df.iloc[0:0].copy()
    kept_df = branch_df.loc[~drop_mask].copy()
    removed_df = branch_df.loc[drop_mask].copy()
    return kept_df, removed_df


def add_plot_time_offsets(df, offset_hours=3.0):
    if df.empty:
        return df.copy()
    work = df.copy()
    work["date_dt"] = pd.to_datetime(work["date_dt"], errors="coerce")
    work["x_plot_dt"] = work["date_dt"]
    normalized_dates = work["date_dt"].dt.normalize()
    for _, day_index in work.groupby(normalized_dates, sort=False).groups.items():
        day_df = work.loc[list(day_index)].sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        count = len(day_df)
        if count <= 1:
            continue
        center = (count - 1) / 2.0
        offsets = [(i - center) * offset_hours for i in range(count)]
        shifted = [pd.to_datetime(dt) + pd.Timedelta(hours=h) for dt, h in zip(day_df["date_dt"], offsets)]
        work.loc[day_df.index, "x_plot_dt"] = shifted
    return work


def load_features_df(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df
    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date", "preview_name", "feature_index"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(work["date"].astype(str))
    work["preview_name"] = work["preview_name"].fillna("").astype(str)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    work["feature_index"] = pd.to_numeric(work["feature_index"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["feature_index"]).copy()
    work["feature_index"] = work["feature_index"].astype(int)
    work["area_km2"] = pd.to_numeric(work["area_km2"], errors="coerce")
    work["display_status"] = work.get("display_status", "").fillna("").astype(str)
    work["branch_id"] = work.get("branch_id", "").fillna("").astype(str)
    if "size_category_04_8" in work.columns:
        work["size_category"] = work["size_category_04_8"].fillna("").astype(str)
    elif "size_category" in work.columns:
        work["size_category"] = work["size_category"].fillna("").astype(str)
    else:
        work["size_category"] = ""
    return work


def apply_common_filters(df):
    if df.empty:
        return df
    work = df.copy()
    if TARGET_SIZE_CATEGORIES:
        work = work[work["size_category"].isin(TARGET_SIZE_CATEGORIES)].copy()
    if TARGET_LAKE_IDS:
        lake_id_set = {int(v) for v in TARGET_LAKE_IDS}
        work = work[work["lake_id"].isin(lake_id_set)].copy()
    if MAX_LAKE_ID and MAX_LAKE_ID > 0:
        work = work[work["lake_id"] <= int(MAX_LAKE_ID)].copy()
    return work


def load_04_3_jpg_map(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return {}
    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date", "preview_name"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(work["date"].astype(str))
    work["preview_name"] = work["preview_name"].fillna("").astype(str)
    work["jpg_path"] = work.get("jpg_path", "").fillna("").astype(str)
    out = {}
    for _, row in work.iterrows():
        key = (int(row["lake_id"]), str(row["date"]), str(row["preview_name"]))
        out[key] = str(row["jpg_path"])
    return out


def build_removed_parent_bridge_edges(source_df, keep_key_set, max_hops=8):
    if source_df.empty or not keep_key_set:
        return []
    work = source_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    node_lookup = {get_node_key(row): row for _, row in work.iterrows()}
    scenes = (
        work[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in scenes.iterrows()]
    parent_scene_lookup = {scene_keys[i]: scene_keys[i - 1] for i in range(1, len(scene_keys))}

    def get_parent_keys(node_key):
        row = node_lookup.get(node_key)
        if row is None:
            return []
        curr_scene = (node_key[0], node_key[1])
        prev_scene = parent_scene_lookup.get(curr_scene)
        out = []
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                if parent_key in node_lookup:
                    out.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in node_lookup:
                out.append(parent_key)
        seen = set()
        deduped = []
        for key in out:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    edge_set = set()
    for child_key in sorted(keep_key_set, key=lambda key: (key[0], key[1], key[2])):
        direct_parents = get_parent_keys(child_key)
        if not direct_parents:
            continue
        if any(parent_key in keep_key_set for parent_key in direct_parents):
            continue

        frontier = [(parent_key, 1) for parent_key in direct_parents]
        seen = {parent_key for parent_key in direct_parents}
        found_parent = None
        while frontier:
            curr_key, hops = frontier.pop(0)
            if curr_key in keep_key_set:
                found_parent = curr_key
                break
            if hops >= int(max_hops):
                continue
            for parent_key in get_parent_keys(curr_key):
                if parent_key in seen:
                    continue
                seen.add(parent_key)
                frontier.append((parent_key, hops + 1))
        if found_parent is not None:
            edge_set.add((found_parent, child_key))
    return list(edge_set)


def get_curve_topology_edges(branch_df, source_df=None):
    if branch_df.empty:
        return []
    node_lookup = {get_node_key(row): row for _, row in branch_df.iterrows()}
    scenes = (
        branch_df[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in scenes.iterrows()]
    parent_scene_lookup = {scene_keys[i]: scene_keys[i - 1] for i in range(1, len(scene_keys))}
    edges = []
    edge_seen = set()

    for _, row in branch_df.iterrows():
        child_key = get_node_key(row)
        curr_scene = (child_key[0], child_key[1])
        prev_scene = parent_scene_lookup.get(curr_scene)
        if prev_scene is None:
            parent_keys = []
        else:
            parent_keys = []
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_keys.append((prev_scene[0], prev_scene[1], int(parent_idx)))
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            parent_keys.append(parent_key)
        for parent_key in parent_keys:
            if parent_key not in node_lookup:
                continue
            edge_id = (parent_key, child_key)
            if edge_id in edge_seen:
                continue
            edge_seen.add(edge_id)
            edges.append(
                {
                    "parent_key": parent_key,
                    "child_key": child_key,
                    "edge_type": "curve_parent",
                }
            )
    if source_df is not None and not source_df.empty:
        keep_key_set = set(node_lookup.keys())
        for parent_key, child_key in build_removed_parent_bridge_edges(source_df, keep_key_set):
            if parent_key not in node_lookup or child_key not in node_lookup:
                continue
            edge_id = (parent_key, child_key)
            if edge_id in edge_seen:
                continue
            edge_seen.add(edge_id)
            edges.append(
                {
                    "parent_key": parent_key,
                    "child_key": child_key,
                    "edge_type": "curve_bridge",
                }
            )
    return edges


def draw_parent_edges(ax, branch_df, source_df=None):
    if branch_df.empty:
        return
    node_lookup = {get_node_key(row): row for _, row in branch_df.iterrows()}
    for edge in get_curve_topology_edges(branch_df, source_df=source_df):
        parent_row = node_lookup.get(edge["parent_key"])
        child_row = node_lookup.get(edge["child_key"])
        if parent_row is None or child_row is None:
            continue
        ax.plot(
            [parent_row.get("x_plot_dt", parent_row["date_dt"]), child_row.get("x_plot_dt", child_row["date_dt"])],
            [parent_row["area_km2"], child_row["area_km2"]],
            color=PURE_BLUE,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
        )


def build_backfill_visual_edges(features_df, keep_key_set, max_backward_scene_lookback=2, min_intersection_ratio=0.01):
    if features_df.empty or not keep_key_set:
        return []
    if gpd is None:
        return []
    work = features_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    node_lookup = {get_node_key(row): row for _, row in work.iterrows()}
    ordered_scenes = (
        work[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    scene_to_index = {scene_key: idx for idx, scene_key in enumerate(scene_keys)}
    scene_node_keys = defaultdict(list)
    for key in keep_key_set:
        scene_node_keys[(key[0], key[1])].append(key)

    edge_set = set()
    for child_key in sorted(keep_key_set, key=lambda key: (key[0], key[1], key[2])):
        child_row = node_lookup.get(child_key)
        if child_row is None:
            continue
        if int(pd.to_numeric(child_row.get("parent_count", 0), errors="coerce")) > 0:
            continue
        if parse_semicolon_node_keys(child_row.get("bridge_parent_keys", "")):
            continue

        child_scene = (child_key[0], child_key[1])
        child_scene_idx = scene_to_index.get(child_scene)
        if child_scene_idx is None:
            continue
        child_geom = child_row.get("geometry", None)
        if child_geom is None or child_geom.is_empty:
            continue

        for back in range(2, int(max_backward_scene_lookback) + 2):
            prev_scene_idx = child_scene_idx - back
            if prev_scene_idx < 0:
                break
            prev_scene = scene_keys[prev_scene_idx]
            candidates = []
            for parent_key in scene_node_keys.get(prev_scene, []):
                parent_row = node_lookup.get(parent_key)
                if parent_row is None:
                    continue
                parent_geom = parent_row.get("geometry", None)
                if parent_geom is None or parent_geom.is_empty:
                    continue
                inter_area = child_geom.intersection(parent_geom).area
                if inter_area <= 0:
                    continue
                overlap_ratio_curr = inter_area / max(child_geom.area, 1e-12)
                overlap_ratio_prev = inter_area / max(parent_geom.area, 1e-12)
                if max(overlap_ratio_curr, overlap_ratio_prev) >= min_intersection_ratio:
                    candidates.append((parent_key, inter_area))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[1], reverse=True)
            edge_set.add((candidates[0][0], child_key))
            break
    return list(edge_set)


def build_child_map(features_df):
    node_lookup = {get_node_key(row): row for _, row in features_df.iterrows()}
    child_map = defaultdict(list)
    ordered_scenes = (
        features_df[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    parent_scene_lookup = {scene_keys[i]: scene_keys[i - 1] for i in range(1, len(scene_keys))}
    for _, row in features_df.iterrows():
        child_key = get_node_key(row)
        curr_scene = (child_key[0], child_key[1])
        prev_scene = parent_scene_lookup.get(curr_scene)
        parent_keys = []
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                if parent_key in node_lookup:
                    parent_keys.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in node_lookup:
                parent_keys.append(parent_key)
        for parent_key in parent_keys:
            child_map[parent_key].append(child_key)
    return dict(child_map), node_lookup, parent_scene_lookup


def build_shared_merge_branch_subset(plot_features, branch_id):
    branch_text = str(branch_id).strip()
    subset = plot_features[plot_features["branch_id"].astype(str).str.strip() == branch_text].copy()
    if subset.empty:
        return subset
    selected_keys = {get_node_key(row) for _, row in subset.iterrows()}
    kept_key_set = {get_node_key(row) for _, row in plot_features.iterrows()}
    child_map, node_lookup, parent_scene_lookup = build_child_map(plot_features)

    def get_kept_parent_keys(node_key):
        row = node_lookup.get(node_key)
        if row is None:
            return []
        out = []
        curr_scene = (node_key[0], node_key[1])
        prev_scene = parent_scene_lookup.get(curr_scene)
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                if parent_key in node_lookup and parent_key in kept_key_set:
                    out.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in node_lookup and parent_key in kept_key_set:
                out.append(parent_key)
        seen = set()
        deduped = []
        for key in out:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    split_entry_keys = set()
    for node_key in selected_keys:
        for parent_key in get_kept_parent_keys(node_key):
            if parent_key in selected_keys:
                continue
            kept_child_keys = [child_key for child_key in child_map.get(parent_key, []) if child_key in kept_key_set]
            if len(kept_child_keys) >= 2:
                split_entry_keys.add(parent_key)

    shared_upstream_keys = set()
    stack = list(split_entry_keys)
    while stack:
        curr_key = stack.pop()
        if curr_key in shared_upstream_keys:
            continue
        shared_upstream_keys.add(curr_key)
        kept_parent_keys = get_kept_parent_keys(curr_key)
        if len(kept_parent_keys) != 1:
            continue
        next_key = kept_parent_keys[0]
        if next_key not in shared_upstream_keys:
            stack.append(next_key)

    merge_entry_keys = set()
    for node_key, row in node_lookup.items():
        if node_key in selected_keys:
            continue
        if node_key not in kept_key_set:
            continue
        curr_scene = (node_key[0], node_key[1])
        prev_scene = parent_scene_lookup.get(curr_scene)
        if prev_scene is None:
            continue
        parent_keys = []
        for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
            parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
            if parent_key in node_lookup:
                parent_keys.append(parent_key)
        if any(parent_key in selected_keys for parent_key in parent_keys):
            merge_entry_keys.add(node_key)

    shared_downstream_keys = set()
    stack = list(merge_entry_keys)
    while stack:
        curr_key = stack.pop()
        if curr_key in shared_downstream_keys:
            continue
        shared_downstream_keys.add(curr_key)
        child_keys = [child_key for child_key in child_map.get(curr_key, []) if child_key in kept_key_set]
        if len(child_keys) != 1:
            continue
        next_key = child_keys[0]
        if next_key not in shared_downstream_keys:
            stack.append(next_key)

    display_keys = selected_keys | shared_upstream_keys | shared_downstream_keys
    out = plot_features[
        plot_features.apply(lambda row: get_node_key(row) in display_keys, axis=1)
    ].copy()
    if out.empty:
        return out

    scene_counts = defaultdict(int)
    scene_has_selected = defaultdict(bool)
    for key in display_keys:
        scene_counts[(key[0], key[1])] += 1
        if key in selected_keys:
            scene_has_selected[(key[0], key[1])] = True
    multi_scene_selected = {
        scene_key
        for scene_key, count in scene_counts.items()
        if count >= 2 and scene_has_selected.get(scene_key, False)
    }
    if multi_scene_selected:
        out = out[
            out.apply(
                lambda row: (
                    (str(row["date"]), str(row["preview_name"])) not in multi_scene_selected
                    or get_node_key(row) in selected_keys
                ),
                axis=1,
            )
        ].copy()
    return out.sort_values(["date_dt", "preview_name", "feature_index"]).copy()


def expand_selected_with_shared_trunk(plot_features, selected_df):
    if plot_features.empty or selected_df.empty:
        return selected_df.copy()
    selected_keys = {get_node_key(row) for _, row in selected_df.iterrows()}
    kept_key_set = {get_node_key(row) for _, row in plot_features.iterrows()}
    child_map, node_lookup, parent_scene_lookup = build_child_map(plot_features)

    merge_entry_keys = set()
    for node_key, row in node_lookup.items():
        if node_key in selected_keys:
            continue
        if node_key not in kept_key_set:
            continue
        curr_scene = (node_key[0], node_key[1])
        prev_scene = parent_scene_lookup.get(curr_scene)
        if prev_scene is None:
            continue
        parent_keys = []
        for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
            parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
            if parent_key in node_lookup:
                parent_keys.append(parent_key)
        if any(parent_key in selected_keys for parent_key in parent_keys):
            merge_entry_keys.add(node_key)

    shared_downstream_keys = set()
    stack = list(merge_entry_keys)
    while stack:
        curr_key = stack.pop()
        if curr_key in shared_downstream_keys:
            continue
        shared_downstream_keys.add(curr_key)
        child_keys = [child_key for child_key in child_map.get(curr_key, []) if child_key in kept_key_set]
        if len(child_keys) != 1:
            continue
        next_key = child_keys[0]
        if next_key not in shared_downstream_keys:
            stack.append(next_key)

    display_keys = selected_keys | shared_downstream_keys
    out = plot_features[
        plot_features.apply(lambda row: get_node_key(row) in display_keys, axis=1)
    ].copy()
    return out.sort_values(["date_dt", "preview_name", "feature_index"]).copy()


def format_branch_plot_label(lake_id, branch_order, total_branch_count):
    if int(total_branch_count) <= 1:
        return f"Lake {int(lake_id)}"
    return f"Lake {int(lake_id)}_{int(branch_order)}"


def render_single_branch_plot(lake_id, branch_order, total_branch_count, branch_df, out_png, source_df=None):
    fig, ax = plt.subplots(figsize=(16, 6), dpi=180)
    draw_parent_edges(ax, branch_df, source_df=source_df)
    ax.scatter(
        branch_df["x_plot_dt"],
        branch_df["area_km2"],
        color=PURE_BLUE,
        s=38,
        edgecolors="none",
        alpha=0.95,
        zorder=3,
    )
    peak_idx = pd.to_numeric(branch_df["area_km2"], errors="coerce").idxmax()
    if pd.notna(peak_idx):
        peak_row = branch_df.loc[peak_idx]
        ax.annotate(
            f"Peak feature: {float(peak_row['area_km2']):.3f}",
            xy=(peak_row.get("x_plot_dt", peak_row["date_dt"]), peak_row["area_km2"]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=11,
            color=PURE_BLUE,
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": PURE_BLUE},
        )

    ax.text(
        0.03,
        0.92,
        format_branch_plot_label(lake_id, branch_order, total_branch_count),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=32,
        fontweight="bold",
        color="black",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(PURE_SWOT_X_START, PURE_SWOT_X_END)
    ax.grid(True, color="#BDBDBD", alpha=0.55, linewidth=1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", labelrotation=22, labelsize=30, pad=34)
    ax.tick_params(axis="y", labelsize=32, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#101010")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def render_single_branch_plot_with_noise_x(lake_id, branch_order, total_branch_count, branch_df, removed_noise_df, out_png, source_df=None):
    fig, ax = plt.subplots(figsize=(16, 6), dpi=180)
    draw_parent_edges(ax, branch_df, source_df=source_df)
    ax.scatter(
        branch_df["x_plot_dt"],
        branch_df["area_km2"],
        color=PURE_BLUE,
        s=38,
        edgecolors="none",
        alpha=0.95,
        zorder=3,
    )
    if removed_noise_df is not None and not removed_noise_df.empty:
        noise_work = removed_noise_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        ax.scatter(
            noise_work["x_plot_dt"],
            noise_work["area_km2"],
            marker="x",
            color="#E53935",
            s=58,
            linewidths=1.4,
            alpha=0.95,
            zorder=4,
        )

    peak_idx = pd.to_numeric(branch_df["area_km2"], errors="coerce").idxmax()
    if pd.notna(peak_idx):
        peak_row = branch_df.loc[peak_idx]
        ax.annotate(
            f"Peak feature: {float(peak_row['area_km2']):.3f}",
            xy=(peak_row.get("x_plot_dt", peak_row["date_dt"]), peak_row["area_km2"]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=11,
            color=PURE_BLUE,
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": PURE_BLUE},
        )

    ax.text(
        0.03,
        0.92,
        format_branch_plot_label(lake_id, branch_order, total_branch_count),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=32,
        fontweight="bold",
        color="black",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(PURE_SWOT_X_START, PURE_SWOT_X_END)
    ax.grid(True, color="#BDBDBD", alpha=0.55, linewidth=1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", labelrotation=22, labelsize=30, pad=34)
    ax.tick_params(axis="y", labelsize=32, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#101010")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def downgrade_between_neighbor_extrema(features_df, local_max_keys, local_min_keys, size_category=None):
    if features_df.empty:
        return local_max_keys, local_min_keys, set()
    tolerance = get_size_tolerance(size_category)
    keep_max = set(local_max_keys)
    keep_min = set(local_min_keys)
    restored_extrema_keys = set()
    for _, branch_df in features_df.groupby("branch_id", dropna=False, sort=False):
        branch_df = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        if len(branch_df) < 3:
            continue
        seq_rows = list(branch_df.itertuples(index=False))
        seq_keys = []
        seq_areas = []
        for row in seq_rows:
            key = (str(getattr(row, "date")), str(getattr(row, "preview_name")), int(getattr(row, "feature_index")))
            area_val = pd.to_numeric(getattr(row, "area_km2"), errors="coerce")
            seq_keys.append(key)
            seq_areas.append(area_val)
        key_to_pos = {key: idx for idx, key in enumerate(seq_keys)}

        for candidate_set in (keep_max, keep_min):
            for node_key in list(candidate_set):
                pos = key_to_pos.get(node_key)
                if pos is None or pos <= 0 or pos >= len(seq_keys) - 1:
                    continue
                left_area = seq_areas[pos - 1]
                curr_area = seq_areas[pos]
                right_area = seq_areas[pos + 1]
                if pd.isna(left_area) or pd.isna(curr_area) or pd.isna(right_area):
                    continue
                lower = min(float(left_area), float(right_area)) - tolerance
                upper = max(float(left_area), float(right_area)) + tolerance
                if lower <= float(curr_area) <= upper:
                    candidate_set.discard(node_key)
                    if tolerance > 0:
                        restored_extrema_keys.add(node_key)
    return keep_max, keep_min, restored_extrema_keys


def get_size_tolerance(size_category=None):
    size_text = str(size_category).strip()
    if size_text == "Large":
        tolerance = 0.2
    elif size_text == "Medium":
        tolerance = 0.05
    elif size_text == "Small":
        tolerance = 0.02
    else:
        tolerance = 0.0
    return float(tolerance)


def detect_short_consecutive_extrema_keys(branch_df, extrema_key_set, max_run_len=3):
    # Pin only short runs of same-direction extrema by consecutive dates.
    # Mixed zig-zag runs like max-min should still be eligible for tolerance rollback.
    if branch_df.empty or not extrema_key_set:
        return set()
    work = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    out = set()
    day_records = []
    for date_text, day_df in work.groupby(work["date"].astype(str), sort=False):
        day_keys = [get_node_key(row) for _, row in day_df.iterrows()]
        has_extrema = any(key in extrema_key_set for key in day_keys)
        day_records.append((str(date_text), has_extrema, day_keys))

    start = None
    for i, (_, has_extrema, _) in enumerate(day_records + [("__end__", False, [])]):
        if has_extrema and start is None:
            start = i
            continue
        if (not has_extrema) and start is not None:
            run_len = i - start
            if 2 <= run_len <= int(max_run_len):
                for _, _, run_day_keys in day_records[start:i]:
                    out.update([key for key in run_day_keys if key in extrema_key_set])
            start = None
    return out


def detect_short_depressed_run_keys(branch_df, size_category=None, min_run_len=2, max_run_len=3):
    if branch_df.empty or len(branch_df) < int(min_run_len) + 2:
        return set()
    work = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    seq_keys = [get_node_key(row) for _, row in work.iterrows()]
    seq_areas = [pd.to_numeric(row["area_km2"], errors="coerce") for _, row in work.iterrows()]
    tolerance = get_size_tolerance(size_category)
    out = set()
    n = len(seq_keys)
    for run_len in range(int(min_run_len), int(max_run_len) + 1):
        for start in range(1, n - run_len):
            end = start + run_len - 1
            if end >= n - 1:
                continue
            left_area = seq_areas[start - 1]
            right_area = seq_areas[end + 1]
            if pd.isna(left_area) or pd.isna(right_area):
                continue
            boundary_floor = min(float(left_area), float(right_area)) - tolerance
            run_areas = seq_areas[start:end + 1]
            if any(pd.isna(v) for v in run_areas):
                continue
            if all(float(v) < boundary_floor for v in run_areas):
                out.update(seq_keys[start:end + 1])
    return out


def protect_true_global_max_keys(features_df, local_max_keys):
    if features_df.empty or not local_max_keys:
        return set()
    work = features_df.sort_values(["branch_id", "date_dt", "preview_name", "feature_index"]).copy()
    protected_keys = set()
    for _, branch_df in work.groupby("branch_id", dropna=False, sort=False):
        branch_df = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy().reset_index(drop=True)
        if len(branch_df) < 3:
            continue
        seq_keys = [get_node_key(row) for _, row in branch_df.iterrows()]
        seq_areas = [pd.to_numeric(row["area_km2"], errors="coerce") for _, row in branch_df.iterrows()]
        valid_areas = [float(v) for v in seq_areas if pd.notna(v)]
        if not valid_areas:
            continue
        branch_global_max = max(valid_areas)
        key_to_pos = {key: idx for idx, key in enumerate(seq_keys)}
        for node_key in [key for key in local_max_keys if key in key_to_pos]:
            pos = key_to_pos[node_key]
            curr_area = seq_areas[pos]
            if pd.isna(curr_area) or float(curr_area) != float(branch_global_max):
                continue
            if pos <= 0 or pos >= len(seq_keys) - 1:
                continue
            left_area = seq_areas[pos - 1]
            right_area = seq_areas[pos + 1]
            kept_areas = [float(v) for i, v in enumerate(seq_areas) if i != pos and pd.notna(v)]
            if not kept_areas:
                continue
            new_global_max = max(kept_areas)
            if (
                (pd.notna(left_area) and float(left_area) == float(new_global_max))
                or (pd.notna(right_area) and float(right_area) == float(new_global_max))
            ):
                protected_keys.add(node_key)
    return protected_keys


def classify_final_curve_size_category(max_area_km2):
    max_area = pd.to_numeric(max_area_km2, errors="coerce")
    if pd.isna(max_area):
        return "Small"
    if float(max_area) > 0.625:
        return "Large"
    if float(max_area) > 0.0625:
        return "Medium"
    return "Small"


def identify_local_extrema_points(features_df, peak_date=None, size_category=None):
    if features_df.empty:
        return set(), set(), set()
    work = features_df.sort_values(["branch_id", "date_dt", "preview_name", "feature_index"]).copy()
    local_max_keys = set()
    local_min_keys = set()
    for _, branch_df in work.groupby("branch_id", dropna=False, sort=False):
        branch_df = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        if len(branch_df) < 3:
            continue

        rows_seq = list(branch_df.itertuples(index=False))
        for pos in range(1, len(rows_seq) - 1):
            prev_row = rows_seq[pos - 1]
            curr_row = rows_seq[pos]
            next_row = rows_seq[pos + 1]
            prev_area = pd.to_numeric(getattr(prev_row, "area_km2"), errors="coerce")
            curr_area = pd.to_numeric(getattr(curr_row, "area_km2"), errors="coerce")
            next_area = pd.to_numeric(getattr(next_row, "area_km2"), errors="coerce")
            if pd.isna(prev_area) or pd.isna(curr_area) or pd.isna(next_area):
                continue
            curr_key = (
                str(getattr(curr_row, "date")),
                str(getattr(curr_row, "preview_name")),
                int(getattr(curr_row, "feature_index")),
            )
            if float(curr_area) > float(prev_area) and float(curr_area) > float(next_area):
                local_max_keys.add(curr_key)
            elif float(curr_area) < float(prev_area) and float(curr_area) < float(next_area):
                local_min_keys.add(curr_key)

    protected_global_max_keys = protect_true_global_max_keys(work, local_max_keys)
    if protected_global_max_keys:
        local_max_keys = {key for key in local_max_keys if key not in protected_global_max_keys}

    keep_max, keep_min, restored_extrema_keys = downgrade_between_neighbor_extrema(
        work,
        local_max_keys,
        local_min_keys,
        size_category=size_category,
    )
    # Tolerance rollback is part of extrema screening: points that already fall
    # within the neighbor envelope are treated as normal fluctuation and should
    # not enter later denoise rules.
    pinned_extrema_keys = set()
    depressed_run_keys = set()
    for _, branch_df in work.groupby("branch_id", dropna=False, sort=False):
        branch_df = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        branch_key_set = {get_node_key(row) for _, row in branch_df.iterrows()}
        branch_max_candidates = set(keep_max) & branch_key_set
        branch_min_candidates = set(keep_min) & branch_key_set
        pinned_extrema_keys.update(
            detect_short_consecutive_extrema_keys(branch_df, branch_max_candidates, max_run_len=3)
        )
        pinned_extrema_keys.update(
            detect_short_consecutive_extrema_keys(branch_df, branch_min_candidates, max_run_len=3)
        )
        depressed_run_keys.update(
            detect_short_depressed_run_keys(branch_df, size_category=size_category, min_run_len=2, max_run_len=3)
        )
    if pinned_extrema_keys:
        keep_max.update({key for key in pinned_extrema_keys if key in keep_max})
        keep_min.update({key for key in pinned_extrema_keys if key in keep_min})
        restored_extrema_keys = {key for key in restored_extrema_keys if key not in pinned_extrema_keys}
    if depressed_run_keys:
        keep_min.update(depressed_run_keys)
        restored_extrema_keys = {key for key in restored_extrema_keys if key not in depressed_run_keys}
    return keep_max, keep_min, restored_extrema_keys


def trim_endpoint_peak_points(df):
    if df.empty:
        return df.copy(), df.iloc[0:0].copy(), {"left": 0, "right": 0}
    work = df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    removed_chunks = []
    removed_counts = {"left": 0, "right": 0}
    while len(work) >= 2:
        changed = False
        first_area = pd.to_numeric(work.iloc[0]["area_km2"], errors="coerce")
        second_area = pd.to_numeric(work.iloc[1]["area_km2"], errors="coerce")
        if pd.notna(first_area) and pd.notna(second_area) and float(first_area) > float(second_area):
            removed_chunks.append(work.iloc[[0]].copy())
            removed_counts["left"] += 1
            work = work.iloc[1:].copy()
            changed = True
        if changed:
            continue
        last_area = pd.to_numeric(work.iloc[-1]["area_km2"], errors="coerce")
        prev_area = pd.to_numeric(work.iloc[-2]["area_km2"], errors="coerce")
        if pd.notna(last_area) and pd.notna(prev_area) and float(last_area) > float(prev_area):
            removed_chunks.append(work.iloc[[-1]].copy())
            removed_counts["right"] += 1
            work = work.iloc[:-1].copy()
            changed = True
        if not changed:
            break
    removed_df = pd.concat(removed_chunks, ignore_index=True) if removed_chunks else work.iloc[0:0].copy()
    return work.reset_index(drop=True), removed_df, removed_counts


def trim_iterative_extrema_points(source_features_df):
    if source_features_df.empty or len(source_features_df) < 3:
        return source_features_df.copy(), source_features_df.iloc[0:0].copy()
    work = source_features_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy().reset_index(drop=True)
    removed_chunks = []
    while len(work) >= 3:
        max_area = pd.to_numeric(work["area_km2"], errors="coerce").max()
        size_category = classify_final_curve_size_category(max_area)
        peak_idx = work["area_km2"].idxmax()
        peak_row = work.loc[peak_idx]
        local_max_keys, local_min_keys, _ = identify_local_extrema_points(
            work,
            peak_date=peak_row["date"],
            size_category=size_category,
        )
        remove_keys = set(local_max_keys) | set(local_min_keys)
        if not remove_keys:
            break
        remove_mask = work.apply(lambda row: get_node_key(row) in remove_keys, axis=1)
        if not remove_mask.any():
            break
        removed_chunks.append(work.loc[remove_mask].copy())
        work = work.loc[~remove_mask].copy().reset_index(drop=True)
    removed_df = pd.concat(removed_chunks, ignore_index=True) if removed_chunks else source_features_df.iloc[0:0].copy()
    return work.reset_index(drop=True), removed_df


def trim_single_global_max_point(df):
    if df.empty:
        return df.copy(), df.iloc[0:0].copy()
    work = df.sort_values(["date_dt", "preview_name", "feature_index"]).copy().reset_index(drop=True)
    area_series = pd.to_numeric(work["area_km2"], errors="coerce")
    if not area_series.notna().any():
        return work, work.iloc[0:0].copy()
    peak_pos = int(area_series.idxmax())
    removed_df = work.iloc[[peak_pos]].copy().reset_index(drop=True)
    kept_df = work.drop(index=peak_pos).reset_index(drop=True)
    return kept_df, removed_df


def revive_between_effective_neighbors(source_df, kept_df):
    if source_df.empty or kept_df.empty:
        return kept_df.copy(), source_df.iloc[0:0].copy()
    source_work = source_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy().reset_index(drop=True)
    kept_key_set = {get_node_key(row) for _, row in kept_df.iterrows()}

    changed = True
    while changed:
        changed = False
        active_flags = [get_node_key(row) in kept_key_set for _, row in source_work.iterrows()]
        for pos in range(1, len(source_work) - 1):
            row = source_work.iloc[pos]
            row_key = get_node_key(row)
            if row_key in kept_key_set:
                continue

            left_pos = None
            for i in range(pos - 1, -1, -1):
                if active_flags[i]:
                    left_pos = i
                    break
            if left_pos is None:
                continue

            right_pos = None
            for i in range(pos + 1, len(source_work)):
                if active_flags[i]:
                    right_pos = i
                    break
            if right_pos is None:
                continue

            left_area = pd.to_numeric(source_work.iloc[left_pos]["area_km2"], errors="coerce")
            curr_area = pd.to_numeric(row["area_km2"], errors="coerce")
            right_area = pd.to_numeric(source_work.iloc[right_pos]["area_km2"], errors="coerce")
            if pd.isna(left_area) or pd.isna(curr_area) or pd.isna(right_area):
                continue

            lower = min(float(left_area), float(right_area))
            upper = max(float(left_area), float(right_area))
            if lower <= float(curr_area) <= upper:
                kept_key_set.add(row_key)
                changed = True

    revived_df = source_work[
        source_work.apply(lambda row: get_node_key(row) in kept_key_set, axis=1)
    ].copy().reset_index(drop=True)
    removed_df = source_work[
        ~source_work.apply(lambda row: get_node_key(row) in kept_key_set, axis=1)
    ].copy().reset_index(drop=True)
    return revived_df, removed_df


def copy_jpg_if_exists(src_path, dst_path):
    if not src_path:
        return False
    if not os.path.exists(src_path):
        return False
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def load_topology_features_gdf(gdb_path):
    if gpd is None or not os.path.exists(gdb_path):
        return None
    gdf = gpd.read_file(gdb_path, layer="topology_features")
    if gdf.empty:
        return gdf
    work = gdf.copy()
    if "date" not in work.columns and "date_" in work.columns:
        work = work.rename(columns={"date_": "date"})
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work["feature_index"] = pd.to_numeric(work.get("feature_index", np.nan), errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date", "preview_name", "feature_index"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["feature_index"] = work["feature_index"].astype(int)
    work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(work["date"].astype(str))
    work["preview_name"] = work["preview_name"].fillna("").astype(str)
    return work


def export_branch_start_gdb(branch_start_df, topo_gdf, out_gdb):
    if gpd is None or topo_gdf is None or topo_gdf.empty or branch_start_df.empty:
        return
    topo_work = topo_gdf.copy()
    topo_work["_join_key"] = (
        topo_work["lake_id"].astype(int).astype(str)
        + "|"
        + topo_work["date"].astype(str)
        + "|"
        + topo_work["preview_name"].astype(str)
        + "|"
        + topo_work["feature_index"].astype(int).astype(str)
    )
    start_work = branch_start_df.copy()
    start_work["_join_key"] = (
        start_work["lake_id"].astype(int).astype(str)
        + "|"
        + start_work["date"].astype(str)
        + "|"
        + start_work["preview_name"].astype(str)
        + "|"
        + start_work["feature_index"].astype(int).astype(str)
    )
    merged = start_work.merge(
        topo_work[["_join_key", "geometry"]],
        on="_join_key",
        how="left",
    )
    merged = merged[merged["geometry"].notna()].copy()
    if merged.empty:
        return
    if os.path.exists(out_gdb):
        shutil.rmtree(out_gdb, ignore_errors=True)
    merged["branch_start_layer"] = merged.apply(build_branch_start_layer_name, axis=1)
    for layer_name, layer_df in merged.groupby("branch_start_layer", sort=False):
        out_gdf = gpd.GeoDataFrame(layer_df.drop(columns=["_join_key"]), geometry="geometry", crs=topo_gdf.crs)
        out_gdf.to_file(out_gdb, layer=str(layer_name), driver="OpenFileGDB")


def export_curve_point_feature_gdbs(curve_point_df, topo_gdf, out_dir):
    if gpd is None or topo_gdf is None or topo_gdf.empty or curve_point_df.empty:
        return
    topo_work = topo_gdf.copy()
    topo_work["_join_key"] = (
        topo_work["lake_id"].astype(int).astype(str)
        + "|"
        + topo_work["date"].astype(str)
        + "|"
        + topo_work["preview_name"].astype(str)
        + "|"
        + topo_work["feature_index"].astype(int).astype(str)
    )
    point_work = curve_point_df.copy()
    point_work["_join_key"] = (
        point_work["lake_id"].astype(int).astype(str)
        + "|"
        + point_work["date"].astype(str)
        + "|"
        + point_work["preview_name"].astype(str)
        + "|"
        + point_work["feature_index"].astype(int).astype(str)
    )
    merged = point_work.merge(
        topo_work[["_join_key", "geometry"]],
        on="_join_key",
        how="left",
    )
    merged = merged[merged["geometry"].notna()].copy()
    if merged.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    merged["curve_point_layer"] = merged.apply(build_curve_point_layer_name, axis=1)
    for lake_id, lake_df in merged.groupby("lake_id", sort=True):
        out_gdb = os.path.join(out_dir, f"lake_{int(lake_id)}.gdb")
        if os.path.exists(out_gdb):
            shutil.rmtree(out_gdb, ignore_errors=True)
        for layer_name, layer_df in lake_df.groupby("curve_point_layer", sort=False):
            out_gdf = gpd.GeoDataFrame(layer_df.drop(columns=["_join_key"]), geometry="geometry", crs=topo_gdf.crs)
            out_gdf.to_file(out_gdb, layer=str(layer_name), driver="OpenFileGDB")


def process_method(method_name):
    paths = get_method_paths(method_name)
    print(f"[05_2_simple] method={method_name}")
    for key in [
        "output_dir",
        "single_branch_plot_dir",
        "single_branch_plot_noise_x_dir",
        "branch_start_jpg_dir",
        "curve_point_jpg_dir",
    ]:
        os.makedirs(paths[key], exist_ok=True)

    features_df = load_features_df(paths["input_features_csv"])
    features_df = apply_common_filters(features_df)
    if features_df.empty:
        print("[05_2_simple] skip: empty features")
        return

    # Use 05_1 denoised result directly.
    kept = features_df[
        (features_df["display_status"] == "kept_display")
        & (features_df["branch_id"].astype(str).str.strip() != "")
    ].copy()
    if kept.empty:
        print("[05_2_simple] skip: no kept_display records")
        return

    kept = kept.sort_values(["lake_id", "date_dt", "preview_name", "feature_index"]).copy()
    kept = add_plot_time_offsets(kept)

    jpg_map = load_04_3_jpg_map(paths["input_04_3_index_csv"])
    topo_gdf = load_topology_features_gdf(paths["input_topology_gdb"])

    branch_start_rows = []
    curve_point_rows = []
    curve_topology_rows = []
    plot_count = 0
    start_jpg_count = 0
    curve_jpg_count = 0

    grouped_lakes = list(kept.groupby("lake_id", sort=True))
    total_lakes = len(grouped_lakes)
    for lake_index, (lake_id, lake_df) in enumerate(grouped_lakes, start=1):
        branch_ids = sorted([v for v in lake_df["branch_id"].dropna().astype(str).unique().tolist() if v.strip()])
        total_branch_count = len(branch_ids)
        for order, branch_id in enumerate(branch_ids, start=1):
            # Directly use the 05_1 branch grouping result; do not expand to
            # shared-merge corridor in simplified 05_2.
            # Align with 05_1_SingleBranch semantics: branch curve includes
            # downstream shared-merge segment.
            branch_source_df = build_shared_merge_branch_subset(lake_df, branch_id)
            if branch_source_df.empty:
                continue
            branch_source_df = branch_source_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
            branch_source_df, manual_removed_df = apply_manual_area_noise_filter(branch_source_df, lake_id)
            if not manual_removed_df.empty:
                print(
                    f"[05_2_simple] manual area-noise removed: lake={int(lake_id)} branch={str(branch_id)} "
                    f"count={len(manual_removed_df)}",
                    flush=True,
                )
            if branch_source_df.empty:
                continue
            endpoint_applied, removed_endpoint_features, removed_counts = trim_endpoint_peak_points(branch_source_df)
            show_black_points = (
                len(removed_endpoint_features) > 0
                and removed_counts.get("left", 0) <= 3
                and removed_counts.get("right", 0) <= 3
            )
            has_other_endpoint_removals = len(removed_endpoint_features) > 0 and not show_black_points
            if len(endpoint_applied) < 5 and show_black_points:
                show_black_points = False
                has_other_endpoint_removals = False
                endpoint_applied = branch_source_df.copy()

            branch_plot_source = branch_source_df if has_other_endpoint_removals else endpoint_applied
            branch_df, _removed_extrema_features = trim_iterative_extrema_points(branch_plot_source)
            if int(lake_id) == 160 and not branch_df.empty:
                branch_df, _removed_special_peak_features = trim_single_global_max_point(branch_df)
            if branch_df.empty:
                continue
            branch_df, removed_noise_df = revive_between_effective_neighbors(branch_source_df, branch_df)
            if branch_df.empty:
                continue
            branch_df = branch_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
            branch_df = add_plot_time_offsets(branch_df)
            if not manual_removed_df.empty:
                removed_noise_df = pd.concat([removed_noise_df, manual_removed_df], ignore_index=True)
            if not removed_noise_df.empty:
                removed_noise_df = removed_noise_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
                removed_noise_df = add_plot_time_offsets(removed_noise_df)

            size_category = str(branch_df["size_category"].iloc[0]) if "size_category" in branch_df.columns else "Unknown"

            out_png = os.path.join(
                paths["single_branch_plot_dir"],
                size_category,
                f"EffectiveAreaCurve_{int(lake_id)}_branch_{branch_id}.png",
            )
            render_single_branch_plot(
                lake_id,
                order,
                total_branch_count,
                branch_df,
                out_png,
                source_df=branch_source_df,
            )
            out_noise_x_png = os.path.join(
                paths["single_branch_plot_noise_x_dir"],
                size_category,
                f"EffectiveAreaCurve_{int(lake_id)}_branch_{branch_id}.png",
            )
            render_single_branch_plot_with_noise_x(
                lake_id,
                order,
                total_branch_count,
                branch_df,
                removed_noise_df,
                out_noise_x_png,
                source_df=branch_source_df,
            )
            plot_count += 1

            node_lookup = {get_node_key(row): row for _, row in branch_df.iterrows()}
            for edge in get_curve_topology_edges(branch_df, source_df=branch_source_df):
                parent_row = node_lookup.get(edge["parent_key"])
                child_row = node_lookup.get(edge["child_key"])
                if parent_row is None or child_row is None:
                    continue
                curve_topology_rows.append(
                    {
                        "lake_id": int(lake_id),
                        "size_category": size_category,
                        "branch_id": str(branch_id),
                        "branch_order": int(order),
                        "edge_type": str(edge["edge_type"]),
                        "parent_date": str(parent_row["date"]),
                        "parent_preview_name": str(parent_row["preview_name"]),
                        "parent_feature_index": int(parent_row["feature_index"]),
                        "parent_area_km2": float(parent_row["area_km2"]),
                        "child_date": str(child_row["date"]),
                        "child_preview_name": str(child_row["preview_name"]),
                        "child_feature_index": int(child_row["feature_index"]),
                        "child_area_km2": float(child_row["area_km2"]),
                    }
                )

            start_candidates = branch_df.head(3).copy()
            start_candidates["area_km2"] = pd.to_numeric(start_candidates["area_km2"], errors="coerce")
            start_candidates = start_candidates[
                start_candidates["area_km2"] > float(BRANCH_START_MIN_AREA_KM2)
            ].copy()
            if not start_candidates.empty:
                start_row = start_candidates.sort_values(
                    ["area_km2", "date_dt", "preview_name", "feature_index"],
                    ascending=[True, True, True, True],
                ).iloc[0]
                start_rec = {
                    "lake_id": int(lake_id),
                    "size_category": size_category,
                    "branch_id": str(branch_id),
                    "branch_order": int(order),
                    "date": str(start_row["date"]),
                    "preview_name": str(start_row["preview_name"]),
                    "feature_index": int(start_row["feature_index"]),
                    "area_km2": float(start_row["area_km2"]),
                    "branch_start_label": f"{int(lake_id)}_{int(order)}",
                }
                start_rec["branch_start_layer"] = build_branch_start_layer_name(start_rec)
                branch_start_rows.append(start_rec)

                start_key = (int(lake_id), str(start_row["date"]), str(start_row["preview_name"]))
                src_jpg = jpg_map.get(start_key, "")
                start_jpg_name = (
                    f"Lake_{int(lake_id)}_Branch_{sanitize_filename(branch_id)}_"
                    f"Start_{str(start_row['date'])}_{sanitize_filename(start_row['preview_name'])}.jpg"
                )
                dst_start_jpg = os.path.join(paths["branch_start_jpg_dir"], size_category, start_jpg_name)
                if copy_jpg_if_exists(src_jpg, dst_start_jpg):
                    start_jpg_count += 1

            for _, point_row in branch_df.iterrows():
                point_key = (int(lake_id), str(point_row["date"]), str(point_row["preview_name"]))
                point_src_jpg = jpg_map.get(point_key, "")
                point_jpg_name = (
                    f"Lake_{int(lake_id)}_Branch_{sanitize_filename(branch_id)}_"
                    f"{str(point_row['date'])}_{sanitize_filename(point_row['preview_name'])}_"
                    f"f{int(point_row['feature_index'])}.jpg"
                )
                point_dst_jpg = os.path.join(paths["curve_point_jpg_dir"], size_category, point_jpg_name)
                copied = copy_jpg_if_exists(point_src_jpg, point_dst_jpg)
                if copied:
                    curve_jpg_count += 1
                curve_point_rows.append(
                    {
                        "lake_id": int(lake_id),
                        "size_category": size_category,
                        "branch_id": str(branch_id),
                        "branch_order": int(order),
                        "date": str(point_row["date"]),
                        "preview_name": str(point_row["preview_name"]),
                        "feature_index": int(point_row["feature_index"]),
                        "area_km2": float(point_row["area_km2"]),
                        "curve_point_layer": build_curve_point_layer_name(
                            {
                                "lake_id": int(lake_id),
                                "branch_order": int(order),
                                "date": str(point_row["date"]),
                                "preview_name": str(point_row["preview_name"]),
                                "feature_index": int(point_row["feature_index"]),
                            }
                        ),
                        "source_jpg_path": str(point_src_jpg),
                        "output_jpg_path": str(point_dst_jpg) if copied else "",
                    }
                )
        print(f"[05_2_simple] processed lake {lake_index}/{total_lakes}: {int(lake_id)}")

    branch_start_df = pd.DataFrame(branch_start_rows)
    curve_point_df = pd.DataFrame(curve_point_rows)
    curve_topology_df = pd.DataFrame(curve_topology_rows)

    branch_start_df.to_csv(paths["branch_start_csv"], index=False, encoding="utf-8-sig")
    curve_point_df.to_csv(paths["curve_point_csv"], index=False, encoding="utf-8-sig")
    curve_topology_df.to_csv(paths["curve_topology_csv"], index=False, encoding="utf-8-sig")
    export_branch_start_gdb(branch_start_df, topo_gdf, paths["branch_start_gdb"])
    export_curve_point_feature_gdbs(curve_point_df, topo_gdf, paths["curve_point_feature_gdb_dir"])

    print(f"[05_2_simple] plots: {plot_count}")
    print(f"[05_2_simple] branch-start records: {len(branch_start_df)} | jpg copied: {start_jpg_count}")
    print(f"[05_2_simple] curve-point records: {len(curve_point_df)} | jpg copied: {curve_jpg_count}")
    print(f"[05_2_simple] curve-topology records: {len(curve_topology_df)}")
    print(f"[05_2_simple] output: {paths['output_dir']}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for method in METHOD_NAMES:
        process_method(method)


if __name__ == "__main__":
    main()



