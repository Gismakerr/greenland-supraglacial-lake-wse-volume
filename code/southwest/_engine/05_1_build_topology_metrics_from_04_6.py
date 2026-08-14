"""
Stage 05_1: build topology metrics directly from Stage 04_6 effective-boundary outputs.

This stage:
1. Reads Stage 04_6 effective-boundary GDB and index CSV.
2. Rebuilds per-scene topology features without going through 04_7/04_8.
3. Applies a lightweight per-scene source filter that mirrors the old 04_8 yellow export.
4. Tracks the main lineage using the largest feature on the peak-water scene as seed.
5. Outputs per-scene metrics, per-feature topology tables, GDB layers, and plots.
"""

import os
from collections import defaultdict

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

try:
    import geopandas as gpd
except ImportError:
    gpd = None

matplotlib.use("Agg")


OUTPUT_ROOT_DIR = str(os.environ.get("OUTPUT_ROOT_DIR", r"X:\2024_西南水位曲线\result1")).strip()
INPUT_04_6_DIR = os.path.join(OUTPUT_ROOT_DIR, "04_3_export_effective_boundaries")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "05_1_build_topology_metrics_from_04_6")
METHOD_NAMES = ["Otsu"]
TARGET_SIZE_CATEGORIES = ["Large", "Medium"]
MAX_LAKE_ID = 0
TARGET_LAKE_IDS = []

INPUT_GDB_LAYER = "effective_boundaries"
INPUT_GDB_PATH_TEMPLATE = os.path.join(INPUT_04_6_DIR, "{method_name}", "04_3_effective_boundaries.gdb")
INPUT_AREA_SUMMARY_TEMPLATE = os.path.join(INPUT_04_6_DIR, "{method_name}", "04_3_effective_boundary_index.csv")

TARGET_CRS = "EPSG:32622"
MIN_INTERSECTION_RATIO = 0.01
MAX_BACKWARD_SCENE_LOOKBACK = 2
MAX_FRAGMENT_BRANCH_NODES = 20
MAX_BRANCH_STITCH_GAP_DAYS = 5
MAX_BRANCH_STITCH_DIST_METERS = 3000.0
MAX_SOURCE_FEATURES_PER_SCENE = 2
AREA_DOMINANCE_RATIO_DROP_THRESHOLD = 10.0
VALID_SCENE_COLOR = "#FFC107"
INVALID_SCENE_COLOR = "#1E88E5"
UNKNOWN_SCENE_COLOR = "royalblue"
BOUNDARY_VALID = "有效边界"
BOUNDARY_INVALID = "无效边界"
BOUNDARY_VALID_ALIASES = {"有效边界", "鏈夋晥杈圭晫"}
BOUNDARY_INVALID_ALIASES = {"无效边界", "鏃犳晥杈圭晫"}

RAW_TOPOLOGY_EDGE_COLOR = "royalblue"
LATE_INVALID_CUTOFF_MONTH_DAY = "0801"
ENABLE_SECONDARY_COMPONENT_FILTER = True
ENABLE_SMALL_DISCONNECTED_COMPONENT_FILTER = False
ENABLE_SPLIT_MERGE_MINOR_BRANCH_FILTER = True
NOISE_STATUS_COLORS = {
    "removed_isolated": "#E53935",
    "removed_secondary_component": "#6D4C41",
    "removed_instant_split_merge": "#8E24AA",
    "removed_split_merge_minor_branch": "#5E35B1",
    "removed_short_premerge_branch": "#FB8C00",
    "removed_short_postsplit_branch": "#00897B",
    "removed_other": "#546E7A",
}
MAX_INSTANT_SPLIT_MERGE_RUN_LEN = 3
MIN_EVENT_BRANCH_NODES = 5
MAX_FORWARD_SCENE_LOOKAHEAD = 3


def get_method_paths(method_name):
    method_dir = os.path.join(OUTPUT_DIR, method_name)
    return {
        "method_name": method_name,
        "input_gdb_path": INPUT_GDB_PATH_TEMPLATE.format(method_name=method_name),
        "input_area_summary_csv": INPUT_AREA_SUMMARY_TEMPLATE.format(method_name=method_name),
        "output_dir": method_dir,
        "metrics_csv": os.path.join(method_dir, "05_1_topology_metrics.csv"),
        "features_csv": os.path.join(method_dir, "05_1_topology_features.csv"),
        "summary_csv": os.path.join(method_dir, "05_1_topology_summary.csv"),
        "branch_summary_base_csv": os.path.join(method_dir, "05_1_topology_branch_summary_base.csv"),
        "raw_plot_features_csv": os.path.join(method_dir, "05_1_raw_plot_features.csv"),
        "output_gdb_path": os.path.join(method_dir, "05_1_topology.gdb"),
        "plots_dir": os.path.join(method_dir, "05_1_TopologyPlots"),
    }


def print_run_summary(method_paths):
    print(f"[05_1_04_6] Build Topology Metrics From 04_6 - {method_paths['method_name']}", flush=True)
    print(f"  input_gdb: {method_paths['input_gdb_path']}", flush=True)
    print(f"  input_area_summary_csv: {method_paths['input_area_summary_csv']}", flush=True)
    print(f"  output_dir: {method_paths['output_dir']}", flush=True)
    print(f"  target_lake_ids: {TARGET_LAKE_IDS if TARGET_LAKE_IDS else 'ALL'}", flush=True)


def print_completion_summary(method_paths, metrics_df, summary_df):
    print(f"[05_1_04_6] Completed - {method_paths['method_name']}", flush=True)
    print(f"  metrics_csv: {method_paths['metrics_csv']}", flush=True)
    print(f"  features_csv: {method_paths['features_csv']}", flush=True)
    print(f"  summary_csv: {method_paths['summary_csv']}", flush=True)
    print(f"  branch_summary_base_csv: {method_paths['branch_summary_base_csv']}", flush=True)
    print(f"  raw_plot_features_csv: {method_paths['raw_plot_features_csv']}", flush=True)
    print(f"  output_gdb: {method_paths['output_gdb_path']}", flush=True)
    print(f"  plots_dir: {method_paths['plots_dir']}", flush=True)
    print(f"  scene_records: {len(metrics_df)}", flush=True)
    print(f"  lakes: {summary_df['lake_id'].nunique() if not summary_df.empty else 0}", flush=True)


def load_area_summary(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        return df
    work = df.copy()
    work["lake_id"] = pd.to_numeric(work["lake_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["lake_id", "date"]).copy()
    work["lake_id"] = work["lake_id"].astype(int)
    work["date"] = work["date"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(work["date"].astype(str))
    work["preview_name"] = work["preview_name"].fillna("").astype(str)
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y%m%d", errors="coerce")
    if "size_category" in work.columns:
        work["size_category_04_8"] = work["size_category"].fillna("Small").astype(str)
    elif "size_category_04_8" in work.columns:
        work["size_category_04_8"] = work["size_category_04_8"].fillna("Small").astype(str)
    else:
        work["size_category_04_8"] = "Small"
    work["scene_boundary_bucket"] = work.get("jpg_bucket", "").fillna("").astype(str).map(normalize_scene_bucket)
    work["area_km2"] = pd.to_numeric(work.get("water_area_km2", np.nan), errors="coerce")
    work["feature_count"] = pd.to_numeric(work.get("effective_features", np.nan), errors="coerce")
    if TARGET_SIZE_CATEGORIES:
        work = work[work["size_category_04_8"].isin(TARGET_SIZE_CATEGORIES)].copy()
    if TARGET_LAKE_IDS:
        lake_id_set = {int(v) for v in TARGET_LAKE_IDS}
        work = work[work["lake_id"].isin(lake_id_set)].copy()
    if MAX_LAKE_ID and MAX_LAKE_ID > 0:
        work = work[work["lake_id"] <= int(MAX_LAKE_ID)].copy()
    return work


def load_topology_source_gdf(gdb_path):
    if gpd is None:
        raise ImportError("geopandas is required to rebuild 05_1 topology from the source GDB")
    gdf = gpd.read_file(gdb_path, layer=INPUT_GDB_LAYER)
    if gdf.empty:
        return gdf
    if "date" not in gdf.columns and "date_" in gdf.columns:
        gdf = gdf.rename(columns={"date_": "date"})
    if str(gdf.crs) != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    gdf["lake_id"] = pd.to_numeric(gdf["lake_id"], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=["lake_id", "date", "preview_name"]).copy()
    gdf["lake_id"] = gdf["lake_id"].astype(int)
    gdf["date"] = gdf["date"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(gdf["date"].astype(str))
    gdf["preview_name"] = gdf["preview_name"].fillna("").astype(str)
    gdf["geom_area_km2"] = pd.to_numeric(gdf["geom_area_km2"], errors="coerce")
    gdf["feature_idx"] = pd.to_numeric(gdf["feature_idx"], errors="coerce")
    if TARGET_SIZE_CATEGORIES and "size_category" in gdf.columns:
        gdf["size_category"] = gdf["size_category"].fillna("").astype(str)
        gdf = gdf[gdf["size_category"].isin(TARGET_SIZE_CATEGORIES)].copy()
    if TARGET_LAKE_IDS:
        lake_id_set = {int(v) for v in TARGET_LAKE_IDS}
        gdf = gdf[gdf["lake_id"].isin(lake_id_set)].copy()
    if MAX_LAKE_ID and MAX_LAKE_ID > 0:
        gdf = gdf[gdf["lake_id"] <= int(MAX_LAKE_ID)].copy()
    if "is_invalid_small_feature" not in gdf.columns:
        gdf["is_invalid_small_feature"] = False
    gdf["is_invalid_small_feature"] = gdf["is_invalid_small_feature"].fillna(False).astype(bool)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf.reset_index(drop=True)


def finalize_source_features(scene_gdf):
    work = scene_gdf.copy()
    if work.empty:
        return work

    valid_mask = ~work["is_invalid_small_feature"].fillna(False)
    invalid_mask = work["is_invalid_small_feature"].fillna(False)
    if int(valid_mask.sum()) > 0 and int(invalid_mask.sum()) > 0:
        work = work.loc[valid_mask].copy()

    work["geom_area_km2"] = pd.to_numeric(work["geom_area_km2"], errors="coerce")
    work = work.sort_values(["geom_area_km2", "feature_idx"], ascending=[False, True]).head(MAX_SOURCE_FEATURES_PER_SCENE).copy()
    if len(work) >= 2:
        first_area = pd.to_numeric(work["geom_area_km2"].iloc[0], errors="coerce")
        second_area = pd.to_numeric(work["geom_area_km2"].iloc[1], errors="coerce")
        if pd.notna(first_area) and pd.notna(second_area) and float(second_area) > 0:
            if float(first_area) / float(second_area) >= AREA_DOMINANCE_RATIO_DROP_THRESHOLD:
                work = work.head(1).copy()
    return work.reset_index(drop=True)


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


def encode_node_key(node_key):
    return f"{str(node_key[0])}|{str(node_key[1])}|{int(node_key[2])}"


def get_node_key(row):
    return (str(row.get("date", "")), str(row.get("preview_name", "")), int(row.get("feature_index", -1)))


def get_scene_bucket_color(bucket):
    bucket_text = normalize_scene_bucket(bucket)
    if bucket_text == BOUNDARY_VALID:
        return VALID_SCENE_COLOR
    if bucket_text == BOUNDARY_INVALID:
        return INVALID_SCENE_COLOR
    return UNKNOWN_SCENE_COLOR


def normalize_scene_bucket(bucket):
    bucket_text = str(bucket).strip()
    if bucket_text in BOUNDARY_VALID_ALIASES:
        return BOUNDARY_VALID
    if bucket_text in BOUNDARY_INVALID_ALIASES:
        return BOUNDARY_INVALID
    return bucket_text


def get_precomputed_display_kept_df(lake_features):
    if lake_features.empty:
        return lake_features.copy()
    if "display_status" not in lake_features.columns:
        return lake_features.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    kept_df = lake_features[lake_features["display_status"].fillna("") == "kept_display"].copy()
    return kept_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()


def lock_blue_after_late_invalid(df):
    if df.empty or "scene_boundary_bucket" not in df.columns:
        return df.copy()
    work = df.copy()
    daily = work[["date", "scene_boundary_bucket"]].drop_duplicates().sort_values("date").copy()
    daily["month_day"] = daily["date"].astype(str).str[-4:]
    bucket_norm = daily["scene_boundary_bucket"].map(normalize_scene_bucket)
    mask = (
        daily["month_day"].gt(LATE_INVALID_CUTOFF_MONTH_DAY)
        & bucket_norm.eq(BOUNDARY_INVALID)
    )
    if not mask.any():
        return work
    lock_start_date = str(daily.loc[mask, "date"].iloc[0])
    work.loc[work["date"].astype(str) >= lock_start_date, "scene_boundary_bucket"] = BOUNDARY_INVALID
    return work


def lock_blue_for_late_peak(df):
    if df.empty or "scene_boundary_bucket" not in df.columns:
        return df.copy()
    work = df.copy()
    changed = True
    while changed and not work.empty:
        changed = False
        active = work[work["scene_boundary_bucket"].map(normalize_scene_bucket) != BOUNDARY_INVALID].copy()
        if active.empty:
            break
        area_series = pd.to_numeric(active["area_km2"], errors="coerce")
        if not area_series.notna().any():
            break
        peak_idx = area_series.idxmax()
        peak_row = active.loc[peak_idx]
        peak_date = str(peak_row["date"])
        if peak_date[-4:] <= LATE_INVALID_CUTOFF_MONTH_DAY:
            break
        work.loc[work["date"].astype(str) == peak_date, "scene_boundary_bucket"] = BOUNDARY_INVALID
        work = lock_blue_after_late_invalid(work)
        changed = True
    return work


def build_parent_scene_lookup(features_df):
    if features_df.empty:
        return {}
    ordered_scenes = (
        features_df[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    return {scene_keys[idx]: scene_keys[idx - 1] for idx in range(1, len(scene_keys))}


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
        shifted = [
            pd.to_datetime(base_dt) + pd.Timedelta(hours=hours)
            for base_dt, hours in zip(day_df["date_dt"], offsets)
        ]
        work.loc[day_df.index, "x_plot_dt"] = shifted
    return work


def build_backfill_visual_edges(features_df, keep_key_set):
    """
    Build cross-scene visual edges for kept nodes that lost immediate parents.
    The inferred edges are used only for plotting and do not overwrite
    parent_indices (which remain immediate-scene topology).
    """
    if features_df.empty or not keep_key_set:
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
        # Skip nodes that already have any topology parent (immediate or bridge).
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

        found = False
        for back in range(2, int(MAX_BACKWARD_SCENE_LOOKBACK) + 2):
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
                if max(overlap_ratio_curr, overlap_ratio_prev) >= MIN_INTERSECTION_RATIO:
                    candidates.append((parent_key, inter_area))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[1], reverse=True)
            # Keep strongest bridge from the nearest available previous scene.
            edge_set.add((candidates[0][0], child_key))
            found = True
            break
        if not found:
            continue

    return list(edge_set)


def compute_tree_layout(topology_df):
    children_map = defaultdict(list)
    parent_map = defaultdict(list)
    node_rows = {}
    for _, row in topology_df.iterrows():
        node_key = get_node_key(row)
        node_rows[node_key] = row
    ordered_scenes = (
        topology_df[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
    )
    scene_ids = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    scene_to_prev = {scene_ids[idx]: scene_ids[idx - 1] for idx in range(1, len(scene_ids))}
    for _, row in topology_df.iterrows():
        node_key = get_node_key(row)
        current_scene = (node_key[0], node_key[1])
        prev_scene = scene_to_prev.get(current_scene)
        valid_parents = []
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], parent_idx)
                if parent_key in node_rows:
                    valid_parents.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in node_rows:
                valid_parents.append(parent_key)
        # Preserve order but remove duplicates.
        dedup_parents = []
        seen_parent = set()
        for parent_key in valid_parents:
            if parent_key in seen_parent:
                continue
            seen_parent.add(parent_key)
            dedup_parents.append(parent_key)
        for parent_key in dedup_parents:
            children_map[parent_key].append(node_key)
            parent_map[node_key].append(parent_key)
    roots = [node_key for node_key in node_rows.keys() if len(parent_map.get(node_key, [])) == 0]
    roots.sort(key=lambda key: (node_rows[key]["date"], -float(node_rows[key]["area_km2"]), key[1]))
    for parent_key, child_list in children_map.items():
        child_list.sort(key=lambda key: (node_rows[key]["date"], -float(node_rows[key]["area_km2"]), key[1]))
    return roots, children_map, parent_map, node_rows


def compute_branch_layout(topology_df):
    children_map = defaultdict(list)
    parent_map = defaultdict(list)
    node_rows = {}
    for _, row in topology_df.iterrows():
        node_key = get_node_key(row)
        node_rows[node_key] = row
    ordered_scenes = (
        topology_df[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
    )
    scene_ids = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    scene_to_prev = {scene_ids[idx]: scene_ids[idx - 1] for idx in range(1, len(scene_ids))}
    for _, row in topology_df.iterrows():
        node_key = get_node_key(row)
        current_scene = (node_key[0], node_key[1])
        prev_scene = scene_to_prev.get(current_scene)
        valid_parents = []
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], parent_idx)
                if parent_key in node_rows:
                    valid_parents.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in node_rows:
                valid_parents.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("branch_match_parent_keys", "")):
            if parent_key in node_rows:
                valid_parents.append(parent_key)
        dedup_parents = []
        seen_parent = set()
        for parent_key in valid_parents:
            if parent_key in seen_parent:
                continue
            seen_parent.add(parent_key)
            dedup_parents.append(parent_key)
        for parent_key in dedup_parents:
            children_map[parent_key].append(node_key)
            parent_map[node_key].append(parent_key)
    roots = [node_key for node_key in node_rows.keys() if len(parent_map.get(node_key, [])) == 0]
    roots.sort(key=lambda key: (node_rows[key]["date"], -float(node_rows[key]["area_km2"]), key[1]))
    for parent_key, child_list in children_map.items():
        child_list.sort(key=lambda key: (node_rows[key]["date"], -float(node_rows[key]["area_km2"]), key[1]))
    return roots, children_map, parent_map, node_rows


def node_rank(node_key, node_rows):
    row = node_rows[node_key]
    is_main = bool(row.get("is_main_lineage", False))
    area = pd.to_numeric(row.get("area_km2", np.nan), errors="coerce")
    area_value = float(area) if pd.notna(area) else -1.0
    return (1 if is_main else 0, area_value, -int(node_key[2]))


def assign_branch_ids(topology_df):
    if topology_df.empty:
        work = topology_df.copy()
        work["branch_id"] = pd.Series(dtype=object)
        return work

    work = topology_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    roots, children_map, parent_map, node_rows = compute_branch_layout(work)
    ordered_keys = sorted(
        node_rows.keys(),
        key=lambda key: (
            pd.to_datetime(node_rows[key].get("date_dt"), errors="coerce"),
            str(node_rows[key].get("preview_name", "")),
            int(key[2]),
        ),
    )

    def choose_primary_child(parent_key):
        child_keys = list(children_map.get(parent_key, []))
        if not child_keys:
            return None
        return max(child_keys, key=lambda key: node_rank(key, node_rows))

    def choose_primary_parent(child_key):
        parent_keys = list(parent_map.get(child_key, []))
        if not parent_keys:
            return None
        return max(parent_keys, key=lambda key: node_rank(key, node_rows))

    def choose_branch_match_parent(child_key):
        row = node_rows.get(child_key)
        if row is None:
            return None
        match_parents = [
            parent_key
            for parent_key in parse_semicolon_node_keys(row.get("branch_match_parent_keys", ""))
            if parent_key in node_rows
        ]
        if not match_parents:
            return None
        return match_parents[0]

    branch_ids = {}
    branch_counter = 1

    def new_branch_id():
        nonlocal branch_counter
        branch_id = f"B{branch_counter:04d}"
        branch_counter += 1
        return branch_id

    for root_key in roots:
        if root_key not in branch_ids:
            branch_ids[root_key] = new_branch_id()

    changed = True
    while changed:
        changed = False
        for node_key in ordered_keys:
            matched_parent = choose_branch_match_parent(node_key)
            if matched_parent is not None and matched_parent in branch_ids:
                matched_branch = str(branch_ids.get(matched_parent, "")).strip()
                if matched_branch and branch_ids.get(node_key) != matched_branch:
                    branch_ids[node_key] = matched_branch
                    changed = True

            if node_key not in branch_ids:
                branch_ids[node_key] = new_branch_id()
                changed = True

            node_branch = branch_ids[node_key]
            child_keys = list(children_map.get(node_key, []))
            if not child_keys:
                continue

            primary_child = choose_primary_child(node_key) if len(child_keys) > 1 else child_keys[0]
            for child_key in child_keys:
                primary_parent = choose_primary_parent(child_key)
                if primary_parent != node_key:
                    continue
                if child_key in branch_ids:
                    continue
                if len(child_keys) > 1 and child_key != primary_child:
                    branch_ids[child_key] = new_branch_id()
                else:
                    branch_ids[child_key] = node_branch
                changed = True

    work["branch_id"] = work.apply(
        lambda row: branch_ids.get((str(row["date"]), str(row["preview_name"]), int(row["feature_index"])), ""),
        axis=1,
    )

    # Merge short disconnected fragments back to nearby main branches
    # (temporal gap + centroid distance), without changing topology edges.
    if not work.empty:
        branch_ids_all = [str(v).strip() for v in work["branch_id"].dropna().astype(str).unique().tolist() if str(v).strip()]
        if len(branch_ids_all) > 2:
            stats = {}
            for bid, g in work.groupby("branch_id", sort=False):
                bid_text = str(bid).strip()
                if not bid_text:
                    continue
                gg = g.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
                start_row = gg.iloc[0]
                end_row = gg.iloc[-1]
                stats[bid_text] = {
                    "count": int(len(gg)),
                    "start_dt": pd.to_datetime(start_row.get("date_dt"), errors="coerce"),
                    "end_dt": pd.to_datetime(end_row.get("date_dt"), errors="coerce"),
                    "start_x": float(pd.to_numeric(start_row.get("centroid_x"), errors="coerce")),
                    "start_y": float(pd.to_numeric(start_row.get("centroid_y"), errors="coerce")),
                    "end_x": float(pd.to_numeric(end_row.get("centroid_x"), errors="coerce")),
                    "end_y": float(pd.to_numeric(end_row.get("centroid_y"), errors="coerce")),
                }

            if len(stats) > 2:
                sorted_ids = sorted(
                    stats.keys(),
                    key=lambda b: (stats[b]["count"], stats[b]["start_dt"]),
                    reverse=True,
                )
                anchor_ids = set(sorted_ids[:2])
                fragment_ids = [
                    b for b in stats.keys() if b not in anchor_ids and stats[b]["count"] <= int(MAX_FRAGMENT_BRANCH_NODES)
                ]

                def _dist(x1, y1, x2, y2):
                    if any(pd.isna(v) for v in [x1, y1, x2, y2]):
                        return np.inf
                    return float(np.hypot(float(x1) - float(x2), float(y1) - float(y2)))

                remap = {}
                for frag in sorted(fragment_ids, key=lambda b: stats[b]["count"]):
                    f = stats[frag]
                    best_target = None
                    best_score = np.inf
                    for cand in anchor_ids:
                        c = stats[cand]
                        if pd.isna(f["start_dt"]) or pd.isna(f["end_dt"]) or pd.isna(c["start_dt"]) or pd.isna(c["end_dt"]):
                            continue

                        gap_prev = (f["start_dt"] - c["end_dt"]).days
                        if 0 <= gap_prev <= int(MAX_BRANCH_STITCH_GAP_DAYS):
                            dist_prev = _dist(c["end_x"], c["end_y"], f["start_x"], f["start_y"])
                            if dist_prev <= float(MAX_BRANCH_STITCH_DIST_METERS):
                                score = gap_prev * 10000.0 + dist_prev
                                if score < best_score:
                                    best_score = score
                                    best_target = cand

                        gap_next = (c["start_dt"] - f["end_dt"]).days
                        if 0 <= gap_next <= int(MAX_BRANCH_STITCH_GAP_DAYS):
                            dist_next = _dist(f["end_x"], f["end_y"], c["start_x"], c["start_y"])
                            if dist_next <= float(MAX_BRANCH_STITCH_DIST_METERS):
                                score = gap_next * 10000.0 + dist_next
                                if score < best_score:
                                    best_score = score
                                    best_target = cand

                    if best_target:
                        remap[frag] = best_target

                if remap:
                    work["branch_id"] = work["branch_id"].astype(str).replace(remap)

    # Display-oriented branch normalization:
    # - before merge: two connected lines -> B0001/B0002
    # - after merge (single-node scene): all become B0001
    if not work.empty:
        work = work.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        scene_order = (
            work[["date", "preview_name", "date_dt"]]
            .drop_duplicates()
            .sort_values(["date_dt", "preview_name"])
            .to_dict("records")
        )
        prev_scene_branch_by_feature = {}

        for scene in scene_order:
            date_val = str(scene["date"])
            preview_val = str(scene["preview_name"])
            scene_mask = (work["date"].astype(str) == date_val) & (work["preview_name"].astype(str) == preview_val)
            scene_df = work.loc[scene_mask].copy()
            if scene_df.empty:
                continue

            scene_df["_area"] = pd.to_numeric(scene_df["area_km2"], errors="coerce").fillna(-1.0)
            scene_df["_parent_count"] = pd.to_numeric(scene_df["parent_count"], errors="coerce").fillna(0).astype(int)
            scene_df["_parent_list"] = scene_df["parent_indices"].apply(parse_semicolon_ints)

            # Initial candidate from immediate parents; fallback by area rank.
            candidates = {}
            inherited_from_parent = {}
            area_sorted_idx = scene_df.sort_values(["_area", "feature_index"], ascending=[False, True]).index.tolist()
            for rank, row_idx in enumerate(area_sorted_idx):
                row = scene_df.loc[row_idx]
                parent_branches = []
                for pidx in row["_parent_list"]:
                    if pidx in prev_scene_branch_by_feature:
                        parent_branches.append(prev_scene_branch_by_feature[pidx])

                if row["_parent_count"] >= 2:
                    cand = "B0001"
                    inherited_from_parent[row_idx] = True
                elif parent_branches:
                    if "B0001" in parent_branches:
                        cand = "B0001"
                    elif "B0002" in parent_branches:
                        cand = "B0002"
                    else:
                        cand = "B0001" if rank == 0 else "B0002"
                    inherited_from_parent[row_idx] = True
                else:
                    cand = "B0001" if rank == 0 else "B0002"
                    inherited_from_parent[row_idx] = False
                candidates[row_idx] = cand

            # Enforce scene-level display rule.
            if len(scene_df) == 1:
                only_idx = scene_df.index[0]
                only_row = scene_df.loc[only_idx]
                only_parent_count = int(only_row["_parent_count"])
                if only_parent_count >= 2:
                    candidates[only_idx] = "B0001"
                elif inherited_from_parent.get(only_idx, False):
                    candidates[only_idx] = candidates.get(only_idx, "B0001")
                else:
                    candidates[only_idx] = "B0001"
            elif len(scene_df) == 2:
                idx_a, idx_b = scene_df.index[0], scene_df.index[1]
                a, b = candidates.get(idx_a, "B0001"), candidates.get(idx_b, "B0002")
                if a == b:
                    parent_rows = [ridx for ridx in [idx_a, idx_b] if inherited_from_parent.get(ridx, False)]
                    if len(parent_rows) == 1:
                        keep_idx = parent_rows[0]
                        other_idx = idx_b if keep_idx == idx_a else idx_a
                        keep_bid = candidates.get(keep_idx, "B0001")
                        candidates[keep_idx] = keep_bid
                        candidates[other_idx] = "B0002" if keep_bid == "B0001" else "B0001"
                    else:
                        top_idx = area_sorted_idx[0]
                        other_idx = area_sorted_idx[1]
                        candidates[top_idx] = "B0001"
                        candidates[other_idx] = "B0002"
                else:
                    # Keep explicit merged child on B0001.
                    for ridx in [idx_a, idx_b]:
                        if int(scene_df.loc[ridx, "_parent_count"]) >= 2:
                            candidates[ridx] = "B0001"
                            other = idx_b if ridx == idx_a else idx_a
                            candidates[other] = "B0002"
                            break

            for ridx, bid in candidates.items():
                work.at[ridx, "branch_id"] = bid

            prev_scene_branch_by_feature = {
                int(scene_df.loc[ridx, "feature_index"]): str(candidates.get(ridx, "B0001"))
                for ridx in scene_df.index
            }

    return work


def classify_final_curve_size_category(max_area_km2):
    max_area = pd.to_numeric(max_area_km2, errors="coerce")
    if pd.isna(max_area):
        return "Small"
    if float(max_area) > 0.625:
        return "Large"
    if float(max_area) > 0.0625:
        return "Medium"
    return "Small"


def prepare_raw_plot_features_df(features_df):
    if features_df.empty:
        return features_df.copy()
    if "display_status" not in features_df.columns:
        features_df = features_df.copy()
        features_df["display_status"] = "kept_display"

    out_frames = []
    for lake_id, lake_df in features_df.groupby("lake_id", sort=True):
        lake_work = lake_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        lake_work = lock_blue_after_late_invalid(lake_work)
        lake_work = lock_blue_for_late_peak(lake_work)

        kept_df = lake_work[lake_work["display_status"].fillna("") == "kept_display"].copy()
        noise_df = lake_work[lake_work["display_status"].fillna("") != "kept_display"].copy()

        if not kept_df.empty:
            kept_df = assign_branch_ids(kept_df)
        else:
            kept_df["branch_id"] = ""

        # Final safeguard:
        # if a short branch (<5 nodes) merges into another kept branch,
        # drop the whole short branch as short-premerge noise.
        if not kept_df.empty and "branch_id" in kept_df.columns:
            kept_work = kept_df.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
            branch_counts = kept_work["branch_id"].astype(str).value_counts()
            candidate_branches = {
                bid for bid, cnt in branch_counts.items()
                if str(bid).strip() and int(cnt) < int(MIN_EVENT_BRANCH_NODES)
            }
            if candidate_branches:
                node_key_to_idx = {
                    (str(row["date"]), str(row["preview_name"]), int(row["feature_index"])): idx
                    for idx, row in kept_work.iterrows()
                }
                scene_order = (
                    kept_work[["date", "preview_name", "date_dt"]]
                    .drop_duplicates()
                    .sort_values(["date_dt", "preview_name"])
                    .reset_index(drop=True)
                )
                scene_keys = [(str(r["date"]), str(r["preview_name"])) for _, r in scene_order.iterrows()]
                next_scene_lookup = {
                    scene_keys[i]: scene_keys[i + 1]
                    for i in range(len(scene_keys) - 1)
                }

                drop_branches = set()
                for bid in candidate_branches:
                    branch_rows = kept_work[kept_work["branch_id"].astype(str) == str(bid)].copy()
                    if branch_rows.empty:
                        continue
                    merges_out = False
                    for idx, row in branch_rows.iterrows():
                        row_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
                        curr_scene = (row_key[0], row_key[1])

                        # Immediate next-scene children
                        next_scene = next_scene_lookup.get(curr_scene)
                        if next_scene is not None:
                            for child_idx in parse_semicolon_ints(row.get("child_indices", "")):
                                child_key = (next_scene[0], next_scene[1], int(child_idx))
                                child_row_idx = node_key_to_idx.get(child_key)
                                if child_row_idx is None:
                                    continue
                                child_branch = str(kept_work.loc[child_row_idx, "branch_id"]).strip()
                                if child_branch and child_branch != str(bid):
                                    merges_out = True
                                    break
                        if merges_out:
                            break

                        # Skip-scene bridge children
                        for child_key in parse_semicolon_node_keys(row.get("bridge_child_keys", "")):
                            child_row_idx = node_key_to_idx.get(child_key)
                            if child_row_idx is None:
                                continue
                            child_branch = str(kept_work.loc[child_row_idx, "branch_id"]).strip()
                            if child_branch and child_branch != str(bid):
                                merges_out = True
                                break
                        if merges_out:
                            break

                    if merges_out:
                        drop_branches.add(str(bid))

                if drop_branches:
                    drop_mask = kept_work["branch_id"].astype(str).isin(drop_branches)
                    drop_df = kept_work.loc[drop_mask].copy()
                    if not drop_df.empty:
                        drop_df["display_status"] = "removed_short_premerge_branch"
                        drop_df["removed_from_display"] = True
                        drop_df["is_isolated_removed"] = False
                        drop_df["is_secondary_component_removed"] = False
                        drop_df["is_instant_split_merge_removed"] = False
                        drop_df["is_short_premerge_branch_removed"] = True
                        drop_df["is_short_postsplit_branch_removed"] = False
                        if "branch_id" in drop_df.columns:
                            drop_df["branch_id"] = ""
                        noise_df = pd.concat([noise_df, drop_df], ignore_index=True, sort=False)
                        kept_df = kept_work.loc[~drop_mask].copy()

        if not noise_df.empty:
            noise_df["branch_id"] = ""

        combined_lake = pd.concat([kept_df, noise_df], ignore_index=True, sort=False)
        combined_lake = add_plot_time_offsets(combined_lake)
        out_frames.append(combined_lake)

    if not out_frames:
        return features_df.iloc[0:0].copy()
    return pd.concat(out_frames, ignore_index=True, sort=False)


def get_branch_marker(branch_id):
    branch_text = str(branch_id).strip()
    markers = ["o", "s"]
    if not branch_text:
        return markers[0]
    branch_num = sum(ord(ch) for ch in branch_text)
    return markers[branch_num % len(markers)]


def build_scene_records(source_gdf, area_df):
    scene_records = []
    grouped = source_gdf.groupby(["lake_id", "date", "preview_name"], sort=False)
    for (lake_id, date_str, preview_name), scene_gdf in grouped:
        scene_gdf = finalize_source_features(scene_gdf)
        if scene_gdf.empty:
            continue
        area_row = area_df[
            (area_df["lake_id"] == int(lake_id))
            & (area_df["date"] == str(date_str))
            & (area_df["preview_name"] == str(preview_name))
        ]
        size_category = "Small"
        date_dt = pd.to_datetime(str(date_str), format="%Y%m%d", errors="coerce")
        scene_boundary_bucket = ""
        if not area_row.empty:
            area_row = area_row.iloc[0]
            size_category = str(area_row.get("size_category_04_8", "Small"))
            scene_boundary_bucket = str(area_row.get("scene_boundary_bucket", ""))
            if pd.notna(area_row.get("date_dt", pd.NaT)):
                date_dt = area_row["date_dt"]
        scene_total_area_km2 = float(pd.to_numeric(scene_gdf["geom_area_km2"], errors="coerce").fillna(0).sum())
        feature_count = int(len(scene_gdf))
        features = []
        for seq_idx, (_, feat_row) in enumerate(scene_gdf.iterrows()):
            geom = feat_row.geometry
            centroid = geom.centroid
            features.append(
                {
                    "feature_index": int(seq_idx),
                    "source_feature_idx": int(feat_row["feature_idx"]) if pd.notna(feat_row["feature_idx"]) else int(seq_idx + 1),
                    "area_km2": float(feat_row["geom_area_km2"]) if pd.notna(feat_row["geom_area_km2"]) else float(geom.area / 1e6),
                    "geom": geom,
                    "centroid_x": float(centroid.x),
                    "centroid_y": float(centroid.y),
                    "parent_indices": [],
                    "child_indices": [],
                    "bridge_parent_keys": [],
                    "bridge_child_keys": [],
                    "branch_match_parent_keys": [],
                    "branch_match_child_keys": [],
                    "is_main_lineage": False,
                }
            )
        scene_records.append(
            {
                "lake_id": int(lake_id),
                "date": str(date_str),
                "preview_name": str(preview_name),
                "date_dt": date_dt,
                "scene_total_area_km2": scene_total_area_km2,
                "feature_count": feature_count,
                "size_category_04_8": size_category,
                "scene_boundary_bucket": scene_boundary_bucket,
                "features": features,
            }
        )
    scene_records.sort(key=lambda item: (item["lake_id"], item["date_dt"], item["preview_name"]))
    return scene_records


def map_feature_links(prev_features, curr_features):
    def _overlap_score(prev_feature, curr_feature):
        curr_geom = curr_feature["geom"]
        prev_geom = prev_feature["geom"]
        if not curr_geom.intersects(prev_geom):
            return 0.0
        inter_area = curr_geom.intersection(prev_geom).area
        if inter_area <= 0:
            return 0.0
        overlap_ratio_curr = inter_area / max(curr_geom.area, 1e-12)
        overlap_ratio_prev = inter_area / max(prev_geom.area, 1e-12)
        if max(overlap_ratio_curr, overlap_ratio_prev) < MIN_INTERSECTION_RATIO:
            return 0.0
        return float(inter_area)

    # For the common 2-branch scenes, enforce one-to-one matching before merge.
    # This avoids cross-linking ("square->circle") when both branches persist.
    if len(prev_features) == 2 and len(curr_features) == 2:
        score_00 = _overlap_score(prev_features[0], curr_features[0])
        score_01 = _overlap_score(prev_features[0], curr_features[1])
        score_10 = _overlap_score(prev_features[1], curr_features[0])
        score_11 = _overlap_score(prev_features[1], curr_features[1])

        same_total = score_00 + score_11
        cross_total = score_01 + score_10
        if max(same_total, cross_total) > 0:
            for curr_feature in curr_features:
                curr_feature["parent_indices"] = []

            if same_total >= cross_total:
                chosen_pairs = [(0, 0, score_00), (1, 1, score_11)]
            else:
                chosen_pairs = [(0, 1, score_01), (1, 0, score_10)]

            for prev_idx, curr_idx, score in chosen_pairs:
                if score > 0:
                    curr_features[curr_idx]["parent_indices"] = [prev_idx]

            for prev_idx, prev_feature in enumerate(prev_features):
                prev_feature["child_indices"] = [
                    idx for idx, curr_feature in enumerate(curr_features) if prev_idx in curr_feature["parent_indices"]
                ]
            return

    for curr_feature in curr_features:
        curr_area = max(curr_feature["geom"].area, 1e-12)
        parents = []
        for prev_idx, prev_feature in enumerate(prev_features):
            if not curr_feature["geom"].intersects(prev_feature["geom"]):
                continue
            inter_area = curr_feature["geom"].intersection(prev_feature["geom"]).area
            if inter_area <= 0:
                continue
            overlap_ratio_curr = inter_area / curr_area
            overlap_ratio_prev = inter_area / max(prev_feature["geom"].area, 1e-12)
            if max(overlap_ratio_curr, overlap_ratio_prev) >= MIN_INTERSECTION_RATIO:
                parents.append((prev_idx, inter_area))
        parents.sort(key=lambda item: item[1], reverse=True)
        curr_feature["parent_indices"] = [idx for idx, _ in parents]
    for prev_idx, prev_feature in enumerate(prev_features):
        prev_feature["child_indices"] = [
            idx for idx, curr_feature in enumerate(curr_features) if prev_idx in curr_feature["parent_indices"]
        ]


def backfill_gap_links_by_intersection(ordered_scenes):
    """
    If a feature has no parent in the immediate previous scene, try to connect it
    to older scenes by actual geometry intersection (within a short lookback).
    """
    if not ordered_scenes or len(ordered_scenes) < 3:
        return

    for scene_idx in range(2, len(ordered_scenes)):
        curr_features = ordered_scenes[scene_idx]["features"]
        for curr_idx, curr_feature in enumerate(curr_features):
            # Keep existing parent links; only repair true gaps.
            if curr_feature.get("parent_indices"):
                continue
            curr_geom = curr_feature["geom"]

            found = False
            for back in range(2, int(MAX_BACKWARD_SCENE_LOOKBACK) + 2):
                prev_scene_idx = scene_idx - back
                if prev_scene_idx < 0:
                    break
                prev_features = ordered_scenes[prev_scene_idx]["features"]
                candidates = []
                for prev_idx, prev_feature in enumerate(prev_features):
                    prev_geom = prev_feature["geom"]
                    inter_area = curr_geom.intersection(prev_geom).area
                    if inter_area <= 0:
                        continue
                    overlap_ratio_curr = inter_area / max(curr_geom.area, 1e-12)
                    overlap_ratio_prev = inter_area / max(prev_geom.area, 1e-12)
                    if max(overlap_ratio_curr, overlap_ratio_prev) >= MIN_INTERSECTION_RATIO:
                        candidates.append((prev_idx, inter_area))
                if not candidates:
                    continue

                candidates.sort(key=lambda item: item[1], reverse=True)
                curr_feature["parent_indices"] = [idx for idx, _ in candidates]
                for prev_idx, _ in candidates:
                    child_indices = [int(v) for v in prev_features[prev_idx].get("child_indices", [])]
                    if curr_idx not in child_indices:
                        child_indices.append(curr_idx)
                    prev_features[prev_idx]["child_indices"] = sorted(set(child_indices))
                found = True
                break

            if not found:
                curr_feature["parent_indices"] = []


def repair_parent_child_indices(ordered_scenes):
    """
    Ensure parent indices always reference the immediate previous scene, then
    rebuild child indices from parent indices.
    """
    if not ordered_scenes or len(ordered_scenes) < 2:
        return

    # First pass: sanitize/repair parent indices against immediate previous scene.
    for idx in range(1, len(ordered_scenes)):
        prev_features = ordered_scenes[idx - 1]["features"]
        curr_features = ordered_scenes[idx]["features"]
        prev_count = len(prev_features)
        for curr_feature in curr_features:
            raw_parents = [int(v) for v in curr_feature.get("parent_indices", [])]
            valid_parents = sorted({p for p in raw_parents if 0 <= p < prev_count})
            if valid_parents:
                curr_feature["parent_indices"] = valid_parents
                continue

            # If parent indices are empty/invalid, try immediate-scene intersection.
            curr_geom = curr_feature["geom"]
            candidates = []
            for prev_idx, prev_feature in enumerate(prev_features):
                prev_geom = prev_feature["geom"]
                inter_area = curr_geom.intersection(prev_geom).area
                if inter_area <= 0:
                    continue
                overlap_ratio_curr = inter_area / max(curr_geom.area, 1e-12)
                overlap_ratio_prev = inter_area / max(prev_geom.area, 1e-12)
                if max(overlap_ratio_curr, overlap_ratio_prev) >= MIN_INTERSECTION_RATIO:
                    candidates.append((prev_idx, inter_area))
            candidates.sort(key=lambda item: item[1], reverse=True)
            curr_feature["parent_indices"] = [p for p, _ in candidates]

    # Second pass: rebuild child indices only from immediate next-scene parents.
    for scene in ordered_scenes:
        for feature in scene["features"]:
            feature["child_indices"] = []
    for idx in range(1, len(ordered_scenes)):
        prev_features = ordered_scenes[idx - 1]["features"]
        curr_features = ordered_scenes[idx]["features"]
        for prev_idx, prev_feature in enumerate(prev_features):
            child_indices = [
                curr_idx
                for curr_idx, curr_feature in enumerate(curr_features)
                if prev_idx in [int(v) for v in curr_feature.get("parent_indices", [])]
            ]
            prev_feature["child_indices"] = child_indices


def build_skip_scene_topology_edges(ordered_scenes):
    """
    Per-feature forward bridging:
    if a feature has no child in the immediate next scene, try linking to the
    scene after next by intersection.
    """
    if not ordered_scenes or len(ordered_scenes) < 3:
        return

    def has_immediate_path(start_scene_idx, start_feature_idx, target_scene_idx, target_feature_idx):
        """
        Return True if target can already be reached from start using only
        immediate-scene child_indices hops.
        """
        if target_scene_idx <= start_scene_idx:
            return False
        current_feature_indices = {int(start_feature_idx)}
        for scene_idx in range(start_scene_idx, target_scene_idx):
            if not current_feature_indices:
                return False
            scene_features = ordered_scenes[scene_idx]["features"]
            next_features = ordered_scenes[scene_idx + 1]["features"]
            next_feature_indices = set()
            for feat_idx in current_feature_indices:
                if feat_idx < 0 or feat_idx >= len(scene_features):
                    continue
                child_indices = [int(v) for v in scene_features[feat_idx].get("child_indices", [])]
                for child_idx in child_indices:
                    if 0 <= child_idx < len(next_features):
                        next_feature_indices.add(child_idx)
            current_feature_indices = next_feature_indices
        return int(target_feature_idx) in current_feature_indices

    for scene in ordered_scenes:
        for feature in scene["features"]:
            feature["bridge_parent_keys"] = []
            feature["bridge_child_keys"] = []

    for idx in range(0, len(ordered_scenes) - 2):
        curr_scene = ordered_scenes[idx]
        next_scene = ordered_scenes[idx + 1]
        curr_features = curr_scene["features"]

        for curr_idx, curr_feature in enumerate(curr_features):
            # Immediate child exists -> keep normal topology only.
            if curr_feature.get("child_indices"):
                continue
            curr_geom = curr_feature["geom"]
            if curr_geom is None or curr_geom.is_empty:
                continue

            for ahead in range(2, int(MAX_FORWARD_SCENE_LOOKAHEAD) + 1):
                target_scene_idx = idx + ahead
                if target_scene_idx >= len(ordered_scenes):
                    break
                target_scene = ordered_scenes[target_scene_idx]
                target_features = target_scene["features"]

                candidates = []
                for target_idx, target_feature in enumerate(target_features):
                    target_geom = target_feature["geom"]
                    if target_geom is None or target_geom.is_empty:
                        continue
                    if not curr_geom.intersects(target_geom):
                        continue
                    inter_area = curr_geom.intersection(target_geom).area
                    if inter_area <= 0:
                        continue
                    overlap_ratio_curr = inter_area / max(curr_geom.area, 1e-12)
                    overlap_ratio_target = inter_area / max(target_geom.area, 1e-12)
                    if max(overlap_ratio_curr, overlap_ratio_target) >= MIN_INTERSECTION_RATIO:
                        candidates.append((target_idx, inter_area))

                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[1], reverse=True)
                chosen_target_idx = None
                for candidate_target_idx, _candidate_score in candidates:
                    # Avoid redundant bridge edges when normal immediate topology
                    # already provides a path to the same target.
                    if has_immediate_path(idx, curr_idx, target_scene_idx, int(candidate_target_idx)):
                        continue
                    chosen_target_idx = int(candidate_target_idx)
                    break
                if chosen_target_idx is None:
                    # All candidates are already reachable by immediate links.
                    continue

                parent_key = (str(curr_scene["date"]), str(curr_scene["preview_name"]), int(curr_idx))
                child_key = (str(target_scene["date"]), str(target_scene["preview_name"]), int(chosen_target_idx))

                bridge_children = list(curr_feature.get("bridge_child_keys", []))
                if child_key not in bridge_children:
                    bridge_children.append(child_key)
                curr_feature["bridge_child_keys"] = bridge_children

                child_feature = target_features[chosen_target_idx]
                bridge_parents = list(child_feature.get("bridge_parent_keys", []))
                if parent_key not in bridge_parents:
                    bridge_parents.append(parent_key)
                child_feature["bridge_parent_keys"] = bridge_parents
                # Use nearest future scene that matches.
                break


def build_merge_split_corridor_edges(ordered_scenes):
    """
    Bridge branch identity across a merged corridor:
    2-node scene -> one or more 1-node scenes -> 2-node scene.

    We keep the normal immediate topology through the merged corridor, but also
    add direct bridge edges from the pre-merge pair to the post-split pair so
    branch continuity can be recovered by spatial intersection.
    """
    if not ordered_scenes or len(ordered_scenes) < 3:
        return

    def overlap_score(src_feature, dst_feature):
        src_geom = src_feature.get("geom")
        dst_geom = dst_feature.get("geom")
        if src_geom is None or dst_geom is None or src_geom.is_empty or dst_geom.is_empty:
            return 0.0
        if not src_geom.intersects(dst_geom):
            return 0.0
        inter_area = src_geom.intersection(dst_geom).area
        if inter_area <= 0:
            return 0.0
        overlap_ratio_src = inter_area / max(src_geom.area, 1e-12)
        overlap_ratio_dst = inter_area / max(dst_geom.area, 1e-12)
        if max(overlap_ratio_src, overlap_ratio_dst) < MIN_INTERSECTION_RATIO:
            return 0.0
        return float(inter_area)

    idx = 0
    while idx < len(ordered_scenes) - 2:
        start_scene = ordered_scenes[idx]
        start_features = start_scene["features"]
        if len(start_features) != 2:
            idx += 1
            continue

        corridor_end = idx + 1
        while corridor_end < len(ordered_scenes) and len(ordered_scenes[corridor_end]["features"]) == 1:
            corridor_end += 1
        if corridor_end >= len(ordered_scenes):
            break

        end_scene = ordered_scenes[corridor_end]
        end_features = end_scene["features"]
        if corridor_end == idx + 1 or len(end_features) != 2:
            idx += 1
            continue

        score_00 = overlap_score(start_features[0], end_features[0])
        score_01 = overlap_score(start_features[0], end_features[1])
        score_10 = overlap_score(start_features[1], end_features[0])
        score_11 = overlap_score(start_features[1], end_features[1])

        same_total = score_00 + score_11
        cross_total = score_01 + score_10
        if max(same_total, cross_total) <= 0:
            idx = corridor_end
            continue

        if same_total >= cross_total:
            chosen_pairs = [(0, 0, score_00), (1, 1, score_11)]
        else:
            chosen_pairs = [(0, 1, score_01), (1, 0, score_10)]

        for start_idx, end_idx, score in chosen_pairs:
            if score <= 0:
                continue

            parent_key = (str(start_scene["date"]), str(start_scene["preview_name"]), int(start_idx))
            child_key = (str(end_scene["date"]), str(end_scene["preview_name"]), int(end_idx))

            start_feature = start_features[start_idx]
            match_children = list(start_feature.get("branch_match_child_keys", []))
            if child_key not in match_children:
                match_children.append(child_key)
            start_feature["branch_match_child_keys"] = match_children

            end_feature = end_features[end_idx]
            match_parents = list(end_feature.get("branch_match_parent_keys", []))
            if parent_key not in match_parents:
                match_parents.append(parent_key)
            end_feature["branch_match_parent_keys"] = match_parents

        idx = corridor_end


def build_lake_topology(scene_records):
    if not scene_records:
        return []
    ordered = sorted(scene_records, key=lambda item: (item["date_dt"], item["preview_name"]))
    for idx in range(1, len(ordered)):
        map_feature_links(ordered[idx - 1]["features"], ordered[idx]["features"])
    repair_parent_child_indices(ordered)
    build_skip_scene_topology_edges(ordered)
    build_merge_split_corridor_edges(ordered)

    peak_scene = max(
        ordered,
        key=lambda item: (
            float(item.get("scene_total_area_km2", 0.0)),
            -item["date_dt"].value if pd.notna(item["date_dt"]) else 0,
            item["preview_name"],
        ),
    )
    if not peak_scene["features"]:
        return ordered

    main_seed_idx = max(range(len(peak_scene["features"])), key=lambda i: peak_scene["features"][i]["area_km2"])
    peak_scene["features"][main_seed_idx]["is_main_lineage"] = True
    peak_pos = ordered.index(peak_scene)
    current_main_idx = int(main_seed_idx)

    for idx in range(peak_pos, 0, -1):
        curr_features = ordered[idx]["features"]
        prev_features = ordered[idx - 1]["features"]
        if not (0 <= current_main_idx < len(curr_features)):
            break
        curr_feature = curr_features[current_main_idx]
        parent_indices = [parent_idx for parent_idx in curr_feature.get("parent_indices", []) if 0 <= parent_idx < len(prev_features)]
        if not parent_indices:
            break
        chosen_parent_idx = int(parent_indices[0])
        prev_features[chosen_parent_idx]["is_main_lineage"] = True
        current_main_idx = chosen_parent_idx

    current_main_idx = int(main_seed_idx)
    for idx in range(peak_pos, len(ordered) - 1):
        curr_features = ordered[idx]["features"]
        next_features = ordered[idx + 1]["features"]
        if not (0 <= current_main_idx < len(curr_features)):
            break
        candidate_children = []
        for next_idx, next_feature in enumerate(next_features):
            if current_main_idx in next_feature.get("parent_indices", []):
                candidate_children.append((next_idx, float(next_feature["area_km2"])))
        if not candidate_children:
            break
        chosen_child_idx = max(candidate_children, key=lambda item: (item[1], -item[0]))[0]
        next_features[chosen_child_idx]["is_main_lineage"] = True
        current_main_idx = int(chosen_child_idx)
    return ordered


def flatten_feature_rows(scene_records):
    rows = []
    for scene in scene_records:
        for feature in scene["features"]:
            rows.append(
                {
                    "lake_id": int(scene["lake_id"]),
                    "date": str(scene["date"]),
                    "preview_name": str(scene["preview_name"]),
                    "date_dt": scene["date_dt"],
                    "size_category_04_8": str(scene["size_category_04_8"]),
                    "scene_total_area_km2": float(scene["scene_total_area_km2"]),
                    "scene_boundary_bucket": str(scene.get("scene_boundary_bucket", "")),
                    "scene_feature_count": int(scene["feature_count"]),
                    "feature_index": int(feature["feature_index"]),
                    "source_feature_idx": int(feature["source_feature_idx"]),
                    "area_km2": float(feature["area_km2"]),
                    "parent_indices": ";".join(str(v) for v in feature.get("parent_indices", [])),
                    "child_indices": ";".join(str(v) for v in feature.get("child_indices", [])),
                    "bridge_parent_keys": ";".join(encode_node_key(v) for v in feature.get("bridge_parent_keys", [])),
                    "bridge_child_keys": ";".join(encode_node_key(v) for v in feature.get("bridge_child_keys", [])),
                    "branch_match_parent_keys": ";".join(encode_node_key(v) for v in feature.get("branch_match_parent_keys", [])),
                    "branch_match_child_keys": ";".join(encode_node_key(v) for v in feature.get("branch_match_child_keys", [])),
                    "parent_count": int(len(feature.get("parent_indices", []))),
                    "child_count": int(len(feature.get("child_indices", []))),
                    "is_main_lineage": bool(feature.get("is_main_lineage", False)),
                    "centroid_x": float(feature["centroid_x"]),
                    "centroid_y": float(feature["centroid_y"]),
                    "geometry": feature["geom"],
                }
            )
    return rows


def compute_tree_layout(topology_df):
    children_map = defaultdict(list)
    parent_map = defaultdict(list)
    node_rows = {}

    for _, row in topology_df.iterrows():
        node_key = get_node_key(row)
        node_rows[node_key] = row

    ordered_scenes = (
        topology_df[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
    )
    scene_ids = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    scene_to_prev = {scene_ids[idx]: scene_ids[idx - 1] for idx in range(1, len(scene_ids))}

    for _, row in topology_df.iterrows():
        node_key = get_node_key(row)
        current_scene = (node_key[0], node_key[1])
        prev_scene = scene_to_prev.get(current_scene)
        valid_parents = []
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], parent_idx)
                if parent_key in node_rows:
                    valid_parents.append(parent_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in node_rows:
                valid_parents.append(parent_key)
        dedup_parents = []
        seen_parent = set()
        for parent_key in valid_parents:
            if parent_key in seen_parent:
                continue
            seen_parent.add(parent_key)
            dedup_parents.append(parent_key)
        for parent_key in dedup_parents:
            children_map[parent_key].append(node_key)
            parent_map[node_key].append(parent_key)

    roots = [node_key for node_key in node_rows.keys() if len(parent_map.get(node_key, [])) == 0]
    roots.sort(key=lambda key: (node_rows[key]["date"], -float(node_rows[key]["area_km2"]), key[1]))
    for parent_key, child_list in children_map.items():
        child_list.sort(key=lambda key: (node_rows[key]["date"], -float(node_rows[key]["area_km2"]), key[1]))
    return roots, children_map, parent_map, node_rows


def build_isolated_node_mask(lake_features):
    if lake_features.empty:
        return pd.Series(dtype=bool)
    parent_count = pd.to_numeric(lake_features.get("parent_count", 0), errors="coerce").fillna(0)
    child_count = pd.to_numeric(lake_features.get("child_count", 0), errors="coerce").fillna(0)
    bridge_parent_count = lake_features.get("bridge_parent_keys", "").fillna("").astype(str).map(
        lambda text: len(parse_semicolon_node_keys(text))
    )
    bridge_child_count = lake_features.get("bridge_child_keys", "").fillna("").astype(str).map(
        lambda text: len(parse_semicolon_node_keys(text))
    )
    parent_count = parent_count + bridge_parent_count
    child_count = child_count + bridge_child_count
    return (parent_count <= 0) & (child_count <= 0)


def build_leaf_endpoint_mask(lake_features):
    if lake_features.empty:
        return pd.Series(dtype=bool)
    parent_count = pd.to_numeric(lake_features.get("parent_count", 0), errors="coerce").fillna(0)
    child_count = pd.to_numeric(lake_features.get("child_count", 0), errors="coerce").fillna(0)
    bridge_parent_count = lake_features.get("bridge_parent_keys", "").fillna("").astype(str).map(
        lambda text: len(parse_semicolon_node_keys(text))
    )
    bridge_child_count = lake_features.get("bridge_child_keys", "").fillna("").astype(str).map(
        lambda text: len(parse_semicolon_node_keys(text))
    )
    parent_count = parent_count + bridge_parent_count
    child_count = child_count + bridge_child_count
    degree = parent_count + child_count
    return degree == 1


def build_largest_connected_component_mask(lake_features):
    if lake_features.empty:
        return pd.Series(dtype=bool)

    work = lake_features.copy()
    node_keys = [
        (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        for _, row in work.iterrows()
    ]
    key_set = set(node_keys)
    adjacency = {key: set() for key in node_keys}

    ordered_scenes = (
        work[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    parent_scene_lookup = {}
    for idx in range(1, len(scene_keys)):
        parent_scene_lookup[scene_keys[idx]] = scene_keys[idx - 1]

    for _, row in work.iterrows():
        node_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        prev_scene = parent_scene_lookup.get((str(row["date"]), str(row["preview_name"])))
        if prev_scene is None:
            continue
        for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
            parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
            if parent_key in key_set:
                adjacency[node_key].add(parent_key)
                adjacency[parent_key].add(node_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in key_set:
                adjacency[node_key].add(parent_key)
                adjacency[parent_key].add(node_key)

    visited = set()
    components = []
    for start_key in node_keys:
        if start_key in visited:
            continue
        stack = [start_key]
        component = []
        visited.add(start_key)
        while stack:
            curr_key = stack.pop()
            component.append(curr_key)
            for next_key in adjacency.get(curr_key, set()):
                if next_key not in visited:
                    visited.add(next_key)
                    stack.append(next_key)
        components.append(component)

    if not components:
        return pd.Series(False, index=work.index)

    area_lookup = {
        (str(row["date"]), str(row["preview_name"]), int(row["feature_index"])): float(pd.to_numeric(row["area_km2"], errors="coerce"))
        for _, row in work.iterrows()
    }
    largest_component = max(
        components,
        key=lambda comp: (
            len(comp),
            sum(area_lookup.get(key, 0.0) for key in comp),
        ),
    )
    largest_set = set(largest_component)
    return pd.Series([key in largest_set for key in node_keys], index=work.index)


def build_small_disconnected_component_mask(lake_features):
    if lake_features.empty:
        return pd.Series(dtype=bool)

    work = lake_features.copy()
    node_keys = [
        (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        for _, row in work.iterrows()
    ]
    key_set = set(node_keys)
    adjacency = {key: set() for key in node_keys}

    ordered_scenes = (
        work[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    parent_scene_lookup = {}
    for idx in range(1, len(scene_keys)):
        parent_scene_lookup[scene_keys[idx]] = scene_keys[idx - 1]

    for _, row in work.iterrows():
        node_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        prev_scene = parent_scene_lookup.get((str(row["date"]), str(row["preview_name"])))
        if prev_scene is not None:
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                if parent_key in key_set:
                    adjacency[node_key].add(parent_key)
                    adjacency[parent_key].add(node_key)
        for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
            if parent_key in key_set:
                adjacency[node_key].add(parent_key)
                adjacency[parent_key].add(node_key)

    visited = set()
    components = []
    for start_key in node_keys:
        if start_key in visited:
            continue
        stack = [start_key]
        comp = []
        visited.add(start_key)
        while stack:
            curr_key = stack.pop()
            comp.append(curr_key)
            for next_key in adjacency.get(curr_key, set()):
                if next_key not in visited:
                    visited.add(next_key)
                    stack.append(next_key)
        components.append(comp)

    if not components:
        return pd.Series(False, index=work.index)

    main_keys = set(
        (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
        for _, row in work[work["is_main_lineage"] == True].iterrows()
    )

    remove_keys = set()
    for comp in components:
        comp_set = set(comp)
        if comp_set & main_keys:
            continue
        remove_keys.update(comp_set)

    return pd.Series([key in remove_keys for key in node_keys], index=work.index)


def build_topology_context(lake_features):
    work = lake_features.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    roots, children_map, parent_map, node_rows = compute_tree_layout(work)
    ordered_scenes = (
        work[["date", "preview_name", "date_dt"]]
        .drop_duplicates()
        .sort_values(["date_dt", "preview_name"])
        .reset_index(drop=True)
    )
    scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
    scene_node_map = defaultdict(list)
    for _, row in work.iterrows():
        node_key = get_node_key(row)
        scene_node_map[(str(row["date"]), str(row["preview_name"]))].append(node_key)
    return {
        "work": work,
        "roots": roots,
        "children_map": children_map,
        "parent_map": parent_map,
        "node_rows": node_rows,
        "scene_keys": scene_keys,
        "scene_node_map": scene_node_map,
    }


def trace_linear_branch_backward(start_key, parent_map, children_map):
    branch_nodes = []
    current_key = start_key
    seen = set()
    is_valid = True
    while current_key is not None and current_key not in seen:
        seen.add(current_key)
        branch_nodes.append(current_key)
        if len(parent_map.get(current_key, [])) > 1:
            is_valid = False
            break
        parent_keys = list(parent_map.get(current_key, []))
        if not parent_keys:
            break
        parent_key = parent_keys[0]
        if len(children_map.get(parent_key, [])) > 1:
            is_valid = False
            break
        current_key = parent_key
    return branch_nodes, is_valid


def trace_linear_branch_forward(start_key, children_map, parent_map):
    branch_nodes = []
    current_key = start_key
    seen = set()
    is_valid = True
    while current_key is not None and current_key not in seen:
        seen.add(current_key)
        branch_nodes.append(current_key)
        if len(children_map.get(current_key, [])) > 1:
            is_valid = False
            break
        child_keys = list(children_map.get(current_key, []))
        if not child_keys:
            break
        child_key = child_keys[0]
        if len(parent_map.get(child_key, [])) > 1:
            is_valid = False
            break
        current_key = child_key
    return branch_nodes, is_valid


def trace_postsplit_branch_forward(start_key, children_map, parent_map):
    branch_nodes = []
    current_key = start_key
    seen = set()
    while current_key is not None and current_key not in seen:
        seen.add(current_key)
        branch_nodes.append(current_key)
        child_keys = list(children_map.get(current_key, []))
        if not child_keys:
            break
        if len(child_keys) > 1:
            break
        child_key = child_keys[0]
        if len(parent_map.get(child_key, [])) > 1:
            break
        current_key = child_key
    return branch_nodes


def trace_branch_until_event_forward(start_key, children_map, parent_map):
    branch_nodes = []
    current_key = start_key
    seen = set()
    while current_key is not None and current_key not in seen:
        seen.add(current_key)
        branch_nodes.append(current_key)
        child_keys = list(children_map.get(current_key, []))
        if not child_keys:
            return branch_nodes, None, "end"
        if len(child_keys) > 1:
            return branch_nodes, current_key, "split"
        next_key = child_keys[0]
        if len(parent_map.get(next_key, [])) > 1:
            return branch_nodes, next_key, "merge"
        current_key = next_key
    return branch_nodes, None, "loop"


def has_merge_before_node(start_key, parent_map, node_rows):
    current_key = start_key
    seen = set()
    while current_key is not None and current_key not in seen:
        seen.add(current_key)
        # Only count merges that still exist in the current filtered topology.
        if len(parent_map.get(current_key, [])) >= 2:
            return True
        parent_keys = list(parent_map.get(current_key, []))
        if len(parent_keys) != 1:
            return False
        current_key = parent_keys[0]
    return False


def build_instant_split_merge_mask(lake_features):
    if lake_features.empty:
        return pd.Series(dtype=bool)
    context = build_topology_context(lake_features)
    remove_keys = set()
    scene_keys = context["scene_keys"]
    scene_node_map = context["scene_node_map"]
    parent_map = context["parent_map"]
    children_map = context["children_map"]

    idx = 1
    while idx < len(scene_keys) - 1:
        prev_scene = scene_keys[idx - 1]
        if len(scene_node_map.get(prev_scene, [])) != 1:
            idx += 1
            continue
        run_start = idx
        while idx < len(scene_keys) - 1 and len(scene_node_map.get(scene_keys[idx], [])) == 2:
            idx += 1
        run_end = idx - 1
        next_scene = scene_keys[idx] if idx < len(scene_keys) else None
        run_len = run_end - run_start + 1
        if run_end < run_start or next_scene is None or len(scene_node_map.get(next_scene, [])) != 1:
            if run_end < run_start:
                idx = run_start + 1
            continue
        if run_len < 1:
            continue
        # Treat only very short 1->2->1 episodes as instant split-merge noise.
        # Longer 2-runs are more likely real branch persistence with occasional
        # single-scene misses, so we keep them.
        if run_len > int(MAX_INSTANT_SPLIT_MERGE_RUN_LEN):
            continue

        prev_key = scene_node_map[prev_scene][0]
        next_key = scene_node_map[next_scene][0]
        run_scenes = scene_keys[run_start:run_end + 1]
        run_valid = True
        for scene_pos, scene_key in enumerate(run_scenes):
            curr_nodes = scene_node_map.get(scene_key, [])
            if len(curr_nodes) != 2:
                run_valid = False
                break
            if scene_pos == 0:
                if not all(prev_key in parent_map.get(curr_key, []) for curr_key in curr_nodes):
                    run_valid = False
                    break
            if scene_pos == len(run_scenes) - 1:
                if not all(next_key in children_map.get(curr_key, []) for curr_key in curr_nodes):
                    run_valid = False
                    break
        if not run_valid:
            continue

        for scene_key in run_scenes:
            curr_nodes = list(scene_node_map.get(scene_key, []))
            minor_key = min(curr_nodes, key=lambda key: (float(context["node_rows"][key]["area_km2"]), key[2]))
            remove_keys.add(minor_key)

    # 2-1-2 singleton suppression:
    # when one scene collapses to a single node between two dual-node scenes,
    # treat that middle singleton as instant split-merge noise so branches stay
    # visually continuous and clearer.
    for idx in range(1, len(scene_keys) - 1):
        prev_scene = scene_keys[idx - 1]
        curr_scene = scene_keys[idx]
        next_scene = scene_keys[idx + 1]
        prev_nodes = list(scene_node_map.get(prev_scene, []))
        curr_nodes = list(scene_node_map.get(curr_scene, []))
        next_nodes = list(scene_node_map.get(next_scene, []))
        if len(prev_nodes) != 2 or len(curr_nodes) != 1 or len(next_nodes) != 2:
            continue

        curr_key = curr_nodes[0]
        curr_parents = set(parent_map.get(curr_key, []))
        curr_children = set(children_map.get(curr_key, []))

        # Require clear connectivity to both sides to avoid suppressing
        # genuinely isolated one-node scenes.
        if all(node in curr_parents for node in prev_nodes) and all(node in curr_children for node in next_nodes):
            remove_keys.add(curr_key)

    return pd.Series([get_node_key(row) in remove_keys for _, row in lake_features.iterrows()], index=lake_features.index)


def build_short_premerge_branch_mask(lake_features, min_branch_nodes=MIN_EVENT_BRANCH_NODES):
    if lake_features.empty:
        return pd.Series(dtype=bool)
    context = build_topology_context(lake_features)
    remove_keys = set()
    parent_map = context["parent_map"]
    children_map = context["children_map"]
    node_rows = context["node_rows"]

    for merge_key, _merge_row in node_rows.items():
        if len(parent_map.get(merge_key, [])) < 2:
            continue
        parent_keys = list(parent_map.get(merge_key, []))
        if len(parent_keys) < 2:
            continue
        branch_infos = []
        for parent_key in parent_keys:
            branch_nodes, is_valid = trace_linear_branch_backward(parent_key, parent_map, children_map)
            if len(branch_nodes) <= 0:
                continue
            branch_area = sum(float(node_rows[key]["area_km2"]) for key in branch_nodes)
            branch_infos.append((parent_key, branch_nodes, branch_area, is_valid))
        if len(branch_infos) < 2:
            continue
        keep_parent_key = max(branch_infos, key=lambda item: (len(item[1]), item[2], item[0][2]))[0]
        for parent_key, branch_nodes, _, is_valid in branch_infos:
            if parent_key == keep_parent_key:
                continue
            if is_valid and len(branch_nodes) < int(min_branch_nodes):
                remove_keys.update(branch_nodes)

    return pd.Series([get_node_key(row) in remove_keys for _, row in lake_features.iterrows()], index=lake_features.index)


def build_short_postsplit_branch_mask(lake_features, min_branch_nodes=MIN_EVENT_BRANCH_NODES):
    if lake_features.empty:
        return pd.Series(dtype=bool)
    context = build_topology_context(lake_features)
    remove_keys = set()
    parent_map = context["parent_map"]
    children_map = context["children_map"]
    node_rows = context["node_rows"]

    for split_key, _split_row in node_rows.items():
        if len(children_map.get(split_key, [])) < 2:
            continue
        child_keys = list(children_map.get(split_key, []))
        if len(child_keys) < 2:
            continue
        branch_infos = []
        for child_key in child_keys:
            branch_nodes = trace_postsplit_branch_forward(child_key, children_map, parent_map)
            if len(branch_nodes) <= 0:
                continue
            branch_area = sum(float(node_rows[key]["area_km2"]) for key in branch_nodes)
            branch_infos.append((child_key, branch_nodes, branch_area))
        if len(branch_infos) < 2:
            continue
        has_prior_merge = has_merge_before_node(split_key, parent_map, node_rows)
        if has_prior_merge:
            keep_child_key = max(branch_infos, key=lambda item: (len(item[1]), item[2], item[0][2]))[0]
        else:
            keep_child_key = max(branch_infos, key=lambda item: (item[2], len(item[1]), item[0][2]))[0]
        for child_key, branch_nodes, _, in branch_infos:
            if child_key == keep_child_key:
                continue
            # If this split has no prior merge upstream, treat the shorter side
            # branch as noise even when it persists for many scenes.
            if not has_prior_merge:
                remove_keys.update(branch_nodes)
                continue
            # After an upstream merge, only remove truly short postsplit branches.
            if len(branch_nodes) < int(min_branch_nodes):
                remove_keys.update(branch_nodes)

    return pd.Series([get_node_key(row) in remove_keys for _, row in lake_features.iterrows()], index=lake_features.index)


def build_split_merge_minor_branch_mask(lake_features):
    if lake_features.empty:
        return pd.Series(dtype=bool)

    context = build_topology_context(lake_features)
    work = context["work"]
    parent_map = context["parent_map"]
    children_map = context["children_map"]
    node_rows = context["node_rows"]
    remove_keys = set()

    for split_key in node_rows.keys():
        child_keys = list(children_map.get(split_key, []))
        if len(child_keys) != 2:
            continue

        branch_infos = []
        for child_key in child_keys:
            branch_nodes, terminal_key, terminal_type = trace_branch_until_event_forward(
                child_key, children_map, parent_map
            )
            if not branch_nodes:
                continue
            branch_area = sum(float(node_rows[key]["area_km2"]) for key in branch_nodes)
            branch_infos.append({
                "child_key": child_key,
                "branch_nodes": branch_nodes,
                "terminal_key": terminal_key,
                "terminal_type": terminal_type,
                "branch_area": branch_area,
            })

        if len(branch_infos) != 2:
            continue
        if any(info["terminal_type"] != "merge" for info in branch_infos):
            continue

        merge_keys = {info["terminal_key"] for info in branch_infos}
        if len(merge_keys) != 1:
            continue
        merge_key = next(iter(merge_keys))
        if merge_key is None or len(parent_map.get(merge_key, [])) < 2:
            continue

        keep_info = max(branch_infos, key=lambda info: (info["branch_area"], len(info["branch_nodes"]), info["child_key"][2]))
        for info in branch_infos:
            if info["child_key"] == keep_info["child_key"]:
                continue
            remove_keys.update(info["branch_nodes"])

    return pd.Series([get_node_key(row) in remove_keys for _, row in work.iterrows()], index=work.index)


def build_display_filter_masks(lake_features):
    if lake_features.empty:
        empty_mask = pd.Series(dtype=bool)
        return {
            "isolated_mask": empty_mask,
            "secondary_component_mask": empty_mask,
            "instant_split_merge_mask": empty_mask,
            "short_premerge_branch_mask": empty_mask,
            "short_postsplit_branch_mask": empty_mask,
            "split_merge_minor_branch_mask": empty_mask,
            "kept_mask": empty_mask,
            "kept_df": lake_features.copy(),
        }

    work = lake_features.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
    isolated_mask = build_isolated_node_mask(work).reindex(work.index, fill_value=False)
    non_isolated = work[~isolated_mask].copy()
    secondary_component_mask = pd.Series(False, index=work.index)
    if ENABLE_SECONDARY_COMPONENT_FILTER:
        component_mask = (
            build_largest_connected_component_mask(non_isolated).reindex(non_isolated.index, fill_value=False)
            if not non_isolated.empty
            else pd.Series(dtype=bool)
        )
        if not non_isolated.empty:
            secondary_component_mask.loc[non_isolated.index] = ~component_mask
    if ENABLE_SMALL_DISCONNECTED_COMPONENT_FILTER:
        small_component_mask = (
            build_small_disconnected_component_mask(non_isolated).reindex(non_isolated.index, fill_value=False)
            if not non_isolated.empty
            else pd.Series(dtype=bool)
        )
        if not non_isolated.empty:
            secondary_component_mask.loc[non_isolated.index] = (
                secondary_component_mask.loc[non_isolated.index] | small_component_mask
            )
    component_df = (
        non_isolated[~secondary_component_mask.loc[non_isolated.index]].copy()
        if not non_isolated.empty
        else work.iloc[0:0].copy()
    )

    instant_split_merge_mask = pd.Series(False, index=work.index)
    short_premerge_branch_mask = pd.Series(False, index=work.index)
    short_postsplit_branch_mask = pd.Series(False, index=work.index)
    split_merge_minor_branch_mask = pd.Series(False, index=work.index)
    if not component_df.empty:
        premerge_base_df = component_df.copy()
        if not premerge_base_df.empty:
            short_premerge_branch_mask.loc[premerge_base_df.index] = build_short_premerge_branch_mask(premerge_base_df).reindex(premerge_base_df.index, fill_value=False)
        # Rebuild topology after each postsplit removal batch so later splits are
        # judged against the remaining structure rather than the original one.
        while True:
            postsplit_base_df = component_df[
                ~(short_premerge_branch_mask.loc[component_df.index] | short_postsplit_branch_mask.loc[component_df.index])
            ].copy()
            if postsplit_base_df.empty:
                break
            iter_mask = build_short_postsplit_branch_mask(postsplit_base_df).reindex(postsplit_base_df.index, fill_value=False)
            if not bool(iter_mask.any()):
                break
            new_remove_index = iter_mask[iter_mask].index.difference(
                short_postsplit_branch_mask[short_postsplit_branch_mask].index
            )
            if len(new_remove_index) == 0:
                break
            short_postsplit_branch_mask.loc[new_remove_index] = True
        instant_base_df = component_df[
            ~(short_premerge_branch_mask.loc[component_df.index] | short_postsplit_branch_mask.loc[component_df.index])
        ].copy()
        if not instant_base_df.empty:
            instant_split_merge_mask.loc[instant_base_df.index] = build_instant_split_merge_mask(instant_base_df).reindex(instant_base_df.index, fill_value=False)

        split_merge_base_df = component_df[
            ~(short_premerge_branch_mask.loc[component_df.index] | short_postsplit_branch_mask.loc[component_df.index] | instant_split_merge_mask.loc[component_df.index])
        ].copy()
        if ENABLE_SPLIT_MERGE_MINOR_BRANCH_FILTER and not split_merge_base_df.empty:
            split_merge_minor_branch_mask.loc[split_merge_base_df.index] = (
                build_split_merge_minor_branch_mask(split_merge_base_df).reindex(split_merge_base_df.index, fill_value=False)
            )

    kept_mask = ~(
        isolated_mask
        | secondary_component_mask
        | instant_split_merge_mask
        | short_premerge_branch_mask
        | short_postsplit_branch_mask
        | split_merge_minor_branch_mask
    )
    return {
        "isolated_mask": isolated_mask,
        "secondary_component_mask": secondary_component_mask,
        "instant_split_merge_mask": instant_split_merge_mask,
        "short_premerge_branch_mask": short_premerge_branch_mask,
        "short_postsplit_branch_mask": short_postsplit_branch_mask,
        "split_merge_minor_branch_mask": split_merge_minor_branch_mask,
        "kept_mask": kept_mask,
        "kept_df": work[kept_mask].copy(),
    }


def build_display_status_rows(features_df):
    status_rows = []
    for lake_id, lake_features in features_df.groupby("lake_id", sort=True):
        lake_features = lake_features.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        if lake_features.empty:
            continue

        filter_masks = build_display_filter_masks(lake_features)
        kept_df = filter_masks["kept_df"]
        kept_keys = {
            (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
            for _, row in kept_df.iterrows()
        }
        leaf_endpoint_mask = build_leaf_endpoint_mask(kept_df).reindex(kept_df.index, fill_value=False) if not kept_df.empty else pd.Series(dtype=bool)

        for _, row in lake_features.iterrows():
            node_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
            if node_key in kept_keys:
                display_status = "kept_display"
            elif bool(filter_masks["isolated_mask"].loc[row.name]):
                display_status = "removed_isolated"
            elif bool(filter_masks["secondary_component_mask"].loc[row.name]):
                display_status = "removed_secondary_component"
            elif bool(filter_masks["instant_split_merge_mask"].loc[row.name]):
                display_status = "removed_instant_split_merge"
            elif bool(filter_masks["split_merge_minor_branch_mask"].loc[row.name]):
                display_status = "removed_split_merge_minor_branch"
            elif bool(filter_masks["short_premerge_branch_mask"].loc[row.name]):
                display_status = "removed_short_premerge_branch"
            elif bool(filter_masks["short_postsplit_branch_mask"].loc[row.name]):
                display_status = "removed_short_postsplit_branch"
            else:
                display_status = "removed_other"
            status_rows.append(
                {
                    "lake_id": int(row["lake_id"]),
                    "date": str(row["date"]),
                    "preview_name": str(row["preview_name"]),
                    "feature_index": int(row["feature_index"]),
                    "display_status": display_status,
                    "removed_from_display": display_status != "kept_display",
                    "is_isolated_removed": display_status == "removed_isolated",
                    "is_secondary_component_removed": display_status == "removed_secondary_component",
                    "is_instant_split_merge_removed": display_status == "removed_instant_split_merge",
                    "is_split_merge_minor_branch_removed": display_status == "removed_split_merge_minor_branch",
                    "is_short_premerge_branch_removed": display_status == "removed_short_premerge_branch",
                    "is_short_postsplit_branch_removed": display_status == "removed_short_postsplit_branch",
                    "is_leaf_endpoint": bool(leaf_endpoint_mask.get(row.name, False)) if not kept_df.empty else False,
                }
            )
    return pd.DataFrame(status_rows)


def build_metrics_df(scene_records):
    metric_rows = []
    for scene in scene_records:
        main_features = [feature for feature in scene["features"] if feature.get("is_main_lineage")]
        metric_rows.append(
            {
                "lake_id": int(scene["lake_id"]),
                "date": str(scene["date"]),
                "preview_name": str(scene["preview_name"]),
                "date_dt": scene["date_dt"],
                "size_category_04_8": str(scene["size_category_04_8"]),
                "scene_total_area_km2": float(scene["scene_total_area_km2"]),
                "scene_boundary_bucket": str(scene.get("scene_boundary_bucket", "")),
                "feature_count": int(scene["feature_count"]),
                "tracked_main_area_km2": float(sum(feature["area_km2"] for feature in main_features)),
                "main_feature_count": int(len(main_features)),
                "has_split": bool(any(len(feature.get("child_indices", [])) >= 2 for feature in main_features)),
                "has_merge": bool(any(len(feature.get("parent_indices", [])) >= 2 for feature in main_features)),
            }
        )
    return pd.DataFrame(metric_rows)


def get_scene_bucket_color(bucket):
    bucket_text = str(bucket).strip()
    if bucket_text == BOUNDARY_VALID:
        return VALID_SCENE_COLOR
    if bucket_text == BOUNDARY_INVALID:
        return INVALID_SCENE_COLOR
    return UNKNOWN_SCENE_COLOR


def build_summary_df(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()
    rows = []
    for lake_id, lake_df in metrics_df.groupby("lake_id", sort=True):
        lake_df = lake_df.sort_values(["date_dt", "preview_name"]).copy()
        peak_idx = lake_df["scene_total_area_km2"].idxmax()
        peak_row = lake_df.loc[peak_idx]
        main_peak_idx = lake_df["tracked_main_area_km2"].idxmax()
        main_peak_row = lake_df.loc[main_peak_idx]
        rows.append(
            {
                "lake_id": int(lake_id),
                "size_category_04_8": str(lake_df["size_category_04_8"].iloc[0]),
                "scene_count": int(len(lake_df)),
                "first_date": lake_df["date"].iloc[0],
                "last_date": lake_df["date"].iloc[-1],
                "max_scene_total_area_km2": float(peak_row["scene_total_area_km2"]),
                "max_scene_total_area_date": str(peak_row["date"]),
                "max_tracked_main_area_km2": float(main_peak_row["tracked_main_area_km2"]),
                "max_tracked_main_area_date": str(main_peak_row["date"]),
                "mean_scene_total_area_km2": float(lake_df["scene_total_area_km2"].mean()),
                "mean_tracked_main_area_km2": float(lake_df["tracked_main_area_km2"].mean()),
                "has_any_split": bool(lake_df["has_split"].fillna(False).any()),
                "has_any_merge": bool(lake_df["has_merge"].fillna(False).any()),
            }
        )
    return pd.DataFrame(rows)


def build_branch_summary_base_df(features_df):
    if features_df.empty:
        return pd.DataFrame()

    rows = []
    for lake_id, lake_features in features_df.groupby("lake_id", sort=True):
        lake_features = lake_features.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        main_df = lake_features[lake_features["is_main_lineage"] == True].copy()
        rows.append(
            {
                "lake_id": int(lake_id),
                "size_category_04_8": str(lake_features["size_category_04_8"].iloc[0]),
                "scene_count": int(lake_features[["date", "preview_name"]].drop_duplicates().shape[0]),
                "feature_record_count": int(len(lake_features)),
                "main_lineage_feature_records": int(len(main_df)),
                "branch_feature_records": int(len(lake_features) - len(main_df)),
                "max_feature_area_km2": float(pd.to_numeric(lake_features["area_km2"], errors="coerce").max()),
                "max_main_lineage_area_km2": float(pd.to_numeric(main_df["area_km2"], errors="coerce").max()) if not main_df.empty else 0.0,
                "has_any_split": bool((pd.to_numeric(lake_features["child_count"], errors="coerce").fillna(0) >= 2).any()),
                "has_any_merge": bool((pd.to_numeric(lake_features["parent_count"], errors="coerce").fillna(0) >= 2).any()),
            }
        )
    return pd.DataFrame(rows)


def export_topology_plots(metrics_df, plot_features_df, plots_dir, method_output_dir):
    if os.path.exists(plots_dir):
        import shutil as _shutil
        _shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    grouped_lakes = list(plot_features_df.groupby("lake_id", sort=True))
    total_lakes = len(grouped_lakes)
    for lake_idx, (lake_id, lake_features) in enumerate(grouped_lakes, 1):
        lake_features = lake_features.sort_values(["date_dt", "preview_name", "feature_index"]).copy()
        if lake_features.empty:
            continue
        kept_features = get_precomputed_display_kept_df(lake_features)
        noise_features = lake_features[lake_features["display_status"].fillna("") != "kept_display"].copy()
        valid_features = kept_features[
            kept_features["scene_boundary_bucket"].map(normalize_scene_bucket) == BOUNDARY_VALID
        ].copy()
        if len(valid_features) < 5:
            continue
        lake_df = metrics_df[metrics_df["lake_id"] == int(lake_id)].sort_values(["date_dt", "preview_name"]).copy()
        active_features = kept_features[
            kept_features["scene_boundary_bucket"].map(normalize_scene_bucket) != BOUNDARY_INVALID
        ].copy()
        peak_source = active_features if not active_features.empty else kept_features
        max_area = pd.to_numeric(peak_source["area_km2"], errors="coerce").max()
        size_category = classify_final_curve_size_category(max_area)
        has_blue_points = kept_features["scene_boundary_bucket"].map(normalize_scene_bucket).eq(BOUNDARY_INVALID).any()
        out_dir = os.path.join(plots_dir, "WithBluePoints", size_category) if has_blue_points else os.path.join(plots_dir, size_category)
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"Topology_{int(lake_id)}.png")

        node_lookup = {
            (str(row["date"]), str(row["preview_name"]), int(row["feature_index"])): row
            for _, row in lake_features.iterrows()
        }
        kept_key_set = {
            (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
            for _, row in kept_features.iterrows()
        }
        child_map = defaultdict(list)
        parent_map = defaultdict(list)
        parent_scene_lookup = {}
        ordered_scenes = (
            lake_features[["date", "preview_name", "date_dt"]]
            .drop_duplicates()
            .sort_values(["date_dt", "preview_name"])
            .reset_index(drop=True)
        )
        scene_keys = [(str(row["date"]), str(row["preview_name"])) for _, row in ordered_scenes.iterrows()]
        for idx in range(1, len(scene_keys)):
            parent_scene_lookup[scene_keys[idx]] = scene_keys[idx - 1]
        next_scene_lookup = {prev_scene: curr_scene for curr_scene, prev_scene in parent_scene_lookup.items()}
        for _, row in lake_features.iterrows():
            node_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
            curr_scene = (node_key[0], node_key[1])
            prev_scene = parent_scene_lookup.get(curr_scene)
            if prev_scene is None:
                continue
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                if parent_key in node_lookup:
                    child_map[parent_key].append(node_key)
                    parent_map[node_key].append(parent_key)

        fig, ax = plt.subplots(figsize=(14, 7.5))
        if not lake_df.empty:
            ax.plot(
                lake_df["date_dt"],
                lake_df["scene_total_area_km2"],
                color="lightgray",
                linewidth=1.8,
                alpha=0.7,
                label="Scene total area",
                zorder=1,
            )

        for _, row in lake_features.iterrows():
            row_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
            if row_key not in kept_key_set:
                continue
            curr_key = (str(row["date"]), str(row["preview_name"]))
            prev_scene = parent_scene_lookup.get(curr_key)
            for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                if prev_scene is None:
                    continue
                parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                if parent_key not in node_lookup:
                    continue
                if parent_key not in kept_key_set:
                    continue
                parent_row = node_lookup[parent_key]
                ax.plot(
                    [parent_row.get("x_plot_dt", parent_row["date_dt"]), row.get("x_plot_dt", row["date_dt"])],
                    [parent_row["area_km2"], row["area_km2"]],
                    color=RAW_TOPOLOGY_EDGE_COLOR,
                    linewidth=0.95,
                    alpha=0.65,
                    zorder=2,
                )
            for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
                if parent_key not in node_lookup:
                    continue
                if parent_key not in kept_key_set:
                    continue
                parent_row = node_lookup[parent_key]
                ax.plot(
                    [parent_row.get("x_plot_dt", parent_row["date_dt"]), row.get("x_plot_dt", row["date_dt"])],
                    [parent_row["area_km2"], row["area_km2"]],
                    color=RAW_TOPOLOGY_EDGE_COLOR,
                    linewidth=0.95,
                    alpha=0.65,
                    zorder=2,
                )
        for parent_key, child_key in build_backfill_visual_edges(lake_features, kept_key_set):
            if parent_key not in node_lookup or child_key not in node_lookup:
                continue
            parent_row = node_lookup[parent_key]
            child_row = node_lookup[child_key]
            ax.plot(
                [parent_row.get("x_plot_dt", parent_row["date_dt"]), child_row.get("x_plot_dt", child_row["date_dt"])],
                [parent_row["area_km2"], child_row["area_km2"]],
                color=RAW_TOPOLOGY_EDGE_COLOR,
                linewidth=0.95,
                alpha=0.65,
                zorder=2,
            )

        for branch_id, branch_df in kept_features.groupby("branch_id", dropna=False, sort=False):
            marker = get_branch_marker(branch_id)
            node_colors = [get_scene_bucket_color(value) for value in branch_df.get("scene_boundary_bucket", "")]
            ax.scatter(
                branch_df.get("x_plot_dt", branch_df["date_dt"]),
                branch_df["area_km2"],
                c=node_colors,
                marker=marker,
                s=34,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
        # Label each visible point with branch_id for raw topology debugging.
        for _, row in kept_features.iterrows():
            bid = str(row.get("branch_id", "")).strip()
            if not bid:
                continue
            bid_digits = "".join(ch for ch in bid if ch.isdigit())
            bid_label = (bid_digits.lstrip("0") if bid_digits else bid) or "0"
            ax.annotate(
                bid_label,
                xy=(row.get("x_plot_dt", row["date_dt"]), row["area_km2"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#263238",
                alpha=0.85,
                zorder=5,
            )
        for status_name, status_df in noise_features.groupby("display_status", sort=False):
            color = NOISE_STATUS_COLORS.get(str(status_name), "#455A64")
            ax.scatter(
                status_df.get("x_plot_dt", status_df["date_dt"]),
                status_df["area_km2"],
                c=color,
                marker="x",
                s=52,
                alpha=0.95,
                linewidths=1.4,
                zorder=4,
            )
        peak_idx = peak_source["area_km2"].idxmax()
        peak_row = peak_source.loc[peak_idx]
        ax.annotate(
            f"Peak feature: {float(peak_row['area_km2']):.3f}",
            xy=(peak_row.get("x_plot_dt", peak_row["date_dt"]), peak_row["area_km2"]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=11,
            color="#D73027",
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#D73027"},
        )
        ax.set_title(f"Lake {int(lake_id)} Topology Branch Curve (Raw)", fontsize=20)
        ax.set_xlabel("Date", fontsize=16)
        ax.set_ylabel("Area (km$^2$)", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(handles=[
            Line2D([0], [0], color="lightgray", lw=1.8, label="Scene total area"),
            Line2D([0], [0], color=RAW_TOPOLOGY_EDGE_COLOR, lw=1.0, label="Topology edges"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=VALID_SCENE_COLOR, markeredgecolor="white", markersize=8, label="Valid scene"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=INVALID_SCENE_COLOR, markeredgecolor="white", markersize=8, label="Invalid scene"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=UNKNOWN_SCENE_COLOR, markeredgecolor="white", markersize=8, label="Scene unmatched"),
            Line2D([0], [0], marker="x", color=NOISE_STATUS_COLORS["removed_secondary_component"], linestyle="None", markersize=8, label="Noise: secondary component"),
            Line2D([0], [0], marker="x", color=NOISE_STATUS_COLORS["removed_short_premerge_branch"], linestyle="None", markersize=8, label="Noise: short premerge"),
            Line2D([0], [0], marker="x", color=NOISE_STATUS_COLORS["removed_short_postsplit_branch"], linestyle="None", markersize=8, label="Noise: short postsplit"),
            Line2D([0], [0], marker="x", color=NOISE_STATUS_COLORS["removed_instant_split_merge"], linestyle="None", markersize=8, label="Noise: instant split-merge"),
            Line2D([0], [0], marker="x", color=NOISE_STATUS_COLORS["removed_split_merge_minor_branch"], linestyle="None", markersize=8, label="Noise: split-merge minor branch"),
            Line2D([0], [0], marker="x", color=NOISE_STATUS_COLORS["removed_isolated"], linestyle="None", markersize=8, label="Noise: isolated"),
        ], loc="upper right", fontsize=11, frameon=True)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)

        # Also export one figure per branch so different branches are not mixed in one plot.
        branch_ids = [
            str(value)
            for value in kept_features["branch_id"].dropna().astype(str).unique().tolist()
            if str(value).strip()
        ]
        for branch_id in branch_ids:
            branch_kept = kept_features[kept_features["branch_id"].astype(str) == branch_id].copy()
            if branch_kept.empty:
                continue

            def get_kept_parent_keys(node_key):
                row = node_lookup.get(node_key)
                if row is None:
                    return []
                out = []
                for parent_key in parent_map.get(node_key, []):
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

            selected_keys = {
                (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
                for _, row in branch_kept.iterrows()
            }
            # Shared trunk before split: if this branch peels off from a public line,
            # include that common upstream segment in this branch plot until the
            # previous merge point (or the root).
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

            # Shared trunk: after this branch merges into the public line, include
            # the common segment until the next split point (or the end).
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

            display_kept_keys = selected_keys | shared_upstream_keys | shared_downstream_keys
            branch_plot_df = kept_features[
                kept_features.apply(
                    lambda row: (str(row["date"]), str(row["preview_name"]), int(row["feature_index"])) in display_kept_keys,
                    axis=1,
                )
            ].copy()
            if branch_plot_df.empty:
                continue
            # In multi-node scenes, keep only the points that truly belong to this
            # branch_id. Shared trunk logic is only meant for the 1-node common
            # corridor, not for carrying the sibling branch through a later split.
            multi_scene_selected = {
                (str(row["date"]), str(row["preview_name"]))
                for _, row in branch_plot_df.iterrows()
                if sum(
                    1
                    for key in display_kept_keys
                    if key[0] == str(row["date"]) and key[1] == str(row["preview_name"])
                ) >= 2
                and any(
                    key in selected_keys
                    for key in display_kept_keys
                    if key[0] == str(row["date"]) and key[1] == str(row["preview_name"])
                )
            }
            if multi_scene_selected:
                branch_plot_df = branch_plot_df[
                    branch_plot_df.apply(
                        lambda row: (
                            (str(row["date"]), str(row["preview_name"])) not in multi_scene_selected
                            or (str(row["date"]), str(row["preview_name"]), int(row["feature_index"])) in selected_keys
                        ),
                        axis=1,
                    )
                ].copy()
                if branch_plot_df.empty:
                    continue

            branch_display_kept = branch_plot_df.copy()
            branch_node_lookup = {
                (str(row["date"]), str(row["preview_name"]), int(row["feature_index"])): row
                for _, row in branch_plot_df.iterrows()
            }
            branch_kept_key_set = {
                (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
                for _, row in branch_display_kept.iterrows()
            }
            branch_peak_source = branch_display_kept[
                branch_display_kept["scene_boundary_bucket"].map(normalize_scene_bucket) != BOUNDARY_INVALID
            ].copy()
            branch_peak_source = branch_peak_source if not branch_peak_source.empty else branch_display_kept

            branch_out_dir = os.path.join(method_output_dir, "05_1_SingleBranch", size_category)
            os.makedirs(branch_out_dir, exist_ok=True)
            branch_out_png = os.path.join(branch_out_dir, f"Topology_{int(lake_id)}_{branch_id}.png")

            fig, ax = plt.subplots(figsize=(14, 7.5))
            if not lake_df.empty:
                ax.plot(
                    lake_df["date_dt"],
                    lake_df["scene_total_area_km2"],
                    color="lightgray",
                    linewidth=1.8,
                    alpha=0.7,
                    label="Scene total area",
                    zorder=1,
                )

            for _, row in branch_plot_df.iterrows():
                row_key = (str(row["date"]), str(row["preview_name"]), int(row["feature_index"]))
                if row_key not in branch_kept_key_set:
                    continue
                curr_key = (str(row["date"]), str(row["preview_name"]))
                prev_scene = parent_scene_lookup.get(curr_key)
                for parent_idx in parse_semicolon_ints(row.get("parent_indices", "")):
                    if prev_scene is None:
                        continue
                    parent_key = (prev_scene[0], prev_scene[1], int(parent_idx))
                    if parent_key not in branch_node_lookup:
                        continue
                    if parent_key not in branch_kept_key_set:
                        continue
                    parent_row = branch_node_lookup[parent_key]
                    ax.plot(
                        [parent_row.get("x_plot_dt", parent_row["date_dt"]), row.get("x_plot_dt", row["date_dt"])],
                        [parent_row["area_km2"], row["area_km2"]],
                        color=RAW_TOPOLOGY_EDGE_COLOR,
                        linewidth=0.95,
                        alpha=0.65,
                        zorder=2,
                    )
                for parent_key in parse_semicolon_node_keys(row.get("bridge_parent_keys", "")):
                    if parent_key not in branch_node_lookup:
                        continue
                    if parent_key not in branch_kept_key_set:
                        continue
                    parent_row = branch_node_lookup[parent_key]
                    ax.plot(
                        [parent_row.get("x_plot_dt", parent_row["date_dt"]), row.get("x_plot_dt", row["date_dt"])],
                        [parent_row["area_km2"], row["area_km2"]],
                        color=RAW_TOPOLOGY_EDGE_COLOR,
                        linewidth=0.95,
                        alpha=0.65,
                        zorder=2,
                    )
            for parent_key, child_key in build_backfill_visual_edges(branch_plot_df, branch_kept_key_set):
                if parent_key not in branch_node_lookup or child_key not in branch_node_lookup:
                    continue
                parent_row = branch_node_lookup[parent_key]
                child_row = branch_node_lookup[child_key]
                ax.plot(
                    [parent_row.get("x_plot_dt", parent_row["date_dt"]), child_row.get("x_plot_dt", child_row["date_dt"])],
                    [parent_row["area_km2"], child_row["area_km2"]],
                    color=RAW_TOPOLOGY_EDGE_COLOR,
                    linewidth=0.95,
                    alpha=0.65,
                    zorder=2,
                )

            node_colors = [get_scene_bucket_color(value) for value in branch_display_kept.get("scene_boundary_bucket", "")]
            ax.scatter(
                branch_display_kept.get("x_plot_dt", branch_display_kept["date_dt"]),
                branch_display_kept["area_km2"],
                c=node_colors,
                marker=get_branch_marker(branch_id),
                s=40,
                alpha=0.86,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )

            peak_idx = branch_peak_source["area_km2"].idxmax()
            peak_row = branch_peak_source.loc[peak_idx]
            ax.annotate(
                f"Peak feature: {float(peak_row['area_km2']):.3f}",
                xy=(peak_row.get("x_plot_dt", peak_row["date_dt"]), peak_row["area_km2"]),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=11,
                color="#D73027",
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#D73027"},
            )
            ax.set_title(f"Lake {int(lake_id)} Topology Branch Curve ({branch_id})", fontsize=20)
            ax.set_xlabel("Date", fontsize=16)
            ax.set_ylabel("Area (km$^2$)", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(handles=[
                Line2D([0], [0], color="lightgray", lw=1.8, label="Scene total area"),
                Line2D([0], [0], color=RAW_TOPOLOGY_EDGE_COLOR, lw=1.0, label="Topology edges"),
                Line2D([0], [0], marker=get_branch_marker(branch_id), color="w", markerfacecolor=VALID_SCENE_COLOR, markeredgecolor="white", markersize=8, label=f"Branch {branch_id}"),
            ], loc="upper right", fontsize=11, frameon=True)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(branch_out_png, dpi=220, bbox_inches="tight")
            plt.close(fig)
        print(f"[05_1] plotted lake {lake_idx}/{total_lakes}: {int(lake_id)}", flush=True)


def process_method(method_name):
    method_paths = get_method_paths(method_name)
    print_run_summary(method_paths)
    os.makedirs(method_paths["output_dir"], exist_ok=True)

    if gpd is None:
        raise ImportError("geopandas is required to run 05_1 from the source GDB")

    area_df = load_area_summary(method_paths["input_area_summary_csv"])
    source_gdf = load_topology_source_gdf(method_paths["input_gdb_path"])
    if area_df.empty or source_gdf.empty:
        print(f"[05_1] skip {method_name}: missing input")
        return

    scene_records = build_scene_records(source_gdf, area_df)
    lake_ids = sorted({record["lake_id"] for record in scene_records})
    print(
        f"[05_1] built scene records: scenes={len(scene_records)}, lakes={len(lake_ids)}",
        flush=True,
    )
    metrics_frames = []
    feature_rows = []
    total_lakes = len(lake_ids)
    for lake_index, lake_id in enumerate(lake_ids, 1):
        lake_scene_records = [record for record in scene_records if record["lake_id"] == lake_id]
        lake_scene_records = build_lake_topology(lake_scene_records)
        metrics_frames.append(build_metrics_df(lake_scene_records))
        feature_rows.extend(flatten_feature_rows(lake_scene_records))
        print(f"[05_1] finished topology lake {lake_index}/{total_lakes}: {int(lake_id)}", flush=True)

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    if not metrics_df.empty:
        metrics_df.sort_values(["lake_id", "date_dt", "preview_name"], inplace=True)
    features_gdf = gpd.GeoDataFrame(feature_rows, geometry="geometry", crs=TARGET_CRS) if feature_rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)
    feature_export_df = pd.DataFrame(feature_rows)
    if not feature_export_df.empty:
        status_df = build_display_status_rows(feature_export_df)
        feature_export_df = feature_export_df.merge(
            status_df,
            on=["lake_id", "date", "preview_name", "feature_index"],
            how="left",
        )
        features_gdf = features_gdf.merge(
            status_df,
            on=["lake_id", "date", "preview_name", "feature_index"],
            how="left",
        )
    summary_df = build_summary_df(metrics_df)
    branch_summary_base_df = build_branch_summary_base_df(feature_export_df)
    raw_plot_features_df = prepare_raw_plot_features_df(feature_export_df)

    metrics_df.to_csv(method_paths["metrics_csv"], index=False, encoding="utf-8-sig")
    if not feature_export_df.empty:
        feature_export_df.drop(columns=["geometry"], inplace=True)
    feature_export_df.to_csv(method_paths["features_csv"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(method_paths["summary_csv"], index=False, encoding="utf-8-sig")
    branch_summary_base_df.to_csv(method_paths["branch_summary_base_csv"], index=False, encoding="utf-8-sig")
    raw_plot_export_df = raw_plot_features_df.copy()
    if "geometry" in raw_plot_export_df.columns:
        raw_plot_export_df = raw_plot_export_df.drop(columns=["geometry"])
    raw_plot_export_df.to_csv(method_paths["raw_plot_features_csv"], index=False, encoding="utf-8-sig")

    if os.path.exists(method_paths["output_gdb_path"]):
        import shutil as _shutil
        _shutil.rmtree(method_paths["output_gdb_path"])
    if not features_gdf.empty:
        print("[05_1] writing topology GDB layers...", flush=True)
        features_gdf.to_file(method_paths["output_gdb_path"], layer="topology_features", driver="OpenFileGDB")
        main_gdf = features_gdf[features_gdf["is_main_lineage"] == True].copy()
        if not main_gdf.empty:
            main_gdf.to_file(method_paths["output_gdb_path"], layer="main_lineage_features", driver="OpenFileGDB")

    print("[05_1] exporting topology plots...", flush=True)
    export_topology_plots(metrics_df, raw_plot_features_df, method_paths["plots_dir"], method_paths["output_dir"])
    print_completion_summary(method_paths, metrics_df, summary_df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for method_name in METHOD_NAMES:
        process_method(method_name)


if __name__ == "__main__":
    main()



