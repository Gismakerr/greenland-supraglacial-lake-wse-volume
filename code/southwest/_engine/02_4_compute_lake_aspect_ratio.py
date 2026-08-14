from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = Path(str(os.environ.get("OUTPUT_ROOT_DIR", "")).strip() or (PROJECT_ROOT / "result1"))
INPUT_SHP = RESULT_ROOT / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons.shp"
OUTPUT_SHP = RESULT_ROOT / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons_WithAspect.shp"
OUTPUT_CSV = RESULT_ROOT / "02_Lake_Extraction" / "04_Filtered_Lakes" / "Water_Max_Filtered_Polygons_WithAspect.csv"


def calc_length_width_by_mrr(geom) -> tuple[float, float, float]:
    if geom is None or geom.is_empty:
        return np.nan, np.nan, np.nan

    try:
        rect = geom.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        if len(coords) < 5:
            return np.nan, np.nan, np.nan

        edges = []
        for i in range(4):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append(float(np.hypot(x2 - x1, y2 - y1)))

        edges = sorted(edges, reverse=True)
        length_m = edges[0]
        width_m = edges[-1]
        if width_m <= 0:
            return length_m, width_m, np.nan
        return length_m, width_m, length_m / width_m
    except Exception:
        return np.nan, np.nan, np.nan


def main() -> None:
    if not INPUT_SHP.exists():
        print(f"[02_4] missing input: {INPUT_SHP}", flush=True)
        return

    gdf = gpd.read_file(INPUT_SHP)
    if gdf.empty:
        print("[02_4] input shapefile is empty", flush=True)
        return

    gdf_utm = gdf.to_crs(epsg=32627)
    vals = gdf_utm.geometry.apply(calc_length_width_by_mrr)
    gdf["length_m"] = vals.apply(lambda x: x[0])
    gdf["width_m"] = vals.apply(lambda x: x[1])
    gdf["lw_ratio"] = vals.apply(lambda x: x[2])

    OUTPUT_SHP.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")
    gdf.drop(columns="geometry", errors="ignore").to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    valid_n = int(np.isfinite(gdf["lw_ratio"]).sum())
    print(f"[02_4] done: total={len(gdf)}, valid_aspect={valid_n}", flush=True)
    print(f"[02_4] shp: {OUTPUT_SHP}", flush=True)
    print(f"[02_4] csv: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
