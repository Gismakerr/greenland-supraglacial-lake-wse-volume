from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent / 'result1'

CLASSIFY_DIR = RESULT_ROOT / '06_4_classify_lake_wse_behaviors_plot'
CLASSIFY_CSV = CLASSIFY_DIR / 'classification_records_dual_axis.csv'
BASE_SHP = RESULT_ROOT / '02_Lake_Extraction' / '04_Filtered_Lakes' / 'Water_Max_Filtered_Polygons.shp'
BRANCH_DIR = RESULT_ROOT / '06_3_plot_branchwise_red_chain_with_merge_anchor_redmin5' / 'Otsu' / 'original' / '06_3_BranchTables' / 'vC'
OUT_SHP = CLASSIFY_DIR / 'lakes_all_with_classified_wse_maxrange_newid_vc_Otsu.shp'
OUT_SHP_HAS1 = CLASSIFY_DIR / 'lakes_has_curve_1_with_wse_maxrange_newid_vc_Otsu.shp'
OUT_SHP_HAS0 = CLASSIFY_DIR / 'lakes_has_curve_0_with_wse_maxrange_newid_vc_Otsu.shp'

PAT = re.compile(r'^lake_(\d+)_branch_(\d+)_stage_vC\.csv$', re.IGNORECASE)


def remove_existing_shapefile(path: Path):
    stem = path.with_suffix('')
    for p in stem.parent.glob(f'{stem.name}.*'):
        if p.suffix.lower() == '.lock':
            continue
        try:
            p.unlink()
        except Exception:
            pass


def detect_col(columns, candidates):
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


if not CLASSIFY_CSV.exists():
    raise FileNotFoundError(f'Missing classify csv: {CLASSIFY_CSV}')
if not BASE_SHP.exists():
    raise FileNotFoundError(f'Missing base shp: {BASE_SHP}')
if not BRANCH_DIR.exists():
    raise FileNotFoundError(f'Missing branch dir: {BRANCH_DIR}')

class_df = pd.read_csv(CLASSIFY_CSV, encoding='utf-8-sig')
if class_df.empty:
    raise RuntimeError('classification_records_dual_axis.csv is empty')

for col in ['lake_id_old', 'branch_index', 'category']:
    if col not in class_df.columns:
        raise RuntimeError(f'Missing column in classify csv: {col}')

class_df['lake_id_old'] = pd.to_numeric(class_df['lake_id_old'], errors='coerce')
class_df['branch_index'] = pd.to_numeric(class_df['branch_index'], errors='coerce')
class_df = class_df.dropna(subset=['lake_id_old', 'branch_index']).copy()
class_df['lake_id_old'] = class_df['lake_id_old'].astype(int)
class_df['branch_index'] = class_df['branch_index'].astype(int)

pair_df = class_df[['lake_id_old', 'branch_index', 'category']].drop_duplicates().reset_index(drop=True)

range_rows = []
missing_csv = 0
for row in pair_df.itertuples(index=False):
    lake_id = int(row.lake_id_old)
    branch = int(row.branch_index)
    csv_path = BRANCH_DIR / f'lake_{lake_id}_branch_{branch}_stage_vC.csv'
    if not csv_path.exists():
        missing_csv += 1
        continue
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        missing_csv += 1
        continue
    if df.empty or 'wse_m' not in df.columns:
        continue
    # Use denoised WSE for range calculation:
    # prefer rows where is_noise == False; fallback to full series if the field is absent.
    if 'is_noise' in df.columns:
        is_noise = df['is_noise'].astype(str).str.strip().str.lower().isin(['1', 'true', 't', 'yes', 'y'])
        df_use = df.loc[~is_noise].copy()
        if df_use.empty:
            continue
    else:
        df_use = df

    wse = pd.to_numeric(df_use['wse_m'], errors='coerce').dropna()
    if wse.empty:
        continue
    max_wse = float(wse.max())
    min_wse = float(wse.min())
    max_rng = max_wse - min_wse
    range_rows.append({
        'lake_id_old': lake_id,
        'branch_index': branch,
        'category': str(row.category),
        'max_wse_m': max_wse,
        'min_wse_m': min_wse,
        'max_rng_m': max_rng,
        'obs_n': int(len(wse)),
    })

range_df = pd.DataFrame(range_rows)
if range_df.empty:
    raise RuntimeError('No valid branch WSE stats computed.')

# lake-level aggregate: max range among classified branches
lake_agg = (
    range_df.sort_values(['lake_id_old', 'max_rng_m'], ascending=[True, False])
    .groupby('lake_id_old', as_index=False)
    .first()
)

# assign new id among lakes with curves: descending max range
ranked = lake_agg.sort_values(['max_rng_m', 'lake_id_old'], ascending=[False, True]).reset_index(drop=True)
ranked['lake_id_new'] = np.arange(1, len(ranked) + 1, dtype=int)

# all features shapefile
base = gpd.read_file(BASE_SHP)
lake_col = detect_col(base.columns, ['lake_id', 'old_lake_i'])
if lake_col is None:
    raise RuntimeError(f'No lake id column in base shapefile: {list(base.columns)}')

base['lake_id_old'] = pd.to_numeric(base[lake_col], errors='coerce')
base = base.dropna(subset=['lake_id_old']).copy()
base['lake_id_old'] = base['lake_id_old'].astype(int)

merge_cols = ['lake_id_old', 'lake_id_new', 'max_rng_m', 'max_wse_m', 'min_wse_m', 'branch_index', 'category', 'obs_n']
out = base.merge(ranked[merge_cols], on='lake_id_old', how='left')
out['has_curve'] = out['lake_id_new'].notna().astype(int)

# keep stable dtypes for shp
for c in ['lake_id_new', 'branch_index', 'obs_n', 'has_curve']:
    out[c] = pd.to_numeric(out[c], errors='coerce')
for c in ['max_rng_m', 'max_wse_m', 'min_wse_m']:
    out[c] = pd.to_numeric(out[c], errors='coerce')
out['category'] = out['category'].fillna('').astype(str)
if 'lake_type' in out.columns:
    out['lake_type'] = out['lake_type'].fillna('').astype(str)
else:
    out['lake_type'] = ''

# behavior 4-class code (0/1/2/3)
behavior_code_map = {
    'drainage_after_recharge': 0,
    'slow_drainage': 1,
    'stable_after_recharge': 2,
    'sudden_drainage': 3,
}
out['bhv4_code'] = out['category'].str.strip().str.lower().map(behavior_code_map)

# Drop non-curve small lakes as requested:
# remove features where has_curve == 0 and lake_type == Small
out = out[~((out['has_curve'] == 0) & (out['lake_type'].str.lower() == 'small'))].copy()

OUT_SHP.parent.mkdir(parents=True, exist_ok=True)
remove_existing_shapefile(OUT_SHP)
out.to_file(OUT_SHP, driver='ESRI Shapefile', encoding='utf-8')

# sidecar csv for full precision (avoid shp field truncation)
out_csv = OUT_SHP.with_suffix('.csv')
out[['lake_id_old', 'lake_type', 'lake_id_new', 'has_curve', 'max_rng_m', 'max_wse_m', 'min_wse_m', 'branch_index', 'category', 'bhv4_code', 'obs_n']].to_csv(out_csv, index=False, encoding='utf-8-sig')

# split by has_curve
out_has1 = out[out['has_curve'] == 1].copy()
out_has0 = out[out['has_curve'] == 0].copy()

remove_existing_shapefile(OUT_SHP_HAS1)
remove_existing_shapefile(OUT_SHP_HAS0)
if not out_has1.empty:
    out_has1.to_file(OUT_SHP_HAS1, driver='ESRI Shapefile', encoding='utf-8')
if not out_has0.empty:
    out_has0.to_file(OUT_SHP_HAS0, driver='ESRI Shapefile', encoding='utf-8')

out_has1.to_csv(OUT_SHP_HAS1.with_suffix('.csv'), index=False, encoding='utf-8-sig')
out_has0.to_csv(OUT_SHP_HAS0.with_suffix('.csv'), index=False, encoding='utf-8-sig')

print(f'[ok] output shp: {OUT_SHP}')
print(f'[ok] output csv: {out_csv}')
print(f'[ok] output shp (has_curve=1): {OUT_SHP_HAS1}')
print(f'[ok] output shp (has_curve=0): {OUT_SHP_HAS0}')
print(f'[ok] total all features: {len(out)}')
print(f'[ok] features with curves: {int(out["has_curve"].sum())}')
print(f'[ok] features without curves: {int((out["has_curve"] == 0).sum())}')
print(f'[ok] classified branches parsed: {len(range_df)} (missing csv: {missing_csv})')
