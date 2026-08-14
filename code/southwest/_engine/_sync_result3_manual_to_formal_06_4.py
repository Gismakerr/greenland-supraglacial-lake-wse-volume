from pathlib import Path
import re
import shutil

root = Path(r'X:\2024_西南水位曲线')
temp_root = root / 'result3' / '_result3_manual_categories_from_result2_07_by_06_2_masks'
formal_root = root / 'result3' / '06_4_classify_lake_wse_behaviors_plot' / '双轴图曲线_老ID'
cats = ['drainage_after_recharge', 'slow_drainage', 'stable_after_recharge', 'sudden_drainage']
pat = re.compile(r'^(Lake_\d+_branch_\d+_vD)(?:.*)?\.png$', re.IGNORECASE)

if formal_root.exists():
    shutil.rmtree(formal_root)
formal_root.mkdir(parents=True, exist_ok=True)

counts = {}
for cat in cats:
    out_dir = formal_root / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted((temp_root / cat).glob('*.png')):
        m = pat.match(src.name)
        if not m:
            continue
        dst_name = m.group(1) + '.png'
        shutil.copy2(src, out_dir / dst_name)
        n += 1
    counts[cat] = n
for k, v in counts.items():
    print(f'{k}\t{v}')
