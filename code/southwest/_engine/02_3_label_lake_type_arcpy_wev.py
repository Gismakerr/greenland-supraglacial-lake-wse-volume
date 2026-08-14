import os
import argparse
import arcpy

arcpy.env.overwriteOutput = True


def parse_args():
    parser = argparse.ArgumentParser(description='Label lake_pos and ice_flag using ArcPy')
    parser.add_argument(
        '--mask',
        default=r'D:\Haoyu_space\03_data\Greenland_IceSheet_IceMap_Merge\Greenland_IceSheet_Main.shp',
    )
    parser.add_argument(
        '--in-dir',
        default=r'X:\2024_西南水位曲线\result1\02_Lake_Extraction\04_Filtered_Lakes',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mask = args.mask
    in_dir = args.in_dir

    shps = [
        os.path.join(in_dir, n)
        for n in os.listdir(in_dir)
        if n.lower().endswith('.shp')
    ]

    if not arcpy.Exists(mask):
        raise RuntimeError(f'Mask not found: {mask}')

    print(f'Found {len(shps)} shapefiles')

    for shp in shps:
        name = os.path.basename(shp)
        print(f'\nProcessing: {name}')

        field_names = [f.name.lower() for f in arcpy.ListFields(shp)]
        if 'lake_pos' not in field_names:
            arcpy.management.AddField(shp, 'lake_pos', 'TEXT', field_length=20)
        if 'ice_flag' not in field_names:
            arcpy.management.AddField(shp, 'ice_flag', 'SHORT')

        arcpy.management.MakeFeatureLayer(shp, 'lake_lyr')
        arcpy.management.CalculateField('lake_lyr', 'lake_pos', '"冰前湖"', 'PYTHON3')
        arcpy.management.CalculateField('lake_lyr', 'ice_flag', '0', 'PYTHON3')

        arcpy.management.MakeFeatureLayer(mask, 'mask_lyr')
        arcpy.management.SelectLayerByLocation('lake_lyr', 'COMPLETELY_WITHIN', 'mask_lyr')

        sel_count = int(arcpy.management.GetCount('lake_lyr')[0])
        if sel_count > 0:
            arcpy.management.CalculateField('lake_lyr', 'lake_pos', '"冰面湖"', 'PYTHON3')
            arcpy.management.CalculateField('lake_lyr', 'ice_flag', '1', 'PYTHON3')

        total = int(arcpy.management.GetCount(shp)[0])
        print(f'  total={total}, ice_surface={sel_count}, fore_ice={total - sel_count}')

    print('\nDone')


if __name__ == '__main__':
    main()
