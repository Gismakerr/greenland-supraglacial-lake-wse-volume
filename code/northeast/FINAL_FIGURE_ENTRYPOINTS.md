# Final representative-figure entry points

## Area-WSE composite

```powershell
python code/08b_compose_representative_area_wse.py
```

Reads the ten locked 3:1 row panels in
`result/08_area_wse_fits/representative_panel_rows/` and writes the stable
PNG/TIFF manuscript figure directly to `result/12_paper_figures/figures/`.

## Six-lake runoff-volume composite

```powershell
python code/10c_plot_representative_runoff_volume.py --combined-rate-labels
```

Reads the three minimal inputs in
`result/10_lake_volume_change/runoff_figure_inputs/` and writes individual
panels plus the six-row combined figure to
`result/10_lake_volume_change/runoff_volume_combined_rate_labels/`.

## Stable-copy collection and validation

```powershell
python code/12_collect_paper_figures.py --run
```

Copies the stage-10/11 stable products, rebuilds the file manifest, and checks
the dimensions and SHA-256 of all ten official manuscript files.
