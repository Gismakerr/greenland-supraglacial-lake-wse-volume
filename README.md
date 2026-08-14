# Estimating Greenland supraglacial lake water levels and volumes from SWOT PIXC observations

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Code license](https://img.shields.io/badge/code-MIT-0B6E69)
![Data license](https://img.shields.io/badge/derived%20data-CC%20BY%204.0-55A89D)
![Status](https://img.shields.io/badge/status-research%20release-E56B5D)

This repository accompanies a study of Greenland supraglacial-lake water-level and volume dynamics using **SWOT PIXC**, **Sentinel-2**, **ICESat-2**, and DEM-derived hydrological information. It provides the ordered analysis code, manuscript figures, compact result tables, and lightweight data used by the interactive lake atlas.

<p align="center">
  <img src="figures/Fig03_Spatial_WSE_variability_and_lake_behaviours.png" width="820" alt="Spatial variability of supraglacial-lake water levels and classified lake behaviours in Greenland">
</p>

## Research overview

Supraglacial lakes store, transfer, and rapidly release meltwater across the Greenland Ice Sheet, but their water-level evolution is difficult to observe consistently with optical imagery alone. This study combines SWOT's pixel-cloud elevations with optical lake-area observations to recover lake water-surface elevation (WSE), characterize filling and drainage behaviour, establish empirical area–WSE relationships, and estimate changes in lake water volume.

The workflow:

1. extracts and quality-controls SWOT PIXC observations over mapped supraglacial lakes;
2. constructs branch-aware WSE time series and retains noisy observations for quality-control tracing;
3. derives Sentinel-2 lake area after spectral-index calculation and cloud screening;
4. classifies seasonal lake behaviour in northeastern and southwestern Greenland;
5. selects linear or quadratic area–WSE models using strict AICc selection with \(R^2 > 0.8\);
6. converts optical area observations to additional WSE and volume estimates; and
7. relates representative volume changes to DEM-derived catchments and drainage networks.

<p align="center">
  <img src="figures/Fig01_Workflow.png" width="700" alt="Study workflow for supraglacial-lake WSE and volume estimation">
</p>

## Study at a glance

| Item | Public-release value |
|---|---:|
| Lake inventory polygons across both study regions | 2,905 |
| Reliable WSE curves — Northeast | 150 |
| Reliable WSE curves — Southwest | 208 |
| Northeast area–WSE fit units | 83 |
| Unique lakes with accepted area–WSE models | 79 |
| Selected linear / quadratic models | 41 / 42 |
| Main observation season | June–August 2024 |

Counts above are calculated from the released canonical tables. Lake 1 and Lake 2 contain branch-aware pre-merger and post-merger units, so the number of fit units is larger than the number of unique lakes.

## Main scientific outputs

- **WSE retrieval and validation** — SWOT PIXC and Raster elevations are evaluated against ICESat-2 observations.
- **Seasonal lake dynamics** — reliable WSE curves reveal continuous filling, repeated filling–drainage, slow drainage, and rapid drainage behaviours.
- **Cloud-gap observations** — SWOT supplies water-level information on dates when optical lake-area retrieval is unavailable or cloud affected.
- **Area–WSE modelling** — AICc is used consistently to select linear or quadratic relationships for accepted lakes.
- **Volume-change estimation** — final area–WSE models and reference water levels are used to derive water-volume time series and their uncertainties.
- **Hydrological context** — DEM-derived catchments, surface drainage networks, and catchment-normalized rates place lake changes in their upstream meltwater context.

<table>
  <tr>
    <td width="50%"><img src="figures/Fig04_Northeastern_Greenland_relative_WSE_curves.png" alt="Northeastern Greenland relative WSE curves"></td>
    <td width="50%"><img src="figures/Fig05_Southwestern_Greenland_relative_WSE_curves.png" alt="Southwestern Greenland relative WSE curves"></td>
  </tr>
  <tr>
    <td align="center"><b>Northeastern Greenland</b></td>
    <td align="center"><b>Southwestern Greenland</b></td>
  </tr>
</table>

<p align="center">
  <img src="figures/Fig08_Representative_DEM_catchments_and_drainage_networks.png" width="820" alt="Representative lake catchments and drainage networks">
  <br><b>Representative DEM-derived catchments and drainage networks</b>
</p>

<details>
<summary><b>View the representative area–WSE relationships and water-frequency panels</b></summary>
<br>
<p align="center"><img src="figures/Fig07_Representative_lake_WSE_area_WSE_fits_and_water_frequency.png" width="700" alt="Representative WSE and area-WSE models"></p>
</details>

## Repository contents

```text
.
├── code/
│   ├── northeast/          # ordered Northeast workflow, including AICc fits and volume estimation
│   └── southwest/          # ordered Southwest workflow; intentionally ends before unavailable products
├── figures/                # final manuscript and supplementary PNG figures
├── results/                # canonical compact CSV result tables
├── web_data/               # lightweight data prepared for the interactive atlas
├── DATA_SOURCES.md         # source-product access and redistribution notes
├── CITATION.cff            # machine-readable citation metadata
├── requirements.txt        # core Python dependencies
├── LICENSE                 # MIT code license
└── LICENSE_DATA.txt        # CC BY 4.0 notice for derived data and figures
```

### Canonical result tables

| File | Description |
|---|---|
| `results/final_wse_branch_timeseries.csv` | Branch-aware WSE observations, uncertainty, noise flags, and dates for both regions |
| `results/lake_attributes_all_regions.csv` | Unified lake inventory, region, behaviour class, WSE range, and observation counts |
| `results/aicc_model_selection.csv` | Linear and quadratic candidate statistics and the final AICc-selected model |
| `results/volume_summary.csv` | Reference levels, volume ranges, rates, model provenance, and observation periods |

## Reproducing the workflow

Create an isolated Python environment and install the public dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Scripts are numbered in execution order. Read [`code/README.md`](code/README.md) and the regional README before running them. Public scripts default to validation or dry-run behaviour where implemented; use `--run` only after configuring local paths to the required external source products.

Some geospatial steps additionally require a working GDAL/PROJ stack. DEM hydrology scripts that use ArcPy require ArcGIS Pro, Spatial Analyst, and a compatible Python environment.

## Source data and redistribution

Raw Sentinel-2 imagery, SWOT PIXC/SLC/Raster products, ICESat-2 files, and DEM rasters are **not redistributed** in this repository. They remain available from their official providers:

- Sentinel-2: Copernicus Data Space Ecosystem / Google Earth Engine
- SWOT PIXC, Raster, and SLC: NASA/JPL PO.DAAC
- ICESat-2 ATL03 and ATL06: NASA NSIDC DAAC
- ArcticDEM: Polar Geospatial Center

See [`DATA_SOURCES.md`](DATA_SOURCES.md) for the complete source-data statement. The larger derived-data package—containing intermediate tables, vectors, plotting data, and workflow quicklooks—is prepared separately for Figshare.

## Figures

The `figures` directory contains the final PNG figures using manuscript numbering. Figure captions and file provenance are listed in [`figures/figure_captions.csv`](figures/figure_captions.csv). TIFF files, candidate figures, backups, and test outputs are intentionally excluded.

## Citation

If you use the code, derived tables, or figures, please cite the accompanying paper and archived Figshare dataset. Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The Figshare DOI will be added when the data record is publicly assigned.

## License

- **Code:** MIT License
- **Derived data and figures:** CC BY 4.0, subject to the terms of the original data providers

## Contact

For questions about the workflow or released data, please open a GitHub issue. The project is maintained by **Haoyu Cao**.
