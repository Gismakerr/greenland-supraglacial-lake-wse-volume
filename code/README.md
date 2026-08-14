# Public workflow

| Step | Northeast | Southwest | Main output |
|---:|---|---|---|
| 01 | Sentinel-2 preprocessing | Sentinel-2 preprocessing | analysis-ready imagery (external/omitted) |
| 02 | lake extraction | lake extraction | lake polygons |
| 03 | effective-area extraction | effective-area extraction | daily lake area |
| 04 | branch topology | branch topology | lake/branch topology |
| 05 | SWOT PIXC extraction | SWOT PIXC extraction | pixel-cloud subsets (external/omitted) |
| 06 | WSE calculation and QC | WSE calculation and QC | SWOT WSE records |
| 07 | final WSE curves and classification | final WSE curves and classification | branch WSE time series |
| 08 | strict R² > 0.8 + AICc linear/quadratic selection | not produced | area–WSE models |
| 09 | observation statistics | observation statistics | observation summaries |
| 10 | relative volume estimation and uncertainty | not produced | volume time series |
| 11 | Lake 1 SLC phase panels | not produced | Fig06 panels |
| 12 | manuscript figure collection | manuscript figure collection | final PNG figures |
| 13 | thematic map data | thematic map data | public vectors and classes |

Daily use should start from the numbered files in each region. `_engine` contains the frozen implementation called by lightweight entry points; it is not a second workflow. `_archive`, candidates, tests, and backups are intentionally absent.
