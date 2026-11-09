# kuststatistiek – BOI kust statistiek

Python port of the KNMI wind and sea water level statistics (de Valk & van den Brink, R scripts delivered
in February 2026), made by HKV for the code review PR5197.30. Every R script has a notebook in
`docs/examples/notebooks` that runs the same steps in the same order. Starting from `set.seed(...)`, a
notebook run gives the same results as the R script, bootstrap samples included.

| R script (delivered Feb2026)             | Python notebook                  | seed    |
|------------------------------------------|----------------------------------|---------|
| `statistics-wind10-impr.Rmd`             | `1_statistics_wind.ipynb`        | 449723  |
| `production-wind10-impr.Rmd`             | `2_production_wind.ipynb`        | –       |
| `statistics-wl10-impr.Rmd`               | `3_statistics_wl.ipynb`          | 342911  |
| `statistics-wl10others-impr.Rmd`         | `4_statistics_wl_others.ipynb`   | 180225  |
| `statistics-wl10-omni-texeln-impr.Rmd`   | `5_statistics_wl_texeln.ipynb`   | 379233  |
| `production-wl10extended-impr.Rmd`       | `6_production_wl.ipynb`          | –       |
| `dependence-wlwind10-impr.Rmd` (production chunk) | `7_dependence.ipynb`    | –       |

## Repository layout

| path | contents |
|------|----------|
| `src/kuststatistiek/` | the Python package |
| `docs/examples/notebooks/` | the notebooks (one per R script), stored without output |
| `docs/` | Quarto website with the notebooks and the API reference (quartodoc) |
| `test/` | pytest tests |

## Installation

The environment is managed with [Pixi](https://pixi.prefix.dev/latest/).

<details>
<summary>Windows</summary>

```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

</details>

<details>
<summary>Linux/Mac</summary>

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

</details>

Then install the environment (including the package itself, editable) and the pre-commit hooks:

```bash
cd boi-kust-statistiek
pixi install
pixi run install
```

Useful tasks: `pixi run pytest`, `pixi run pre-commit`, `pixi run docs` (preview of the website).

## How to run

1. Set where the input data are and where the results go, either with environment variables
   (`KUSTSTATISTIEK_DATA_PATH`, `KUSTSTATISTIEK_OUTPUT_PATH`, `KUSTSTATISTIEK_FIG_PATH`) or by editing
   `input_path` / `output_path` in the **Settings** cell at the top of each notebook. The input data are the
   `data` folder of KNMI's archive `Update_statistiek_zeewaterstand_wind_06_archive_Feb2026`; they are not
   in this repository.
2. Open the notebook in VS Code (or Jupyter) and select the pixi environment (`.pixi/envs/default`).
   Check `stations` and `nb` (number of bootstrap samples) in the Settings cell where relevant.
3. Run all cells (*Run All*). Each step corresponds to an R chunk; the chunk name is given in its heading.

Run the notebooks in the order of their numbers: later notebooks read the results of earlier ones
(2 needs 1; 4 needs 3; 6 needs 2, 3 and 4; 5 and 7 only need the input data).
Notebook 5 reads `texno1.csv` ... `texno5.csv` from the input folder; these files are not in KNMI's archive
`data` folder.
The statistics notebooks with bootstraps are slow: about 1 minute per bootstrap sample per station
(100 samples x 7 stations is about half a day per notebook).

`stations` can be any selection of stations, in any order (e.g. only `["hoekvanh"]`): every station
still gets exactly the random numbers it gets in the R run with all stations. The notebooks first
replay the random draws of the R run for all stations of `r_stations` (the station list of the R
run, in R order; takes about half a minute, no fits) and keep, per station and step, the position
in the random stream where R starts drawing for that station (`kuststatistiek/rstream.py`).
To get the same results as an R run, `nb` and `r_stations` must be those of that R run.
Every cell that draws random numbers starts from `set.seed` itself or from its own copy of the stored stream
position, so any cell can be re-run (e.g. after an interruption) without changing the random numbers.
Notebooks 2 and 6 skip stations whose results are not there yet. The dependence tables (notebook 7)
have one row per station in the order of `stations`.

The defaults of `input_path`, `output_path`, `nb` and `r_stations` are set in `src/kuststatistiek/settings.py`
(`DATA_PATH`, `OUTPUT_PATH`, `NB`, `WIND_STATIONS` / `WL_STATIONS` / `OTHER_STATIONS`).
The texts in the notebooks are in Dutch.

Output files have the R names with extension `.pkl` (a dict with exactly the variable names of the R
`save()` call), CSV tables have the R names and the same text format as R's `write.csv`.

## Package `kuststatistiek`

| module          | contents |
|-----------------|----------|
| `rrandom.py`    | R's random numbers: Mersenne-Twister with R's `set.seed` scrambling, `runif`, `rnorm` (inversion, R's `qnorm`), `sample` (R >= 3.6 rejection sampling) |
| `rstream.py`    | R's random stream split up per station: replays the random draws of an R run and keeps the stream position per station and step, so stations can be computed in any order or selection |
| `rutils.py`     | R semantics: `approx`, `quantile` (type 7), `signif`, `sum/mean/sd` (long-double accumulation), `loess` (scikit-misc = R's netlib code), `%*%`, `write.csv` formatting |
| `gw.py`         | `FitGW_iHilli`, `qGWdisp`, `h` (EVTools), `esextr`, `collect`, `assigncat`, `taildir` |
| `evtools.py`    | `EI`, `Wfit`, `tau2upot`, `tau2u10`, `correctfrequencies`, `eimdl` |
| `analysis.py`   | the computational chunks (fits, bootstraps, smoothing, combination for the other stations) |
| `production.py` | Weibull refit and conditional probability tables (22.5 and 30 degrees, omni) |
| `io/`           | readers of KNMI wind data, tide gauge data, Texel Noordzee data, SEAS5(-DCSM5) files, a minimal `.RData` reader (`rdata.py`) |
| `settings.py`   | constants, direction bins (statistics: north = 0; production: north = 360), station tables |
| `plotting.py`   | optional figures |

Requirements: numpy, pandas, scipy, pyreadr, scikit-misc, tqdm, matplotlib (and ipykernel to run the notebooks); all are in the pixi environment.

## Reproducibility notes

- Deterministic results agree with R to rounding level (relative differences <= ~1e-10, mostly
  bit-identical): R on Windows uses a different `log`/`atan2` and long-double accumulation.
- The random draws are bit-identical to R, except that ~1 in 3000 normal draws may differ in the
  last bit (again R's Windows `log`); this has no visible effect.
- The results of an R session only match if that session ran from `set.seed` through the chunks in
  order. HKV's R result files in `Uitvoering\R` were partly produced in separate sessions (e.g. the
  water level bootstraps of den Helder and later stations, the measurement bootstraps of 4 March and
  the wind `boot2` of Huibertgat and Delfzijl), so those bootstrap results cannot be matched.
- R quirks are reproduced on purpose (marked in the code), e.g. in `statistics-wl10others` the
  bootstrap quantile map of the directional sectors uses the 45-degree sector samples, and in
  `estimaterho.R` the sample fraction is overwritten by a count after the first sector.
- `readwlmeas.R` uses `readr::read_table2`, which no longer exists in readr >= 2.2 (use `read_table`).
- `production-wind10-impr.Rmd` computes its tables with the direction distribution in
  `data/hoekvanh_winddirs.RData`, not with the one it writes itself (chunk `eiseas5`, the csv used by
  the water level scripts); they differ by 1e-4 in the 225/240 degree sector. Python (notebook 2) instead uses
  the distribution it computes itself (the direction csv files), so in that sector the last digit of the wind
  tables can differ from KNMI.
- `production-wl10extended-impr.Rmd` does not run as delivered: it reads the direction csv with
  `read.csv(skip = 1)` (one data row) and uses `Table22h[2, j]`, which gives an error. KNMI's delivered
  tables correspond to the sector probabilities `Table22h[2, ]` / `Table30[2, ]` of
  `data/hoekvanh_winddirs.RData` (as in `production-wind10-impr.Rmd`). Python (notebook 6) instead uses the
  direction csv files written by notebook 2, as notebook 2 does, so the 225/240 degree sector can differ from
  KNMI in the last digit.
- `statistics-wl10-impr.Rmd`, chunk `dir-estimate-meas2`: the vector-averaged wind direction
  `(atan2(S, C)*(1/dc)) %% 360` is 0 or 360 degrees for wind from exactly north, depending on the platform
  (KNMI: 0, R on Windows: 360). 0 and 360 fall in different 45-degree sectors, which changes the directional
  water level statistics by up to ~3 cm. Python sets |angle| < 1e-9 to 0 first, which reproduces KNMI's tables.
- `loess` with span > 1 (trend of short tide gauge records, e.g. OS11) is not supported by
  scikit-misc; `rutils` implements R's rule for that case (all points, radius^2 = max dist^2 x span).
- Some R formulas lose precision, e.g. `1 - exp(-x)` for x ~ 1e-11 in the probability tables
  (use `-expm1(-x)`); in such cells the last of the 4 significant digits can differ (1 cell found).

## Verification

The delivered R scripts and the Python port were rerun with a reduced configuration (3 tide gauges and 2 wind stations, nb = 3, OS11, Texel Noordzee, dependence,
production tables) and all outputs were compared. Result (September 2026, R 4.5.3 on Windows):

| output | result |
|--------|--------|
| processed measurements (wind, water level + wind direction), settings | bit-identical / <= 4e-14 |
| GW fits SEAS5(-DCSM5), measurements (`stats_multi`, `stats_wind`, `meas0`, `multi_meas`) | <= 8e-10 relative |
| bootstrap samples incl. model error (`stats_wl_bootdata`, `meas_wl_bootdata`, `meas1/2/3`, `stats_wind_boot1/2`, `bootdata`) | same random samples; values <= 8e-10 relative |
| OS11 (others), Texel Noordzee (incl. scale bootstrap) | <= 2e-10 relative |
| all CSV tables: water level (4 stations x 2), Texel, s-values and correlation coefficients (2 x 2) | text-identical |
| wind CSV tables (2 stations x 2 sector sets x 2) | text-identical except 1 cell (4th digit, see above) |

## Contact

For questions about this package contact P. Kindermann (`P.Kindermann@hkv.nl`).
