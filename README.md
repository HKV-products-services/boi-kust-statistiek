# Wind- en zeewaterstandstatistiek voor BOI

Python-versie van de KNMI-statistiek voor wind en zeewaterstand (de Valk & van den Brink, R-scripts versie
februari 2026), gemaakt door HKV voor de code review PR5197.30. Elk R-script heeft een notebook in
`docs/examples/notebooks` dat dezelfde stappen in dezelfde volgorde uitvoert. Vanaf `set.seed(...)` geeft een
run van het notebook dezelfde resultaten als het R-script, inclusief de bootstrap-samples.

| R-script (versie feb. 2026)            | Python-notebook                  | seed    |
|------------------------------------------|----------------------------------|---------|
| `statistics-wind10-impr.Rmd`             | `1_statistics_wind.ipynb`        | 449723  |
| `production-wind10-impr.Rmd`             | `2_production_wind.ipynb`        | –       |
| `statistics-wl10-impr.Rmd`               | `3_statistics_wl.ipynb`          | 342911  |
| `statistics-wl10others-impr.Rmd`         | `4_statistics_wl_others.ipynb`   | 180225  |
| `statistics-wl10-omni-texeln-impr.Rmd`   | `5_statistics_wl_texeln.ipynb`   | 379233  |
| `production-wl10extended-impr.Rmd`       | `6_production_wl.ipynb`          | –       |
| `dependence-wlwind10-impr.Rmd` (production-chunk) | `7_dependence.ipynb`    | –       |

## Opbouw van de repository

| pad | inhoud |
|-----|--------|
| `src/kuststatistiek/` | het Python-package |
| `docs/examples/notebooks/` | de notebooks (één per R-script), opgeslagen zonder uitvoer |
| `docs/` | Quarto-website met de notebooks en de API-referentie (quartodoc) |
| `test/` | pytest-tests |

## Installatie

De omgeving wordt beheerd met [Pixi](https://pixi.prefix.dev/latest/). Pixi is een package- en omgevingsbeheerder. Met pixi installeert iedereen met één commando precies dezelfde, vastgelegde projectomgeving, inclusief de juiste Python-versie, zonder handmatige acties met conda, venv of pip.

<details>
<summary>Windows</summary>

1. Installeer pixi eenmalig, in PowerShell:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```
Sluit PowerShell en open opnieuw.
</details>

<details>
<summary>Linux/Mac</summary>

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

</details>

2. Clone de repository in een gewenste werkmap, bijv.:
```bash
cd C:\Users\<username>\GitHub
git clone https://github.com/HKV-products-services/boi-kust-statistiek.git
cd boi-kust-statistiek
```
3. Installeer daarna de omgeving (inclusief het package zelf, editable) en de pre-commit hooks:

```bash
pixi install
```
Handige taken: `pixi run pytest`, `pixi run pre-commit`, `pixi run docs` (preview van de website).

4. Open in VS Code (of ander programma waarin je met Python werkt) de geïnstalleerde Folder (C:\Users\<username>\GitHub\boi-kust-statistiek) en zorg dat je de bijbehorende pixi omgeving selecteert als Python kernel (.pixi\envs\default\python.exe).

De repository (zonder omgeving) kan ook geïnstalleerd worden met:
```bash
pip install boi-kust-statistiek
```

## Gebruik

1. Geef aan waar de invoerdata staan en waar de resultaten heen moeten, met de omgevingsvariabelen
   (`KUSTSTATISTIEK_DATA_PATH`, `KUSTSTATISTIEK_OUTPUT_PATH`, `KUSTSTATISTIEK_FIG_PATH`) of door
   `input_path` / `output_path` aan te passen in de cel **Settings** bovenaan elk notebook. De invoerdata zijn de
   map `data` uit het KNMI-archief `Update_statistiek_zeewaterstand_wind_06_archive_Feb2026`; ze staan niet
   in deze repository, maar kunnen opgevraagd worden bij Rijkswaterstaat, KNMI of HKV.
2. Open het notebook in VS Code (of Jupyter) en kies de pixi-omgeving (`.pixi/envs/default`).
   Controleer waar nodig `stations` en `nb` (aantal bootstrap-samples) in de cel Settings.
3. Voer alle cellen uit (*Run All*). Elke stap komt overeen met een R-chunk; de naam van de chunk staat in de kop.

Draai de notebooks in de volgorde van hun nummers: latere notebooks lezen de resultaten van eerdere
(2 heeft 1 nodig; 4 heeft 3 nodig; 6 heeft 2, 3 en 4 nodig; 5 en 7 hebben alleen de invoerdata nodig).
De statistiek-notebooks met bootstraps zijn traag: ongeveer 1 minuut per bootstrap-sample per station
(100 samples x 7 stations is ongeveer een halve dag per notebook).

`stations` kan elke selectie van stations zijn, in elke volgorde (bijv. alleen `["hoekvanh"]`): elk station
krijgt nog steeds precies de toevalsgetallen die het in de R-run met alle stations krijgt. De notebooks spelen
eerst de trekkingen van de R-run na voor alle stations van `r_stations` (de stationslijst van de R-run, in
R-volgorde; duurt ongeveer een halve minuut, zonder fits) en bewaren per station en stap de positie in de
toevalsreeks waar R voor dat station begint te trekken (`kuststatistiek/rstream.py`).
Om dezelfde resultaten als een R-run te krijgen, moeten `nb` en `r_stations` gelijk zijn aan die van die R-run.
Elke cel die toevalsgetallen trekt, begint zelf met `set.seed` of met een eigen kopie van de bewaarde positie
in de reeks, zodat elke cel opnieuw kan worden uitgevoerd (bijv. na een onderbreking) zonder dat de
toevalsgetallen veranderen.
Notebooks 2 en 6 slaan stations over waarvan de resultaten er nog niet zijn. De afhankelijkheidstabellen
(notebook 7) hebben één rij per station, in de volgorde van `stations`.

De standaardwaarden van `input_path`, `output_path`, `nb` en `r_stations` staan in `src/kuststatistiek/settings.py`
(`DATA_PATH`, `OUTPUT_PATH`, `NB`, `WIND_STATIONS` / `WL_STATIONS` / `OTHER_STATIONS`).

Uitvoerbestanden hebben de R-namen met extensie `.pkl` (een dict met precies de variabelenamen van de
R-aanroep `save()`); CSV-tabellen hebben de R-namen en hetzelfde tekstformaat als R's `write.csv`.

## Voorbewerking van de SEAS5/DCSM-5-modeldata

De invoerbestanden `<station>_multi_int.RData` en `<station>_wind.RData` zijn door KNMI gemaakt met de
R-scripts `readmixed.R`, `concat_multi_ex.R` en `concat_wind.R` (in het KNMI-archief, naast `data`). Deze
scripts zijn geport naar `kuststatistiek.preprocessing`:

| R-script              | Python                       | resultaat |
|-----------------------|------------------------------|-----------|
| `readmixed.R`         | `read_mixed` / `read_mixed_file` | per SEAS5-run en station de hoogwaterstanden van DCSM-5 met wind en windschuifspanning van het dichtstbijzijnde SEAS5-roosterpunt: `<station>_<run>_int.nc` |
| `concat_multi_ex.R`   | `concat_multi`               | `<station>_multi_int.nc` (dt, wl, surge, mstress, dstress, mwind, dwind, timelength) |
| `concat_wind.R`       | `concat_wind`                | `<station>_wind.nc` (dt, mstress, dstress, mwind, dwind, timelength) |

```python
from kuststatistiek.preprocessing import read_mixed, concat_multi

# de paden in selection_stress_wind_tide.txt verwijzen naar de KNMI-servers; vertaal ze naar de lokale kopie
read_mixed("selection_stress_wind_tide.txt", out_dir="collocated/vlissing",
           path_map={"/net/pc200003/nobackup_1/users/brinkh": "D:/SEAS5"}, stations=["vlissing"],
           lat=[51.42], lon=[3.50])
concat_multi("collocated", stations=["vlissing"], out_path="data")
```

- Python kan geen R-lijsten naar `.RData` schrijven; de voorbewerking schrijft daarom netCDF met dezelfde
  variabelenamen. `read_seas5_wind` en `read_seas5_multi` lezen het `.RData`-bestand als dat er is, en anders
  het `.nc`-bestand. `concat_multi` leest zowel de `.RData`-bestanden van `readmixed.R` als de `.nc`-bestanden
  van `read_mixed`.
- Voor de ruwe modeldata (netCDF-bestanden van SEAS5 en DCSM-5 op de KNMI-servers, zie
  `selection_stress_wind_tide.txt`) is toegang via KNMI nodig; ze zitten niet in het archief.
- De ports zijn op synthetische data vergeleken met de R-scripts (R 4.5.3): met `discard_first_month=False` zijn
  de uitkomsten identiek, op 1 ulp in `mstress`/`mwind` na in enkele gevallen waarin `sqrt` van R onder Windows
  niet correct afrondt.

Correcties van de bevindingen uit de code review (standaard aan; `discard_first_month=False` geeft het gedrag
van de R-scripts):

- **Eerste maand (inspeelperiode) van elke run.** In `selection_stress_wind_tide.txt` hebben 6 runs (ensemble 00
  van 198101 t/m 198106) `tstart = 1`, waardoor hun eerste maand in `<station>_multi_int` terechtkomt; daarnaast
  beginnen 500 februari-runs in schrikkeljaren op 29 februari. `read_mixed` gooit daarom alles weg vóór de eerste
  dag van de maand na de start van de run (naast `tstart`/`tend`). De geleverde `*_multi_int.RData` bevatten deze
  data nog; dit is pas opgelost als de bestanden opnieuw worden aangemaakt.
- **`concat_wind` kan de SEAS5-netCDF-bestanden lezen.** `concat_wind.R` laadt halfjaarbestanden die KNMI uit de
  SEAS5-bestanden heeft gemaakt (niet geleverd) en geeft een error op de `.nc`-bestanden zelf. `concat_wind` leest
  de `.nc`-bestanden (`seasonal_system5.<datum>.<ensemble>.nc`, eventueel in submappen met `recursive=True`)
  direct: wind en windschuifspanning van het dichtstbijzijnde roosterpunt, maanden 2 t/m 7 van elke run (dezelfde
  periode als de waterstanden). Halfjaarbestanden (`.RData`) worden zoals in R gebruikt; dan is station j kolom j,
  en gebruikt `concat_wind.R` uit het archief de stationslijst met `harlinge`, terwijl de geleverde data
  `westters_wind.RData` bevat. Bij de `.nc`-bestanden kan elke stationslijst (bijv. `WIND_STATIONS`).
- **Texel Noordzee** (`SEAS5_hwloc_230.RData`) is niet met `readmixed.R` gemaakt en het script ontbreekt; dit kan
  niet worden gereconstrueerd. `read_mixed` kan wel extra stations verwerken (`stations`, `lat`, `lon`) als de
  variabelenaam in de DCSM-5-bestanden bekend is.

## Package `kuststatistiek`

| module          | inhoud |
|-----------------|--------|
| `rrandom.py`    | R's toevalsgetallen: Mersenne-Twister met R's `set.seed`-scrambling, `runif`, `rnorm` (inversie, R's `qnorm`), `sample` (rejection sampling van R >= 3.6) |
| `rstream.py`    | R's toevalsreeks opgesplitst per station: speelt de trekkingen van een R-run na en bewaart de positie in de reeks per station en stap, zodat stations in elke volgorde of selectie kunnen worden doorgerekend |
| `rutils.py`     | R-semantiek: `approx`, `quantile` (type 7), `signif`, `sum/mean/sd` (sommatie in long double), `loess` (scikit-misc = de netlib-code van R), `%*%`, opmaak van `write.csv` |
| `gw.py`         | `FitGW_iHilli`, `qGWdisp`, `h` (EVTools), `esextr`, `collect`, `assigncat`, `taildir` |
| `evtools.py`    | `EI`, `Wfit`, `tau2upot`, `tau2u10`, `correctfrequencies`, `eimdl` |
| `analysis.py`   | de rekenchunks (fits, bootstraps, afvlakking, combinatie voor de overige stations) |
| `production.py` | Weibull-herfit en tabellen met conditionele kansen (22,5 en 30 graden, omni) |
| `preprocessing.py` | voorbewerking van de SEAS5/DCSM-5-modeldata (`readmixed.R`, `concat_multi_ex.R`, `concat_wind.R`) |
| `io/`           | inleesroutines voor KNMI-winddata, getijmeetstations, Texel Noordzee-data, SEAS5(-DCSM5)-bestanden, een minimale `.RData`-lezer (`rdata.py`), netCDF (`netcdf.py`) |
| `settings.py`   | constanten, richtingsklassen (statistiek: noord = 0; productie: noord = 360), stationstabellen |
| `plotting.py`   | optionele figuren |

Benodigd: numpy, pandas, scipy, pyreadr, netCDF4, scikit-misc, tqdm, matplotlib (en ipykernel om de notebooks te draaien); deze zitten allemaal in de pixi-omgeving.

## Contact

Voor vragen over dit package: neem contact op met P. Kindermann (`P.Kindermann@hkv.nl`).
