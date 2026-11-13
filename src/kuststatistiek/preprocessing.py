"""
Preprocessing of the SEAS5 / DCSM-5 model data: ports of the KNMI R scripts readmixed.R,
concat_multi_ex.R and concat_wind.R (archive Update_statistiek_zeewaterstand_wind_06_archive_Feb2026).

These scripts make the input files <station>_multi_int.RData and <station>_wind.RData of the statistics
notebooks from the model output on the KNMI servers:

1. `read_mixed` (readmixed.R): for each SEAS5 run listed in selection_stress_wind_tide.txt, collocate
   the high-tide water levels of DCSM-5 with the SEAS5 wind and wind stress of the nearest grid point.
   Writes one file <station>_<run>_int per station and run.
2. `concat_multi` (concat_multi_ex.R): concatenate the files of step 1 per station to
   <station>_multi_int (dt, wl, surge, mstress, dstress, mwind, dwind, timelength).
3. `concat_wind` (concat_wind.R): concatenate the SEAS5 wind and wind stress of all runs per station
   to <station>_wind (dt, mstress, dstress, mwind, dwind, timelength).

Corrections of the findings of the code review (section "Data voorbewerking"); the R behaviour is
available with ``discard_first_month=False``:

- First month of each run (spin-up). In selection_stress_wind_tide.txt, 6 runs (member 00 of
  198101-198106) have tstart = 1, so their first month ends up in <station>_multi_int; 500 runs of
  February in leap years start at 29 February. `read_mixed` discards everything before the first day of
  the month after the start of the run (in addition to tstart / tend).
- concat_wind.R loads half-year files that KNMI made from the SEAS5 netCDF files (not in the archive)
  and fails on the netCDF files themselves. `concat_wind` also reads the SEAS5 netCDF files directly:
  wind and wind stress of the grid point nearest to each station, months 2 to 7 of each run (the same
  window as the water levels in selection_stress_wind_tide.txt).
- Texel Noordzee: SEAS5_hwloc_230.RData was not made with readmixed.R and its script is missing; this
  cannot be reconstructed here.

The Python versions write netCDF instead of .RData (Python cannot write R lists); `read_seas5_wind` and
`read_seas5_multi` read both. Times are POSIXct seconds (since 1970-01-01 UTC), as in R. `concat_multi`
reads the files of step 1 written by R (.RData) as well as by Python (.nc).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kuststatistiek.io.netcdf import (
    nc_dim_time,
    nc_dim_vals,
    ncvar_get,
    read_nc,
    write_nc,
)
from kuststatistiek.io.rdata import read_rdata
from kuststatistiek.io.read_wl import hwextract
from kuststatistiek.rutils import r_approx, r_round_half_even
from kuststatistiek.settings import (
    CONCAT_MULTI_STATIONS,
    CONCAT_WIND_STATIONS,
    MIXED_LAT,
    MIXED_LON,
    MIXED_STATIONS,
)

WLEVELS_TIME_VARS = ("dt_tide", "dt_wl")


# ------------------------------------------------------------------------------------------
# readmixed.R
# ------------------------------------------------------------------------------------------
def read_selection(path):
    """read.table("selection_stress_wind_tide.txt", sep = " "): one row per SEAS5 run with the water
    level file (V1), the first and last time index to keep (V2, V3; 1-based), the tide-only file (V6)
    and the SEAS5 wind/stress file (V7)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            rows.append(
                {
                    "wl": v[0],
                    "tstart": int(v[1]),
                    "tend": int(v[2]),
                    "datestart": v[3],
                    "dateend": v[4],
                    "tide": v[5],
                    "wind": v[6],
                }
            )
    return rows


def month_start(t, months=0):
    """POSIXct seconds of the first day of the month of `t` (POSIXct seconds), plus `months` months."""
    period = pd.Timestamp(float(t), unit="s").to_period("M") + months
    return period.to_timestamp().value / 1e9


def _map_path(path, path_map):
    for old, new in (path_map or {}).items():
        if path.startswith(old):
            return str(Path(new) / path[len(old) :].lstrip("/"))
    return path


def collocate_station(
    val_wl, val_tide, dt_wl, dt_tide, dt_wind, wind, tstart, tend, tmin=None
):
    """Body of the station loop of readmixed.R for one run and one station.

    `val_wl` and `val_tide` are the water level and tide-only series (same time axis); `dt_wl`,
    `dt_tide` and `dt_wind` the POSIXct seconds of the water level, tide and wind series; `wind` a
    dict with the SEAS5 series u10, v10, iews, inss at the nearest grid point; `tstart` and `tend`
    the 1-based indices in dt_wl of the period to keep. `tmin` is an optional earliest time to keep
    (POSIXct seconds); it is not in R and is used to discard the first month of the run.

    Returns the R list `wlevels` as a dict: tide, wl, dt_tide, dt_wl, u10, v10, iews, inss.
    """
    val_wl = np.asarray(val_wl, dtype=float)
    tide, ind = hwextract(np.asarray(val_tide, dtype=float))  # high waters of the tide

    # highest water level within +-36 samples (6 h) of each tidal high water
    n = len(val_wl)
    indic = np.clip(ind[:, None] + np.arange(-36, 37)[None, :], 0, n - 1)
    v = val_wl[indic]
    allna = np.all(np.isnan(v), axis=1)  # which.max(all NA) -> NA
    imax = np.argmax(
        np.where(np.isnan(v), -np.inf, v), axis=1
    )  # which.max: first maximum
    wlind = indic[np.arange(len(ind)), imax]
    wl = np.where(allna, np.nan, val_wl[wlind])
    dt_tide = np.asarray(dt_tide, dtype=float)
    dt_t = dt_tide[ind]
    dt_w = np.where(allna, np.nan, dt_tide[wlind])  # R uses dt_tide here, not dt_wl

    # remove data at beginning/end; as in R, a row with an NA condition is kept with all values NA
    dt_wl = np.asarray(dt_wl, dtype=float)
    t0 = dt_wl[tstart - 1] if 1 <= tstart <= len(dt_wl) else np.nan
    t1 = dt_wl[tend - 1] if 1 <= tend <= len(dt_wl) else np.nan
    if tmin is not None and not np.isnan(t0):
        t0 = max(t0, tmin)
    na = np.isnan(dt_w) | np.isnan(t0) | np.isnan(t1)
    with np.errstate(invalid="ignore"):
        keep = na | ((dt_w >= t0) & (dt_w <= t1))
    na = na[keep]
    wlevels = {
        "tide": np.where(na, np.nan, tide[keep]),
        "wl": np.where(na, np.nan, wl[keep]),
        "dt_tide": np.where(na, np.nan, dt_t[keep]),
        "dt_wl": np.where(na, np.nan, dt_w[keep]),
    }

    # matching wind/stress: linear interpolation in time
    nw = len(dt_wind)
    rind = r_approx(dt_wind, np.arange(1, nw + 1, dtype=float), wlevels["dt_wl"])
    ok = ~np.isnan(rind)
    ind0 = np.floor(rind[ok]).astype(int) - 1
    ind1 = np.ceil(rind[ok]).astype(int) - 1
    lambda1 = rind[ok] - np.floor(rind[ok])
    lambda0 = 1 - lambda1
    for key in ("u10", "v10", "iews", "inss"):
        temp = np.asarray(wind[key], dtype=float)
        out = np.full(len(rind), np.nan)
        out[ok] = temp[ind0] * lambda0 + temp[ind1] * lambda1
        wlevels[key] = out
    return wlevels


def read_mixed_file(
    wl_file,
    tide_file,
    wind_file,
    tstart,
    tend,
    out_name,
    out_dir=".",
    stations=MIXED_STATIONS,
    lat=MIXED_LAT,
    lon=MIXED_LON,
    discard_first_month=True,
):
    """One iteration of the loop over the SEAS5 runs in readmixed.R. Writes
    <out_dir>/<station>_<out_name>_int.nc for each station; returns the paths written (an empty list,
    as in R, if the time units of the wind file cannot be read).

    With `discard_first_month`, high waters before the first day of the month after the start of the
    water level series (the start of the run) are removed, also if tstart points to an earlier time.
    discard_first_month=False gives the results of readmixed.R."""
    from netCDF4 import Dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with (
        Dataset(wl_file) as nc_wl,
        Dataset(tide_file) as nc_tide,
        Dataset(wind_file) as nc_wind,
    ):
        dt_wl = nc_dim_time(nc_wl)
        dt_tide = nc_dim_time(nc_tide)
        dt_wind = nc_dim_time(nc_wind)
        if dt_wind is None:
            print(f"wind file wrongly formatted in: {wind_file}")
            return written
        tmin = month_start(dt_wl[0], 1) if discard_first_month else None
        lon_vals = nc_dim_vals(nc_wind, "lon")
        lat_vals = nc_dim_vals(nc_wind, "lat")
        for station, la, lo in zip(stations, lat, lon):
            index = {
                "lon": int(np.argmin(np.abs(lon_vals - lo))),
                "lat": int(np.argmin(np.abs(lat_vals - la))),
            }
            wind = {
                k: ncvar_get(nc_wind, k, index) for k in ("u10", "v10", "iews", "inss")
            }
            wlevels = collocate_station(
                ncvar_get(nc_wl, station),
                ncvar_get(nc_tide, station),
                dt_wl,
                dt_tide,
                dt_wind,
                wind,
                tstart,
                tend,
                tmin,
            )
            path = out_dir / f"{station}_{out_name}_int.nc"
            write_nc(path, wlevels, time_vars=WLEVELS_TIME_VARS, dim="hw")
            written.append(path)
    return written


def read_mixed(
    selection_file,
    out_dir=".",
    path_map=None,
    stations=MIXED_STATIONS,
    lat=MIXED_LAT,
    lon=MIXED_LON,
    runs=None,
    discard_first_month=True,
):
    """readmixed.R: collocate high-tide water levels, tide, wind and wind stress of all SEAS5 runs in
    `selection_file` (selection_stress_wind_tide.txt).

    - `out_dir`: directory of the output files <station>_<run>_int.nc (R: the working directory).
    - `path_map`: optional {old prefix: new prefix} to translate the paths in the selection file (on
      the KNMI servers, /net/pc200003/...) to the local location of the model output.
    - `stations`, `lat`, `lon`: stations (variable names in the water level files) and their
      coordinates, used to pick the nearest SEAS5 grid point (default: those of readmixed.R).
    - `runs`: optional selection of row numbers (0-based) of the selection file; None = all runs.
      R's test mode is `read_mixed_file` with the test files.
    - `discard_first_month`: discard the first month (spin-up) of every run, also for the runs whose
      tstart in the selection file does not (see the module documentation); False = as readmixed.R.

    Returns the paths of the files written.
    """
    selection = read_selection(selection_file)
    if runs is not None:
        selection = [selection[i] for i in runs]
    written = []
    for row in tqdm(selection, desc="readmixed"):
        # R: substr(files_wl[ii], start = 78, stop = 97), the file name without .nc
        out_name = Path(row["wl"]).stem
        written += read_mixed_file(
            _map_path(row["wl"], path_map),
            _map_path(row["tide"], path_map),
            _map_path(row["wind"], path_map),
            row["tstart"],
            row["tend"],
            out_name,
            out_dir,
            stations,
            lat,
            lon,
            discard_first_month,
        )
    return written


# ------------------------------------------------------------------------------------------
# concat_multi_ex.R and concat_wind.R
# ------------------------------------------------------------------------------------------
def _direction(x, y):
    """round((atan2(x, y)*(180/pi)) %% 360)"""
    return r_round_half_even(np.mod(np.arctan2(x, y) * (180 / np.pi), 360))


def _is_netcdf(path):
    with open(path, "rb") as f:
        return f.read(4) in (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF")


def _load_wlevels(path):
    """`wlevels` of a file of readmixed (R: .RData, Python: .nc)."""
    if _is_netcdf(path):
        return read_nc(path, decode_times=False)
    return read_rdata(path)["wlevels"]


def concat_multi(path, stations=CONCAT_MULTI_STATIONS, out_path=None):
    """concat_multi_ex.R: concatenate the files <station>_*_int of `read_mixed` in <path>/<station>/
    to <out_path>/<station>_multi_int.nc (out_path defaults to path, as in R).

    Returns the paths written."""
    path = Path(path)
    out_path = Path(out_path) if out_path is not None else path
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for station in stations:
        print(station)
        inpath = path / station
        if not inpath.is_dir():
            continue
        flist = sorted(
            f
            for f in inpath.iterdir()
            if station in f.name and "multi" not in f.name and f.is_file()
        )
        nf = len(flist)
        if nf == 0:
            continue
        parts = {k: [] for k in ("dt", "wl", "surge", "iews", "inss", "u10", "v10")}
        for f in flist:
            w = _load_wlevels(f)
            # R's NA is a signalling NaN: numpy warns on arithmetic with it, the result is NaN
            parts["dt"].append(w["dt_wl"])
            parts["wl"].append(w["wl"])
            with np.errstate(invalid="ignore"):
                parts["surge"].append(w["wl"] - w["tide"])
            for k in ("iews", "inss", "u10", "v10"):
                parts[k].append(w[k])
        d = {k: np.concatenate([np.atleast_1d(v) for v in p]) for k, p in parts.items()}
        ind = ~(
            np.isnan(d["wl"])
            | np.isnan(d["surge"])
            | np.isnan(d["iews"])
            | np.isnan(d["inss"])
        )
        d = {k: v[ind] for k, v in d.items()}
        out = {
            "dt": d["dt"],
            "wl": d["wl"],
            "surge": d["surge"],
            "mstress": np.sqrt(d["iews"] ** 2 + d["inss"] ** 2),
            "dstress": _direction(-d["iews"], -d["inss"]),
            "mwind": np.sqrt(d["u10"] ** 2 + d["v10"] ** 2),
            "dwind": _direction(-d["u10"], -d["v10"]),
            "timelength": 0.5 * nf,  # time in years
        }
        fname = out_path / f"{station}_multi_int.nc"
        write_nc(fname, out, time_vars=("dt",))
        written.append(fname)
    return written


def _station_coordinates(stations, lat, lon):
    if lat is None or lon is None:
        coords = dict(zip(MIXED_STATIONS, zip(MIXED_LAT, MIXED_LON)))
        missing = [st for st in stations if st not in coords]
        if missing:
            raise ValueError(f"no coordinates for {missing}: pass lat and lon")
        lat = [coords[st][0] for st in stations]
        lon = [coords[st][1] for st in stations]
    return list(lat), list(lon)


def _read_seas5_point(ds, path, lat, lon, discard_first_month, n_months):
    """Wind and wind stress at the grid point nearest to (lat, lon) of a SEAS5 netCDF file
    (seasonal_system5.<date>.<member>.nc), within months 2 .. 1 + n_months of the run."""
    dt = nc_dim_time(ds)
    if dt is None:
        print(f"wind file wrongly formatted in: {path}")
        return None
    keep = np.ones(len(dt), dtype=bool)
    if discard_first_month:
        keep &= dt >= month_start(dt[0], 1)
        if n_months is not None:
            keep &= dt < month_start(dt[0], 1 + n_months)
    index = {
        "lon": int(np.argmin(np.abs(nc_dim_vals(ds, "lon") - lon))),
        "lat": int(np.argmin(np.abs(nc_dim_vals(ds, "lat") - lat))),
    }
    d = {k: ncvar_get(ds, k, index)[keep] for k in ("iews", "inss", "u10", "v10")}
    d["dt"] = dt[keep]
    return d


def _load_wind_file(path, j, lat, lon, discard_first_month, n_months):
    """Series of station j (dt, iews, inss, u10, v10) of one input file of concat_wind:

    - a SEAS5 netCDF file (dimensions lon and lat): grid point nearest to (lat[j], lon[j]);
    - an .RData file as loaded by concat_wind.R (dt and matrices time x station): column j;
    - a netCDF file with dt and matrices time x station (as the .RData files): column j.

    Returns None if the time units of a SEAS5 file cannot be read."""
    if _is_netcdf(path):
        from netCDF4 import Dataset

        with Dataset(path) as ds:
            if "lon" in ds.dimensions and "lat" in ds.dimensions:
                return _read_seas5_point(
                    ds, path, lat[j], lon[j], discard_first_month, n_months
                )
            w = {
                k: np.ma.filled(np.ma.asarray(ds.variables[k][:], dtype=float), np.nan)
                for k in ("dt", "iews", "inss", "u10", "v10")
            }
    else:
        w = read_rdata(path)
    d = {k: np.asarray(w[k], dtype=float)[:, j] for k in ("iews", "inss", "u10", "v10")}
    d["dt"] = np.ravel(w["dt"]).astype(float)
    return d


def concat_wind(
    path,
    stations=CONCAT_WIND_STATIONS,
    out_path=None,
    lat=None,
    lon=None,
    discard_first_month=True,
    n_months=6,
    recursive=False,
):
    """concat_wind.R: concatenate the SEAS5 wind and wind stress of all runs to
    <out_path>/<station>_wind.nc (out_path defaults to path, as in R).

    The input files are all files in `path` with "nc" in the name (as in R; with `recursive`, also in
    subdirectories, e.g. seasonal/system5/<yyyymm>/). These can be

    - the SEAS5 netCDF files themselves (seasonal_system5.<date>.<member>.nc): wind and wind stress of
      the grid point nearest to each station (coordinates `lat`, `lon`; default: those of readmixed.R).
      With `discard_first_month`, only months 2 .. 1 + n_months of each run are kept (default: the
      6 months after the spin-up month, as for the water levels); with discard_first_month=False the
      whole file. concat_wind.R cannot read these files.
    - the half-year files that concat_wind.R loads (.RData with dt and matrices time x station):
      station j is column j, so `stations` must be in the column order of the files. These files are
      used as they are (they were already cut by KNMI).

    Returns the paths written."""
    path = Path(path)
    out_path = Path(out_path) if out_path is not None else path
    out_path.mkdir(parents=True, exist_ok=True)
    lat, lon = _station_coordinates(stations, lat, lon)
    own = {
        f"{s}_wind.nc" for s in stations
    }  # outputs of an earlier run (R writes .RData)
    candidates = path.rglob("*") if recursive else path.iterdir()
    flist = sorted(
        f for f in candidates if "nc" in f.name and f.name not in own and f.is_file()
    )
    written = []
    for j, station in enumerate(
        stations
    ):  # as in R: all files are read for each station
        print(station)
        parts = [
            _load_wind_file(f, j, lat, lon, discard_first_month, n_months)
            for f in flist
        ]
        parts = [p for p in parts if p is not None]
        nf = len(parts)
        if nf == 0:
            continue
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        ind = ~(
            np.isnan(d["u10"])
            | np.isnan(d["v10"])
            | np.isnan(d["iews"])
            | np.isnan(d["inss"])
        )
        d = {k: v[ind] for k, v in d.items()}
        out = {
            "dt": d["dt"],
            "mstress": np.sqrt(d["iews"] ** 2 + d["inss"] ** 2),
            "dstress": _direction(-d["iews"], -d["inss"]),
            "mwind": np.sqrt(d["u10"] ** 2 + d["v10"] ** 2),
            "dwind": _direction(-d["u10"], -d["v10"]),
            "timelength": 0.5 * nf,  # time in years
        }
        fname = out_path / f"{station}_wind.nc"
        write_nc(fname, out, time_vars=("dt",))
        written.append(fname)
    return written
