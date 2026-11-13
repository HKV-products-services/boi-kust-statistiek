"""Tests of the preprocessing (readmixed.R, concat_multi_ex.R, concat_wind.R) on small synthetic data.

The ports were checked against the R scripts (R 4.5.3, ncdf4 1.24) on synthetic model output: the
results are identical, except mstress/mwind in a few near-tie cases where sqrt() of R on Windows is not
correctly rounded (1 ulp).
"""

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from kuststatistiek.io.netcdf import read_nc, write_nc  # noqa: E402
from kuststatistiek.io.read_wind import read_seas5_multi, read_seas5_wind  # noqa: E402
from kuststatistiek.preprocessing import (  # noqa: E402
    collocate_station,
    concat_multi,
    concat_wind,
    month_start,
    read_mixed_file,
    read_selection,
)

T0 = 536457600.0  # 1987-01-01 00:00 UTC in POSIXct seconds


def _tide(n):
    t = np.arange(n) * 600.0  # 10-minute values
    return T0 + t, np.cos(2 * np.pi * t / (12.42 * 3600))


def test_collocate_station():
    dt, tide = _tide(1000)
    wl = tide.copy()
    wl[80:85] += 0.5  # surge peak 4-8 samples after the 2nd tidal high water (index 75)
    wind_dt = T0 + np.arange(0, 1000 * 600.0, 3600.0)
    wind = {k: np.arange(len(wind_dt), dtype=float) for k in ("u10", "v10", "iews", "inss")}
    w = collocate_station(wl, tide, dt, dt, wind_dt, wind, tstart=10, tend=1000)
    # tidal high waters at 75, 149, 224, ... (hwextract never returns the first sample)
    assert w["dt_tide"][0] == dt[75]
    assert w["dt_wl"][0] == dt[80]  # first maximum of wl within +-36 samples
    assert w["wl"][0] == wl[80]
    assert w["tide"][0] == tide[75]
    # wind interpolated linearly in time: hourly index = (t - T0) / 3600
    np.testing.assert_allclose(w["u10"], (w["dt_wl"] - T0) / 3600)
    assert set(w) == {"tide", "wl", "dt_tide", "dt_wl", "u10", "v10", "iews", "inss"}


def test_collocate_station_missing_values():
    dt, tide = _tide(1000)
    wl = tide.copy()
    wl[113:186] = np.nan  # no water level around the 2nd tidal high water (index 149)
    wind_dt = T0 + 13 * 3600 + np.arange(0, 1000 * 600.0, 3600.0)  # wind starts after the 1st
    wind = {k: np.ones(len(wind_dt)) for k in ("u10", "v10", "iews", "inss")}
    w = collocate_station(wl, tide, dt, dt, wind_dt, wind, tstart=1, tend=1000)
    # R: which.max of only NA -> NA; the row is kept with all values NA
    assert np.isnan(w["wl"][1]) and np.isnan(w["dt_wl"][1]) and np.isnan(w["tide"][1])
    assert not np.isnan(w["wl"][0]) and np.isnan(w["u10"][0])  # before the wind (rule = 1)
    assert w["u10"][2] == 1.0


def _write_model(tmp_path, n=1000):
    dt, tide = _tide(n)
    tmin = (dt - T0) / 60
    for name, values in (("wl.nc", tide + 0.1), ("tide.nc", tide)):
        with netCDF4.Dataset(tmp_path / name, "w") as ds:
            ds.createDimension("time", n)
            t = ds.createVariable("time", "f8", ("time",))
            t.units = "minutes since 1987-01-01 00:00"
            t[:] = tmin
            v = ds.createVariable("vlissing", "f4", ("time",), fill_value=np.float32(-999))
            v[:] = values
    lon, lat = np.array([3.0, 3.5, 4.0]), np.array([51.0, 51.5])
    nt = n // 6
    with netCDF4.Dataset(tmp_path / "wind.nc", "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 3)
        t = ds.createVariable("time", "i4", ("time",))
        t.units = "hours since 1987-01-01 00:00:00"
        t[:] = np.arange(nt)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        for k in ("u10", "v10", "iews", "inss"):
            v = ds.createVariable(k, "i2", ("time", "lat", "lon"), fill_value=np.int16(-32767))
            v.scale_factor, v.add_offset = 0.5, 10.0
            v.set_auto_maskandscale(False)
            raw = np.zeros((nt, 2, 3), dtype=np.int16)
            raw[:, 1, 1] = 4  # nearest grid point of Vlissingen (51.42, 3.50): 10 + 0.5*4 = 12
            v[:] = raw


def test_read_mixed_file(tmp_path):
    _write_model(tmp_path)
    (path,) = read_mixed_file(
        tmp_path / "wl.nc",
        tmp_path / "tide.nc",
        tmp_path / "wind.nc",
        1,
        1000,
        "SEAS5_tide_198701-00",
        tmp_path / "out",
        stations=["vlissing"],
        lat=[51.42],
        lon=[3.50],
        discard_first_month=False,  # as readmixed.R; the series is shorter than a month
    )
    assert path.name == "vlissing_SEAS5_tide_198701-00_int.nc"
    w = read_nc(path, decode_times=False)
    np.testing.assert_allclose(w["wl"] - w["tide"], 0.1, atol=1e-6)
    ok = ~np.isnan(w["u10"])
    assert ok.sum() >= len(ok) - 1 and np.all(w["u10"][ok] == 12.0)
    assert w["dt_wl"][0] == T0 + 75 * 600  # first tidal high water


def test_month_start():
    assert month_start(T0 + 86400 * 10) == T0  # 1987-01-11 -> 1987-01-01
    assert month_start(T0, 1) == T0 + 31 * 86400  # 1987-02-01
    t = 951782400.0  # 2000-02-29 (leap year)
    assert month_start(t, 1) == t + 86400  # 2000-03-01


def test_read_mixed_discards_first_month(tmp_path):
    """Finding of the code review: the first month (spin-up) of a run must not be used, also when
    tstart in the selection file is 1 (member 00 of 198101-198106)."""
    n = 6 * 24 * 45  # 45 days from 1987-01-01
    _write_model(tmp_path, n)
    args = (tmp_path / "wl.nc", tmp_path / "tide.nc", tmp_path / "wind.nc", 1, n, "run")
    kw = dict(stations=["vlissing"], lat=[51.42], lon=[3.50])
    (fixed,) = read_mixed_file(*args, tmp_path / "fixed", **kw)
    (asr,) = read_mixed_file(*args, tmp_path / "r", discard_first_month=False, **kw)
    feb1 = T0 + 31 * 86400
    w, r = read_nc(fixed, decode_times=False), read_nc(asr, decode_times=False)
    assert r["dt_wl"][0] < feb1 and w["dt_wl"][0] >= feb1
    assert w["dt_wl"][0] - feb1 < 12.5 * 3600
    np.testing.assert_array_equal(w["wl"], r["wl"][r["dt_wl"] >= feb1])


def test_concat_wind_seas5_netcdf(tmp_path):
    """Finding of the code review: concat_wind.R cannot read the SEAS5 netCDF files; concat_wind
    reads them directly (nearest grid point, months 2-7 of the run)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    nt = 4 * 250  # 6-hourly, 250 days from 1987-01-01
    lon, lat = np.array([3.0, 3.5, 4.0]), np.array([51.0, 51.5])
    for member in range(2):
        with netCDF4.Dataset(raw / f"seasonal_system5.19870101.{member}.nc", "w") as ds:
            ds.createDimension("time", nt)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 3)
            t = ds.createVariable("time", "i4", ("time",))
            t.units = "hours since 1987-01-01 00:00:00"
            t[:] = np.arange(nt) * 6
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            for k in ("u10", "v10", "iews", "inss"):
                v = ds.createVariable(k, "f4", ("time", "lat", "lon"))
                x = np.ones((nt, 2, 3), dtype=np.float32)
                x[:, 1, 1] = 3.0  # nearest grid point of Vlissingen (51.42, 3.50)
                v[:] = x
    concat_wind(raw, stations=["vlissing"], out_path=tmp_path)
    d = read_seas5_wind("vlissing", tmp_path)
    assert d["dt"][0] == np.datetime64("1987-02-01T00:00")
    assert d["dt"][-1] == np.datetime64("1987-07-31T18:00")
    assert len(d["dt"]) == 2 * 4 * 181  # Feb-Jul 1987: 181 days, 2 members
    np.testing.assert_allclose(d["mwind"], 3 * np.sqrt(2))
    assert d["dwind"][0] == 225  # u = v = 3: wind from the south-west


def test_read_selection(tmp_path):
    f = tmp_path / "selection.txt"
    f.write_text(
        "#waterleve-file  tstart tend datestart dateend tideonly-file  wind+stress+msl-file\n"
        "/a/SEAS5_tide_198101-00.nc 00001 30528 01010000 07312350 /b/tide.nc /c/wind.nc\n"
    )
    (row,) = read_selection(f)
    assert row["tstart"] == 1 and row["tend"] == 30528 and row["wind"] == "/c/wind.nc"


def _wlevels(t, u10, v10, iews, inss, wl, tide):
    return {
        "tide": tide,
        "wl": wl,
        "dt_tide": t,
        "dt_wl": t,
        "u10": u10,
        "v10": v10,
        "iews": iews,
        "inss": inss,
    }


def test_concat_multi(tmp_path):
    st = tmp_path / "vlissing"
    st.mkdir()
    one = np.ones(4)
    # wind from the north (u = 0, v < 0) and from the east (u < 0, v = 0)
    a = _wlevels(T0 + np.arange(4.0), 0 * one, -one, -one, 0 * one, 2 * one, one)
    b = _wlevels(T0 + 10 + np.arange(4.0), -3 * one, 0 * one, 0 * one, -4 * one, 3 * one, one)
    b["wl"][2] = np.nan  # removed
    write_nc(st / "vlissing_SEAS5_tide_198101-00_int.nc", a, time_vars=("dt_tide", "dt_wl"))
    write_nc(st / "vlissing_SEAS5_tide_198107-00_int.nc", b, time_vars=("dt_tide", "dt_wl"))
    concat_multi(tmp_path, stations=["vlissing"])
    d = read_seas5_multi("vlissing", tmp_path)  # no .RData -> reads the .nc file
    assert len(d["wl"]) == 7 and d["timelength"][0] == 1.0
    np.testing.assert_array_equal(d["surge"], [1, 1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(d["dwind"], [0, 0, 0, 0, 90, 90, 90])
    np.testing.assert_array_equal(d["dstress"], [90, 90, 90, 90, 0, 0, 0])
    np.testing.assert_array_equal(d["mwind"], [1, 1, 1, 1, 3, 3, 3])
    np.testing.assert_array_equal(d["mstress"], [1, 1, 1, 1, 4, 4, 4])
    assert d["dt"].dtype.kind == "M" and d["dt"][4] == np.datetime64("1987-01-01T00:00:10")


def test_concat_wind(tmp_path):
    for k in range(2):
        with netCDF4.Dataset(tmp_path / f"wind_{k}.nc", "w") as ds:
            ds.createDimension("time", 3)
            ds.createDimension("station", 2)
            ds.createVariable("dt", "f8", ("time",))[:] = T0 + k * 1e6 + np.arange(3) * 21600
            for name, j in (("u10", 0), ("v10", 1), ("iews", 0), ("inss", 1)):
                v = ds.createVariable(name, "f8", ("time", "station"))
                v[:] = np.full((3, 2), -1.0) if j else np.zeros((3, 2))
    concat_wind(tmp_path, stations=["vlissing", "hoekvanh"])
    d = read_seas5_wind("hoekvanh", tmp_path)
    assert len(d["dt"]) == 6
    np.testing.assert_array_equal(d["dwind"], np.zeros(6))
    np.testing.assert_array_equal(d["mstress"], np.ones(6))
    # a second run ignores the output files of the first run
    concat_wind(tmp_path, stations=["vlissing", "hoekvanh"])
    assert len(read_seas5_wind("vlissing", tmp_path)["dt"]) == 6
