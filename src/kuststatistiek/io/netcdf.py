"""
NetCDF input and output for the preprocessing of the SEAS5 / DCSM-5 model data.

- `ncvar_get` and `nc_dim_time` read the model output as R's ncdf4 does (ncvar_get, nc$dim$time).
- `write_nc` and `read_nc` store the preprocessed series. Python cannot write R lists to .RData, so the
  Python preprocessing writes netCDF files with the same variable names as the .RData files of the R
  scripts. Times are stored as seconds since 1970-01-01 UTC (R's POSIXct).
"""

import numpy as np
import pandas as pd

EPOCH_UNITS = "seconds since 1970-01-01 00:00:00"
_UNIT_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}


def _netcdf4():
    import netCDF4  # imported here: only needed for the preprocessing and the netCDF input files

    return netCDF4


def _missing_mask(raw, var):
    """Values that ncdf4 sets to NA: _FillValue / missing_value (or the default fill value)."""
    netCDF4 = _netcdf4()
    attrs = var.ncattrs()
    missing = [var.getncattr(a) for a in ("_FillValue", "missing_value") if a in attrs]
    if not missing:
        key = var.dtype.str[1:]
        if key in netCDF4.default_fillvals:
            missing = [netCDF4.default_fillvals[key]]
    mask = np.isnan(raw)
    for mv in np.ravel(missing).astype(float):
        if np.isnan(mv):
            continue
        if np.issubdtype(
            var.dtype, np.floating
        ):  # ncdf4 compares floats with a relative tolerance
            mask |= np.abs(raw - mv) < abs(mv) * 1e-5
        else:
            mask |= raw == mv
    return mask


def ncvar_get(ds, name, index=None):
    """ncdf4::ncvar_get(nc, name): values as double, missing values as NaN, scale_factor and
    add_offset applied (in double, as in R). `index` maps dimension names to an index, e.g.
    {"lon": ilon, "lat": ilat} for ncvar_get(..., start = c(ilon, ilat, 1), count = c(1, 1, -1)).
    Degenerate dimensions are dropped (R: collapse_degen = TRUE)."""
    var = ds.variables[name]
    var.set_auto_maskandscale(False)
    index = index or {}
    idx = tuple(index.get(d, slice(None)) for d in var.dimensions)
    raw = np.asarray(var[idx], dtype=float)
    values = np.where(_missing_mask(raw, var), np.nan, raw)
    attrs = var.ncattrs()
    if "scale_factor" in attrs:
        values = values * float(var.getncattr("scale_factor"))
    if "add_offset" in attrs:
        values = values + float(var.getncattr("add_offset"))
    return np.squeeze(values)


def nc_dim_vals(ds, name):
    """nc$dim$<name>$vals: the coordinate values (1..n if there is no coordinate variable)."""
    if name in ds.variables:
        v = ds.variables[name]
        v.set_auto_maskandscale(False)
        return np.asarray(v[:], dtype=float).ravel()
    return np.arange(1, len(ds.dimensions[name]) + 1, dtype=float)


def nc_dim_time(ds, name="time"):
    """Times of dimension `name` as seconds since 1970-01-01 UTC (R: ymd_hm(...) + dminutes(vals)).
    Returns None if the reference time in the units cannot be read (R: off is NA)."""
    units = ds.variables[name].getncattr("units")
    unit, _, ref = units.partition(" since ")
    try:
        off = pd.Timestamp(ref.strip())
    except ValueError:
        return None
    if off is pd.NaT:
        return None
    if off.tzinfo is not None:
        off = off.tz_convert("UTC").tz_localize(None)
    factor = _UNIT_SECONDS[unit.strip().lower().rstrip("s")]
    return off.value / 1e9 + nc_dim_vals(ds, name) * factor


def seconds_to_datetime64(t):
    """POSIXct seconds -> datetime64[us] (NaN -> NaT), as pyreadr returns POSIXct."""
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, np.datetime64("NaT", "us"), dtype="datetime64[us]")
    ok = np.isfinite(t)
    out[ok] = np.round(t[ok] * 1e6).astype(np.int64).astype("datetime64[us]")
    return out


def write_nc(path, data, time_vars=(), dim="index", attrs=None):
    """Write 1-D series of equal length (and scalars) to a compressed netCDF file. Variables in
    `time_vars` are POSIXct seconds and get the units 'seconds since 1970-01-01 00:00:00'."""
    netCDF4 = _netcdf4()
    lengths = {np.size(v) for v in data.values() if np.ndim(v) > 0}
    if len(lengths) > 1:
        raise ValueError(f"series of unequal length: {lengths}")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension(dim, lengths.pop() if lengths else 0)
        for k, v in (attrs or {}).items():
            ds.setncattr(k, v)
        for name, values in data.items():
            values = np.asarray(values, dtype=float)
            if values.ndim == 0:
                var = ds.createVariable(name, "f8", ())
                var.assignValue(values)
                continue
            var = ds.createVariable(
                name, "f8", (dim,), zlib=True, complevel=4, fill_value=np.nan
            )
            if name in time_vars:
                var.units = EPOCH_UNITS
                var.calendar = "standard"
            var[:] = values


def read_nc(path, decode_times=True):
    """Read all variables of a file written by `write_nc` -> {name: 1-D numpy array}. With
    decode_times, time variables are returned as datetime64[us] (as pyreadr returns POSIXct)."""
    netCDF4 = _netcdf4()
    out = {}
    with netCDF4.Dataset(path) as ds:
        for name, var in ds.variables.items():
            var.set_auto_maskandscale(True)
            values = np.atleast_1d(
                np.ma.filled(np.ma.asarray(var[:], dtype=float), np.nan)
            )
            if decode_times and getattr(var, "units", None) == EPOCH_UNITS:
                values = seconds_to_datetime64(values)
            out[name] = values
    return out
