"""Readers for KNMI hourly wind data and SEAS5 wind stress (ports of ReadKNMIhourly.R, ReadKNMIpothourly.R)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pyreadr


def _knmi_sorted(data, icol_date, icol_hour, cols):
    """dt <- ymd_h(date*100 + hour); s <- sort.int(dt, index.return = TRUE): rows with NA dates are
    dropped by sort.int; hour 24 rolls over to 00 of the next day (as lubridate)."""
    ok = np.isfinite(data[:, icol_date]) & np.isfinite(data[:, icol_hour])
    data = data[ok]
    d = pd.to_datetime(
        pd.Series(data[:, icol_date].astype(np.int64)).astype(str), format="%Y%m%d"
    )
    dt = (d + pd.to_timedelta(data[:, icol_hour].astype(np.int64), unit="h")).to_numpy()
    order = np.argsort(dt, kind="stable")
    return {"date": dt[order], "data": data[order][:, cols]}


def _rbind(frames):
    """rbind(data.matrix(...)) of the files: by column position (the header line of some files is a data row)."""
    return np.vstack(
        [f.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) for f in frames]
    )


def read_knmi_hourly(station, datadir):
    """ReadKNMIhourly.R: uurgeg_<station>* files; returns dates and columns 4.. (DD, FH, ...)."""
    files = sorted(Path(datadir).glob(f"uurgeg_{station}*"))
    data = _rbind(
        [pd.read_csv(f, skiprows=31, header=0, skipinitialspace=True) for f in files]
    )
    return _knmi_sorted(data, 1, 2, slice(3, None))


def read_knmi_pothourly(station, datadir):
    """ReadKNMIpothourly.R: potwind_<station>* files; returns dates and columns 3 and 5."""
    files = sorted(Path(datadir).glob(f"potwind_{station}*"))
    data = _rbind(
        [pd.read_csv(f, skiprows=21, header=0, skipinitialspace=True) for f in files]
    )
    return _knmi_sorted(data, 0, 1, [2, 4])


def read_seas5_wind(station, input_path):
    """<station>_wind.RData: dt, mstress, dstress, mwind, dwind (6-hourly SEAS5)."""
    d = pyreadr.read_r(Path(input_path) / f"{station}_wind.RData")
    return {
        k: d[k][k].to_numpy()
        for k in ["dt", "mstress", "dstress", "mwind", "dwind"]
        if k in d
    }


def read_seas5_multi(station, input_path):
    """<station>_multi_int.RData: SEAS5-DCSM5 high-tide water levels with collocated wind."""
    d = pyreadr.read_r(Path(input_path) / f"{station}_multi_int.RData")
    return {k: d[k][k].to_numpy() for k in d.keys()}
