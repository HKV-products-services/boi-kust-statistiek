"""
Readers for measured high-tide water levels (ports of readwlmeas.R, readwlothermeas.R,
readwlmeastexeln.R) and collocation of water levels with measured wind (chunk
'Process measurements of wind direction and collocate with water level measurements'
of statistics-wl10-impr.Rmd).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from kuststatistiek.rutils import r_mean, r_loess_fit, r_loess_predict, r_approx
from kuststatistiek.io.read_wind import read_knmi_hourly


def _trend_correct(hwdate, hwvalues, trend_fit=True):
    """Common part of the readers: annual means, loess trend, detrending, cut at July 1 of the last year."""
    hwyear = hwdate.year.to_numpy()
    yr = pd.unique(hwyear)  # unique(year(hwdate)), order of appearance
    lyr = len(yr)
    ind = np.ones(lyr, dtype=bool)
    dates = set(hwdate.normalize())
    if pd.Timestamp(f"{yr.max()}-12-31") not in dates:  # !any(grepl(d1, hwdate))
        ind[lyr - 1] = False
    if pd.Timestamp(f"{yr.min()}-01-01") not in dates:
        ind[0] = False
    hwmean = np.full(lyr, np.nan)
    for i in np.flatnonzero(ind):
        hwmean[i] = r_mean(hwvalues[hwyear == yr[i]])
    if trend_fit:
        # loess(y ~ t, span = 84/length(yr), degree = 1, surface = "direct", statistics = "exact")
        mdl = r_loess_fit(
            yr.astype(float),
            hwmean,
            span=84 / len(yr),
            degree=1,
            surface="direct",
            statistics="exact",
        )
        trend = r_loess_predict(mdl, yr.astype(float))
        off = np.max(trend)
        trend = trend - off
        idx = hwyear - hwyear.min()  # trend[hwyear - min(hwyear) + 1]
        tr = np.full(len(idx), np.nan)
        ok = idx < len(trend)
        tr[ok] = trend[idx[ok]]
        hwstand = hwvalues - tr
        hwmean = hwmean - off
    else:  # Texel Noordzee: trend <- 0
        trend = 0.0
        hwstand = hwvalues.copy()
    d1 = pd.Timestamp(f"{yr.max()}-07-01")
    keep = hwdate < d1
    keep = np.asarray(keep)
    return {
        "t": hwdate[keep],
        "wspeed": hwstand[keep],
        "hwtrend": {"trend": trend, "yr": yr, "mean": hwmean},
    }


def read_wlmeas(path, filename):
    """readwlmeas.R: files vlis/hoek/ijmu/held/harl/delf.txt (flag 1 = high water)."""
    hw = pd.read_csv(
        Path(path) / f"{filename}.txt",
        sep=r"\s+",
        header=None,
        skiprows=5,
        usecols=[0, 1, 2, 3],
        dtype={0: str, 1: str},
    )
    hwdate = pd.to_datetime(hw[0] + " " + hw[1], format="%d-%m-%Y %H:%M")
    sel = (hw[2] == 1).to_numpy()
    return _trend_correct(
        pd.DatetimeIndex(hwdate[sel]), hw[3].to_numpy(dtype=float)[sel] * 0.01
    )


def read_wlothermeas(path, filename):
    """readwlothermeas.R: files hansweert/os11/denoever/westters/lauwerso/huibertg.txt ('hoogwater')."""
    hw = pd.read_csv(
        Path(path) / f"{filename}.txt",
        sep=r"\s+",
        header=None,
        skiprows=4,
        dtype={1: str, 2: str},
    )
    hwdate = pd.to_datetime(hw[1] + " " + hw[2], format="%d-%m-%Y %H:%M")
    sel = (hw[3] == "hoogwater").to_numpy()
    return _trend_correct(
        pd.DatetimeIndex(hwdate[sel]), hw[4].to_numpy(dtype=float)[sel] * 0.01
    )


def read_station_wlmeas(station, path, measfile):
    """Dispatch to readwlmeas / readwlothermeas as in the R scripts."""
    fname, fmt = measfile[station]
    return read_wlmeas(path, fname) if fmt == 1 else read_wlothermeas(path, fname)


def hwextract(data):
    """hwextract() of readwlmeastexeln.R and readmixed.R: peaks at least 54 samples (9 h) apart.

    Returns the high-water values and their 0-based indices in `data`."""
    lw = 6 * 9
    ld = len(data)
    with np.errstate(invalid="ignore"):
        ind = (
            np.flatnonzero(data[1 : ld - 1] >= np.maximum(data[: ld - 2], data[2:])) + 1
        )
    phw = data[ind]
    idd = np.flatnonzero(np.diff(ind) < lw)
    while len(idd) > 0:
        remove = idd[phw[idd] < phw[idd + 1]]
        if len(remove) > 0:
            ind = np.delete(ind, remove)
            phw = np.delete(phw, remove)
        idd = np.flatnonzero(np.diff(ind) < lw)
        if len(idd) > 0:
            remove = idd[phw[idd + 1] <= phw[idd]] + 1
            if len(remove) > 0:
                ind = np.delete(ind, remove)
                phw = np.delete(phw, remove)
            idd = np.flatnonzero(np.diff(ind) < lw)
    keep = phw > 0
    return phw[keep], ind[keep]


def read_wlmeastexeln(
    path, files=("texno1.csv", "texno2.csv", "texno3.csv", "texno4.csv")
):
    """readwlmeastexeln.R (the R script reads texno1-4; texno5 holds the data after 2017)."""
    parts = [
        pd.read_csv(Path(path) / f, sep=";", usecols=[91, 92, 94, 95], dtype=str)
        for f in files
    ]
    data = pd.concat(parts, ignore_index=True)
    waarde = pd.to_numeric(data["waarde"], errors="coerce").to_numpy(dtype=float)
    waarde[waarde > 99999999] = np.nan
    datetime = pd.to_datetime(
        data["datum"].str.strip() + " " + data["tijd"].str.strip(),
        format="%d-%m-%Y %H:%M",
    )
    dt = datetime.to_numpy()
    # remove the 2nd occurrence of duplicated times; as in R, unique() (order of appearance) is indexed
    # with the positions of the duplicates in table() (sorted order)
    ud = pd.unique(dt)
    _, counts = np.unique(dt, return_counts=True)
    nd = ud[np.flatnonzero(counts > 1)]
    keep = np.ones(len(dt), dtype=bool)
    for t in nd:
        ii = np.flatnonzero(dt == t)
        if len(ii) >= 2:
            keep[ii[1]] = False
    dt = dt[keep]
    waarde = waarde[keep]
    phw, ind = hwextract(waarde)
    return _trend_correct(pd.DatetimeIndex(dt[ind]), phw * 0.01, trend_fit=False)


def link_wlmeas_windmeas(wlm, dtm, windstation, datadir):
    """Collocate high-tide water levels with the vector-averaged measured wind of the preceding 3 hours."""
    A = read_knmi_hourly(windstation, datadir)
    dt_wind = A["date"]
    winddir = A["data"][:, 0].copy()
    windsp = A["data"][:, 1] * 0.1
    idd = winddir > 360
    winddir[idd] = np.nan
    windsp[idd] = np.nan
    wlmm = np.asarray(wlm, dtype=float).ravel()
    dtmm = np.asarray(dtm, dtype="datetime64[ns]")
    Nmm = len(wlmm)
    n = len(dt_wind)
    tw = (
        dt_wind.astype("datetime64[s]").astype(np.int64).astype(float)
    )  # POSIXct seconds
    tm = dtmm.astype("datetime64[s]").astype(np.int64).astype(float)
    ind = np.floor(
        r_approx(tw, np.arange(1, n + 1), tm)
    )  # floor(approx(dt_wind, 1:n, dtmm)$y)
    ok = ~np.isnan(ind)
    Nmm = Nmm * (ok.sum() / len(wlmm))
    ind = ind[ok].astype(np.int64)
    wlmm, dtmm = wlmm[ok], dtmm[ok]
    dc = np.pi / 180
    X0 = winddir[ind - 1] * dc
    X1 = winddir[np.maximum(1, ind - 1) - 1] * dc
    X2 = winddir[np.maximum(1, ind - 2) - 1] * dc
    Z0 = windsp[ind - 1]
    Z1 = windsp[np.maximum(1, ind - 1) - 1]
    Z2 = windsp[np.maximum(1, ind - 2) - 1]
    S = np.sin(X0) * Z0 + np.sin(X1) * Z1 + np.sin(X2) * Z2
    C = np.cos(X0) * Z0 + np.cos(X1) * Z1 + np.cos(X2) * Z2
    # R: (atan2(S, C)*(1/dc)) %% 360. For wind from (almost exactly) north atan2 is +-1e-14, and depending on
    # the platform %% 360 gives 0 or 360 deg, which fall in different 45-degree sectors. Here such angles are
    # set to 0 first, so north is always 0 deg, as in KNMI's delivered tables.
    a = np.arctan2(S, C) * (1 / dc)
    wd = np.mod(np.where(np.abs(a) < 1e-9, 0.0, a), 360)
    ws = (Z0 + Z1 + Z2) * (1 / 3)
    ok = ~np.isnan(wd + ws)
    Nmm = round(Nmm * (ok.sum() / len(wlmm)))
    return {
        "wlmm": wlmm[ok],
        "dtmm": dtmm[ok],
        "winddir": wd[ok],
        "windsp": ws[ok],
        "Nmm": Nmm,
    }
