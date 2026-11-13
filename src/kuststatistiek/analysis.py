"""
The computational chunks of the KNMI statistics scripts, one function per R chunk.

Every function returns a dict with the variable names of the corresponding R save() call.
Random draws are taken from an RRandom stream (kuststatistiek.rrandom) in the same order as
in R; the caller is responsible for calling the chunks in R's order.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from kuststatistiek.gw import FitGW_iHilli, esextr, taildir, collect, qGWdisp
from kuststatistiek.rutils import (
    r_approx,
    r_apply_sd,
    r_rowsums,
    r_loess_fit,
    r_loess_predict,
)


def _fixdata(var, N, pthreshold, es, theta0=None, theta0Std=None, caseId=None):
    return {
        "metadata": var.metadata(caseId),
        "N": N,
        "factor": var.factor,
        "pthreshold": pthreshold,
        "kthresh": 1,
        "mthresh": 1,
        "theta0": theta0,
        "theta0Std": theta0Std,
        "es": es,
        "x": var.x,
        "freq": var.freq,
    }


# ==========================================================================================
# SEAS5 / SEAS5-DCSM5 omni and directional tails
# (chunk 'GW omni and directional tail estimates from SEAS5' resp.
#  'OMNI GW tail estimates from SEAS5-DCSM5, and analysis of directionality of extremes')
# ==========================================================================================
def fit_seas5_tails(values, dirs, var, bins, caseId=None):
    X = -np.sort(-np.asarray(values, dtype=float))
    N = len(X)
    l0 = np.ceil(N * var.pthreshold).astype(np.int64)
    n = int(l0.max()) + 20
    esGWfull = FitGW_iHilli(
        X[:n], p=var.p, N=N, r11=1, sigma=np.inf, metadata=var.metadata(caseId)
    )
    esGWfull["freq"] = var.freq
    esGWfull["EI"] = var.EIvalue
    esGW = esextr(esGWfull, l0)
    k = var.kthresh - 1
    if (
        var.name == "wind"
    ):  # directional threshold mthresh (0.012), omni kthresh (0.003)
        fixdata = _fixdata(
            var,
            N,
            var.pthreshold[var.mthresh - 1],
            esextr(esGWfull, l0[k]),
            caseId=caseId,
        )
    else:  # water level: kthresh (0.012) for both
        fixdata = _fixdata(
            var, N, var.pthreshold[k], esextr(esGW, l0[k]), caseId=caseId
        )
    d = taildir(fixdata, values, dirs, bins)
    return {
        "esGWdir": d["esGWdir"],
        "esGW": esGW,
        "esGWfull": esGWfull,
        "pbin0": d["pbin0"],
        "x": d["x"],
        "mu": d["mu"],
        "Mu": d["Mu"],
        "mu0": d["mu0"],
        "Mu0": d["Mu0"],
        "qmu": d["qmu"],
        "Qfreq": d["Qfreq"],
        "Qfreq0": d["Qfreq0"],
    }


# ==========================================================================================
# block bootstrap of SEAS5 data (ensemble members = blocks)
# ==========================================================================================
def seas5_blocks(dt, rng, nb):
    """Blocks = 6-month SEAS5 runs, separated by date jumps; blocks <- sample(1:nblock, nb*m, TRUE)."""
    # seconds; via datetime64[ns] because pyreadr may return datetime64[us] (asi8 is then in us)
    t = np.asarray(dt, dtype="datetime64[ns]").astype(np.int64).astype(float) / 1e9
    ddt = np.diff(t) / 86400.0  # difftime(..., units = "days")
    idd = np.concatenate([[True], (ddt > 2) | (ddt < 0)])
    blockno = np.cumsum(idd)
    nblock = int(idd.sum())
    m = int(round(nblock))
    blocks = rng.sample(nblock, nb * m).reshape(nb, m).T  # dim(blocks) <- c(m, nb)
    return blockno, blocks


def draw_seas5_blocks_only(n_blocks, rng, nb):
    """Consume the random draws of seas5_blocks() for a station that is not computed."""
    rng.sample(n_blocks, nb * n_blocks)


def bootstrap_seas5(
    values,
    dirs,
    blockno,
    blocks,
    var,
    bins,
    caseId=None,
    dir_threshold=None,
    modelerror=None,
    rng=None,
    keep_quantiles=True,
):
    """Bootstrap of the omni and directional GW estimates from SEAS5 data.

    modelerror None: chunk 'Bootstrap ... WITHOUT model error' (shape estimated)
    modelerror dict(theta00bs, theta00dirbs): chunk 'WITH model error' (wind: shapes fixed at the
      perturbed values, stress scaled by exp(rnorm(1)*scalestd) drawn from `rng` per sample)
    dir_threshold: sample fraction for the directional fits (wind: mthresh, wl: kthresh)
    """
    values = np.asarray(values, dtype=float)
    dirs = np.asarray(dirs, dtype=float)
    lp, lf, nb = len(var.pthreshold), len(var.freq), blocks.shape[1]
    lcats, lcats0 = bins.lcats, bins.lcats0
    theta0bs = np.full((lp, nb), np.nan)
    theta0dirbs = np.full((lcats, nb), np.nan)
    Q0bs = np.full((lf, lcats0, nb), np.nan)
    Qbs = np.full((lf, lcats, nb), np.nan)
    qbs = np.full((lp, lf, nb), np.nan)
    k = var.kthresh - 1
    for ib in tqdm(range(nb), desc=f"bootstrap {caseId}", leave=False):
        ibs = collect(blockno, blocks[:, ib]) - 1
        vbs = values[ibs]
        if modelerror is not None and var.name == "wind":
            vbs = vbs * np.exp(rng.rnorm(1)[0] * var.scalestd)
        dbs = dirs[ibs]
        Nbs = len(vbs)
        X = -np.sort(-vbs)
        l0 = np.round(Nbs * var.pthreshold).astype(np.int64)
        n = int(l0.max()) + 20
        fixedpar = None
        if modelerror is not None:
            t = modelerror["theta00bs"][:, ib]
            fixedpar = {"theta0": t, "theta0Std": 0 * t}
        esGWbs = FitGW_iHilli(
            X[:n],
            p=var.p,
            N=Nbs,
            r11=1,
            fixedpar=fixedpar,
            l0=l0,
            sigma=np.inf,
            metadata=var.metadata(caseId),
        )
        esGWbs["freq"] = var.freq
        if modelerror is None:
            theta0bs[:, ib] = esGWbs["tailindex"]
        qbs[:, :, ib] = esGWbs["quantile"]
        td0 = td0s = None
        if modelerror is not None:
            td0 = modelerror["theta00dirbs"][:, ib]
            td0s = td0 * 0
        fixdata = _fixdata(
            var, Nbs, dir_threshold, esextr(esGWbs, l0[k]), td0, td0s, caseId
        )
        d = taildir(fixdata, vbs, dbs, bins)
        if modelerror is None:
            for i in range(lcats):
                theta0dirbs[i, ib] = min(1.0, float(d["esGWdir"][i]["tailindex"][0]))
        Qbs[:, :, ib] = d["Qfreq"]
        Q0bs[:, :, ib] = d["Qfreq0"]
    out = {"theta0bs": theta0bs, "theta0dirbs": theta0dirbs}
    if keep_quantiles:
        out.update(
            {
                "Qbs": Qbs,
                "Q0bs": Q0bs,
                "qbs": qbs,
                "Qsd": r_apply_sd(Qbs, 2),
                "Q0sd": r_apply_sd(Q0bs, 2),
                "qsd": r_apply_sd(qbs, 2),
                "q0bs": qbs[k, :, :],
            }
        )
    return out


def model_error_draws(theta0bs, theta0dirbs, var, rng):
    """theta00bs <- theta0bs + modelstd*rep(rnorm(nb), each= lp);
    theta00dirbs <- pmin(1, theta0dirbs + modelstd*rnorm(lcats*nb)); dim(.) <- c(lcats, nb)"""
    lp, nb = theta0bs.shape
    lcats = theta0dirbs.shape[0]
    theta00bs = theta0bs + var.modelstd * rng.rnorm(nb)[None, :]
    theta00dirbs = np.minimum(
        1, theta0dirbs + var.modelstd * rng.rnorm(lcats * nb).reshape(nb, lcats).T
    )
    return theta00bs, theta00dirbs


# ==========================================================================================
# water level: measurements with SEAS5-DCSM5 shape parameters
# ==========================================================================================
def fit_meas_fixed_shape(stats_multi, wlm, wlmm, winddir, var, bins, caseId=None):
    """Chunk 'OMNI GW tail estimates from measurements with shape parameter from SEAS5-DCSM5 data'.

    Returns (meas0, esmeasGW): meas0 with the variables of <station>_stats_multi_meas0.RData.
    """
    esGW, esGWdir, esGWfull = (
        stats_multi["esGW"],
        stats_multi["esGWdir"],
        stats_multi["esGWfull"],
    )
    theta0 = np.asarray(esGW["tailindex"], dtype=float)
    theta0dir = np.array(
        [esGWdir[i]["tailindex"][0] for i in range(bins.lcats)], dtype=float
    )
    md = {"timelength": None, "caseId": caseId, "varname": var.varname}
    X = -np.sort(-np.asarray(wlm, dtype=float).ravel())
    Nm = len(X)
    # all thresholds, without use of SEAS5 shapes
    pfullmax = np.exp(-np.min(esGWfull["y"]))
    n = int(np.ceil(Nm * pfullmax))
    esmeasGWpure = FitGW_iHilli(X[:n], var.p, N=Nm, r11=1, sigma=np.inf, metadata=md)
    esmeasGWpure["freq"] = var.freq
    # all thresholds, with use of SEAS5 shapes
    theta0full = r_approx(esGWfull["y"], esGWfull["tailindex"], esmeasGWpure["y"])
    esmeasGWfull = FitGW_iHilli(
        X[:n],
        var.p,
        N=Nm,
        r11=1,
        l0=esmeasGWpure["l"],
        fixedpar={"theta0": theta0full, "theta0Std": theta0full * 0},
        sigma=np.inf,
        metadata=md,
    )
    esmeasGWfull["freq"] = var.freq
    # standard thresholds, with use of SEAS5 shapes
    l0 = np.maximum(10, np.round(Nm * var.pthreshold)).astype(np.int64)
    n = int(l0.max()) + 10
    esmeasGW = FitGW_iHilli(
        X[:n],
        var.p,
        N=Nm,
        r11=1,
        fixedpar={"theta0": theta0, "theta0Std": theta0 * 0},
        l0=l0,
        sigma=np.inf,
        metadata=md,
    )
    esmeasGW["freq"] = var.freq
    fixdata = {
        "metadata": md,
        "N": len(wlmm),
        "factor": var.factor,
        "pthreshold": var.pthreshold[var.mthresh - 1],
        "kthresh": 1,
        "mthresh": 1,
        "theta0": theta0dir,
        "theta0Std": theta0dir * 0,
        "es": esextr(esmeasGW, l0[var.kthresh - 1]),
        "x": var.x,
        "freq": var.freq,
    }
    d = taildir(fixdata, wlmm, winddir, bins)
    meas0 = {
        "esmeasGWpure": esmeasGWpure,
        "esmeasGWfull": esmeasGWfull,
        "x": var.x,
        "mum": d["mu"],
        "Mum": d["Mu"],
        "pbin0m": d["pbin0"],
        "mum0": d["mu0"],
        "Mum0": d["Mu0"],
        "qmum": d["qmu"],
        "freq": var.freq,
        "Qfreqm": d["Qfreq"],
        "Qfreq0m": d["Qfreq0"],
    }
    return meas0, esmeasGW


def meas_bootstrap_years(wlm, dtm, wlmm, dtmm, winddir, rng, nb):
    """Year resampling of chunk 'Bootstrap uncertainty estimation without model error'
    (contents of <station>_meas_wl_bootdata.RData)."""
    dtm = pd.DatetimeIndex(dtm)
    dtmm = pd.DatetimeIndex(dtmm)
    ind = ~dtm.isin(dtmm)  # water levels without collocated wind
    dtm = dtm[ind]
    wlm = np.asarray(wlm, dtype=float).ravel()[ind]
    yrmm = pd.unique(dtmm.year.to_numpy())
    yrm = pd.unique(dtm.year.to_numpy())
    yrmbs = rng.sample(yrm, nb * len(yrm)).reshape(nb, len(yrm)).T
    yrmmbs = rng.sample(yrmm, nb * len(yrmm)).reshape(nb, len(yrmm)).T
    return {
        "nb": nb,
        "yrmbs": yrmbs,
        "yrmmbs": yrmmbs,
        "wlm": wlm,
        "dtm": dtm.to_numpy(),
        "dtmm": dtmm.to_numpy(),
        "wlmm": np.asarray(wlmm, dtype=float),
        "winddir": np.asarray(winddir, dtype=float),
    }


def meas_bootstrap(bootdata, theta_omni, theta_dir, var, bins, caseId=None, pure=False):
    """Loop of the measurement bootstraps (meas1: pure=True with theta0bs; meas2: theta00bs, model error).

    Returns qmbs, Qmbs, Q0mbs, Qmsd, Q0msd, qmsd (and qmpurebs, thetampurebs if pure=True).
    """
    lp, lf = len(var.pthreshold), len(var.freq)
    nb = bootdata["yrmbs"].shape[1]
    Q0mbs = np.full((lf, bins.lcats0, nb), np.nan)
    Qmbs = np.full((lf, bins.lcats, nb), np.nan)
    qmbs = np.full((lp, lf, nb), np.nan)
    qmpurebs = np.full((lp, lf, nb), np.nan)
    thetampurebs = np.full((lp, nb), np.nan)
    yr_mm = pd.DatetimeIndex(bootdata["dtmm"]).year.to_numpy()
    yr_m = pd.DatetimeIndex(bootdata["dtm"]).year.to_numpy()
    md = {"timelength": None, "caseId": caseId, "varname": var.varname}
    for ib in tqdm(range(nb), desc=f"meas bootstrap {caseId}", leave=False):
        ibs = collect(yr_mm, bootdata["yrmmbs"][:, ib]) - 1
        wlmmbs = bootdata["wlmm"][ibs]
        winddirbs = bootdata["winddir"][ibs]
        ibs = collect(yr_m, bootdata["yrmbs"][:, ib]) - 1
        wlmbs = np.concatenate([wlmmbs, bootdata["wlm"][ibs]])
        X = -np.sort(-wlmbs)
        Nbs = len(X)
        l0 = np.round(Nbs * var.pthreshold).astype(np.int64)
        n = int(l0.max()) + 20
        if pure:
            es = FitGW_iHilli(
                X[:n], var.p, N=Nbs, r11=1, l0=l0, sigma=np.inf, metadata=md
            )
            qmpurebs[:, :, ib] = es["quantile"]
            thetampurebs[:, ib] = es["tailindex"]
        t = theta_omni[:, ib]
        esmeasbs = FitGW_iHilli(
            X[:n],
            var.p,
            N=Nbs,
            r11=1,
            fixedpar={"theta0": t, "theta0Std": 0 * t},
            l0=l0,
            sigma=np.inf,
            metadata=md,
        )
        esmeasbs["freq"] = var.freq
        qmbs[:, :, ib] = esmeasbs["quantile"]
        td = theta_dir[:, ib]
        fixdata = {
            "metadata": md,
            "N": len(wlmmbs),
            "factor": var.factor,
            "pthreshold": var.pthreshold[var.mthresh - 1],
            "kthresh": 1,
            "mthresh": 1,
            "theta0": td,
            "theta0Std": td * 0,
            "es": esextr(esmeasbs, l0[var.kthresh - 1]),
            "x": var.x,
            "freq": var.freq,
        }
        d = taildir(fixdata, wlmmbs, winddirbs, bins)
        Qmbs[:, :, ib] = d["Qfreq"]
        Q0mbs[:, :, ib] = d["Qfreq0"]
    out = {
        "qmbs": qmbs,
        "Qmbs": Qmbs,
        "Q0mbs": Q0mbs,
        "Qmsd": r_apply_sd(Qmbs, 2),
        "Q0msd": r_apply_sd(Q0mbs, 2),
        "qmsd": r_apply_sd(qmbs, 2),
    }
    if pure:
        out.update({"qmpurebs": qmpurebs, "thetampurebs": thetampurebs})
    return out


def _smooth_B(A, cats0, min_finite=None):
    """B <- loess smoothed log-ratios A over directions (3 copies of the circle, middle one kept)."""
    lc0 = len(cats0)
    ca = np.concatenate([cats0 - 360, cats0, cats0 + 360])
    B = np.full_like(A, np.nan)
    for k in range(A.shape[0]):
        a = np.tile(A[k, :], 3)
        ok = ~np.isnan(a) & np.isfinite(a)
        use = np.any(ok) if min_finite is None else ok.sum() > min_finite
        if use:
            m = r_loess_fit(ca[ok], a[ok], span=1 / 4, degree=1)
            if np.all(ok):
                b = m.outputs.fitted_values
            else:
                b = r_loess_predict_inside(m, ca, ca[ok])
            B[k, :] = b[lc0 : 2 * lc0]
    return B


def r_loess_predict_inside(m, newx, xfit):
    """predict.loess with surface = 'interpolate': NA outside the range of the data."""
    out = np.full(len(newx), np.nan)
    inside = (newx >= xfit.min()) & (newx <= xfit.max())
    out[inside] = r_loess_predict(m, newx[inside])
    return out


def smooth_directionality(stats_multi, meas2, var, bins):
    """Chunk 'Smooth the directionality' (contents of <station>_stats_multi_meas3.RData)."""
    Mu0, mu0 = stats_multi["Mu0"], stats_multi["mu0"]
    Mum0, mum0 = meas2["Mum0"], meas2["mum0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.log(Mum0 / mum0[:, None]) / np.log(Mu0 / mu0[:, None])
        B = _smooth_B(A, bins.cats0)
        Mum0 = np.exp(B * np.log(Mu0 / mu0[:, None])) * mum0[:, None]
        temp = r_rowsums(Mum0)
        Mum0 = Mum0 * (np.exp(-var.x) * var.factor / temp)[:, None]
    lfreq = np.log(var.freq)
    qq = meas2["qmum"]["quantile"]
    Qfreq0m = np.full((len(var.freq), bins.lcats0), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(bins.lcats0):
            temp = np.log(Mum0[:, i])
            ok = ~np.isnan(temp)
            Qfreq0m[:, i] = r_approx(temp[ok], qq[ok], lfreq)
    Qfreqm = np.array(meas2["Qfreqm"], dtype=float, copy=True)
    Qfreqm[:, bins.lcats0] = r_approx(np.log(meas2["mum"]), qq, lfreq)
    Qfreqm[:, : bins.lcats0] = np.nan
    out = dict(meas2)
    out.update({"Mum0": Mum0, "Qfreq0m": Qfreq0m, "Qfreqm": Qfreqm})
    return out


# ==========================================================================================
# other stations (statistics-wl10others-impr.Rmd, chunk 'Make the combinations ...')
# ==========================================================================================
def combine_other_station(
    own_stats_multi,
    omniref_stats_multi,
    omniref_meas2,
    dirref_stats_multi,
    dirref_meas2,
    shift,
    var,
    bins,
    station,
):
    lc0 = bins.lcats0
    lfreq = np.log(var.freq)
    k = var.kthresh - 1
    # Q-ratios from the omni reference station
    Qratio_omni = (
        omniref_meas2["qmum"]["quantile"] / omniref_stats_multi["qmu"]["quantile"]
    )
    Qmbs_ref = omniref_meas2["Qmbs"]
    Qfreqm_ref = omniref_meas2["Qfreqm"]
    Qfreq0m_ref = omniref_meas2["Qfreq0m"]
    if station == "os11":
        Qratio_omni = Qratio_omni * 0.98  # it decreases from N to S
    # log-Mu-ratios from the directional reference station
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.log(dirref_meas2["Mum0"] / dirref_meas2["mum0"][:, None]) / np.log(
            dirref_stats_multi["Mu0"] / dirref_stats_multi["mu0"][:, None]
        )
    B = _smooth_B(A, bins.cats0, min_finite=3 * lc0 - 7)  # if (sum(id) > length(a)-7)
    # apply to the estimates from SEAS5-DCSM5 data of the station itself
    Mu0, mu0, mu, esGW = (
        own_stats_multi["Mu0"],
        own_stats_multi["mu0"],
        own_stats_multi["mu"],
        own_stats_multi["esGW"],
    )
    mum0 = dirref_meas2["mum0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        Mum0 = np.exp(B * np.log(Mu0 / mu0[:, None])) * mum0[:, None]
        temp = r_rowsums(Mum0)
        Mum0 = Mum0 * (np.exp(-var.x) * var.factor / temp)[:, None]
    p = esGW["k"][k] / esGW["N"]
    qmu = qGWdisp(
        mu / var.factor,
        p,
        esGW["location"][k],
        esGW["logdisp"][k],
        esGW["tailindex"][k],
        esGW["locationStd"][k],
        esGW["logdispStd"][k],
        esGW["tailindexStd"][k],
    )
    qmum = dict(qmu)
    qmum["quantile"] = qmu["quantile"] * Qratio_omni + shift
    Qfreq0m = np.full((len(var.freq), lc0), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(lc0):
            temp = np.log(Mum0[:, i])
            ok = ~np.isnan(temp)
            Qfreq0m[:, i] = r_approx(temp[ok], qmum["quantile"][ok], lfreq)
    mum = dirref_meas2["mum"]
    Qfreqm = np.array(dirref_meas2["Qfreqm"], dtype=float, copy=True)
    Qfreqm[:, lc0] = r_approx(np.log(mum), qmum["quantile"], lfreq)
    Qfreqm[:, :lc0] = np.nan
    # quantile map of the bootstrap samples of the omni reference station (loess, surface = "direct")
    nb = Qmbs_ref.shape[2]
    Qmbs = np.full_like(dirref_meas2["Qmbs"], np.nan)
    m = r_loess_fit(
        Qfreqm_ref[:, lc0], Qfreqm[:, lc0], span=0.75, degree=2, surface="direct"
    )
    for ib in range(nb):
        Qmbs[:, lc0, ib] = r_loess_predict(m, Qmbs_ref[:, lc0, ib])
    Q0mbs = np.full_like(dirref_meas2["Q0mbs"], np.nan)
    for i in range(lc0):
        m = r_loess_fit(
            Qfreq0m_ref[:, i], Qfreq0m[:, i], span=0.75, degree=2, surface="direct"
        )
        for ib in range(nb):
            # NB: as in the R script, the wide-sector samples Qmbs_ref are mapped here (not Q0mbs_ref)
            Q0mbs[:, i, ib] = r_loess_predict(m, Qmbs_ref[:, i, ib])
    return {
        "x": var.x,
        "mum": mum,
        "Mum": dirref_meas2["Mum"],
        "pbin0m": dirref_meas2["pbin0m"],
        "mum0": mum0,
        "Mum0": Mum0,
        "qmum": qmum,
        "freq": var.freq,
        "Qfreqm": Qfreqm,
        "Qfreq0m": Qfreq0m,
        "Qmbs": Qmbs,
        "Q0mbs": Q0mbs,
        "Qmsd": r_apply_sd(Qmbs, 2),
        "Q0msd": r_apply_sd(Q0mbs, 2),
    }
