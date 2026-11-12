"""
Production of the BOI tables (ports of production-wind10-impr.Rmd, production-wl10extended-impr.Rmd
and the table chunk of statistics-wl10-omni-texeln-impr.Rmd).

Output files have the same names and the same text format (write.csv) as the R scripts.
"""

import numpy as np

from kuststatistiek.evtools import Wfit, tau2upot, correctfrequencies
from kuststatistiek.gw import assigncat
from kuststatistiek.rutils import (
    r_approx,
    r_signif,
    r_sd,
    r_sum,
    r_matvec,
    r_as_character,
    write_r_csv,
    save_pkl,
)

DELTA = (
    1 / 365.25
)  # time step in years for tables of 12-hourly maxima in the winter half year


def r_seq_by(frm, to, by):
    """seq(from, to, by) as in R's seq.default."""
    n = int((to - frm) / by + 1e-10)
    x = frm + np.arange(n + 1) * by
    return np.minimum(x, to) if by > 0 else np.maximum(x, to)


def _rownames(rp, sdname):
    return (
        ["shape", "scale", "reffreq"]
        + [f"RV {r_as_character(r)} yr" for r in rp]
        + [f"{sdname} {r_as_character(r)} yr" for r in rp]
    )


def _colnames(cats):
    return ["omni"] + [r_as_character(c) for c in cats]


def _rev_sd(a):
    """rev(apply(a, 1, sd, na.rm = TRUE)) for a (freq x samples) array."""
    return np.array([r_sd(row) for row in a])[::-1]


def _write_wbltail(path, W, rownames, colnames):
    W = r_signif(W, 6)
    write_r_csv(path, [W[:, j] for j in range(W.shape[1])], colnames, rownames=rownames)
    return W


def _write_table(path, x, pc0, Pc0, firstcol, cats):
    T = np.column_stack([x, r_signif(pc0, 4), r_signif(Pc0, 4)])
    write_r_csv(
        path,
        [T[:, j] for j in range(T.shape[1])],
        [firstcol, "omni"] + [r_as_character(c) for c in cats],
    )
    return T


def read_direction_table(path):
    """Probabilities of the sectors from the windrichting csv (R: read.csv(skip = 1)"""
    lines = path.read_text().splitlines()[1:]
    return np.array([float(v) for v in lines[1].split(",")[1:]])


def read_winddirs_rdata(path):
    """load("hoekvanh_winddirs.RData"): Table22h[2, ], Table30[2, ] (sector probabilities)."""
    from kuststatistiek.io.rdata import read_rdata

    r = read_rdata(path)
    return r["Table22h"][1].copy(), r["Table30"][1].copy()


# ==========================================================================================
def direction_distribution(dstress, pbins, paths):
    """Chunk 'Distribution of wind direction at Hoek van holland from SEAS5 data'."""
    out = {}
    for tag, lb, ub, cats, title in [
        (
            "22hdeg",
            pbins.lbin0,
            pbins.ubin0,
            pbins.cats0,
            "Windsnelheidsverdeling Hoek van Holland (22.5 graden)",
        ),
        (
            "30deg",
            pbins.lbin30,
            pbins.ubin30,
            pbins.cats30,
            "Windsnelheidsverdeling Hoek van Holland (30 graden)",
        ),
    ]:
        cat = assigncat(dstress, lb, ub)
        pdir = np.round(cat.mean(axis=0), 4)
        dif = r_sum(pdir) - 1
        ii = int(np.argmax(pdir))
        pdir[ii] = pdir[ii] - dif
        fname = paths.out(f"hoekvanh_2035_windrichting_{tag}_versionFeb2026.csv")
        rows = [
            title,
            ",".join(['"richting"'] + [r_as_character(c) for c in cats]),
            ",".join(['"kans"'] + [r_as_character(v) for v in pdir]),
        ]
        fname.write_text(
            "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
        )  # LF line ends, as KNMI's tables
        out[tag] = pdir
    return out


# ==========================================================================================
def _cond_prob_wind(qwind, mu0, Mu, ptab):
    """Conditional 12-hourly exceedance probability tables (potential wind, 1 m/s steps)."""
    qlim = (int(np.ceil(np.nanmin(qwind))), int(np.floor(np.nanmax(qwind))))
    x = np.arange(1, qlim[1] + 1)
    lx, lc = len(x), Mu.shape[1]
    qrange = np.flatnonzero((x >= qlim[0]) & (x <= qlim[1]))
    Pc0 = np.full((lx, lc), np.nan)
    MU = np.full((lx, lc), np.nan)
    mU = np.full(lx, np.nan)

    def extrap(pc):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log(-np.log(pc))
            q1 = qlim[0]
            if q1 > 1:
                idx = np.arange(1, q1)  # 1:(qlim[1]-1)
                z[idx - 1] = z[q1 - 1] + (idx - q1) * (z[q1] - z[q1 - 1])
                pc[idx - 1] = np.exp(-np.exp(z[idx - 1]))
        return pc

    ind = ~np.isnan(mu0) & (mu0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mU[qrange] = np.exp(
            r_approx(qwind[ind], np.log(mu0[ind]), x[qrange].astype(float))
        )
        pc0 = extrap(np.maximum(0, 1 - np.exp(-mU * DELTA)))
        for j in range(lc):
            ind = ~np.isnan(Mu[:, j]) & (Mu[:, j] > 0)
            MU[qrange, j] = np.exp(
                r_approx(qwind[ind], np.log(Mu[ind, j]), x[qrange].astype(float))
            )
            Pc0[:, j] = extrap(np.maximum(0, 1 - np.exp(-MU[:, j] * (DELTA / ptab[j]))))
    return x, pc0, Pc0


def _cond_prob_wl(qwl, mum0, Mum, ptab):
    """Conditional 12-hourly exceedance probability tables (water level, 0.1 m steps)."""
    qlim = np.array([np.ceil(np.min(qwl) * 10), np.floor(np.max(qwl) * 10)]) * 0.1
    x = r_seq_by(0.1, qlim[1], 0.1)
    lx = len(x)
    qrange = np.flatnonzero((x >= qlim[0]) & (x <= qlim[1]))
    Pc0 = None
    mU = np.full(lx, np.nan)
    ind = ~np.isnan(mum0) & (mum0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mU[qrange] = np.exp(r_approx(qwl[ind], np.log(mum0[ind]), x[qrange]))
        pc0 = np.maximum(0, 1 - np.exp(-mU * DELTA))
        if Mum is not None:
            lc = Mum.shape[1]
            Pc0 = np.full((lx, lc), np.nan)
            for j in range(lc):
                ind = ~np.isnan(Mum[:, j]) & (Mum[:, j] > 0)
                MU = np.exp(r_approx(qwl[ind], np.log(Mum[ind, j]), x[qrange]))
                col = np.full(lx, np.nan)
                col[qrange] = MU
                Pc0[:, j] = np.maximum(0, 1 - np.exp(-col * (DELTA / ptab[j])))
    # if (qrange[1] > min(x)): rows 1:(qrange[1]-1) set to 1 (note: 1:0 = c(1, 0) sets row 1)
    q1 = qrange[0] + 1
    rows = np.arange(0, max(q1 - 1, 1))
    if Pc0 is not None:
        Pc0[rows, :] = 1
    pc0[rows] = 1
    return x, pc0, Pc0


# ==========================================================================================
def weibull_refit_wind(
    station,
    boot2,
    eimdl,
    EIvalue,
    factor,
    freq,
    pbins,
    ptab,
    dirbins,
    paths,
    resolutionfactor=1.15,
):
    """Chunks 'Refit ordinary 2-par. Weibull tails ...' (dirbins '22h') and 'Same refit for 30 deg' ('30')."""
    lf = len(freq)
    lf2 = lf - 2
    freq2 = freq[:lf2]
    rp2 = 1 / freq2[::-1]
    Mu0, mu0 = np.array(boot2["Mu0"], dtype=float), np.array(boot2["mu0"], dtype=float)
    mu00 = mu0.copy()
    corrfactor = correctfrequencies(mu0 / factor, eimdl, EIvalue)
    mu0 = mu0 * corrfactor
    Mu0 = Mu0 * corrfactor[:, None]
    tag = "22hdeg" if dirbins == "22h" else "30deg"
    cats = pbins.cats0 if dirbins == "22h" else pbins.cats30
    lc = len(cats)
    npar = 3
    W = np.full((npar + lf2 * 2, lc + 1), np.nan)
    qstress = boot2["qmu"]["quantile"]
    qwind = tau2upot(qstress * resolutionfactor)
    # correct bootstrap samples q0bs and Q0bs for the change in frequency (generates NaN's)
    qmod = r_approx(mu0, qstress, mu00)
    q0modbs = r_approx(np.append(qstress, 100), np.append(qmod, 100), boot2["q0bs"])
    Q0modbs = r_approx(np.append(qstress, 100), np.append(qmod, 100), boot2["Q0bs"])
    q0bs_wind = tau2upot(q0modbs * resolutionfactor)
    Q0bs_wind = tau2upot(Q0modbs * resolutionfactor)
    if dirbins == "30":
        nb = boot2["q0bs"].shape[1]
        Mu30 = np.full((Mu0.shape[0], lc), np.nan)
        weight = np.full((len(pbins.lbin0), lc), np.nan)
        for i in range(lc):
            lam = np.minimum(1, np.maximum(0, pbins.ubin30[i] - pbins.lbin0) / 22.5)
            nu = np.minimum(1, np.maximum(0, pbins.ubin00 - pbins.lbin30[i]) / 22.5)
            w = lam + nu - 1
            if pbins.ubin30[i] < pbins.lbin30[i]:
                w = w + 1
            weight[:, i] = w
            Mu30[:, i] = r_matvec(Mu0, w)
        qw = r_seq_by(1, np.ceil(np.nanmax(qwind)) + 5, 1)
        Mu0bs = np.full((len(qw), len(pbins.lbin0)), np.nan)
        Q30bs = np.full((lf2, lc, nb), np.nan)
        for ib in range(nb):
            for i in range(len(pbins.lbin0)):
                temp = Q0bs_wind[:lf2, i, ib]
                it = ~np.isnan(temp)
                if (
                    it.sum() > lf2 - 2
                ):  # otherwise Mu0bs[, i] keeps its previous value (as in R)
                    esbs = Wfit(freq2[it], temp[it], 2)
                    Mu0bs[:, i] = esbs["mu0"] * np.exp(
                        -((qw / esbs["sigma"]) ** (1 / esbs["rho"]))
                    )
            for i in range(lc):
                with np.errstate(divide="ignore", invalid="ignore"):
                    Q30bs[:, i, ib] = r_approx(
                        np.log(r_matvec(Mu0bs, weight[:, i])), qw, np.log(freq2)
                    )
        Mudir = Mu30
    else:
        Mudir = Mu0
    # omni
    ind = (mu0 <= 1) & (mu0 > 1e-10)
    es = Wfit(mu0[ind], qwind[ind], 2)
    W[0, 0], W[1, 0], W[2, 0] = es["rho"], es["sigma"], es["mu0"]
    W[npar : npar + lf2, 0] = r_signif(
        es["sigma"] * np.log(es["mu0"] * rp2) ** es["rho"], 3
    )
    mufit = es["mu0"] * np.exp(-((qwind / es["sigma"]) ** (1 / es["rho"])))
    mu0[ind] = mufit[ind]
    temp = _rev_sd(q0bs_wind)[2:lf]
    W[npar + lf2 :, 0] = r_signif(temp, 3) / W[npar : npar + lf2, 0]  # relative errors
    # directional
    for i in range(lc):
        mu1 = Mudir[:, i].copy()
        ind = (mu1 <= 1) & (mu1 > 1e-10)
        es = Wfit(mu1[ind], qwind[ind], 2)
        W[0, i + 1], W[1, i + 1], W[2, i + 1] = es["rho"], es["sigma"], es["mu0"]
        W[npar : npar + lf2, i + 1] = r_signif(
            es["sigma"] * np.log(es["mu0"] * rp2) ** es["rho"], 3
        )
        mufit = es["mu0"] * np.exp(-((qwind / es["sigma"]) ** (1 / es["rho"])))
        mu1[ind] = mufit[ind]
        Mudir[:, i] = mu1
        temp = (
            _rev_sd(Q0bs_wind[:, i, :])[2:lf]
            if dirbins == "22h"
            else _rev_sd(Q30bs[:, i, :])
        )
        W[npar + lf2 :, i + 1] = r_signif(temp, 3) / W[npar : npar + lf2, i + 1]
    x, pc0, Pc0 = _cond_prob_wind(qwind, mu0, Mudir, ptab)
    T = _write_table(
        paths.out(f"{station}_2023_potwind_condprob_12hour_{tag}_versionFeb2026.csv"),
        x,
        pc0,
        Pc0,
        "potential wind speed [m/s]",
        cats,
    )
    W = _write_wbltail(
        paths.out(f"{station}_2023_potwind_Wbltails_{tag}_versionFeb2026.csv"),
        W,
        _rownames(rp2, "SDrel"),
        _colnames(cats),
    )
    if dirbins == "22h":
        save_pkl(
            paths.out(f"{station}_2023_potwind_stats_refit_22hdeg_versionFeb2026.pkl"),
            cats0=cats,
            lcats0=lc,
            Wbltail=W,
            Table=T,
            mu0=mu0,
            Mu0=Mudir,
            qwind=qwind,
            qstress=qstress,
        )
    else:
        save_pkl(
            paths.out(
                f"{station}_2023_potwind_stats_wind_refit_30deg_versionFeb2026.pkl"
            ),
            cats30=cats,
            lcats30=lc,
            Wbltail=W,
            Table=T,
            mu0=mu0,
            Mu30=Mudir,
            qwind=qwind,
            qstress=qstress,
        )
    return W, T


# ==========================================================================================
def weibull_refit_wl(station, meas3, freq, pbins, ptab, dirbins, paths, SLR):
    """production-wl10extended-impr.Rmd: 22.5 deg (dirbins '22h') and 30 deg ('30') tables."""
    lf = len(freq)
    lf1, lf2 = lf - 1, lf - 2
    freq1, freq2 = freq[:lf1], freq[:lf2]
    rp1 = 1 / freq1[::-1]
    tag = "22hdeg" if dirbins == "22h" else "30deg"
    cats = pbins.cats0 if dirbins == "22h" else pbins.cats30
    lc = len(cats)
    npar = 3
    W = np.full((npar + lf1 * 2, lc + 1), np.nan)
    qwl = meas3["qmum"]["quantile"] + SLR
    mum0 = np.array(meas3["mum0"], dtype=float)
    Mum0 = np.array(meas3["Mum0"], dtype=float)
    if dirbins == "30":
        Q0mbs = meas3["Q0mbs"]
        nb = meas3["Qmbs"].shape[2]
        Mum30 = np.full((Mum0.shape[0], lc), np.nan)
        weight = np.full((len(pbins.lbin0), lc), np.nan)
        for i in range(lc):
            lam = np.minimum(1, np.maximum(0, pbins.ubin30[i] - pbins.lbin0) / 22.5)
            nu = np.minimum(1, np.maximum(0, pbins.ubin00 - pbins.lbin30[i]) / 22.5)
            w = lam + nu - 1
            if pbins.ubin30[i] < pbins.lbin30[i]:
                w = w + 1
            weight[:, i] = w
            Mum30[:, i] = r_matvec(Mum0, w)
        qw = r_seq_by(0.2, np.ceil(np.max(qwl)) + 1, 0.2)
        Mum0bs = np.full((len(qw), len(pbins.lbin0)), np.nan)
        Q30bs = np.full((lf1, lc, nb), np.nan)
        for ib in range(nb):
            for i in range(len(pbins.lbin0)):
                esbs = Wfit(freq2, Q0mbs[:lf2, i, ib], 2)
                Mum0bs[:, i] = esbs["mu0"] * np.exp(
                    -((qw / esbs["sigma"]) ** (1 / esbs["rho"]))
                )
            for i in range(lc):
                with np.errstate(divide="ignore", invalid="ignore"):
                    Q30bs[:, i, ib] = r_approx(
                        np.log(r_matvec(Mum0bs, weight[:, i])), qw, np.log(freq1)
                    )
        Mudir = Mum30
    else:
        Mudir = Mum0
    ind = ~np.isnan(mum0) & (mum0 <= 1) & (mum0 > 1e-10)
    es = Wfit(mum0[ind], qwl[ind], 2)
    W[0, 0], W[1, 0], W[2, 0] = es["rho"], es["sigma"], es["mu0"]
    W[npar : npar + lf1, 0] = r_signif(
        es["sigma"] * np.log(es["mu0"] * rp1) ** es["rho"], 3
    )
    mufit = es["mu0"] * np.exp(-((qwl / es["sigma"]) ** (1 / es["rho"])))
    mum0[ind] = mufit[ind]
    W[npar + lf1 :, 0] = r_signif(
        meas3["Qmsd"][:, 16][::-1][1:lf], 3
    )  # rev(Qmsd[, 17])[2:lf]
    for i in range(lc):
        mu1 = Mudir[:, i].copy()
        ind = (mu1 <= 1) & (mu1 > 1e-10)
        es = Wfit(mu1[ind], qwl[ind], 2)
        W[0, i + 1], W[1, i + 1], W[2, i + 1] = es["rho"], es["sigma"], es["mu0"]
        W[npar : npar + lf1, i + 1] = r_signif(
            es["sigma"] * np.log(es["mu0"] * rp1) ** es["rho"], 3
        )
        mufit = es["mu0"] * np.exp(-((qwl / es["sigma"]) ** (1 / es["rho"])))
        mu1[ind] = mufit[ind]
        Mudir[:, i] = mu1
        if dirbins == "22h":
            temp = meas3["Q0msd"][:, i][::-1][1:lf]  # rev(Q0msd[, i])[2:lf]
        else:
            temp = _rev_sd(Q30bs[:, i, :])
        W[npar + lf1 :, i + 1] = r_signif(temp, 3)
    x, pc0, Pc0 = _cond_prob_wl(qwl, mum0, Mudir, ptab)
    T = _write_table(
        paths.out(f"{station}_2035_wl_condprob_12hour_{tag}_versionFeb2026.csv"),
        x,
        pc0,
        Pc0,
        "high-tide water level [m]",
        cats,
    )
    W = _write_wbltail(
        paths.out(f"{station}_2035_wl_Wbltails_{tag}_versionFeb2026.csv"),
        W,
        _rownames(rp1, "SD"),
        _colnames(cats),
    )
    if dirbins == "22h":
        save_pkl(
            paths.out(f"{station}_2035_wl_stats_refit_22hdeg_versionFeb2026.pkl"),
            cats0=cats,
            lcats0=lc,
            Wbltail=W,
            Table=T,
            mum0=mum0,
            Mum0=Mudir,
            qwl=qwl,
        )
    else:
        save_pkl(
            paths.out(f"{station}_2035_wl_stats_refit_30deg_versionFeb2026.pkl"),
            cats30=cats,
            lcats30=lc,
            Wbltail=W,
            Table=T,
            mum0=mum0,
            Mum30=Mudir,
            qwl=qwl,
        )
    return W, T


def weibull_refit_wl_omni(station, mum0, qmum, qmsd, freq, pbins, paths, SLR):
    """Table chunk of statistics-wl10-omni-texeln-impr.Rmd (omni only; direction columns are NA)."""
    lf = len(freq)
    lf1 = lf - 1
    rp1 = 1 / freq[:lf1][::-1]
    lc = len(pbins.cats0)
    npar = 3
    W = np.full((npar + lf1 * 2, lc + 1), np.nan)
    qwl = qmum["quantile"] + SLR
    mum0 = np.array(mum0, dtype=float)
    ind = ~np.isnan(mum0) & (mum0 <= 1) & (mum0 > 1e-10)
    es = Wfit(mum0[ind], qwl[ind], 2)
    W[0, 0], W[1, 0], W[2, 0] = es["rho"], es["sigma"], es["mu0"]
    W[npar : npar + lf1, 0] = r_signif(
        es["sigma"] * np.log(es["mu0"] * rp1) ** es["rho"], 3
    )
    mufit = es["mu0"] * np.exp(-((qwl / es["sigma"]) ** (1 / es["rho"])))
    mum0[ind] = mufit[ind]
    W[npar + lf1 :, 0] = r_signif(np.asarray(qmsd)[::-1][1:lf], 3)
    x, pc0, _ = _cond_prob_wl(qwl, mum0, None, None)
    Pc0 = np.full((len(x), lc), np.nan)
    T = _write_table(
        paths.out(f"{station}_2035_wl_condprob_12hour_omni_versionFeb2026.csv"),
        x,
        pc0,
        Pc0,
        "high-tide water level [m]",
        pbins.cats0,
    )
    W = _write_wbltail(
        paths.out(f"{station}_2035_wl_Wbltails_omni_versionFeb2026.csv"),
        W,
        _rownames(rp1, "SD"),
        _colnames(pbins.cats0),
    )
    save_pkl(
        paths.out(f"{station}_2035_wl_stats_refit_omni.pkl"),
        Wbltail=W,
        Table=T,
        mum0=mum0,
        qwl=qwl,
    )
    return W, T
