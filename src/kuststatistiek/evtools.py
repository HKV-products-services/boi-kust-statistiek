"""
Extreme-value helper functions: ports of EVTools::EI and of the KNMI scripts
Wfit.R, tau2upot.R, tau2u10.R, u102tau.R, correctfrequencies.R.
"""

import numpy as np

from kuststatistiek.rutils import r_approx


def EI(X, l=50, ngr=20):  # noqa: E741 (argument name as in EVTools)
    """EVTools::EI: extremal index estimates (upcrossings and Ferro & Segers) for ngr thresholds."""
    X = np.asarray(X, dtype=float).ravel()
    sX = -np.sort(-X)
    n = len(X)
    gr = np.linspace(np.log(l), np.log(n), ngr)  # seq(log(l), log(n), length.out = ngr)
    k = np.ceil(np.exp(gr))
    p = k / n
    EIupcross = np.full(ngr, np.nan)
    EIFS = np.full(ngr, np.nan)
    for i in range(ngr):
        if k[i] > n:  # sX[k] is NA in R
            continue
        s = sX[int(k[i]) - 1]
        id2 = (X[:-1] <= s) & (X[1:] > s)
        id1 = X[1:] > s
        EIupcross[i] = id2.sum() / id1.sum()
        S = np.flatnonzero(X > s)
        if len(S) > 1:
            T = np.diff(S)
            sT, lT = T.sum(), len(T)
            if T.max() > 2:
                theta = 2 * (sT - lT) ** 2 / (np.sum((T - 1) * (T - 2)) * lT)
            else:
                theta = 2 * sT**2 / (np.sum(T**2) * lT)
            EIFS[i] = min(1.0, theta)
    return {"k": k, "p": p, "EIupcross": EIupcross, "EIFS": EIFS}


def fit_eimdl(p, EIFS):
    """eimdl <- lm(lEI ~ y, EIm[EIm$y < 5.5 & EIm$y > 0.05, ]); returns the coefficients (intercept, slope)."""
    y = -np.log(np.asarray(p, dtype=float))
    lEI = np.log(np.asarray(EIFS, dtype=float))
    sel = (y < 5.5) & (y > 0.05) & np.isfinite(lEI)
    A = np.column_stack([np.ones(sel.sum()), y[sel]])
    coef, *_ = np.linalg.lstsq(A, lEI[sel], rcond=None)
    return {"coefficients": coef}


def correctfrequencies(p, eimdl, EIvalue):
    """correctfrequencies.R: correction factors for the higher serial dependence in the low range."""
    EIlim = EIvalue / 6  # EI for hourly data
    y = -np.log(p)
    a = eimdl["coefficients"][1]
    b = eimdl["coefficients"][0] - np.log(EIlim)
    scales = -b / a
    return np.exp(b * (1 - np.tanh(y / scales)))


def tau2upot(tau):
    """tau2upot.R: wind stress [Pa] to potential wind speed [m/s]."""
    tau = np.asarray(tau, dtype=float)
    rho, kappa = 1.27, 0.4
    u10gr = np.arange(601) * 0.1  # seq(0, 60, 0.1)
    Cd = 0.001 * (0.8 + 0.065 * u10gr)  # Wu
    taugr = Cd * rho * u10gr**2
    with np.errstate(divide="ignore", invalid="ignore"):
        lz0 = np.log(10) - kappa / np.sqrt(Cd)
        upotgr = u10gr * (np.log(60) - lz0) / (np.log(10) - lz0) * (1 / 1.31)
    return r_approx(taugr, upotgr, tau)


def tau2u10(tau, char=0.018, maxcd=np.inf):
    """tau2u10.R: wind stress to 10-m wind speed using the Charnock relation."""
    rho, g, kappa = 1.27, 9.81, 0.4
    ustar = np.sqrt(np.asarray(tau, dtype=float) / rho)
    z0 = char * ustar**2 / g
    cdsqrt = np.minimum(np.sqrt(maxcd), kappa / np.log(10 / z0 + 1))
    return ustar / cdsqrt


def Wfit(mu, y, npar):
    """Wfit.R: grid-search least-squares fit of a 2- (or 3-) parameter Weibull tail.

    npar = 2: log(mu) = c - d * y^(1/rho), returns sigma = d^(-rho), mu0 = exp(c).
    The regression for each rho is the ordinary least-squares line (as lm in R).
    """
    mu = np.asarray(mu, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(mu) | np.isnan(y))  # lm: na.action = na.omit
    mu, y = mu[ok], y[ok]
    rhogrid = 0.1 + np.arange(951) * 0.002  # seq(0.1, 2, 0.002)
    if npar == 2:
        Y = np.log(mu)
        Xg = y[None, :] ** (1.0 / rhogrid[:, None])  # (nrho, n)
    elif npar == 3:
        Y = y
        Xg = (-np.log(mu / 10))[None, :] ** rhogrid[:, None]
    else:
        raise ValueError("npar must be 2 or 3")
    xm = Xg.mean(axis=1, keepdims=True)
    ym = Y.mean()
    slope = np.sum((Xg - xm) * (Y - ym), axis=1) / np.sum((Xg - xm) ** 2, axis=1)
    icpt = ym - slope * xm[:, 0]
    if npar == 2:
        c, d = icpt, -slope
        res = Y[None, :] - c[:, None] + d[:, None] * Xg
        mse = np.mean(res**2, axis=1)
        i = int(np.argmin(mse))
        return {
            "x0": 0.0,
            "sigma": d[i] ** (-rhogrid[i]),
            "rho": rhogrid[i],
            "mu0": np.exp(c[i]),
            "mse": mse[i],
        }
    res = Y[None, :] - (icpt[:, None] + slope[:, None] * Xg)
    mse = np.mean(res**2, axis=1)
    i = int(np.argmin(mse))
    return {
        "x0": icpt[i],
        "sigma": slope[i],
        "rho": rhogrid[i],
        "mu0": 0.1,
        "mse": mse[i],
    }
