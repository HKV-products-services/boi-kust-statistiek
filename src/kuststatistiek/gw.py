"""
Generalized Weibull (GW) tail estimation and directional tail analysis.

Line-by-line ports of
  EVTools::FitGW_iHilli, EVTools::qGWdisp, EVTools::h       (R package EVTools)
  esextr.R, collect.R, taildir.R                            (KNMI R scripts)
Index conventions: R is 1-based; comments give the R expression where helpful.
"""

import copy
import numpy as np

from kuststatistiek.rutils import r_quantile, r_approx, r_rowsums


# ------------------------------------------------------------------------------------------
def h(theta, lam):
    """EVTools::h: (lambda^theta - 1)/theta, log(lambda) for theta ~ 0."""
    eps = 1e-10
    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(lam, dtype=float)
    nt, nl = theta.size, lam.size
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if nt == 1:
            t = float(theta.ravel()[0])
            return (lam**t - 1) / t if abs(t) > eps else np.log(lam)
        if nl == nt:
            x = np.log(lam).astype(float).copy()
            i = np.abs(theta) > eps
            x[i] = (lam[i] ** theta[i] - 1) / theta[i]
            return x
        if nl == 1:
            lv = float(lam.ravel()[0])
            x = np.full(theta.shape, np.log(lv))
            i = np.abs(theta) > eps
            x[i] = (lv ** theta[i] - 1) / theta[i]
            return x
        # matrix(h(rep(theta, each = nl), rep(lambda, nt)), nl, nt): element [i, j] = h(theta_j, lambda_i)
        return h(np.repeat(theta, nl), np.tile(lam, nt)).reshape(nt, nl).T


# ------------------------------------------------------------------------------------------
def FitGW_iHilli(
    X, p=None, N=0, r11=1, fixedpar=None, l0=None, sigma=np.inf, metadata=None
):
    """Port of EVTools::FitGW_iHilli (GW tail fit with implicit joint shape/scale solution)."""
    fixedpar = fixedpar or {}
    theta0 = fixedpar.get("theta0")
    theta0Std = fixedpar.get("theta0Std")
    logdisp0 = fixedpar.get("logdisp0")
    logdisp0Std = fixedpar.get("logdisp0Std")
    theta0 = None if theta0 is None else np.atleast_1d(np.asarray(theta0, dtype=float))
    theta0Std = (
        None if theta0Std is None else np.atleast_1d(np.asarray(theta0Std, dtype=float))
    )
    len_theta0 = 0 if theta0 is None else len(theta0)
    len_theta0Std = 0 if theta0Std is None else len(theta0Std)
    l0 = None if l0 is None else np.atleast_1d(np.asarray(l0, dtype=np.int64))
    len_l0 = 0 if l0 is None else len(l0)
    p = None if p is None else np.atleast_1d(np.asarray(p, dtype=float))

    X = np.asarray(X, dtype=float).ravel()
    n = len(X)
    if n == 0:
        return None
    if N < n:
        N = n
    L = np.arange(1, n + 1)
    H = np.concatenate([[0.0], np.cumsum(1.0 / L)])
    th = np.sum(1.0 / np.arange(1, N + 1)) - H  # sum(1/(1:N)) - H
    th = th[:n]  # th[i-1] = vartheta_{i,n}
    if len_l0 == 0:
        l = np.arange(10, n)  # noqa: E741 (10:(n-1), name as in EVTools)
    else:
        l = l0[l0 < n]  # noqa: E741
    k = l.copy()
    if sigma < np.inf:
        raise NotImplementedError("sigma < Inf is not used in this project")
    k = np.minimum(np.maximum(1, k), n)
    k = np.maximum(k, l)
    X0 = -np.sort(-X)
    keep = (k > 2) & (X0[np.minimum(k, n) - 1] > -np.inf) & (l > 0) & (k < n) & (k >= l)
    k, l = k[keep], l[keep]  # noqa: E741
    nl = len(k)
    theta = thetaStd = g = q = qStd = None
    logdisp = logdispStd = thetaraw = thetarawStd = X0lStd = kraw = None
    if nl > 0:
        mk = int(k.max())
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hill0 = np.cumsum(X0[: mk - 1]) / np.arange(1, mk) - X0[1:mk]
            bad = hill0 <= 0
            if np.any(bad):
                hill0[bad] = np.min(hill0[~bad])
            if len_theta0 == 0:
                u = np.cumsum(np.log(th[1 : mk - 1])) / np.arange(1, mk - 1) - np.log(
                    th[2:mk]
                )
                g0 = hill0[: mk - 1] * th[1:mk]
                thetaraw = (
                    np.cumsum(np.log(g0[: mk - 2])) / np.arange(1, mk - 1)
                    - np.log(g0[1 : mk - 1])
                ) / u
                kraw = np.arange(1, mk - 1) + 2
                theta = thetaraw[k - 3]  # thetaraw[k-2]
                r11value = (
                    np.interp(kraw / N, r11["p"], r11["r"])
                    if isinstance(r11, dict)
                    else r11
                )
                thetarawStd = th[kraw - 1] * np.sqrt(r11value / kraw)
                thetaStd = thetarawStd[k - 3]
            elif len_theta0 == len_l0:
                theta = theta0 = theta0[:nl]
                if len_theta0Std == 0:
                    thetaStd = np.zeros(nl)
                elif len_theta0Std == len_l0:
                    thetaStd = theta0Std[:nl]
                else:
                    thetaStd = np.repeat(theta0Std[0], nl)
            else:
                theta = np.repeat(theta0[0], nl)
                theta0 = theta0[:1]
                thetaStd = (
                    np.repeat(theta0Std[0], nl) if len_theta0Std > 0 else np.zeros(nl)
                )

            if len_theta0 == 0:
                thetagrid = -2.005 + np.arange(402) * 0.01  # seq(-2.005, 2.005, 0.01)
                err = np.full(nl, np.inf)
                g = np.full(nl, np.nan)
                dw = np.full(nl, np.nan)
                dd = np.full(nl, np.nan)
                thetaref = np.full(nl, np.nan)
                a1 = np.arange(1, mk)
                a2 = np.arange(1, mk - 1)
                th1, th2 = th[: mk - 1], th[1:mk]
                lth1, lth2 = np.log(th1), np.log(th2)
                for ti in thetagrid:
                    temp = np.cumsum(h(ti, th1)) / a1
                    w = th2 ** (-ti) * temp + h(ti, 1 / th2)
                    w1 = np.cumsum(np.log(w[: mk - 2])) / a2 - np.log(w[1 : mk - 1])
                    temp1 = np.cumsum(lth1 * th1**ti) / a1
                    temp2 = np.cumsum(th1**ti) / a1
                    temp3 = (th2 ** (-ti) * (temp1 - lth2 * temp2) - w) / ti
                    temp4 = (
                        np.cumsum(temp3[: mk - 2] / w[: mk - 2]) / a2
                        - temp3[1 : mk - 1] / w[1 : mk - 1]
                    )
                    temp4 = np.append(
                        temp4, 2 * temp4[mk - 3] - temp4[mk - 4]
                    )  # temp4[mk-1]
                    err1 = np.abs(ti + 1 - theta + w1[k - 3] / u[k - 3])
                    idx = err1 < err
                    if not np.any(np.isnan(err1)):  # R: if (!any(is.na(id)))
                        if np.any(idx):
                            thetaref[idx] = ti
                            err[idx] = err1[idx]
                            li = l[idx] - 2  # l[id]-1 (R)
                            g[idx] = hill0[li] / w[li]
                            dw[idx] = temp3[li]
                            dd[idx] = 1 + temp4[li] / w[li]
                dd[np.isnan(dd)] = 1
                theta = thetaref
                thetaStd = thetaStd / dd
                thetaStd = np.maximum.accumulate(thetaStd[::-1])[
                    ::-1
                ]  # rev(cummax(rev()))
            elif len(theta0) == 1:
                ti = float(theta0[0])
                if abs(ti) < 1e-6:
                    ti = 1e-6
                temp = np.cumsum(h(ti, th[: mk - 1])) / np.arange(1, mk)
                w = th[1:mk] ** (-ti) * temp + h(ti, 1 / th[1:mk])
                temp1 = np.cumsum(
                    np.log(th[: mk - 1]) * th[: mk - 1] ** ti
                ) / np.arange(1, mk)
                temp2 = np.cumsum(th[: mk - 1] ** ti) / np.arange(1, mk)
                temp3 = (
                    th[1:mk] ** (-ti) * (temp1 - np.log(th[1:mk]) * temp2) - w
                ) / ti
                g = hill0[l - 2] / w[l - 2]
                dw = temp3[l - 2]
            else:
                g = np.full(nl, np.nan)
                dw = np.full(nl, np.nan)
                a1 = np.arange(1, mk)
                lth1, lth2 = np.log(th[: mk - 1]), np.log(th[1:mk])
                for i in range(nl):
                    ti = float(theta0[i])
                    if abs(ti) < 1e-6:
                        ti = 1e-6
                    # only the element l[i]-1 of the cumulative sums is needed
                    m = int(l[i]) - 1  # w[l[i]-1] (R) -> w[m-1]
                    temp = np.cumsum(h(ti, th[:m]))[-1] / m
                    w = th[m] ** (-ti) * temp + h(ti, 1 / th[m])
                    temp1 = np.cumsum(lth1[:m] * th[:m] ** ti)[-1] / m
                    temp2 = np.cumsum(th[:m] ** ti)[-1] / m
                    temp3 = (th[m] ** (-ti) * (temp1 - lth2[m - 1] * temp2) - w) / ti
                    g[i] = hill0[m - 1] / w
                    dw[i] = temp3

            r11value = (
                np.interp(l / N, r11["p"], r11["r"]) if isinstance(r11, dict) else r11
            )
            if logdisp0 is None or np.size(logdisp0) == 0:
                logdisp = np.log(g / np.maximum(X0[l - 1], 1e-4))
                logdispStd = np.sqrt(r11value / l)
            else:
                logdisp0 = np.atleast_1d(logdisp0)
                g = X0[l - 1] * np.exp(logdisp0[0] if len(logdisp0) != nl else logdisp0)
                logdisp = (
                    np.repeat(logdisp0[0], nl) if len(logdisp0) != nl else logdisp0
                )
                logdispStd = np.repeat(np.atleast_1d(logdisp0Std)[0], nl)
            X0lStd = hill0[l - 2] * np.sqrt(r11value / l)
            if p is not None and len(p) > 0:
                lp = len(p)
                q = np.full((nl, lp), np.nan)
                qStd = np.full((nl, lp), np.nan)
                for i in range(lp):
                    lam = -np.log(p[i]) / th[l - 1]
                    ha = h(theta, lam)
                    q[:, i] = X0[l - 1] + g * ha
                    ha1 = (1 / theta) * (lam**theta * np.log(lam) - ha)
                    iz = np.abs(theta) < 1e-10
                    if np.any(iz):
                        ha1 = np.where(iz, 0.5 * np.log(lam) ** 2, ha1)
                    deriv = ha1 - g * ha * dw / hill0[l - 2]
                    var = (
                        g**2 * (ha**2 * logdispStd**2 + deriv**2 * thetaStd**2)
                        + X0lStd**2
                    )
                    qStd[:, i] = np.sqrt(var)
    return {
        "k": k,
        "l": l,
        "y": th[l - 1],
        "N": N,
        "sigma": sigma,
        "r11": r11,
        "tailindex": theta,
        "tailindexStd": thetaStd,
        "scale": g,
        "logdisp": logdisp,
        "logdispStd": logdispStd,
        "location": X0[l - 1],
        "locationStd": X0lStd,
        "p": p,
        "quantile": q,
        "quantileStd": qStd,
        "tailindexraw": thetaraw,
        "tailindexrawStd": thetarawStd,
        "kraw": kraw,
        "orderstats": X0,
        "df": "GW",
        "method": "FitGW_iHilli",
        "metadata": metadata,
    }


# ------------------------------------------------------------------------------------------
def qGWdisp(
    p, p0, q0, logdisp, theta, q0Std=np.nan, logdispStd=np.nan, thetaStd=np.nan
):
    """Port of EVTools::qGWdisp: GW quantiles (and standard errors) from location/dispersion/shape."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.log(p) / np.log(p0)
        ha = h(theta, lam)
        disp = np.exp(logdisp)
        q = q0 * (1 + disp * ha)
        dha = (1 / theta) * (lam**theta * np.log(lam) - ha)
        if np.any(np.abs(theta) < np.finfo(float).eps):
            dha = 0.5 * np.log(lam) ** 2
        varq0 = q0Std**2 * (q / q0) ** 2
        varlogdisp = (ha * disp * q0) ** 2 * logdispStd**2
        vartheta = (disp * q0) ** 2 * dha**2 * thetaStd**2
        qStd = np.sqrt(varq0 + varlogdisp + vartheta)
    return {
        "quantile": q,
        "quantileStd": qStd,
        "varlocation": varq0,
        "varlogdisp": varlogdisp,
        "vartailindex": vartheta,
    }


# ------------------------------------------------------------------------------------------
_ES_VECTORS = [
    "k",
    "l",
    "y",
    "tailindex",
    "tailindexStd",
    "scale",
    "logdisp",
    "logdispStd",
    "location",
    "locationStd",
]


def esextr(es, l0):
    """esextr.R: estimates for the thresholds l0 (in the order of l0, NA where absent)."""
    l0 = np.atleast_1d(np.asarray(l0))
    esl = np.asarray(es["l"])
    id_ = np.isin(esl, l0)
    id0 = np.isin(l0, esl)
    out = dict(es)
    for f in _ES_VECTORS:
        v = np.full(len(l0), np.nan)
        v[id0] = np.asarray(es[f], dtype=float)[id_]
        out[f] = v
    for f in ["quantile", "quantileStd"]:
        a = np.full((len(l0), np.asarray(es[f]).shape[1]), np.nan)
        a[id0, :] = np.asarray(es[f])[id_, :]
        out[f] = a
    return out


# ------------------------------------------------------------------------------------------
def collect(blockno, blocks):
    """collect.R: indices (1-based) of the resampled series, in the order of the original series."""
    blockno = np.asarray(blockno, dtype=np.int64)
    blocks = np.asarray(blocks, dtype=np.int64)
    counts = np.bincount(blocks - blocks.min(), minlength=0)
    rel = blockno - blocks.min()
    ni = np.zeros(len(blockno), dtype=np.int64)
    ok = (rel >= 0) & (rel < len(counts))
    ni[ok] = counts[rel[ok]]
    return np.repeat(np.arange(1, len(blockno) + 1), ni)


# ------------------------------------------------------------------------------------------
def assigncat(y, lbin, ubin):
    """assigncat.R: logical matrix (len(y) x nbins) of direction-bin membership."""
    y = np.asarray(y, dtype=float)
    lbin = np.asarray(lbin, dtype=float)
    ubin = np.asarray(ubin, dtype=float)
    if len(lbin) == 0:
        return np.ones((len(y), 1), dtype=bool)
    cat = np.zeros((len(y), len(lbin)), dtype=bool)
    for i in range(len(lbin)):
        if ubin[i] > lbin[i]:
            cat[:, i] = (y >= lbin[i]) & (y < ubin[i])
        else:
            cat[:, i] = (y >= lbin[i]) | (y < ubin[i])
    return cat


def rank_transform(v, N):
    """wlt <- -log(index) + log(N+1), index = ranks of -v (ties in order of occurrence)."""
    v = np.asarray(v, dtype=float).ravel()
    order = np.argsort(-v, kind="stable")
    index = np.empty(len(v), dtype=np.int64)
    index[order] = np.arange(1, len(v) + 1)
    return -np.log(index) + np.log(N + 1)


def taildir(fixdata, wlmm, winddir, bins, r11values=None):
    """Port of taildir.R: directional GW tails, consistent with the omni tail.

    fixdata: dict with metadata, N, factor, pthreshold (scalar), kthresh, mthresh (1-based),
             theta0, theta0Std (None or arrays of length lcats), es, x, freq
    bins:    kuststatistiek.settings.Bins (lbin, ubin, cats, lbinsm, ubinsm, lbin0, ubin0, cats0, binlink)
    """
    N = fixdata["N"]
    factor = fixdata["factor"]
    x = fixdata["x"]
    es = fixdata["es"]
    pthreshold = float(fixdata["pthreshold"])
    kthresh = fixdata["kthresh"] - 1
    mthresh = fixdata["mthresh"] - 1
    theta0 = fixdata.get("theta0")
    theta0Std = fixdata.get("theta0Std")
    freq = np.asarray(fixdata["freq"], dtype=float)
    lfreq = np.log(freq)
    lcats = len(bins.cats)
    lcats0 = len(bins.cats0)
    metadata = fixdata.get("metadata")
    esGWdir = [None] * lcats
    esGWdir0 = [None] * lcats0

    wlt = rank_transform(wlmm, N)
    cat = assigncat(winddir, bins.lbin, bins.ubin)
    catsm = assigncat(winddir, bins.lbinsm, bins.ubinsm)
    cat0 = assigncat(winddir, bins.lbin0, bins.ubin0)
    pbin0 = cat0.mean(axis=0)

    for i in range(lcats):
        X = -np.sort(-wlt[cat[:, i]])
        pbin = cat[:, i].mean()
        pbinsm = catsm[:, i].mean()
        Nbin = int(np.round(N * pbin))
        p = freq / factor / pbin
        l0 = np.atleast_1d(np.maximum(20, np.ceil(Nbin * pthreshold))).astype(np.int64)
        n = min(int(l0.max()) + 20, len(X))
        fixedpar = None
        if theta0 is not None and len(theta0) >= lcats:
            fixedpar = {"theta0": theta0[i], "theta0Std": theta0Std[i]}
        es_i = FitGW_iHilli(
            X[:n],
            p=p,
            N=Nbin,
            r11=1,
            fixedpar=fixedpar,
            l0=l0,
            sigma=np.inf,
            metadata=metadata,
        )
        ind = np.flatnonzero(es_i["tailindex"] > 1)
        if len(ind) > 0:  # adjust these estimates to have tailindex = 1
            temp = FitGW_iHilli(
                X[:n],
                p=p,
                N=Nbin,
                r11=1,
                l0=l0[ind],
                fixedpar={"theta0": 1.0, "theta0Std": 0.0},
                sigma=np.inf,
            )
            for f in [
                "tailindex",
                "tailindexStd",
                "scale",
                "logdisp",
                "logdispStd",
                "location",
                "locationStd",
            ]:
                es_i[f] = np.array(es_i[f], dtype=float)
                es_i[f][ind] = temp[f]
            es_i["quantile"][ind, :] = temp["quantile"]
            es_i["quantileStd"][ind, :] = temp["quantileStd"]
        if i < lcats - 1:
            ind = np.flatnonzero(es_i["tailindex"] == 1)
            if (
                len(ind) > 0
            ):  # bound the directional tail by the assumed exponential omni tail
                es_i["scale"][ind] = np.minimum(es_i["scale"][ind], es_i["y"][ind])
            ratio = r_quantile(wlt[catsm[:, i]], 1 - pthreshold) / es_i["location"]
            es_i["location"] = es_i["location"] * ratio
            es_i["logdisp"] = es_i["logdisp"] - np.log(ratio)
            es_i["quantile"] = es_i["quantileStd"] = None
        es_i["pbin"] = pbinsm
        es_i["freq"] = freq
        esGWdir[i] = es_i

    L = len(x)
    Mu = np.full((L, lcats), np.nan)
    Mu0 = np.full((L, lcats0), np.nan)

    def empirical(vals, xs):
        X1 = -np.sort(-vals)
        return r_approx(X1, np.arange(1, len(X1) + 1), xs, method="constant") / len(X1)

    def trailing_na(col):
        isna = np.isnan(col)
        return np.cumsum(isna[::-1])[::-1] == np.arange(
            1, len(col) + 1
        )  # rev(cumsum(rev(.)))==1:L

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for i in range(lcats):
            e = esGWdir[i]
            p = e["k"][mthresh] / e["N"]
            rho = min(1.0, e["tailindex"][mthresh])
            f = e["scale"][mthresh]
            Q = e["location"][mthresh]
            ind = x >= Q
            Mu[ind, i] = (
                factor
                * e["pbin"]
                * p ** (np.maximum(0, 1 + rho * ((x[ind] - Q) / f)) ** (1 / rho))
            )
            ind = x < Q
            if np.sum(ind) > 0:
                Mu[ind, i] = factor * e["pbin"] * empirical(wlt[catsm[:, i]], x[ind])
            Mu[trailing_na(Mu[:, i]), i] = 0
            links = np.flatnonzero(bins.binlink == i + 1)
            if i < lcats - 1 and len(links) > 0:
                for ii in links:
                    e0 = copy.deepcopy(e)  # esGWdir0[ii] <- esGWdir[i]
                    ratio = (
                        r_quantile(wlt[cat0[:, ii]], 1 - pthreshold) / e0["location"]
                    )
                    e0["location"] = e0["location"] * ratio
                    e0["logdisp"] = e0["logdisp"] - np.log(ratio)
                    e0["pbin"] = pbin0[ii]
                    esGWdir0[ii] = e0
                    Q0 = e0["location"][mthresh]
                    ind = x >= Q0
                    Mu0[ind, ii] = (
                        factor
                        * pbin0[ii]
                        * p
                        ** (np.maximum(0, 1 + rho * ((x[ind] - Q0) / f)) ** (1 / rho))
                    )
                    ind = x < Q0
                    if np.sum(ind) > 0:
                        Mu0[ind, ii] = (
                            factor * pbin0[ii] * empirical(wlt[cat0[:, ii]], x[ind])
                        )
                    Mu0[trailing_na(Mu0[:, ii]), ii] = 0

        mu = r_rowsums(Mu[:, : lcats - 1])
        Mu = Mu * (np.exp(-x) * factor / mu)[:, None]
        mu0 = r_rowsums(Mu0)
        Mu0 = Mu0 * (np.exp(-x) * factor / mu0)[:, None]
        mu0 = mu = np.exp(-x) * factor

        # quantiles at the standard frequencies (mu0 = mu now)
        p = es["k"][kthresh] / es["N"]
        qmu = qGWdisp(
            mu / factor,
            p,
            es["location"][kthresh],
            es["logdisp"][kthresh],
            es["tailindex"][kthresh],
            es["locationStd"][kthresh],
            es["logdispStd"][kthresh],
            es["tailindexStd"][kthresh],
        )
        Qfreq = np.full((len(freq), lcats), np.nan)
        for i in range(lcats - 1):
            temp = np.log(Mu[:, i])
            ok = ~np.isnan(temp)
            Qfreq[:, i] = r_approx(temp[ok], qmu["quantile"][ok], lfreq)
        Qfreq[:, lcats - 1] = r_approx(np.log(mu), qmu["quantile"], lfreq)
        Qfreq0 = np.full((len(freq), lcats0), np.nan)
        for i in range(lcats0):
            temp = np.log(Mu0[:, i])
            ok = ~np.isnan(temp)
            Qfreq0[:, i] = r_approx(temp[ok], qmu["quantile"][ok], lfreq)

    return {
        "esGWdir": esGWdir,
        "x": x,
        "mu": mu,
        "Mu": Mu,
        "esGWdir0": esGWdir0,
        "mu0": mu0,
        "Mu0": Mu0,
        "pbin0": pbin0,
        "qmu": qmu,
        "freq": freq,
        "Qfreq": Qfreq,
        "Qfreq0": Qfreq0,
    }
