"""
Small numerical helpers that reproduce R semantics exactly (or to the last bit where possible).

R functions reproduced: approx, signif, quantile (type 7), sum/mean/sd (R accumulates in
long double), loess (via scikit-misc, which wraps the same netlib code as R), %*% (reference
BLAS summation order), and write.csv / write.table number formatting.
"""

import math
from fractions import Fraction
from pathlib import Path
import pickle

import numpy as np


# ----------------------------------------------------------------------------------------
# pickle i/o: every output file of the R scripts (.RData) has a .pkl counterpart holding a
# dict with exactly the variable names of the R save() call.
# ----------------------------------------------------------------------------------------
def save_pkl(path, **objects):
    with open(Path(path), "wb") as f:
        pickle.dump(objects, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------------------
# arithmetic
# ----------------------------------------------------------------------------------------
def r_sum(x):
    """sum(x): R accumulates in long double; math.fsum (correctly rounded) matches it."""
    x = np.asarray(x, dtype=float).ravel()
    if np.any(np.isnan(x)):
        return np.nan
    return math.fsum(x)


def r_rowsums(a):
    """apply(a, 1, sum) for a 2-d array (NA propagates, as in R without na.rm)."""
    a = np.asarray(a, dtype=float)
    return np.array([r_sum(row) for row in a])


def _exact_mean(x):
    return float(sum(map(Fraction, x)) / len(x))


def r_mean(x):
    """mean(x) (R: long double sum, then a refinement step; ~ correctly rounded)."""
    x = np.asarray(x, dtype=float).ravel()
    if np.any(np.isnan(x)):
        return np.nan
    return _exact_mean(x) if len(x) > 0 else np.nan


def r_sd(x):
    """sd(x, na.rm = TRUE) as computed by R's var (two-pass, long double accumulation)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    m = _exact_mean(x)
    d = x - m
    p = d * d
    return math.sqrt(float(sum(map(Fraction, p)) / (n - 1)))


def r_apply_sd(a, axis):
    """apply(a, margin, sd, na.rm = TRUE) where `axis` is the axis that is reduced."""
    a = np.moveaxis(np.asarray(a, dtype=float), axis, -1)
    out = np.empty(a.shape[:-1])
    for idx in np.ndindex(out.shape):
        out[idx] = r_sd(a[idx])
    return out


def r_matvec(A, w):
    """A %*% w with the summation order of the reference BLAS (dgemm) used by R."""
    A = np.asarray(A, dtype=float)
    c = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        c = c + w[j] * A[:, j]
    return c


def r_quantile(x, probs):
    """quantile(x, probs) (type 7) as implemented in R's quantile.default."""
    x = np.sort(np.asarray(x, dtype=float))
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    n = len(x)
    index = 1 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = x[lo - 1].copy()
    xhi = x[hi - 1]
    i = (index > lo) & (xhi != qs)
    h = (index - lo)[i]
    qs[i] = (1 - h) * qs[i] + h * xhi[i]
    return qs


def r_round_half_even(x):
    return np.round(x)  # R's round(x) (digits = 0) is also IEC 60559 round-half-even


def r_signif(x, digits):
    """signif(x, digits): port of R's fprec() (nmath/fprec.c)."""
    x = np.asarray(x, dtype=float)
    out = np.array(x, dtype=float, copy=True)
    flat = out.ravel()
    dig = max(1, int(round(digits)))
    max10e = 308
    for k, v in enumerate(flat):
        if not np.isfinite(v) or v == 0.0:
            continue
        sgn = 1.0
        if v < 0:
            sgn, v = -1.0, -v
        l10 = math.log10(v)
        e10 = int(dig - 1 - math.floor(l10))
        if abs(l10) < max10e - 2:
            p10 = 1.0
            if e10 > max10e:
                p10 = 10.0 ** (e10 - max10e)
                e10 = max10e
            if e10 > 0:
                pow10 = _pow_di(10.0, e10)
                flat[k] = sgn * (_nearbyint((v * pow10) * p10) / pow10) / p10
            else:
                pow10 = _pow_di(10.0, -e10)
                flat[k] = sgn * (_nearbyint(v / pow10) * pow10)
        else:  # very large or small numbers: not needed here
            flat[k] = sgn * float(f"{v:.{dig - 1}e}")
    return out.reshape(x.shape) if x.ndim else float(out)


def _pow_di(x, n):
    """R_pow_di: repeated squaring, exact for 10^n with n <= 22."""
    xn = 1.0
    if n < 0:
        return 1.0 / _pow_di(x, -n)
    while n:
        if n & 1:
            xn *= x
        n >>= 1
        if n:
            x *= x
    return xn


def _nearbyint(v):
    return float(np.rint(v))


# ----------------------------------------------------------------------------------------
# approx (R's stats::approx with rule = 1 or 2, ties = mean, method linear or constant)
# ----------------------------------------------------------------------------------------
def r_approx(x, y, xout, method="linear", rule=1, na_rm=True):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    xout = np.asarray(xout, dtype=float)
    shape = xout.shape
    v = xout.ravel()
    if na_rm:
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y = x[ok], y[ok]
    else:
        ok = ~np.isnan(x)
        x, y = x[ok], y[ok]
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    if len(x) > 1 and np.any(x[1:] == x[:-1]):  # ties = mean
        ux, start = np.unique(x, return_index=True)
        ends = np.append(start[1:], len(x))
        y = np.array(
            [y[s] if e - s == 1 else r_mean(y[s:e]) for s, e in zip(start, ends)]
        )
        x = ux
    n = len(x)
    out = np.full(v.shape, np.nan)
    if n == 0:
        return out.reshape(shape)
    ylow, yhigh = (np.nan, np.nan) if rule == 1 else (y[0], y[-1])
    fin = ~np.isnan(v)
    vv = v[fin]
    res = np.empty(vv.shape)
    below = vv < x[0]
    above = vv > x[-1]
    res[below] = ylow
    res[above] = yhigh
    inr = ~(below | above)
    if n == 1:
        res[inr] = y[0]
    else:
        w = vv[inr]
        j = np.clip(np.searchsorted(x, w, side="right"), 1, n - 1)
        i = j - 1
        if method == "linear":
            with np.errstate(invalid="ignore", divide="ignore"):
                r = y[i] + (y[j] - y[i]) * ((w - x[i]) / (x[j] - x[i]))
        else:  # constant, f = 0
            r = y[i].copy()
        eqj = w == x[j]
        r[eqj] = y[j][eqj]
        eqi = w == x[i]
        r[eqi] = y[i][eqi]
        res[inr] = r
    out[fin] = res
    return out.reshape(shape)


# ----------------------------------------------------------------------------------------
# loess (R: stats::loess), via scikit-misc which wraps the same netlib 'dloess' code
# ----------------------------------------------------------------------------------------
class _DirectLoessSpanGT1:
    """loess(y ~ x, span > 1, surface = "direct") for one predictor (scikit-misc refuses span > 1).
    As in R's ehg127: all points are used and the squared neighbourhood radius is
    max(dist^2) * span; tricube weights, local polynomial least squares at each point."""

    class _Pred:
        def __init__(self, v):
            self.values = v

    def __init__(self, x, y, span, degree):
        self.x, self.y, self.span, self.degree = x, y, span, degree

    def predict(self, newx):
        x, y = self.x, self.y
        out = np.empty(len(newx))
        for i, q in enumerate(np.asarray(newx, dtype=float)):
            dist = (x - q) ** 2
            w = np.sqrt(dist / (np.max(dist) * self.span))
            w = np.sqrt((1 - w**3) ** 3)
            B = np.column_stack([w * (x - q) ** k for k in range(self.degree + 1)])
            out[i] = np.linalg.lstsq(B, w * y, rcond=None)[0][0]
        return self._Pred(out)


def r_loess_fit(
    x, y, span=0.75, degree=2, surface="interpolate", statistics="approximate"
):
    from skmisc.loess import loess

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))  # na.action = na.omit
    if span > 1:
        if surface != "direct":
            raise NotImplementedError(
                "span > 1 only implemented for surface = 'direct'"
            )
        return _DirectLoessSpanGT1(x[ok], y[ok], span, degree)
    m = loess(
        x[ok],
        y[ok],
        span=span,
        degree=degree,
        surface=surface,
        statistics=statistics,
        family="gaussian",
    )
    m.fit()
    return m


def r_loess_predict(model, newx):
    newx = np.asarray(newx, dtype=float)
    out = np.full(newx.shape, np.nan)
    ok = ~np.isnan(newx)
    if np.any(ok):
        out[ok] = model.predict(newx[ok]).values
    return out


# ----------------------------------------------------------------------------------------
# write.csv / write.table (numbers formatted like R: 15 significant digits, fixed or
# scientific notation, whichever is narrower)
# ----------------------------------------------------------------------------------------
def r_format_num(v, digits=15):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    if v == 0:
        return "0"
    # minimal number of significant digits (<= digits) that represents v at `digits` precision
    s = f"{abs(v):.{digits - 1}e}"
    mant, ex = s.split("e")
    kpower = int(ex)
    mdig = mant.replace(".", "").rstrip("0")
    nsig = max(1, len(mdig))
    neg = 1 if v < 0 else 0
    # R rounds to `digits` significant digits first (value used for both notations)
    rgt_fixed = max(0, nsig - kpower - 1)
    left = kpower + 1 if kpower >= 0 else 1
    w_fixed = neg + left + (rgt_fixed + 1 if rgt_fixed > 0 else 0)
    w_sci = neg + (nsig + 1 if nsig > 1 else 1) + (4 if abs(kpower) < 100 else 5)
    if w_fixed <= w_sci:
        return f"{v:.{rgt_fixed}f}"
    m = mdig[0] + ("." + mdig[1:] if nsig > 1 else "")
    return f"{'-' if neg else ''}{m}e{'-' if kpower < 0 else '+'}{abs(kpower):02d}"


def _cell(v, quote_strings=True):
    if isinstance(v, str):
        return f'"{v}"' if quote_strings else v
    return r_format_num(v)


def write_r_csv(path, columns, colnames, rownames=None, row_names=True):
    """write.csv(data, file) for a table given as a list of columns.

    columns: list of 1-d sequences (numeric or str); colnames: list of str;
    rownames: list of str (default "1", "2", ...); row_names=False drops the row-name column.
    """
    ncol = len(columns)
    nrow = len(columns[0]) if ncol else 0
    lines = []
    head = (['""'] if row_names else []) + [f'"{c}"' for c in colnames]
    lines.append(",".join(head))
    for r in range(nrow):
        cells = []
        if row_names:
            rn = rownames[r] if rownames is not None else str(r + 1)
            cells.append(f'"{rn}"')
        for c in range(ncol):
            cells.append(_cell(columns[c][r]))
        lines.append(",".join(cells))
    Path(path).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )  # LF line ends, as KNMI's tables


def r_as_character(v):
    """as.character(<double>) as used for row/column labels (15 significant digits)."""
    return r_format_num(float(v))
