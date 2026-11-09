"""
Bit-exact reimplementation of R's default random number generation, so that the
Python scripts can reproduce the bootstrap samples and random perturbations of the
R scripts.

Reproduces (R >= 3.6 defaults: RNGkind("Mersenne-Twister", "Inversion", "Rejection")):
    set.seed(seed)                      -> RRandom(seed)
    runif(n)                            -> rng.runif(n)
    rnorm(n)                            -> rng.rnorm(n)
    sample(x, size, replace = TRUE)     -> rng.sample(x, size)       (x: array or int n for 1:n)

The draws are consumed from one stream in call order, exactly as in R. So the Python
scripts must make the same calls in the same order as the R scripts (including calls for
stations that are not computed; use the `size` of the draw to keep the streams aligned).
"""

import numpy as np

_I2_32M1 = 2.328306437080797e-10  # 1/(2^32 - 1), used by R's fixup()
_SCALE = 2.3283064365386963e-10  # 2^-32, used by R's MT_genrand()
_BIG = 134217728.0  # 2^27, used by R's inversion method for rnorm


def r_qnorm(p):
    """R's qnorm5(p, 0, 1, lower_tail=TRUE, log_p=FALSE) (Wichura AS241), same operation order."""
    p = np.asarray(p, dtype=float)
    q = p - 0.5
    out = np.empty_like(p)

    c = np.abs(q) <= 0.425
    if np.any(c):
        qq = q[c]
        r = 0.180625 - qq * qq
        out[c] = (
            qq
            * (
                (
                    (
                        (
                            (
                                (
                                    (r * 2509.0809287301226727 + 33430.575583588128105)
                                    * r
                                    + 67265.770927008700853
                                )
                                * r
                                + 45921.953931549871457
                            )
                            * r
                            + 13731.693765509461125
                        )
                        * r
                        + 1971.5909503065514427
                    )
                    * r
                    + 133.14166789178437745
                )
                * r
                + 3.387132872796366608
            )
            / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r
                                        * 5226.495278852854561  # coefficient as corrected in R >= 4.3
                                        + 28729.085735721942674
                                    )
                                    * r
                                    + 39307.89580009271061
                                )
                                * r
                                + 21213.794301586595867
                            )
                            * r
                            + 5394.1960214247511077
                        )
                        * r
                        + 687.1870074920579083
                    )
                    * r
                    + 42.313330701600911252
                )
                * r
                + 1.0
            )
        )
    t = ~c
    if np.any(t):
        qq = q[t]
        pp = p[t]
        r = np.sqrt(
            -np.log(np.where(qq > 0, (0.5 - pp) + 0.5, pp))
        )  # R_DT_CIv(p) = 0.5 - p + 0.5
        val = np.empty_like(r)
        a = r <= 5.0
        if np.any(a):
            ra = r[a] - 1.6
            val[a] = (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ra * 7.7454501427834140764e-4
                                        + 0.0227238449892691845833
                                    )
                                    * ra
                                    + 0.24178072517745061177
                                )
                                * ra
                                + 1.27045825245236838258
                            )
                            * ra
                            + 3.64784832476320460504
                        )
                        * ra
                        + 5.7694972214606914055
                    )
                    * ra
                    + 4.6303378461565452959
                )
                * ra
                + 1.42343711074968357734
            ) / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ra * 1.05075007164441684324e-9
                                        + 5.475938084995344946e-4
                                    )
                                    * ra
                                    + 0.0151986665636164571966
                                )
                                * ra
                                + 0.14810397642748007459
                            )
                            * ra
                            + 0.68976733498510000455
                        )
                        * ra
                        + 1.6763848301838038494
                    )
                    * ra
                    + 2.05319162663775882187
                )
                * ra
                + 1.0
            )
        b = ~a
        if np.any(b):
            rb = r[b] - 5.0
            val[b] = (
                (
                    (
                        (
                            (
                                (
                                    (
                                        rb * 2.01033439929228813265e-7
                                        + 2.71155556874348757815e-5
                                    )
                                    * rb
                                    + 0.0012426609473880784386
                                )
                                * rb
                                + 0.026532189526576123093
                            )
                            * rb
                            + 0.29656057182850489123
                        )
                        * rb
                        + 1.7848265399172913358
                    )
                    * rb
                    + 5.4637849111641143699
                )
                * rb
                + 6.6579046435011037772
            ) / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        rb * 2.04426310338993978564e-15
                                        + 1.4215117583164458887e-7
                                    )
                                    * rb
                                    + 1.8463183175100546818e-5
                                )
                                * rb
                                + 7.868691311456132591e-4
                            )
                            * rb
                            + 0.0148753612908506148525
                        )
                        * rb
                        + 0.13692988092273580531
                    )
                    * rb
                    + 0.59983220655588793769
                )
                * rb
                + 1.0
            )
        out[t] = np.where(qq < 0.0, -val, val)
    return out


class RRandom:
    """One R random-number stream: RRandom(seed) is equivalent to set.seed(seed)."""

    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):  # initial scrambling (R: Randomize)
            s = (69069 * s + 1) & 0xFFFFFFFF
        iseed = np.empty(625, dtype=np.uint64)
        for j in range(625):  # RNG_Init: dummy[0] (= mti) + mt[624]
            s = (69069 * s + 1) & 0xFFFFFFFF
            iseed[j] = s
        # FixupSeeds(initial=TRUE): mti = N = 624, so the next draw regenerates the state
        self._bitgen = np.random.MT19937()
        self._bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": iseed[1:].astype(np.uint32), "pos": 624},
        }
        self._buf = np.empty(0)
        self._pos = 0

    # ---- uniform stream with look-ahead (needed for rejection sampling) ----
    def _peek(self, n):
        avail = len(self._buf) - self._pos
        if avail < n:
            m = max(n - avail, 1 << 16)
            raw = self._bitgen.random_raw(m).astype(np.float64) * _SCALE
            raw[raw <= 0.0] = 0.5 * _I2_32M1  # R's fixup()
            raw[(1.0 - raw) <= 0.0] = 1.0 - 0.5 * _I2_32M1
            self._buf = np.concatenate([self._buf[self._pos :], raw])
            self._pos = 0
        return self._buf[self._pos : self._pos + n]

    def _advance(self, n):
        self._pos += n

    def unif_rand(self, n):
        u = self._peek(n).copy()
        self._advance(n)
        return u

    # ---- R functions ----
    def runif(self, n, a=0.0, b=1.0):
        return a + (b - a) * self.unif_rand(int(n))

    def rnorm(self, n, mean=0.0, sd=1.0):
        n = int(n)
        u = self.unif_rand(2 * n).reshape(n, 2)
        z = np.floor(_BIG * u[:, 0]) + u[:, 1]  # u = (int)(BIG*u) + unif_rand()
        return mean + sd * r_qnorm(z / _BIG)

    def unif_index(self, dn, size):
        """R_unif_index(dn) (sample.kind = 'Rejection') repeated `size` times; returns 0-based indices."""
        size = int(size)
        if size == 0:
            return np.empty(0, dtype=np.int64)
        bits = int(np.ceil(np.log2(dn)))
        k = bits // 16 + 1  # uniforms per attempt in rbits()
        mask = (1 << bits) - 1
        out = np.empty(size, dtype=np.int64)
        filled = 0
        while filled < size:
            need = size - filled
            natt = int(need * 1.3) + 64
            u = self._peek(natt * k).reshape(natt, k)
            v = np.zeros(natt, dtype=np.int64)
            for c in range(k):
                v = 65536 * v + np.floor(u[:, c] * 65536).astype(np.int64)
            dv = v & mask
            ok = np.flatnonzero(dv < dn)
            if len(ok) >= need:
                last = ok[need - 1]
                out[filled:] = dv[ok[:need]]
                self._advance((last + 1) * k)
                filled = size
            else:
                out[filled : filled + len(ok)] = dv[ok]
                filled += len(ok)
                self._advance(natt * k)
        return out

    def sample(self, x, size):
        """sample(x, size, replace = TRUE); x an array, or an int n meaning 1:n."""
        if np.ndim(x) == 0:
            return self.unif_index(int(x), size) + 1
        x = np.asarray(x)
        return x[self.unif_index(len(x), size)]
