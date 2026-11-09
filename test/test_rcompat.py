"""Tests of the R-compatible functions against reference values from R 4.5.3."""

import numpy as np

from kuststatistiek.rrandom import RRandom
from kuststatistiek.rutils import r_approx, r_quantile, r_signif
from kuststatistiek.settings import Bins, ProductionBins, wl_settings, wind_settings


def test_runif():
    # R: set.seed(42); runif(3)
    rng = RRandom(42)
    expected = [0.91480604349635541, 0.93707541329786181, 0.28613953478634357]
    np.testing.assert_array_equal(rng.runif(3), expected)


def test_rnorm():
    # R: set.seed(42); rnorm(3)
    rng = RRandom(42)
    expected = [1.37095844714666848, -0.56469817139608869, 0.36312841133733920]
    np.testing.assert_allclose(rng.rnorm(3), expected, rtol=1e-15)


def test_sample():
    # R: set.seed(42); sample(10, 8, replace = TRUE)
    rng = RRandom(42)
    np.testing.assert_array_equal(rng.sample(10, 8), [1, 5, 1, 9, 10, 4, 2, 10])


def test_set_seed_restarts_stream():
    rng = RRandom(42)
    first = rng.runif(5)
    rng.set_seed(42)
    np.testing.assert_array_equal(rng.runif(5), first)


def test_quantile_type7():
    # R: quantile(c(3,1,4,1,5,9,2,6), c(0, 0.1, 0.25, 0.5, 0.9, 1))
    x = [3, 1, 4, 1, 5, 9, 2, 6]
    q = r_quantile(x, [0, 0.1, 0.25, 0.5, 0.9, 1])
    np.testing.assert_allclose(
        q, [1.0, 1.0, 1.75, 3.5, 6.8999999999999995, 9.0], rtol=1e-15
    )


def test_signif():
    # R: signif(c(123456, 0.00123456, -7.8385e-12, 2.5), 3)
    out = r_signif(np.array([123456, 0.00123456, -7.8385e-12, 2.5]), 3)
    np.testing.assert_array_equal(out, [123000.0, 0.00123, -7.84e-12, 2.5])


def test_approx_rule1():
    # R: approx(c(1,2,3), c(10,20,40), xout = c(0.5, 1.5, 2.5, 3.5))$y  ->  NA 15 30 NA
    y = np.asarray(r_approx([1, 2, 3], [10, 20, 40], [0.5, 1.5, 2.5, 3.5]), dtype=float)
    np.testing.assert_array_equal(np.isnan(y), [True, False, False, True])
    np.testing.assert_array_equal(y[1:3], [15.0, 30.0])


def test_bins():
    b = Bins()
    assert b.lcats == 17 and b.lcats0 == 16  # 16 sectors of 22.5 degrees + omni
    assert b.cats[-1] == 180.0
    pb = ProductionBins()
    assert (
        len(pb.cats30) == 12 and pb.cats30[-1] == 360.0
    )  # production bins: north = 360


def test_var_settings():
    assert wl_settings().seed == 342911
    assert wind_settings().seed == 449723
    assert wind_settings(nb=3).nb == 3
