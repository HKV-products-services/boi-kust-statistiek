"""
R's random number stream, split up per station.

In the R scripts all random numbers come from one stream (set.seed), drawn chunk after chunk and, within
a chunk, station after station. The numbers a station gets therefore depend on all stations before it.
The functions below replay only the random draws of an R run, for the full station list of that run,
and keep for every station and chunk a copy of the stream at the point where R starts drawing for that
station. A notebook can then compute any selection of stations, in any order, with exactly the random
numbers of the R run. This only reads the data (no fits).

Each function takes the seed of the R script and starts from set.seed itself, so a notebook cell that
calls it can be re-run on its own. (An RRandom stream is accepted too; it is then continued and advanced.)

Every lookup streams[station][step] returns a new copy of the stored stream, so the stored position never
moves: re-running a notebook cell (e.g. after an interruption) gives the same random numbers again.
"""

import copy

from kuststatistiek.rrandom import RRandom
from kuststatistiek.settings import Bins, WL_STATIONS, WL_WINDSTATIONS, WL_MEASFILE
from kuststatistiek.io.read_wind import read_seas5_wind, read_seas5_multi
from kuststatistiek.io.read_wl import read_station_wlmeas, link_wlmeas_windmeas
from kuststatistiek.analysis import seas5_blocks, meas_bootstrap_years


class _Snapshots(dict):
    """{step: stream}; every lookup returns a fresh copy of the stored stream."""

    def __getitem__(self, step):
        return copy.deepcopy(super().__getitem__(step))


def _start(seed):
    """set.seed(seed); an RRandom stream is used as it is."""
    return seed if isinstance(seed, RRandom) else RRandom(seed)


def wind_random_streams(seed, stations, nb, input_path, meas_N=()):
    """statistics-wind10, from set.seed: first the draws of chunk proc-meas2 (runif(Nm) for each measuring
    station, Nm in meas_N), then the bootstrap chunks.
    Returns {station: {"bootstrap1": stream, "bootstrap2": stream}}."""
    rng = _start(seed)
    for Nm in meas_N:  # chunk proc-meas2: runif(Nm)
        rng.runif(Nm)
    lcats = Bins().lcats
    streams = {st: _Snapshots() for st in stations}
    for st in (
        stations
    ):  # chunk dir-estimate-bootstrap1: blocks <- sample(1:nblock, nb*m, TRUE)
        streams[st]["bootstrap1"] = copy.deepcopy(rng)
        seas5_blocks(read_seas5_wind(st, input_path)["dt"], rng, nb)
    for st in (
        stations
    ):  # chunk dir-estimate-bootstrap2: rnorm(nb), rnorm(lcats*nb), rnorm(1) per sample
        streams[st]["bootstrap2"] = copy.deepcopy(rng)
        rng.rnorm(nb)
        rng.rnorm(lcats * nb)
        rng.rnorm(nb)
    return streams


def wl_random_streams(seed, stations, nb, input_path):
    """statistics-wl10, from set.seed.
    Returns {station: {"eimeas", "bootstrap", "meas_bootstrap", "model_error": stream}}."""
    rng = _start(seed)
    lcats = Bins().lcats
    windstation = dict(zip(WL_STATIONS, WL_WINDSTATIONS))
    streams = {st: _Snapshots() for st in stations}
    meas = {}
    for st in stations:  # chunk eimeas: runif(lw)
        streams[st]["eimeas"] = copy.deepcopy(rng)
        a = read_station_wlmeas(st, input_path, WL_MEASFILE)
        wlm = a["wspeed"] + (rng.runif(len(a["wspeed"])) - 0.5) * 0.0001
        meas[st] = (wlm, a["t"].to_numpy())
    for st in (
        stations
    ):  # chunk dir-estimate-bootstrap: blocks <- sample(1:nblock, nb*m, TRUE)
        streams[st]["bootstrap"] = copy.deepcopy(rng)
        seas5_blocks(read_seas5_multi(st, input_path)["dt"], rng, nb)
    for st in (
        stations
    ):  # chunk dir-estimate-meas-bootstrap: sample(yrm, ...), sample(yrmm, ...)
        streams[st]["meas_bootstrap"] = copy.deepcopy(rng)
        wlm, dtm = meas[st]
        c = link_wlmeas_windmeas(wlm, dtm, windstation[st], input_path)
        meas_bootstrap_years(wlm, dtm, c["wlmm"], c["dtmm"], c["winddir"], rng, nb)
    for st in (
        stations
    ):  # chunk dir-estimate-meas-bootstrap-moderr: rnorm(nb), rnorm(lcats*nb)
        streams[st]["model_error"] = copy.deepcopy(rng)
        rng.rnorm(nb)
        rng.rnorm(lcats * nb)
    return streams


def others_random_streams(seed, stations, input_path):
    """statistics-wl10others, from set.seed (only chunk eimeas draws random numbers).
    Returns {station: {"eimeas": stream}}."""
    rng = _start(seed)
    streams = {st: _Snapshots() for st in stations}
    for st in stations:  # chunk eimeas: runif(lw)
        streams[st]["eimeas"] = copy.deepcopy(rng)
        rng.runif(len(read_station_wlmeas(st, input_path, WL_MEASFILE)["wspeed"]))
    return streams


def texel_random_streams(seed, Nm, n_years, nb):
    """statistics-wl10-omni-texeln, from set.seed: runif(Nm) (eimeas), rnorm(nb) (shape), sample(yrm, nb*n_years)
    plus rnorm(1) per sample (bootstrap with model error), sample(yrm, nb*n_years) (bootstrap of the scale).
    Returns {"eimeas", "shape", "bootstrap", "scale_bootstrap": stream}."""
    rng = _start(seed)
    streams = _Snapshots()
    streams["eimeas"] = copy.deepcopy(rng)
    rng.runif(Nm)
    streams["shape"] = copy.deepcopy(rng)
    rng.rnorm(nb)
    streams["bootstrap"] = copy.deepcopy(rng)
    rng.sample(n_years, nb * n_years)
    for _ in range(nb):
        rng.rnorm(1)
    streams["scale_bootstrap"] = copy.deepcopy(rng)
    return streams
