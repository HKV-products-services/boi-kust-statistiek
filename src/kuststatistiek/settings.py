"""
Settings of the KNMI R scripts (Feb2026 versions), in one place.

Station lists are given in the order of the R scripts: the random-number stream depends on
the order in which stations are processed, so do not reorder them.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# default locations; set the environment variables KUSTSTATISTIEK_DATA_PATH, KUSTSTATISTIEK_OUTPUT_PATH
# and KUSTSTATISTIEK_FIG_PATH, or pass input_path / output_path / fig_path to Paths().
DATA_PATH = Path(
    os.environ.get(
        "KUSTSTATISTIEK_DATA_PATH",
        r"C:\Users\kindermann\Documents\Projecten\PR5197_Code review\Update_statistiek_zeewaterstand_wind_06_archive_Feb2026 2\Update_statistiek_zeewaterstand_wind_06_archive_Feb2026\data",
    )
)
OUTPUT_PATH = Path(
    os.environ.get(
        "KUSTSTATISTIEK_OUTPUT_PATH",
        r"C:\Users\kindermann\Documents\Projecten\PR5197_Code review\Python\results\data",
    )
)
FIG_PATH = Path(
    os.environ.get(
        "KUSTSTATISTIEK_FIG_PATH",
        r"C:\Users\kindermann\Documents\Projecten\PR5197_Code review\Python\results\figuren",
    )
)

# default bootstrap sample size (R: nb <- 100); use the nb of the R run to get the same random numbers
NB = 100


@dataclass
class Paths:
    input_path: Path = DATA_PATH  # delivered input data (colpath / measpath in R)
    output_path: Path = OUTPUT_PATH  # all output files (R: working directory)
    fig_path: Path = FIG_PATH

    def __post_init__(self):
        self.input_path, self.output_path, self.fig_path = map(
            Path, (self.input_path, self.output_path, self.fig_path)
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fig_path.mkdir(parents=True, exist_ok=True)

    def out(self, name):
        return self.output_path / name


# ------------------------------------------------------------------------------------------
FREQ = np.float_power(10.0, np.arange(-7, 2))  # freq <- 10^(-7:1)
PTHRESHOLD = 0.0015 * 2.0 ** (-4 + np.arange(19) * 0.5)  # 0.0015*2^seq(-4, 5, 0.5)
PN = 0.95


@dataclass
class Bins:
    """Direction bins as defined in the statistics scripts (directions %% 360: north = 0)."""

    lbin0: np.ndarray = field(init=False)
    ubin0: np.ndarray = field(init=False)
    cats0: np.ndarray = field(init=False)
    lbin: np.ndarray = field(init=False)
    ubin: np.ndarray = field(init=False)
    cats: np.ndarray = field(init=False)
    lbinsm: np.ndarray = field(init=False)
    ubinsm: np.ndarray = field(init=False)
    binlink: np.ndarray = field(init=False)

    def __post_init__(self):
        lbin0 = 11.25 + np.arange(16) * 22.5  # seq(11.25, 348.75, 22.5)
        ubin0 = lbin0 + 22.5
        cats0 = lbin0 + (ubin0 - lbin0) / 2
        self.ubin0, self.lbin0, self.cats0 = ubin0 % 360, lbin0 % 360, cats0 % 360
        lbin = (self.lbin0 - 11.25) % 360  # widened (overlapping 45 deg) bins
        ubin = (self.ubin0 + 11.25) % 360
        self.cats = np.append(self.cats0 % 360, 180.0)  # + omni bin
        self.ubin = np.append(ubin, 0.0)
        self.lbin = np.append(lbin, 0.0)
        lc = len(self.cats)
        self.ubinsm = (self.ubin - 11.25) % 360  # narrow bins, including omni
        self.lbinsm = (self.lbin + 11.25) % 360
        self.ubinsm[lc - 1] = self.ubin[lc - 1]
        self.lbinsm[lc - 1] = self.lbin[lc - 1]
        binlink = np.zeros(16, dtype=int)  # R: binlink (1-based bin number, 0 = NA)
        for i in range(lc - 1):
            if self.ubinsm[i] > self.lbinsm[i]:
                ind = (self.cats0 > self.lbinsm[i]) & (self.cats0 <= self.ubinsm[i])
            else:
                ind = (self.cats0 > self.lbinsm[i]) | (self.cats0 <= self.ubinsm[i])
            binlink[ind] = i + 1
        self.binlink = binlink

    @property
    def lcats(self):
        return len(self.cats)

    @property
    def lcats0(self):
        return len(self.cats0)


@dataclass
class ProductionBins:
    """Direction bins as defined in the production scripts (1+(x-1) %% 360: north = 360)."""

    lbin0: np.ndarray = field(init=False)
    ubin0: np.ndarray = field(init=False)
    cats0: np.ndarray = field(init=False)
    ubin00: np.ndarray = field(init=False)
    lbin30: np.ndarray = field(init=False)
    ubin30: np.ndarray = field(init=False)
    cats30: np.ndarray = field(init=False)

    def __post_init__(self):
        def m(v):
            return 1 + (v - 1) % 360

        lbin0 = 11.25 + np.arange(16) * 22.5
        ubin0 = lbin0 + 22.5
        cats0 = lbin0 + (ubin0 - lbin0) / 2
        self.ubin0, self.lbin0, self.cats0 = m(ubin0), m(lbin0), m(cats0)
        self.ubin00 = m(self.ubin0)
        self.ubin00[-1] += 360
        lbin30 = 15.0 + np.arange(12) * 30  # seq(15, 345, 30)
        ubin30 = lbin30 + 30
        cats30 = lbin30 + (ubin30 - lbin30) / 2
        self.ubin30, self.lbin30, self.cats30 = m(ubin30), m(lbin30), m(cats30)


@dataclass
class VarSettings:
    """Variable-specific settings (wind stress from SEAS5, or high-tide water level)."""

    name: str  # "wind" or "wl"
    EIvalue: float
    delt: float  # time step in years
    kthresh: int  # 1-based index in PTHRESHOLD (omni threshold)
    mthresh: int  # 1-based index in PTHRESHOLD (directional threshold)
    xgrid: tuple  # (from, n) of seq(from, 8, 0.025)
    modelstd: float
    scalestd: float
    seed: int
    varname: str
    nb: int = NB
    freq: np.ndarray = field(default_factory=lambda: FREQ.copy())
    pthreshold: np.ndarray = field(default_factory=lambda: PTHRESHOLD.copy())

    @property
    def factor(self):
        return self.EIvalue / self.delt

    @property
    def p(self):
        return self.freq / self.factor

    @property
    def x(self):
        start, n = self.xgrid
        return (start + np.arange(n) * 0.025) * np.log(10) + np.log(self.factor)

    def metadata(self, caseId=None):
        return {
            "varname": self.varname,
            "varunit": "m",
            "timeunit": "year",
            "timestep": self.delt,
            "caseId": caseId,
        }


def wind_settings(**kw):
    s = dict(
        name="wind",
        EIvalue=6,
        delt=6 / 24 / 365.25,
        kthresh=11,
        mthresh=15,
        xgrid=(-3.2, 449),
        modelstd=0.17,
        scalestd=0.05,
        seed=449723,
        varname="wind stress",
    )
    s.update(kw)
    return VarSettings(**s)


def wl_settings(**kw):
    s = dict(
        name="wl",
        EIvalue=1,
        delt=12.42 / 24 / 365.25,
        kthresh=15,
        mthresh=19,
        xgrid=(-2.8, 433),
        modelstd=0.10,
        scalestd=0.0,
        seed=342911,
        varname="high-tide water level",
    )
    s.update(kw)
    return VarSettings(**s)


RESOLUTIONFACTOR = 1.15  # SEAS5 stress bias correction
CHARNOCK = 0.032

# ------------------------------------------------------------------------------------------
# station tables (order as in the R scripts)
# ------------------------------------------------------------------------------------------
WIND_STATIONS = [
    "vlissing",
    "hoekvanh",
    "ijmuiden",
    "denhelde",
    "westters",
    "huibertg",
    "delfzijl",
]
WIND_STATIONNAME = [
    "Vlissingen",
    "Hoek van Holland",
    "IJmuiden",
    "den Helder",
    "West Terschelling",
    "Huibertgat",
    "Delfzijl",
]
WIND_MEASSTATIONS = [
    308,
    320,
    225,
    229,
    242,
    250,
    285,
]  # 250 = potential wind (W. Terschelling)
WIND_MEASSTATIONNAME = [
    "Cadzand",
    "LEG",
    "IJmuiden",
    "Texelhors",
    "Vlieland",
    "West Terschelling",
    "Huibertgat",
]
WIND_CSTATIONSYS5 = [
    "vlissing",
    "hoekvanh",
    "ijmuiden",
    "denhelde",
    "westters",
    "westters",
    "huibertg",
]

WL_STATIONS = [
    "vlissing",
    "hoekvanh",
    "ijmuiden",
    "denhelde",
    "harlinge",
    "delfzijl",
    "hansweert",
]
WL_STATIONNAME = [
    "Vlissingen",
    "Hoek van Holland",
    "IJmuiden",
    "den Helder",
    "Harlingen",
    "Delfzijl",
    "Hansweert",
]
WL_WINDSTATIONS = [308, 320, 225, 235, 242, 285, 308]
# measurement files: readwlmeas() (format 1) or readwlothermeas() (format 2)
WL_MEASFILE = {
    "vlissing": ("vlis", 1),
    "hoekvanh": ("hoek", 1),
    "ijmuiden": ("ijmu", 1),
    "denhelde": ("held", 1),
    "harlinge": ("harl", 1),
    "delfzijl": ("delf", 1),
    "hansweert": ("hansweert", 2),
    "os11": ("os11", 2),
    "denoever": ("denoever", 2),
    "westters": ("westters", 2),
    "lauwerso": ("lauwerso", 2),
    "huibertg": ("huibertg", 2),
}

OTHER_STATIONS = ["os11", "denoever", "westters", "lauwerso", "huibertg"]
OTHER_STATIONNAME = ["OS11", "DenOever", "WestTerschelling", "Lauwersoog", "Huibertgat"]
OTHER_OMNIREF = ["hoekvanh", "denhelde", "harlinge", "harlinge", "harlinge"]
OTHER_DIRREF = ["vlissing", "denhelde", "harlinge", "harlinge", "harlinge"]
OTHER_SHIFTS = [0.0, 0.1, 0.05, 0.2, 0.0]
OTHER_SEED = 180225

TEXEL_SEED = 379233

PRODUCTION_WL_STATIONS = [
    "hansweert",
    "vlissing",
    "os11",
    "hoekvanh",
    "ijmuiden",
    "denhelde",
    "denoever",
    "westters",
    "harlinge",
    "lauwerso",
    "huibertg",
    "delfzijl",
]
PRODUCTION_WL_STATIONNAME = [
    "Hansweert",
    "Vlissingen",
    "OS11",
    "Hoek van Holland",
    "IJmuiden",
    "den Helder",
    "den Oever",
    "West Terschelling",
    "Harlingen",
    "Lauwersoog",
    "Huibertgat",
    "Delfzijl",
]
