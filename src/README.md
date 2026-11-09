# kuststatistiek

Python port of the KNMI wind and sea water level statistics (de Valk & van den Brink, Feb2026 R scripts),
made by HKV for the BOI coastal statistics (PR5197). The port reproduces the R results, bootstrap samples
included: it reimplements R's random number generator (`set.seed`, `runif`, `rnorm`, `sample`) and the
R functions that are used (`approx`, `quantile`, `loess`, `write.csv`, ...).

See the [documentation](https://hkv-products-services.github.io/boi-kust-statistiek) for the example notebooks.
