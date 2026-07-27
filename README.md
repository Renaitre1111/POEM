# POEM: Phase Geometry Modeling with Harmonic Modulation

Official implementation of **POEM** for long-term time series forecasting.

## Datasets

Download the benchmark datasets and unzip to `./dataset/`:

```
dataset/
  ETTh1.csv   ETTh2.csv   ETTm1.csv   ETTm2.csv
  weather.csv  electricity.csv  traffic.csv
```

## Quick Start

Train and evaluate POEM on a single dataset:

```bash
bash scripts/POEM/etth1.sh
```

Available scripts: `etth1.sh`, `etth2.sh`, `ettm1.sh`, `ettm2.sh`, `weather.sh`, `electricity.sh`, `traffic.sh`.

Summarize results into a comparison table:

```bash
python summarize_results.py
```

