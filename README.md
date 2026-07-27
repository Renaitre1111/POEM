# POEM: Phase Geometry Modeling with Harmonic Modulation

Official implementation of **POEM** for long-term time series forecasting.
POEM models periodic dynamics in a phase-aligned geometry using adaptive phase
interaction and harmonic phase modulation.

------

## 📦 Overview

 This repository includes:

- 🔧 Scripts for training and evaluation
- 📊 Preprocessed datasets and download links
- 📝 Detailed instructions for reproducibility

------

## 📁 Dataset Preparation

**Benchmark datasets**

The paper evaluates POEM on seven public datasets:

```
ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic
```

📥 Download from [Google Drive](https://drive.google.com/file/d/1ypgCc6iQ2Z8IB_9CY3If_KMRNQKBsI3J/view?usp=sharing), then unzip to the `./dataset/` directory.

## ⚙️ Implementation Details

- **Phase geometry construction** folds each variable into phase-aligned cycles.
- **Adaptive phase interaction** combines local circular differences with
  low-rank global phase interaction.
- **Harmonic phase modulation** adapts features through learned harmonic bases.
- **Geometry-guided forecast reconstruction** maps the phase representation
  back to the requested prediction horizon.

------

## 🚀 Running Experiments

To train and evaluate the current POEM model on a given dataset:

```bash
bash ./scripts/POEM/etth1.sh
```

To reproduce PhaseFormer with its original hyperparameters and this repository's
seed, use the matching script:

```bash
bash ./scripts/PhaseFormer/etth1.sh
```

The complete seven-dataset suites are available through:

```bash
bash ./scripts/POEM/run_all.sh
bash ./scripts/PhaseFormer/run_all.sh
```

Scripts, checkpoints, logs, and results are separated into `POEM/` and
`PhaseFormer/` families under their respective top-level directories.

------

## 📄 Citation

If you find this project helpful, please cite us:

```bibtex
@inproceedings{poem,
  title={POEM: Phase Geometry Modeling with Harmonic Modulation for Long-term Time Series Forecasting},
  author={Anonymous},
  year={2026}
}
```



------

## Acknowledgements

We would like to thank the authors of the following open-source projects for their valuable contributions, which provides significant help for our work:

- [**SparseTSF** (ICML 2024)](https://github.com/lss-1138/SparseTSF)
- [**TFB** (VLDB 2024)](https://github.com/decisionintelligence/TFB)
- [**Time-Series-Library** (THUML)](https://github.com/thuml/Time-Series-Library)

We gratefully acknowledge their contributions to the time series forecasting community.
