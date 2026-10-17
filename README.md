<h1 align="center">H2VLR-benchmarks</h1>

<h3 align="center">A Reproducible Benchmark of Few-Shot Anomaly Detection Methods for H2VLR</h3>

<p align="center">
  <a href="#-news"><b>News</b></a> •
  <a href="#-overview"><b>Overview</b></a> •
  <a href="#-methods"><b>Methods</b></a> •
  <a href="#-start"><b>Start</b></a> •
  <a href="#-email"><b>Email</b></a>
</p>

<p align="center">
  <img alt="type" src="https://img.shields.io/badge/type-benchmark-blue">
  <img alt="task" src="https://img.shields.io/badge/task-few--shot%20anomaly%20detection-purple">
  <img alt="domain" src="https://img.shields.io/badge/domain-industrial%20%7C%20medical-green">
  <a href="https://github.com/fayeH2/H2VLR-benchmarks"><img alt="stars" src="https://img.shields.io/github/stars/fayeH2/H2VLR-benchmarks?style=social"></a>
  <img alt="license" src="https://img.shields.io/badge/license-see%20method%20subdirs-yellow">
</p>

This repository provides the **reproduction code of the comparison methods** used in the **H2VLR** project, covering both **industrial** and **medical** few-shot anomaly detection (FSAD). 

## 📢 News

- **2026-10-17**: The reproduction code of all comparison methods is released.
- **2026-10-17**: Supported datasets include industrial **MVTec-AD / VisA / BTAD / MPDD** and medical **BrainMRI / BUSI / LiverCT**.

## 📊 Overview

This benchmark collects the **8 representative FSAD baselines** adopted in H2VLR. For every method we preserve:

- the **model code** (as released by the authors),
- the **data loaders** used for the industrial / medical protocols,
- the **required configs** (e.g. prompt / class-histogram files).


### Industrial & Medical Protocols

| Domain | Datasets | Setting |
| :--- | :--- | :--- |
| Industrial | MVTec-AD, VisA, BTAD, MPDD, … | few-shot (1 / 2 / 4 / 8-shot), image- & pixel-level |
| Medical | BrainMRI, BUSI, LiverCT | few-shot, image- & pixel-level |

## 🔬 Methods

| Method | Venue | Domain | Entry point | Directory |
| :--- | :---: | :---: | :--- | :--- |
| **PatchCore** | CVPR 2022 | Industrial | `patchcore/` | [`PatchCore/`](./PatchCore) |
| **RegAD** | ECCV 2022 | Industrial | `train.py` / `test.py` | [`RegAD/`](./RegAD) |
| **FastRecon** | ICCV 2023 | Industrial | `main.py` | [`FastRecon/`](./FastRecon) |
| **PromptAD** | CVPR 2024 | Industrial | `train_cls.py`, `train_industrial.py` | [`PromptAD/`](./PromptAD) |
| **MediCLIP** | MICCAI 2024 | Medical | `run_medical.py` | [`MediCLIP/`](./MediCLIP) |
| **INP-Former** | CVPR 2025 | Industrial | `INP_Former_Industrial.py` | [`INP-Former/`](./INP-Former) |
| **KAG-prompt** | AAAI 2025 | Industrial | `code/` | [`KAG-prompt/`](./KAG-prompt) |
| **UniVAD** | CVPR 2025 | Industrial | `test_univad.py` | [`UniVAD/`](./UniVAD) |

## 📁 Repository Structure

```text
H2VLR-benchmarks/
├── PatchCore/        # PatchCore (CVPR 2022)
├── RegAD/            # RegAD (ECCV 2022)
├── FastRecon/        # FastRecon (ICCV 2023)
├── PromptAD/         # PromptAD (CVPR 2024)
├── MediCLIP/         # MediCLIP (MICCAI 2024)
├── INP-Former/       # INP-Former (CVPR 2025)
├── KAG-prompt/       # KAG-prompt (AAAI 2025)
├── UniVAD/           # UniVAD (CVPR 2025)
└── .gitignore
```

## 🔱 Start

### Environment

Each method ships its own `requirements.txt`. A common starting point is:

```bash
conda create -n h2vlr-bench python=3.9
conda activate h2vlr-bench
pip install torch torchvision          # match your CUDA version
# then per method:
pip install -r <method>/requirements.txt
```

### Datasets

Arrange the datasets in a single root and point each method to it:

```text
/path/to/datasets/
├── industrial/
│   ├── MVTec/
│   ├── VisA/
│   ├── BTAD/
│   └── MPDD/
└── medical/
    ├── BrainMRI/
    ├── BUSI/
    └── LiverCT/
```

Then set the corresponding argument (e.g. `--data-root`) or environment variable in the entry script of each method.

### Run a method

Every method follows the same two-step recipe — **configure, then run**:

```bash
cd <method>            # e.g. cd RegAD
python train.py ...    # train on support set (few-shot)
python test.py  ...    # evaluate image-/pixel-level AUROC, AP, F1
```



## 📧 Email

Contact us via email: [jianghong@std.uestc.edu.cn](mailto:jianghong@std.uestc.edu.cn)

## 🏷️ License

This benchmark aggregates several third-party repositories for academic comparison. **Each method retains its original license** (see the `LICENSE` file inside each sub-directory). 
