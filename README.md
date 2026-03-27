# retinal_images

## Project Description

This repository contains the code for MPHY0050 coursework 2. The aim of the project is to train models for classification of Myopic Maculopathytrain using the [MMAC dataset](https://liveuclac.sharepoint.com/sites/MPHY0050/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FMPHY0050%2FShared%20Documents%2FMPHY0050%5F2025%5F2026%2FCourseworks%2FCurrent%2FGroupProject%2FData%2FTraining&viewid=e45938e4%2D54ab%2D4ebd%2Dad49%2De289b76fe39a)

The implementations will include the following:
- Baseline classification solution
- Solution including strategies to account for imbalance data
- Solution using multi-task learning
- Solution with explainability markers
- Solution incorporating uncertainty evaluation
- Solution accounting for some source of bias
- A mixture of at least 3 out of (imbalance, multi-task, interpretability, uncertainty, bias)

These models are evaluated using a comparative validation experiment

## Dependencies

- pyyaml
- numpy
- pandas
- torch
- torchvision
- pillow
- matplotlib
- scikit-learn
- transformers

Dependencies are listed in requirements.txt

## Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `data.yaml` with the path to your local copy of the [MMAC dataset](https://liveuclac.sharepoint.com/sites/MPHY0050/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FMPHY0050%2FShared%20Documents%2FMPHY0050%5F2025%5F2026%2FCourseworks%2FCurrent%2FGroupProject%2FData%2FTraining&viewid=e45938e4%2D54ab%2D4ebd%2Dad49%2De289b76fe39a):

```yaml
data_root: "/path/to/your/data"
```

## Training

The main training script is `train.py`. It fine-tunes a pretrained ConvNeXtV2 model for myopic maculopathy classification and evaluates using accuracy, macro F1, and quadratic weighted kappa.

To run a single experiment:
```bash
uv run python train.py --config configs/exp1_weighted_loss.yaml
```

To run all experiments:
```bash
uv run python train.py
```

Experiment configs are stored in `configs/`. Each experiment overrides the base config in `configs/base.yaml`. Model weights are saved to `weights/`.


