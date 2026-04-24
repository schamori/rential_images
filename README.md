# retinal_images

## Project Description

This repository contains the code for MPHY0050 coursework 2. The aim of the project is to train models for classification of myopic maculopathy using the [MMAC dataset](https://liveuclac.sharepoint.com/sites/MPHY0050/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FMPHY0050%2FShared%20Documents%2FMPHY0050%5F2025%5F2026%2FCourseworks%2FCurrent%2FGroupProject%2FData%2FTraining&viewid=e45938e4%2D54ab%2D4ebd%2Dad49%2De289b76fe39a)

The implementations will include the following:

- Baseline classification solution
- Solution including strategies to account for imbalance data
- Solution using multi-task learning
- Solution with explainability markers
- Solution incorporating uncertainty evaluation
- Solution accounting for some source of bias
- A mixture of at least 3 out of (imbalance, multi-task, interpretability, uncertainty, bias)

Models are compared using macro F1 as the primary metric (given class imbalance), alongside accuracy and quadratic weighted kappa.

## Repository Layout

```
.
├── configs/              # Experiment configs, inheriting from base.yaml
├── weights/              # Saved model checkpoints (created at training time)
├── train.py              # Main training / evaluation entry point
├── dataset.py            # FundusDataset and FundusDatasetMTL
├── multitask_model.py    # Shared-encoder multi-head MTL model
├── losses.py             # Loss functions incl. MultiTaskLoss / get_multitask_loss
├── requirements.txt
└── data.yaml             # Local dataset path (created by you, see Setup)
```

## Dependencies

All dependencies are pinned in `requirements.txt`. Key libraries include PyTorch, torchvision, HuggingFace Transformers, scikit-learn, pandas, and PyYAML.

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

The main entry point is `train.py`. It fine-tunes a pretrained ConvNeXt V2 Tiny backbone for 5-class myopic maculopathy classification, reporting accuracy, macro F1, and quadratic weighted kappa on the validation set. Early stopping is driven by macro F1.

### Configs

Every experiment is defined by a YAML config in `configs/`, which overrides fields from `configs/base.yaml`. Typical overrides include the loss function, learning rate, augmentation settings, and multi-task flags.

### Running Experiments

To run a single experiment:

```bash
python train.py --config configs/exp1_weighted_loss.yaml
```

All configs in `configs/` sequentially:

```bash
python train.py
```

Checkpoints are written to `weights/` under the experiment name.

### Multi-task learning

Multi-task variants are gated behind `multitask: true` in the experiment config, which switches `train.py` to the `FundusDatasetMTL` / multi-head model path. Auxiliary task weights (e.g. `lambda_centre`, `lambda_age`) and which auxiliaries to enable are set in the same config.
