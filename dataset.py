import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split

from augmentations import get_train_transform, get_val_transform

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data.yaml")) as f:
    data_path = yaml.safe_load(f)

DATA = data_path["data_root"]
TRAIN_IMG = f"{DATA}/Training/Training_Images"
LABELS_CSV = f"{DATA}/Training/Training_LabelsDemographic.csv"

class FundusDataset(Dataset):
    def __init__(self, df, img_dir, img_size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = get_train_transform(img_size) if augment else get_val_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        return self.transform(img), int(row["myopic_maculopathy_grade"])

class FundusDatasetMTL(Dataset):
    def __init__(self, df, img_dir, img_size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = get_train_transform(img_size) if augment else get_val_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        img_tensor = self.transform(img)
        grade = int(row["myopic_maculopathy_grade"])

        # age: normalise to [0, 1], handle missing values
        if pd.isna(row["age"]):
            age_norm = 0.0
            age_valid = 0.0
        else:
            age_norm = float(row["age"]) / 100.0
            age_valid = 1.0

        # data centre: map to 0/1 (centre 1 → 0, centre 2 → 1)
        centre = float(row["data_center"]) - 1.0
        return img_tensor, grade, age_norm, age_valid, centre


def get_class_weights(df):
    """Inverse-frequency weights per class (for weighted loss / sampler)."""
    counts = df["myopic_maculopathy_grade"].value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * len(counts)  # normalise
    return torch.tensor(weights, dtype=torch.float32)


def get_loaders(cfg):
    df = pd.read_csv(LABELS_CSV)
    labels = df["myopic_maculopathy_grade"].values

    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.15, stratify=labels, random_state=42
    )
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    use_mtl = cfg.get("multitask", False)
    DsClass = FundusDatasetMTL if use_mtl else FundusDataset
    train_ds = DsClass(train_df, TRAIN_IMG, cfg["img_size"], augment=True)
    val_ds   = DsClass(val_df,   TRAIN_IMG, cfg["img_size"], augment=False)

    # ── Sampler selection ─────────────────────────────────────────────────────
    sampler = None
    if cfg["sampler"] == "weighted":
        # WeightedRandomSampler: minority classes drawn more often (oversampling)
        # oversample_strength: 1.0 = full inverse-freq, 0.5 = sqrt (softer), 2.0 = more aggressive
        # oversample_epoch_mult: num_samples = mult * len(train_ds) (more passes over minority)
        train_labels = train_df["myopic_maculopathy_grade"].values
        class_counts = np.bincount(train_labels)
        strength = cfg.get("oversample_strength", 1.0)
        sample_weights = (1.0 / class_counts[train_labels]) ** strength
        n_samples = int(len(train_ds) * cfg.get("oversample_epoch_mult", 1.0))
        sampler = WeightedRandomSampler(sample_weights, num_samples=n_samples, replacement=True)

    elif cfg["sampler"] == "undersample":
        # Random undersampling: keep at most min-class-count samples per class
        min_count = train_df["myopic_maculopathy_grade"].value_counts().min()
        keep_idx = (
            train_df.groupby("myopic_maculopathy_grade")
            .apply(lambda g: g.sample(min_count, random_state=42))
            .index.get_level_values(1)
        )
        train_ds = DsClass(train_df.loc[keep_idx], TRAIN_IMG, cfg["img_size"], augment=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader, df
