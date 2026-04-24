import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (f1_score, cohen_kappa_score, classification_report)
from sklearn.model_selection import train_test_split

import os
import glob
import yaml
import torch
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import get_class_weights
from dataset import LABELS_CSV, TRAIN_IMG, TEST_LABELS_CSV, TEST_IMG, FundusDataset, FundusDatasetMTL
from transformers import ConvNextV2ForImageClassification
from multitask_model import MultiTaskConvNeXt
from losses import get_loss, get_multitask_loss
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ToTensor\\(\\) is deprecated.*")
warnings.filterwarnings("ignore", message=".*id2label.*")
tqdm.disable = True

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["No pathology", "Tessellated", "Diffuse CRA", "Patchy CRA", "Macular atrophy"]

def load_cfg(path):
    """Load config (supports base.yaml override)."""
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        cfg.update(yaml.safe_load(f))
    return cfg

def build_model(cfg, num_classes, device):
    """Rebuild single-task model architecture."""                           
    model = ConvNextV2ForImageClassification.from_pretrained(
        cfg["model_id"],
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    hidden_dim = model.classifier.in_features
    
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, num_classes),
    ).to(device)
    return model


def remap_classifier_keys_for_model(state, model):
    """Map legacy classifier checkpoint keys to the classifier layout expected by `model`."""
    model_keys = set(model.state_dict().keys())

    expected_weight = None
    for k in ("classifier.0.weight", "classifier.1.weight", "classifier.weight"):
        if k in model_keys:
            expected_weight = k
            break

    if expected_weight is None:
        return state

    source_weight = None
    for k in ("classifier.0.weight", "classifier.1.weight", "classifier.weight"):
        if k in state:
            source_weight = k
            break

    if source_weight is None:
        return state

    expected_bias = expected_weight.replace("weight", "bias")
    source_bias = source_weight.replace("weight", "bias")

    if source_weight != expected_weight:
        state[expected_weight] = state.pop(source_weight)
        if source_bias in state:
            state[expected_bias] = state.pop(source_bias)

    # Remove stale classifier aliases that are not expected by the current model.
    for k in (
        "classifier.weight", "classifier.bias",
        "classifier.0.weight", "classifier.0.bias",
        "classifier.1.weight", "classifier.1.bias",
    ):
        if k in state and k not in model_keys:
            state.pop(k)

    return state


def build_fixed_eval_loaders(cfg):
    """Build deterministic train/val/test loaders using the same split as training."""
    df = pd.read_csv(LABELS_CSV)
    test_df = pd.read_csv(TEST_LABELS_CSV)
    labels = df["myopic_maculopathy_grade"].values

    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.15, stratify=labels, random_state=42
    )
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    use_mtl = cfg.get("multitask", False)
    DsClass = FundusDatasetMTL if use_mtl else FundusDataset

    # For evaluation, disable augmentation on all splits.
    train_ds = DsClass(train_df, TRAIN_IMG, cfg["img_size"], augment=False)
    val_ds = DsClass(val_df, TRAIN_IMG, cfg["img_size"], augment=False)
    test_ds = DsClass(test_df, TEST_IMG, cfg["img_size"], augment=False)

    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    return train_loader, val_loader, test_loader, df, train_df, val_df, test_df


def format_class_counts(df):
    counts = df["myopic_maculopathy_grade"].value_counts().sort_index()
    return " ".join([f"C{int(c)}:{int(n)}" for c, n in counts.items()])

def evaluate(model, loader, device, criterion, CLASS_NAMES): 
    """Evaluate single task model on a validation set, returning loss, acc, F1, kappa, and report."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).logits
            total_loss += criterion(logits, labels).item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_mac = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa  = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
    return {
        "loss": total_loss / len(loader),
        "acc": acc, 
        "f1_macro": f1_mac, 
        "f1_weighted": f1_wt, 
        "kappa": kappa,
        "report": report,
    }

def evaluate_ensemble(models, loader, device, criterion, CLASS_NAMES): 
    """
    Evaluate an ensemble of single-task models.
    Averages logits across models before computing metrics.
    """
    for m in models:
        m.eval()

    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            
            logits_list = [model(imgs).logits for model in models]
            avg_logits = sum(logits_list) / len(models)

            total_loss += criterion(avg_logits, labels).item()

            preds = avg_logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_mac = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa  = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "kappa": kappa,
        "report": report,
    }

def evaluate_mtl(model, loader, device, criterion, CLASS_NAMES): 
    """
    Evaluate multi-task model on a validation set. Only classification metrics are reported.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss_cls = 0
    n_batches = 0
 
    with torch.no_grad():
        for batch in loader:
            imgs, grades, ages, age_valid, centres = batch
            imgs   = imgs.to(device)
            grades = grades.to(device)
            ages   = ages.float().to(device)
            age_valid = age_valid.float().to(device)
            centres   = centres.float().to(device)
 
            cls_logits, age_pred, centre_pred = model(imgs)
 
            total, loss_dict = criterion(
                cls_logits, age_pred, centre_pred,
                grades, ages, age_valid, centres
            )
            total_loss_cls += loss_dict["cls"]
            n_batches += 1
 
            all_preds.extend(cls_logits.argmax(1).cpu().tolist())
            all_labels.extend(grades.cpu().tolist())
 
    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_mac = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa  = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES,
                                   zero_division=0)
    return {
        "loss": total_loss_cls / max(n_batches, 1),
        "acc": acc, 
        "f1_macro": f1_mac, 
        "f1_weighted": f1_wt, 
        "kappa": kappa,
        "report": report,
    }

def evaluate_mtl_ensemble(models, loader, device, criterion, CLASS_NAMES): 
    """
    Evaluate an ensemble of multi-task models.
    Only classification metrics are reported (same as evaluate_mtl).
    """
    for m in models:
        m.eval()

    all_preds, all_labels = [], []
    total_loss_cls = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            imgs, grades, ages, age_valid, centres = batch

            imgs   = imgs.to(device)
            grades = grades.to(device)
            ages   = ages.float().to(device)
            age_valid = age_valid.float().to(device)
            centres   = centres.float().to(device)

            cls_logits_list = []
            age_preds = []
            centre_preds = []

            for model in models:
                cls_logits, age_pred, centre_pred = model(imgs)
                cls_logits_list.append(cls_logits)
                age_preds.append(age_pred)
                centre_preds.append(centre_pred)

            cls_logits = sum(cls_logits_list) / len(models)

            age_pred = sum(age_preds) / len(models)
            centre_pred = sum(centre_preds) / len(models)

            total, loss_dict = criterion(
                cls_logits, age_pred, centre_pred,
                grades, ages, age_valid, centres
            )

            total_loss_cls += loss_dict["cls"]
            n_batches += 1

            preds = cls_logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(grades.cpu().tolist())

    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_mac = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa  = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return {
        "loss": total_loss_cls / max(n_batches, 1),
        "acc": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "kappa": kappa,
        "report": report,
    }

def evaluate_test(models, loader, device, criterion, CLASS_NAMES, is_mtl=False):
    """
    Unified test evaluation entry point.

    Handles:
    - single model vs ensemble
    - single-task vs multi-task
    """

    if len(models) == 1:
        model = models[0]

        if is_mtl:
            return evaluate_mtl(model, loader, device, criterion, CLASS_NAMES)
        else:
            return evaluate(model, loader, device, criterion, CLASS_NAMES)

    if is_mtl:
        return evaluate_mtl_ensemble(models, loader, device, criterion, CLASS_NAMES)
    else:
        return evaluate_ensemble(models, loader, device, criterion, CLASS_NAMES)
    
def run_split_evaluation(split_name, exp_name, models, loader, device, criterion, cfg, CLASS_NAMES):
    """Runs evaluation for one named data split."""

    is_mtl = cfg.get("multitask", False)

    print(f"\n{'-'*60}")
    print(f"{split_name} RESULTS — {exp_name}")

    results = evaluate_test(
        models=models,
        loader=loader,
        device=device,
        criterion=criterion,
        CLASS_NAMES=CLASS_NAMES,
        is_mtl=is_mtl
    )

    print(f"  Accuracy:    {results['acc']:.4f}")
    print(f"  F1 macro:    {results['f1_macro']:.4f}")
    print(f"  F1 weighted: {results['f1_weighted']:.4f}")
    print(f"  Kappa:       {results['kappa']:.4f}")
    print(f"\nPer-class report:\n{results['report']}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to folder containing checkpoint .pt files")
    parser.add_argument("--configs", required=True,
                        help="Path to folder containing config .yaml files")
    parser.add_argument("--eval_splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to evaluate")
    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu")

    config_paths = sorted(
        glob.glob(os.path.join(args.configs, "**", "*.yaml"), recursive=True)
    )
    config_paths = [
        p for p in config_paths
        if os.path.basename(p) != "base.yaml"
    ]

    split_list = [s.lower() for s in args.eval_splits]
    print(f"\nRunning evaluation ({', '.join(split_list)}) on {len(config_paths)} configs\n")

    results_summary = []

    for cfg_path in config_paths:
        cfg = load_cfg(cfg_path)
        exp_name = os.path.splitext(os.path.basename(cfg_path))[0]

        print(f"\n{'='*70}")
        print(f"TESTING: {exp_name}")

        # ── DATA ─────────────────────────────────────────────
        train_loader, val_loader, test_loader, df, train_df, val_df, test_df = build_fixed_eval_loaders(cfg)
        class_weights = get_class_weights(df)

        print(f"  Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        print(f"  Train class counts: {format_class_counts(train_df)}")
        print(f"  Val class counts:   {format_class_counts(val_df)}")
        print(f"  Test class counts:  {format_class_counts(test_df)}")

        # ── MODEL SETUP ──────────────────────────────────────
        is_mtl = cfg.get("multitask", False)
        is_ensemble = cfg.get("ensemble", False)

        n_models = cfg.get("ensemble_n", 1) if is_ensemble else 1
        seeds = cfg.get("ensemble_seeds", [42] * n_models)[:n_models]

        models = []
        missing = False

        for seed in seeds:
            path = (
                os.path.join(args.weights, f"{exp_name}_seed{seed}.pt")
                if is_ensemble
                else os.path.join(args.weights, f"{exp_name}.pt")
            )

            if not os.path.exists(path):
                print(f"  [skip] checkpoint not found: {path}")
                missing = True
                break

            if is_mtl:
                model = MultiTaskConvNeXt(cfg).to(device)
            else:
                model = build_model(cfg, num_classes=5, device=device)

            state = torch.load(path, map_location=device)
            if not is_mtl:
                state = remap_classifier_keys_for_model(state, model)
            model.load_state_dict(state, strict=True)

            model.eval()
            models.append(model)

        if missing:
            continue

        # ── LOSS ─────────────────────────────────────────────
        criterion = (
            get_multitask_loss(cfg, class_weights, device)
            if is_mtl else
            get_loss(cfg, class_weights, device)
        )

        # ── RUN TEST ─────────────────────────────────────────
        split_to_loader = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        split_results = {}
        for split in split_list:
            split_results[split] = run_split_evaluation(
                split_name=split.upper(),
                exp_name=exp_name,
                models=models,
                loader=split_to_loader[split],
                device=device,
                criterion=criterion,
                cfg=cfg,
                CLASS_NAMES=CLASS_NAMES,
            )

        results_summary.append((exp_name, split_results))

    # ── FINAL SUMMARY ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'Experiment':<30} {'Split':<6} {'Acc':>6} {'F1':>6} {'Kappa':>6}")
    print("-" * 70)

    for name, split_results in results_summary:
        for split in split_list:
            r = split_results[split]
            print(
                f"{name:<30} "
                f"{split:<6} "
                f"{r['acc']:>6.3f} "
                f"{r['f1_macro']:>6.3f} "
                f"{r['kappa']:>6.3f}"
            )


if __name__ == "__main__":
    main()