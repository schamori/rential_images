import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (f1_score, cohen_kappa_score, classification_report)

import os
import glob
import yaml
import torch
import pandas as pd
import argparse
import warnings
from tqdm import tqdm

from dataset import get_loaders, get_class_weights
from dataset import TEST_LABELS_CSV, TEST_IMG, FundusDataset, FundusDatasetMTL
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
        torch.nn.Dropout(0.3),
        torch.nn.Linear(hidden_dim, num_classes),
    ).to(device)
    return model

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
    
def run_test_evaluation(exp_name, models, test_loader, device, criterion, cfg, CLASS_NAMES):
    """Runs final test evaluation for an experiment."""

    is_mtl = cfg.get("multitask", False)

    print(f"\n{'-'*60}")
    print(f"TEST RESULTS — {exp_name}")

    results = evaluate_test(
        models=models,
        loader=test_loader,
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
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    config_paths = sorted(
        glob.glob(os.path.join(args.configs, "**", "*.yaml"), recursive=True)
    )
    config_paths = [
        p for p in config_paths
        if os.path.basename(p) != "base.yaml"
    ]

    print(f"\nRunning TEST evaluation on {len(config_paths)} configs\n")

    results_summary = []

    for cfg_path in config_paths:
        cfg = load_cfg(cfg_path)
        exp_name = os.path.splitext(os.path.basename(cfg_path))[0]

        print(f"\n{'='*70}")
        print(f"TESTING: {exp_name}")

        # ── DATA ─────────────────────────────────────────────
        train_loader, val_loader, test_loader, df = get_loaders(cfg, return_test=True)
        class_weights = get_class_weights(df)

        # ── MODEL SETUP ──────────────────────────────────────
        is_mtl = cfg.get("multitask", False)
        is_ensemble = cfg.get("ensemble", False)

        n_models = cfg.get("ensemble_n", 1) if is_ensemble else 1
        seeds = cfg.get("ensemble_seeds", [42] * n_models)[:n_models]

        models = []

        for seed in seeds:
            path = (
                os.path.join(args.weights, f"{exp_name}_seed{seed}.pt")
                if is_ensemble
                else os.path.join(args.weights, f"{exp_name}.pt")
            )

            if is_mtl:
                model = MultiTaskConvNeXt(cfg).to(device)
            else:
                model = build_model(cfg, num_classes=5, device=device)

            state = torch.load(path, map_location=device)
            model.load_state_dict(state, strict=True)

            model.eval()
            models.append(model)

        # ── LOSS ─────────────────────────────────────────────
        criterion = (
            get_multitask_loss(cfg, class_weights, device)
            if is_mtl else
            get_loss(cfg, class_weights, device)
        )

        # ── RUN TEST ─────────────────────────────────────────
        results = run_test_evaluation(
            exp_name=exp_name,
            models=models,
            test_loader=test_loader,
            device=device,
            criterion=criterion,
            cfg=cfg,
            CLASS_NAMES=CLASS_NAMES
        )

        results_summary.append((exp_name, results))

    # ── FINAL SUMMARY ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL TEST SUMMARY")
    print(f"{'Experiment':<30} {'Acc':>6} {'F1':>6} {'Kappa':>6}")
    print("-" * 70)

    for name, r in results_summary:
        print(
            f"{name:<30} "
            f"{r['acc']:>6.3f} "
            f"{r['f1_macro']:>6.3f} "
            f"{r['kappa']:>6.3f}"
        )


if __name__ == "__main__":
    main()