"""
Training Script for MMAC classification -- supports both single-task and multi-task modes:

Usage:
    python train.py --config configs/base.yaml                 # single-task
    python train.py --config configs/exp1_weighted_loss.yaml
    python train.py --config configs/exp3_ensemble.yaml
    python train.py --config configs/exp5_mtl_age.yaml         # multi-task
    python train.py                                            # run all configs
 
Multi-task mode is activated when the config contains  multitask: true.
"""
import argparse, copy, os
import yaml
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from transformers import ConvNextV2ForImageClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from dataset import get_loaders, get_class_weights, LABELS_CSV
from losses import get_loss , get_multitask_loss
from multitask_model import MultiTaskConvNeXt

CLASS_NAMES = ["No pathology", "Tessellated", "Diffuse CRA", "Patchy CRA", "Macular atrophy"]

HERE       = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(HERE, "weights")
os.makedirs(WEIGHT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
def load_cfg(path):
    """Load a config yaml, with the option to override a base.yaml."""
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f) # load base config
    with open(path) as f:
        cfg.update(yaml.safe_load(f))   # experiment overrides base
    return cfg

# ── Metrics ────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion): # single task
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
        "acc": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt, "kappa": kappa,
        "report": report,
    }

def evaluate_mtl(model, loader, device, criterion): # multi task
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
 
            # compute full MTL loss for logging
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
        "acc": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt, "kappa": kappa,
        "report": report,
    }

# ── Single training run ────────────────────────────────────────────────────────
def train_one(cfg, seed, device, train_loader, val_loader, class_weights, save_path):
    """Train a single task model, returning the best one after early stopping."""
    torch.manual_seed(seed)
    model = ConvNextV2ForImageClassification.from_pretrained(
        cfg["model_id"], num_labels=cfg["num_classes"], ignore_mismatched_sizes=True
    ).to(device)

    model.classifier.add_module("mc_dropout", nn.Dropout(p=0.2))

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0))
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg.get("t_max", 100), eta_min=cfg.get("min_lr", 1e-6))
    criterion = get_loss(cfg, class_weights, device)
    best_f1, patience_left = 0.0, cfg["patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs).logits, labels)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            train_loss += loss.item()
        
        scheduler.step()

        m = evaluate(model, val_loader, device, criterion)
        print(f"  Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={m['loss']:.4f}  acc={m['acc']:.3f}  "
              f"f1_macro={m['f1_macro']:.3f}  kappa={m['kappa']:.3f}")

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            patience_left = cfg["patience"]
            torch.save(model.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch} — best f1_macro={best_f1:.3f}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# ── Single multi-task training run ────────────────────────────────────────────────────────
def train_one_mtl(cfg, seed, device, train_loader, val_loader, class_weights, save_path):
    """Train a multi-task model, returning the best one after early stopping."""
    torch.manual_seed(seed)
    model = MultiTaskConvNeXt(cfg).to(device)
 
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = get_multitask_loss(cfg, class_weights, device)
    best_f1, patience_left = 0.0, cfg["patience"]
 
    for epoch in range(1, cfg["epochs"] + 1):
        # DANN: ramp up gradient reversal strength (Ganin et al. schedule)
        if cfg.get("dann", False):
            p = epoch / cfg["epochs"]
            grl_lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            model.set_grl_lambda(grl_lam)    

        model.train()
        train_losses = {"cls": 0, "age": 0, "centre": 0, "total": 0}
        n_batches = 0
 
        for batch in train_loader:
            imgs, grades, ages, age_valid, centres = batch
            imgs      = imgs.to(device)
            grades    = grades.to(device)
            ages      = ages.float().to(device)
            age_valid = age_valid.float().to(device)
            centres   = centres.float().to(device)
 
            cls_logits, age_pred, centre_pred = model(imgs)
 
            total, loss_dict = criterion(
                cls_logits, age_pred, centre_pred,
                grades, ages, age_valid, centres
            )
 
            optimiser.zero_grad()
            total.backward()
            optimiser.step()
 
            for k in train_losses:
                train_losses[k] += loss_dict[k]
            n_batches += 1
 
        # validation (classification metrics only)
        m = evaluate_mtl(model, val_loader, device, criterion)
 
        # logging
        tl = {k: v / n_batches for k, v in train_losses.items()}
        grl_str = f"  grl_λ={grl_lam:.3f}" if cfg.get("dann", False) else ""
        print(f"  Epoch {epoch:3d}/{cfg['epochs']}  "
              f"loss_total={tl['total']:.4f}  "
              f"loss_cls={tl['cls']:.4f}  "
              f"loss_age={tl['age']:.4f}  "
              f"loss_ctr={tl['centre']:.4f}  |  "
              f"val_acc={m['acc']:.3f}  "
              f"val_f1mac={m['f1_macro']:.3f}  "
              f"val_kappa={m['kappa']:.3f}{grl_str}")
 
        # early stopping on F1
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            patience_left = cfg["patience"]
            torch.save(model.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch} — best f1_macro={best_f1:.3f}")
                break
 
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# ── Main ───────────────────────────────────────────────────────────────────────
def run_experiment(config_path, device):
    """Run a single experiment specified by a config yaml, returning results."""
    cfg = load_cfg(config_path)
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"\n{'='*60}")
    print(f"=== {exp_name} | device={device} ===")
    print(yaml.dump(cfg, default_flow_style=False))

    train_loader, val_loader, df = get_loaders(cfg)
    class_weights = get_class_weights(df)

    is_mtl = cfg.get("multitask", False)

    if is_mtl:
        # ── multi-task path ──────────────────────────────────────────
        model = train_one_mtl(
            cfg, seed=42, device=device,
            train_loader=train_loader, val_loader=val_loader,
            class_weights=class_weights,
            save_path=os.path.join(WEIGHT_DIR, f"{exp_name}.pt"),
        )
        criterion = get_multitask_loss(cfg, class_weights, device)
        m = evaluate_mtl(model, val_loader, device, criterion)

    elif not cfg["ensemble"]:
        # ── single-task path ──────────────────────────────────────────
        model = train_one(cfg, seed=42, device=device,
                          train_loader=train_loader, val_loader=val_loader,
                          class_weights=class_weights,
                          save_path=os.path.join(WEIGHT_DIR, f"{exp_name}.pt"))
        m = evaluate(model, val_loader, device, get_loss(cfg, class_weights, device))
    else:
        # train N models, average logits
        models = []
        for i, seed in enumerate(cfg["ensemble_seeds"][:cfg["ensemble_n"]]):
            print(f"\n--- Ensemble member {i+1}/{cfg['ensemble_n']} (seed={seed}) ---")
            m_path = os.path.join(WEIGHT_DIR, f"{exp_name}_seed{seed}.pt")
            m = train_one(cfg, seed=seed, device=device,
                          train_loader=train_loader, val_loader=val_loader,
                          class_weights=class_weights, save_path=m_path)
            models.append(m)

        # Ensemble evaluation
        for mod in models:
            mod.eval()
        all_preds, all_labels = [], []
        criterion = get_loss(cfg, class_weights, device)
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                avg_logits = sum(mod(imgs).logits for mod in models) / len(models)
                all_preds.extend(avg_logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.tolist())

        m = {
            "acc":         sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
            "f1_macro":    f1_score(all_labels, all_preds, average="macro",    zero_division=0),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "kappa":       cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
            "report":      classification_report(all_labels, all_preds,
                                                 target_names=CLASS_NAMES, zero_division=0),
        }

    print(f"\n{'='*60}")
    print(f"Final results — {exp_name}")
    print(f"  Accuracy:    {m['acc']:.4f}")
    print(f"  F1 macro:    {m['f1_macro']:.4f}")
    print(f"  F1 weighted: {m['f1_weighted']:.4f}")
    print(f"  QW Kappa:    {m['kappa']:.4f}")
    print(f"\nPer-class report:\n{m['report']}")
    return exp_name, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Path to a config yaml. Omit to run all configs in configs/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if args.config:
        configs = [args.config]
    else:
        # run every non-base config in configs/
        import glob as _glob
        cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        configs = sorted(
            p for p in _glob.glob(os.path.join(cfg_dir, "**", "*.yaml"), recursive=True)
            if os.path.basename(p) != "base.yaml"
        )
        print(f"No --config specified. Running all {len(configs)} experiments:\n"
              + "\n".join(f"  {c}" for c in configs))

    summary = []
    for cfg_path in configs:
        exp_name, m = run_experiment(cfg_path, device)
        summary.append((exp_name, m))

    if len(summary) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'Experiment':<30} {'Acc':>6} {'F1mac':>6} {'F1wt':>6} {'Kappa':>6}")
        print("-" * 60)
        for name, m in summary:
            print(f"{name:<30} {m['acc']:>6.3f} {m['f1_macro']:>6.3f} "
                  f"{m['f1_weighted']:>6.3f} {m['kappa']:>6.3f}")


if __name__ == "__main__":
    main()
