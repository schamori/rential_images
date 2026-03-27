"""
Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/exp1_weighted_loss.yaml
    python train.py --config configs/exp3_ensemble.yaml
"""
import argparse, copy, os
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from transformers import ConvNextV2ForImageClassification

from dataset import get_loaders, get_class_weights, LABELS_CSV
from losses import get_loss
import pandas as pd

CLASS_NAMES = ["No pathology", "Tessellated", "Diffuse CRA", "Patchy CRA", "Macular atrophy"]

HERE       = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(HERE, "weights")
os.makedirs(WEIGHT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
def load_cfg(path):
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        cfg.update(yaml.safe_load(f))   # experiment overrides base
    return cfg

# ── Metrics ────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion):
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
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES,
                                   zero_division=0)
    return {
        "loss": total_loss / len(loader),
        "acc": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt, "kappa": kappa,
        "report": report,
    }

# ── Single training run ────────────────────────────────────────────────────────
def train_one(cfg, seed, device, train_loader, val_loader, class_weights, save_path):
    torch.manual_seed(seed)
    model = ConvNextV2ForImageClassification.from_pretrained(
        cfg["model_id"], num_labels=cfg["num_classes"], ignore_mismatched_sizes=True
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = get_loss(cfg, class_weights, device)
    best_acc, patience_left = 0.0, cfg["patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs).logits, labels)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            train_loss += loss.item()

        m = evaluate(model, val_loader, device, criterion)
        print(f"  Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={m['loss']:.4f}  acc={m['acc']:.3f}  "
              f"f1_macro={m['f1_macro']:.3f}  kappa={m['kappa']:.3f}")

        if m["acc"] > best_acc:
            best_acc = m["acc"]
            patience_left = cfg["patience"]
            torch.save(model.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch} — best acc={best_acc:.3f}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# ── Main ───────────────────────────────────────────────────────────────────────
def run_experiment(config_path, device):
    cfg = load_cfg(config_path)
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"\n{'='*60}")
    print(f"=== {exp_name} | device={device} ===")
    print(yaml.dump(cfg, default_flow_style=False))

    train_loader, val_loader, df = get_loaders(cfg)
    class_weights = get_class_weights(df)

    if not cfg["ensemble"]:
        model = train_one(cfg, seed=42, device=device,
                          train_loader=train_loader, val_loader=val_loader,
                          class_weights=class_weights,
                          save_path=os.path.join(WEIGHT_DIR, f"{exp_name}.pt"))
        m = evaluate(model, val_loader, device, get_loss(cfg, class_weights, device))
    else:
        # Train N models, average logits
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

        from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.config:
        configs = [args.config]
    else:
        # run every non-base config in configs/
        cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        configs = sorted(
            os.path.join(cfg_dir, f)
            for f in os.listdir(cfg_dir)
            if f.endswith(".yaml") and f != "base.yaml"
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
