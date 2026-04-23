import argparse, os, re, yaml, random, warnings
import numpy as np 
import torch, torch.nn as nn
import matplotlib.pyplot as plt, matplotlib.cm as mcm
import pandas as pd

from collections import defaultdict
from PIL import Image
from transformers import ConvNextV2ForImageClassification
from transformers.utils import logging
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score

from dataset import TRAIN_IMG, LABELS_CSV, get_loaders, get_class_weights
from train import CLASS_NAMES, load_cfg, get_class_weights, get_loss, evaluate
from augmentations import get_ttda_transform, tensor_to_pil
from uncertainty_plotting import HERE, UNC_DIR, plot_confusion_grid, plot_reliability_grid, plot_uncertainty_metrics_grid, plot_entropy_vs_variance_scatter, save, results_table

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
logging.disable_progress_bar()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# plt.rcParams.update({"axes.facecolor": "#1a1a2e", "figure.facecolor": "#16213e",
#                       "text.color": "white", "axes.titlecolor": "white"})

# ── Model loading and inference ───────────────────────────────────────────────—
def load_ensembles(ckpt_dir):
    all_ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    ensembles = defaultdict(list)

    for ckpt in all_ckpts:
        match = re.search(r"ensemble_(\w+)_seed\d+", ckpt)
        if not match:
            continue
        exp_name = match.group(1)
        ensembles[exp_name].append(os.path.join(ckpt_dir, ckpt))

    for exp_name in ensembles:
        ensembles[exp_name] = sorted(
            ensembles[exp_name],
            key=lambda x: int(re.search(r"_seed(\d+)", x).group(1))
        )

    print("\nDetected ensembles:")
    for k, v in ensembles.items():
        print(f"  {k}: {len(v)} models")

    return dict(ensembles)

class LogitModel(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).logits

def load_model(ckpt):
    base = ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-tiny-1k-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    hidden_dim = base.classifier.in_features 
    base.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(hidden_dim, NUM_CLASSES),)
    base.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return LogitModel(base).eval().to(DEVICE)

def enable_mc_dropout(model):
    activated = False
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            activated = True
    if not activated:
        print("Warning: No Dropout layers found to activate for MC Dropout!")
    if activated:
        print("MC Dropout enabled: Dropout layers set to train mode during inference.")

def inference(model, val_loader, mcdo=False, ttda=False, T=5):
    all_probs, all_vars, all_labels, all_preds, all_MI, all_vars_epis, all_vars_alea = [], [], [], [], [], [], []

    model.eval()
    if mcdo:
        enable_mc_dropout(model)
    if ttda:
        ttda_transforms = get_ttda_transform(n_aug=4)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            all_labels.append(labels.cpu())

            mc_passes = T if mcdo else 1

            probs_mc = []        
            vars_ttda_mc = []     

            for t in range(mc_passes):

                if ttda:
                    probs_aug = []

                    for transform in ttda_transforms:
                        aug_imgs = torch.stack([transform(img) for img in imgs]).to(DEVICE)
                        logits = model(aug_imgs)
                        if torch.isnan(logits).any():
                            print("NaN logits detected!")
                        probs_aug.append(torch.softmax(logits, dim=1))

                    probs_aug = torch.stack(probs_aug)  

                    mean_ttda = probs_aug.mean(dim=0)   
                    var_ttda  = probs_aug.var(dim=0, unbiased=False)

                    probs_mc.append(mean_ttda)
                    vars_ttda_mc.append(var_ttda)

                else:
                    logits = model(imgs)
                    probs = torch.softmax(logits, dim=1)

                    probs_mc.append(probs)

            probs_mc = torch.stack(probs_mc)               

            mean_probs = probs_mc.mean(dim=0)             
            var_epistemic = probs_mc.var(dim=0, unbiased=False)           

            MI_probs = mutual_information(probs_mc) if mcdo else None

            if ttda:
                vars_ttda_mc = torch.stack(vars_ttda_mc)    
                var_aleatoric = vars_ttda_mc.mean(dim=0)    
            else:
                var_aleatoric = torch.zeros_like(mean_probs)

            var_total = var_epistemic + var_aleatoric

            all_probs.append(mean_probs.cpu())
            all_vars.append(var_total.cpu())   
            all_preds.append(mean_probs.argmax(dim=1).cpu())
            all_vars_epis.append(var_epistemic.cpu())
            all_vars_alea.append(var_aleatoric.cpu())

            if MI_probs is not None:
                all_MI.append(MI_probs.cpu())

    all_probs  = torch.cat(all_probs)
    all_vars   = torch.cat(all_vars) if len(all_vars) > 0 else None
    all_labels = torch.cat(all_labels)
    all_preds  = torch.cat(all_preds)
    all_MI     = torch.cat(all_MI) if len(all_MI) > 0 else None
    all_vars_epis = torch.cat(all_vars_epis) if len(all_vars_epis) > 0 else None
    all_vars_alea = torch.cat(all_vars_alea) if len(all_vars_alea) > 0 else None
    
    return all_probs, all_vars, all_labels, all_preds, all_MI, all_vars_epis, all_vars_alea

def evaluate_single_model(name, path, val_loader, mcdo=True, ttda=False):
    model = load_model(path).to(DEVICE)
    probs, vars, labels, preds, MI_dropout, vars_epis, vars_alea = inference(model, val_loader, mcdo=mcdo, ttda=ttda)
    del model
    torch.cuda.empty_cache()

    print_class_distribution(name, labels, preds)
    
    return probs, vars, labels, preds, MI_dropout, vars_epis, vars_alea

def evaluate_ensemble(name, paths, val_loader, ttda=False):
    all_probs_list, all_vars_alea = [], []
    labels = None
    for p in paths:
        model = load_model(p).to(DEVICE)
        if labels is None:
            probs, _, labels, _, _, _, vars_alea = inference(model, val_loader, mcdo=False, ttda=ttda)
            all_probs_list.append(probs)
            all_vars_alea.append(vars_alea)
        else:
            probs, _, _, _, _, _, vars_alea = inference(model, val_loader, mcdo=False, ttda=ttda)
            all_probs_list.append(probs)
            all_vars_alea.append(vars_alea)
        del model
        torch.cuda.empty_cache()

    stacked_probs = torch.stack(all_probs_list)  
    probs = stacked_probs.mean(dim=0)       

    vars_epis = stacked_probs.var(dim=0)
    vars_alea = torch.stack(all_vars_alea).mean(dim=0) if ttda else torch.zeros_like(vars_epis)
    vars = vars_epis + vars_alea

    preds = probs.argmax(dim=1)   
    MI_ensemble = mutual_information(stacked_probs)

    print_class_distribution(name, labels, preds)

    return probs, vars, labels, preds, MI_ensemble, vars_epis, vars_alea

def evaluate(name, paths, val_loader, mode="deterministic"):
    if mode == "deterministic":
        path = random.choice(paths)
        return evaluate_single_model(name, path, val_loader, mcdo=False, ttda=False)
    elif mode == "mcdo":
        path = random.choice(paths)
        return evaluate_single_model(name, path, val_loader, mcdo=True, ttda=False)
    elif mode == "ensemble":
        return evaluate_ensemble(name, paths, val_loader)
    elif mode == "ttda":
        path = random.choice(paths)
        return evaluate_single_model(name, path, val_loader, mcdo=False, ttda=True)
    elif mode == "mcdo_ttda":
        path = random.choice(paths)
        return evaluate_single_model(name, path, val_loader, mcdo=True, ttda=True)
    elif mode == "ensemble_ttda":
        return evaluate_ensemble(name, paths, val_loader, ttda=True)
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")

# ── Uncertainty analysis ———————──────────────────────────────────────────────—

def print_class_distribution(name, labels, preds):
    print("\nClass distribution:")
    print(f"{'Class':<20} {'#GT':>5} {'#Pred':>7}")
    for i, name in enumerate(CLASS_NAMES):
        n_gt   = (labels == i).sum().item()
        n_pred = (preds == i).sum().item()
        print(f"{name:<20} {n_gt:>5} {n_pred:>7}")

def shannon_entropy(probs, base=torch.e):
    probs = probs.clamp(min=1e-12)
    log_base = np.log(base)
    entropy = -torch.sum(probs * torch.log(probs) / log_base, dim=1)
    return entropy

def mutual_information(probs, eps=1e-12):
    probs = probs.clamp(min=eps)
    mean_probs = probs.mean(dim=0)
    H_mean = -torch.sum(mean_probs * torch.log(mean_probs), dim=1)
    H_each = -torch.sum(probs * torch.log(probs), dim=2)
    E_H = H_each.mean(dim=0)
    MI = H_mean - E_H
    return MI

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True,
                        help="Path to folder containing checkpoint .pt files")
    args = parser.parse_args()
    ckpt_dir = args.checkpoints

    ensembles = load_ensembles(ckpt_dir)

    print("Loading validation data...")
    cfg = load_cfg(os.path.join(HERE, "configs/base.yaml"))
    _, val_loader, df = get_loaders(cfg)

    results = {}

    for name, paths in ensembles.items():

        # Single model | No MC Dropout | No TTDA
        r = f"{name}_deterministic"
        print(f"\n{'='*60}")
        print(f"\nEvaluating single model - {r}")
        all_probs, all_vars, all_labels, all_preds, _, _, _ = evaluate(name, paths, val_loader, mode="deterministic")
        entropies = shannon_entropy(all_probs)

        results[r] = {
            "model": name,
            "method": "deterministic",
            "probs": all_probs, 
            "vars": None, 
            "labels": all_labels, 
            "preds": all_preds,
            "entropy": entropies,
            "MI_dropout": None,
            "MI_ensemble": None,
            "vars_epistemic": None,
            "vars_aleatoric": None
            }
        results_table(results[r], r)

        # Single model | MC Dropout | No TTDA
        r = f"{name}_mcdo"
        print(f"\n{'='*60}")
        print(f"\nEvaluating single model - {r}")
        all_probs, all_vars, all_labels, all_preds, MI_dropout, _, _ = evaluate(name, paths, val_loader, mode="mcdo")
        entropies = shannon_entropy(all_probs)
        
        results[r] = {
            "model": name,
            "method": "mcdo",
            "probs": all_probs, 
            "vars": all_vars, 
            "labels": all_labels, 
            "preds": all_preds,
            "entropy": entropies,
            "MI_dropout": MI_dropout,
            "MI_ensemble": None,
            "vars_epistemic": None,
            "vars_aleatoric": None
            }
        results_table(results[r], r)

        # Ensemble | No MC Dropout | No TTDA
        r = f"{name}_ensemble"
        print(f"\n{'='*60}")
        print(f"\nEvaluating ensemble - {r}")
        all_probs, all_vars, all_labels, all_preds, MI_ensemble, _, _ = evaluate(name, paths, val_loader, mode="ensemble")
        entropies = shannon_entropy(all_probs)
        
        results[r] = {
            "model": name,
            "method": "ensemble",
            "probs": all_probs, 
            "vars": all_vars, 
            "labels": all_labels, 
            "preds": all_preds,
            "entropy": entropies,
            "MI_dropout": None,
            "MI_ensemble": MI_ensemble,
            "vars_epistemic": None,
            "vars_aleatoric": None
            }
        results_table(results[r], r)
        
        # Single model | No MC Dropout | TTDA
        r = f"{name}_ttda"
        print(f"\n{'='*60}")
        print(f"\nEvaluating single model - {r}")
        all_probs, all_vars, all_labels, all_preds, _, _, _ = evaluate(name, paths, val_loader, mode="ttda")
        entropies = shannon_entropy(all_probs)
        
        results[r] = {
            "model": name,
            "method": "ttda",
            "probs": all_probs, 
            "vars": all_vars, 
            "labels": all_labels, 
            "preds": all_preds,
            "entropy": entropies,
            "MI_dropout": None,
            "MI_ensemble": None,
            "vars_epistemic": None,
            "vars_aleatoric": None
            }
        results_table(results[r], r)

        # Single model | MC Dropout | TTDA
        # r = f"{name}_mcdo_ttda"
        # print(f"\n{'='*60}")
        # print(f"\nEvaluating single model - {r}")
        # all_probs, all_vars, all_labels, all_preds, MI_dropout, vars_epis, vars_alea =  evaluate(name, paths, val_loader, mode="mcdo_ttda")
        # entropies = shannon_entropy(all_probs)
        
        # results[r] = {
        #     "model": name,
        #     "method": "mcdo_ttda",
        #     "probs": all_probs, 
        #     "vars": all_vars, 
        #     "labels": all_labels, 
        #     "preds": all_preds,
        #     "entropy": entropies,
        #     "MI_dropout": MI_dropout,
        #     "MI_ensemble": None,
        #     "vars_epistemic": vars_epis,
        #     "vars_aleatoric": vars_alea
        #     }
        # results_table(results[r], r)
        
        # Ensemble | No MC Dropout | TTDA
        # r = f"{name}_ensemble_ttda"
        # print(f"\n{'='*60}")
        # print(f"\nEvaluating ensemble - {r}")
        # all_probs, all_vars, all_labels, all_preds, MI_ensemble, vars_epis, vars_alea = evaluate(name, paths, val_loader, mode="ensemble_ttda")
        # entropies = shannon_entropy(all_probs)

        # results[r] = {
        #     "model": name,
        #     "method": "ensemble_ttda",
        #     "probs": all_probs, 
        #     "vars": all_vars, 
        #     "labels": all_labels, 
        #     "preds": all_preds,
        #     "entropy": entropies,
        #     "MI_dropout": None,
        #     "MI_ensemble": MI_ensemble,
        #     "vars_epistemic": vars_epis,
        #     "vars_aleatoric": vars_alea
        #     }
        # results_table(results[r], r)


    plot_confusion_grid(results, CLASS_NAMES, save_path=os.path.join(UNC_DIR, "confusion_grid.png"))
    plot_reliability_grid(results, n_bins=15)
    plot_uncertainty_metrics_grid(results)
    plot_entropy_vs_variance_scatter(results)

    print(f"\nDone — all plots saved to {UNC_DIR}/")

if __name__ == "__main__":
    main()