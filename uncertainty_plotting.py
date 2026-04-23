import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from scipy.stats import spearmanr, pearsonr

from train import CLASS_NAMES

HERE        = os.path.dirname(os.path.abspath(__file__))
UNC_DIR     = os.path.join(HERE, "unc_plots")

def save(fig, name):
    os.makedirs(UNC_DIR, exist_ok=True)
    path = os.path.join(UNC_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  {path}")

def prepare_values(values):
    if values is None:
        return None
    values = values.cpu()
    if values.ndim > 1:
        values = values.mean(dim=1)
    return values.numpy()

def summarize(name, values, correct_mask):
    if values is None:
        return
    mean = values.mean()
    std = values.std()
    corr = values[correct_mask].mean() if correct_mask.any() else np.nan
    incorr = values[~correct_mask].mean() if (~correct_mask).any() else np.nan
    print(f"{name:<12} {mean:8.4f} {std:8.4f} {corr:10.4f} {incorr:10.4f}")

def correlate(name, values, error):
    if values is None:
        return
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return
    sp, sp_p = spearmanr(values[mask], error[mask])
    pr, pr_p = pearsonr(values[mask], error[mask])
    n = mask.sum()
    print(f"{name:<12} {sp:10.4f} {sp_p:10.2e} {pr:10.4f} {pr_p:10.2e} {n:6d}")

def results_table(data, r):
    preds = data["preds"].cpu().numpy()
    labels = data["labels"].cpu().numpy()

    acc = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)

    print(f"\nClassification results — {r}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 macro:    {f1_macro:.4f}")
    print(f"  F1 weighted: {f1_weighted:.4f}")
    print(f"  QW Kappa:    {kappa:.4f}")
    print(f"\nPer-class report:\n{report}")

    print("\nUncertainty summary:")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Correct':>10} {'Incorrect':>10}")

    correct_mask = preds == labels
    error = (~correct_mask).astype(int)

    entropy_vals = prepare_values(data.get("entropy"))
    var_vals = prepare_values(data.get("vars"))

    MI_vals = data.get("MI_dropout")
    if MI_vals is None:
        MI_vals = data.get("MI_ensemble")
    MI_vals = prepare_values(MI_vals)

    summarize("Entropy", entropy_vals, correct_mask)
    summarize("Variance", var_vals, correct_mask)
    summarize("MI", MI_vals, correct_mask)

    print("\nUncertainty-error correlation results")
    print(f"{'Metric':<12} {'Spearman':>10} {'p-val':>10} {'Pearson':>10} {'p-val':>10} {'N':>6}")

    correlate("Entropy", entropy_vals, error)
    correlate("Variance", var_vals, error)
    correlate("MI", MI_vals, error)

def plot_confusion_grid(results, class_names, save_path=None, normalize=True):
    models = sorted(set(v["model"] for v in results.values()))
    methods = sorted(set(v["method"] for v in results.values()))

    n_rows, n_cols = len(models), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, model in enumerate(models):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            key = next((k for k, v in results.items() if v["model"] == model and v["method"] == method), None)

            if key is None:
                ax.axis("off")
                continue

            labels = results[key]["labels"].cpu().numpy()
            preds  = results[key]["preds"].cpu().numpy()

            num_classes = len(class_names)
            cm = np.zeros((num_classes, num_classes), dtype=float)

            for t, p in zip(labels, preds):
                cm[t, p] += 1

            if normalize:
                cm = cm / cm.sum(axis=1, keepdims=True)
                cm = np.nan_to_num(cm)

            im = ax.imshow(cm)

            if r == 0:
                ax.set_title(method.upper(), fontsize=13)
            if c == 0:
                ax.set_ylabel(model, fontsize=13)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Confusion Matrices", fontsize=18)
    fig.tight_layout()

    save(fig, "confusion_grid")

def compute_reliability(probs, labels, n_bins=15):
    confidences, preds = probs.max(dim=1)
    correct = preds.eq(labels)

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2).numpy()
    bin_width = 1 / n_bins

    bin_acc, bin_conf, bin_frac  = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)

        if mask.sum() == 0:
            continue

        bc = confidences[mask].mean().item()
        ba = correct[mask].float().mean().item()
        bf = mask.float().sum().item() / len(confidences)

        bin_conf[i], bin_acc[i], bin_frac[i] = bc, ba, bf
        ece += bf * abs(bc - ba)

    return {
        "bin_centres": bin_centres,
        "bin_width": bin_width,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_frac": bin_frac,
        "ece": ece
    }

def plot_reliability_grid(results, n_bins=15):

    models = sorted(set(v["model"] for v in results.values()))
    methods = sorted(set(v["method"] for v in results.values()))

    n_rows, n_cols = len(models) * 2, len(methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, gridspec_kw={"height_ratios": [2, 1] * len(models)})

    for m_idx, model in enumerate(models):
        for c, method in enumerate(methods):

            ax_top = axes[2*m_idx, c]
            ax_bot = axes[2*m_idx + 1, c]

            key = next((k for k, v in results.items() if v["model"] == model and v["method"] == method), None)

            if key is None:
                ax_top.axis("off")
                ax_bot.axis("off")
                continue

            data = results[key]
            stats = compute_reliability(data["probs"], data["labels"], n_bins)

            bc = stats["bin_centres"]
            bw = stats["bin_width"]
            acc = stats["bin_acc"]
            conf = stats["bin_conf"]
            frac = stats["bin_frac"]
            ece = stats["ece"]

            ax_top.bar(bc, acc, width=bw * 0.9, color="#4C72B0", alpha=0.85)
            ax_top.bar(bc, np.clip(conf - acc, 0, None), width=bw * 0.9, bottom=acc, color="#DD8452", alpha=0.7)
            ax_top.bar(bc, np.clip(acc - conf, 0, None), width=bw * 0.9, bottom=conf, color="#55A868", alpha=0.7)

            ax_top.plot([0, 1], [0, 1], "k--", lw=1)
            ax_top.set_ylim(0, 1)

            if m_idx == 0:
                ax_top.set_title(method.upper())

            if c == 0:
                ax_top.set_ylabel(f"{model}\nAccuracy")

            ax_top.text(0.05, 0.9, f"ECE={ece:.3f}", transform=ax_top.transAxes, fontsize=9)
            ax_bot.bar(bc, frac, width=bw * 0.9, color="#4C72B0", alpha=0.85)
            if c == 0:
                ax_bot.set_ylabel("Frac")
            ax_bot.set_xlabel("Confidence")
            ax_bot.set_ylim(0, frac.max() * 1.1)

    plt.tight_layout()
    save(fig, "reliability_grid")

def plot_uncertainty_metrics_grid(results):
    models = sorted(set(v["model"] for v in results.values()))
    methods = sorted(set(v["method"] for v in results.values()))
    metrics = ["entropy", "variance", "MI"]

    for model in models:
        model_results = {k: v for k, v in results.items() if v["model"] == model}
        n_rows, n_cols = len(methods), len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))

        if n_rows == 1:
            axes = axes[None, :]
        if n_cols == 1:
            axes = axes[:, None]

        for c, metric in enumerate(metrics):
            axes[0, c].set_title(metric.upper())   

        for r, method in enumerate(methods):
            for c, metric in enumerate(metrics):
                ax = axes[r, c]
                key = next((k for k, v in model_results.items() if v["method"] == method), None)

                if key is None:
                    ax.axis("off")
                    continue

                data = model_results[key]
                labels, preds = data["labels"],data["preds"]
                correct_mask = (preds == labels)

                if metric == "entropy":
                    values = data["entropy"]

                elif metric == "variance":
                    if data["vars"] is None:
                        ax.axis("off")
                        continue
                    values = data["vars"].mean(dim=1) 

                elif metric == "MI":
                    values = data["MI_dropout"] if data["MI_dropout"] is not None else data["MI_ensemble"]
                    if values is None:
                        ax.axis("off")
                        continue

                values = values.cpu()
                correct_vals = values[correct_mask].numpy()
                incorrect_vals = values[~correct_mask].numpy()

                ax.boxplot([correct_vals, incorrect_vals], labels=["Correct", "Incorrect"], showfliers=False)
                if c == 0:
                    ax.set_ylabel(method)

        fig.suptitle(f"Uncertainty vs Error — {model}", fontsize=14)
        plt.tight_layout()
        save(fig, f"uncertainty_box_{model}")

def plot_entropy_vs_variance_scatter(results):
    models = sorted(set(v["model"] for v in results.values()))
    methods = ["mcdo", "ensemble", "ttda"]

    n_rows, n_cols = len(models), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, model in enumerate(models):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            key = next((k for k, v in results.items() if v["model"] == model and v["method"] == method), None)
            
            if key is None:
                ax.axis("off")
                continue

            data = results[key]
            entropy, var, labels, preds = data["entropy"], data["vars"], data["labels"], data["preds"]

            if var is None:
                ax.axis("off")
                continue

            var = var.mean(dim=1)          
            correct = (preds == labels)

            entropy = entropy.cpu().numpy()
            var = var.cpu().numpy()
            correct = correct.cpu().numpy()

            ax.scatter(entropy[correct], var[correct], alpha=0.5, s=10, label="Correct")
            ax.scatter(entropy[~correct], var[~correct], alpha=0.5, s=10, label="Incorrect")
            if r == 0:
                ax.set_title(method.upper())
            if c == 0:
                ax.set_ylabel(f"{model}\nVariance")
            ax.set_xlabel("Entropy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    save(fig, "entropy_vs_variance_scatter")