"""
Statistical comparison of trained methods on a fixed test set.

This script:
1) Runs each method checkpoint on the same test set.
2) Saves per-sample predictions/errors for each method.
3) Runs paired permutation tests between methods.
4) Aggregates method ranks across imbalance-aware metrics.

Usage example:
  python statistical_comparison.py \
    --configs configs \
    --weights weights \
    --output_dir plots/stat_tests \
    --n_permutations 2000 \
    --alpha 0.01 \
    --method1 exp1_weighted_loss \
    --method2 exp2_oversampling
"""

import argparse
import glob
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from transformers import ConvNextV2ForImageClassification

from dataset import TEST_LABELS_CSV, get_loaders
from multitask_model import MultiTaskConvNeXt


NUM_CLASSES = 5
RANK_METRICS = ["kappa", "fbeta_macro", "pr_auc_macro"]
PAIRWISE_METRICS = ["f1_macro", "kappa", "fbeta_macro", "pr_auc_macro"]


@dataclass
class MethodSpec:
    name: str
    cfg_path: str
    cfg: Dict
    is_mtl: bool
    is_ensemble: bool
    checkpoints: List[str]


def load_cfg(path: str) -> Dict:
    """Load yaml config with base override from the same folder."""
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def build_single_task_model(cfg: Dict, device: str) -> nn.Module:
    model = ConvNextV2ForImageClassification.from_pretrained(
        cfg["model_id"],
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    hidden_dim = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(hidden_dim, NUM_CLASSES),
    ).to(device)
    return model


def checkpoint_paths_for_method(name: str, cfg: Dict, weights_dir: str) -> List[str]:
    is_ensemble = cfg.get("ensemble", False)
    if not is_ensemble:
        return [os.path.join(weights_dir, f"{name}.pt")]

    n_models = cfg.get("ensemble_n", 1)
    seeds = cfg.get("ensemble_seeds", [42] * n_models)[:n_models]
    return [os.path.join(weights_dir, f"{name}_seed{seed}.pt") for seed in seeds]


def discover_methods(configs_dir: str, weights_dir: str, selected: List[str]) -> List[MethodSpec]:
    """Find all config methods that have the full expected checkpoint set."""
    selected_set = set(selected) if selected else None

    cfg_paths = sorted(glob.glob(os.path.join(configs_dir, "**", "*.yaml"), recursive=True))
    cfg_paths = [p for p in cfg_paths if os.path.basename(p) != "base.yaml"]

    methods: List[MethodSpec] = []
    for cfg_path in cfg_paths:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        if selected_set and name not in selected_set:
            continue

        cfg = load_cfg(cfg_path)
        checkpoints = checkpoint_paths_for_method(name, cfg, weights_dir)
        if not all(os.path.exists(p) for p in checkpoints):
            continue

        methods.append(
            MethodSpec(
                name=name,
                cfg_path=cfg_path,
                cfg=cfg,
                is_mtl=cfg.get("multitask", False),
                is_ensemble=cfg.get("ensemble", False),
                checkpoints=checkpoints,
            )
        )

    if selected_set:
        by_name = {m.name: m for m in methods}
        ordered = [by_name[n] for n in selected if n in by_name]
        missing = [n for n in selected if n not in by_name]
        if missing:
            print("[warn] Requested methods missing config/weights:")
            for n in missing:
                print(f"  - {n}")
        methods = ordered

    return methods


def load_models_for_method(spec: MethodSpec, device: str) -> List[nn.Module]:
    models: List[nn.Module] = []
    for ckpt_path in spec.checkpoints:
        if spec.is_mtl:
            model = MultiTaskConvNeXt(spec.cfg).to(device)
        else:
            model = build_single_task_model(spec.cfg, device=device)

        state = torch.load(ckpt_path, map_location=device)
        for old_key, new_key in [
            ("classifier.weight", "classifier.1.weight"),
            ("classifier.bias", "classifier.1.bias"),
        ]:
            if old_key in state and new_key not in state:
                state[new_key] = state.pop(old_key)

        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def predict_method(spec: MethodSpec, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, predictions, and probabilities for a method on the test set."""
    _, _, test_loader, _ = get_loaders(spec.cfg, return_test=True)
    models = load_models_for_method(spec, device)

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch in test_loader:
        if spec.is_mtl:
            imgs, labels, *_ = batch
        else:
            imgs, labels = batch

        imgs = imgs.to(device, non_blocking=True)

        logits_sum = None
        for model in models:
            if spec.is_mtl:
                logits, _, _ = model(imgs)
            else:
                logits = model(imgs).logits

            logits_sum = logits if logits_sum is None else logits_sum + logits

        avg_logits = logits_sum / len(models)
        probs = torch.softmax(avg_logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    prob = np.concatenate(all_probs, axis=0).astype(np.float64)
    return y_true, y_pred, prob


def macro_pr_auc(y_true: np.ndarray, prob: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Macro PR AUC with safe handling for classes absent in the test split."""
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    ap_values: List[float] = []
    for c in range(num_classes):
        positives = int(y_bin[:, c].sum())
        negatives = int(len(y_true) - positives)
        if positives == 0 or negatives == 0:
            continue
        ap_values.append(float(average_precision_score(y_bin[:, c], prob[:, c])))

    if not ap_values:
        return float("nan")
    return float(np.mean(ap_values))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, beta: float) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "fbeta_macro": float(
            fbeta_score(y_true, y_pred, beta=beta, average="macro", zero_division=0)
        ),
        "pr_auc_macro": macro_pr_auc(y_true, prob),
    }


def metric_functions(beta: float) -> Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]]:
    return {
        "f1_macro": lambda y, p, pr: float(f1_score(y, p, average="macro", zero_division=0)),
        "kappa": lambda y, p, pr: float(cohen_kappa_score(y, p, weights="quadratic")),
        "fbeta_macro": lambda y, p, pr: float(
            fbeta_score(y, p, beta=beta, average="macro", zero_division=0)
        ),
        "pr_auc_macro": lambda y, p, pr: macro_pr_auc(y, pr),
    }


def stable_uint_seed(seed: int, metric: str, m1: str, m2: str) -> int:
    key = f"{seed}|{metric}|{m1}|{m2}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def paired_permutation_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    prob_a: np.ndarray,
    pred_b: np.ndarray,
    prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_permutations: int,
    seed: int,
) -> Dict[str, float]:
    """Paired permutation test by swapping method tags sample-wise."""
    observed_diff = metric_fn(y_true, pred_a, prob_a) - metric_fn(y_true, pred_b, prob_b)

    n = len(y_true)
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        swap = rng.integers(0, 2, size=n, dtype=np.int8).astype(bool)

        perm_pred_a = np.where(swap, pred_b, pred_a)
        perm_pred_b = np.where(swap, pred_a, pred_b)
        perm_prob_a = np.where(swap[:, None], prob_b, prob_a)
        perm_prob_b = np.where(swap[:, None], prob_a, prob_b)

        perm_diffs[i] = metric_fn(y_true, perm_pred_a, perm_prob_a) - metric_fn(
            y_true, perm_pred_b, perm_prob_b
        )

    p_right = float((np.sum(perm_diffs >= observed_diff) + 1) / (n_permutations + 1))
    p_left = float((np.sum(perm_diffs <= observed_diff) + 1) / (n_permutations + 1))
    p_two_sided = float(
        (np.sum(np.abs(perm_diffs) >= abs(observed_diff)) + 1) / (n_permutations + 1)
    )

    return {
        "observed_diff": float(observed_diff),
        "perm_mean": float(np.mean(perm_diffs)),
        "perm_std": float(np.std(perm_diffs)),
        "p_value_method1_gt_method2": p_right,
        "p_value_method2_gt_method1": p_left,
        "p_value_two_sided": p_two_sided,
    }


def build_rankings(
    method_metrics: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    rank_metrics: List[str],
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_values = method_metrics.set_index("method")
    methods = method_metrics["method"].tolist()

    rows: List[Dict] = []
    for metric in rank_metrics:
        sub = pairwise_df[pairwise_df["metric"] == metric]
        wins = {m: 0 for m in methods}

        for _, r in sub.iterrows():
            if bool(r["method1_better_at_alpha"]):
                wins[r["method1"]] += 1
            if bool(r["method2_better_at_alpha"]):
                wins[r["method2"]] += 1

        ordered = sorted(
            methods,
            key=lambda m: (-wins[m], -metric_values.loc[m, metric], m),
        )

        for rank, method in enumerate(ordered, start=1):
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "significant_wins": wins[method],
                    "metric_value": float(metric_values.loc[method, metric]),
                    "rank": rank,
                    "alpha": alpha,
                }
            )

    rank_df = pd.DataFrame(rows).sort_values(["metric", "rank", "method"]).reset_index(drop=True)

    agg = (
        rank_df.groupby("method", as_index=False)
        .agg(
            aggregated_rank_score=("rank", "sum"),
            mean_rank=("rank", "mean"),
            total_significant_wins=("significant_wins", "sum"),
        )
        .sort_values(["aggregated_rank_score", "mean_rank", "method"])
        .reset_index(drop=True)
    )
    agg["overall_rank"] = np.arange(1, len(agg) + 1)
    return rank_df, agg


def export_per_sample(
    out_dir: str,
    method_name: str,
    image_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
) -> None:
    n = len(y_true)
    idx = np.arange(n)

    df = pd.DataFrame(
        {
            "image": image_names,
            "true_label": y_true,
            "pred_label": y_pred,
            "correct": (y_true == y_pred).astype(int),
            "error": (y_true != y_pred).astype(int),
            "pred_confidence": prob.max(axis=1),
            "true_class_prob": prob[idx, y_true],
        }
    )

    for c in range(prob.shape[1]):
        df[f"prob_class_{c}"] = prob[:, c]

    out_path = os.path.join(out_dir, f"{method_name}_per_sample.csv")
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs", help="Config folder")
    parser.add_argument("--weights", default="weights", help="Weights folder")
    parser.add_argument(
        "--output_dir",
        default="plots/stat_tests",
        help="Where to save CSV/JSON outputs",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional explicit list of method names (config stem names)",
    )
    parser.add_argument("--method1", default=None, help="Primary method for focused comparison")
    parser.add_argument("--method2", default=None, help="Second method for focused comparison")
    parser.add_argument("--n_permutations", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = (
        "cuda:1"
        if torch.cuda.device_count() > 1
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    os.makedirs(args.output_dir, exist_ok=True)
    per_sample_dir = os.path.join(args.output_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    methods = discover_methods(args.configs, args.weights, args.methods)
    if len(methods) < 2:
        raise RuntimeError("Need at least 2 methods with available checkpoints.")

    print(f"Found {len(methods)} methods with available checkpoints:")
    for m in methods:
        kind = "MTL" if m.is_mtl else "single-task"
        ens = f"ensemble({len(m.checkpoints)})" if m.is_ensemble else "single"
        print(f"  - {m.name:28s} [{kind}, {ens}]")

    test_df = pd.read_csv(TEST_LABELS_CSV)
    image_names = test_df["image"].astype(str).tolist()
    labels_csv = test_df["myopic_maculopathy_grade"].astype(int).to_numpy()

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    metrics_rows: List[Dict] = []

    print("\nRunning inference for each method on fixed test set...")
    for spec in tqdm(methods, desc="Methods", unit="method"):
        y_true, y_pred, prob = predict_method(spec, device=device)

        if len(y_true) != len(labels_csv):
            raise RuntimeError(f"Label length mismatch for {spec.name}: {len(y_true)} vs {len(labels_csv)}")
        if not np.array_equal(y_true, labels_csv):
            raise RuntimeError(f"Label order mismatch detected for method {spec.name}.")

        predictions[spec.name] = {"y_true": y_true, "y_pred": y_pred, "prob": prob}
        export_per_sample(per_sample_dir, spec.name, image_names, y_true, y_pred, prob)

        m = compute_metrics(y_true, y_pred, prob, beta=args.beta)
        m["method"] = spec.name
        metrics_rows.append(m)

    method_metrics = pd.DataFrame(metrics_rows).sort_values("method").reset_index(drop=True)
    method_metrics.to_csv(os.path.join(args.output_dir, "method_metrics.csv"), index=False)

    metric_fns = metric_functions(beta=args.beta)

    print("\nRunning paired permutation tests...")
    pair_rows: List[Dict] = []
    pairs = list(itertools.combinations([m.name for m in methods], 2))

    for metric in PAIRWISE_METRICS:
        fn = metric_fns[metric]
        for m1, m2 in tqdm(pairs, desc=f"Metric={metric}", unit="pair"):
            d1 = predictions[m1]
            d2 = predictions[m2]

            out = paired_permutation_test(
                y_true=d1["y_true"],
                pred_a=d1["y_pred"],
                prob_a=d1["prob"],
                pred_b=d2["y_pred"],
                prob_b=d2["prob"],
                metric_fn=fn,
                n_permutations=args.n_permutations,
                seed=stable_uint_seed(args.seed, metric, m1, m2),
            )

            pair_rows.append(
                {
                    "metric": metric,
                    "method1": m1,
                    "method2": m2,
                    **out,
                    "alpha": args.alpha,
                    "method1_better_at_alpha": bool(
                        out["observed_diff"] > 0
                        and out["p_value_method1_gt_method2"] < args.alpha
                    ),
                    "method2_better_at_alpha": bool(
                        out["observed_diff"] < 0
                        and out["p_value_method2_gt_method1"] < args.alpha
                    ),
                }
            )

    pairwise_df = pd.DataFrame(pair_rows).sort_values(["metric", "method1", "method2"]).reset_index(drop=True)
    pairwise_df.to_csv(os.path.join(args.output_dir, "pairwise_permutation_results.csv"), index=False)

    rank_df, aggregated_df = build_rankings(
        method_metrics=method_metrics,
        pairwise_df=pairwise_df,
        rank_metrics=RANK_METRICS,
        alpha=args.alpha,
    )
    rank_df.to_csv(os.path.join(args.output_dir, "metric_rankings.csv"), index=False)
    aggregated_df.to_csv(os.path.join(args.output_dir, "aggregated_ranking.csv"), index=False)

    if args.method1 is not None and args.method2 is not None:
        if args.method1 not in predictions or args.method2 not in predictions:
            raise RuntimeError("--method1/--method2 not found in discovered methods.")
        focus_m1 = args.method1
        focus_m2 = args.method2
    else:
        focus_m1, focus_m2 = methods[0].name, methods[1].name

    focus_rows = []
    for metric in PAIRWISE_METRICS:
        fn = metric_fns[metric]
        d1 = predictions[focus_m1]
        d2 = predictions[focus_m2]
        out = paired_permutation_test(
            y_true=d1["y_true"],
            pred_a=d1["y_pred"],
            prob_a=d1["prob"],
            pred_b=d2["y_pred"],
            prob_b=d2["prob"],
            metric_fn=fn,
            n_permutations=args.n_permutations,
            seed=stable_uint_seed(args.seed + 17, metric, focus_m1, focus_m2),
        )
        focus_rows.append(
            {
                "metric": metric,
                "method1": focus_m1,
                "method2": focus_m2,
                **out,
                "alpha": args.alpha,
                "method1_better_at_alpha": bool(
                    out["observed_diff"] > 0
                    and out["p_value_method1_gt_method2"] < args.alpha
                ),
                "method2_better_at_alpha": bool(
                    out["observed_diff"] < 0
                    and out["p_value_method2_gt_method1"] < args.alpha
                ),
            }
        )

    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(os.path.join(args.output_dir, "method1_vs_method2.csv"), index=False)
    with open(os.path.join(args.output_dir, "method1_vs_method2.json"), "w", encoding="utf-8") as f:
        json.dump(focus_rows, f, indent=2)

    summary = {
        "n_methods": len(methods),
        "n_permutations": args.n_permutations,
        "alpha": args.alpha,
        "beta": args.beta,
        "rank_metrics": RANK_METRICS,
        "pairwise_metrics": PAIRWISE_METRICS,
        "focused_pair": [focus_m1, focus_m2],
        "winner": aggregated_df.iloc[0]["method"],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Outputs saved under: {args.output_dir}")
    print(f"Overall winner (aggregated rank): {aggregated_df.iloc[0]['method']}")
    print(f"Focused pair: {focus_m1} vs {focus_m2}")


if __name__ == "__main__":
    main()