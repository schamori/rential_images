"""
XAI / Interpretability analysis for the ConvNextV2 MMAC model.
Each method produces a 3×5 grid: rows = [Input, Attribution, Overlay], cols = grades 0-4.
Usage:
    python explain.py --checkpoint weights/exp1_weighted_loss.pt
"""
from email.mime import base
import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt, matplotlib.cm as mcm
from PIL import Image
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from augmentations import get_val_transform
from dataset import TRAIN_IMG, LABELS_CSV
import pandas as pd

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
CLASS_NAMES = ["No pathology", "Tessellated", "Diffuse CRA", "Patchy CRA", "Macular atrophy"]
HERE        = os.path.dirname(os.path.abspath(__file__))
XAI_DIR     = os.path.join(HERE, "xai_plots")

plt.rcParams.update({"axes.facecolor": "#1a1a2e", "figure.facecolor": "#16213e",
                      "text.color": "white", "axes.titlecolor": "white"})


# ── Model & data ───────────────────────────────────────────────────────────────
class LogitModel(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).logits

def load_model(ckpt):
    base = ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-tiny-1k-224", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    base.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return LogitModel(base).eval().to(DEVICE)

def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    tensor = get_val_transform()(img).unsqueeze(0).to(DEVICE)
    return tensor, np.array(img) / 255.0

def grad_inp(t): return t.clone().requires_grad_(True)


# ── Visualisation helpers ──────────────────────────────────────────────────────
def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-8)

def norm_pct(a, pct=99):
    """Percentile clip then normalise — prevents outliers washing out structure."""
    lo, hi = np.percentile(a, 100 - pct), np.percentile(a, pct)
    return norm(np.clip(a, lo, hi))

def to_heat(attr, signed=False):
    a = attr.squeeze().cpu().detach().float().numpy()
    if a.ndim == 3: a = a.mean(0) if signed else np.abs(a).max(0)
    return a

def blend(img, h, alpha=0.75, cmap="inferno"):
    """Overlay always uses abs + percentile-clipped heatmap for maximum contrast."""
    return np.clip(alpha * mcm.get_cmap(cmap)(norm_pct(np.abs(h)))[..., :3] + (1 - alpha) * img, 0, 1)

def save(fig, name):
    os.makedirs(XAI_DIR, exist_ok=True)
    path = os.path.join(XAI_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  {path}")

def per_class_grid(title, fname, method_fn, data, signed=False, cmap="inferno"):
    """3 rows × 5 cols: [Input | Attribution | Overlay] × [Grade 0..4]."""
    fig, axes = plt.subplots(3, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold", color="white")
    for ax, rl in zip(axes[:, 0], ["Input", "Attribution", "Overlay"]):
        ax.set_ylabel(rl, fontsize=11, color="white", labelpad=6)

    for col, (tensor, img_np, lbl) in enumerate(data):
        h = method_fn(tensor, img_np, lbl)
        axes[0, col].set_title(f"Grade {lbl}\n{CLASS_NAMES[lbl]}", fontsize=9, color="white")
        axes[0, col].imshow(img_np)
        if signed:
            lim = np.percentile(np.abs(h), 99)   # clip top 1% so structure is visible
            axes[1, col].imshow(h, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            axes[1, col].imshow(norm_pct(h), cmap=cmap)
        axes[2, col].imshow(blend(img_np, h, cmap=cmap))

    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); save(fig, fname)


# ── 1. Internal representations ───────────────────────────────────────────────
def plot_activation_maps(model, data):
    """4 feature-map channels × 5 classes grid."""
    n_ch = 4
    fig, axes = plt.subplots(n_ch, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4 * n_ch))
    fig.suptitle("Feature Activation Maps  (last dwconv, 4 channels × 5 grades)",
                 fontsize=11, fontweight="bold", color="white")

    for col, (tensor, _, lbl) in enumerate(data):
        acts = {}
        hook = model.m.convnextv2.encoder.stages[-1].layers[-1].dwconv.register_forward_hook(
            lambda *a: acts.update({"f": a[2].detach()}))
        with torch.no_grad(): model(tensor)
        hook.remove()
        maps = acts["f"][0].cpu().numpy()  # [768, 7, 7]
        axes[0, col].set_title(f"Grade {lbl}\n{CLASS_NAMES[lbl]}", fontsize=9, color="white")
        for row in range(n_ch):
            axes[row, col].imshow(maps[row * 10], cmap="viridis")
            if col == 0: axes[row, col].set_ylabel(f"Ch {row*10}", fontsize=8, color="white")

    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); save(fig, "1a_activation_maps")


def plot_weight_maps(model):
    """Stem patch-embedding kernels — model weights, not input-dependent."""
    w = model.m.convnextv2.embeddings.patch_embeddings.weight.cpu().detach().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Stem Conv Kernel Weights  (first 16 of 96 filters, RGB)",
                 fontsize=11, fontweight="bold", color="white")
    for i, ax in enumerate(axes.flat):
        if i < 16:
            k = w[i].transpose(1, 2, 0)
            ax.imshow(norm(k)); ax.set_title(f"Filter {i}", fontsize=8, color="white")
        ax.axis("off")
    plt.tight_layout(); save(fig, "1b_weight_maps")


# ── 2. Gradient-based methods ─────────────────────────────────────────────────
def plot_gradient_methods(model, data):
    from captum.attr import Saliency, GuidedBackprop, IntegratedGradients, NoiseTunnel

    methods = [
        ("Vanilla Saliency",     "2a_saliency",
         lambda t, i, l: to_heat(Saliency(model).attribute(grad_inp(t), target=l)),
         False, "inferno"),
        ("Guided Backprop",      "2b_guided_backprop",
         lambda t, i, l: to_heat(GuidedBackprop(model).attribute(grad_inp(t), target=l), signed=True),
         True, "RdBu_r"),
        ("Integrated Gradients", "2c_integrated_gradients",
         lambda t, i, l: to_heat(IntegratedGradients(model).attribute(
             grad_inp(t), torch.zeros_like(t), target=l, n_steps=50), signed=True),
         True, "RdBu_r"),
        ("SmoothGrad",           "2d_smoothgrad",
         lambda t, i, l: to_heat(NoiseTunnel(Saliency(model)).attribute(
             grad_inp(t), target=l, nt_samples=20, stdevs=0.1, nt_type="smoothgrad")),
         False, "inferno"),
    ]
    for title, fname, fn, signed, cmap in methods:
        per_class_grid(title, fname, fn, data, signed=signed, cmap=cmap)


# ── 3. CAM methods ────────────────────────────────────────────────────────────
def plot_cam_methods(model, data):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    tl = [model.m.convnextv2.encoder.stages[-1].layers[-1].dwconv]

    for Cls, name, fname in [(GradCAM, "GradCAM", "3a_gradcam"),
                              (GradCAMPlusPlus, "GradCAM++", "3b_gradcam_pp")]:
        def fn(t, i, l, Cls=Cls):
            with Cls(model=model, target_layers=tl) as cam:
                return cam(input_tensor=t, targets=[ClassifierOutputTarget(l)])[0]
        per_class_grid(name, fname, fn, data, signed=False, cmap="jet")


# ── 4. Relevance propagation ──────────────────────────────────────────────────
def _lrp(model, tensor, label):
    x = grad_inp(tensor); model(x)[0, label].backward()
    return to_heat(x * x.grad, signed=True)

def plot_relevance(model, data):
    from captum.attr import DeepLift, GradientShap

    methods = [
        ("DeepLIFT",     "4a_deeplift",
         lambda t, i, l: to_heat(DeepLift(model).attribute(grad_inp(t), torch.zeros_like(t), target=l), signed=True),
         True, "RdBu_r"),
        ("GradientSHAP", "4b_gradient_shap",
         lambda t, i, l: to_heat(GradientShap(model).attribute(grad_inp(t), torch.zeros_like(t), target=l, n_samples=30), signed=True),
         True, "RdBu_r"),
        ("LRP (ε-rule)", "4c_lrp",
         lambda t, i, l: _lrp(model, t, l),
         True, "RdBu_r"),
    ]
    for title, fname, fn, signed, cmap in methods:
        per_class_grid(title, fname, fn, data, signed=signed, cmap=cmap)


# ── 5. Model-agnostic ─────────────────────────────────────────────────────────
def plot_agnostic(model, data):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    def predict_fn(imgs):
        ts = torch.stack([get_val_transform()(Image.fromarray(i)) for i in imgs]).to(DEVICE)
        with torch.no_grad(): return torch.softmax(model(ts), 1).cpu().numpy()

    # LIME uses mark_boundaries so handled separately (1 row × 5 cols)
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 5))
    fig.suptitle("LIME  (positive superpixels highlighted)", fontsize=13, fontweight="bold", color="white")
    for col, (tensor, img_np, lbl) in enumerate(data):
        exp = lime_image.LimeImageExplainer(random_state=42).explain_instance(
            (img_np * 255).astype(np.uint8), predict_fn, top_labels=1, num_samples=300)
        temp, mask = exp.get_image_and_mask(lbl, positive_only=True, num_features=8, hide_rest=False)
        axes[col].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 0.8, 0)))
        axes[col].set_title(f"Grade {lbl}\n{CLASS_NAMES[lbl]}", fontsize=9, color="white")
        axes[col].axis("off")
    plt.tight_layout(); save(fig, "5a_lime")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    df        = pd.read_csv(LABELS_CSV)
    grade_col = "myopic_maculopathy_grade"
    model     = load_model(args.checkpoint)

    print("Loading one image per class:")
    data = []
    for cls in range(NUM_CLASSES):
        path = os.path.join(TRAIN_IMG, df[df[grade_col] == cls].iloc[0]["image"])
        t, i = load_image(path)
        with torch.no_grad(): pred = model(t).argmax(1).item()
        print(f"  Grade {cls} ({CLASS_NAMES[cls]}): pred={pred}  [{os.path.basename(path)}]")
        data.append((t, i, cls))

    print("\nGenerating XAI plots →")
    plot_activation_maps(model, data)
    plot_weight_maps(model)
    plot_gradient_methods(model, data)
    plot_cam_methods(model, data)
    plot_relevance(model, data)
    plot_agnostic(model, data)
    print(f"\nDone — all plots saved to {XAI_DIR}/")


if __name__ == "__main__":
    main()
