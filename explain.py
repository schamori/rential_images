"""
XAI / Interpretability analysis for the ConvNextV2 MMAC model.
Usage:
    python explain.py --checkpoint base.pt
    python explain.py --checkpoint base.pt --image path/to/img.png --label 2
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt, matplotlib.cm as mcm
from PIL import Image
from transformers import ConvNextV2ForImageClassification
from augmentations import get_val_transform, IMAGENET_MEAN, IMAGENET_STD
from dataset import TRAIN_IMG, LABELS_CSV
import pandas as pd

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
CLASS_NAMES = ["No pathology", "Tessellated", "Diffuse CRA", "Patchy CRA", "Macular atrophy"]

plt.rcParams.update({"axes.facecolor": "#1a1a2e", "figure.facecolor": "#16213e",
                      "text.color": "white", "axes.titlecolor": "white"})


# ── Model & data ───────────────────────────────────────────────────────────────
class LogitModel(nn.Module):
    """Thin wrapper so HF ModelOutput → plain logit tensor (needed by captum/grad-cam)."""
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

def to_heat(attr, signed=False):
    a = attr.squeeze().cpu().detach().float().numpy()
    if a.ndim == 3: a = a.mean(0) if signed else np.abs(a).max(0)
    return a

def blend(img, h, alpha=0.55, cmap="inferno"):
    return np.clip(alpha * mcm.get_cmap(cmap)(norm(h))[..., :3] + (1 - alpha) * img, 0, 1)

def save(fig, name):
    os.makedirs("xai_plots", exist_ok=True)
    fig.savefig(f"xai_plots/{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  xai_plots/{name}.png")

def method_row(axes, title, img, h, signed=False, cmap="inferno"):
    """One row: original | attribution | overlay."""
    for ax in axes: ax.axis("off")
    axes[0].imshow(img);                 axes[0].set_title("Input",    fontsize=9)
    if signed:
        lim = np.abs(h).max()
        im = axes[1].imshow(h, cmap="RdBu_r", vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)
    else:
        axes[1].imshow(norm(h), cmap=cmap)
    axes[1].set_title(title,     fontsize=9)
    axes[2].imshow(blend(img, h, cmap=cmap)); axes[2].set_title("Overlay", fontsize=9)


# ── 1. Internal representations ───────────────────────────────────────────────
def plot_activation_maps(model, tensor, img_np):
    """Feature maps from last depthwise conv (7×7 spatial, 768 channels)."""
    acts = {}
    h = model.m.convnextv2.encoder.stages[-1].layers[-1].dwconv.register_forward_hook(
        lambda *a: acts.update({"f": a[2].detach()}))
    with torch.no_grad(): model(tensor)
    h.remove()

    maps = acts["f"][0].cpu().numpy()   # [768, 7, 7]
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Feature Activation Maps  (last dwconv — 16 of 768 channels)",
                 fontsize=11, fontweight="bold", color="white")
    for i, ax in enumerate(axes.flat):
        if i < 16: ax.imshow(maps[i * 5], cmap="viridis"); ax.set_title(f"Ch {i*5}", fontsize=8, color="white")
        ax.axis("off")
    plt.tight_layout(); save(fig, "1a_activation_maps")


def plot_weight_maps(model):
    """Stem patch-embedding kernels [96 × 3 × 4 × 4] → RGB thumbnails."""
    w = model.m.convnextv2.embeddings.patch_embeddings.weight.cpu().detach().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Stem Conv Kernel Weights  (first 16 of 96 filters, RGB)",
                 fontsize=11, fontweight="bold", color="white")
    for i, ax in enumerate(axes.flat):
        if i < 16:
            k = w[i].transpose(1, 2, 0)   # [4,4,3]
            ax.imshow(norm(k)); ax.set_title(f"Filter {i}", fontsize=8, color="white")
        ax.axis("off")
    plt.tight_layout(); save(fig, "1b_weight_maps")


# ── 2. Gradient-based methods ─────────────────────────────────────────────────
def plot_gradient_methods(model, tensor, img_np, label):
    from captum.attr import Saliency, GuidedBackprop, IntegratedGradients, NoiseTunnel

    x   = grad_inp(tensor)
    bl  = torch.zeros_like(tensor)

    attrs = [
        ("Vanilla Saliency",
         to_heat(Saliency(model).attribute(x, target=label)), False),
        ("Guided Backprop",
         to_heat(GuidedBackprop(model).attribute(x, target=label), signed=True), True),
        ("Integrated Gradients",
         to_heat(IntegratedGradients(model).attribute(x, bl, target=label, n_steps=50), signed=True), True),
        ("SmoothGrad",
         to_heat(NoiseTunnel(Saliency(model)).attribute(
             x, target=label, nt_samples=20, stdevs=0.1, nt_type="smoothgrad")), False),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(11, 14))
    fig.suptitle("Gradient-Based Backpropagation Methods", fontsize=13, fontweight="bold", color="white")
    for i, (name, h, signed) in enumerate(attrs):
        method_row(axes[i], name, img_np, h, signed=signed)
    plt.tight_layout(); save(fig, "2_gradient_methods")


# ── 3. CAM methods ────────────────────────────────────────────────────────────
def plot_cam_methods(model, tensor, img_np, label):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # last depthwise conv — 7×7 spatial feature map ideal for localization
    target_layers = [model.m.convnextv2.encoder.stages[-1].layers[-1].dwconv]
    targets       = [ClassifierOutputTarget(label)]

    cams = []
    for Cls, name in [(GradCAM, "GradCAM"), (GradCAMPlusPlus, "GradCAM++")]:
        with Cls(model=model, target_layers=target_layers) as cam:
            cams.append((name, cam(input_tensor=tensor, targets=targets)[0]))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle("Class Activation Mapping", fontsize=13, fontweight="bold", color="white")
    for i, (name, h) in enumerate(cams):
        method_row(axes[i], name, img_np, h, cmap="jet")
    plt.tight_layout(); save(fig, "3_cam_methods")


# ── 4. Relevance propagation ──────────────────────────────────────────────────
def plot_relevance(model, tensor, img_np, label):
    from captum.attr import DeepLift, GradientShap

    bl = torch.zeros_like(tensor)
    attrs = [
        ("DeepLIFT",
         to_heat(DeepLift(model).attribute(grad_inp(tensor), bl, target=label), signed=True), True),
        ("GradientSHAP",
         to_heat(GradientShap(model).attribute(grad_inp(tensor), bl, target=label, n_samples=30), signed=True), True),
    ]

    # LRP (epsilon rule) — R = gradient × input, exact for linear layers,
    # principled approximation for non-linear layers (GRN, LayerNorm in ConvNextV2)
    x = grad_inp(tensor); model(x)[0, label].backward()
    attrs.append(("LRP (ε-rule)", to_heat(x * x.grad, signed=True), True))

    n = len(attrs)
    fig, axes = plt.subplots(n, 3, figsize=(11, 4.5 * n))
    if n == 1: axes = axes[None]
    fig.suptitle("Relevance Propagation", fontsize=13, fontweight="bold", color="white")
    for i, (name, h, signed) in enumerate(attrs):
        method_row(axes[i], name, img_np, h, signed=signed, cmap="RdBu_r")
    plt.tight_layout(); save(fig, "4_relevance_propagation")


# ── 5. Model-agnostic ─────────────────────────────────────────────────────────
def plot_agnostic(model, tensor, img_np, label):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from captum.attr import GradientShap

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model-Agnostic Methods", fontsize=13, fontweight="bold", color="white")
    for ax in axes: ax.axis("off")

    # LIME — perturbs superpixels and fits a local linear model
    def predict_fn(imgs):
        ts = torch.stack([get_val_transform()(Image.fromarray(i)) for i in imgs]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(model(ts), 1).cpu().numpy()

    exp = lime_image.LimeImageExplainer(random_state=42).explain_instance(
        (img_np * 255).astype(np.uint8), predict_fn, top_labels=1, num_samples=500)
    temp, mask = exp.get_image_and_mask(label, positive_only=True, num_features=8, hide_rest=False)
    axes[0].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 0.8, 0)))
    axes[0].set_title("LIME  (positive superpixels highlighted)", fontsize=10, color="white")

    # GradientSHAP — Shapley values via gradient × noise baseline
    bl = torch.zeros_like(tensor)
    shap_h = to_heat(GradientShap(model).attribute(
        grad_inp(tensor), bl, target=label, n_samples=50), signed=True)
    lim = np.abs(shap_h).max()
    im = axes[1].imshow(shap_h, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)
    axes[1].set_title("GradientSHAP  (red=supports, blue=contradicts)", fontsize=10, color="white")

    plt.tight_layout(); save(fig, "5_model_agnostic")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image",  default=None, help="Path to a fundus image")
    parser.add_argument("--label",  type=int, default=None, help="Ground-truth class (0-4)")
    args = parser.parse_args()

    if args.image is None:
        df  = pd.read_csv(LABELS_CSV)
        row = df.iloc[0]
        args.image = os.path.join(TRAIN_IMG, row["image"])
        if args.label is None:
            args.label = int(row["myopic_maculopathy_grade"])

    model = load_model(args.checkpoint)
    tensor, img_np = load_image(args.image)

    with torch.no_grad():
        pred = model(tensor).argmax(1).item()
    label = args.label if args.label is not None else pred

    print(f"Image : {os.path.basename(args.image)}")
    print(f"Pred  : C{pred} — {CLASS_NAMES[pred]}")
    print(f"Target: C{label} — {CLASS_NAMES[label]}")
    print("Generating XAI plots →")

    plot_activation_maps(model, tensor, img_np)
    plot_weight_maps(model)
    plot_gradient_methods(model, tensor, img_np, label)
    plot_cam_methods(model, tensor, img_np, label)
    plot_relevance(model, tensor, img_np, label)
    plot_agnostic(model, tensor, img_np, label)

    print("Done — all plots saved to xai_plots/")


if __name__ == "__main__":
    main()
