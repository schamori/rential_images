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
from dataset import TEST_IMG, TEST_LABELS_CSV
import pandas as pd

DEVICE      = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    hidden_dim = base.classifier.in_features
    base.classifier = nn.Sequential( nn.Linear(hidden_dim, NUM_CLASSES))
    state = torch.load(ckpt, map_location=DEVICE)

    # Support checkpoints saved with classifier keys as:
    # classifier.weight / classifier.bias (plain Linear),
    # classifier.0.* (Sequential index 0), or classifier.1.* (Sequential index 1).
    model_keys = set(base.state_dict().keys())
    if "classifier.0.weight" in model_keys:
        exp_w, exp_b = "classifier.0.weight", "classifier.0.bias"
    elif "classifier.1.weight" in model_keys:
        exp_w, exp_b = "classifier.1.weight", "classifier.1.bias"
    else:
        exp_w, exp_b = "classifier.weight", "classifier.bias"

    for src in ("classifier.weight", "classifier.0.weight", "classifier.1.weight"):
        if src in state:
            if src != exp_w and exp_w not in state:
                state[exp_w] = state.pop(src)
            break

    for src in ("classifier.bias", "classifier.0.bias", "classifier.1.bias"):
        if src in state:
            if src != exp_b and exp_b not in state:
                state[exp_b] = state.pop(src)
            break

    base.load_state_dict(state, strict=True)
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
            (img_np * 255).astype(np.uint8), predict_fn, labels=(lbl,), num_samples=300)
        temp, mask = exp.get_image_and_mask(lbl, positive_only=True, num_features=8, hide_rest=False)
        axes[col].imshow(mark_boundaries(temp / 255.0, mask, color=(1, 0.8, 0)))
        axes[col].set_title(f"Grade {lbl}\n{CLASS_NAMES[lbl]}", fontsize=9, color="white")
        axes[col].axis("off")
    plt.tight_layout(); save(fig, "5a_lime")


# ── 6. Robustness / sanity checks ─────────────────────────────────────────────
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _denorm(t):
    m, s = MEAN_T.to(t.device), STD_T.to(t.device)
    return (t * s + m).squeeze(0).permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)

def _gradcam(model, tensor, label):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    tl = [model.m.convnextv2.encoder.stages[-1].layers[-1].dwconv]
    with GradCAM(model=model, target_layers=tl) as cam:
        return cam(input_tensor=tensor, targets=[ClassifierOutputTarget(label)])[0]


def plot_input_robustness(model, tensor, label):
    """GradCAM under superficial input perturbations.
    A faithful XAI method should give STABLE attributions — the class evidence is
    still there regardless of brightness/noise/rotation."""
    import torchvision.transforms.functional as TF

    perturbs = [
        ("Original",          tensor),
        ("+ Gaussian noise",  tensor + 0.15 * torch.randn_like(tensor)),
        ("Brightness +30%",   tensor * 1.3),
        ("Brightness −30%",   tensor * 0.7),
        ("Rotated 10°",       TF.rotate(tensor, 10)),
        ("Blur (σ=2)",        TF.gaussian_blur(tensor, kernel_size=7, sigma=2.0)),
    ]

    n = len(perturbs)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    fig.suptitle("Input Robustness — GradCAM under superficial input changes\n"
                 "(a faithful method should stay stable)",
                 fontsize=12, fontweight="bold", color="white")
    for col, (name, t) in enumerate(perturbs):
        img = _denorm(t)
        h   = _gradcam(model, t, label)
        axes[0, col].imshow(img); axes[0, col].set_title(name, fontsize=10, color="white")
        axes[1, col].imshow(blend(img, h, cmap="jet"))
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); save(fig, "6a_input_robustness")


def plot_model_robustness(ckpt, tensor, img_np, label):
    """Cascading weight randomization (Adebayo et al. 2018 sanity check).
    A faithful XAI method should CHANGE as the model is destroyed top-down.
    If attributions stay similar → method is just an edge-detector, not explaining the model."""
    def randomize(model, patterns):
        for name, p in model.named_parameters():
            if any(pat in name for pat in patterns):
                if p.dim() >= 2: nn.init.kaiming_normal_(p)
                else: nn.init.zeros_(p)
        return model

    stages = [
        ("Trained",                load_model(ckpt)),
        ("Rand classifier",        randomize(load_model(ckpt), ["classifier"])),
        ("… + last stage",         randomize(load_model(ckpt), ["classifier", "stages.3"])),
        ("… + last 2 stages",      randomize(load_model(ckpt), ["classifier", "stages.3", "stages.2"])),
        ("Fully random",           randomize(load_model(ckpt), [""])),
    ]

    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5))
    fig.suptitle("Model Robustness — cascading weight randomization (Adebayo 2018)\n"
                 "(faithful XAI should shift; edge-detector-like methods won't)",
                 fontsize=12, fontweight="bold", color="white")
    for col, (name, m) in enumerate(stages):
        h = _gradcam(m, tensor, label)
        axes[col].imshow(blend(img_np, h, cmap="jet"))
        axes[col].set_title(name, fontsize=10, color="white")
        axes[col].axis("off")
    plt.tight_layout(); save(fig, "6b_model_robustness")


# ── 7. Quantitative robustness benchmark ──────────────────────────────────────
def _rel_change(orig, pert):
    """L2 change normalised against initial magnitude: ||pert - orig|| / ||orig||.
    Scale-free, directly comparable across methods with different dynamic ranges."""
    return float(np.linalg.norm(pert - orig) / (np.linalg.norm(orig) + 1e-8))


def _all_methods():
    """Return {name: fn(model, tensor, label) -> 2D heatmap} for every XAI method."""
    from captum.attr import Saliency, GuidedBackprop, IntegratedGradients, NoiseTunnel, DeepLift, GradientShap
    from lime import lime_image
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    def _cam(Cls, m, t, l):
        tl = [m.m.convnextv2.encoder.stages[-1].layers[-1].dwconv]
        with Cls(model=m, target_layers=tl) as cam:
            return cam(input_tensor=t, targets=[ClassifierOutputTarget(l)])[0]

    def _to_uint8(img):
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return arr
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _predict_for_lime(m, imgs):
        ts = torch.stack([get_val_transform()(Image.fromarray(_to_uint8(i))) for i in imgs]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(m(ts), dim=1).cpu().numpy()

    def _lime_heat(m, t, l):
        # Keep sample count moderate so LIME is feasible inside the full benchmark loop.
        img_uint8 = _to_uint8(_denorm(t))
        exp = lime_image.LimeImageExplainer(random_state=42).explain_instance(
            img_uint8,
            lambda imgs: _predict_for_lime(m, imgs),
            labels=(int(l),),
            num_samples=120,
        )
        heat = np.zeros_like(exp.segments, dtype=np.float32)
        for seg_id, weight in exp.local_exp[int(l)]:
            heat[exp.segments == seg_id] = weight
        return heat

    return {
        "Saliency":    lambda m, t, l: to_heat(Saliency(m).attribute(grad_inp(t), target=l)),
        "Guided BP":   lambda m, t, l: to_heat(GuidedBackprop(m).attribute(grad_inp(t), target=l), signed=True),
        "IntegGrad":   lambda m, t, l: to_heat(IntegratedGradients(m).attribute(grad_inp(t), torch.zeros_like(t), target=l, n_steps=20), signed=True),
        "SmoothGrad":  lambda m, t, l: to_heat(NoiseTunnel(Saliency(m)).attribute(grad_inp(t), target=l, nt_samples=10, stdevs=0.1, nt_type="smoothgrad")),
        "GradCAM":     lambda m, t, l: _cam(GradCAM, m, t, l),
        "GradCAM++":   lambda m, t, l: _cam(GradCAMPlusPlus, m, t, l),
        "DeepLIFT":    lambda m, t, l: to_heat(DeepLift(m).attribute(grad_inp(t), torch.zeros_like(t), target=l), signed=True),
        "GradShap":    lambda m, t, l: to_heat(GradientShap(m).attribute(grad_inp(t), torch.zeros_like(t), target=l, n_samples=20), signed=True),
        "LRP (ε)":     lambda m, t, l: _lrp(m, t, l),
        "LIME":        lambda m, t, l: _lime_heat(m, t, l),
    }


def plot_robustness_benchmark(model, ckpt, tensor, img_np, label):
    """Run input + model robustness tests for every XAI method, normalised change metric."""
    import torchvision.transforms.functional as TF

    # Input perturbations (superficial — class should be unchanged)
    input_perturbs = [
        tensor + 0.15 * torch.randn_like(tensor),
        tensor * 1.3,
        tensor * 0.7,
        TF.rotate(tensor, 10),
        TF.gaussian_blur(tensor, kernel_size=7, sigma=2.0),
    ]

    # Model perturbations (Adebayo cascading randomization)
    def randomize(m, patterns):
        for name, p in m.named_parameters():
            if any(pat in name for pat in patterns):
                if p.dim() >= 2: nn.init.kaiming_normal_(p)
                else: nn.init.zeros_(p)
        return m

    model_perturbs = [
        randomize(load_model(ckpt), ["classifier"]),
        randomize(load_model(ckpt), ["classifier", "stages.3"]),
        randomize(load_model(ckpt), ["classifier", "stages.3", "stages.2"]),
        randomize(load_model(ckpt), [""]),
    ]

    methods = _all_methods()
    results = {}

    print("\nRobustness benchmark (this takes a few minutes)…")
    for name, fn in methods.items():
        print(f"  {name}…", end=" ", flush=True)
        try:
            base_attr = fn(model, tensor, label)
            # Input robustness: ideally LOW (stable to superficial changes)
            input_score = float(np.mean([_rel_change(base_attr, fn(model, tp, label)) for tp in input_perturbs]))
            # Model robustness: ideally HIGH (faithful to trained weights)
            model_score = float(np.mean([_rel_change(base_attr, fn(mp, tensor, label)) for mp in model_perturbs]))
            results[name] = (input_score, model_score)
            print(f"input_Δ={input_score:.2f}  model_Δ={model_score:.2f}")
        except Exception as e:
            print(f"skipped ({type(e).__name__})")

    # ── Summary plot ──────────────────────────────────────────────────────────
    # Sort by "faithfulness margin" = model_change - input_change (higher = better)
    ordered = sorted(results.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)
    names  = [k for k, _ in ordered]
    inputs = [v[0] for _, v in ordered]
    models = [v[1] for _, v in ordered]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(names))
    ax.barh(y - 0.2, inputs, height=0.4, color="#4fc3f7", label="Input-perturbation Δ (want LOW — stable)")
    ax.barh(y + 0.2, models, height=0.4, color="#f06292", label="Model-randomization Δ (want HIGH — faithful)")

    for i, (inp, mod) in enumerate(zip(inputs, models)):
        ax.text(inp + 0.02, i - 0.2, f"{inp:.2f}", va="center", fontsize=9, color="white")
        ax.text(mod + 0.02, i + 0.2, f"{mod:.2f}", va="center", fontsize=9, color="white")

    ax.set_yticks(y); ax.set_yticklabels(names, color="white")
    ax.set_xlabel("Relative change  ‖Δ attribution‖ / ‖baseline‖  (dimensionless)", color="white")
    ax.set_title("XAI Robustness Benchmark — sorted by faithfulness margin (model Δ − input Δ)\n"
                 "Ideal: short blue bar (stable), long pink bar (faithful)",
                 fontsize=12, fontweight="bold", color="white")
    ax.legend(loc="lower right", framealpha=0.15, labelcolor="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("white")
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout(); save(fig, "7_robustness_benchmark")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    df        = pd.read_csv(TEST_LABELS_CSV)
    grade_col = "myopic_maculopathy_grade"
    model     = load_model(args.checkpoint)

    print("Loading one image per class (from test set):")
    data = []
    for cls in range(NUM_CLASSES):
        path = os.path.join(TEST_IMG, df[df[grade_col] == cls].iloc[0]["image"])
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

    # Robustness tests — use grade 2 sample as a single fixed example
    sample_t, sample_img, sample_lbl = next((d for d in data if d[2] == 2), data[2])
    plot_input_robustness(model, sample_t, sample_lbl)
    plot_model_robustness(args.checkpoint, sample_t, sample_img, sample_lbl)
    plot_robustness_benchmark(model, args.checkpoint, sample_t, sample_img, sample_lbl)

    print(f"\nDone — all plots saved to {XAI_DIR}/")


if __name__ == "__main__":
    main()
