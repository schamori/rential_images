"""
Light augmentations for retinal fundus images.
Run directly to visualise all transforms on a sample image:
    python augmentations.py
"""
import random
import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image, ImageFilter
import yaml


# ── Custom transforms ─────────────────────────────────────────────────────────

class RandomDefocus:
    """Simulate lens defocus with a randomly sized Gaussian blur."""
    def __init__(self, radius=(0.5, 2.0)):
        self.radius = radius

    def __call__(self, img):
        r = random.uniform(*self.radius)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomGamma:
    """Gamma correction — shifts overall image brightness non-linearly."""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        return F.adjust_gamma(img, gamma)


class RandomGaussianNoise:
    """Additive Gaussian noise (sensor noise simulation)."""
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0, 1)


class RandomNonLinearIntensity:
    """S-curve / power-law intensity correction."""
    def __init__(self, alpha_range=(0.85, 1.15)):
        self.alpha_range = alpha_range

    def __call__(self, img):
        alpha = random.uniform(*self.alpha_range)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, alpha)
        return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


class RandomNonLinearIntensityTTDA:
    def __init__(self, alpha_range=(0.85, 1.15)):
        self.alpha_range = alpha_range

    def __call__(self, tensor):
        alpha = random.uniform(*self.alpha_range)
        return torch.pow(tensor, alpha)
    
class RandomGammaTTDA:
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, x):
        x = torch.clamp(x, 0, 1)
        gamma = random.uniform(*self.gamma_range)
        return torch.clamp(x.pow(gamma), 0, 1)
    
class RandomGaussianNoiseTTDA:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        return (x + torch.randn_like(x) * self.std).clamp(0, 1)


# ── Public API ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        # orientation invariance
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),               
        # photometric — all kept light
        T.ColorJitter(brightness=0.2, contrast=0.2),
        RandomDefocus(radius=(0.0, 1.5)),           
        RandomGamma(gamma_range=(0.85, 1.15)),      
        RandomNonLinearIntensity(alpha_range=(0.9, 1.1)),
        T.ToTensor(),
        RandomGaussianNoise(std=0.015),            
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_val_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_ttda_transform(img_size=224, n_aug=5):
    ttda_transforms = []
    for _ in range(n_aug):
        ttda_transforms.append(
            T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
    return ttda_transforms

IMAGENET_STD_T  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)

def tensor_to_pil(t):
    t = t.cpu() * IMAGENET_STD_T + IMAGENET_MEAN_T
    t = t.clamp(0, 1)
    return ToPILImage()(t)

# ── Visualisation (run this file directly) ───────────────────────────────────

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    
    with open("./data.yaml") as f:
        data_path = yaml.safe_load(f)

    DATA = data_path["data_root"]
    IMG_DIR = f"{DATA}/Training/Training_Images"
    
    sample = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    orig = Image.open(sample).convert("RGB")

    # show each augmentation individually (no normalise so colours are readable)
    named_transforms = [
        ("Original",                  T.Resize((224, 224))),
        ("H/V Flip + Rotation",       T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(p=1), T.RandomRotation(15)])),
        ("Brightness & Contrast",     T.Compose([T.Resize((224,224)), T.ColorJitter(brightness=0.4, contrast=0.4)])),
        ("Defocus blur",              T.Compose([T.Resize((224,224)), RandomDefocus((1.0, 2.0))])),
        ("Gamma / lighting",          T.Compose([T.Resize((224,224)), RandomGamma((0.6, 1.4))])),
        ("Non-linear intensity",      T.Compose([T.Resize((224,224)), RandomNonLinearIntensity((0.7, 1.3))])),
        ("Gaussian noise",            T.Compose([T.Resize((224,224)), T.ToTensor(),
                                                 RandomGaussianNoise(std=0.05),
                                                 T.Lambda(lambda t: t.permute(1,2,0).numpy())])),
        ("Full train pipeline",       T.Compose([T.Resize((224,224)),
                                                 T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                                                 T.RandomRotation(15),
                                                 T.ColorJitter(brightness=0.2, contrast=0.2),
                                                 RandomDefocus((0.0, 1.5)),
                                                 RandomGamma((0.85, 1.15)),
                                                 RandomNonLinearIntensity((0.9, 1.1))])),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, tfm) in zip(axes.flat, named_transforms):
        out = tfm(orig)
        if isinstance(out, np.ndarray):       # already HWC float
            ax.imshow(out.clip(0, 1))
        elif isinstance(out, torch.Tensor):   # CHW tensor
            ax.imshow(out.permute(1, 2, 0).numpy().clip(0, 1))
        else:                                 # PIL
            ax.imshow(out)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    plt.suptitle("Augmentation visualisation", fontsize=12)
    plt.tight_layout()
    plt.savefig("augmentations_preview.png", dpi=130)
    plt.show()
    print("Saved augmentations_preview.png")
    