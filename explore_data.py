"""
Data exploration for MMAC Task 1 - Myopic Maculopathy Classification
5 classes: 0=No pathology, 1=Tessellated fundus, 2=Diffuse chorioretinal atrophy,
           3=Patchy chorioretinal atrophy, 4=Macular atrophy
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

DATA = "/home/moritz/Applied_ai_cw_2/Data"
TRAIN_IMG = f"{DATA}/Training/Training_Images"
LABELS_CSV = f"{DATA}/Training/Training_LabelsDemographic.csv"
TEST_IMG = f"{DATA}/Testing/Testing_Images"
TEST_CSV = f"{DATA}/Testing/Testing_LabelDemographic.csv"

df = pd.read_csv(LABELS_CSV)
df_test = pd.read_csv(TEST_CSV)
CLASS_NAMES = {
    0: "No pathology",
    1: "Tessellated fundus",
    2: "Diffuse CRA",
    3: "Patchy CRA",
    4: "Macular atrophy",
}

# ── 0. Dataset split overview ─────────────────────────────────────────────────
n_train, n_test = len(df), len(df_test)
total = n_train + n_test
print(f"Dataset split:  train={n_train}  test={n_test}  total={total}")
print(f"               ({n_train/total*100:.1f}% / {n_test/total*100:.1f}%)")

# ── 1. Class imbalance ────────────────────────────────────────────────────────
counts      = df["myopic_maculopathy_grade"].value_counts().sort_index()
counts_test = df_test["myopic_maculopathy_grade"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(NUM_CLASSES := 5)
w = 0.4
bars_tr = axes[0].bar(x - w/2, counts.values,      w, label=f"Train (n={n_train})", color="steelblue")
bars_te = axes[0].bar(x + w/2, counts_test.reindex(range(NUM_CLASSES), fill_value=0).values,
                      w, label=f"Test  (n={n_test})",  color="coral")
for bar in bars_tr:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(int(bar.get_height())), ha="center", fontsize=8)
for bar in bars_te:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(int(bar.get_height())), ha="center", fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"C{i}\n{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)], fontsize=8)
axes[0].set_title("Class distribution — train vs test")
axes[0].set_ylabel("Count")
axes[0].legend()

# imbalance ratio (train only — what the model sees)
majority = counts.max()
ratios = majority / counts
axes[1].bar([f"C{i}" for i in counts.index], ratios.values, color=plt.cm.tab10.colors[:5])
axes[1].axhline(1, color="grey", linestyle="--", linewidth=0.8)
axes[1].set_title("Imbalance ratio in train (majority / class)")
axes[1].set_ylabel("Ratio")
for i, (idx, r) in enumerate(ratios.items()):
    axes[1].text(i, r + 0.1, f"{r:.1f}×", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=120)
plt.close()
print("Saved class_distribution.png")
print("\nTrain counts:")
print(counts.to_string())
print("\nTest counts:")
print(counts_test.to_string())
print(f"\nMax imbalance ratio (train): {ratios.max():.1f}× (C{ratios.idxmax()} vs majority)")

# ── 2. Example images per class ───────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(2, 5, hspace=0.4)

for cls in range(5):
    samples = df[df["myopic_maculopathy_grade"] == cls]["image"].values[:2]
    for row, fname in enumerate(samples):
        ax = fig.add_subplot(gs[row, cls])
        img = Image.open(os.path.join(TRAIN_IMG, fname))
        ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title(f"C{cls}: {CLASS_NAMES[cls]}\n(n={counts[cls]})", fontsize=8)

plt.suptitle("2 examples per class  —  800×800 RGB fundus images", fontsize=11)
plt.savefig("example_images.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved example_images.png")

# ── 3. Quick image stats (size / channels sanity check) ───────────────────────
sizes = set()
for fname in df["image"].values[:20]:
    img = Image.open(os.path.join(TRAIN_IMG, fname))
    sizes.add((img.size, img.mode))
print(f"\nUnique (size, mode) in first 20 images: {sizes}")
