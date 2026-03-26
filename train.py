"""
ConvNextV2 fine-tuning for Myopic Maculopathy grading (5 classes).
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from transformers import ConvNextV2ForImageClassification, AutoImageProcessor

# ── Config ─────────────────────────────────────────────────────────────────────
DATA = "/home/moritz/Applied_ai_cw_2/Data"
TRAIN_IMG = f"{DATA}/Training/Training_Images"
LABELS_CSV = f"{DATA}/Training/Training_LabelsDemographic.csv"
MODEL_ID = "facebook/convnextv2-tiny-1k-224"
EPOCHS = 256
BATCH = 128j
LR = 3e-4
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset ────────────────────────────────────────────────────────────────────
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

class FundusDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB")
        return transform(img), int(row["myopic_maculopathy_grade"])

df = pd.read_csv(LABELS_CSV)
dataset = FundusDataset(df, TRAIN_IMG)

labels = df["myopic_maculopathy_grade"].values
train_idx, val_idx = train_test_split(
    range(len(df)), test_size=0.15, stratify=labels, random_state=42
)
train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4)

# ── Model ──────────────────────────────────────────────────────────────────────
model = ConvNextV2ForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,   # replace classifier head
).to(DEVICE)

optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

PATIENCE = 35
best_acc, patience_left = 0.0, PATIENCE

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        loss = criterion(model(imgs).logits, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    val_acc = correct / total
    print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss/len(train_loader):.4f}  val_acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_left = PATIENCE
        torch.save(model.state_dict(), "convnextv2_mmac.pt")
    else:
        patience_left -= 1
        if patience_left == 0:
            print(f"Early stopping — best val_acc={best_acc:.3f}")
            break

print(f"Saved best model (val_acc={best_acc:.3f})")
