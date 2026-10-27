import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.peta_dataset import PetaDataset, compute_pos_weight
from src.peta_model import PetaAttrModel

device = "cuda" if torch.cuda.is_available() else "cpu"
num_bins = 61
num_colors = 11
epochs = 30
batch_size = 16
lr = 3e-4
wd = 1e-4

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

peta_root = ROOT / "datasets" / "PETA" / "PETA dataset" / "3DPeS" / "archive"
label_file = peta_root / "Label.txt"

def main():
    train_ds = PetaDataset(str(peta_root), str(label_file), split="train", val_split=0.2, seed=42, transform=train_tf)
    val_ds   = PetaDataset(str(peta_root), str(label_file), split="val",   val_split=0.2, seed=42, transform=val_tf)

    pos_weight = compute_pos_weight(train_ds, num_bins=num_bins).to(device)
    # Use reduction='none' so we can mask ignored labels
    bin_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    color_crit = nn.CrossEntropyLoss(ignore_index=-1)

    # Set num_workers=0 if you still have multiprocessing issues
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = PetaAttrModel(num_binary=num_bins, num_color_classes=num_colors).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    def step(loader, train=True):
        model.train(train)
        total = 0
        for imgs, bin_t, col_t in loader:
            imgs = imgs.to(device)
            bin_t = bin_t.to(device)  # (B, 61), 0/1, -1 for ignore
            col_t = col_t.to(device)  # (B, 4), 0–10 or -1 for ignore

            bin_logits, color_logits = model(imgs)

            # Binary loss with masking
            bin_mask = (bin_t != -1)                       # (B, 61) bool
            bin_targets = bin_t.clamp(min=0)               # replace -1 with 0
            bin_loss_raw = bin_crit(bin_logits, bin_targets)  # (B, 61)
            valid = bin_mask.float().sum()
            if valid > 0:
                bin_loss = (bin_loss_raw * bin_mask.float()).sum() / valid
            else:
                bin_loss = torch.zeros((), device=device)

            # Color losses (CE handles ignore_index=-1)
            color_loss = 0.0
            for i in range(4):
                color_loss = color_loss + color_crit(color_logits[i], col_t[:, i])

            loss = bin_loss + color_loss

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item()
        return total / max(1, len(loader))

    for ep in range(epochs):
        tr = step(train_loader, train=True)
        va = step(val_loader, train=False)
        sched.step()
        print(f"Epoch {ep+1}/{epochs} train {tr:.4f} val {va:.4f}")

    out_path = ROOT / "models" / "peta" / "peta_resnet50_multihead.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))

if __name__ == "__main__":
    # Uncomment if you still see multiprocessing issues:
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main()