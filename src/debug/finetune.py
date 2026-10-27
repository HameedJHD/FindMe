import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from PIL import Image

# --- config ---
csv_path = Path("peta_train_val.csv")   # your CSV with image_path + attrs
img_root = Path(".")                    # root to prepend to image paths if needed
checkpoint = "resnet_ckpt.pth"          # your existing ResNet state_dict
num_attrs = 35                          # set to your attribute count (e.g., 35 or 61)
batch_size = 32
epochs = 15
lr = 1e-3
weight_decay = 1e-4
freeze_backbone = False
num_workers = 4

# --- dataset ---
class PetaDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.imgs = df["image_path"].tolist()
        self.labels = df.drop(columns=["image_path"]).values.astype("float32")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(img_root / self.imgs[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        y = torch.from_numpy(self.labels[idx])
        return img, y

tf_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
tf_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

df = pd.read_csv(csv_path)
# split
val_frac = 0.2
val_size = int(len(df) * val_frac)
df_val = df.sample(val_size, random_state=42)
df_train = df.drop(df_val.index)

train_ds = PetaDataset(df_train, tf_train)
val_ds   = PetaDataset(df_val, tf_val)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# --- model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
state = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(state, strict=False)  # old head may mismatch
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_attrs)

if freeze_backbone:
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = False

model.to(device)

# optional pos_weight for imbalance: tensor of shape [num_attrs]
pos_weight = None  # or torch.tensor([...], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def run_epoch(loader, train_mode=True):
    model.train(train_mode)
    total = 0
    loss_sum = 0.0
    correct_sum = 0
    count_sum = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if train_mode: optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct_sum += (preds == labels).sum().item()
        count_sum += labels.numel()
        total += imgs.size(0)
    return loss_sum / total, correct_sum / count_sum  # loss per image, mean accuracy over bits

best_acc = 0.0
for epoch in range(1, epochs + 1):
    train_loss, train_bitacc = run_epoch(train_loader, True)
    val_loss, val_bitacc = run_epoch(val_loader, False)
    scheduler.step()
    print(f"Epoch {epoch}: train loss {train_loss:.4f} bit-acc {train_bitacc:.3f} | val loss {val_loss:.4f} bit-acc {val_bitacc:.3f}")
    if val_bitacc > best_acc:
        best_acc = val_bitacc
        torch.save(model.state_dict(), "resnet_peta_finetuned_best.pth")
        print("  saved resnet_peta_finetuned_best.pth")