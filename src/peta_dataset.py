import torch
from torch.utils.data import Dataset
from PIL import Image
import os, glob, numpy as np, random

BINARY_ATTRS = [
    "accessoryHeadphone","personalLess15","personalLess30","personalLess45","personalLess60",
    "personalLarger60","carryingBabyBuggy","carryingBackpack","hairBald","footwearBoots",
    "lowerBodyCapri","carryingOther","carryingShoppingTro","carryingUmbrella",
    "lowerBodyCasual","upperBodyCasual","personalFemale","carryingFolder","lowerBodyFormal",
    "upperBodyFormal","accessoryHairBand","accessoryHat","lowerBodyHotPants","upperBodyJacket",
    "lowerBodyJeans","accessoryKerchief","footwearLeatherShoes","upperBodyLogo","hairLong",
    "lowerBodyLongSkirt","upperBodyLongSleeve","lowerBodyPlaid","lowerBodyThinStripes",
    "carryingLuggageCase","personalMale","carryingMessengerBag","accessoryMuffler",
    "accessoryNothing","carryingNothing","upperBodyNoSleeve","upperBodyPlaid",
    "carryingPlasticBags","footwearSandals","footwearShoes","hairShort","lowerBodyShorts",
    "upperBodyShortSleeve","lowerBodyShortSkirt","footwearSneaker","footwearStocking",
    "upperBodyThinStripes","upperBodySuit","carryingSuitcase","lowerBodySuits",
    "accessorySunglasses","upperBodySweater","upperBodyThickStripes","lowerBodyTrousers",
    "upperBodyTshirt","upperBodyOther","upperBodyVNeck"
]
BIN2IDX = {k.lower(): i for i, k in enumerate(BINARY_ATTRS)}

COLOR_VOCAB = ["black","blue","brown","green","grey","orange","pink","purple","red","white","yellow"]
COLOR_TO_ID = {c: i for i, c in enumerate(COLOR_VOCAB)}
COLOR_REGIONS = {"footwear": 0, "hair": 1, "lowerbody": 2, "upperbody": 3}

def parse_tokens_to_targets(tokens):
    bin_targets = np.zeros(len(BINARY_ATTRS), dtype=float)
    color_targets = np.full(4, -1, dtype=int)
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        tl = t.lower()
        matched_color = False
        for region, ridx in COLOR_REGIONS.items():
            if tl.startswith(region):
                color = tl[len(region):]
                if color in COLOR_TO_ID:
                    color_targets[ridx] = COLOR_TO_ID[color]
                    matched_color = True
                    break
        if matched_color:
            continue
        if tl in BIN2IDX:
            bin_targets[BIN2IDX[tl]] = 1.0
    return bin_targets, color_targets

class PetaDataset(Dataset):
    def __init__(self, root, label_file, split="train", val_split=0.2, seed=42, transform=None):
        self.root = root
        self.transform = transform
        self.file_index = self._build_index(root)

        with open(label_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        samples = []
        dropped = 0
        for ln in lines:
            parts = ln.split()
            if len(parts) < 2:
                continue
            img_id, tokens = parts[0], parts[1:]
            if img_id not in self.file_index:
                dropped += 1
                continue
            bin_t, col_t = parse_tokens_to_targets(tokens)
            samples.append((img_id, bin_t, col_t))

        if dropped:
            print(f"[PETA] Dropped {dropped} labels with missing images; kept {len(samples)}")

        random.seed(seed)
        random.shuffle(samples)
        n_val = int(len(samples) * val_split)
        if split == "train":
            self.samples = samples[n_val:]
        else:
            self.samples = samples[:n_val]

    def _build_index(self, root):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        index = {}
        for ext in exts:
            for path in glob.glob(os.path.join(root, "**", ext), recursive=True):
                base = os.path.basename(path)
                index[base] = path
        if not index:
            raise RuntimeError(f"No images found under {root}")
        return index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, bins, colors = self.samples[idx]
        img_path = self.file_index[img_id]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        bin_targets = torch.tensor(bins, dtype=torch.float32)
        color_targets = torch.tensor(colors, dtype=torch.long)
        return img, bin_targets, color_targets

def compute_pos_weight(dataset, num_bins=61):
    counts_pos = np.zeros(num_bins)
    counts_total = np.zeros(num_bins)
    for _, bins, _ in dataset.samples:
        counts_pos += bins
        counts_total += 1
    pos_weight = (counts_total - counts_pos) / np.clip(counts_pos, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)