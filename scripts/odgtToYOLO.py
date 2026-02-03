import json
from pathlib import Path
from PIL import Image

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_DIR = Path("../../CrowdHuman")      # Root folder containing images + ODGT
SPLITS = ["train", "val"]                # Your dataset splits

IMAGE_DIR = DATA_DIR / "images"
ODGT_FILES = {
    "train": DATA_DIR / "annotation_train.odgt",
    "val":   DATA_DIR / "annotation_val.odgt"
}

LABEL_DIR = DATA_DIR / "labels"          # Where YOLO .txt files will go

# Ensure label folders exist
for split in SPLITS:
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def xywh_to_yolo(box, img_w, img_h):
    """Convert [x, y, w, h] to YOLO normalized format."""
    x, y, w, h = box
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def load_odgt(path):
    """Load ODGT annotation file (one JSON per line)."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

# ----------------------------
# MAIN LOOP
# ----------------------------
for split in SPLITS:
    print(f"Processing {split} split...")
    odgt_path = ODGT_FILES[split]
    records = load_odgt(odgt_path)

    for record in records:
        img_id = record["ID"]
        img_path = IMAGE_DIR / split / f"{img_id}.jpg"

        if not img_path.exists():
            print(f"WARNING: Image not found: {img_path}")
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        lines = []
        for obj in record["gtboxes"]:
            head_attr = obj.get("head_attr", {})           # safe fallback
            ignore_flag = head_attr.get("ignore", 0)
            occ = head_attr.get("occ", 0)
            unsure = head_attr.get("unsure", 0)

            # Skip ignored / fully occluded / unsure boxes
            if ignore_flag == 1 or occ == 2 or unsure == 1:
                continue

            # Convert fbox to YOLO format
            box = obj["fbox"]
            x_c, y_c, w_n, h_n = xywh_to_yolo(box, img_w, img_h)
            line = f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
            lines.append(line)

        # Write label file
        label_path = LABEL_DIR / split / f"{img_id}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Finished {split} split. Labels saved in {LABEL_DIR / split}")
