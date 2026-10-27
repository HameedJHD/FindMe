import json, sqlite3
from pathlib import Path

# Copy from train/peta_dataset.py
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
COLOR_VOCAB = ["black","blue","brown","green","grey","orange","pink","purple","red","white","yellow"]
COLOR_NAMES = ["footwear","hair","lowerbody","upperbody"]

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "database" / "detections.sqlite"

def decode_row(row, topk=5):
    frame_idx, tid, conf, bin_js, col_js = row
    bin_probs = json.loads(bin_js)
    color_probs = json.loads(col_js)

    top = sorted(zip(range(len(bin_probs)), bin_probs), key=lambda x: x[1], reverse=True)[:topk]
    bin_readable = [(BINARY_ATTRS[i], round(p, 3)) for i, p in top]

    colors = []
    for region_idx, probs in enumerate(color_probs):
        if not probs:
            continue
        ci = max(range(len(probs)), key=lambda j: probs[j])
        colors.append(f"{COLOR_NAMES[region_idx]}={COLOR_VOCAB[ci]} ({probs[ci]:.3f})")

    return bin_readable, colors

def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT frame_idx, track_id, det_conf, bin_probs, color_probs FROM detections LIMIT 50").fetchall()
    for r in rows:
        bins, colors = decode_row(r, topk=5)
        print(f"frame {r[0]} track {r[1]} det_conf {r[2]:.2f}")
        print("  bins:", bins)
        print("  colors:", colors)
    conn.close()

if __name__ == "__main__":
    main()