import json, sqlite3, cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "database" / "detections.sqlite"
VIDEO_ROOT = ROOT / "data" / "videos"
OUTPUT_VIDEO = ROOT / "output" / "query_result.mp4"

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
BIN_IDX = {k: i for i, k in enumerate(BINARY_ATTRS)}

COLOR_VOCAB = ["black","blue","brown","green","grey","orange","pink","purple","red","white","yellow"]
COLOR_NAMES = ["footwear","hair","lowerbody","upperbody"]
COLOR_IDX = {name: i for i, name in enumerate(COLOR_NAMES)}
COLOR_VAL_IDX = {c: i for i, c in enumerate(COLOR_VOCAB)}

def matches_attr(bin_probs, color_probs, bin_requirements, color_requirements, bin_thresh=0.5, color_thresh=0.5):
    for name in bin_requirements:
        if name not in BIN_IDX: 
            return False
        if bin_probs[BIN_IDX[name]] < bin_thresh:
            return False
    for region, color in color_requirements.items():
        ridx = COLOR_IDX[region]
        cidx = COLOR_VAL_IDX[color]
        if color_probs[ridx][cidx] < color_thresh:
            return False
    return True

def collect_matches(bin_requirements, color_requirements, bin_thresh=0.5, color_thresh=0.5):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT video_id, frame_idx, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_conf, bin_probs, color_probs "
        "FROM detections"
    ).fetchall()
    conn.close()

    results = {}
    for r in rows:
        video_id, frame_idx, tid, x1, y1, x2, y2, conf, bin_js, col_js = r
        bin_probs = json.loads(bin_js)
        color_probs = json.loads(col_js)
        if not matches_attr(bin_probs, color_probs, bin_requirements, color_requirements, bin_thresh, color_thresh):
            continue
        results.setdefault((video_id, tid), []).append((frame_idx, (x1, y1, x2, y2), conf, bin_probs, color_probs))
    return results

def render_tracks(match_dict):
    # Assumes one video_id; if multiple, extend to loop over videos.
    if not match_dict:
        print("No matches found.")
        return
    (video_id, _) = next(iter(match_dict.keys()))
    cap = cv2.VideoCapture(str(VIDEO_ROOT / video_id))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Detected FPS:", fps)
    if fps <= 0:  # fallback if OpenCV fails
        fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


    # Build frame->list of boxes for all matched tracks
    frame_map = {}
    for (vid, tid), entries in match_dict.items():
        for frame_idx, box, conf, bin_probs, color_probs in entries:
            frame_map.setdefault(frame_idx, []).append((tid, box, conf, bin_probs, color_probs))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in frame_map:
            for tid, box, conf, bin_probs, color_probs in frame_map[idx]:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # simple label: track id and footwear color argmax
                footwear_probs = color_probs[COLOR_IDX["footwear"]]
                cidx = max(range(len(footwear_probs)), key=lambda j: footwear_probs[j])
                color_label = COLOR_VOCAB[cidx]
                cv2.putText(frame, f"ID {tid} {color_label} conf {conf:.2f}",
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    print("Saved to", OUTPUT_VIDEO)

if __name__ == "__main__":
    # Example: male + carryingBackpack + footwear black + hair black
    bin_requirements = ["lowerBodyShorts" , "accessoryMuffler"]
    color_requirements = {"hair": "black"}
    matches_found = collect_matches(bin_requirements, color_requirements, bin_thresh=0.5, color_thresh=0.4)
    print(f"Matched {len(matches_found)} tracks")
    render_tracks(matches_found)
    