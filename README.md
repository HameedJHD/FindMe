# People Detection + Attribute Pipeline

This project detects people with YOLO, tracks them (ByteTrack), extracts attributes (PETA-trained multi-head CNN), and stores results in a database.

## Files

- `train/peta_dataset.py` — Loads PETA labels (sparse tokens), builds an index of images, splits train/val, computes `pos_weight`.
- `train/peta_model.py` — Multi-head ResNet50: 61 binary logits + 4×11-class color heads.
- `train/peta.py` — Training loop for the attribute model (BCE for binaries, CE for colors), uses `pos_weight`, supports `num_workers=0` to avoid multiprocessing issues.
- `train/peta_resnet50_multihead.pt` — Trained attribute model weights (output).
- `train/bytetrack.py` — ByteTrack wrapper scaffold. Replace the stub with a real ByteTrack implementation; keeps API `update(dets) -> [x1,y1,x2,y2,track_id,score]`.
- `train/db.py` — SQLite helper; creates `detections` table and inserts per-track attributes.
- `train/attributes.py` — Attribute inference wrapper: loads `peta_resnet50_multihead.pt`, preprocesses crops, returns binary and color probabilities.
- `train/run_pipeline.py` — End-to-end inference: YOLO detect → ByteTrack → crop persons → attribute model → DB insert.
- `datasets/crowdhuman.yaml` — YOLO dataset config (person class only, train/val paths).

## How to run

1. **Train YOLO (Python)**  
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8m.pt")  # or yolov8s.pt for speed
   model.train(data="datasets/crowdhuman.yaml", epochs=200, imgsz=960)
   ```
2. **Train attributes**  
   ```bash
   python train/peta.py
   ```
3. **Run pipeline**  
   - Replace the ByteTrack stub with your actual tracker, or install a ByteTrack implementation.
   - Set `DET_MODEL` and `ATTR_MODEL` in `train/run_pipeline.py`.
   ```bash
   python train/run_pipeline.py
   ```
   Results are stored in `detections.sqlite`.

## Notes

- If you see missing-image drops, `peta_dataset.py` logs how many labels were skipped.
- For lower GPU memory, reduce `batch_size` in `peta.py` (e.g., 8–16) and use `torch.cuda.amp`.
- To increase FPS: use YOLO TensorRT FP16, `imgsz=832`, and run attributes every N frames per track (`ATTR_EVERY_N_FRAMES`).