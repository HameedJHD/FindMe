import sqlite3
import json
from pathlib import Path

class DbClient:
    def __init__(self, db_path="detections.sqlite"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-200000")  # ~200 MB cache
        self._pending = []
        self._batch_size = 1000
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            frame_idx INTEGER,
            timestamp_ms REAL,
            track_id INTEGER,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            det_conf REAL,
            bin_probs TEXT,      -- JSON array length 61
            color_probs TEXT     -- JSON array of 4 arrays length 11
        );
        """)
        self.conn.commit()

    def insert(self, video_id, frame_idx, timestamp_ms, track_id, box, det_conf, attr_out):
        bin_probs = attr_out["bin_probs"]
        color_probs = attr_out["color_probs"]
        x1, y1, x2, y2 = box
        self._pending.append((
            video_id, frame_idx, float(timestamp_ms), int(track_id),
            float(x1), float(y1), float(x2), float(y2), float(det_conf),
            json.dumps(bin_probs), json.dumps(color_probs)
        ))
        if len(self._pending) >= self._batch_size:
            self._flush()

    def _flush(self):
        if not self._pending:
            return
        self.conn.execute("BEGIN")
        self.conn.executemany(
            "INSERT INTO detections (video_id, frame_idx, timestamp_ms, track_id, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_conf, bin_probs, color_probs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._pending
        )
        self.conn.commit()
        self._pending.clear()

    def close(self):
        self._flush()
        self.conn.close()