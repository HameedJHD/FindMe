"""
Re-ID feature extraction + track linking for the FindMe pipeline.

Two classes:

    ReidModel    – Thin wrapper around FastReID's DefaultPredictor.
                   Takes a list of BGR crops, returns L2-normalized
                   embedding tensors.

    TrackLinker  – Stateful, in-memory. Watches the stream of tracks
                   coming out of ByteTrack and re-identifies people
                   who left the frame and came back. Maintains a map
                   of (raw ByteTrack id) -> (canonical id) so that the
                   rest of the pipeline only ever sees stable IDs.

Both are intended to live for the duration of one video pipeline run
and then be discarded. Nothing is persisted to disk.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# FastReID source location
# ──────────────────────────────────────────────────────────────────────
# FastReID is not pip-installable, so we add its source folder to
# sys.path. The user is expected to drop the unzipped FastReID source
# at <project_root>/vendor/fast-reid-1.3.0/.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FASTREID_ROOT = _PROJECT_ROOT / "vendor" / "fast-reid-1.3.0"

if not _FASTREID_ROOT.exists():
    raise ImportError(
        f"FastReID source not found at {_FASTREID_ROOT}. "
        "Unzip the FastReID 1.3.0 archive there before importing this module."
    )

if str(_FASTREID_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASTREID_ROOT))

from fastreid.config import get_cfg          # noqa: E402
from fastreid.engine import DefaultPredictor # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Re-ID model wrapper
# ──────────────────────────────────────────────────────────────────────
class ReidModel:
    """BGR crops → L2-normalized feature embeddings.

    Mirrors the public surface of `src.attributes.AttrModel` so it can
    be used the same way from the pipeline.
    """

    def __init__(self, config_path, weights_path, device=0):
        cfg = get_cfg()
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.DEVICE = (
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        cfg.freeze()

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

        # FastReID stores INPUT.SIZE_TEST as (H, W).
        h, w = cfg.INPUT.SIZE_TEST
        self.input_h = int(h)
        self.input_w = int(w)

    @torch.no_grad()
    def batch_predict(self, crops):
        """Compute embeddings for a batch of BGR crops.

        Args:
            crops: list of np.ndarray (H, W, 3) in BGR order.

        Returns:
            torch.Tensor (N, D), L2-normalized, on CPU.
            Empty (0, 0) tensor if the input list is empty.
        """
        if not crops:
            return torch.empty((0, 0))

        imgs = []
        for c in crops:
            # Match FastReID's FeatureExtractionDemo: BGR -> RGB, then
            # resize to (W, H) using INTER_CUBIC. cv2.resize takes (W, H)
            # while INPUT.SIZE_TEST is (H, W).
            rgb = c[:, :, ::-1]
            resized = cv2.resize(
                rgb,
                (self.input_w, self.input_h),
                interpolation=cv2.INTER_CUBIC,
            )
            imgs.append(resized.astype("float32").transpose(2, 0, 1))

        batch = torch.from_numpy(np.stack(imgs))   # (N, 3, H, W)
        feats = self.predictor(batch)              # (N, D)
        feats = F.normalize(feats, dim=1)
        return feats.cpu()


# ──────────────────────────────────────────────────────────────────────
# Track linker
# ──────────────────────────────────────────────────────────────────────
class TrackLinker:
    """Re-identify ByteTrack ids that fragmented when a person briefly
    left the frame.

    Lifecycle of a person, from the linker's point of view:

        1. ByteTrack hands us a raw_id we've never seen.
        2. We compare the person's embedding against the embeddings of
           every track that "went lost" within the last
           `max_lost_frames` frames.
        3. If the best cosine similarity is above `match_threshold`,
           we treat this as the same person — we reuse the old
           canonical id and remap the new raw_id to it.
        4. Otherwise it's a genuinely new person and we mint a new
           canonical id.

    All state is in-memory and discarded when the pipeline run ends.

    Tunables:
        match_threshold  Cosine similarity above which two tracks are
                         considered the same person. 0.6 is a sane
                         default for FastReID Market1501 models.
                         Raise it to be stricter (fewer false merges,
                         more ID switches), lower it to be lenient.
        max_lost_frames  How long after disappearing a track is still
                         eligible for re-matching. At 30fps, 600 frames
                         ≈ 20 seconds.
        ema_alpha        Smoothing factor for active embeddings.
                         Higher = trust the running average more,
                         less reactive to noisy single frames.
    """

    def __init__(
        self,
        match_threshold: float = 0.6,
        max_lost_frames: int = 600,
        ema_alpha: float = 0.9,
    ):
        self.match_threshold = match_threshold
        self.max_lost_frames = max_lost_frames
        self.ema_alpha = ema_alpha

        # canonical_id -> {"emb": Tensor(D,), "last_frame": int}
        self.active: dict = {}
        # canonical_id -> {"emb": Tensor(D,), "lost_frame": int}
        self.lost: dict = {}
        # raw_track_id (from ByteTrack) -> canonical_id
        self.id_map: dict = {}

        self._next_canonical = 1

    # ---- public API ---------------------------------------------------
    def update(self, raw_ids, embeddings, frame_idx):
        """Process the tracks that came out of ByteTrack for one frame.

        Args:
            raw_ids:    list of int, ByteTrack track IDs
            embeddings: torch.Tensor (N, D), L2-normalized, on CPU,
                        in the same order as raw_ids
            frame_idx:  int, current frame index

        Returns:
            list of int canonical ids, in the same order as raw_ids.
        """
        canonical_ids = []
        seen_canonical_this_frame = set()

        for i, raw_id in enumerate(raw_ids):
            emb = embeddings[i]

            cached_cid = self.id_map.get(raw_id)
            if cached_cid is not None and cached_cid not in seen_canonical_this_frame:
                # Already linked to a canonical id, and that id hasn't
                # been claimed by another raw_id in this same frame.
                cid = cached_cid
                self._touch_active(cid, emb, frame_idx)
            else:
                # Either this raw_id is brand new, or its cached
                # canonical id was already taken this frame (which
                # means ByteTrack recycled the raw id onto a different
                # person). Re-match from scratch — and drop the stale
                # cached mapping so we don't keep colliding.
                if cached_cid is not None:
                    del self.id_map[raw_id]
                cid = self._match_or_new(
                    emb, frame_idx, seen_canonical_this_frame
                )
                self.id_map[raw_id] = cid

            canonical_ids.append(cid)
            seen_canonical_this_frame.add(cid)

        # Anyone who was active but didn't show up this frame moves to
        # the "lost" pool, where they're eligible for re-matching.
        self._sweep_lost(seen_canonical_this_frame, frame_idx)

        # Forget tracks that have been lost for too long.
        self._forget_old(frame_idx)

        return canonical_ids

    # ---- internals ----------------------------------------------------
    def _touch_active(self, cid, emb, frame_idx):
        """Update an active track's running embedding (EMA)."""
        if cid in self.active:
            old = self.active[cid]["emb"]
            new = self.ema_alpha * old + (1.0 - self.ema_alpha) * emb
            new = F.normalize(new, dim=0)
            self.active[cid]["emb"] = new
            self.active[cid]["last_frame"] = frame_idx
        else:
            # Was lost (or never registered as active) — promote.
            self.active[cid] = {"emb": emb, "last_frame": frame_idx}
            self.lost.pop(cid, None)

    def _match_or_new(self, emb, frame_idx, seen_this_frame):
        """Try to match `emb` against a recently-lost track.

        `seen_this_frame` holds canonical ids already claimed earlier
        in this frame — we never let two raw tracks in the same frame
        collapse onto the same canonical id.
        """
        if not self.lost:
            return self._new_canonical(emb, frame_idx)

        best_cid = None
        best_sim = -1.0
        for cid, info in self.lost.items():
            if cid in seen_this_frame:
                continue
            sim = float(torch.dot(emb, info["emb"]).item())
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_cid is not None and best_sim >= self.match_threshold:
            # Re-identified. Restore canonical id, drop from lost pool.
            self.active[best_cid] = {"emb": emb, "last_frame": frame_idx}
            del self.lost[best_cid]
            return best_cid

        return self._new_canonical(emb, frame_idx)

    def _new_canonical(self, emb, frame_idx):
        cid = self._next_canonical
        self._next_canonical += 1
        self.active[cid] = {"emb": emb, "last_frame": frame_idx}
        return cid

    def _sweep_lost(self, seen_this_frame, frame_idx):
        for cid in list(self.active.keys()):
            if cid not in seen_this_frame:
                self.lost[cid] = {
                    "emb": self.active[cid]["emb"],
                    "lost_frame": frame_idx,
                }
                del self.active[cid]

    def _forget_old(self, frame_idx):
        cutoff = frame_idx - self.max_lost_frames
        if cutoff <= 0:
            return
        for cid in list(self.lost.keys()):
            if self.lost[cid]["lost_frame"] < cutoff:
                # Drop the canonical id and any raw ids that pointed at it.
                # (raw ids in id_map are harmless to leave, but cleaning up
                # avoids unbounded memory growth on very long videos.)
                del self.lost[cid]
                stale = [r for r, c in self.id_map.items() if c == cid]
                for r in stale:
                    del self.id_map[r]
