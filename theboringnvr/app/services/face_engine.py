"""Face recognition engine (InsightFace, optional Intel OpenVINO acceleration).

Per person event we analyse 3-5 snapshots (multi-frame voting):
  1. every frame: detect faces, keep the best one (detector score x size x sharpness)
  2. embed it and compare with all known embeddings via cosine similarity
  3. vote: known person with the highest summed score wins, otherwise the sharpest frame becomes a
     review-queue candidate
The winning frame is cropped into a circular avatar (PNG with alpha).
"""
from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import DATA_DIR

log = logging.getLogger("nvr.face")


@dataclass
class Candidate:
    embedding: np.ndarray
    quality: float
    crop_png: bytes
    person_id: int | None
    score: float


@dataclass
class FaceResult:
    person_id: int | None      # None = unknown -> review queue
    score: float
    embedding: np.ndarray
    crop_png: bytes
    best_person_id: int | None  # closest known person even if below threshold (for review hints)
    frames_used: int


def circle_crop(img: np.ndarray, bbox, size: int = 256, margin: float = 0.45) -> bytes:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    scale = size / side
    m = np.array([[scale, 0, -(cx - side / 2) * scale], [0, scale, -(cy - side / 2) * scale]], dtype=np.float32)
    out = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    rgba = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ss = 4  # supersampled mask for smooth edges
    mask = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size * ss - 1, size * ss - 1), fill=255)
    rgba.putalpha(mask.resize((size, size), Image.LANCZOS))
    buf = io.BytesIO()
    rgba.save(buf, "PNG", optimize=True)
    return buf.getvalue()


class FaceEngine:
    def __init__(self, cfg, db) -> None:
        self.cfg, self.db = cfg, db
        self._app = None
        self._lock = threading.Lock()
        self._ver = -1
        self._ids = np.zeros(0, dtype=np.int64)
        self._mat = np.zeros((0, 512), dtype=np.float32)
        self.provider_in_use = "not loaded"

    # ---------- model ----------
    def _build(self, providers):
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=self.cfg.model_name, root=str(DATA_DIR / "models"),
                           allowed_modules=["detection", "recognition"], providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        app.get(np.zeros((480, 640, 3), dtype=np.uint8))   # warm-up / provider sanity check
        return app

    def ensure_loaded(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            import onnxruntime as ort
            avail = ort.get_available_providers()
            dev = self.cfg.inference_device
            if "OpenVINOExecutionProvider" in avail and dev != "CPU_ORT":
                try:
                    self._app = self._build([("OpenVINOExecutionProvider", {"device_type": dev}),
                                             "CPUExecutionProvider"])
                    self.provider_in_use = f"OpenVINO ({dev})"
                except Exception as exc:  # noqa: BLE001
                    log.warning("OpenVINO (%s) failed (%s) - falling back to ONNX Runtime CPU", dev, exc)
            if self._app is None:
                self._app = self._build(["CPUExecutionProvider"])
                self.provider_in_use = "ONNX Runtime CPU"
            log.info("Face engine ready: model=%s provider=%s", self.cfg.model_name, self.provider_in_use)

    # ---------- matching ----------
    def _refresh(self) -> None:
        if self._ver != self.db.emb_version:
            self._ids, self._mat = self.db.embedding_matrix()
            self._ver = self.db.emb_version

    def match(self, emb: np.ndarray) -> tuple[int | None, float]:
        """Best cosine similarity per person (embeddings are L2-normalised)."""
        self._refresh()
        if not len(self._ids):
            return None, 0.0
        sims = self._mat @ emb
        best_pid, best = None, -1.0
        for pid in np.unique(self._ids):
            s = float(sims[self._ids == pid].max())
            if s > best:
                best_pid, best = int(pid), s
        return best_pid, best

    # ---------- analysis ----------
    def _best_face(self, img: np.ndarray):
        faces = self._app.get(img)
        best, best_q = None, 0.0
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            size = min(x2 - x1, y2 - y1)
            if f.det_score < self.cfg.min_det_score or size < self.cfg.min_face_px:
                continue
            xi1, yi1, xi2, yi2 = [max(int(v), 0) for v in (x1, y1, x2, y2)]
            patch = cv2.cvtColor(img[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size else 0.0
            q = float(f.det_score) * min(1.0, size / 160.0) * min(1.0, 0.3 + sharp / 200.0)
            if q > best_q:
                best, best_q = f, q
        return best, best_q

    def analyze_frames(self, jpegs: list[bytes]) -> FaceResult | None:
        """Blocking - call via asyncio.to_thread."""
        self.ensure_loaded()
        cands: list[Candidate] = []
        for jpg in jpegs:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            face, q = self._best_face(img)
            if face is None:
                continue
            emb = face.normed_embedding.astype(np.float32)
            pid, score = self.match(emb)
            cands.append(Candidate(emb, q, circle_crop(img, face.bbox), pid, score))
        if not cands:
            return None

        # --- vote ---
        thr = self.cfg.match_threshold
        weights: dict[int, float] = {}
        for c in cands:
            if c.person_id is not None and c.score >= thr:
                weights[c.person_id] = weights.get(c.person_id, 0.0) + c.score * c.quality
        unknown_w = sum(c.quality for c in cands if c.person_id is None or c.score < thr) * thr

        if weights and max(weights.values()) >= unknown_w:
            winner = max(weights, key=weights.get)
            mine = [c for c in cands if c.person_id == winner and c.score >= thr]
            best = max(mine, key=lambda c: c.score * c.quality)
            return FaceResult(winner, best.score, best.embedding, best.crop_png, winner, len(cands))

        best = max(cands, key=lambda c: c.quality)
        return FaceResult(None, best.score, best.embedding, best.crop_png, best.person_id, len(cands))

    def embed_photo(self, jpg: bytes) -> tuple[np.ndarray, bytes] | None:
        """Enrol from an uploaded photo: largest valid face -> (embedding, circular crop)."""
        self.ensure_loaded()
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = [f for f in self._app.get(img) if f.det_score >= 0.5]
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return f.normed_embedding.astype(np.float32), circle_crop(img, f.bbox)
