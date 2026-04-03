"""Ingestion pipeline — orchestrates the full video processing chain."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from looking_glass.config import DATA_DIR
from looking_glass.sources.base import Frame


class Detector(Protocol):
    def detect(self, image: npt.NDArray[np.uint8], classes: list[str]) -> list[dict]: ...


class Tracker(Protocol):
    def update(self, detections: list[dict], frame: npt.NDArray[np.uint8]) -> list[dict]: ...
    def reset(self) -> None: ...


class Embedder(Protocol):
    def encode_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]: ...
    def encode_text(self, text: str) -> npt.NDArray[np.float32]: ...
    def dim(self) -> int: ...


class Captioner(Protocol):
    def caption(self, image: npt.NDArray[np.uint8]) -> str: ...
    def exhaustive_caption(self, image: npt.NDArray[np.uint8]) -> str: ...
    def grounded_detection(self, image: npt.NDArray[np.uint8], text: str) -> list[dict]: ...


class VectorStore(Protocol):
    def upsert_frame(
        self, frame_id: str, vector: npt.NDArray[np.float32], payload: dict
    ) -> None: ...
    def upsert_crop(
        self, crop_id: str, vector: npt.NDArray[np.float32], payload: dict
    ) -> None: ...



DB_PATH = DATA_DIR / "tracks.db"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database for tracks."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            track_id INTEGER,
            camera_id TEXT,
            class_name TEXT,
            first_seen REAL,
            last_seen REAL,
            frame_count INTEGER DEFAULT 1,
            PRIMARY KEY (track_id, camera_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            camera_id TEXT,
            timestamp REAL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            class_name TEXT,
            score REAL,
            frame_path TEXT
        )
    """)
    conn.commit()
    return conn


@dataclass
class IngestionPipeline:
    """Orchestrates the full ingestion pipeline."""

    detector: Detector
    tracker: Tracker
    embedder: Embedder
    captioner: Captioner
    store: VectorStore
    target_fps: float = 1.0
    classes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.classes:
            from looking_glass.config import OPEN_VOCAB_SET
            self.classes = list(OPEN_VOCAB_SET)

    def ingest_frames(self, camera_id: str, frames: list[Frame]) -> dict:
        """Run the full pipeline on a list of frames. Returns stats."""
        from looking_glass.ingestion.frame_sampler import FrameSampler

        sampler = FrameSampler(self.target_fps)
        src_fps = 25.0
        if len(frames) >= 2:
            dt = frames[1].timestamp - frames[0].timestamp
            if dt > 0:
                src_fps = 1.0 / dt

        sampled = sampler.sample(frames, src_fps)
        self.tracker.reset()

        conn = init_db()
        stats = {"frames_total": len(frames), "frames_sampled": len(sampled),
                 "detections": 0, "tracks": set(), "captions": 0}
        frames_dir = DATA_DIR / "frames" / camera_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        for frame in sampled:
            # Save frame image
            frame_id = f"{camera_id}_f{frame.frame_idx:06d}"
            frame_path = frames_dir / f"{frame_id}.jpg"
            if not frame_path.exists():
                import cv2
                cv2.imwrite(str(frame_path), frame.image)

            # Detect
            dets = self.detector.detect(frame.image, self.classes)
            stats["detections"] += len(dets)

            # Track
            tracked = self.tracker.update(dets, frame.image)
            for t in tracked:
                tid = t.get("track_id", 0)
                stats["tracks"].add(tid)
                # SQLite
                sql_tracks = (
                    "INSERT OR REPLACE INTO tracks"
                    " (track_id, camera_id, class_name,"
                    " first_seen, last_seen, frame_count)"
                    " VALUES (?, ?, ?, ?, ?,"
                    " COALESCE((SELECT frame_count FROM tracks"
                    " WHERE track_id=? AND camera_id=?), 0)+1)"
                )
                conn.execute(
                    sql_tracks,
                    (
                        tid, camera_id,
                        t.get("class_name", ""),
                        frame.timestamp, frame.timestamp,
                        tid, camera_id,
                    ),
                )
                bbox = t.get("bbox", (0, 0, 0, 0))
                sql_dets = (
                    "INSERT INTO detections"
                    " (track_id, camera_id, timestamp,"
                    " x1, y1, x2, y2,"
                    " class_name, score, frame_path)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                conn.execute(
                    sql_dets,
                    (
                        tid, camera_id, frame.timestamp,
                        *bbox,
                        t.get("class_name", ""),
                        t.get("score", 0.0),
                        str(frame_path),
                    ),
                )

            # Short caption for grounded detection (Florence-2)
            short_cap = self.captioner.caption(frame.image)

            # Exhaustive caption via Ollama VLM for rich queryable descriptions
            cap = self.captioner.exhaustive_caption(frame.image)
            stats["captions"] += 1

            # Florence-2 grounded detection: find bboxes for everything in the caption.
            # This automatically detects all objects/people/actions without a fixed class list.
            grounded_dets: list[dict] = []
            if short_cap:
                try:
                    raw_gd = self.captioner.grounded_detection(frame.image, short_cap)
                    grounded_dets = [
                        {
                            "bbox": list(g["bbox"]),
                            "class_name": g.get("label", ""),
                            "score": 1.0,  # grounded detections don't have confidence
                        }
                        for g in raw_gd
                        if g.get("bbox")
                    ]
                except Exception:
                    pass  # graceful fallback if grounding fails

            # Merge: YOLO-World tracked detections + Florence-2 grounded detections
            # YOLO provides tracked objects with IDs; Florence provides caption-grounded bboxes
            all_detections = [
                {
                    "bbox": t.get("bbox"),
                    "class_name": t.get("class_name"),
                    "score": t.get("score"),
                    "track_id": t.get("track_id"),
                }
                for t in tracked
            ] + grounded_dets

            # Embed frame
            frame_vec = self.embedder.encode_image(frame.image)
            self.store.upsert_frame(frame_id, frame_vec, {
                "camera_id": camera_id,
                "timestamp": frame.timestamp,
                "frame_idx": frame.frame_idx,
                "frame_path": str(frame_path),
                "caption": cap,
                "detections": json.dumps(all_detections),
            })

            # Embed crops from tracked objects (YOLO-World)
            for t in tracked:
                bbox = t.get("bbox", (0, 0, 0, 0))
                x1, y1, x2, y2 = [int(c) for c in bbox]
                h, w = frame.image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame.image[y1:y2, x1:x2]
                    crop_vec = self.embedder.encode_image(crop)
                    crop_id = f"{frame_id}_t{t.get('track_id', 0)}"
                    self.store.upsert_crop(crop_id, crop_vec, {
                        "camera_id": camera_id,
                        "timestamp": frame.timestamp,
                        "frame_path": str(frame_path),
                        "bbox": list(bbox),
                        "class_name": t.get("class_name", ""),
                        "track_id": t.get("track_id", 0),
                    })

        conn.commit()
        conn.close()
        stats["tracks"] = len(stats["tracks"])
        return stats
