"""Ingest all normalized video clips into the vector store."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from looking_glass.config import NORMALIZED_DIR
from looking_glass.sources.file_source import FileVideoSource


def main(mock: bool = False) -> None:
    if mock:
        from looking_glass.ingestion.pipeline import (
            IngestionPipeline,
            MockCaptioner,
            MockDetector,
            MockEmbedder,
            MockTracker,
            MockVectorStore,
        )

        pipeline = IngestionPipeline(
            detector=MockDetector(),
            tracker=MockTracker(),
            embedder=MockEmbedder(),
            captioner=MockCaptioner(),
            store=MockVectorStore(),
        )
    else:
        from looking_glass.models.detector import OpenVocabDetector
        from looking_glass.models.tracker import MultiObjectTracker
        from looking_glass.models.embedder import FrameEmbedder
        from looking_glass.models.captioner import DenseCaptioner
        from looking_glass.store.vector_store import QdrantStore

        store = QdrantStore()
        pipeline = IngestionPipeline(
            detector=OpenVocabDetector(),
            tracker=MultiObjectTracker(),
            embedder=FrameEmbedder(),
            captioner=DenseCaptioner(),
            store=store,
        )

    clips = sorted(NORMALIZED_DIR.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {NORMALIZED_DIR}")
        return

    print(f"Found {len(clips)} clips to ingest.")
    total_start = time.time()

    for clip in clips:
        print(f"\nIngesting {clip.name}...")
        source = FileVideoSource(clip)
        frames = source.frames()
        stats = pipeline.ingest_frames(source.camera_id(), frames)
        print(f"  {stats['frames_sampled']}/{stats['frames_total']} frames, "
              f"{stats['detections']} detections, {stats['tracks']} tracks")

    elapsed = time.time() - total_start
    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    use_mock = "--mock" in sys.argv
    main(mock=use_mock)
