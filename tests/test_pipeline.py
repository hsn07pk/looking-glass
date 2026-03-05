"""Tests for the ingestion pipeline with mocks."""

import numpy as np

from looking_glass.sources.base import Frame
from looking_glass.ingestion.pipeline import (
    IngestionPipeline,
    MockCaptioner,
    MockDetector,
    MockEmbedder,
    MockTracker,
    MockVectorStore,
)


def _make_frames(n: int = 50) -> list[Frame]:
    return [
        Frame(
            camera_id="cam01",
            timestamp=i / 25.0,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_idx=i,
        )
        for i in range(n)
    ]


class TestMockPipeline:
    def test_ingest_mock(self):
        store = MockVectorStore()
        pipeline = IngestionPipeline(
            detector=MockDetector(),
            tracker=MockTracker(),
            embedder=MockEmbedder(),
            captioner=MockCaptioner(),
            store=store,
        )
        frames = _make_frames(50)
        stats = pipeline.ingest_frames("cam01", frames)
        assert stats["frames_total"] == 50
        assert stats["frames_sampled"] > 0
        assert stats["detections"] > 0
        assert len(store.frames) > 0
