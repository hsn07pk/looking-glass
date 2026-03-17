"""Natural language search — the core query engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from looking_glass.models.embedder import FrameEmbedder
from looking_glass.store.vector_store import QdrantStore


@dataclass
class SearchResult:
    """A single search result."""
    camera_id: str
    timestamp: float
    score: float
    frame_path: str
    detections: list[dict] = field(default_factory=list)
    caption: str = ""


@dataclass
class NLSearch:
    """End-to-end natural language search over video frames."""

    embedder: FrameEmbedder
    store: QdrantStore

    @classmethod
    def from_default_config(cls) -> NLSearch:
        """Create with default configuration."""
        return cls(
            embedder=FrameEmbedder(),
            store=QdrantStore(),
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for frames matching a natural language query."""
        query_vec = self.embedder.encode_text(query)

        # Search both frames and crops
        frame_hits = self.store.search_frames(query_vec, top_k=top_k * 2)
        crop_hits = self.store.search_crops(query_vec, top_k=top_k * 2)

        # Merge and deduplicate by frame_path
        seen: dict[str, dict] = {}
        for hit in frame_hits + crop_hits:
            path = hit.get("frame_path", "")
            if path not in seen or hit.get("score", 0) > seen[path].get("score", 0):
                seen[path] = hit

        # Sort by score and take top_k
        ranked = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        results = []
        for hit in ranked:
            dets_raw = hit.get("detections", "[]")
            if isinstance(dets_raw, str):
                try:
                    dets = json.loads(dets_raw)
                except json.JSONDecodeError:
                    dets = []
            else:
                dets = dets_raw if isinstance(dets_raw, list) else []

            results.append(SearchResult(
                camera_id=hit.get("camera_id", ""),
                timestamp=hit.get("timestamp", 0.0),
                score=hit.get("score", 0.0),
                frame_path=hit.get("frame_path", ""),
                detections=dets,
                caption=hit.get("caption", ""),
            ))
        return results
