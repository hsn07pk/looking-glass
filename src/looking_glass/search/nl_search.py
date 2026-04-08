from __future__ import annotations

import json
from dataclasses import dataclass, field

from looking_glass.models.embedder import FrameEmbedder
from looking_glass.store.vector_store import QdrantStore


@dataclass
class SearchResult:
    camera_id: str
    timestamp: float
    score: float
    frame_path: str
    detections: list[dict] = field(default_factory=list)
    caption: str = ""


@dataclass
class NLSearch:

    embedder: FrameEmbedder
    store: QdrantStore

    @classmethod
    def from_default_config(cls) -> NLSearch:
        return cls(
            embedder=FrameEmbedder(),
            store=QdrantStore(),
        )

    @staticmethod
    def _normalize_score(raw: float) -> float:
        """Rescale cosine similarity to 0-100% for display."""
        low, high = 0.04, 0.15
        normalized = (raw - low) / (high - low)
        return max(0.01, min(0.99, normalized))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_vec = self.embedder.encode_text(query)
        frame_hits = self.store.search_frames(query_vec, top_k=max(top_k * 5, 50))
        crop_hits = self.store.search_crops(query_vec, top_k=top_k * 3)

        frame_index: dict[str, dict] = {}
        for hit in frame_hits:
            path = hit.get("frame_path", "")
            if path and (path not in frame_index or hit.get("score", 0) > frame_index[path].get("score", 0)):
                frame_index[path] = hit

        seen: dict[str, dict] = dict(frame_index)
        for hit in crop_hits:
            path = hit.get("frame_path", "")
            if not path:
                continue
            frame_meta = frame_index.get(path)
            if path not in seen:
                if frame_meta:
                    hit["detections"] = frame_meta.get("detections", "[]")
                    hit["caption"] = frame_meta.get("caption", "")
                seen[path] = hit
            elif hit.get("score", 0) > seen[path].get("score", 0):
                prev = seen[path]
                hit["detections"] = prev.get("detections", hit.get("detections", "[]"))
                hit["caption"] = prev.get("caption", hit.get("caption", ""))
                seen[path] = hit

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
                score=self._normalize_score(hit.get("score", 0.0)),
                frame_path=hit.get("frame_path", ""),
                detections=dets,
                caption=hit.get("caption", ""),
            ))
        return results
