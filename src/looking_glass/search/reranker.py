"""Reranker using Florence-2 grounding to verify search results."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from looking_glass.models.captioner import DenseCaptioner
from looking_glass.search.nl_search import SearchResult


@dataclass
class Reranker:
    """Rerank search results using Florence-2 grounded detection."""

    captioner: DenseCaptioner

    def rerank(self, results: list[SearchResult], query: str, top_k: int = 5) -> list[SearchResult]:
        """Rerank results by grounding the query on each frame."""
        scored = []
        for r in results[:top_k * 2]:  # Check more than we need
            try:
                img = cv2.imread(r.frame_path)
                if img is None:
                    scored.append((r, r.score))
                    continue
                groundings = self.captioner.grounded_detection(img, query)
                # Boost score if grounding finds something
                boost = min(len(groundings) * 0.1, 0.3)
                scored.append((r, r.score + boost))
            except Exception:
                scored.append((r, r.score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]
