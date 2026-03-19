"""Qdrant vector store for frames and crops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class QdrantStore:
    """Qdrant embedded-mode vector store."""

    path: str = ""
    vector_dim: int = 768
    _client: QdrantClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.path:
            from looking_glass.config import QDRANT_DIR
            self.path = str(QDRANT_DIR)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=self.path)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        for name in ("frames", "crops"):
            if not self._client.collection_exists(name):  # type: ignore[union-attr]
                self._client.create_collection(  # type: ignore[union-attr]
                    name,
                    vectors_config=VectorParams(
                        size=self.vector_dim, distance=Distance.COSINE
                    ),
                )

    def upsert_frame(
        self, frame_id: str, vector: npt.NDArray[np.float32], payload: dict
    ) -> None:
        """Upsert a frame embedding."""
        self._client.upsert(  # type: ignore[union-attr]
            "frames",
            points=[
                PointStruct(
                    id=self._str_to_int_id(frame_id),
                    vector=vector.tolist(),
                    payload={**payload, "frame_id": frame_id},
                )
            ],
        )

    def upsert_crop(
        self, crop_id: str, vector: npt.NDArray[np.float32], payload: dict
    ) -> None:
        """Upsert a crop embedding."""
        self._client.upsert(  # type: ignore[union-attr]
            "crops",
            points=[
                PointStruct(
                    id=self._str_to_int_id(crop_id),
                    vector=vector.tolist(),
                    payload={**payload, "crop_id": crop_id},
                )
            ],
        )

    def search_frames(
        self, query_vec: npt.NDArray[np.float32], top_k: int = 20
    ) -> list[dict]:
        """Search frames by vector similarity."""
        results = self._client.query_points(  # type: ignore[union-attr]
            "frames",
            query=query_vec.tolist(),
            limit=top_k,
        )
        return [
            {**p.payload, "score": p.score}
            for p in results.points
            if p.payload is not None
        ]

    def search_crops(
        self, query_vec: npt.NDArray[np.float32], top_k: int = 20
    ) -> list[dict]:
        """Search crops by vector similarity."""
        results = self._client.query_points(  # type: ignore[union-attr]
            "crops",
            query=query_vec.tolist(),
            limit=top_k,
        )
        return [
            {**p.payload, "score": p.score}
            for p in results.points
            if p.payload is not None
        ]

    def count(self, collection: str = "frames") -> int:
        """Get count of points in a collection."""
        info = self._client.get_collection(collection)  # type: ignore[union-attr]
        return info.points_count or 0

    @staticmethod
    def _str_to_int_id(s: str) -> int:
        """Convert string ID to a positive integer for Qdrant."""
        return abs(hash(s)) % (2**63)
