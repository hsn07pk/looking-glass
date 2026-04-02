"""Search route."""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from looking_glass.api.deps import get_search
from looking_glass.api.schemas import (
    Detection,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Natural language search over video frames."""
    try:
        nl_search = get_search()
        results = nl_search.search(req.q, top_k=req.top_k)
    except Exception as exc:
        logger.error(f"Search failed for query '{req.q}': {exc}")
        return SearchResponse(results=[], query=req.q, total=0)
    # Filter noisy detections for clean display
    _NOISY_CLASSES = {"phone", "smartphone", "camera"}
    _NOISY_LABELS = {
        "one hand", "his hands", "her hands", "their hands", "hand", "hands",
        "the table", "a table", "table", "ceiling", "wall", "floor",
        "The office", "office", "object", "objects", "various objects",
    }

    items = []
    for r in results:
        dets = [
            Detection(
                bbox=d.get("bbox") or d.get("bboxes"),
                class_name=d.get("class_name", d.get("class", "")),
                score=d.get("score", 0.0),
                track_id=d.get("track_id"),
            )
            for d in (r.detections or [])
            if d.get("class_name", d.get("class", "")) not in _NOISY_CLASSES
            and d.get("class_name", d.get("class", "")) not in _NOISY_LABELS
        ]
        # Limit to top 5 detections per frame for clean display
        dets = dets[:5]
        items.append(SearchResultItem(
            camera_id=r.camera_id,
            timestamp=r.timestamp,
            score=r.score,
            frame_path=r.frame_path,
            detections=dets,
            caption=r.caption,
        ))
    return SearchResponse(results=items, query=req.q, total=len(items))
