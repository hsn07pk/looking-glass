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
    items = []
    for r in results:
        dets = [
            Detection(
                bbox=d.get("bbox") or d.get("bboxes"),
                class_name=d.get("class", d.get("class_name", "")),
                score=d.get("score", 0.0),
                track_id=d.get("track_id"),
            )
            for d in (r.detections or [])
        ]
        items.append(SearchResultItem(
            camera_id=r.camera_id,
            timestamp=r.timestamp,
            score=r.score,
            frame_path=r.frame_path,
            detections=dets,
            caption=r.caption,
        ))
    return SearchResponse(results=items, query=req.q, total=len(items))
