"""Analytics route — question answering over video data."""

from __future__ import annotations

from fastapi import APIRouter

from looking_glass.api.deps import get_analytics
from looking_glass.api.schemas import AnalyticsRequest, AnalyticsResponse

router = APIRouter()


@router.post("/analytics/ask", response_model=AnalyticsResponse)
async def ask_analytics(req: AnalyticsRequest) -> AnalyticsResponse:
    """Answer an analytics question about the video data."""
    analytics = get_analytics()
    answer = analytics.answer(req.q)
    return AnalyticsResponse(answer=answer, query=req.q)
