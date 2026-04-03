"""Pydantic v2 schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel


class SearchRequest(BaseModel):
    q: str
    top_k: int = 10


class Detection(BaseModel):
    bbox: list[float] | None = None
    class_name: str = ""
    score: float = 0.0
    track_id: int | None = None


class SearchResultItem(BaseModel):
    camera_id: str
    timestamp: float
    score: float
    frame_path: str
    detections: list[Detection] = []
    caption: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    total: int


class AlertRuleRequest(BaseModel):
    q: str
    threshold: float = 0.07
    camera_filter: str | None = None


class AlertRuleResponse(BaseModel):
    id: str
    query: str
    threshold: float
    camera_filter: str | None = None


class AnalyticsRequest(BaseModel):
    q: str


class AnalyticsResponse(BaseModel):
    answer: str
    query: str


class CameraInfo(BaseModel):
    camera_id: str
    clip_name: str
    frame_count: int = 0


class HealthResponse(BaseModel):
    ok: bool
    models_loaded: bool
    frame_count: int
