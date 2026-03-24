"""Dependency injection for FastAPI — singleton model instances."""

from __future__ import annotations

from functools import lru_cache

from looking_glass.alerts.engine import AlertEngine
from looking_glass.analytics.counter import AnalyticsRouter, PeopleCounter
from looking_glass.models.embedder import FrameEmbedder
from looking_glass.search.nl_search import NLSearch
from looking_glass.store.vector_store import QdrantStore


@lru_cache(maxsize=1)
def get_embedder() -> FrameEmbedder:
    return FrameEmbedder()


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore()


@lru_cache(maxsize=1)
def get_search() -> NLSearch:
    return NLSearch(embedder=get_embedder(), store=get_store())


@lru_cache(maxsize=1)
def get_alert_engine() -> AlertEngine:
    return AlertEngine()


@lru_cache(maxsize=1)
def get_analytics() -> AnalyticsRouter:
    return AnalyticsRouter(counter=PeopleCounter())
