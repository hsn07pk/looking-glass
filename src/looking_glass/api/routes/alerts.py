"""Alerts routes — CRUD + WebSocket."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect

from looking_glass.api.deps import get_alert_engine, get_embedder, get_store
from looking_glass.api.schemas import AlertRuleRequest, AlertRuleResponse

router = APIRouter()

# Connected WebSocket clients
_ws_clients: list[WebSocket] = []


def _scan_frames_for_rule_sync(rule_id: str) -> list[dict]:
    """Scan stored frame embeddings against a newly registered alert rule (sync)."""
    engine = get_alert_engine()
    store = get_store()
    rule = engine.rules.get(rule_id)
    if not rule:
        return []

    hits = store.search_frames(rule.embedding, top_k=50)
    alerts = []
    seen_cameras: set[str] = set()
    for hit in hits:
        score = hit.get("score", 0.0)
        cam = hit.get("camera_id", "")
        if score >= rule.threshold and cam not in seen_cameras:
            seen_cameras.add(cam)
            alerts.append({
                "rule_id": rule.id,
                "query": rule.query,
                "camera_id": cam,
                "timestamp": hit.get("timestamp", 0.0),
                "score": round(score, 4),
            })
    return alerts


@router.post("/alerts/rules", response_model=AlertRuleResponse)
async def create_alert_rule(req: AlertRuleRequest) -> AlertRuleResponse:
    """Register a new alert rule and scan existing frames."""
    engine = get_alert_engine()
    embedder = get_embedder()
    rule_id = engine.register(
        query=req.q,
        threshold=req.threshold,
        camera_filter=req.camera_filter,
        embedder=embedder,
    )
    rule = engine.rules[rule_id]

    # Scan existing frames and broadcast alerts via WebSocket
    fired = _scan_frames_for_rule_sync(rule_id)
    for alert in fired:
        await broadcast_alert(alert)

    return AlertRuleResponse(
        id=rule.id, query=rule.query,
        threshold=rule.threshold, camera_filter=rule.camera_filter,
    )


@router.get("/alerts/rules")
async def list_alert_rules() -> list[dict]:
    """List all active alert rules."""
    return get_alert_engine().list_rules()


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str) -> dict:
    """Delete an alert rule."""
    removed = get_alert_engine().remove(rule_id)
    return {"removed": removed}


@router.websocket("/alerts/ws")
async def alerts_websocket(ws: WebSocket) -> None:
    """WebSocket for live alert notifications."""
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            # Keep connection alive, client can send heartbeats
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.remove(ws)


async def broadcast_alert(alert: dict) -> None:
    """Send an alert to all connected WebSocket clients."""
    message = json.dumps(alert)
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)
