from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_settings: dict = {
    "show_bboxes": True,
    "show_captions": True,
    "search_top_k": 10,
    "alert_threshold": 0.07,
    "bbox_min_confidence": 0.35,
    "vision_model": "minicpm-v",
    "llm_model": "llama3.2:3b",
    "detection_classes": "auto",
}


class SettingsUpdate(BaseModel):
    key: str
    value: str | int | float | bool


@router.get("/settings")
async def get_settings() -> dict:
    return _settings


@router.post("/settings")
async def update_setting(req: SettingsUpdate) -> dict:
    if req.key in _settings:
        _settings[req.key] = req.value
        return {"ok": True, "key": req.key, "value": req.value}
    return {"ok": False, "error": f"Unknown setting: {req.key}"}
