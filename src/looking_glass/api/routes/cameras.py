"""Cameras route — list available cameras."""

from __future__ import annotations

from fastapi import APIRouter

from looking_glass.api.schemas import CameraInfo
from looking_glass.config import NORMALIZED_DIR

router = APIRouter()


@router.get("/cameras", response_model=list[CameraInfo])
async def list_cameras() -> list[CameraInfo]:
    """List all available camera feeds."""
    clips = sorted(NORMALIZED_DIR.glob("*.mp4"))
    return [
        CameraInfo(
            camera_id=c.stem.split("_")[0],
            clip_name=c.name,
        )
        for c in clips
    ]
