from __future__ import annotations

from pydantic import BaseModel


class FramePayload(BaseModel):
    camera_id: str
    timestamp: float
    frame_idx: int
    frame_path: str
    caption: str = ""
    detections: str = "[]"  # JSON string


class CropPayload(BaseModel):
    camera_id: str
    timestamp: float
    frame_path: str
    bbox: list[float]
    class_name: str = ""
    track_id: int = 0
