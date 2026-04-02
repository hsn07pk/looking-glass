"""Cameras route — list available cameras and tracking data."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from looking_glass.api.schemas import CameraInfo
from looking_glass.config import DATA_DIR, NORMALIZED_DIR

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


@router.get("/cameras/{camera_id}/tracks")
async def get_camera_tracks(camera_id: str) -> list[dict]:
    """Get all tracked detections for a camera, sorted by timestamp.

    Returns a list of {track_id, timestamp, x1, y1, x2, y2, class_name, score}
    used for interpolating bounding boxes during video playback.
    """
    db_path = DATA_DIR / "tracks.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT track_id, timestamp, x1, y1, x2, y2, class_name, score "
        "FROM detections WHERE camera_id=? ORDER BY timestamp",
        (camera_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
