"""File-based video source using decord."""

from __future__ import annotations

from pathlib import Path

import cv2

from looking_glass.sources.base import Frame, VideoSource


class FileVideoSource(VideoSource):
    """Reads an MP4 file and yields frames."""

    def __init__(self, path: str | Path, fps: float = 25.0) -> None:
        self.path = Path(path)
        self._camera_id = self.path.stem.split("_")[0]
        self._fps = fps

    def camera_id(self) -> str:
        return self._camera_id

    def frames(self) -> list[Frame]:
        """Read all frames from the video file using OpenCV."""
        result: list[Frame] = []
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or self._fps
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = idx / src_fps
            result.append(
                Frame(
                    camera_id=self._camera_id,
                    timestamp=timestamp,
                    image=frame,
                    frame_idx=idx,
                )
            )
            idx += 1
        cap.release()
        return result
