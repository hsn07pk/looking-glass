"""Webcam video source stub — extensibility seam."""

from __future__ import annotations

from looking_glass.sources.base import Frame, VideoSource


class WebcamSource(VideoSource):
    """Stub for live webcam input. Not implemented for demo."""

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id

    def camera_id(self) -> str:
        return f"webcam{self._device_id}"

    def frames(self) -> list[Frame]:
        raise NotImplementedError("WebcamSource is a stub for future extensibility.")
