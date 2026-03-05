"""RTSP video source stub — extensibility seam."""

from __future__ import annotations

from looking_glass.sources.base import Frame, VideoSource


class RTSPSource(VideoSource):
    """Stub for RTSP stream input. Not implemented for demo."""

    def __init__(self, url: str = "") -> None:
        self._url = url

    def camera_id(self) -> str:
        return "rtsp0"

    def frames(self) -> list[Frame]:
        raise NotImplementedError("RTSPSource is a stub for future extensibility.")
