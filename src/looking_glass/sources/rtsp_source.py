from __future__ import annotations

from looking_glass.sources.base import Frame, VideoSource


class RTSPSource(VideoSource):

    def __init__(self, url: str = "") -> None:
        self._url = url

    def camera_id(self) -> str:
        return "rtsp0"

    def frames(self) -> list[Frame]:
        raise NotImplementedError
