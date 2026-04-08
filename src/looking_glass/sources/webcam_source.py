from __future__ import annotations

from looking_glass.sources.base import Frame, VideoSource


class WebcamSource(VideoSource):

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id

    def camera_id(self) -> str:
        return f"webcam{self._device_id}"

    def frames(self) -> list[Frame]:
        raise NotImplementedError
