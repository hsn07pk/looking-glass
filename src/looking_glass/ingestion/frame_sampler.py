from __future__ import annotations

from looking_glass.sources.base import Frame


class FrameSampler:

    def __init__(self, target_fps: float = 1.0) -> None:
        self.target_fps = target_fps

    def sample(self, frames: list[Frame], source_fps: float = 25.0) -> list[Frame]:
        if not frames:
            return []
        step = max(1, int(source_fps / self.target_fps))
        return frames[::step]
