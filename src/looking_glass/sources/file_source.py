from __future__ import annotations

from pathlib import Path

import cv2

from looking_glass.sources.base import Frame, VideoSource


class FileVideoSource(VideoSource):

    def __init__(
        self, path: str | Path, fps: float = 25.0, sample_fps: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self._camera_id = self.path.stem.split("_")[0]
        self._fps = fps
        self._sample_fps = sample_fps  # 0 = read all, >0 = subsample at read time

    def camera_id(self) -> str:
        return self._camera_id

    def frames(self) -> list[Frame]:
        result: list[Frame] = []
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or self._fps
        step = max(1, int(src_fps / self._sample_fps)) if self._sample_fps > 0 else 1
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
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
