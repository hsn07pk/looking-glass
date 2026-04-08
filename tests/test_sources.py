import numpy as np
import pytest

from looking_glass.sources.base import Frame, VideoSource
from looking_glass.sources.file_source import FileVideoSource
from looking_glass.sources.webcam_source import WebcamSource
from looking_glass.sources.rtsp_source import RTSPSource
from looking_glass.ingestion.frame_sampler import FrameSampler


def _make_frames(n: int, camera_id: str = "cam01", fps: float = 25.0) -> list[Frame]:
    return [
        Frame(
            camera_id=camera_id,
            timestamp=i / fps,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            frame_idx=i,
        )
        for i in range(n)
    ]


class TestFrameSampler:
    def test_sample_25fps_to_1fps(self):
        frames = _make_frames(50, fps=25.0)  # 2 seconds
        sampler = FrameSampler(target_fps=1.0)
        sampled = sampler.sample(frames, source_fps=25.0)
        assert len(sampled) == 2  # ~1 per second

    def test_sample_empty(self):
        sampler = FrameSampler(target_fps=1.0)
        assert sampler.sample([], source_fps=25.0) == []

    def test_sample_preserves_metadata(self):
        frames = _make_frames(25, fps=25.0)
        sampler = FrameSampler(target_fps=1.0)
        sampled = sampler.sample(frames, source_fps=25.0)
        assert all(f.camera_id == "cam01" for f in sampled)


class TestStubs:
    def test_webcam_stub_raises(self):
        src = WebcamSource()
        assert src.camera_id() == "webcam0"
        with pytest.raises(NotImplementedError):
            src.frames()

    def test_rtsp_stub_raises(self):
        src = RTSPSource()
        assert src.camera_id() == "rtsp0"
        with pytest.raises(NotImplementedError):
            src.frames()
