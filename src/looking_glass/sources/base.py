"""Abstract base for video sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Frame:
    """A single video frame with metadata."""

    camera_id: str
    timestamp: float
    image: npt.NDArray[np.uint8]  # HWC, BGR
    frame_idx: int = 0


class VideoSource(ABC):
    """Abstract base class for video sources."""

    @abstractmethod
    def frames(self) -> list[Frame]:
        """Return all frames from this source."""
        ...

    @abstractmethod
    def camera_id(self) -> str:
        """Return the camera identifier."""
        ...
