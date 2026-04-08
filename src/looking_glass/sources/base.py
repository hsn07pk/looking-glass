from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Frame:

    camera_id: str
    timestamp: float
    image: npt.NDArray[np.uint8]  # HWC, BGR
    frame_idx: int = 0


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> list[Frame]: ...

    @abstractmethod
    def camera_id(self) -> str: ...
