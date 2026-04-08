from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from ultralytics import YOLOWorld

from looking_glass.config import OPEN_VOCAB_SET


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class OpenVocabDetector:

    model_name: str = "yolov8m-worldv2.pt"
    conf_threshold: float = 0.40
    iou_threshold: float = 0.5
    _model: YOLOWorld | None = field(default=None, repr=False)
    _classes_set: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._model = YOLOWorld(self.model_name)
        self._model.to(_device())

    def detect(
        self, image: npt.NDArray[np.uint8], classes: list[str] | None = None,
    ) -> list[dict]:
        assert self._model is not None

        target_classes = classes if classes else OPEN_VOCAB_SET
        if target_classes != self._classes_set:
            self._model.set_classes(target_classes)
            self._classes_set = list(target_classes)

        results = self._model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[dict] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            score = boxes.conf[i].item()
            cls_id = int(boxes.cls[i].item())
            class_name = result.names[cls_id]
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "score": round(score, 4),
            })
        return detections
