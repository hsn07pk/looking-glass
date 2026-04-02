"""ByteTrack multi-object tracker via supervision."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import supervision as sv


@dataclass
class MultiObjectTracker:
    """Wraps supervision.ByteTrack for frame-to-frame tracking."""

    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 25
    _tracker: sv.ByteTrack | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate,
        )

    def update(
        self, detections: list[dict], frame: npt.NDArray[np.uint8],
    ) -> list[dict]:
        """Update tracker with new detections.

        Input dicts must have: bbox (x1,y1,x2,y2), class_name, score.
        Returns same dicts with added track_id.
        """
        if not detections:
            return []

        assert self._tracker is not None

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["score"] for d in detections], dtype=np.float32)

        # Map class names to integer IDs for supervision
        class_names = [d["class_name"] for d in detections]
        unique_classes = sorted(set(class_names))
        name_to_id = {name: idx for idx, name in enumerate(unique_classes)}
        class_id = np.array([name_to_id[n] for n in class_names], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        results: list[dict] = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            cls_name = unique_classes[cid] if cid < len(unique_classes) else "unknown"
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            results.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": cls_name,
                "score": round(score, 4),
                "track_id": tid,
            })
        return results

    def reset(self) -> None:
        """Reset tracker state between videos."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate,
        )
