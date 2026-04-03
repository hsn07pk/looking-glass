"""Alert engine — register NL queries and fire when similarity exceeds threshold."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class AlertRule:
    """A registered alert rule."""
    id: str
    query: str
    embedding: npt.NDArray[np.float32]
    threshold: float = 0.07
    camera_filter: str | None = None
    created_at: float = 0.0


@dataclass
class FiredAlert:
    """A fired alert event."""
    rule_id: str
    query: str
    camera_id: str
    timestamp: float
    score: float
    frame_path: str = ""


@dataclass
class AlertEngine:
    """Manages alert rules and checks frames against them."""

    rules: dict[str, AlertRule] = field(default_factory=dict)
    _cooldowns: dict[str, float] = field(default_factory=dict)
    cooldown_sec: float = 5.0

    def register(self, query: str, threshold: float = 0.25,
                 camera_filter: str | None = None,
                 embedder: object | None = None) -> str:
        """Register an alert rule. Returns the rule ID."""
        rule_id = str(uuid.uuid4())[:8]

        if embedder is None:
            from looking_glass.models.embedder import FrameEmbedder
            embedder = FrameEmbedder()

        embedding = embedder.encode_text(query)  # type: ignore[union-attr]

        rule = AlertRule(
            id=rule_id,
            query=query,
            embedding=embedding,
            threshold=threshold,
            camera_filter=camera_filter,
            created_at=time.time(),
        )
        self.rules[rule_id] = rule
        return rule_id

    def remove(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        return self.rules.pop(rule_id, None) is not None

    def list_rules(self) -> list[dict]:
        """List all active rules."""
        return [
            {"id": r.id, "query": r.query, "threshold": r.threshold,
             "camera_filter": r.camera_filter}
            for r in self.rules.values()
        ]

    def check_frame(
        self,
        frame_embedding: npt.NDArray[np.float32],
        camera_id: str,
        timestamp: float,
        frame_path: str = "",
    ) -> list[FiredAlert]:
        """Check a frame embedding against all active rules."""
        fired = []
        now = time.time()

        for rule in self.rules.values():
            # Camera filter
            if rule.camera_filter and rule.camera_filter != camera_id:
                continue

            # Cooldown
            last_fire = self._cooldowns.get(rule.id, 0.0)
            if now - last_fire < self.cooldown_sec:
                continue

            # Cosine similarity
            sim = float(np.dot(frame_embedding, rule.embedding))
            if sim >= rule.threshold:
                fired.append(FiredAlert(
                    rule_id=rule.id,
                    query=rule.query,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    score=sim,
                    frame_path=frame_path,
                ))
                self._cooldowns[rule.id] = now

        return fired
