"""People counter and analytics engine using track data + Ollama."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from looking_glass.config import DATA_DIR


@dataclass
class PeopleCounter:
    """Count unique people (tracks) from SQLite."""

    db_path: str = ""

    def __post_init__(self) -> None:
        if not self.db_path:
            self.db_path = str(DATA_DIR / "tracks.db")

    def count(self, camera: str | None = None, class_name: str = "person") -> int:
        """Count distinct track IDs, optionally filtered by camera."""
        conn = sqlite3.connect(self.db_path)
        if camera:
            row = conn.execute(
                "SELECT COUNT(DISTINCT track_id) FROM tracks WHERE camera_id=? AND class_name=?",
                (camera, class_name),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(DISTINCT track_id) FROM tracks WHERE class_name=?",
                (class_name,),
            ).fetchone()
        conn.close()
        return row[0] if row else 0

    def count_all(self, camera: str | None = None) -> int:
        """Count all distinct tracks regardless of class."""
        conn = sqlite3.connect(self.db_path)
        if camera:
            row = conn.execute(
                "SELECT COUNT(DISTINCT track_id) FROM tracks WHERE camera_id=?",
                (camera,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(DISTINCT track_id) FROM tracks",
            ).fetchone()
        conn.close()
        return row[0] if row else 0

    def time_window(self, camera: str | None = None) -> tuple[float, float]:
        """Get the time window of tracked activity."""
        conn = sqlite3.connect(self.db_path)
        if camera:
            row = conn.execute(
                "SELECT MIN(first_seen), MAX(last_seen) FROM tracks WHERE camera_id=?",
                (camera,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MIN(first_seen), MAX(last_seen) FROM tracks",
            ).fetchone()
        conn.close()
        if row and row[0] is not None:
            return (row[0], row[1])
        return (0.0, 0.0)


@dataclass
class AnalyticsRouter:
    """Routes analytics questions to the right handler."""

    counter: PeopleCounter
    _chat: object = None

    def _get_chat(self) -> object:
        if self._chat is None:
            from looking_glass.models.vlm_chat import OllamaChat
            self._chat = OllamaChat()
        return self._chat

    def answer(self, question: str) -> str:
        """Answer an analytics question."""
        q = question.lower()

        # Fast path: keyword detection
        camera = self._extract_camera(q)
        if any(kw in q for kw in ["how many", "count", "number of"]):
            count = self.counter.count_all(camera)
            window = self.counter.time_window(camera)
            window_str = f"{window[0]:.0f}s to {window[1]:.0f}s" if window[0] > 0 else "all time"
            # Use Ollama to phrase it nicely
            chat = self._get_chat()
            prompt = (
                f"The user asked: '{question}'. "
                f"The data shows {count} unique tracked objects "
                f"in the time window {window_str}. "
                f"Give a one-sentence answer."
            )
            try:
                return chat.answer(  # type: ignore[union-attr]
                    "You are a video analytics assistant. Answer concisely.",
                    prompt,
                )
            except Exception:
                return f"{count} objects tracked ({window_str})."

        # Smart path: let Ollama interpret
        chat = self._get_chat()
        count = self.counter.count_all(camera)
        try:
            return chat.answer(  # type: ignore[union-attr]
                "You are a video analytics assistant with access to camera footage data. "
                f"Total tracked objects: {count}. Answer the user's question concisely.",
                question,
            )
        except Exception:
            return f"I found {count} tracked objects across all cameras."

    @staticmethod
    def _extract_camera(text: str) -> str | None:
        """Extract camera ID from text."""
        match = re.search(r"cam(?:era)?\s*(\d+)", text)
        if match:
            return f"cam{match.group(1).zfill(2)}"
        if "lobby" in text:
            return "cam04"
        if "street" in text:
            return "cam01"
        if "doorstep" in text:
            return "cam08"
        return None
