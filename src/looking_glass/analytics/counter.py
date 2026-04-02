"""People counter and analytics engine using track data + Ollama."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

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

    # Classes known to produce false positives from YOLO-World
    _NOISY_CLASSES = {"phone", "smartphone", "camera"}

    def summary(self, camera: str | None = None) -> str:
        """Get a detailed summary of tracked objects for a camera or all cameras."""
        conn = sqlite3.connect(self.db_path)
        if camera:
            rows = conn.execute(
                "SELECT class_name, COUNT(DISTINCT track_id) as cnt "
                "FROM tracks WHERE camera_id=? GROUP BY class_name ORDER BY cnt DESC",
                (camera,),
            ).fetchall()
            window = self.time_window(camera)
            total = self.count_all(camera)
            header = f"Camera {camera}"
        else:
            rows = conn.execute(
                "SELECT class_name, COUNT(DISTINCT track_id) as cnt "
                "FROM tracks GROUP BY class_name ORDER BY cnt DESC",
            ).fetchall()
            window = self.time_window()
            total = self.count_all()
            header = "All cameras"

        lines = [f"{header}: {total} total tracked objects"]
        lines.append(f"Time window: {window[0]:.1f}s to {window[1]:.1f}s")
        if rows:
            lines.append("Breakdown by type:")
            for class_name, cnt in rows:
                if class_name not in self._NOISY_CLASSES:
                    lines.append(f"  - {class_name}: {cnt}")

        # Per-camera breakdown when no specific camera
        if not camera:
            cam_rows = conn.execute(
                "SELECT camera_id, class_name, COUNT(DISTINCT track_id) as cnt "
                "FROM tracks GROUP BY camera_id, class_name ORDER BY camera_id, cnt DESC",
            ).fetchall()
            cam_data: dict[str, list[str]] = {}
            for cam_id, cls, cnt in cam_rows:
                if cls not in self._NOISY_CLASSES:
                    cam_data.setdefault(cam_id, []).append(f"{cnt} {cls}")
            lines.append("Per camera:")
            for cam_id in sorted(cam_data):
                lines.append(f"  {cam_id}: {', '.join(cam_data[cam_id])}")

        conn.close()
        return "\n".join(lines)


@dataclass
class AnalyticsRouter:
    """Routes analytics questions to the right handler."""

    counter: PeopleCounter
    _chat: object = None
    _history: list[dict] = field(default_factory=list)

    def _get_chat(self) -> object:
        if self._chat is None:
            from looking_glass.models.vlm_chat import OllamaChat
            self._chat = OllamaChat()
        return self._chat

    def _build_search_query(self, question: str) -> str:
        """Build a descriptive search query from the question + conversation history.

        SigLIP works best with descriptive phrases, not questions.
        Use Ollama to convert the question (with history context) into a visual search query.
        """
        if not self._history:
            # No history — extract key nouns/descriptors from the question
            # Remove question words to make it more descriptive
            q = question.lower()
            for word in ["what", "which", "where", "how", "many", "is", "are", "the",
                         "do", "does", "did", "can", "could", "?", "please", "tell me"]:
                q = q.replace(word, "")
            return q.strip() or question

        # With history: ask Ollama to synthesize a visual search query
        chat = self._get_chat()
        history_text = "\n".join(
            f"  Q: {h['question']}\n  A: {h['answer']}" for h in self._history[-3:]
        )
        try:
            result = chat.answer(  # type: ignore[union-attr]
                "You convert questions into short visual search descriptions. "
                "Output ONLY a 3-8 word descriptive phrase, nothing else.",
                f"Conversation so far:\n{history_text}\n\n"
                f"New question: {question}\n\n"
                f"Write a short descriptive phrase to search for in video frame captions:",
            )
            return result.strip().strip('"').strip("'") or question
        except Exception:
            return question

    def _search_captions(self, question: str) -> str:
        """Search video frames for visual context matching the question."""
        try:
            from looking_glass.api.deps import get_search
            search = get_search()

            search_query = self._build_search_query(question)
            results = search.search(search_query, top_k=5)
            if not results:
                return ""
            lines = [f"Visual search results for '{search_query}':"]
            for r in results:
                if r.caption:
                    lines.append(f"  - {r.camera_id} at {r.timestamp:.1f}s: {r.caption}")
            return "\n".join(lines)
        except Exception:
            return ""

    def answer(self, question: str) -> str:
        """Answer an analytics question."""
        q = question.lower()
        camera = self._extract_camera(q)

        # Get tracking data from SQLite
        data_summary = self.counter.summary(camera)

        # Search frame captions for visual/action context
        visual_context = self._search_captions(question)

        chat = self._get_chat()
        prompt = f"The user asked: '{question}'\n\n"

        # Include recent conversation history for context
        if self._history:
            prompt += "Previous conversation:\n"
            for h in self._history[-3:]:
                prompt += f"  Q: {h['question']}\n  A: {h['answer']}\n"
            prompt += "\n"

        prompt += (
            f"Tracking data:\n{data_summary}\n\n"
        )
        if visual_context:
            prompt += f"{visual_context}\n\n"
        prompt += (
            f"Answer the user's question based ONLY on the data above. "
            f"Include specific details from the frame captions (colors, clothing, objects) when available. "
            f"Do NOT invent details not present in the data. "
            f"Be concise (1-2 sentences)."
        )
        try:
            answer = chat.answer(  # type: ignore[union-attr]
                "You are a video surveillance analytics assistant. "
                "Answer ONLY based on the tracking data and visual search results provided. "
                "When captions mention colors, clothing, or descriptions, include them in your answer. "
                "Never invent or hallucinate details not present in the data.",
                prompt,
            )
        except Exception:
            answer = data_summary

        # Save to history for follow-up questions
        self._history.append({"question": question, "answer": answer})
        if len(self._history) > 10:
            self._history = self._history[-10:]

        return answer

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
