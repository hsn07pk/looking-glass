"""Ollama LLM chat wrapper for analytics phrasing."""

from __future__ import annotations

from dataclasses import dataclass

import ollama
from pydantic import BaseModel


class AnalyticsAnswer(BaseModel):
    """Structured response schema for analytics questions."""
    answer: str


@dataclass
class OllamaChat:
    """Local LLM chat via Ollama for analytics question answering."""

    model: str = "llama3.2:3b"

    def answer(self, system: str, user: str) -> str:
        """Send a system+user prompt and return the assistant's reply.

        Uses structured JSON output with temperature=0 to minimize hallucination.
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=AnalyticsAnswer.model_json_schema(),
            options={"temperature": 0},
        )
        # Parse structured response
        import json
        try:
            parsed = json.loads(response.message.content)
            return parsed.get("answer", response.message.content)
        except (json.JSONDecodeError, AttributeError):
            return response.message.content

    def smoke(self) -> bool:
        """Quick health check: can we reach Ollama and get a reply?"""
        try:
            reply = self.answer("Reply in exactly 3 words.", "Hello")
            return bool(reply.strip())
        except Exception:
            return False
