"""Ollama LLM chat wrapper for analytics phrasing."""

from __future__ import annotations

from dataclasses import dataclass

import ollama


@dataclass
class OllamaChat:
    """Local LLM chat via Ollama for analytics question answering."""

    model: str = "llama3.2:3b"

    def answer(self, system: str, user: str) -> str:
        """Send a system+user prompt and return the assistant's reply."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.message.content

    def smoke(self) -> bool:
        """Quick health check: can we reach Ollama and get a reply?"""
        try:
            reply = self.answer("Reply in exactly 3 words.", "Hello")
            return bool(reply.strip())
        except Exception:
            return False
