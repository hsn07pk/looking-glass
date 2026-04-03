"""Florence-2 dense captioner + Ollama VLM exhaustive captioner."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

EXHAUSTIVE_PROMPT = """Describe this surveillance camera frame exhaustively. Cover ALL of the following:
1. PEOPLE: count, gender, estimated age, clothing (colors, type), accessories, posture, actions
2. VEHICLES: type, color, make if visible, license plate text if readable
3. OBJECTS: all visible objects, their colors, positions
4. BACKGROUND: buildings, signs (transcribe text), furniture, weather, lighting
5. ACTIONS: what is happening, interactions between people/objects
6. SPATIAL: where things are relative to each other (left, right, foreground, background)
Be specific about colors, materials, and details. Do not omit anything visible."""


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DenseCaptioner:
    """Wraps Florence-2 for captioning and grounded detection."""

    model_id: str = "microsoft/Florence-2-base"
    _model: AutoModelForCausalLM | None = field(default=None, repr=False)
    _processor: AutoProcessor | None = field(default=None, repr=False)
    _device: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self._device = _device()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()  # type: ignore[union-attr]

    def _run_task(self, image: Image.Image, task: str, text_input: str = "") -> dict:
        """Run a Florence-2 task and return parsed output."""
        assert self._model is not None
        assert self._processor is not None

        prompt = task if not text_input else task + text_input
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )
        return parsed  # type: ignore[return-value]

    def _to_pil(self, image: npt.NDArray[np.uint8]) -> Image.Image:
        """Convert BGR numpy array to RGB PIL Image."""
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image[:, :, ::-1])
        return Image.fromarray(image)

    def caption(self, image: npt.NDArray[np.uint8]) -> str:
        """Generate a short caption for the image."""
        pil_img = self._to_pil(image)
        result = self._run_task(pil_img, "<DETAILED_CAPTION>")
        return result.get("<DETAILED_CAPTION>", "")

    def dense_caption(self, image: npt.NDArray[np.uint8]) -> str:
        """Generate a more detailed caption."""
        pil_img = self._to_pil(image)
        result = self._run_task(pil_img, "<MORE_DETAILED_CAPTION>")
        return result.get("<MORE_DETAILED_CAPTION>", "")

    def exhaustive_caption(self, image: npt.NDArray[np.uint8]) -> str:
        """Generate an exhaustive, paragraph-length caption using Ollama VLM.

        Uses minicpm-v (or llava fallback) to produce detailed descriptions
        covering people, clothing colors, objects, actions, spatial layout.
        """
        import cv2
        import ollama

        # Save frame to temp file for Ollama
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
            # Convert BGR to RGB for saving
            if image.ndim == 3 and image.shape[2] == 3:
                cv2.imwrite(tmp_path, image)
            else:
                cv2.imwrite(tmp_path, image)

        try:
            response = ollama.chat(
                model="minicpm-v",
                messages=[{
                    "role": "user",
                    "content": EXHAUSTIVE_PROMPT,
                    "images": [tmp_path],
                }],
                options={"temperature": 0},
            )
            return response.message.content
        except Exception:
            # Fallback to Florence-2 MORE_DETAILED_CAPTION
            return self.dense_caption(image)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def grounded_detection(
        self, image: npt.NDArray[np.uint8], text: str,
    ) -> list[dict]:
        """Run grounded detection: find bboxes for objects described in text.

        Returns list of dicts with 'bbox' (x1,y1,x2,y2) and 'label'.
        """
        pil_img = self._to_pil(image)
        result = self._run_task(pil_img, "<CAPTION_TO_PHRASE_GROUNDING>", text)
        grounding = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        bboxes = grounding.get("bboxes", [])
        labels = grounding.get("labels", [])

        detections = []
        for i, bbox in enumerate(bboxes):
            label = labels[i] if i < len(labels) else ""
            detections.append({
                "bbox": tuple(bbox),
                "label": label,
            })
        return detections
