"""SigLIP frame and text embedder via open_clip."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class FrameEmbedder:
    """Encodes images and text into a shared SigLIP embedding space."""

    model_name: str = "ViT-B-16-SigLIP"
    pretrained: str = "webli"
    _model: nn.Module | None = field(default=None, repr=False)
    _preprocess: Callable | None = field(default=None, repr=False)  # type: ignore[type-arg]
    _tokenizer: Callable | None = field(default=None, repr=False)  # type: ignore[type-arg]
    _dim: int = field(default=0, repr=False)
    _device: str = field(default="", repr=False)
    _logit_scale: float = field(default=1.0, repr=False)
    _logit_bias: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._device = _device()
        model, preprocess = open_clip.create_model_from_pretrained(
            self.model_name,
            pretrained=self.pretrained,
            device=self._device,
        )
        self._model = model
        self._model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._dim = 768  # SigLIP B/16 is always 768

        # Extract learned SigLIP temperature and bias for proper scoring.
        # SigLIP computes: logits = cosine_sim * exp(logit_scale) + logit_bias
        # then: probability = sigmoid(logits)
        if hasattr(model, 'logit_scale'):
            self._logit_scale = float(model.logit_scale.exp().item())
        if hasattr(model, 'logit_bias'):
            self._logit_bias = float(model.logit_bias.item())

    def encode_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Encode a BGR/RGB numpy image to a unit-norm embedding vector."""
        assert self._model is not None
        assert self._preprocess is not None

        # Convert numpy BGR (OpenCV) to PIL RGB
        if image.ndim == 3 and image.shape[2] == 3:
            pil_img = Image.fromarray(image[:, :, ::-1])
        else:
            pil_img = Image.fromarray(image)

        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)  # type: ignore[union-attr]

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = F.normalize(features, dim=-1)

        return features[0].cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> npt.NDArray[np.float32]:
        """Encode a text query to a unit-norm embedding vector."""
        assert self._model is not None
        assert self._tokenizer is not None

        tokens = self._tokenizer([text]).to(self._device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = F.normalize(features, dim=-1)

        return features[0].cpu().numpy().astype(np.float32)

    def cosine_to_probability(self, cosine_sim: float) -> float:
        """Convert raw cosine similarity to SigLIP probability using learned params.

        SigLIP uses: prob = sigmoid(cosine_sim * temperature + bias)
        This gives meaningful 0-1 probabilities instead of raw cosine scores.
        """
        logit = cosine_sim * self._logit_scale + self._logit_bias
        return 1.0 / (1.0 + np.exp(-logit))

    def dim(self) -> int:
        """Return embedding dimensionality (768 for SigLIP B/16)."""
        return self._dim
