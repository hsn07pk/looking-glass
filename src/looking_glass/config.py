"""Central configuration for Looking Glass."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
NORMALIZED_DIR = VIDEOS_DIR / "normalized"
MODELS_DIR = ROOT_DIR / "models"
QDRANT_DIR = DATA_DIR / "qdrant_storage"

OPEN_VOCAB_SET = [
    "person", "man", "woman", "child",
    "car", "truck", "construction truck", "dump truck", "van", "bus", "bicycle", "motorcycle",
    "bag", "backpack", "duffel bag", "handbag", "suitcase", "package", "box",
    "jacket", "yellow jacket", "high-visibility vest",
    "dog", "cat",
    "door", "window", "entrance",
    "umbrella", "helmet",
]
