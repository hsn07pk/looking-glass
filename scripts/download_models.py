"""Download all models needed for Looking Glass."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    print("=== Downloading YOLO-World ===")
    from ultralytics import YOLO
    YOLO("yolov8s-worldv2.pt")
    print("YOLO-World: OK")

    print("\n=== Downloading SigLIP ===")
    import open_clip
    open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    print("SigLIP: OK")

    print("\n=== Downloading Florence-2 ===")
    from transformers import AutoModelForCausalLM, AutoProcessor
    AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    print("Florence-2: OK")

    print("\n=== Pulling Ollama model ===")
    result = subprocess.run(
        ["ollama", "pull", "llama3.2:3b"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Ollama llama3.2:3b: OK")
    else:
        print(f"Ollama pull failed: {result.stderr}")
        print("Make sure ollama is running: ollama serve")

    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()
