"""Generate synthetic demo clips for Looking Glass using OpenCV.

Each clip simulates a camera feed with colored objects to test detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
NORM_DIR = ROOT / "data" / "videos" / "normalized"


def generate_clip(
    path: Path,
    duration: float = 10.0,
    fps: float = 25.0,
    size: tuple[int, int] = (1280, 720),
    bg_color: tuple[int, int, int] = (40, 40, 40),
    objects: list[dict] | None = None,
) -> None:
    """Generate a synthetic video clip with moving colored objects."""
    if path.exists():
        print(f"  {path.name}: already exists")
        return

    w, h = size
    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    for i in range(total_frames):
        frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
        # Add some noise for texture
        noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        if objects:
            for obj in objects:
                x = obj["x"] + int(obj.get("dx", 0) * i / total_frames)
                y = obj["y"] + int(obj.get("dy", 0) * i / total_frames)
                ow, oh = obj["w"], obj["h"]
                color = obj["color"]
                # Draw filled rectangle
                cv2.rectangle(frame, (x, y), (x + ow, y + oh), color, -1)
                # Optional label
                if "label" in obj:
                    cv2.putText(frame, obj["label"], (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add timestamp overlay
        ts = f"{i / fps:.1f}s"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

        writer.write(frame)

    writer.release()
    print(f"  {path.name}: generated ({total_frames} frames)")


def main() -> None:
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    clips = [
        ("cam01_street_truck.mp4", (80, 80, 80), [
            {"x": 100, "y": 200, "w": 400, "h": 200, "color": (30, 120, 220), "label": "truck", "dx": 200},
        ]),
        ("cam02_indoor_bag.mp4", (60, 60, 70), [
            {"x": 400, "y": 350, "w": 150, "h": 100, "color": (60, 80, 140), "label": "bag"},
            {"x": 200, "y": 100, "w": 60, "h": 180, "color": (100, 80, 60), "dx": 100},
        ]),
        ("cam03_office_handshake.mp4", (70, 70, 80), [
            {"x": 300, "y": 100, "w": 100, "h": 300, "color": (120, 80, 60)},
            {"x": 500, "y": 120, "w": 100, "h": 300, "color": (60, 80, 120)},
        ]),
        ("cam04_lobby_traffic.mp4", (90, 90, 90), [
            {"x": 50, "y": 100, "w": 60, "h": 180, "color": (150, 60, 60), "dx": 300},
            {"x": 200, "y": 120, "w": 60, "h": 180, "color": (60, 150, 60), "dx": 250},
            {"x": 400, "y": 110, "w": 60, "h": 180, "color": (60, 60, 150), "dx": 200},
            {"x": 600, "y": 130, "w": 60, "h": 180, "color": (150, 150, 60), "dx": 100},
            {"x": 800, "y": 100, "w": 60, "h": 180, "color": (60, 150, 150), "dx": -200},
        ]),
        ("cam05_tourist_photo.mp4", (100, 130, 160), [
            {"x": 400, "y": 100, "w": 80, "h": 250, "color": (80, 60, 40)},
            {"x": 420, "y": 130, "w": 40, "h": 60, "color": (20, 20, 20), "label": "phone"},
        ]),
        ("cam06_intersection_car.mp4", (50, 50, 50), [
            {"x": 50, "y": 300, "w": 200, "h": 100, "color": (40, 40, 200), "label": "red car", "dx": 800},
        ]),
        ("cam07_crowd_jacket.mp4", (70, 70, 70), [
            {"x": 300, "y": 100, "w": 80, "h": 220, "color": (0, 220, 220), "label": "yellow jacket"},
            {"x": 500, "y": 120, "w": 60, "h": 200, "color": (100, 80, 60), "dx": -100},
        ]),
        ("cam08_doorstep_package.mp4", (80, 70, 60), [
            {"x": 400, "y": 400, "w": 150, "h": 100, "color": (60, 100, 140), "label": "package"},
        ]),
    ]

    print("Generating synthetic demo clips...")
    for name, bg, objects in clips:
        generate_clip(NORM_DIR / name, bg_color=bg, objects=objects)

    # Write manifest
    clip_files = sorted(NORM_DIR.glob("*.mp4"))
    manifest = {
        "clips": [
            {"file": c.name, "camera_id": c.stem.split("_")[0], "description": c.stem}
            for c in clip_files
        ],
        "count": len(clip_files),
    }
    manifest_path = ROOT / "data" / "clips_manifest.yaml"
    manifest_path.write_text(yaml.dump(manifest, default_flow_style=False))
    print(f"\nManifest: {len(clip_files)} clips written to {manifest_path}")


if __name__ == "__main__":
    main()
