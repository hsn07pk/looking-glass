from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from looking_glass.config import DATA_DIR, NORMALIZED_DIR

CHECKPOINT_DIR = DATA_DIR / "checkpoints"


def load_checkpoint(camera_id: str) -> dict | None:
    cp_file = CHECKPOINT_DIR / f"{camera_id}.json"
    if cp_file.exists():
        return json.loads(cp_file.read_text())
    return None


def save_checkpoint(camera_id: str, data: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_file = CHECKPOINT_DIR / f"{camera_id}.json"
    cp_file.write_text(json.dumps(data, indent=2))


def main() -> None:
    from looking_glass.ingestion.pipeline import IngestionPipeline
    from looking_glass.models.detector import OpenVocabDetector
    from looking_glass.models.tracker import MultiObjectTracker
    from looking_glass.models.embedder import FrameEmbedder
    from looking_glass.models.captioner import DenseCaptioner
    from looking_glass.store.vector_store import QdrantStore

    store = QdrantStore()
    pipeline = IngestionPipeline(
        detector=OpenVocabDetector(),
        tracker=MultiObjectTracker(),
        embedder=FrameEmbedder(),
        captioner=DenseCaptioner(),
        store=store,
    )

    clips = sorted(NORMALIZED_DIR.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {NORMALIZED_DIR}")
        return

    completed = []
    pending = []
    for clip in clips:
        cam_id = clip.stem.split("_")[0]
        cp = load_checkpoint(cam_id)
        if cp and cp.get("status") == "done":
            completed.append(cam_id)
        else:
            pending.append(clip)

    print(f"Found {len(clips)} clips total.")
    if completed:
        print(f"Already completed: {', '.join(completed)}")
    print(f"To process: {len(pending)} clips")

    if not pending:
        print("All cameras already ingested. Delete data/checkpoints/ to re-ingest.")
        return

    total_start = time.time()
    total_frames = 0

    for i, clip in enumerate(pending):
        from looking_glass.sources.file_source import FileVideoSource

        cam_id = clip.stem.split("_")[0]
        cp = load_checkpoint(cam_id)

        print(f"\n[{i+1}/{len(pending)}] Ingesting {clip.name}...")

        save_checkpoint(cam_id, {
            "status": "in_progress",
            "clip": clip.name,
            "started_at": time.time(),
        })

        try:
            source = FileVideoSource(clip, sample_fps=2.0)
            frames = source.frames()

            clip_start = time.time()
            stats = pipeline.ingest_frames(source.camera_id(), frames)
            clip_elapsed = time.time() - clip_start

            total_frames += stats["frames_sampled"]
            fps = stats["frames_sampled"] / clip_elapsed if clip_elapsed > 0 else 0

            print(f"  {stats['frames_sampled']}/{stats['frames_total']} frames, "
                  f"{stats['detections']} detections, {stats['tracks']} tracks "
                  f"({clip_elapsed:.0f}s, {fps:.2f} fps)")

            save_checkpoint(cam_id, {
                "status": "done",
                "clip": clip.name,
                "frames_sampled": stats["frames_sampled"],
                "frames_total": stats["frames_total"],
                "detections": stats["detections"],
                "tracks": stats["tracks"],
                "elapsed_sec": round(clip_elapsed, 1),
                "completed_at": time.time(),
            })

            elapsed_so_far = time.time() - total_start
            remaining = len(pending) - (i + 1)
            if i > 0:
                avg_per_clip = elapsed_so_far / (i + 1)
                eta = avg_per_clip * remaining
                print(f"  ETA for remaining {remaining} clips: ~{eta/60:.0f} min")

        except Exception as e:
            print(f"  ERROR: {e}")
            save_checkpoint(cam_id, {
                "status": "error",
                "clip": clip.name,
                "error": str(e),
                "failed_at": time.time(),
            })
            continue

    elapsed = time.time() - total_start
    print(f"\nDone. {total_frames} frames across {len(pending)} clips in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
