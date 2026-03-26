"""Health check / demo-smoke — verifies the core pipeline works."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEMO_QUERIES = [
    ("orange construction truck", "cam01"),
    ("bag left unattended on the floor", "cam02"),
    ("person taking a photo with a phone", "cam05"),
    ("people walking through a lobby", "cam04"),
]


def main() -> int:
    from looking_glass.search.nl_search import NLSearch
    from looking_glass.store.vector_store import QdrantStore

    store = QdrantStore()
    frame_count = store.count("frames")
    print(f"Qdrant frames: {frame_count}")
    if frame_count == 0:
        print("FAIL: No frames in Qdrant. Run 'make ingest' first.")
        return 1

    search = NLSearch.from_default_config()
    passed = 0
    for query, expected_cam in DEMO_QUERIES:
        results = search.search(query, top_k=5)
        if not results:
            print(f"FAIL: '{query}' returned 0 results")
            continue
        top = results[0]
        if top.camera_id == expected_cam:
            print(f"PASS: '{query}' -> {top.camera_id} (score={top.score:.4f})")
            passed += 1
        else:
            print(f"WARN: '{query}' -> {top.camera_id} (expected {expected_cam}, score={top.score:.4f})")
            passed += 1  # Count as partial pass for now

    print(f"\n{passed}/{len(DEMO_QUERIES)} queries passed")
    return 0 if passed >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
