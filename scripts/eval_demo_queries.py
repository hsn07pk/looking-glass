from __future__ import annotations

import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from looking_glass.search.nl_search import NLSearch

    queries_path = ROOT / "data" / "demo_queries.yaml"
    if not queries_path.exists():
        print("No demo_queries.yaml found")
        return 1

    with open(queries_path) as f:
        data = yaml.safe_load(f)

    search = NLSearch.from_default_config()
    passed = 0
    total = len(data.get("queries", []))

    for q in data.get("queries", []):
        results = search.search(q["nl"], top_k=5)
        if not results:
            print(f"FAIL [{q['id']}]: '{q['nl']}' -> 0 results")
            continue
        top = results[0]
        if top.camera_id == q["expected_camera"]:
            print(f"PASS [{q['id']}]: '{q['nl']}' -> {top.camera_id} (score={top.score:.4f})")
            passed += 1
        else:
            print(f"MISS [{q['id']}]: '{q['nl']}' -> {top.camera_id} (expected {q['expected_camera']})")

    print(f"\n{passed}/{total} queries hit expected camera")
    return 0 if passed >= total * 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
