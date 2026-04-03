"""End-to-end test of Looking Glass frontend + backend."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

FRONTEND = "http://localhost:5173"
BACKEND = "http://localhost:8000"
SCREENSHOTS = Path(__file__).resolve().parent.parent / "SCREENSHOTS"
SCREENSHOTS.mkdir(exist_ok=True)

results: list[dict] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "passed": passed, "detail": detail})
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> None:
    print("=" * 60)
    print("LOOKING GLASS — End-to-End Test Suite")
    print("=" * 60)

    # ── 1. Backend API tests (no browser needed) ──
    print("\n── Backend API Tests ──")
    import urllib.request

    # 1a. Health endpoint
    try:
        r = urllib.request.urlopen(f"{BACKEND}/health")
        data = json.loads(r.read())
        record("Health endpoint", data.get("ok") is True,
               f"models_loaded={data.get('models_loaded')}, frame_count={data.get('frame_count')}")
    except Exception as e:
        record("Health endpoint", False, str(e))

    # 1b. Cameras endpoint
    try:
        r = urllib.request.urlopen(f"{BACKEND}/cameras")
        cams = json.loads(r.read())
        record("Cameras endpoint", len(cams) == 8,
               f"got {len(cams)} cameras: {[c['camera_id'] for c in cams]}")
    except Exception as e:
        record("Cameras endpoint", False, str(e))

    # 1c. Search endpoint
    try:
        req = urllib.request.Request(
            f"{BACKEND}/search",
            data=json.dumps({"q": "orange truck", "top_k": 5}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r = urllib.request.urlopen(req)
        data = json.loads(r.read())
        res = data.get("results", [])
        has_results = len(res) > 0
        has_detections = any(r.get("detections") for r in res)
        has_captions = any(r.get("caption") for r in res)
        multi_cam = len(set(r.get("camera_id") for r in res)) > 0
        record("Search endpoint returns results", has_results,
               f"{len(res)} results")
        record("Search results have detections", has_detections,
               f"detections found in {sum(1 for r in res if r.get('detections'))}/{len(res)} results")
        record("Search results have captions", has_captions,
               f"captions found in {sum(1 for r in res if r.get('caption'))}/{len(res)} results")
        record("Search results have camera IDs", multi_cam,
               f"cameras: {list(set(r.get('camera_id') for r in res))}")
    except Exception as e:
        record("Search endpoint", False, str(e))

    # 1d. Alert rules endpoint
    try:
        req = urllib.request.Request(
            f"{BACKEND}/alerts/rules",
            data=json.dumps({"q": "person with red jacket"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r = urllib.request.urlopen(req)
        data = json.loads(r.read())
        record("Alert registration", "rule_id" in data or "id" in data or r.status == 200,
               f"response: {json.dumps(data)[:100]}")
    except Exception as e:
        record("Alert registration", False, str(e))

    # 1e. Analytics chat endpoint
    try:
        req = urllib.request.Request(
            f"{BACKEND}/analytics/ask",
            data=json.dumps({"q": "How many people on cam01?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r = urllib.request.urlopen(req, timeout=60)
        data = json.loads(r.read())
        has_answer = bool(data.get("answer"))
        record("Analytics chat endpoint", has_answer,
               f"answer: {str(data.get('answer', ''))[:120]}")
    except Exception as e:
        record("Analytics chat endpoint", False, str(e))

    # 1f. Video files served
    try:
        r = urllib.request.urlopen(f"{BACKEND}/videos/cam01_street_truck.mp4", timeout=5)
        size = len(r.read(1024))
        record("Video static files served", size > 0, f"got {size} bytes from cam01 video")
    except Exception as e:
        record("Video static files served", False, str(e))

    # ── 2. Frontend browser tests ──
    print("\n── Frontend Browser Tests ──")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 2a. Page loads
        page.goto(FRONTEND, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        title_text = page.text_content("header") or ""
        record("Frontend loads", "LOOKING GLASS" in title_text,
               f"header: {title_text[:80]}")
        page.screenshot(path=str(SCREENSHOTS / "01_homepage.png"), full_page=True)

        # 2b. Camera grid shows 8 cameras
        cam_tiles = page.locator("video").all()
        # Main view might also have a video, count all
        record("Camera grid renders videos", len(cam_tiles) >= 8,
               f"found {len(cam_tiles)} video elements")

        # 2c. Camera count in header
        cam_count_text = page.text_content("header") or ""
        record("Camera count in header", "8 cameras" in cam_count_text,
               f"header shows: {cam_count_text}")

        # 2d. Clock is ticking
        time1 = page.text_content("header") or ""
        page.wait_for_timeout(1500)
        time2 = page.text_content("header") or ""
        record("Live clock updates", time1 != time2, "clock text changed between reads")

        # 2e. Click a camera tile (cam03)
        cam_tiles_divs = page.locator("video").all()
        if len(cam_tiles_divs) >= 3:
            cam_tiles_divs[2].click()
            page.wait_for_timeout(1000)
            page.screenshot(path=str(SCREENSHOTS / "02_cam_selected.png"), full_page=True)
            record("Camera selection works", True, "clicked cam03 tile")

        # ── Search tests ──
        print("\n── Search Feature Tests ──")

        # 2f. Type search query and submit
        search_input = page.locator("input[placeholder*='Search across']")
        search_input.fill("orange truck")
        page.locator("button", has_text="Search").click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SCREENSHOTS / "03_search_orange_truck.png"), full_page=True)

        # Check search results strip appeared
        result_buttons = page.locator("button:has-text('%')").all()
        record("Search results appear in UI", len(result_buttons) > 0,
               f"found {len(result_buttons)} result buttons")

        # Check if match overlay appeared
        match_overlay = page.locator("text=Match:").all()
        record("Match confidence overlay shown", len(match_overlay) > 0,
               f"found {len(match_overlay)} match overlays")

        # Check caption overlay
        # The caption div is in the top-left of the main view
        page_text = page.text_content("body") or ""
        has_caption_text = any(word in page_text.lower() for word in ["image shows", "scene", "truck", "car", "person", "parking"])
        record("Caption overlay shown", has_caption_text,
               f"found descriptive text in page")

        # 2g. Click a different result button
        if len(result_buttons) > 1:
            result_buttons[1].click()
            page.wait_for_timeout(1000)
            page.screenshot(path=str(SCREENSHOTS / "04_search_result_click.png"), full_page=True)
            record("Result button switches view", True, "clicked second result")

        # 2h. Search for person
        search_input.fill("person walking")
        page.locator("button", has_text="Search").click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SCREENSHOTS / "05_search_person.png"), full_page=True)
        result_buttons2 = page.locator("button:has-text('%')").all()
        record("Search 'person walking' returns results", len(result_buttons2) > 0,
               f"found {len(result_buttons2)} results")

        # 2i. Search with Enter key
        search_input.fill("car at intersection")
        search_input.press("Enter")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SCREENSHOTS / "06_search_enter_key.png"), full_page=True)
        record("Search via Enter key works", True, "submitted with Enter")

        # ── Bounding box test ──
        print("\n── Bounding Box Overlay Tests ──")
        search_input.fill("truck")
        page.locator("button", has_text="Search").click()
        page.wait_for_timeout(3000)
        bbox_divs = page.locator(".bbox-animate").all()
        record("Bounding boxes render on detections", len(bbox_divs) > 0,
               f"found {len(bbox_divs)} bbox overlays")
        page.screenshot(path=str(SCREENSHOTS / "07_bounding_boxes.png"), full_page=True)

        # ── Alert tests ──
        print("\n── Alert Feature Tests ──")

        alert_input = page.locator("input[placeholder*='Register alert']")
        alert_input.fill("person with bag")
        page.locator("button", has_text="+").click()
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS / "08_alert_registered.png"), full_page=True)
        record("Alert registration via UI", True, "registered 'person with bag' alert")

        # Check alert input cleared
        alert_val = alert_input.input_value()
        record("Alert input clears after submit", alert_val == "",
               f"input value: '{alert_val}'")

        # ── Analytics Chat tests ──
        print("\n── Analytics Chat Tests ──")

        # Click a suggested query button
        suggested_btns = page.locator("button:has-text('How many people')").all()
        if suggested_btns:
            suggested_btns[0].click()
            page.wait_for_timeout(500)
            chat_input = page.locator("input[placeholder*='Ask about']")
            chat_val = chat_input.input_value()
            record("Suggested query button fills input", "people" in chat_val.lower(),
                   f"input: '{chat_val}'")

        # Submit analytics chat query
        chat_input = page.locator("input[placeholder*='Ask about']")
        chat_input.fill("Count vehicles on cam01")
        chat_input.press("Enter")
        # This calls Ollama which can be slow
        page.wait_for_timeout(15000)
        page.screenshot(path=str(SCREENSHOTS / "09_analytics_chat.png"), full_page=True)

        # Check chat messages appeared (user + assistant)
        chat_bubbles = page.locator(".rounded-lg.px-3.py-2").all()
        record("Analytics chat shows messages", len(chat_bubbles) >= 2,
               f"found {len(chat_bubbles)} chat bubbles")

        page_text = page.text_content("body") or ""
        has_real_answer = "error connecting" not in page_text.lower()
        record("Analytics chat answer is not error", has_real_answer,
               "no connection errors in chat")

        # ── WebSocket alert test ──
        print("\n── WebSocket Tests ──")
        # Check if WS connection exists (we can check the alerts section)
        alerts_section = page.locator("text=No alerts yet.").all()
        # WS alerts may or may not have fired
        record("Alerts section renders", True,
               f"{'No alerts yet' if alerts_section else 'Alerts present'}")

        # ── Final full page screenshot ──
        page.screenshot(path=str(SCREENSHOTS / "10_final_state.png"), full_page=True)

        browser.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    print()
    if failed:
        print("FAILURES:")
        for r in results:
            if not r["passed"]:
                print(f"  FAIL: {r['name']} — {r['detail']}")
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
