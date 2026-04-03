.PHONY: setup detect download-models ingest serve frontend demo demo-smoke test eval preflight verify clean fallback

setup:
	uv sync --extra dev
	cd frontend && pnpm install
	mkdir -p LOGS .checkpoints BENCHMARKS data/frames models
	@echo "Dependencies installed."

detect:
	uv run python scripts/detect_system.py

download-models:
	uv run python scripts/download_models.py

ingest:
	uv run python scripts/ingest_all.py

serve:
	uv run uvicorn looking_glass.api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && pnpm dev

demo:
	@echo "Starting Looking Glass demo..."
	@uv run uvicorn looking_glass.api.main:app --host 0.0.0.0 --port 8000 &
	@sleep 3
	@cd frontend && pnpm dev &
	@sleep 2
	@open http://localhost:5173 2>/dev/null || xdg-open http://localhost:5173 2>/dev/null || true
	@echo "Demo running at http://localhost:5173 — Press Ctrl+C to stop."
	@wait

demo-smoke:
	uv run python scripts/health_check.py

test:
	uv run pytest -x --tb=short

eval:
	uv run python scripts/eval_demo_queries.py

preflight:
	uv run python scripts/health_check.py
	uv run python scripts/eval_demo_queries.py
	uv run python scripts/verify_state.py

verify:
	uv run python scripts/verify_state.py

fallback:
	uv run python scripts/screenshot_fallback.py

clean:
	rm -rf data/qdrant_storage data/tracks.db data/frames .venv __pycache__ frontend/node_modules frontend/dist
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
