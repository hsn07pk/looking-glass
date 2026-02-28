.PHONY: setup detect download-models ingest serve frontend demo demo-smoke test eval preflight verify clean

setup:
	uv sync --extra dev
	@echo "✓ Python dependencies installed"

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
	@echo "Starting backend..."
	uv run uvicorn looking_glass.api.main:app --host 0.0.0.0 --port 8000 &
	@sleep 2
	@echo "Starting frontend..."
	cd frontend && pnpm dev &
	@sleep 2
	@echo "Opening browser..."
	open http://localhost:5173 || xdg-open http://localhost:5173 || true
	@echo "Demo running. Press Ctrl+C to stop."
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
	$(MAKE) demo-smoke

verify:
	uv run python scripts/verify_state.py

clean:
	rm -rf data/qdrant_storage .venv __pycache__ frontend/node_modules frontend/dist
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
