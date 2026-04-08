from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from looking_glass.api.routes import alerts, analytics, cameras, search, settings
from looking_glass.api.schemas import HealthResponse
from looking_glass.config import DATA_DIR


def create_app() -> FastAPI:
    app = FastAPI(
        title="Looking Glass",
        description="Natural language video intelligence API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(search.router, tags=["search"])
    app.include_router(alerts.router, tags=["alerts"])
    app.include_router(analytics.router, tags=["analytics"])
    app.include_router(cameras.router, tags=["cameras"])
    app.include_router(settings.router, tags=["settings"])

    videos_dir = DATA_DIR / "videos" / "normalized"
    frames_dir = DATA_DIR / "frames"
    if videos_dir.exists():
        app.mount("/videos", StaticFiles(directory=str(videos_dir)), name="videos")
    if frames_dir.exists():
        app.mount("/frames", StaticFiles(directory=str(frames_dir)), name="frames")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        from looking_glass.api.deps import get_store

        try:
            store = get_store()
            count = store.count("frames")
            return HealthResponse(
                ok=count > 0,
                models_loaded=count > 0,
                frame_count=count,
            )
        except Exception:
            return HealthResponse(ok=False, models_loaded=False, frame_count=0)

    return app


app = create_app()
