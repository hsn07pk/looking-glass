# Looking Glass

Natural-language video search for security camera footage. Query across multiple camera feeds using plain English, get real-time alerts when objects of interest appear, and ask analytical questions about what's happening on screen.

Built for **Springineering Challenge 2026** at the University of Oulu, ITEE.

Runs 100% offline. No API keys, no cloud. All models run locally via Ollama and open-weight checkpoints.

## Requirements

- Python 3.11+, Node 20+, ffmpeg
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Ollama](https://ollama.com) with `llama3.2:3b` and `minicpm-v`
- 16 GB RAM minimum (32 GB recommended)
- ~20 GB free disk for model weights

## Quick start

```bash
git clone https://github.com/hsn07pk/spring-engineering.git
cd spring-engineering
make setup
make download-models
make ingest
make demo
```

Then open **http://localhost:5173** and try:

- `"orange construction truck"` — finds the truck on cam01
- `"bag left unattended"` — highlights the bag on cam02
- `"person taking a photo"` — locates the tourist on cam05
- `"how many people in the lobby today"` — analytics via the chat panel

## Architecture

```
MP4 clips
  -> Frame sampling (1 fps)
  -> YOLO-World detection + ByteTrack tracking
  -> SigLIP frame/crop embeddings + Florence-2 / MiniCPM-V captions
  -> Qdrant vector store (embedded mode)
  -> FastAPI backend  (/search, /alerts, /analytics, /cameras)
  -> React + Vite frontend (multi-camera grid, bounding boxes, chat)
```

## Project structure

```
src/looking_glass/
  models/       # detector, tracker, embedder, captioner, vlm_chat
  ingestion/    # frame sampler, pipeline
  search/       # NL search, reranker
  store/        # Qdrant wrapper, schemas
  alerts/       # rule engine, real-time alerts
  analytics/    # people counter, LLM-powered Q&A
  api/          # FastAPI routes
  sources/      # video source abstractions

frontend/       # React + Vite + Tailwind
scripts/        # ingest, download, eval, health check
tests/          # pytest suite
demo/           # presentation files
```

## Tech stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLO-World v2 (open vocabulary) |
| Tracking | ByteTrack via supervision |
| Embeddings | SigLIP ViT-B-16 |
| Captioning | Florence-2 + MiniCPM-V (Ollama) |
| Analytics LLM | Llama 3.2 3B (Ollama) |
| Vector DB | Qdrant (embedded mode) |
| Backend | FastAPI + uvicorn |
| Frontend | React + Vite + Tailwind CSS |
| Package mgmt | uv (Python), pnpm (JS) |

## License

MIT
