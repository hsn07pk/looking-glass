# Looking Glass

Natural language search across security camera footage.
Built for Springineering Challenge 2026, University of Oulu.
**$0 in API costs. 100% local. Zero cloud dependencies.**

## Requirements
- Python 3.11+, Node 20+, ffmpeg, uv, ollama
- 16 GB RAM minimum (32 recommended)
- 20 GB free disk
- See SYSTEM.md after first run

## Quick start
```
git clone <repo>
cd looking-glass
make setup
make detect            # writes SYSTEM.md
make download-models   # ~6 GB pull
make ingest
make demo
```

Open http://localhost:5173 and try:
- "orange construction truck"
- "bag left unattended"
- "person taking a photo"
- "how many people in the lobby today"

## Architecture
## Built with
SigLIP · Florence-2 · YOLO-World · ByteTrack · Qdrant · FastAPI · React · Vite · uv · **Ollama (Llama 3.2)**
