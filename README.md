# Vyapar-Mitra — Backend

FastAPI service for the Vyapar-Mitra business-advisory app: hyper-local business
discovery, deterministic feasibility + loan/EMI planning, government-scheme
matching, and a grounded AI advisory chatbot. The web client lives in a separate repo.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
copy .env.example .env           # then fill in OPENROUTER_API_KEY
uvicorn main:app --host 127.0.0.1 --port 8000
```

Check: `http://127.0.0.1:8000/api/health` → `{"status":"ok"}`

## Environment

| var | required | notes |
|---|---|---|
| `OPENROUTER_API_KEY` | **yes** for AI features | https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | no | default `google/gemini-2.5-flash` |
| `OPENROUTER_FREE_MODEL` | no | rate-limit fallback |
| `GOOGLE_MAPS_API_KEY` | no | Google Places; falls back to OpenStreetMap |

All LLM calls route through OpenRouter via `ai_client.py`.

## Deploy → Render

**Blueprint:** Render → New + → Blueprint → this repo (reads `render.yaml`), then set
`OPENROUTER_API_KEY` in the service's Environment tab.

**Manual:**
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- Env: `OPENROUTER_API_KEY`, `PYTHON_VERSION=3.11.9`

CORS is open (`allow_origins=["*"]`). Point the frontend at the deployed URL with
`VITE_BACKEND_URL`.

> Render's free tier sleeps after ~15 min idle — first request then takes 30–50 s.

## Notes

- `requirements.txt` is intentionally slim (no `torch`/`transformers`). The only thing
  that needs them is the optional DistilBERT expense classifier (`/api/upload-expenses`),
  whose 267 MB weights aren't in the repo. Every other endpoint works without it; the
  categorizer just returns `"Unknown"`. For local use: `pip install -r requirements-ml.txt`
  plus the model files.
