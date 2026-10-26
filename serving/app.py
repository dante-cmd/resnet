"""FastAPI inference service.

    uvicorn serving.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health            liveness probe
    GET  /ready             readiness probe (model loaded)
    POST /predict           multipart image file -> top-k predictions
    POST /predict/url       {"url": ...} -> top-k predictions
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from resnet_mlops.predict import Predictor

CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best.pt")

app = FastAPI(title="resnet-mlops", version="0.1.0")
_predictor: Predictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    _predictor = Predictor(CHECKPOINT)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "classes": len(_predictor.classes)}


class UrlRequest(BaseModel):
    url: HttpUrl
    top_k: int = 5


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5) -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")
    return _predictor.predict_bytes(payload, top_k=top_k)


@app.post("/predict/url")
async def predict_url(req: UrlRequest) -> dict:
    import httpx

    if _predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        payload = httpx.get(str(req.url), timeout=10.0, follow_redirects=True).content
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"failed to fetch image: {exc}") from exc
    return _predictor.predict_bytes(payload, top_k=req.top_k)
