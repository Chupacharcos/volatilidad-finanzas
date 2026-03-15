"""
FastAPI standalone — Detección de Volatilidad Anómala
Puerto: 8092

Arrancar:
  cd /var/www/volatilidad-anomala
  /var/www/chatbot/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8092
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ML Volatilidad Anómala — TFT-lite",
    version="1.0",
    description="Detección de volatilidad anómala en activos financieros mediante Temporal Fusion Transformer + sentimiento FinBERT-inspired.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://adrianmoreno-dev.com", "http://127.0.0.1", "http://localhost"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

from router import router
app.include_router(router)


@app.get("/")
def root():
    return {"service": "ml-volatilidad-anomala", "status": "ok", "port": 8092}
