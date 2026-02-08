"""FastAPI application for the Whisper Ops web dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tools.web_dashboard.api.routes import build_router
from tools.web_dashboard.core.run_manager import RunManager

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
run_manager = RunManager(base_dir=BASE_DIR)

app = FastAPI(title="Whisper Ops Web Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(build_router(run_manager))
