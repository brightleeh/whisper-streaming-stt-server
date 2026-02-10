"""FastAPI application for the Whisper Ops web dashboard."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tools.web_dashboard.api.routes import build_router
from tools.web_dashboard.core.run_manager import RunManager

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
run_manager = RunManager(base_dir=BASE_DIR)

app = FastAPI(title="Whisper Ops Web Dashboard", version="0.1.0")

cors_raw = os.getenv("WEB_DASHBOARD_CORS_ORIGINS", "")
if cors_raw:
    cors_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
else:
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

if "*" in cors_origins:
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = cors_origins
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(build_router(run_manager))
