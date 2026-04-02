"""
api/app.py

FastAPI application entry point.

Wires together:
  - CORS middleware
  - Structured logging
  - Lifespan (graph pre-warm on startup)
  - All route handlers
  - Global exception handlers
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    stream  = sys.stdout,
    level   = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Pre-warms the LangGraph instance so the first request isn't slow.
    """
    logger.info("Starting Autonomous Job Finder API...")
    from api.dependencies import get_graph
    try:
        get_graph()
        logger.info("LangGraph pipeline ready.")
    except Exception as e:
        logger.error(f"Failed to initialise graph: {e}")
        # Don't crash on startup — let individual requests fail with a clear error

    yield  # app runs here

    logger.info("Shutting down Autonomous Job Finder API.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Autonomous Job Finder API",
        description = (
            "Multi-agent pipeline that takes a resume as input and autonomously "
            "surfaces the best-matched job opportunities, ranked by semantic fit "
            "and enriched with real-world hiring signals."
        ),
        version     = "0.1.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # In development, allow all origins.
    # In production, restrict to your frontend domain.
    allowed_origins = (
        ["*"]
        if os.getenv("APP_ENV", "development") == "development"
        else os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = allowed_origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1", tags=["pipeline"])

    # ── Global exception handlers ─────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            content     = {
                "detail": "An unexpected error occurred. Please try again.",
                "code":   "internal_server_error",
            },
        )

    return app


# ── App instance ──────────────────────────────────────────────────────────────

app = create_app()


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = os.getenv("APP_ENV", "development") == "development",
        log_level = os.getenv("LOG_LEVEL", "info").lower(),
    )
