"""
main.py  –  FastAPI application entrypoint

This is the single file Railway (or any deployment server) boots.
It wires together:
  • App configuration
  • Lifespan (startup / shutdown events)
  • CORS middleware
  • Global exception handler
  • All API routers
  • Root / health-check endpoints
"""

import time
import os
from dotenv import load_dotenv

# Load .env before anything else so all os.getenv() calls pick up the values
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.logger import logging

# Import the router AND the service-initialiser from the backend package
from backend.api import router as review_router, init_services


# ──────────────────────────────────────────────────────────────────────────────
# 1. Application Configuration
# ──────────────────────────────────────────────────────────────────────────────

class AppConfig:
    """Central config – values fall back to sane defaults when env vars are absent."""
    TITLE: str       = "PlayStore Review Analyzer API"
    DESCRIPTION: str = "End-to-end Play Store review sentiment analysis & insight service"
    VERSION: str     = "1.0.0"
    DEBUG: bool      = os.getenv("DEBUG", "False").lower() == "true"
    # In production you should replace "*" with your actual frontend origin
    ALLOW_ORIGINS: list = os.getenv("ALLOW_ORIGINS", "*").split(",")
    HOST: str        = "0.0.0.0"
    PORT: int        = int(os.getenv("PORT", 8000))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Lifespan  (startup + shutdown)
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup (before requests) and on shutdown."""
    logging.info(f"Starting {AppConfig.TITLE} v{AppConfig.VERSION} ...")
    try:
        init_services()   # warm up ML models & external API clients
        logging.info("All services initialised – server is ready to accept requests.")
    except Exception as exc:
        # Log, but don't crash – /api/status will report services as not ready
        logging.critical(f"Service initialisation failed: {exc}", exc_info=True)

    yield  # ← application runs here

    logging.info(f"Shutting down {AppConfig.TITLE} ...")


# ──────────────────────────────────────────────────────────────────────────────
# 3. FastAPI Instance
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=AppConfig.TITLE,
    description=AppConfig.DESCRIPTION,
    version=AppConfig.VERSION,
    lifespan=lifespan,
    debug=AppConfig.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Middleware
# ──────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Global Exception Handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches any unhandled exception and returns a clean JSON error."""
    logging.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Routers
# ──────────────────────────────────────────────────────────────────────────────

# All review-analysis endpoints are served under /api
# e.g.  POST /api/analyze
#        GET /api/status
app.include_router(review_router, prefix="/api", tags=["Play Store Reviews"])


# ──────────────────────────────────────────────────────────────────────────────
# 7. Core Endpoints  (root & health-check)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Monitoring"], summary="API welcome message")
async def root():
    return {
        "message": f"Welcome to the {AppConfig.TITLE}",
        "version": AppConfig.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", tags=["Monitoring"], summary="Health check")
async def health_check():
    """Lightweight liveness probe used by Railway and load balancers."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": AppConfig.VERSION,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Local Development Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.info(f"Launching dev server on {AppConfig.HOST}:{AppConfig.PORT}")
    uvicorn.run(
        "main:app",
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        reload=AppConfig.DEBUG,
    )
