import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.core.database import init_database
from backend.services.streaming_service import streaming_service
from backend.api.health import router as health_router
from backend.api.datasets import router as datasets_router
from backend.api.models import router as models_router
from backend.api.inference import router as inference_router
from backend.api.experiments import router as experiments_router
from backend.api.streaming import router as streaming_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("eeg_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database & registered built-in checkpoints
    logger.info("Initializing EEG Platform Backend...")
    await init_database()
    logger.info("Database initialized successfully.")
    yield
    # Shutdown: Stop active streams
    logger.info("Shutting down EEG Platform Backend...")
    streaming_service.stop_stream()

app = FastAPI(
    title="EEG Trial-Minimisation & Research Platform API",
    description="Full-stack scientific backend for EEG dataset inspection, preprocessing, model inference, trial minimisation, GAN augmentation, and streaming visualization.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list + ["*"], # Allow development frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router)
app.include_router(datasets_router)
app.include_router(models_router)
app.include_router(inference_router)
app.include_router(experiments_router)
app.include_router(streaming_router)

# Root status / SPA fallback
from pathlib import Path
from fastapi.responses import FileResponse

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if (frontend_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

@app.get("/")
async def root():
    if (frontend_dist / "index.html").exists():
        return FileResponse(frontend_dist / "index.html")
    return {
        "platform": "EEG Trial Minimisation Research Platform",
        "status": "online",
        "docs_url": "/docs",
        "api_health": "/api/health"
    }

@app.get("/{full_path:path}")
async def serve_spa_catchall(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
        return {"detail": "Not Found"}
    target = frontend_dist / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    if (frontend_dist / "index.html").exists():
        return FileResponse(frontend_dist / "index.html")
    return {"platform": "EEG Trial Minimisation Research Platform", "status": "online"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
