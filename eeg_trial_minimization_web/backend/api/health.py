import os
import sys
import shutil
import platform
from datetime import datetime, timezone
from fastapi import APIRouter
from backend.core.config import settings
from backend.core.database import get_db_connection
from backend.models.schemas import HealthResponse, SystemInfoResponse, InventoryResponse
from backend.services.inventory_service import scan_project_inventory
from backend.services.streaming_service import streaming_service

router = APIRouter(prefix="/api", tags=["Health & System"])

@router.get("/health", response_model=HealthResponse)
async def check_health():
    db_ok = False
    model_count = 0
    active_model_name = None

    try:
        async with get_db_connection() as db:
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM models")
            row = await cursor.fetchone()
            model_count = row["cnt"] if row else 0

            cursor = await db.execute("SELECT name FROM models WHERE is_active = 1 LIMIT 1")
            act_row = await cursor.fetchone()
            active_model_name = act_row["name"] if act_row else None

            db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database_connected=db_ok,
        models_registered=model_count,
        active_model=active_model_name,
        stream_active=streaming_service.is_streaming,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    import torch
    
    # Calculate free disk space on project drive
    drive = str(settings.EEG_PROJECT_ROOT.anchor)
    try:
        disk_usage = shutil.disk_usage(drive if drive else "D:\\")
        free_gb = round(disk_usage.free / (1024 ** 3), 2)
    except Exception:
        free_gb = 0.0

    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Laptop Standard Environment)"

    return SystemInfoResponse(
        os=f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        device_name=dev_name,
        disk_free_gb=free_gb,
        paths=settings.validate_paths()
    )


@router.get("/project/inventory", response_model=InventoryResponse)
async def get_project_inventory():
    return scan_project_inventory()
