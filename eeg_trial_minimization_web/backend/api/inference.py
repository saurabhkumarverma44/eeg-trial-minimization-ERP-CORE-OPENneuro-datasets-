from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from backend.models.schemas import SingleInferenceRequest, SingleInferenceResponse, BatchInferenceRequest, BatchInferenceResponse
from backend.services.inference_service import run_single_inference, run_batch_inference
from backend.core.database import get_db_connection

router = APIRouter(prefix="/api/inference", tags=["Inference"])

@router.post("/single", response_model=SingleInferenceResponse)
async def predict_single_epoch(req: SingleInferenceRequest):
    return await run_single_inference(req)

@router.post("/batch", response_model=BatchInferenceResponse)
async def predict_batch_epochs(req: BatchInferenceRequest):
    return await run_batch_inference(
        model_id=req.model_id,
        dataset_name=req.dataset_name,
        split=req.split,
        max_samples=req.max_samples
    )

@router.get("/history")
async def get_inference_history(limit: int = 50):
    async with get_db_connection() as db:
        cursor = await db.execute("""
            SELECT * FROM prediction_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
