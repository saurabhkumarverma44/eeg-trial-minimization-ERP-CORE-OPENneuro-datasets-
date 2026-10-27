from typing import List, Optional
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.models.schemas import ModelRegistryItem, ModelValidationResponse
from backend.services.model_service import (
    list_models,
    save_uploaded_model,
    validate_model,
    activate_model,
    deactivate_model,
    delete_model
)

router = APIRouter(prefix="/api/models", tags=["Model Registry & Management"])

@router.get("", response_model=List[ModelRegistryItem])
async def get_all_models():
    return await list_models()

@router.post("/upload", response_model=ModelRegistryItem)
async def upload_model_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    architecture: str = Form(...),
    num_channels: int = Form(31),
    sampling_rate: float = Form(1000.0),
    class_labels_json: Optional[str] = Form(None),
    preprocessing_json: Optional[str] = Form(None)
):
    labels = json.loads(class_labels_json) if class_labels_json else None
    prep = json.loads(preprocessing_json) if preprocessing_json else None
    return await save_uploaded_model(
        file=file,
        name=name,
        architecture=architecture,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        class_labels=labels,
        preprocessing_info=prep
    )

@router.post("/{model_id}/validate", response_model=ModelValidationResponse)
async def trigger_model_validation(model_id: str):
    return await validate_model(model_id)

@router.post("/{model_id}/activate")
async def trigger_model_activation(model_id: str):
    await activate_model(model_id)
    return {"status": "success", "message": f"Model '{model_id}' activated."}

@router.post("/{model_id}/deactivate")
async def trigger_model_deactivation(model_id: str):
    await deactivate_model(model_id)
    return {"status": "success", "message": f"Model '{model_id}' deactivated."}

@router.delete("/{model_id}")
async def trigger_model_deletion(model_id: str):
    await delete_model(model_id)
    return {"status": "success", "message": f"Model '{model_id}' successfully removed."}
