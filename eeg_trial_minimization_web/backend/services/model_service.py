import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile, HTTPException
import torch
from backend.core.config import settings
from backend.core.database import get_db_connection
from backend.core.security import sanitize_filename, validate_safe_path
from backend.models.schemas import ModelRegistryItem, ModelValidationResponse
from backend.models.eegnet import EEGNet, load_eegnet_checkpoint
from backend.models.cnn1d import ERPConv1DNet, load_cnn1d_checkpoint

async def list_models() -> List[ModelRegistryItem]:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT * FROM models ORDER BY is_active DESC, created_at DESC")
        rows = await cursor.fetchall()
        models = []
        for r in rows:
            models.append(ModelRegistryItem(
                id=r["id"],
                name=r["name"],
                filename=r["filename"],
                filepath=r["filepath"],
                format=r["format"],
                architecture=r["architecture"],
                num_channels=r["num_channels"],
                sampling_rate=r["sampling_rate"],
                input_shape=r["input_shape"],
                class_labels=json.loads(r["class_labels"]) if r["class_labels"] else None,
                preprocessing_info=json.loads(r["preprocessing_info"]) if r["preprocessing_info"] else None,
                is_active=bool(r["is_active"]),
                is_builtin=bool(r["is_builtin"]),
                validation_status=r["validation_status"],
                validation_notes=r["validation_notes"],
                created_at=str(r["created_at"])
            ))
        return models


async def get_active_model() -> Optional[ModelRegistryItem]:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT * FROM models WHERE is_active = 1 LIMIT 1")
        r = await cursor.fetchone()
        if not r:
            return None
        return ModelRegistryItem(
            id=r["id"],
            name=r["name"],
            filename=r["filename"],
            filepath=r["filepath"],
            format=r["format"],
            architecture=r["architecture"],
            num_channels=r["num_channels"],
            sampling_rate=r["sampling_rate"],
            input_shape=r["input_shape"],
            class_labels=json.loads(r["class_labels"]) if r["class_labels"] else None,
            preprocessing_info=json.loads(r["preprocessing_info"]) if r["preprocessing_info"] else None,
            is_active=bool(r["is_active"]),
            is_builtin=bool(r["is_builtin"]),
            validation_status=r["validation_status"],
            validation_notes=r["validation_notes"],
            created_at=str(r["created_at"])
        )


async def activate_model(model_id: str) -> bool:
    async with get_db_connection() as db:
        # First ensure the model exists and is valid
        cursor = await db.execute("SELECT id, validation_status FROM models WHERE id = ?", (model_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found.")
        if row["validation_status"] != "valid":
            raise HTTPException(status_code=400, detail="Cannot activate an unvalidated or invalid model. Please validate first.")

        await db.execute("UPDATE models SET is_active = 0")
        await db.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
        await db.commit()
        return True


async def deactivate_model(model_id: str) -> bool:
    async with get_db_connection() as db:
        await db.execute("UPDATE models SET is_active = 0 WHERE id = ?", (model_id,))
        await db.commit()
        return True


async def save_uploaded_model(
    file: UploadFile,
    name: str,
    architecture: str,
    num_channels: int,
    sampling_rate: float,
    class_labels: Optional[List[str]] = None,
    preprocessing_info: Optional[Dict[str, Any]] = None
) -> ModelRegistryItem:
    settings.EEG_MODEL_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(file.filename or "uploaded_model.pt")
    ext = Path(clean_name).suffix.lower()

    if ext not in [".pt", ".pth", ".onnx"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Must be .pt, .pth, or .onnx")

    model_id = f"model-{uuid.uuid4().hex[:8]}"
    dest_filename = f"{model_id}_{clean_name}"
    dest_path = settings.EEG_MODEL_UPLOAD_ROOT / dest_filename

    # Read with chunking and size limit enforcement
    total_bytes = 0
    max_bytes = settings.max_upload_size_bytes

    with open(dest_path, "wb") as out_f:
        while chunk := await file.read(1024 * 1024): # 1MB chunks
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                out_f.close()
                if dest_path.exists():
                    dest_path.unlink()
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeded maximum permitted size of {settings.MAX_UPLOAD_SIZE_MB} MB."
                )
            out_f.write(chunk)

    input_shape = f"[1, {num_channels}, 1001]"
    default_classes = class_labels or ["Class 0", "Class 1"]

    async with get_db_connection() as db:
        await db.execute("""
            INSERT INTO models (
                id, name, filename, filepath, format, architecture,
                num_channels, sampling_rate, input_shape, class_labels,
                preprocessing_info, is_active, is_builtin, validation_status, validation_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 'pending', 'Uploaded. Pending validation.')
        """, (
            model_id,
            name,
            clean_name,
            str(dest_path),
            ext,
            architecture,
            num_channels,
            sampling_rate,
            input_shape,
            json.dumps(default_classes),
            json.dumps(preprocessing_info or {})
        ))
        await db.commit()

    return ModelRegistryItem(
        id=model_id,
        name=name,
        filename=clean_name,
        filepath=str(dest_path),
        format=ext,
        architecture=architecture,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        input_shape=input_shape,
        class_labels=default_classes,
        preprocessing_info=preprocessing_info,
        is_active=False,
        is_builtin=False,
        validation_status="pending",
        validation_notes="Uploaded. Pending validation.",
        created_at=str(time.time())
    )


async def validate_model(model_id: str) -> ModelValidationResponse:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found.")

        filepath = row["filepath"]
        architecture = row["architecture"]
        num_channels = row["num_channels"]

    if not os.path.exists(filepath):
        return ModelValidationResponse(
            model_id=model_id,
            is_valid=False,
            architecture=architecture,
            num_channels=num_channels,
            sampling_rate=1000.0,
            message=f"File not found on disk at {filepath}"
        )

    # Perform architecture and weight validation
    try:
        t0 = time.perf_counter()
        if architecture.lower() == "eegnet":
            model, diag = load_eegnet_checkpoint(filepath, in_channels=num_channels, num_classes=2, device="cpu")
            test_input = torch.randn(1, num_channels, 1001)
            with torch.no_grad():
                out = model(test_input)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            is_valid = diag["is_fully_compatible"]
            msg = f"EEGNet validated successfully ({inference_ms:.1f}ms test inference)."
        elif architecture.lower() in ["cnn1d", "erpconv1dnet"]:
            model, diag = load_cnn1d_checkpoint(filepath, in_channels=num_channels, num_classes=2, device="cpu")
            test_input = torch.randn(1, num_channels, 1001)
            with torch.no_grad():
                out = model(test_input)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            is_valid = diag["is_fully_compatible"]
            msg = f"1D CNN validated successfully ({inference_ms:.1f}ms test inference)."
        else:
            is_valid = False
            inference_ms = None
            diag = {"missing_keys": [], "unexpected_keys": []}
            msg = f"Unknown or unsupported architecture: '{architecture}'."

        status_str = "valid" if is_valid else "warning"
        async with get_db_connection() as db:
            await db.execute("""
                UPDATE models
                SET validation_status = ?, validation_notes = ?
                WHERE id = ?
            """, (status_str, msg, model_id))
            await db.commit()

        return ModelValidationResponse(
            model_id=model_id,
            is_valid=is_valid,
            architecture=architecture,
            num_channels=num_channels,
            sampling_rate=row["sampling_rate"],
            missing_keys=diag.get("missing_keys", []),
            unexpected_keys=diag.get("unexpected_keys", []),
            test_inference_success=True,
            test_inference_ms=round(inference_ms, 2) if inference_ms else None,
            message=msg
        )

    except Exception as e:
        err_msg = f"Validation failed: {str(e)}"
        async with get_db_connection() as db:
            await db.execute("""
                UPDATE models
                SET validation_status = 'invalid', validation_notes = ?
                WHERE id = ?
            """, (err_msg, model_id))
            await db.commit()

        return ModelValidationResponse(
            model_id=model_id,
            is_valid=False,
            architecture=architecture,
            num_channels=num_channels,
            sampling_rate=row["sampling_rate"],
            message=err_msg
        )


async def delete_model(model_id: str) -> bool:
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found.")

        # Safety rule: Built-in research models MUST NEVER be deleted!
        if row["is_builtin"]:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Built-in research checkpoints cannot be deleted through the application."
            )

        filepath = Path(row["filepath"])
        # Ensure file is inside upload root before unlinking
        if settings.EEG_MODEL_UPLOAD_ROOT.resolve() in filepath.resolve().parents:
            if filepath.exists():
                filepath.unlink()

        await db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        await db.commit()
        return True
