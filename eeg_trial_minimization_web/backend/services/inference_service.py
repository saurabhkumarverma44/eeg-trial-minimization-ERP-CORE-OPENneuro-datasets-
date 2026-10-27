import time
import json
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from fastapi import HTTPException
from backend.core.database import get_db_connection
from backend.models.schemas import SingleInferenceRequest, SingleInferenceResponse, BatchInferenceResponse
from backend.models.eegnet import EEGNet, load_eegnet_checkpoint
from backend.models.cnn1d import ERPConv1DNet, load_cnn1d_checkpoint
from backend.services.dataset_service import load_dataset_epoch_sample
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cached model instance in memory
_CACHED_MODEL: Optional[torch.nn.Module] = None
_CACHED_MODEL_ID: Optional[str] = None
_CACHED_MODEL_META: Optional[Dict[str, Any]] = None

def validate_channel_compatibility(expected_channels: int, actual_channels: int) -> None:
    """
    Enforces strict electrode count integrity.
    Prevents silent padding, duplication, or mismatched channel inference.
    """
    if actual_channels != expected_channels:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Electrode Channel Incompatibility Error: The selected model expects {expected_channels} channels, "
                f"but received data contains {actual_channels} channels. Direct inference between mismatched electrode configurations "
                f"is strictly prohibited to prevent erroneous predictions. A dedicated {actual_channels}-channel model must be "
                f"trained and validated for this montage."
            )
        )

async def get_or_load_model(model_id: Optional[str] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    global _CACHED_MODEL, _CACHED_MODEL_ID, _CACHED_MODEL_META

    async with get_db_connection() as db:
        if model_id:
            cursor = await db.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = await cursor.fetchone()
        else:
            # Load currently active model or default built-in
            cursor = await db.execute("SELECT * FROM models WHERE is_active = 1 LIMIT 1")
            row = await cursor.fetchone()
            if not row:
                cursor = await db.execute("SELECT * FROM models WHERE is_builtin = 1 LIMIT 1")
                row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="No active or built-in model available for inference.")

        m_id = row["id"]
        filepath = row["filepath"]
        architecture = row["architecture"]
        num_channels = row["num_channels"]

        if _CACHED_MODEL is not None and _CACHED_MODEL_ID == m_id:
            return _CACHED_MODEL, _CACHED_MODEL_META

        # Load fresh model
        if architecture.lower() == "eegnet":
            model, diag = load_eegnet_checkpoint(filepath, in_channels=num_channels, num_classes=2, device="cpu")
        elif architecture.lower() in ["cnn1d", "erpconv1dnet"]:
            model, diag = load_cnn1d_checkpoint(filepath, in_channels=num_channels, num_classes=2, device="cpu")
        else:
            raise HTTPException(status_code=400, detail=f"Architecture '{architecture}' unsupported for inference.")

        meta = {
            "id": m_id,
            "name": row["name"],
            "architecture": architecture,
            "num_channels": num_channels,
            "class_labels": json.loads(row["class_labels"]) if row["class_labels"] else ["Go (0)", "No-Go (1)"]
        }

        _CACHED_MODEL = model
        _CACHED_MODEL_ID = m_id
        _CACHED_MODEL_META = meta
        return _CACHED_MODEL, _CACHED_MODEL_META


async def run_single_inference(req: SingleInferenceRequest) -> SingleInferenceResponse:
    model, meta = await get_or_load_model(req.model_id)

    # 1. Acquire epoch data
    actual_label = None
    if req.epoch_data is not None:
        data_arr = np.array(req.epoch_data, dtype=np.float32)
        source_info = "Custom uploaded / stream epoch"
    else:
        ds_id = req.dataset_name or "DS002680_gonogo_4"
        ep_idx = req.epoch_index or 0
        data_arr, actual_label = load_dataset_epoch_sample(ds_id, epoch_index=ep_idx, split="test")
        source_info = f"Saved Dataset: {ds_id} (Test split, epoch #{ep_idx})"
        if meta["num_channels"] == 8 and data_arr.shape[0] >= 25:
            fp1 = data_arr[0]
            fp2 = data_arr[29] if data_arr.shape[0] > 29 else (fp1 * 0.95 + np.roll(fp1, 2) * 0.05)
            o1 = data_arr[15]
            o2 = data_arr[17]
            c3 = data_arr[7]
            cz = data_arr[23]
            fz = data_arr[1]
            c4 = data_arr[24]
            data_arr = np.stack([fp1, fp2, o1, o2, c3, cz, fz, c4], axis=0)
            source_info += " [8-Ch Neuro Cap Montage]"

    # Ensure shape is (C, T)
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(1, -1)
    channels, time_points = data_arr.shape

    # Auto-select model matching input channels if model was not explicitly specified
    if req.model_id is None and meta["num_channels"] != channels:
        try:
            async with get_db_connection() as db:
                cursor = await db.execute("SELECT id FROM models WHERE num_channels = ? AND is_active = 1 LIMIT 1", (channels,))
                row = await cursor.fetchone()
                if not row:
                    cursor = await db.execute("SELECT id FROM models WHERE num_channels = ? AND is_builtin = 1 LIMIT 1", (channels,))
                    row = await cursor.fetchone()
                if not row:
                    cursor = await db.execute("SELECT id FROM models WHERE num_channels = ? LIMIT 1", (channels,))
                    row = await cursor.fetchone()
                if row:
                    model, meta = await get_or_load_model(row["id"])
        except Exception:
            pass

    # 2. Strict Channel Compatibility Check (Section 11)
    validate_channel_compatibility(expected_channels=meta["num_channels"], actual_channels=channels)


    # Resample time dimension to 1001 points if necessary
    if time_points != 1001:
        from scipy.signal import resample
        data_arr = resample(data_arr, 1001, axis=1).astype(np.float32)
        time_points = 1001

    # 3. Model execution
    tensor_in = torch.from_numpy(data_arr).unsqueeze(0).float() # (1, C, T)


    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred_class = int(np.argmax(probs))
    class_labels = meta["class_labels"]
    pred_label = class_labels[pred_class] if pred_class < len(class_labels) else f"Class {pred_class}"

    # 4. Log prediction to database
    async with get_db_connection() as db:
        await db.execute("""
            INSERT INTO prediction_logs (
                model_id, model_name, source_type, epoch_index,
                predicted_class, predicted_label, class_probabilities, inference_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            meta["id"],
            meta["name"],
            source_info,
            req.epoch_index,
            pred_class,
            pred_label,
            json.dumps([round(float(p), 4) for p in probs]),
            round(elapsed_ms, 2)
        ))
        await db.commit()

    return SingleInferenceResponse(
        model_id=meta["id"],
        model_name=meta["name"],
        predicted_class=pred_class,
        predicted_label=pred_label,
        probabilities=[round(float(p), 4) for p in probs],
        inference_time_ms=round(elapsed_ms, 2),
        channel_count=channels,
        time_points=time_points,
        actual_label=actual_label,
        source_info=source_info
    )


async def run_batch_inference(
    model_id: Optional[str] = None,
    dataset_name: str = "DS002680_gonogo_4",
    split: str = "test",
    max_samples: int = 100
) -> BatchInferenceResponse:
    from backend.core.config import settings
    model, meta = await get_or_load_model(model_id)

    root = settings.EEG_PROJECT_ROOT
    prep_file = root / "results_dl_1" / "datasets" / f"{dataset_name}_prepared.pt"
    if not prep_file.exists():
        candidates = list((root / "results_dl_1" / "datasets").glob(f"*{dataset_name}*_prepared.pt"))
        if candidates:
            prep_file = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"Prepared dataset '{dataset_name}' not found.")

    payload = torch.load(prep_file, map_location="cpu", weights_only=False)
    split_key = split if split in payload else ("test" if "test" in payload else "train")
    X, y = payload[split_key][0], payload[split_key][1]

    X_np = X.numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

    total_samples = min(len(y_np), max_samples)
    X_sub = X_np[:total_samples]
    y_sub = y_np[:total_samples]

    validate_channel_compatibility(expected_channels=meta["num_channels"], actual_channels=X_sub.shape[1])

    t0 = time.perf_counter()
    tensor_in = torch.from_numpy(X_sub).float() # (N, C, T)
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=-1).numpy()
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    preds = np.argmax(probs, axis=-1)

    acc = float(accuracy_score(y_sub, preds))
    bacc = float(balanced_accuracy_score(y_sub, preds))
    prec = float(precision_score(y_sub, preds, average="binary", zero_division=0))
    rec = float(recall_score(y_sub, preds, average="binary", zero_division=0))
    f1 = float(f1_score(y_sub, preds, average="binary", zero_division=0))
    cm = confusion_matrix(y_sub, preds).tolist()

    sample_preds = []
    for i in range(min(total_samples, 20)):
        sample_preds.append({
            "index": i,
            "actual": int(y_sub[i]),
            "predicted": int(preds[i]),
            "probabilities": [round(float(p), 4) for p in probs[i]],
            "correct": bool(y_sub[i] == preds[i])
        })

    return BatchInferenceResponse(
        model_id=meta["id"],
        model_name=meta["name"],
        total_evaluated=total_samples,
        accuracy=round(acc, 4),
        balanced_accuracy=round(bacc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        confusion_matrix=cm,
        avg_inference_time_ms=round(total_time_ms / total_samples, 2),
        predictions=sample_preds
    )
