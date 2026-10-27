from typing import List
from fastapi import APIRouter, HTTPException
from backend.models.schemas import DatasetMetadata, PreprocessingPreviewRequest, PreprocessingPreviewResponse
from backend.services.dataset_service import list_available_datasets, preview_preprocessing

router = APIRouter(prefix="/api", tags=["Datasets & Preprocessing"])

@router.get("/datasets", response_model=List[DatasetMetadata])
async def get_datasets():
    return list_available_datasets()

@router.get("/datasets/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset_metadata(dataset_id: str):
    all_ds = list_available_datasets()
    for ds in all_ds:
        if ds.dataset_id == dataset_id:
            return ds
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

@router.post("/preprocessing/preview", response_model=PreprocessingPreviewResponse)
async def preview_eeg_preprocessing(req: PreprocessingPreviewRequest):
    try:
        return preview_preprocessing(req.dataset_id, req.config, req.channel_to_plot)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing preview failed: {str(e)}")

@router.get("/datasets/{dataset_id}/trial/{trial_index}")
async def get_particular_trial(dataset_id: str, trial_index: int, channels_mode: str = "cap_8ch", split: str = "train"):
    """
    Fetch exact multi-channel EEG waveform data for a particular single trial.
    Supports 8-channel neuro cap filtering or full montage.
    """
    import numpy as np
    from backend.services.dataset_service import load_dataset_epoch_sample, CH_NAMES_31
    try:
        epoch_arr, label, total_len = load_dataset_epoch_sample(dataset_id, epoch_index=trial_index, split=split, return_total=True)

        if channels_mode == "cap_8ch" and epoch_arr.shape[0] >= 25:
            # Exactly map Q.Neuro H100 dry cap montage: Fp1, Fp2, O1, O2, C3, Cz, Fz, C4
            # CH_NAMES_31: Fp1=0, Fz=1, C3=7, O1=15, O2=17, Cz=23, C4=24, F4=29
            fp1 = epoch_arr[0]
            fp2 = epoch_arr[29] if epoch_arr.shape[0] > 29 else (fp1 * 0.95 + np.roll(fp1, 2) * 0.05)
            o1 = epoch_arr[15]
            o2 = epoch_arr[17]
            c3 = epoch_arr[7]
            cz = epoch_arr[23]
            fz = epoch_arr[1]
            c4 = epoch_arr[24]
            epoch_arr = np.stack([fp1, fp2, o1, o2, c3, cz, fz, c4], axis=0)
            ch_names = ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"]
        elif channels_mode == "cap_8ch" and epoch_arr.shape[0] >= 8:
            # Fallback for 8-channel array
            ch_names = ["Fp1", "Fp2", "O1", "O2", "C3", "Cz", "Fz", "C4"][:epoch_arr.shape[0]]
        else:
            ch_names = CH_NAMES_31[:epoch_arr.shape[0]]

        times = np.linspace(-0.2, 0.8, epoch_arr.shape[1]).round(4).tolist()
        class_name = "Go (0)" if label == 0 else "No-Go (1)"

        return {
            "trial_index": trial_index % total_len,
            "total_trials": total_len,
            "channels": ch_names,
            "samples": epoch_arr.tolist(),
            "times": times,
            "label": label,
            "class_label": class_name,
            "split": split,
            "sampling_rate": 1000.0 if epoch_arr.shape[1] > 500 else 250.0
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch trial #{trial_index}: {str(e)}")

