from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    models_registered: int
    active_model: Optional[str] = None
    stream_active: bool = False
    timestamp: str

class SystemInfoResponse(BaseModel):
    os: str
    python_version: str
    torch_version: str
    cuda_available: bool
    device_name: str
    disk_free_gb: float
    paths: Dict[str, Any]

class InventoryItem(BaseModel):
    name: str
    category: str
    path: str
    size_mb: float
    file_type: str
    status: str
    description: str

class InventoryResponse(BaseModel):
    project_root: str
    items: List[InventoryItem]
    total_files: int
    total_size_mb: float

class DatasetMetadata(BaseModel):
    dataset_id: str
    name: str
    path: str
    file_type: str
    size_mb: float
    num_channels: int
    channel_names: List[str]
    sampling_rate: float
    duration_sec: Optional[float] = None
    n_epochs: Optional[int] = None
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    subjects: Optional[List[str]] = None
    task: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

class PreprocessingConfig(BaseModel):
    l_freq: float = 0.1
    h_freq: float = 40.0
    notch_freq: Optional[float] = None
    rereference: str = "average"
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Optional[List[float]] = [-0.2, 0.0]
    reject_uv: Optional[float] = 100.0

class PreprocessingPreviewRequest(BaseModel):
    dataset_id: str
    config: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    channel_to_plot: Optional[str] = None

class PreprocessingPreviewResponse(BaseModel):
    channel_name: str
    time_points: List[float]
    raw_signal: List[float]
    filtered_signal: List[float]
    retained_epochs: int
    rejected_epochs: int
    status: str
    message: str

class ModelRegistryItem(BaseModel):
    id: str
    name: str
    filename: str
    filepath: str
    format: str
    architecture: str
    num_channels: int
    sampling_rate: float
    input_shape: Optional[str] = None
    class_labels: Optional[List[str]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None
    is_active: bool
    is_builtin: bool
    validation_status: str
    validation_notes: Optional[str] = None
    created_at: str

class ModelValidationResponse(BaseModel):
    model_id: str
    is_valid: bool
    architecture: str
    num_channels: int
    sampling_rate: float
    missing_keys: List[str] = Field(default_factory=list)
    unexpected_keys: List[str] = Field(default_factory=list)
    test_inference_success: bool = False
    test_inference_ms: Optional[float] = None
    message: str

class SingleInferenceRequest(BaseModel):
    model_id: Optional[str] = None
    epoch_index: Optional[int] = None
    dataset_name: Optional[str] = "DS002680_gonogo_4"
    epoch_data: Optional[List[List[float]]] = None # Optional custom (C, T)

class SingleInferenceResponse(BaseModel):
    model_id: str
    model_name: str
    predicted_class: int
    predicted_label: str
    probabilities: List[float]
    inference_time_ms: float
    channel_count: int
    time_points: int
    actual_label: Optional[int] = None
    source_info: str

class BatchInferenceRequest(BaseModel):
    model_id: Optional[str] = None
    dataset_name: str = "DS002680_gonogo_4"
    split: str = "test"
    max_samples: int = 100

class BatchInferenceResponse(BaseModel):
    model_id: str
    model_name: str
    total_evaluated: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    avg_inference_time_ms: float
    predictions: List[Dict[str, Any]]

class ExperimentCreateRequest(BaseModel):
    name: str
    dataset_name: str = "DS002680"
    task_name: str = "gonogo_4"
    model_type: str = "EEGNet"
    model_id: Optional[str] = None
    trial_counts: List[Union[int, str]] = [5, 10, 20, 30, 50, 75, 100, 150, 200, "full"]
    num_folds: int = 5
    random_seed: int = 42
    cv_strategy: str = "stratified"
    epochs: int = 15
    batch_size: int = 32
    use_gan_augmentation: bool = False
    gan_synthetic_ratio: float = 0.5

class ExperimentSummary(BaseModel):
    id: str
    name: str
    dataset_name: str
    task_name: str
    model_type: str
    model_id: Optional[str] = None
    status: str
    progress: float
    current_step: Optional[str] = None
    trial_counts: List[Any]
    summary_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str

class StreamStatus(BaseModel):
    is_streaming: bool
    adapter_type: str
    sampling_rate: float
    channel_count: int
    channel_names: List[str]
    samples_transmitted: int
    dropped_chunks: int
    latency_ms: float
    last_event_code: Optional[int] = None
    source_label: str

class HardwareAdapterInfo(BaseModel):
    id: str
    name: str
    status: str
    description: str
    supported_channels: List[int]
    is_available: bool
    requires_external_service: bool
    config_params: Dict[str, Any]
