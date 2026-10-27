import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Core directories
    EEG_PROJECT_ROOT: Path = Field(default=Path(r"D:\erp_trial_min_backup_20260626\erp_trial_min"))
    EEG_WEB_ROOT: Path = Field(default=Path(r"D:\erp_trial_min_backup_20260626\erp_trial_min\eeg_trial_minimization_web"))
    EEG_RESULTS_ROOT: Path = Field(default=Path(r"D:\erp_trial_min_backup_20260626\erp_trial_min\eeg_trial_minimization_web\storage\results"))
    EEG_MODEL_UPLOAD_ROOT: Path = Field(default=Path(r"D:\erp_trial_min_backup_20260626\erp_trial_min\eeg_trial_minimization_web\storage\uploads"))
    EEG_DATABASE_PATH: Path = Field(default=Path(r"D:\erp_trial_min_backup_20260626\erp_trial_min\eeg_trial_minimization_web\storage\database.sqlite"))

    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    # Security & Constraints
    MAX_UPLOAD_SIZE_MB: int = 250
    ALLOWED_EXTENSIONS: str = ".pt,.pth,.onnx,.fif,.edf,.bdf,.vhdr"
    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def allowed_extensions_set(self) -> set:
        return {ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()}

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def validate_paths(self) -> dict:
        """Inspect and return validation state of all critical paths."""
        return {
            "project_root_exists": self.EEG_PROJECT_ROOT.exists(),
            "project_root_path": str(self.EEG_PROJECT_ROOT),
            "web_root_exists": self.EEG_WEB_ROOT.exists(),
            "web_root_path": str(self.EEG_WEB_ROOT),
            "results_root_exists": self.EEG_RESULTS_ROOT.exists(),
            "results_root_path": str(self.EEG_RESULTS_ROOT),
            "model_upload_root_exists": self.EEG_MODEL_UPLOAD_ROOT.exists(),
            "model_upload_root_path": str(self.EEG_MODEL_UPLOAD_ROOT),
            "database_dir_exists": self.EEG_DATABASE_PATH.parent.exists(),
            "database_path": str(self.EEG_DATABASE_PATH),
        }

settings = Settings()
