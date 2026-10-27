import os
from pathlib import Path
from typing import Optional, Union
from fastapi import HTTPException
from backend.core.config import settings

class SecurityError(Exception):
    pass

def sanitize_filename(filename: str) -> str:
    """Strip dangerous characters and directory separators from a filename."""
    name = os.path.basename(filename)
    # Remove any null bytes or control chars
    clean = "".join(c for c in name if c.isalnum() or c in "._- ")
    return clean.strip()

def validate_safe_path(target_path: Union[str, Path], allowed_roots: Optional[list] = None) -> Path:
    """
    Ensure the path is contained within one of the authorized directories,
    preventing path traversal attacks (e.g., ../../../windows/system32).
    """
    if allowed_roots is None:
        allowed_roots = [
            settings.EEG_PROJECT_ROOT,
            settings.EEG_WEB_ROOT,
            settings.EEG_RESULTS_ROOT,
            settings.EEG_MODEL_UPLOAD_ROOT,
        ]

    resolved_target = Path(target_path).resolve()
    for root in allowed_roots:
        resolved_root = Path(root).resolve()
        try:
            if resolved_target == resolved_root or resolved_root in resolved_target.parents:
                return resolved_target
        except ValueError:
            continue

    raise HTTPException(
        status_code=403,
        detail=f"Access denied: Path '{target_path}' is outside authorized directories."
    )

def validate_file_extension(filename: str, allowed_exts: Optional[set] = None) -> str:
    ext = Path(filename).suffix.lower()
    if allowed_exts is None:
        allowed_exts = settings.allowed_extensions_set
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(allowed_exts))}"
        )
    return ext
