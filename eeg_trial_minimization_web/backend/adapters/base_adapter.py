from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseEEGAdapter(ABC):
    """Abstract interface for all live and playback EEG acquisition adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to signal source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully release resources and close connections."""
        pass

    @abstractmethod
    def get_channels(self) -> List[str]:
        """Return list of active electrode channel names."""
        pass

    @abstractmethod
    def get_sampling_rate(self) -> float:
        """Return sampling frequency in Hz."""
        pass

    @abstractmethod
    def get_next_chunk(self, chunk_size: int = 50) -> Optional[Dict[str, Any]]:
        """
        Produce next chunk of EEG data.
        Returns dict containing:
        - 'samples': List[List[float]] with shape (num_channels, chunk_size)
        - 'timestamps': List[float]
        - 'events': List[Dict[str, Any]] (e.g., [{'sample_offset': 12, 'code': 1}])
        - 'is_simulated': bool
        - 'source_type': str
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return detailed status report and connection diagnostics."""
        pass
