"""Base adapter interface for data loading."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..core.schema import Fixture, Match


class BaseAdapter(ABC):
    """Base class for data adapters."""
    
    def __init__(self, data_dir: Path):
        """Initialize adapter.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
    
    @abstractmethod
    def load_matches(self, season: str) -> List[Match]:
        """Load historical matches for a season.
        
        Args:
            season: Season identifier (e.g., '2023-24')
            
        Returns:
            List of Match objects
        """
        pass
    
    @abstractmethod
    def load_fixtures(self, season: str) -> List[Fixture]:
        """Load upcoming fixtures for a season.
        
        Args:
            season: Season identifier
            
        Returns:
            List of Fixture objects
        """
        pass
    
    @abstractmethod
    def get_available_seasons(self) -> List[str]:
        """Get list of available seasons.
        
        Returns:
            List of season identifiers
        """
        pass
    
    @abstractmethod
    def validate_data_format(self, filepath: Path) -> bool:
        """Validate that data file matches expected format.
        
        Args:
            filepath: Path to data file
            
        Returns:
            True if format is valid
        """
        pass
