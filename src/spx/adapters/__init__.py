"""Adapter registry for different leagues and sports."""

from typing import Dict, List, Type

from .base import BaseAdapter


class AdapterRegistry:
    """Registry for data adapters."""
    
    _adapters: Dict[str, Type[BaseAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[BaseAdapter]) -> None:
        """Register an adapter.
        
        Args:
            name: Adapter name/identifier
            adapter_class: Adapter class
        """
        cls._adapters[name] = adapter_class
    
    @classmethod
    def get_adapter(cls, name: str) -> Type[BaseAdapter]:
        """Get adapter by name.
        
        Args:
            name: Adapter name
            
        Returns:
            Adapter class
            
        Raises:
            KeyError: If adapter not found
        """
        if name not in cls._adapters:
            available = list(cls._adapters.keys())
            raise KeyError(f"Adapter '{name}' not found. Available: {available}")
        
        return cls._adapters[name]
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapters.
        
        Returns:
            List of adapter names
        """
        return list(cls._adapters.keys())


# Import and register adapters
def register_all_adapters() -> None:
    """Register all available adapters."""
    try:
        from .football.epl import EPLAdapter
        AdapterRegistry.register("EPLAdapter", EPLAdapter)
        AdapterRegistry.register("epl", EPLAdapter)
    except ImportError:
        pass  # Adapter not available
    
    # Add more adapters here as they're created
    # from .football.bundesliga import BundesligaAdapter
    # AdapterRegistry.register("BundesligaAdapter", BundesligaAdapter)


# Register adapters on import
register_all_adapters()
