"""
DuckDB Configuration for Space Analytics Demo
"""
import os
from typing import Dict, Any

try:
    from .secure_config import get_secure_token
    SECURE_CONFIG_AVAILABLE = True
except ImportError:
    SECURE_CONFIG_AVAILABLE = False


class DuckDBConfig:
    """Configuration settings for DuckDB connections and setup."""
    
    def __init__(self):
        self.database_path = self._get_config("DUCKDB_DATABASE_PATH", "space_analytics.db")
        self.motherduck_token = self._get_config("MOTHERDUCK_TOKEN")
        self.max_memory = "4GB"
        self.threads = os.cpu_count() or 4
    
    def _get_config(self, key: str, default: str = None) -> str:
        """Get configuration value using secure hierarchy."""
        if SECURE_CONFIG_AVAILABLE:
            return get_secure_token(key) or default
        return os.getenv(key, default)
        
    def get_connection_config(self) -> Dict[str, Any]:
        """Get DuckDB connection configuration."""
        config = {
            'config': {
                'memory_limit': self.max_memory,
                'threads': self.threads,
            }
        }
        return config
    
    def get_motherduck_config(self) -> Dict[str, str]:
        """Get MotherDuck connection configuration."""
        if not self.motherduck_token:
            raise ValueError("MOTHERDUCK_TOKEN environment variable is required")
        
        return {
            'database': f'md:space_analytics?motherduck_token={self.motherduck_token}',
            'token': self.motherduck_token
        }
    
    def get_extensions(self) -> list[str]:
        """Get list of DuckDB extensions to install."""
        return [
            'httpfs',
            'parquet', 
            'json',
            'iceberg'
        ]


# Default configuration instance
duckdb_config = DuckDBConfig()