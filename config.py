"""
Configuration management for the NLQ system.
Handles environment-specific settings and resource allocation.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    connection_pool_size: int = 5
    chunk_size: int = 50000
    memory_limit: str = "2GB"
    threads: int = 4
    enable_parallel: bool = True
    cache_size: str = "512MB"
    db_path: str = "nlq.duckdb"


@dataclass
class ModelConfig:
    """LLM model configuration settings."""
    model_path: str = "models/llama-3-sqlcoder-8b.Q6_K.gguf"
    summarizer_model_path: str = "models/Mistral-7B-Instruct-v0.1.Q6_K.gguf"
    n_ctx: int = 2048
    n_threads: int = 6
    n_gpu_layers: int = 20
    temperature: float = 0.0
    max_tokens: int = 512
    verbose: bool = False
    summarizer_temperature: float = 0.2
    summarizer_max_tokens: int = 384


@dataclass
class CacheConfig:
    """Caching configuration settings."""
    enable_query_cache: bool = True
    enable_result_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    cache_dir: str = "cache"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/nlq_system.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_performance_logging: bool = True


@dataclass
class SystemConfig:
    """System and performance configuration."""
    max_memory_usage: float = 0.8  # 80% of available RAM
    enable_profiling: bool = False
    profiling_output_dir: str = "profiling"
    enable_monitoring: bool = True
    health_check_interval: int = 60  # seconds


class Config:
    """Main configuration class that manages all settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.cache = CacheConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
        
        # Auto-detect system resources and adjust settings
        self._auto_configure_resources()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def _auto_configure_resources(self):
        """Auto-configure settings based on available system resources."""
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Adjust database settings based on memory
        if memory_gb < 8:
            self.database.chunk_size = 25000
            self.database.memory_limit = "1GB"
            self.database.cache_size = "256MB"
            self.model.n_ctx = 1024
            self.model.n_gpu_layers = 10
        elif memory_gb >= 16:
            self.database.chunk_size = 100000
            self.database.memory_limit = "4GB"
            self.database.cache_size = "1GB"
            self.model.n_ctx = 4096
            self.model.n_gpu_layers = 35
        
        # Adjust threading based on CPU
        self.database.threads = min(cpu_count, 8)
        self.model.n_threads = min(cpu_count - 1, 8)
        
        print(f"Auto-configured for {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model settings
        if os.getenv("MODEL_PATH"):
            self.model.model_path = os.getenv("MODEL_PATH")
        if os.getenv("SUMMARIZER_MODEL_PATH"):
            self.model.summarizer_model_path = os.getenv("SUMMARIZER_MODEL_PATH")
        if os.getenv("MODEL_N_CTX"):
            self.model.n_ctx = int(os.getenv("MODEL_N_CTX"))
        if os.getenv("MODEL_N_GPU_LAYERS"):
            self.model.n_gpu_layers = int(os.getenv("MODEL_N_GPU_LAYERS"))
        
        # Database settings
        if os.getenv("DB_CHUNK_SIZE"):
            self.database.chunk_size = int(os.getenv("DB_CHUNK_SIZE"))
        if os.getenv("DB_MEMORY_LIMIT"):
            self.database.memory_limit = os.getenv("DB_MEMORY_LIMIT")
        
        # Logging settings
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
        if os.getenv("ENABLE_PROFILING"):
            self.system.enable_profiling = os.getenv("ENABLE_PROFILING").lower() == "true"
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if "database" in config_data:
                for key, value in config_data["database"].items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            if "model" in config_data:
                for key, value in config_data["model"].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)
            
            if "cache" in config_data:
                for key, value in config_data["cache"].items():
                    if hasattr(self.cache, key):
                        setattr(self.cache, key, value)
            
            if "logging" in config_data:
                for key, value in config_data["logging"].items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
            
            if "system" in config_data:
                for key, value in config_data["system"].items():
                    if hasattr(self.system, key):
                        setattr(self.system, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        config_data = {
            "database": self.database.__dict__,
            "model": self.model.__dict__,
            "cache": self.cache.__dict__,
            "logging": self.logging.__dict__,
            "system": self.system.__dict__
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "max_allowed_gb": (memory.total * self.system.max_memory_usage) / (1024**3)
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Check if SQL model file exists
        if not os.path.exists(self.model.model_path):
            errors.append(f"Model file not found: {self.model.model_path}")
        # Summarizer model is recommended; warn if missing but do not fail validation
        if not os.path.exists(self.model.summarizer_model_path):
            print(f"Warning: Summarizer model not found: {self.model.summarizer_model_path}. "
                  f"Will fallback to SQL model for summaries.")
        
        # Check memory limits
        memory_info = self.get_memory_info()
        if memory_info["available_gb"] < 2:
            errors.append("Insufficient memory available (< 2GB)")
        
        # Check chunk size is reasonable
        if self.database.chunk_size < 1000 or self.database.chunk_size > 1000000:
            errors.append(f"Chunk size {self.database.chunk_size} is outside reasonable range (1000-1000000)")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Global configuration instance
config = Config()