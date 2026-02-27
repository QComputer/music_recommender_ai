"""
Configuration module for Music Recommender AI.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the music recommender."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        self._config: Dict[str, Any] = {}
        self._config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        default_config = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
        if os.path.exists(default_config):
            self.load_from_file(default_config)
        else:
            self._config = self._get_builtin_defaults()
    
    def _get_builtin_defaults(self) -> Dict[str, Any]:
        """Return built-in default configuration."""
        return {
            "scanner": {
                "music_root": "./music",
                "supported_formats": ["mp3", "flac", "wav", "m4a"],
                "output_dir": "./data",
                "manifest_filename": "manifest.parquet",
            },
            "metadata": {
                "normalize_strings": True,
                "fields": ["artist", "album", "title", "genre", "year", 
                          "track_number", "duration", "bitrate", "sample_rate"],
            },
            "audio": {
                "sample_rate": 22050,
                "mono": True,
                "duration_limit": None,
            },
            "features": {
                "n_mfcc": 20,
                "n_mels": 128,
                "extract_melspec": False,
                "n_workers": 4,
                "checkpoint_file": "extraction_checkpoint.json",
            },
            "embeddings": {
                "enabled": False,
                "model": None,
                "embed_dim": 128,
                "aggregation": "mean",
                "storage": "numpy",
                "hdf5_path": "embeddings.h5",
            },
            "recommender": {
                "normalization": "standard",
                "backend": "sklearn",
                "default_k": 10,
                "metadata_boost": 0.0,
                "use_embeddings": False,
            },
            "faiss": {
                "index_type": "Flat",
                "n_clusters": 100,
                "hnsw_m": 32,
                "hnsw_ef_construction": 200,
            },
            "hnswlib": {
                "M": 32,
                "ef_construction": 200,
                "ef": 100,
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "music_recommender.log",
            },
            "test": {
                "data_dir": "./tests/data",
                "sample_size": 10,
            },
        }

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Resolve relative paths
        self._resolve_paths()
    
    def _resolve_paths(self) -> None:
        """Resolve relative paths in configuration."""
        config_dir = os.path.dirname(os.path.abspath(self._config_path or ""))
        
        # Resolve music_root
        if "scanner" in self._config:
            music_root = self._config["scanner"].get("music_root", "./music")
            if not os.path.isabs(music_root):
                self._config["scanner"]["music_root"] = os.path.join(config_dir, music_root)
            
            output_dir = self._config["scanner"].get("output_dir", "./data")
            if not os.path.isabs(output_dir):
                self._config["scanner"]["output_dir"] = os.path.join(config_dir, output_dir)
        
        # Resolve embedding storage path
        if "embeddings" in self._config:
            hdf5_path = self._config["embeddings"].get("hdf5_path", "embeddings.h5")
            if not os.path.isabs(hdf5_path):
                self._config["embeddings"]["hdf5_path"] = os.path.join(config_dir, hdf5_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key (e.g., "scanner.music_root")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def scanner(self) -> Dict[str, Any]:
        """Get scanner configuration."""
        return self._config.get("scanner", {})
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata extraction configuration."""
        return self._config.get("metadata", {})
    
    @property
    def audio(self) -> Dict[str, Any]:
        """Get audio preprocessing configuration."""
        return self._config.get("audio", {})
    
    @property
    def features(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return self._config.get("features", {})
    
    @property
    def embeddings(self) -> Dict[str, Any]:
        """Get embedding extraction configuration."""
        return self._config.get("embeddings", {})
    
    @property
    def recommender(self) -> Dict[str, Any]:
        """Get recommender configuration."""
        return self._config.get("recommender", {})
    
    @property
    def faiss(self) -> Dict[str, Any]:
        """Get FAISS configuration."""
        return self._config.get("faiss", {})
    
    @property
    def hnswlib(self) -> Dict[str, Any]:
        """Get HNSWlib configuration."""
        return self._config.get("hnswlib", {})
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self._config.get("api", {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get("logging", {})
    
    @property
    def test(self) -> Dict[str, Any]:
        """Get test configuration."""
        return self._config.get("test", {})
    
    @property
    def music_root(self) -> str:
        """Get music root directory."""
        return self.get("scanner.music_root", "./music")
    
    @property
    def output_dir(self) -> str:
        """Get output directory."""
        return self.get("scanner.output_dir", "./data")
    
    @property
    def manifest_path(self) -> str:
        """Get manifest file path."""
        output_dir = self.output_dir
        filename = self.get("scanner.manifest_filename", "manifest.parquet")
        return os.path.join(output_dir, filename)
    
    @property
    def sample_rate(self) -> int:
        """Get target sample rate."""
        return self.get("audio.sample_rate", 22050)
    
    @property
    def n_mfcc(self) -> int:
        """Get number of MFCC coefficients."""
        return self.get("features.n_mfcc", 20)
    
    @property
    def backend(self) -> str:
        """Get similarity search backend."""
        return self.get("recommender.backend", "sklearn")
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save configuration.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Configuration instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def set_config(config: Config) -> None:
    """Set global configuration instance.
    
    Args:
        config: Configuration instance.
    """
    global _config_instance
    _config_instance = config
