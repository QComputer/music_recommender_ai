"""
Feature extraction module for audio files using librosa.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from music_recommender.config import Config


logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extractor for audio features using librosa."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize feature extractor.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        
        # Audio settings
        self.sample_rate = self.config.get("audio.sample_rate", 22050)
        self.mono = self.config.get("audio.mono", True)
        self.duration_limit = self.config.get("audio.duration_limit", None)
        
        # Feature settings
        self.n_mfcc = self.config.get("features.n_mfcc", 20)
        self.n_mels = self.config.get("features.n_mels", 128)
        self.extract_melspec = self.config.get("features.extract_melspec", False)
        
        # Try to import librosa
        self._librosa = None
        self._init_librosa()
    
    def _init_librosa(self) -> None:
        """Initialize librosa."""
        try:
            import librosa
            self._librosa = librosa
            logger.info("Librosa loaded successfully")
        except ImportError:
            logger.error("Librosa not available. Feature extraction will fail.")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Tuple of (audio data, sample rate).
        """
        if self._librosa is None:
            raise RuntimeError("Librosa not available")
        
        try:
            # Load audio with resampling
            y, sr = self._librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono,
                duration=self.duration_limit
            )
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio from {file_path}: {e}")
            raise
    
    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """Extract all features from audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Dictionary of extracted features.
        """
        if self._librosa is None:
            raise RuntimeError("Librosa not available")
        
        features = {}
        
        try:
            # Load audio
            y, sr = self.load_audio(file_path)
            
            # Get duration
            features["duration"] = len(y) / sr
            
            # --- MFCCs ---
            mfccs = self._librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features["mfcc_mean"] = mfccs.mean(axis=1).tolist()
            features["mfcc_std"] = mfccs.std(axis=1).tolist()
            
            # --- Delta MFCCs ---
            delta_mfccs = self._librosa.feature.delta(mfccs)
            features["delta_mfcc_mean"] = delta_mfccs.mean(axis=1).tolist()
            features["delta_mfcc_std"] = delta_mfccs.std(axis=1).tolist()
            
            # --- Chroma ---
            chroma = self._librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma_mean"] = chroma.mean(axis=1).tolist()
            features["chroma_std"] = chroma.std(axis=1).tolist()
            
            # --- Spectral features ---
            # Spectral centroid
            spectral_centroid = self._librosa.feature.spectral_centroid(y=y, sr=sr)
            features["spectral_centroid_mean"] = float(spectral_centroid.mean())
            features["spectral_centroid_std"] = float(spectral_centroid.std())
            
            # Spectral bandwidth
            spectral_bandwidth = self._librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features["spectral_bandwidth_mean"] = float(spectral_bandwidth.mean())
            features["spectral_bandwidth_std"] = float(spectral_bandwidth.std())
            
            # Spectral rolloff
            spectral_rolloff = self._librosa.feature.spectral_rolloff(y=y, sr=sr)
            features["spectral_rolloff_mean"] = float(spectral_rolloff.mean())
            features["spectral_rolloff_std"] = float(spectral_rolloff.std())
            
            # Spectral contrast
            spectral_contrast = self._librosa.feature.spectral_contrast(y=y, sr=sr)
            features["spectral_contrast_mean"] = spectral_contrast.mean(axis=1).tolist()
            features["spectral_contrast_std"] = spectral_contrast.std(axis=1).tolist()
            
            # --- RMS energy ---
            rms = self._librosa.feature.rms(y=y)
            features["rms_mean"] = float(rms.mean())
            features["rms_std"] = float(rms.std())
            
            # --- Zero crossing rate ---
            zcr = self._librosa.feature.zero_crossing_rate(y)
            features["zcr_mean"] = float(zcr.mean())
            features["zcr_std"] = float(zcr.std())
            
            # --- Tempo and beats ---
            tempo, beats = self._librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)
            features["beat_count"] = int(beats)
            
            # --- Mel spectrogram (optional) ---
            if self.extract_melspec:
                mel_spec = self._librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
                mel_spec_db = self._librosa.power_to_db(mel_spec, ref=np.max)
                features["melspec_mean"] = mel_spec_db.mean(axis=1).tolist()
                features["melspec_std"] = mel_spec_db.std(axis=1).tolist()
                features["melspec_max"] = mel_spec_db.max(axis=1).tolist()
            
            features["success"] = True
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            features["success"] = False
            features["error"] = str(e)
        
        return features
    
    def extract_features_flat(self, file_path: str) -> Dict[str, Any]:
        """Extract features and flatten to single-level dict.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Dictionary with flattened feature values.
        """
        features = self.extract_features(file_path)
        
        if not features.get("success", False):
            return features
        
        flat_features = {}
        
        # Handle list features by expanding them
        for key, value in features.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    flat_features[f"{key}_{i}"] = v
            else:
                flat_features[key] = value
        
        return flat_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces.
        
        Returns:
            List of feature names.
        """
        names = []
        
        # MFCC
        for i in range(self.n_mfcc):
            names.append(f"mfcc_mean_{i}")
            names.append(f"mfcc_std_{i}")
        
        # Delta MFCC
        for i in range(self.n_mfcc):
            names.append(f"delta_mfcc_mean_{i}")
            names.append(f"delta_mfcc_std_{i}")
        
        # Chroma (12 pitch classes)
        for i in range(12):
            names.append(f"chroma_mean_{i}")
            names.append(f"chroma_std_{i}")
        
        # Spectral
        names.extend([
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
        ])
        
        # Spectral contrast (7 bands)
        for i in range(7):
            names.append(f"spectral_contrast_mean_{i}")
            names.append(f"spectral_contrast_std_{i}")
        
        # Energy and ZCR
        names.extend([
            "rms_mean",
            "rms_std",
            "zcr_mean",
            "zcr_std",
        ])
        
        # Tempo and beats
        names.extend(["tempo", "beat_count"])
        
        if self.extract_melspec:
            for i in range(self.n_mels):
                names.append(f"melspec_mean_{i}")
                names.append(f"melspec_std_{i}")
                names.append(f"melspec_max_{i}")
        
        return names
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension.
        
        Returns:
            Number of features.
        """
        return len(self.get_feature_names())


def features_to_array(features: Dict[str, Any]) -> np.ndarray:
    """Convert feature dict to numpy array for similarity search.
    
    Args:
        features: Dictionary of features.
        
    Returns:
        Feature array.
    """
    # Filter out non-numeric and metadata fields
    numeric_features = {}
    for key, value in features.items():
        if key in ["success", "error", "duration", "beat_count"]:
            continue
        if isinstance(value, (int, float)):
            numeric_features[key] = value
        elif isinstance(value, list):
            # Flatten list features
            for i, v in enumerate(value):
                numeric_features[f"{key}_{i}"] = v
    
    # Sort by key for consistent ordering
    sorted_keys = sorted(numeric_features.keys())
    return np.array([numeric_features[k] for k in sorted_keys], dtype=np.float32)


def normalize_features(
    features: np.ndarray, 
    method: str = "standard",
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize feature array.
    
    Args:
        features: Feature array (n_samples, n_features).
        method: Normalization method (standard, minmax, none).
        mean: Pre-computed mean (for standard normalization).
        std: Pre-computed std (for standard normalization).
        
    Returns:
        Tuple of (normalized features, mean, std).
    """
    if method == "none" or method is None:
        return features, np.zeros(features.shape[1]), np.ones(features.shape[1])
    
    if method == "standard":
        if mean is None:
            mean = features.mean(axis=0)
        if std is None:
            std = features.std(axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (features - mean) / std
        return normalized, mean, std
    
    elif method == "minmax":
        min_val = features.min(axis=0)
        max_val = features.max(axis=0)
        
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        
        normalized = (features - min_val) / range_val
        return normalized, min_val, range_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
