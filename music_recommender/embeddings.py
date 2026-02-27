"""
Embedding extraction module using VGGish or YAMNet.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from music_recommender.config import Config


logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extractor for audio embeddings using VGGish or YAMNet."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize embedding extractor.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        
        # Settings
        self.enabled = self.config.get("embeddings.enabled", False)
        self.model_name = self.config.get("embeddings.model", None)
        self.embed_dim = self.config.get("embeddings.embed_dim", 128)
        self.aggregation = self.config.get("embeddings.aggregation", "mean")
        self.storage = self.config.get("embeddings.storage", "numpy")
        self.hdf5_path = self.config.get("embeddings.hdf5_path", "embeddings.h5")
        
        # Audio settings
        self.sample_rate = self.config.get("audio.sample_rate", 22050)
        
        # Model and session
        self._model = None
        self._session = None
        self._graph = None
        
        if self.enabled and self.model_name:
            self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the embedding model."""
        if not self.enabled:
            logger.info("Embedding extraction is disabled")
            return
        
        if self.model_name == "vggish":
            self._init_vggish()
        elif self.model_name == "yamnet":
            self._init_yamnet()
        else:
            logger.warning(f"Unknown embedding model: {self.model_name}")
            self.enabled = False
    
    def _init_vggish(self) -> None:
        """Initialize VGGish model."""
        try:
            # Try to import vggish
            import vggish
            import vggish_input
            import vggish_params
            
            self._vggish_params = vggish_params
            self._vggish_input = vggish_input
            self._vggish = vggish
            
            # Try to load model
            try:
                self._model = vggish.define_vggish_slim()
                # Try to load pretrained weights if available
                weights_path = vggish_params.EXAMPLE_PARAMS.get('checkpoint_filename')
                if weights_path and os.path.exists(weights_path):
                    vggish.load_vggish_slim_checkpoint(self._model, weights_path)
                
                self._embed_dim = 128
                logger.info("VGGish model initialized")
            except Exception as e:
                logger.warning(f"Could not initialize VGGish model: {e}")
                self._model = None
                self.enabled = False
                
        except ImportError:
            logger.warning("VGGish not available. Install with: pip install vggish")
            self.enabled = False
    
    def _init_yamnet(self) -> None:
        """Initialize YAMNet model."""
        try:
            # Try to import tensorflow and tensorflow_hub
            import tensorflow as tf
            import tensorflow_hub as hub
            
            self._tf = tf
            self._hub = hub
            
            # Try to load YAMNet from TensorFlow Hub
            try:
                # YAMNet model URL
                yamnet_url = "https://tfhub.dev/google/yamnet/1"
                self._model = hub.load(yamnet_url)
                self._embed_dim = 1024
                logger.info("YAMNet model initialized")
            except Exception as e:
                logger.warning(f"Could not load YAMNet model: {e}")
                self._model = None
                self.enabled = False
                
        except ImportError:
            logger.warning("TensorFlow or TensorFlow Hub not available for YAMNet")
            self.enabled = False
    
    def extract_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Extract embedding from audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Embedding array or None if extraction fails.
        """
        if not self.enabled or self._model is None:
            return None
        
        try:
            if self.model_name == "vggish":
                return self._extract_vggish(file_path)
            elif self.model_name == "yamnet":
                return self._extract_yamnet(file_path)
        except Exception as e:
            logger.error(f"Error extracting embedding from {file_path}: {e}")
            return None
    
    def _extract_vggish(self, file_path: str) -> Optional[np.ndarray]:
        """Extract VGGish embedding from audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Embedding array or None.
        """
        try:
            # Load audio using librosa
            import librosa
            y, sr = librosa.load(file_path, sr=self._vggish_params.SAMPLE_RATE, mono=True)
            
            # Convert to VGGish input format
            examples = self._vggish_input.waveform_to_examples(y, sr)
            
            if len(examples) == 0:
                logger.warning(f"No valid VGGish examples for {file_path}")
                return None
            
            # Get embeddings
            with self._tf.Session() as sess:
                self._model.initialize(sess)
                [embeddings] = sess.run(self._model['embeddings'], 
                                        feed_dict={self._model['features']: examples})
            
            # Aggregate
            return self._aggregate_embeddings(embeddings)
            
        except Exception as e:
            logger.error(f"VGGish extraction error: {e}")
            return None
    
    def _extract_yamnet(self, file_path: str) -> Optional[np.ndarray]:
        """Extract YAMNet embedding from audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Embedding array or None.
        """
        try:
            # Load audio using librosa
            import librosa
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Run YAMNet
            scores, embeddings, spectrogram = self._model(y)
            
            # embeddings shape: (n_frames, 1024)
            embeddings = embeddings.numpy()
            
            if len(embeddings) == 0:
                logger.warning(f"No valid YAMNet embeddings for {file_path}")
                return None
            
            # Aggregate
            return self._aggregate_embeddings(embeddings)
            
        except Exception as e:
            logger.error(f"YAMNet extraction error: {e}")
            return None
    
    def _aggregate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Aggregate embeddings using configured method.
        
        Args:
            embeddings: Array of embeddings (n_frames, embed_dim).
            
        Returns:
            Aggregated embedding.
        """
        if self.aggregation == "mean":
            return embeddings.mean(axis=0)
        elif self.aggregation == "max":
            return embeddings.max(axis=0)
        elif self.aggregation == "mean_max":
            mean_emb = embeddings.mean(axis=0)
            max_emb = embeddings.max(axis=0)
            return np.concatenate([mean_emb, max_emb])
        else:
            return embeddings.mean(axis=0)
    
    def extract_batch(
        self, 
        file_paths: List[str],
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings from multiple files.
        
        Args:
            file_paths: List of file paths.
            show_progress: Whether to show progress.
            
        Returns:
            Dictionary mapping file paths to embeddings.
        """
        results = {}
        
        iterator = file_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(file_paths, desc="Extracting embeddings")
            except ImportError:
                pass
        
        for file_path in iterator:
            embedding = self.extract_embedding(file_path)
            if embedding is not None:
                results[file_path] = embedding
        
        return results
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension.
        """
        if self.aggregation == "mean_max":
            return self.embed_dim * 2
        return self.embed_dim
    
    @property
    def is_available(self) -> bool:
        """Check if embedding extraction is available.
        
        Returns:
            True if embeddings can be extracted.
        """
        return self.enabled and self._model is not None


class CPUEmbeddingExtractor(EmbeddingExtractor):
    """Fallback embedding extractor using simple features for CPU-only systems."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize CPU embedding extractor."""
        # Force disable neural embeddings
        config = config or Config()
        config.set("embeddings.enabled", False)
        config.set("embeddings.model", None)
        
        super().__init__(config)
        
        # But enable our simple embeddings
        self._use_simple = True
        self.embed_dim = 64  # Simple embedding dimension
    
    def _init_model(self) -> None:
        """Initialize simple embedding extractor."""
        self.enabled = True
        self._model = "simple"
        logger.info("Using simple CPU embedding extractor")
    
    def extract_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Extract simple embedding from audio file.
        
        Uses MFCC statistics as a simple embedding.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Embedding array or None.
        """
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Extract MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Compute statistics
            mean_mfcc = mfccs.mean(axis=1)
            std_mfcc = mfccs.std(axis=1)
            max_mfcc = mfccs.max(axis=1)
            min_mfcc = mfccs.min(axis=1)
            
            # Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            mean_delta = delta_mfccs.mean(axis=1)
            std_delta = delta_mfccs.std(axis=1)
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mean_chroma = chroma.mean(axis=1)
            
            # Combine
            embedding = np.concatenate([
                mean_mfcc, std_mfcc, max_mfcc, min_mfcc,
                mean_delta, std_delta, mean_chroma
            ])
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting simple embedding: {e}")
            return None
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension.
        """
        return 64


def create_embedding_extractor(config: Optional[Config] = None) -> EmbeddingExtractor:
    """Create appropriate embedding extractor based on config.
    
    Args:
        config: Configuration object.
        
    Returns:
        Embedding extractor instance.
    """
    extractor = EmbeddingExtractor(config)
    
    if extractor.is_available:
        return extractor
    
    # Fall back to simple CPU extractor
    return CPUEmbeddingExtractor(config)
