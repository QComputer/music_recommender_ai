"""
Recommender module with multiple similarity search backends.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from music_recommender.config import Config
from music_recommender.features import normalize_features
from music_recommender.storage import Storage


logger = logging.getLogger(__name__)


class Recommender:
    """Content-based music recommender with multiple backends."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize recommender.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.storage = Storage(self.config)
        
        # Settings
        self.normalization = self.config.get("recommender.normalization", "standard")
        self.backend = self.config.get("recommender.backend", "sklearn")
        self.default_k = self.config.get("recommender.default_k", 10)
        self.metadata_boost = self.config.get("recommender.metadata_boost", 0.0)
        self.use_embeddings = self.config.get("recommender.use_embeddings", False)
        
        # State
        self._index = None
        self._features = None
        self._embeddings = None
        self._track_ids = None
        self._manifest = None
        self._mean = None
        self._std = None
        self._embedding_dim = None
        
        # Initialize backend
        self._backend_obj = None
        self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize the similarity search backend."""
        if self.backend == "sklearn":
            self._init_sklearn()
        elif self.backend == "faiss":
            self._init_faiss()
        elif self.backend == "hnswlib":
            self._init_hnswlib()
        else:
            logger.warning(f"Unknown backend: {self.backend}, using sklearn")
            self._init_sklearn()
    
    def _init_sklearn(self) -> None:
        """Initialize scikit-learn backend."""
        try:
            from sklearn.neighbors import NearestNeighbors
            self._backend_obj = NearestNeighbors(
                n_neighbors=self.default_k + 1,  # +1 because query track might be in index
                metric="cosine",
                algorithm="brute"  # brute force for accuracy
            )
            logger.info("Using sklearn NearestNeighbors backend")
        except ImportError:
            logger.error("scikit-learn not available")
            self._backend_obj = None
    
    def _init_faiss(self) -> None:
        """Initialize FAISS backend."""
        try:
            import faiss
            self._faiss = faiss
            logger.info("FAISS backend available")
        except ImportError:
            logger.warning("FAISS not available, falling back to sklearn")
            self.backend = "sklearn"
            self._init_sklearn()
    
    def _init_hnswlib(self) -> None:
        """Initialize HNSWlib backend."""
        try:
            import hnswlib
            self._hnswlib = hnswlib
            logger.info("HNSWlib backend available")
        except ImportError:
            logger.warning("HNSWlib not available, falling back to sklearn")
            self.backend = "sklearn"
            self._init_sklearn()
    
    def load_data(
        self, 
        manifest_path: Optional[str] = None,
        load_embeddings: bool = False
    ) -> None:
        """Load manifest and build feature matrix.
        
        Args:
            manifest_path: Path to manifest Parquet file.
            load_embeddings: Whether to load embeddings.
        """
        # Load manifest
        if manifest_path is None:
            manifest_path = self.storage.manifest_path
        
        self._manifest = self.storage.load_manifest(manifest_path)
        
        if self._manifest.empty:
            logger.warning("Manifest is empty")
            return
        
        # Get track IDs
        self._track_ids = self._manifest["track_id"].tolist()
        
        # Build feature matrix
        feature_cols = [c for c in self._manifest.columns 
                       if c.startswith(("mfcc_", "chroma_", "spectral_", "rms_", "zcr_", "tempo"))]
        
        if feature_cols:
            self._features = self._manifest[feature_cols].values.astype(np.float32)
            logger.info(f"Loaded {len(feature_cols)} audio features")
        else:
            logger.warning("No feature columns found in manifest")
            self._features = None
        
        # Normalize features
        if self._features is not None and self.normalization != "none":
            self._features, self._mean, self._std = normalize_features(
                self._features, method=self.normalization
            )
        
        # Load embeddings if requested
        if load_embeddings and self.use_embeddings:
            self._embeddings = self.storage.load_embeddings(self._track_ids)
            if self._embeddings:
                self._embedding_dim = next(iter(self._embeddings.values())).shape[0]
                logger.info(f"Loaded embeddings with dim {self._embedding_dim}")
    
    def build_index(
        self, 
        manifest_path: Optional[str] = None,
        use_embeddings: bool = False
    ) -> None:
        """Build similarity index from features.
        
        Args:
            manifest_path: Path to manifest Parquet file.
            use_embeddings: Whether to use embeddings instead of features.
        """
        self.load_data(manifest_path, load_embeddings=use_embeddings)
        
        if self._manifest is None or self._manifest.empty:
            logger.error("No data to build index from")
            return
        
        # Determine what to index
        if use_embeddings and self._embeddings:
            data_matrix = np.array(list(self._embeddings.values()))
            logger.info(f"Building index on embeddings ({data_matrix.shape})")
        elif self._features is not None:
            data_matrix = self._features
            logger.info(f"Building index on features ({data_matrix.shape})")
        else:
            logger.error("No feature or embedding data available")
            return
        
        # Build index based on backend
        if self.backend == "sklearn":
            self._build_sklearn_index(data_matrix)
        elif self.backend == "faiss":
            self._build_faiss_index(data_matrix)
        elif self.backend == "hnswlib":
            self._build_hnswlib_index(data_matrix)
        
        # Save index
        self.storage.save_index(self._index)
        logger.info("Index built and saved")
    
    def _build_sklearn_index(self, data_matrix: np.ndarray) -> None:
        """Build sklearn NearestNeighbors index."""
        self._backend_obj = self._backend_obj.__class__(
            n_neighbors=min(self.default_k + 1, len(data_matrix)),
            metric="cosine",
            algorithm="brute"
        )
        self._backend_obj.fit(data_matrix)
        self._index = self._backend_obj
    
    def _build_faiss_index(self, data_matrix: np.ndarray) -> None:
        """Build FAISS index."""
        # Normalize for cosine similarity
        norms = np.linalg.norm(data_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        data_normalized = data_matrix / norms
        
        d = data_matrix.shape[1]
        
        # Choose index type based on size
        n_vectors = len(data_matrix)
        
        if n_vectors < 10000:
            # Use exact search for small libraries
            index = self._faiss.IndexFlatIP(d)
        else:
            # Use IVF for larger libraries
            n_clusters = min(100, n_vectors // 100)
            quantizer = self._faiss.IndexFlatIP(d)
            index = self._faiss.IndexIVFFlat(quantizer, d, n_clusters)
            index.train(data_normalized)
        
        index.add(data_normalized.astype(np.float32))
        
        self._index = index
        self._data_normalized = data_normalized
    
    def _build_hnswlib_index(self, data_matrix: np.ndarray) -> None:
        """Build HNSWlib index."""
        # Get HNSW settings
        M = self.config.get("hnswlib.M", 32)
        ef_construction = self.config.get("hnswlib.ef_construction", 200)
        ef = self.config.get("hnswlib.ef", 100)
        
        dim = data_matrix.shape[1]
        num_elements = len(data_matrix)
        
        index = self._hnswlib.HnswIndex(space='cosine', dim=dim)
        index.set_ef(ef)
        index.init_index(max_elements=num_elements, M=M, ef_construction=ef_construction)
        
        # Add items
        index.add_items(data_matrix)
        
        self._index = index
    
    def recommend(
        self, 
        track_id: str, 
        k: Optional[int] = None,
        use_embeddings: bool = False,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recommendations for a track.
        
        Args:
            track_id: Track ID to find recommendations for.
            k: Number of recommendations.
            use_embeddings: Whether to use embeddings.
            exclude_self: Whether to exclude the query track.
            
        Returns:
            List of recommendation dictionaries with track info and scores.
        """
        k = k or self.default_k
        
        if self._manifest is None or self._manifest.empty:
            logger.error("No index loaded")
            return []
        
        # Find track index
        try:
            track_idx = self._track_ids.index(track_id)
        except ValueError:
            logger.error(f"Track ID not found: {track_id}")
            return []
        
        # Get feature vector
        if use_embeddings and self._embeddings:
            if track_id not in self._embeddings:
                logger.warning(f"Embedding not found for track: {track_id}")
                return []
            query = self._embeddings[track_id].reshape(1, -1)
        elif self._features is not None:
            query = self._features[track_idx].reshape(1, -1)
        else:
            logger.error("No features or embeddings available")
            return []
        
        # Search
        if self.backend == "sklearn":
            distances, indices = self._search_sklearn(query, k + 1)
        elif self.backend == "faiss":
            distances, indices = self._search_faiss(query, k + 1)
        elif self.backend == "hnswlib":
            distances, indices = self._search_hnswlib(query, k + 1)
        else:
            logger.error(f"Unknown backend: {self.backend}")
            return []
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip self
            if exclude_self and self._track_ids[idx] == track_id:
                continue
            
            if len(results) >= k:
                break
            
            # Get track info
            track_id_result = self._track_ids[idx]
            track_info = self._manifest[
                self._manifest["track_id"] == track_id_result
            ].iloc[0].to_dict()
            
            # Convert distance to similarity (0 = identical, 2 = opposite)
            similarity = 1 - dist
            
            # Apply metadata boost if configured
            if self.metadata_boost > 0:
                similarity = self._apply_metadata_boost(
                    track_id, track_id_result, similarity
                )
            
            results.append({
                "track_id": track_id_result,
                "track_info": track_info,
                "distance": float(dist),
                "similarity": float(similarity),
            })
        
        return results
    
    def _search_sklearn(
        self, 
        query: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using sklearn."""
        if self._index is None:
            # Build index if not built
            if self._features is not None:
                self._build_sklearn_index(self._features)
            else:
                raise ValueError("No features available")
        
        distances, indices = self._index.kneighbors(query, n_neighbors=k)
        return distances, indices
    
    def _search_faiss(
        self, 
        query: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS."""
        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query_normalized = query / norm
        else:
            query_normalized = query
        
        if hasattr(self._index, 'search'):
            distances, indices = self._index.search(
                query_normalized.astype(np.float32), k
            )
            return distances, indices.astype(np.int64)
        else:
            # For exact search
            distances, indices = self._index.search(
                query_normalized.astype(np.float32), k
            )
            return distances, indices
    
    def _search_hnswlib(
        self, 
        query: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using HNSWlib."""
        labels, distances = self._index.knn_query(query, k=k)
        
        # Convert distances to proper format (HNSW returns distances, not 1-dist)
        return distances.reshape(1, -1), labels.reshape(1, -1)
    
    def _apply_metadata_boost(
        self, 
        seed_track_id: str, 
        candidate_track_id: str,
        similarity: float
    ) -> float:
        """Apply metadata-based similarity boost.
        
        Args:
            seed_track_id: Seed track ID.
            candidate_track_id: Candidate track ID.
            similarity: Current similarity score.
            
        Returns:
            Boosted similarity score.
        """
        # Get track metadata
        seed = self._manifest[self._manifest["track_id"] == seed_track_id].iloc[0]
        candidate = self._manifest[
            self._manifest["track_id"] == candidate_track_id
        ].iloc[0]
        
        # Check genre match
        genre_match = 0
        if pd.notna(seed.get("genre")) and pd.notna(candidate.get("genre")):
            if str(seed["genre"]).lower() == str(candidate["genre"]).lower():
                genre_match = 0.2
        
        # Check artist match
        artist_match = 0
        if pd.notna(seed.get("artist")) and pd.notna(candidate.get("artist")):
            if str(seed["artist"]).lower() == str(candidate["artist"]).lower():
                artist_match = 0.3
        
        # Apply boost
        boost = genre_match + artist_match
        boosted = similarity + (boost * self.metadata_boost)
        
        return min(1.0, boosted)
    
    def recommend_batch(
        self, 
        track_ids: List[str],
        k: Optional[int] = None,
        use_embeddings: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get recommendations for multiple tracks.
        
        Args:
            track_ids: List of track IDs.
            k: Number of recommendations per track.
            use_embeddings: Whether to use embeddings.
            
        Returns:
            Dictionary mapping track_id to list of recommendations.
        """
        results = {}
        for track_id in track_ids:
            results[track_id] = self.recommend(
                track_id, k=k, use_embeddings=use_embeddings
            )
        return results
    
    def get_similar_tracks(
        self, 
        file_path: str = None,
        track_id: str = None,
        k: Optional[int] = None,
        use_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Get similar tracks by file path or track ID.
        
        Args:
            file_path: Path to audio file (used to compute track_id).
            track_id: Track ID.
            k: Number of recommendations.
            use_embeddings: Whether to use embeddings.
            
        Returns:
            List of similar tracks.
        """
        from music_recommender.scanner import Scanner
        
        if track_id is None and file_path is None:
            raise ValueError("Either file_path or track_id must be provided")
        
        if track_id is None:
            # Compute track_id from file
            scanner = Scanner(self.config)
            track_id = scanner.compute_track_id(file_path)
        
        return self.recommend(track_id, k=k, use_embeddings=use_embeddings)
    
    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded.
        
        Returns:
            True if index is available.
        """
        return self._index is not None or self._features is not None
    
    @property
    def track_count(self) -> int:
        """Get number of tracks in index.
        
        Returns:
            Number of tracks.
        """
        return len(self._track_ids) if self._track_ids else 0


def create_recommender(
    config: Optional[Config] = None,
    manifest_path: Optional[str] = None,
    use_embeddings: bool = False
) -> Recommender:
    """Create and initialize recommender.
    
    Args:
        config: Configuration object.
        manifest_path: Path to manifest file.
        use_embeddings: Whether to use embeddings.
        
    Returns:
        Initialized recommender.
    """
    recommender = Recommender(config)
    recommender.build_index(manifest_path, use_embeddings=use_embeddings)
    return recommender
