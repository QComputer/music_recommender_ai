"""
Pipeline module that orchestrates the full music recommendation workflow.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from music_recommender.config import Config
from music_recommender.scanner import Scanner
from music_recommender.metadata import MetadataExtractor
from music_recommender.features import FeatureExtractor, normalize_features
from music_recommender.embeddings import create_embedding_extractor
from music_recommender.storage import Storage
from music_recommender.recommender import Recommender


logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full music recommendation pipeline."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        
        # Components
        self.scanner = Scanner(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.embedding_extractor = create_embedding_extractor(self.config)
        self.storage = Storage(self.config)
        self.recommender = None
    
    def run_scan(
        self,
        root_dir: Optional[str] = None,
        compute_checksum: bool = False
    ) -> List[Dict[str, Any]]:
        """Run scan step.
        
        Args:
            root_dir: Music root directory.
            compute_checksum: Whether to compute file checksums.
            
        Returns:
            List of file information dictionaries.
        """
        logger.info("Running scan step")
        
        if root_dir:
            self.config.set("scanner.music_root", root_dir)
        
        files = self.scanner.scan(compute_checksum=compute_checksum)
        
        # Save manifest
        self.storage.save_manifest(files)
        
        logger.info(f"Scan complete: {len(files)} files")
        
        return files
    
    def run_extract(
        self,
        manifest_path: Optional[str] = None,
        parallel: bool = True,
        workers: Optional[int] = None
    ) -> pd.DataFrame:
        """Run extraction step.
        
        Args:
            manifest_path: Path to manifest file.
            parallel: Whether to use parallel processing.
            workers: Number of worker processes.
            
        Returns:
            Updated DataFrame with features.
        """
        logger.info("Running extraction step")
        
        if workers:
            self.config.set("features.n_workers", workers)
        
        # Load existing manifest
        if manifest_path:
            df = self.storage.load_manifest(manifest_path)
        else:
            df = self.storage.load_manifest()
        
        if df.empty:
            logger.error("No manifest found. Run scan first.")
            return df
        
        # Process each track
        from tqdm import tqdm
        
        track_ids = df["track_id"].tolist()
        file_paths = df["file_path"].tolist()
        
        results = []
        for track_id, file_path in tqdm(zip(track_ids, file_paths), total=len(track_ids)):
            try:
                # Extract metadata
                metadata = self.metadata_extractor.extract(file_path)
                
                # Extract features
                features = self.feature_extractor.extract_features_flat(file_path)
                
                # Extract embedding if enabled
                embedding = None
                if self.embedding_extractor.is_available:
                    embedding = self.embedding_extractor.extract_embedding(file_path)
                
                # Combine
                track_data = {
                    "track_id": track_id,
                    "file_path": file_path,
                }
                track_data.update(metadata)
                
                # Add feature values (not metadata)
                feature_names = self.feature_extractor.get_feature_names()
                for name in feature_names:
                    if name in features:
                        track_data[name] = features[name]
                
                # Mark embedding availability
                if embedding is not None:
                    track_data["has_embedding"] = True
                else:
                    track_data["has_embedding"] = False
                
                results.append(track_data)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    "track_id": track_id,
                    "file_path": file_path,
                    "error": str(e),
                })
        
        # Convert to DataFrame and save
        result_df = pd.DataFrame(results)
        self.storage.save_manifest(result_df.to_dict("records"))
        
        logger.info(f"Extraction complete: {len(result_df)} tracks processed")
        
        return result_df
    
    def run_build_index(
        self,
        manifest_path: Optional[str] = None,
        use_embeddings: bool = False
    ) -> Recommender:
        """Build similarity index.
        
        Args:
            manifest_path: Path to manifest file.
            use_embeddings: Whether to use embeddings.
            
        Returns:
            Trained recommender.
        """
        logger.info("Building similarity index")
        
        self.recommender = Recommender(self.config)
        self.recommender.build_index(manifest_path, use_embeddings=use_embeddings)
        
        logger.info(f"Index built with {self.recommender.track_count} tracks")
        
        return self.recommender
    
    def run_full_pipeline(
        self,
        root_dir: Optional[str] = None,
        use_embeddings: bool = False,
        compute_checksum: bool = False
    ) -> Recommender:
        """Run the full pipeline.
        
        Args:
            root_dir: Music root directory.
            use_embeddings: Whether to use embeddings.
            compute_checksum: Whether to compute checksums.
            
        Returns:
            Trained recommender.
        """
        # Scan
        self.run_scan(root_dir, compute_checksum)
        
        # Extract
        self.run_extract()
        
        # Build index
        recommender = self.run_build_index(use_embeddings=use_embeddings)
        
        logger.info("Pipeline complete")
        
        return recommender
    
    def get_recommendations(
        self,
        track_id: Optional[str] = None,
        file_path: Optional[str] = None,
        k: int = 10,
        use_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Get recommendations.
        
        Args:
            track_id: Track ID.
            file_path: Audio file path.
            k: Number of recommendations.
            use_embeddings: Whether to use embeddings.
            
        Returns:
            List of recommendations.
        """
        if self.recommender is None:
            self.recommender = self.run_build_index(use_embeddings=use_embeddings)
        
        return self.recommender.get_similar_tracks(
            file_path=file_path,
            track_id=track_id,
            k=k,
            use_embeddings=use_embeddings
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary of statistics.
        """
        stats = {
            "manifest_exists": self.storage.exists(),
            "track_count": self.storage.get_track_count(),
        }
        
        if stats["manifest_exists"]:
            df = self.storage.load_manifest()
            
            stats["total_tracks"] = len(df)
            
            if "genre" in df.columns:
                genres = df["genre"].dropna()
                if not genres.empty:
                    stats["unique_genres"] = int(genres.nunique())
            
            if "artist" in df.columns:
                artists = df["artist"].dropna()
                if not artists.empty:
                    stats["unique_artists"] = int(artists.nunique())
            
            # Features
            feature_cols = [c for c in df.columns if c.startswith(("mfcc_", "chroma_", "spectral_"))]
            stats["feature_count"] = len(feature_cols)
            
            # Embeddings
            if "has_embedding" in df.columns:
                stats["tracks_with_embeddings"] = int(df["has_embedding"].sum())
        
        return stats


def run_pipeline(
    config: Optional[Config] = None,
    root_dir: Optional[str] = None,
    use_embeddings: bool = False,
    backend: str = "sklearn"
) -> Recommender:
    """Run the full pipeline.
    
    Args:
        config: Configuration object.
        root_dir: Music root directory.
        use_embeddings: Whether to use embeddings.
        backend: Similarity search backend.
        
    Returns:
        Trained recommender.
    """
    if config is None:
        config = Config()
    
    if backend:
        config.set("recommender.backend", backend)
    
    pipeline = Pipeline(config)
    return pipeline.run_full_pipeline(root_dir, use_embeddings)
