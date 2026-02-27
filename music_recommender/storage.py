"""
Storage module for managing data persistence.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from music_recommender.config import Config


logger = logging.getLogger(__name__)


class Storage:
    """Storage manager for Parquet manifests and embeddings."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize storage manager.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.output_dir = self.config.output_dir
        self.manifest_filename = self.config.get(
            "scanner.manifest_filename", "manifest.parquet"
        )
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def manifest_path(self) -> str:
        """Get manifest file path."""
        return os.path.join(self.output_dir, self.manifest_filename)
    
    @property
    def embeddings_dir(self) -> str:
        """Get embeddings directory."""
        return os.path.join(self.output_dir, "embeddings")
    
    def save_manifest(
        self, 
        data: List[Dict[str, Any]], 
        path: Optional[str] = None
    ) -> str:
        """Save manifest to Parquet file (or CSV as fallback).
        
        Args:
            data: List of track dictionaries.
            path: Optional custom path. Uses default if None.
            
        Returns:
            Path to saved file.
        """
        if path is None:
            path = self.manifest_path
        
        df = pd.DataFrame(data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Try parquet first, fall back to CSV
        parquet_path = path
        csv_path = path.replace('.parquet', '.csv') if path.endswith('.parquet') else path + '.csv'
        
        try:
            # Save to Parquet
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Manifest saved to: {parquet_path}")
            return parquet_path
        except Exception as e:
            logger.warning(f"Could not save as Parquet, falling back to CSV: {e}")
            try:
                df.to_csv(csv_path, index=False)
                logger.info(f"Manifest saved to: {csv_path}")
                return csv_path
            except Exception as e2:
                logger.error(f"Could not save manifest: {e2}")
                raise
    
    def load_manifest(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load manifest from Parquet file (or CSV as fallback).
        
        Args:
            path: Optional custom path. Uses default if None.
            
        Returns:
            DataFrame with track data.
        """
        if path is None:
            path = self.manifest_path
        
        # Try parquet first, then CSV
        if not os.path.exists(path):
            # Try CSV version
            csv_path = path.replace('.parquet', '.csv') if path.endswith('.parquet') else path + '.csv'
            if os.path.exists(csv_path):
                path = csv_path
            else:
                logger.warning(f"Manifest not found: {path}")
                return pd.DataFrame()
        
        try:
            df = pd.read_parquet(path)
            logger.info(f"Manifest loaded: {len(df)} tracks from {path}")
            return df
        except Exception as e:
            logger.warning(f"Could not read Parquet, trying CSV: {e}")
            try:
                df = pd.read_csv(path)
                logger.info(f"Manifest loaded: {len(df)} tracks from {path}")
                return df
            except Exception as e2:
                logger.warning(f"Could not load manifest: {e2}")
                return pd.DataFrame()
    
    def update_manifest(
        self, 
        updates: List[Dict[str, Any]],
        path: Optional[str] = None
    ) -> pd.DataFrame:
        """Update manifest with new or modified tracks.
        
        Args:
            updates: List of track dictionaries with updates.
            path: Optional custom path. Uses default if None.
            
        Returns:
            Updated DataFrame.
        """
        if path is None:
            path = self.manifest_path
        
        # Load existing manifest or create new
        if os.path.exists(path):
            df = self.load_manifest(path)
        else:
            df = pd.DataFrame()
        
        # Convert updates to DataFrame
        update_df = pd.DataFrame(updates)
        
        if df.empty:
            # New manifest
            df = update_df
        else:
            # Merge: update existing, add new
            if "track_id" in df.columns and "track_id" in update_df.columns:
                # Create index from track_id
                existing_ids = set(df["track_id"])
                new_updates = update_df[~update_df["track_id"].isin(existing_ids)]
                
                if not new_updates.empty:
                    df = pd.concat([df, new_updates], ignore_index=True)
        
        # Save updated manifest
        self.save_manifest(df.to_dict("records"), path)
        
        return df
    
    def save_embeddings_numpy(
        self, 
        embeddings: Dict[str, np.ndarray],
        prefix: str = ""
    ) -> List[str]:
        """Save embeddings as numpy files.
        
        Args:
            embeddings: Dictionary mapping track_id to embedding array.
            prefix: Optional prefix for filenames.
            
        Returns:
            List of saved file paths.
        """
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        saved_paths = []
        for track_id, embedding in embeddings.items():
            filename = f"{prefix}{track_id}.npy"
            filepath = os.path.join(self.embeddings_dir, filename)
            np.save(filepath, embedding)
            saved_paths.append(filepath)
        
        logger.info(f"Saved {len(saved_paths)} embeddings to {self.embeddings_dir}")
        return saved_paths
    
    def load_embeddings_numpy(
        self, 
        track_ids: List[str],
        prefix: str = ""
    ) -> Dict[str, np.ndarray]:
        """Load embeddings from numpy files.
        
        Args:
            track_ids: List of track IDs to load.
            prefix: Optional prefix for filenames.
            
        Returns:
            Dictionary mapping track_id to embedding array.
        """
        embeddings = {}
        
        for track_id in track_ids:
            filename = f"{prefix}{track_id}.npy"
            filepath = os.path.join(self.embeddings_dir, filename)
            
            if os.path.exists(filepath):
                embeddings[track_id] = np.load(filepath)
        
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings
    
    def save_embeddings_hdf5(
        self, 
        embeddings: Dict[str, np.ndarray],
        key: str = "embeddings"
    ) -> str:
        """Save embeddings to HDF5 file.
        
        Args:
            embeddings: Dictionary mapping track_id to embedding array.
            key: Dataset key in HDF5 file.
            
        Returns:
            Path to saved file.
        """
        import h5py
        
        hdf5_path = self.config.get("embeddings.hdf5_path", "embeddings.h5")
        filepath = os.path.join(self.output_dir, hdf5_path)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            for track_id, embedding in embeddings.items():
                f.create_dataset(
                    f"{key}/{track_id}", 
                    data=embedding, 
                    compression="gzip"
                )
        
        logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
        return filepath
    
    def load_embeddings_hdf5(
        self, 
        track_ids: List[str],
        key: str = "embeddings"
    ) -> Dict[str, np.ndarray]:
        """Load embeddings from HDF5 file.
        
        Args:
            track_ids: List of track IDs to load.
            key: Dataset key in HDF5 file.
            
        Returns:
            Dictionary mapping track_id to embedding array.
        """
        import h5py
        
        hdf5_path = self.config.get("embeddings.hdf5_path", "embeddings.h5")
        filepath = os.path.join(self.output_dir, hdf5_path)
        
        if not os.path.exists(filepath):
            logger.warning(f"HDF5 file not found: {filepath}")
            return {}
        
        embeddings = {}
        
        with h5py.File(filepath, 'r') as f:
            for track_id in track_ids:
                dataset_path = f"{key}/{track_id}"
                if dataset_path in f:
                    embeddings[track_id] = f[dataset_path][:]
        
        logger.info(f"Loaded {len(embeddings)} embeddings from HDF5")
        return embeddings
    
    def save_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray]
    ) -> str:
        """Save embeddings using configured storage method.
        
        Args:
            embeddings: Dictionary mapping track_id to embedding array.
            
        Returns:
            Path to saved file(s).
        """
        storage = self.config.get("embeddings.storage", "numpy")
        
        if storage == "hdf5":
            return self.save_embeddings_hdf5(embeddings)
        else:
            self.save_embeddings_numpy(embeddings)
            return self.embeddings_dir
    
    def load_embeddings(
        self, 
        track_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Load embeddings using configured storage method.
        
        Args:
            track_ids: List of track IDs to load.
            
        Returns:
            Dictionary mapping track_id to embedding array.
        """
        storage = self.config.get("embeddings.storage", "numpy")
        
        if storage == "hdf5":
            return self.load_embeddings_hdf5(track_ids)
        else:
            return self.load_embeddings_numpy(track_ids)
    
    def save_normalization_params(
        self, 
        mean: np.ndarray, 
        std: np.ndarray,
        path: Optional[str] = None
    ) -> str:
        """Save normalization parameters.
        
        Args:
            mean: Mean array.
            std: Standard deviation array.
            path: Optional custom path.
            
        Returns:
            Path to saved parameters.
        """
        if path is None:
            path = os.path.join(self.output_dir, "normalization_params.npz")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=mean, std=std)
        
        logger.info(f"Normalization params saved to: {path}")
        return path
    
    def load_normalization_params(
        self, 
        path: Optional[str] = None
    ) -> tuple:
        """Load normalization parameters.
        
        Args:
            path: Optional custom path.
            
        Returns:
            Tuple of (mean, std) arrays.
        """
        if path is None:
            path = os.path.join(self.output_dir, "normalization_params.npz")
        
        if not os.path.exists(path):
            logger.warning(f"Normalization params not found: {path}")
            return None, None
        
        data = np.load(path)
        return data["mean"], data["std"]
    
    def save_index(
        self, 
        index: Any,
        path: Optional[str] = None
    ) -> str:
        """Save similarity index.
        
        Args:
            index: Index object to save.
            path: Optional custom path.
            
        Returns:
            Path to saved index.
        """
        import pickle
        
        if path is None:
            path = os.path.join(self.output_dir, "similarity_index.pkl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(index, f)
        
        logger.info(f"Index saved to: {path}")
        return path
    
    def load_index(
        self, 
        path: Optional[str] = None
    ) -> Any:
        """Load similarity index.
        
        Args:
            path: Optional custom path.
            
        Returns:
            Loaded index object.
        """
        import pickle
        
        if path is None:
            path = os.path.join(self.output_dir, "similarity_index.pkl")
        
        if not os.path.exists(path):
            logger.warning(f"Index not found: {path}")
            return None
        
        with open(path, 'rb') as f:
            index = pickle.load(f)
        
        logger.info(f"Index loaded from: {path}")
        return index
    
    def exists(self) -> bool:
        """Check if manifest exists.
        
        Returns:
            True if manifest file exists.
        """
        return os.path.exists(self.manifest_path)
    
    def get_track_count(self) -> int:
        """Get number of tracks in manifest.
        
        Returns:
            Number of tracks or 0 if manifest doesn't exist.
        """
        if not self.exists():
            return 0
        
        df = self.load_manifest()
        return len(df)
