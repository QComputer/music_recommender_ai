"""
Tests for the storage module.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from music_recommender.storage import Storage
from music_recommender.config import Config


class TestStorage:
    """Tests for Storage class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage instance."""
        config = Config()
        config.set("scanner.output_dir", temp_dir)
        return Storage(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return [
            {
                "track_id": "abc123",
                "file_path": "/music/track1.mp3",
                "artist": "Test Artist",
                "title": "Test Track",
            },
            {
                "track_id": "def456",
                "file_path": "/music/track2.mp3",
                "artist": "Another Artist",
                "title": "Another Track",
            },
        ]
    
    def test_save_manifest(self, storage, sample_data):
        """Test saving manifest."""
        path = storage.save_manifest(sample_data)
        
        assert os.path.exists(path)
        assert path.endswith(".parquet")
    
    def test_load_manifest(self, storage, sample_data):
        """Test loading manifest."""
        storage.save_manifest(sample_data)
        
        df = storage.load_manifest()
        
        assert len(df) == 2
        assert "track_id" in df.columns
        assert "artist" in df.columns
    
    def test_load_nonexistent_manifest(self, storage):
        """Test loading nonexistent manifest."""
        df = storage.load_manifest()
        
        assert df.empty
    
    def test_update_manifest(self, storage, sample_data):
        """Test updating manifest."""
        # Save initial
        storage.save_manifest(sample_data[:1])
        
        # Update with new
        updated = storage.update_manifest(sample_data[1:])
        
        # Should have both
        assert len(updated) == 2
    
    def test_save_embeddings_numpy(self, storage):
        """Test saving embeddings as numpy files."""
        embeddings = {
            "track1": np.random.randn(128),
            "track2": np.random.randn(128),
        }
        
        paths = storage.save_embeddings_numpy(embeddings)
        
        assert len(paths) == 2
        assert os.path.exists(storage.embeddings_dir)
    
    def test_load_embeddings_numpy(self, storage):
        """Test loading numpy embeddings."""
        embeddings = {
            "track1": np.random.randn(128),
            "track2": np.random.randn(128),
        }
        
        storage.save_embeddings_numpy(embeddings)
        
        loaded = storage.load_embeddings_numpy(["track1", "track2"])
        
        assert len(loaded) == 2
        np.testing.assert_array_equal(loaded["track1"], embeddings["track1"])
    
    def test_save_embeddings_hdf5(self, storage):
        """Test saving embeddings to HDF5."""
        storage.config.set("embeddings.storage", "hdf5")
        
        embeddings = {
            "track1": np.random.randn(128),
            "track2": np.random.randn(128),
        }
        
        path = storage.save_embeddings_hdf5(embeddings)
        
        assert os.path.exists(path)
    
    def test_save_normalization_params(self, storage):
        """Test saving normalization params."""
        mean = np.random.randn(100)
        std = np.ones(100)
        
        path = storage.save_normalization_params(mean, std)
        
        assert os.path.exists(path)
    
    def test_load_normalization_params(self, storage):
        """Test loading normalization params."""
        mean = np.random.randn(100)
        std = np.ones(100)
        
        storage.save_normalization_params(mean, std)
        
        loaded_mean, loaded_std = storage.load_normalization_params()
        
        np.testing.assert_array_equal(mean, loaded_mean)
        np.testing.assert_array_equal(std, loaded_std)
    
    def test_save_and_load_index(self, storage):
        """Test saving and loading index."""
        # Create mock index
        index = {"data": np.random.randn(10, 5)}
        
        path = storage.save_index(index)
        
        assert os.path.exists(path)
        
        loaded = storage.load_index(path)
        
        assert loaded is not None
        np.testing.assert_array_equal(loaded["data"], index["data"])
    
    def test_exists(self, storage, sample_data):
        """Test exists check."""
        assert not storage.exists()
        
        storage.save_manifest(sample_data)
        
        assert storage.exists()
    
    def test_get_track_count(self, storage, sample_data):
        """Test getting track count."""
        assert storage.get_track_count() == 0
    
    def test_save_and_get_track_count(self, storage, sample_data):
        """Test track count after save."""
        storage.save_manifest(sample_data)
        
        assert storage.get_track_count() == 2
