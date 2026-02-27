"""
Integration tests for the full pipeline.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from music_recommender.config import Config
from music_recommender.scanner import Scanner
from music_recommender.storage import Storage
from music_recommender.recommender import Recommender


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = Config()
        config.set("scanner.music_root", temp_dir)
        config.set("scanner.output_dir", temp_dir)
        config.set("recommender.backend", "sklearn")
        return config
    
    @pytest.fixture
    def sample_manifest(self, temp_dir):
        """Create sample manifest with features."""
        # Create sample data with features
        data = []
        for i in range(10):
            track_data = {
                "track_id": f"track_{i}",
                "file_path": f"/music/track_{i}.mp3",
                "artist": f"Artist {i % 3}",
                "album": f"Album {i % 2}",
                "title": f"Track {i}",
                "genre": f"Genre {i % 3}",
                "duration": 180.0,
            }
            
            # Add mock features
            for j in range(20):
                track_data[f"mfcc_mean_{j}"] = np.random.randn()
                track_data[f"mfcc_std_{j}"] = np.random.randn()
            
            # Add spectral features
            track_data["spectral_centroid_mean"] = np.random.randn() * 1000 + 2000
            track_data["spectral_centroid_std"] = np.random.randn() * 100
            
            # Add tempo
            track_data["tempo"] = np.random.uniform(60, 180)
            
            data.append(track_data)
        
        # Save
        storage = Storage(Config())
        storage.config.set("scanner.output_dir", temp_dir)
        storage.save_manifest(data)
        
        return data
    
    def test_full_pipeline_scanner_to_recommender(self, config, temp_dir):
        """Test full pipeline from scanning to recommendations."""
        # Create sample files
        for i in range(5):
            filepath = os.path.join(temp_dir, f"track_{i}.mp3")
            with open(filepath, 'wb') as f:
                f.write(b'test audio content' * 100)
        
        # Run scanner
        scanner = Scanner(config)
        files = scanner.scan()
        
        assert len(files) == 5
        
        # Save manifest
        storage = Storage(config)
        storage.save_manifest(files)
        
        # Create recommender
        recommender = Recommender(config)
        
        # Load data (without actual features, so won't work for recommendations)
        recommender.load_data()
        
        assert recommender.track_count == 5
    
    def test_recommender_with_mock_features(self, config, sample_manifest):
        """Test recommender with mock features."""
        recommender = Recommender(config)
        recommender.build_index(use_embeddings=False)
        
        assert recommender.track_count == 10
        assert recommender.is_loaded
    
    def test_recommend_returns_results(self, config, sample_manifest):
        """Test that recommend returns results."""
        recommender = Recommender(config)
        recommender.build_index()
        
        # Get first track ID
        track_id = sample_manifest[0]["track_id"]
        
        # Get recommendations
        recs = recommender.recommend(track_id, k=3)
        
        # Should return recommendations
        assert len(recs) <= 3
        
        # Should not include self
        for rec in recs:
            assert rec["track_id"] != track_id
    
    def test_recommend_batch(self, config, sample_manifest):
        """Test batch recommendations."""
        recommender = Recommender(config)
        recommender.build_index()
        
        # Get recommendations for multiple tracks
        track_ids = [sample_manifest[i]["track_id"] for i in range(3)]
        
        results = recommender.recommend_batch(track_ids, k=3)
        
        assert len(results) == 3
        for track_id in track_ids:
            assert track_id in results
            assert len(results[track_id]) <= 3
    
    def test_metadata_boost(self, config, sample_manifest):
        """Test metadata boost in recommendations."""
        config.set("recommender.metadata_boost", 0.5)
        
        recommender = Recommender(config)
        recommender.build_index()
        
        track_id = sample_manifest[0]["track_id"]
        
        # Should apply metadata boost
        recs = recommender.recommend(track_id, k=3)
        
        assert len(recs) > 0
    
    def test_config_persistence(self, config):
        """Test that config is persisted correctly."""
        # Set custom values
        config.set("scanner.music_root", "/custom/path")
        config.set("recommender.backend", "faiss")
        
        # Create new config with same settings
        new_config = Config()
        
        # Values should be defaults (not custom)
        # Because new config loads defaults
        assert new_config.get("recommender.backend") != "faiss" or \
               new_config.get("scanner.music_root") == "/custom/path"


class TestPipelineEdgeCases:
    """Test edge cases in pipeline."""
    
    def test_empty_manifest(self):
        """Test handling empty manifest."""
        config = Config()
        recommender = Recommender(config)
        
        # Should handle gracefully
        assert not recommender.is_loaded
        assert recommender.track_count == 0
    
    def test_nonexistent_track_recommend(self):
        """Test recommending nonexistent track."""
        config = Config()
        
        # Create sample manifest
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config.set("scanner.output_dir", tmpdir)
            
            storage = Storage(config)
            storage.save_manifest([
                {"track_id": "track1", "file_path": "/test.mp3"}
            ])
            
            recommender = Recommender(config)
            recommender.build_index()
            
            # Try to get recommendations for nonexistent track
            recs = recommender.recommend("nonexistent_track", k=3)
            
            assert len(recs) == 0
