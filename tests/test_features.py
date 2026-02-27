"""
Tests for the feature extraction module.
"""

import os
import tempfile
import numpy as np
import pytest

from music_recommender.features import FeatureExtractor, normalize_features
from music_recommender.config import Config


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor."""
        config = Config()
        return FeatureExtractor(config)
    
    def test_get_feature_names(self, extractor):
        """Test getting feature names."""
        names = extractor.get_feature_names()
        
        # Should have MFCC features
        assert any("mfcc_mean" in n for n in names)
        
        # Should have spectral features
        assert any("spectral_centroid" in n for n in names)
        
        # Should have tempo
        assert "tempo" in names
    
    def test_get_feature_dim(self, extractor):
        """Test getting feature dimension."""
        dim = extractor.get_feature_dim()
        
        # Should be reasonable size
        assert dim > 100
    
    def test_normalize_features_standard(self):
        """Test standard normalization."""
        data = np.random.randn(10, 20)
        
        normalized, mean, std = normalize_features(data, method="standard")
        
        # Check normalization
        assert normalized.shape == data.shape
        assert mean.shape == (20,)
        assert std.shape == (20,)
        
        # Check mean close to 0
        assert np.abs(normalized.mean(axis=0)).max() < 0.1
    
    def test_normalize_features_minmax(self):
        """Test min-max normalization."""
        data = np.random.randn(10, 20) * 10 + 5
        
        normalized, min_val, range_val = normalize_features(data, method="minmax")
        
        # Check normalization
        assert normalized.shape == data.shape
        
        # Check range 0-1
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_normalize_features_none(self):
        """Test no normalization."""
        data = np.random.randn(10, 20)
        
        normalized, mean, std = normalize_features(data, method="none")
        
        # Should be unchanged
        np.testing.assert_array_equal(normalized, data)
    
    def test_features_to_array(self):
        """Test converting features to array."""
        from music_recommender.features import features_to_array
        
        features = {
            "mfcc_mean_0": 1.0,
            "mfcc_mean_1": 2.0,
            "mfcc_std_0": 0.5,
            "tempo": 120.0,
            "rms_mean": 0.1,
        }
        
        arr = features_to_array(features)
        
        # Should be float32
        assert arr.dtype == np.float32
        
        # Should have 5 values
        assert len(arr) == 5


class TestNormalizeFeatures:
    """Tests for normalization."""
    
    def test_zero_std(self):
        """Test handling zero standard deviation."""
        data = np.ones((10, 5))
        
        normalized, mean, std = normalize_features(data, method="standard")
        
        # Should not have NaN or inf
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
    
    def test_preserves_shape(self):
        """Test that normalization preserves shape."""
        data = np.random.randn(100, 50)
        
        normalized, _, _ = normalize_features(data, method="standard")
        
        assert normalized.shape == data.shape
