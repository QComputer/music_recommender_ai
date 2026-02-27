"""
Tests for the metadata extraction module.
"""

import os
import tempfile
import pytest

from music_recommender.metadata import MetadataExtractor
from music_recommender.config import Config


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create metadata extractor."""
        return MetadataExtractor()
    
    def test_normalize_string(self, extractor):
        """Test string normalization."""
        # Test basic normalization
        result = extractor.normalize_string("  Test  String  ")
        assert result == "Test String"
        
        # Test unicode
        result = extractor.normalize_string("café")
        assert "café" in result
        
        # Test None
        result = extractor.normalize_string(None)
        assert result is None
    
    def test_extract_empty_file(self, extractor):
        """Test extracting from nonexistent file."""
        metadata = extractor.extract("/nonexistent/file.mp3")
        
        # Should return default values
        assert metadata["artist"] is None
        assert metadata["title"] is None
        assert metadata["album"] is None
    
    def test_extract_without_backend(self):
        """Test extractor without backend available."""
        # This will work if neither mutagen nor tinytag is available
        extractor = MetadataExtractor()
        
        # Should not raise error
        metadata = extractor.extract("/nonexistent/file.mp3")
        assert isinstance(metadata, dict)
    
    def test_extract_batch(self, extractor):
        """Test batch extraction."""
        files = ["/nonexistent/file1.mp3", "/nonexistent/file2.flac"]
        
        results = extractor.extract_batch(files, show_progress=False)
        
        assert len(results) == 2
        assert "/nonexistent/file1.mp3" in results
        assert "/nonexistent/file2.flac" in results


class TestDurationFormatting:
    """Tests for duration formatting."""
    
    def test_get_duration_formatted(self):
        """Test duration formatting."""
        from music_recommender.metadata import get_duration_formatted
        
        assert get_duration_formatted(60) == "1:00"
        assert get_duration_formatted(90) == "1:30"
        assert get_duration_formatted(0) == "0:00"
        assert get_duration_formatted(None) is None
        assert get_duration_formatted(125.5) == "2:05"
