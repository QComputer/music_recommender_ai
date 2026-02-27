"""
Tests for the scanner module.
"""

import os
import tempfile
import pytest

from music_recommender.scanner import Scanner
from music_recommender.config import Config


class TestScanner:
    """Tests for Scanner class."""
    
    @pytest.fixture
    def temp_music_dir(self):
        """Create temporary music directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            subdir = os.path.join(tmpdir, "music")
            os.makedirs(subdir)
            
            # Create test files
            test_files = [
                "test1.mp3",
                "test2.flac",
                "test3.wav",
                "test4.m4a",
                "test5.txt",  # Should be ignored
            ]
            
            for filename in test_files:
                filepath = os.path.join(subdir, filename)
                with open(filepath, 'wb') as f:
                    f.write(b'test content')
            
            yield tmpdir
    
    def test_find_audio_files(self, temp_music_dir):
        """Test finding audio files."""
        config = Config()
        config.set("scanner.music_root", temp_music_dir)
        
        scanner = Scanner(config)
        files = list(scanner.find_audio_files())
        
        # Should find 4 audio files, not the .txt
        assert len(files) == 4
        
        extensions = [f.suffix.lower().lstrip('.') for f in files]
        assert 'mp3' in extensions
        assert 'flac' in extensions
        assert 'wav' in extensions
        assert 'm4a' in extensions
        assert 'txt' not in extensions
    
    def test_compute_track_id(self):
        """Test track ID computation."""
        config = Config()
        scanner = Scanner(config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'test content')
            filepath = f.name
        
        try:
            track_id = scanner.compute_track_id(filepath)
            
            # Should be 64 character hex string
            assert len(track_id) == 64
            assert all(c in '0123456789abcdef' for c in track_id)
            
            # Same file should produce same ID
            track_id2 = scanner.compute_track_id(filepath)
            assert track_id == track_id2
            
        finally:
            os.unlink(filepath)
    
    def test_scan(self, temp_music_dir):
        """Test scanning directory."""
        config = Config()
        config.set("scanner.music_root", temp_music_dir)
        
        scanner = Scanner(config)
        files = scanner.scan()
        
        assert len(files) == 4
        
        # Check structure
        for f in files:
            assert 'track_id' in f
            assert 'file_path' in f
            assert 'file_size' in f
            assert 'mtime' in f
    
    def test_scan_nonexistent_dir(self):
        """Test scanning nonexistent directory."""
        config = Config()
        config.set("scanner.music_root", "/nonexistent/path")
        
        scanner = Scanner(config)
        files = scanner.scan()
        
        assert len(files) == 0
    
    def test_get_file_count(self, temp_music_dir):
        """Test file count."""
        config = Config()
        config.set("scanner.music_root", temp_music_dir)
        
        scanner = Scanner(config)
        count = scanner.get_file_count()
        
        assert count == 4
