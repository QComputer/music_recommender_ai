"""
Music Recommender AI - A content-based music recommendation system.

This package provides tools for:
- Scanning local music libraries
- Extracting audio features and metadata
- Building similarity indexes
- Recommending similar tracks
"""

__version__ = "0.1.0"
__author__ = "Music Recommender AI Team"
__license__ = "MIT"

from music_recommender.config import Config
from music_recommender.scanner import Scanner
from music_recommender.metadata import MetadataExtractor
from music_recommender.features import FeatureExtractor
from music_recommender.embeddings import EmbeddingExtractor
from music_recommender.storage import Storage
from music_recommender.recommender import Recommender
from music_recommender.api import app

__all__ = [
    "Config",
    "Scanner",
    "MetadataExtractor", 
    "FeatureExtractor",
    "EmbeddingExtractor",
    "Storage",
    "Recommender",
    "app",
]
