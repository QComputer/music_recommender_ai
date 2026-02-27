"""
Metadata extraction module for audio files.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from music_recommender.config import Config


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extractor for audio file metadata using mutagen or tinytag."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize metadata extractor.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.normalize_strings = self.config.get("metadata.normalize_strings", True)
        
        # Try to import mutagen first, fall back to tinytag
        self._extractor = None
        self._backend = None
        self._init_extractor()
    
    def _init_extractor(self) -> None:
        """Initialize the metadata extractor backend."""
        # Try mutagen
        try:
            import mutagen
            from mutagen import File as MutagenFile
            self._mutagen_file = MutagenFile
            self._backend = "mutagen"
            logger.info("Using mutagen for metadata extraction")
            return
        except ImportError:
            pass
        
        # Try tinytag
        try:
            from tinytag import TinyTag
            self._tinytag = TinyTag
            self._backend = "tinytag"
            logger.info("Using tinytag for metadata extraction")
            return
        except ImportError:
            pass
        
        # Neither available
        logger.warning("Neither mutagen nor tinytag available. Metadata extraction will be limited.")
        self._backend = None
    
    def normalize_string(self, s: Any) -> Optional[str]:
        """Normalize string for consistency.
        
        Args:
            s: String to normalize.
            
        Returns:
            Normalized string or None.
        """
        if s is None:
            return None
        
        # Convert to string
        s = str(s)
        
        if self.normalize_strings:
            # Normalize unicode
            s = unicodedata.normalize('NFKD', s)
            # Strip whitespace
            s = s.strip()
            # Remove extra whitespace
            s = re.sub(r'\s+', ' ', s)
        
        return s if s else None
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Dictionary of metadata fields.
        """
        metadata = {
            "artist": None,
            "album": None,
            "title": None,
            "genre": None,
            "year": None,
            "track_number": None,
            "duration": None,
            "bitrate": None,
            "sample_rate": None,
        }
        
        if self._backend is None:
            logger.warning(f"No metadata backend available for: {file_path}")
            return metadata
        
        try:
            if self._backend == "mutagen":
                metadata = self._extract_mutagen(file_path)
            elif self._backend == "tinytag":
                metadata = self._extract_tinytag(file_path)
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
        
        # Normalize string fields
        for key in ["artist", "album", "title", "genre"]:
            if metadata.get(key):
                metadata[key] = self.normalize_string(metadata[key])
        
        return metadata
    
    def _extract_mutagen(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using mutagen.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Dictionary of metadata fields.
        """
        metadata = {}
        
        try:
            audio = self._mutagen_file(file_path)
            
            if audio is None:
                return metadata
            
            # Basic info
            metadata["duration"] = audio.info.length if hasattr(audio.info, 'length') else None
            metadata["bitrate"] = audio.info.bitrate if hasattr(audio.info, 'bitrate') else None
            metadata["sample_rate"] = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else None
            
            # Tags
            if audio.tags:
                # Common tag keys
                tag_mappings = {
                    'artist': ['artist', 'TPE1', '\xa9ART'],
                    'album': ['album', 'TALB', '\xa9alb'],
                    'title': ['title', 'TIT2', '\xa9nam'],
                    'genre': ['genre', 'TCON', '\xa9gen'],
                    'year': ['date', 'TDRC', '\xa9day'],
                    'track_number': ['tracknumber', 'TRCK', 'trkn'],
                }
                
                for key, tags in tag_mappings.items():
                    for tag in tags:
                        if tag in audio.tags:
                            value = audio.tags[tag]
                            # Handle track number specially
                            if key == "track_number":
                                if hasattr(value, 'text'):
                                    value = value.text[0] if value.text else None
                                elif isinstance(value, list):
                                    value = str(value[0]) if value else None
                                else:
                                    value = str(value)
                                # Extract number from string like "1/12"
                                if value and '/' in value:
                                    value = value.split('/')[0]
                            else:
                                if hasattr(value, 'text'):
                                    value = value.text[0] if value.text else None
                                elif isinstance(value, list):
                                    value = str(value[0]) if value else None
                                else:
                                    value = str(value)
                            
                            if value:
                                metadata[key] = value
                                break
                
                # Handle year as integer
                if metadata.get("year"):
                    try:
                        year_str = str(metadata["year"])
                        # Extract first 4 digits
                        match = re.search(r'\d{4}', year_str)
                        if match:
                            metadata["year"] = int(match.group())
                        else:
                            metadata["year"] = None
                    except (ValueError, TypeError):
                        metadata["year"] = None
        
        except Exception as e:
            logger.debug(f"Mutagen extraction error for {file_path}: {e}")
        
        return metadata
    
    def _extract_tinytag(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using tinytag.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Dictionary of metadata fields.
        """
        metadata = {}
        
        try:
            audio = self._tinytag.get(file_path)
            
            if audio is None:
                return metadata
            
            # Basic info
            metadata["duration"] = audio.duration
            metadata["bitrate"] = audio.bitrate
            metadata["sample_rate"] = audio.sample_rate
            
            # Tags
            metadata["artist"] = audio.artist
            metadata["album"] = audio.album
            metadata["title"] = audio.title
            metadata["genre"] = audio.genre
            
            # Year
            if audio.year:
                try:
                    metadata["year"] = int(audio.year)
                except (ValueError, TypeError):
                    metadata["year"] = None
            
            # Track number
            if audio.track:
                try:
                    metadata["track_number"] = int(audio.track)
                except (ValueError, TypeError):
                    metadata["track_number"] = None
        
        except Exception as e:
            logger.debug(f"TinyTag extraction error for {file_path}: {e}")
        
        return metadata
    
    def extract_batch(
        self, 
        file_paths: list,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Extract metadata from multiple files.
        
        Args:
            file_paths: List of file paths.
            show_progress: Whether to show progress.
            
        Returns:
            Dictionary mapping file paths to metadata dictionaries.
        """
        results = {}
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(file_paths, desc="Extracting metadata")
            except ImportError:
                iterator = file_paths
        else:
            iterator = file_paths
        
        for file_path in iterator:
            results[file_path] = self.extract(file_path)
        
        return results


def get_duration_formatted(seconds: Optional[float]) -> Optional[str]:
    """Format duration in seconds to MM:SS format.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted string or None.
    """
    if seconds is None:
        return None
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
