"""
Scanner module for finding audio files and computing stable track IDs.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from music_recommender.config import Config


logger = logging.getLogger(__name__)


class Scanner:
    """Scanner for finding and inventorying audio files."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize scanner.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.supported_formats = self.config.get(
            "scanner.supported_formats", ["mp3", "flac", "wav", "m4a"]
        )
        self.music_root = self.config.music_root
        self.output_dir = self.config.output_dir
    
    def find_audio_files(self, root_dir: Optional[str] = None) -> Iterator[Path]:
        """Recursively find all audio files in directory.
        
        Args:
            root_dir: Root directory to search. Uses config music_root if None.
            
        Yields:
            Path objects for each audio file found.
        """
        root = root_dir or self.music_root
        
        if not os.path.exists(root):
            logger.warning(f"Music directory does not exist: {root}")
            return
        
        logger.info(f"Scanning for audio files in: {root}")
        
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext in self.supported_formats:
                    yield Path(dirpath) / filename
    
    def compute_track_id(
        self, 
        file_path: str, 
        file_size: Optional[int] = None,
        mtime: Optional[float] = None
    ) -> str:
        """Compute stable track ID from file metadata.
        
        Uses SHA256 of path + size + modification time for stability
        across file moves (as long as path remains the same).
        
        Args:
            file_path: Absolute path to the file.
            file_size: File size in bytes. If None, computed from file.
            mtime: Modification time. If None, computed from file.
            
        Returns:
            64-character hex string track ID.
        """
        if file_size is None or mtime is None:
            stat = os.stat(file_path)
            if file_size is None:
                file_size = stat.st_size
            if mtime is None:
                mtime = stat.st_mtime
        
        # Create stable identifier from path, size, and mtime
        # Normalize path for consistency
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        
        id_string = f"{normalized_path}|{file_size}|{mtime}"
        track_id = hashlib.sha256(id_string.encode('utf-8')).hexdigest()
        
        return track_id
    
    def compute_file_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of entire file.
        
        Args:
            file_path: Path to file.
            
        Returns:
            64-character hex string checksum.
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def scan(
        self, 
        root_dir: Optional[str] = None,
        compute_checksum: bool = False
    ) -> List[Dict]:
        """Scan directory and build file inventory.
        
        Args:
            root_dir: Root directory to scan. Uses config if None.
            compute_checksum: Whether to compute file checksums.
            
        Returns:
            List of dictionaries with file information.
        """
        root = root_dir or self.music_root
        files = []
        
        logger.info(f"Starting scan of: {root}")
        
        for audio_path in self.find_audio_files(root_dir=root):
            try:
                stat = os.stat(audio_path)
                
                file_info = {
                    "file_path": str(audio_path),
                    "file_name": audio_path.name,
                    "file_ext": audio_path.suffix.lower().lstrip('.'),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                    "track_id": self.compute_track_id(
                        str(audio_path), 
                        file_size=stat.st_size, 
                        mtime=stat.st_mtime
                    ),
                }
                
                if compute_checksum:
                    file_info["checksum"] = self.compute_file_checksum(str(audio_path))
                
                files.append(file_info)
                
            except Exception as e:
                logger.error(f"Error scanning {audio_path}: {e}")
        
        logger.info(f"Scan complete. Found {len(files)} audio files.")
        
        return files
    
    def scan_with_checkpoint(
        self,
        root_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None
    ) -> List[Dict]:
        """Scan with checkpointing to skip unchanged files.
        
        Args:
            root_dir: Root directory to scan.
            checkpoint_file: Path to checkpoint file.
            
        Returns:
            List of file information dictionaries.
        """
        import json
        
        root = root_dir or self.music_root
        
        if checkpoint_file is None:
            checkpoint_file = os.path.join(
                self.output_dir, 
                self.config.get("features.checkpoint_file", "extraction_checkpoint.json")
            )
        
        # Load existing checkpoint
        checkpoint = {}
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Scan files
        all_files = self.scan(root_dir=root, compute_checksum=False)
        
        # Filter to only changed files
        changed_files = []
        for file_info in all_files:
            track_id = file_info["track_id"]
            existing = checkpoint.get(track_id)
            
            if existing is None:
                # New file
                changed_files.append(file_info)
            elif (existing.get("mtime") != file_info["mtime"] or
                  existing.get("file_size") != file_info["file_size"]):
                # Modified file
                changed_files.append(file_info)
        
        logger.info(
            f"Checkpoint scan complete. {len(changed_files)} changed files "
            f"out of {len(all_files)} total."
        )
        
        return changed_files
    
    def save_checkpoint(
        self, 
        files: List[Dict], 
        checkpoint_file: Optional[str] = None
    ) -> None:
        """Save scan checkpoint.
        
        Args:
            files: List of file information dictionaries.
            checkpoint_file: Path to checkpoint file.
        """
        import json
        
        if checkpoint_file is None:
            checkpoint_file = os.path.join(
                self.output_dir,
                self.config.get("features.checkpoint_file", "extraction_checkpoint.json")
            )
        
        checkpoint = {}
        for f in files:
            checkpoint[f["track_id"]] = {
                "file_path": f["file_path"],
                "mtime": f["mtime"],
                "file_size": f["file_size"],
            }
        
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved to: {checkpoint_file}")
    
    def get_file_count(self, root_dir: Optional[str] = None) -> int:
        """Get count of audio files in directory.
        
        Args:
            root_dir: Root directory to count. Uses config if None.
            
        Returns:
            Number of audio files.
        """
        return sum(1 for _ in self.find_audio_files(root_dir))
