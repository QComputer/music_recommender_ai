#!/usr/bin/env python
"""
Demo script that runs the full pipeline: scan -> extract -> build-index -> recommend
"""

import os
import sys
import tempfile
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_audio_files(output_dir: str, num_files: int = 10):
    """Create sample audio files for demo.
    
    This creates simple test audio files. In production, you would
    use your actual music library.
    """
    import numpy as np
    import soundfile as sf
    
    logger.info(f"Creating {num_files} sample audio files...")
    
    sample_rate = 22050
    duration = 5  # 5 seconds
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        # Generate different "genres" with different frequency characteristics
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Different base frequencies for different "artists"
        freq = 220 + (i % 3) * 110  # A3, A4, or A5
        
        # Generate audio
        audio = np.sin(2 * np.pi * freq * t) * 0.3
        
        # Add some variation
        audio += np.random.randn(len(audio)) * 0.05
        
        # Save
        filename = os.path.join(output_dir, f"demo_track_{i:02d}.wav")
        sf.write(filename, audio, sample_rate)
    
    logger.info(f"Created {num_files} sample files in {output_dir}")
    return output_dir


def run_demo():
    """Run the full demo pipeline."""
    # Create temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        music_dir = os.path.join(tmpdir, "music")
        data_dir = os.path.join(tmpdir, "data")
        
        # Create sample audio files
        create_sample_audio_files(music_dir, num_files=10)
        
        # Import after creating temp dir
        from music_recommender.config import Config
        from music_recommender.pipeline import Pipeline
        
        # Create config
        config = Config()
        config.set("scanner.music_root", music_dir)
        config.set("scanner.output_dir", data_dir)
        config.set("recommender.backend", "sklearn")
        
        # Create pipeline
        pipeline = Pipeline(config)
        
        # Step 1: Scan
        logger.info("=" * 50)
        logger.info("STEP 1: Scanning music directory")
        logger.info("=" * 50)
        files = pipeline.run_scan()
        logger.info(f"Found {len(files)} audio files")
        
        # Step 2: Extract
        logger.info("=" * 50)
        logger.info("STEP 2: Extracting features and metadata")
        logger.info("=" * 50)
        df = pipeline.run_extract()
        logger.info(f"Extracted features for {len(df)} tracks")
        
        # Show feature columns
        feature_cols = [c for c in df.columns if c.startswith(("mfcc_", "chroma_", "spectral_"))]
        logger.info(f"Generated {len(feature_cols)} feature columns")
        
        # Step 3: Build index
        logger.info("=" * 50)
        logger.info("STEP 3: Building similarity index")
        logger.info("=" * 50)
        recommender = pipeline.run_build_index()
        logger.info(f"Built index with {recommender.track_count} tracks")
        
        # Step 4: Get recommendations
        logger.info("=" * 50)
        logger.info("STEP 4: Getting recommendations")
        logger.info("=" * 50)
        
        # Get first track
        first_track_id = df.iloc[0]["track_id"]
        first_title = df.iloc[0].get("title", "Unknown")
        
        logger.info(f"Getting recommendations for: {first_title}")
        
        recommendations = pipeline.get_recommendations(
            track_id=first_track_id,
            k=3
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)
        print(f"\nSeed track: {first_title}")
        print(f"Track ID: {first_track_id}\n")
        
        for i, rec in enumerate(recommendations, 1):
            track_info = rec["track_info"]
            title = track_info.get("title", "Unknown")
            artist = track_info.get("artist", "Unknown Artist")
            similarity = rec["similarity"]
            
            print(f"{i}. {title} by {artist}")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Track ID: {rec['track_id']}")
            print()
        
        # Show statistics
        logger.info("=" * 50)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 50)
        stats = pipeline.get_statistics()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nDemo complete!")
        
        # Keep temp directory for inspection if needed
        logger.info(f"Data saved to: {data_dir}")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)
