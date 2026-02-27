"""
CLI module for music recommender commands.
"""

import logging
import os
import sys
from typing import Optional

import click

from music_recommender.config import Config, get_config, set_config
from music_recommender.scanner import Scanner
from music_recommender.metadata import MetadataExtractor
from music_recommender.features import FeatureExtractor
from music_recommender.embeddings import create_embedding_extractor
from music_recommender.storage import Storage
from music_recommender.recommender import Recommender


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level"
)
@click.pass_context
def cli(ctx, config: str, log_level: str):
    """Music Recommender AI - Content-based music recommendation system."""
    # Setup logging
    setup_logging(log_level)
    
    # Load configuration
    cfg = Config(config) if config else Config()
    set_config(cfg)
    
    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@cli.command()
@click.option(
    "--root",
    "-r",
    type=click.Path(exists=True),
    help="Music root directory"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for manifest"
)
@click.option(
    "--checksum/--no-checksum",
    default=False,
    help="Compute file checksums"
)
@click.pass_context
def scan(ctx, root: str, output_dir: str, checksum: bool):
    """Scan music directory and create inventory manifest."""
    config = ctx.obj["config"]
    
    if root:
        config.set("scanner.music_root", root)
    if output_dir:
        config.set("scanner.output_dir", output_dir)
    
    scanner = Scanner(config)
    
    click.echo(f"Scanning: {config.music_root}")
    files = scanner.scan(compute_checksum=checksum)
    
    # Save to manifest
    storage = Storage(config)
    storage.save_manifest(files)
    
    click.echo(f"Found {len(files)} audio files")
    click.echo(f"Manifest saved to: {storage.manifest_path}")


@cli.command()
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True),
    help="Path to manifest file (from scan)"
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="Use parallel extraction"
)
@click.option(
    "--workers",
    "-w",
    type=int,
    help="Number of worker processes"
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip files already processed"
)
@click.pass_context
def extract(ctx, manifest: str, parallel: bool, workers: int, skip_existing: bool):
    """Extract metadata and features from audio files."""
    config = ctx.obj["config"]
    
    if workers:
        config.set("features.n_workers", workers)
    
    # Initialize components
    scanner = Scanner(config)
    metadata_extractor = MetadataExtractor(config)
    feature_extractor = FeatureExtractor(config)
    embedding_extractor = create_embedding_extractor(config)
    storage = Storage(config)
    
    # Load manifest
    if manifest:
        df = storage.load_manifest(manifest)
    else:
        df = storage.load_manifest()
    
    if df.empty:
        click.echo("No manifest found. Run 'scan' first.", err=True)
        sys.exit(1)
    
    click.echo(f"Processing {len(df)} tracks...")
    
    # Process tracks
    from tqdm import tqdm
    
    track_ids = df["track_id"].tolist()
    file_paths = df["file_path"].tolist()
    
    results = []
    for i, (track_id, file_path) in enumerate(tqdm(zip(track_ids, file_paths), total=len(track_ids))):
        # Get existing data if skipping
        if skip_existing:
            existing = df[df["track_id"] == track_id]
            if not existing.empty:
                # Check if features already exist
                has_features = any(c.startswith("mfcc_") for c in existing.columns)
                if has_features:
                    results.append(existing.iloc[0].to_dict())
                    continue
        
        try:
            # Extract metadata
            metadata = metadata_extractor.extract(file_path)
            
            # Extract features
            features = feature_extractor.extract_features_flat(file_path)
            
            # Extract embeddings if enabled
            embedding = None
            if embedding_extractor.is_available:
                embedding = embedding_extractor.extract_embedding(file_path)
            
            # Combine
            track_data = {
                "track_id": track_id,
                "file_path": file_path,
            }
            track_data.update(metadata)
            
            # Add features
            for key, value in features.items():
                if key not in ["success", "error"]:
                    track_data[key] = value
            
            # Add embedding path reference
            if embedding is not None:
                track_data["has_embedding"] = True
                track_data["embedding_dim"] = len(embedding)
            else:
                track_data["has_embedding"] = False
            
            results.append(track_data)
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            results.append({
                "track_id": track_id,
                "file_path": file_path,
                "error": str(e),
            })
    
    # Save updated manifest
    storage.save_manifest(results)
    
    click.echo(f"Extraction complete. Updated manifest saved to: {storage.manifest_path}")


@cli.command()
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True),
    help="Path to manifest file"
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["sklearn", "faiss", "hnswlib"]),
    help="Similarity search backend"
)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use embeddings instead of features"
)
@click.pass_context
def build_index(ctx, manifest: str, backend: str, embeddings: bool):
    """Build similarity search index."""
    config = ctx.obj["config"]
    
    if backend:
        config.set("recommender.backend", backend)
    
    storage = Storage(config)
    
    click.echo("Building similarity index...")
    click.echo(f"  Backend: {config.backend}")
    click.echo(f"  Using embeddings: {embeddings}")
    
    recommender = Recommender(config)
    recommender.build_index(manifest, use_embeddings=embeddings)
    
    click.echo(f"Index built successfully with {recommender.track_count} tracks")


@cli.command()
@click.option(
    "--track-id",
    "-t",
    help="Track ID to find recommendations for"
)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True),
    help="Audio file path (computes track ID)"
)
@click.option(
    "--k",
    "-k",
    type=int,
    default=10,
    help="Number of recommendations"
)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use embeddings for similarity"
)
@click.pass_context
def recommend(ctx, track_id: str, file: str, k: int, embeddings: bool):
    """Get track recommendations."""
    config = ctx.obj["config"]
    
    if not track_id and not file:
        click.echo("Either --track-id or --file must be provided", err=True)
        sys.exit(1)
    
    # Initialize recommender
    storage = Storage(config)
    recommender = Recommender(config)
    
    if not recommender.is_loaded:
        click.echo("Building index...")
        recommender.build_index(use_embeddings=embeddings)
    
    # Get recommendations
    recommendations = recommender.get_similar_tracks(
        file_path=file,
        track_id=track_id,
        k=k,
        use_embeddings=embeddings
    )
    
    if not recommendations:
        click.echo("No recommendations found")
        return
    
    # Display results
    click.echo(f"\nTop {k} recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        track_info = rec["track_info"]
        title = track_info.get("title", "Unknown")
        artist = track_info.get("artist", "Unknown Artist")
        album = track_info.get("album", "Unknown Album")
        
        click.echo(f"{i}. {title} - {artist}")
        click.echo(f"   Album: {album}")
        click.echo(f"   Similarity: {rec['similarity']:.3f}")
        click.echo(f"   Track ID: {rec['track_id']}")
        click.echo()


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to"
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to bind to"
)
@click.option(
    "--reload/--no-reload",
    default=False,
    help="Enable auto-reload"
)
@click.pass_context
def serve(ctx, host: str, port: int, reload: bool):
    """Start the API server."""
    import uvicorn
    
    config = ctx.obj["config"]
    config.set("api.host", host)
    config.set("api.port", port)
    config.set("api.reload", reload)
    
    click.echo(f"Starting API server at http://{host}:{port}")
    
    # Import and run app
    from music_recommender.api import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True),
    help="Path to manifest file"
)
@click.pass_context
def stats(ctx, manifest: str):
    """Show dataset statistics."""
    config = ctx.obj["config"]
    storage = Storage(config)
    
    df = storage.load_manifest(manifest)
    
    if df.empty:
        click.echo("Manifest is empty")
        return
    
    click.echo(f"\nDataset Statistics")
    click.echo(f"==================\n")
    click.echo(f"Total tracks: {len(df)}")
    
    # Genre distribution
    if "genre" in df.columns:
        genres = df["genre"].dropna()
        if not genres.empty:
            click.echo(f"\nTop genres:")
            for genre, count in genres.value_counts().head(10).items():
                click.echo(f"  {genre}: {count}")
    
    # Artist distribution
    if "artist" in df.columns:
        artists = df["artist"].dropna()
        if not artists.empty:
            click.echo(f"\nTop artists:")
            for artist, count in artists.value_counts().head(10).items():
                click.echo(f"  {artist}: {count}")
    
    # Feature columns
    feature_cols = [c for c in df.columns if c.startswith(("mfcc_", "chroma_", "spectral_", "rms_", "zcr_", "tempo"))]
    if feature_cols:
        click.echo(f"\nFeature columns: {len(feature_cols)}")
    
    # Embeddings
    if "has_embedding" in df.columns:
        n_with_embeddings = df["has_embedding"].sum()
        click.echo(f"Tracks with embeddings: {n_with_embeddings}")
    
    click.echo()


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
