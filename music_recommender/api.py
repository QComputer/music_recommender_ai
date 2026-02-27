"""
FastAPI application for music recommendation API.
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from music_recommender.config import Config, get_config
from music_recommender.recommender import Recommender
from music_recommender.storage import Storage


# Create FastAPI app
app = FastAPI(
    title="Music Recommender AI",
    description="Content-based music recommendation API",
    version="0.1.0",
)


# Global state
_recommender: Optional[Recommender] = None
_config: Optional[Config] = None


def get_recommender() -> Recommender:
    """Get or create recommender instance."""
    global _recommender, _config
    
    if _recommender is None:
        # Load config
        config = get_config()
        
        # Initialize recommender
        storage = Storage(config)
        _recommender = Recommender(config)
        
        if _recommender.is_loaded:
            return _recommender
        
        # Build index if not loaded
        use_embeddings = config.get("recommender.use_embeddings", False)
        _recommender.build_index(use_embeddings=use_embeddings)
    
    return _recommender


# Request/Response models
class TrackInfo(BaseModel):
    """Track information model."""
    track_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    duration: Optional[float] = None
    year: Optional[int] = None


class Recommendation(BaseModel):
    """Recommendation response model."""
    track_id: str
    track_info: TrackInfo
    distance: float
    similarity: float


class RecommendRequest(BaseModel):
    """Recommendation request model."""
    track_id: Optional[str] = None
    file_path: Optional[str] = None
    k: int = Query(default=10, ge=1, le=100)
    use_embeddings: bool = False


class RecommendResponse(BaseModel):
    """Recommendation response wrapper."""
    seed_track_id: str
    recommendations: List[Recommendation]


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global _config
    _config = get_config()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Music Recommender AI",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    recommender = get_recommender()
    return {
        "status": "healthy",
        "tracks_indexed": recommender.track_count if recommender else 0,
    }


@app.get("/tracks")
async def list_tracks(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List all tracks in the database."""
    recommender = get_recommender()
    
    if recommender._manifest is None:
        raise HTTPException(status_code=404, detail="No tracks found")
    
    df = recommender._manifest
    
    tracks = []
    for i in range(offset, min(offset + limit, len(df))):
        row = df.iloc[i]
        tracks.append({
            "track_id": row["track_id"],
            "title": row.get("title"),
            "artist": row.get("artist"),
            "album": row.get("album"),
            "genre": row.get("genre"),
            "duration": row.get("duration"),
        })
    
    return {
        "total": len(df),
        "limit": limit,
        "offset": offset,
        "tracks": tracks,
    }


@app.get("/tracks/{track_id}")
async def get_track(track_id: str):
    """Get track information."""
    recommender = get_recommender()
    
    if recommender._manifest is None:
        raise HTTPException(status_code=404, detail="No tracks found")
    
    df = recommender._manifest
    track = df[df["track_id"] == track_id]
    
    if track.empty:
        raise HTTPException(status_code=404, detail="Track not found")
    
    row = track.iloc[0]
    return {
        "track_id": row["track_id"],
        "file_path": row.get("file_path"),
        "title": row.get("title"),
        "artist": row.get("artist"),
        "album": row.get("album"),
        "genre": row.get("genre"),
        "year": row.get("year"),
        "duration": row.get("duration"),
        "bitrate": row.get("bitrate"),
        "sample_rate": row.get("sample_rate"),
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get track recommendations."""
    if not request.track_id and not request.file_path:
        raise HTTPException(
            status_code=400, 
            detail="Either track_id or file_path must be provided"
        )
    
    recommender = get_recommender()
    
    try:
        # Get recommendations
        recommendations = recommender.get_similar_tracks(
            file_path=request.file_path,
            track_id=request.track_id,
            k=request.k,
            use_embeddings=request.use_embeddings,
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail="No recommendations found"
            )
        
        # Determine seed track ID
        seed_track_id = request.track_id
        if not seed_track_id:
            from music_recommender.scanner import Scanner
            scanner = Scanner()
            seed_track_id = scanner.compute_track_id(request.file_path)
        
        # Format response
        recs = []
        for rec in recommendations:
            track_info = rec["track_info"]
            recs.append(Recommendation(
                track_id=rec["track_id"],
                track_info=TrackInfo(
                    track_id=rec["track_id"],
                    title=track_info.get("title"),
                    artist=track_info.get("artist"),
                    album=track_info.get("album"),
                    genre=track_info.get("genre"),
                    duration=track_info.get("duration"),
                    year=track_info.get("year"),
                ),
                distance=rec["distance"],
                similarity=rec["similarity"],
            ))
        
        return RecommendResponse(
            seed_track_id=seed_track_id,
            recommendations=recs,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
async def recommend_get(
    track_id: Optional[str] = Query(default=None),
    file_path: Optional[str] = Query(default=None),
    k: int = Query(default=10, ge=1, le=100),
    use_embeddings: bool = Query(default=False),
):
    """Get track recommendations (GET version)."""
    request = RecommendRequest(
        track_id=track_id,
        file_path=file_path,
        k=k,
        use_embeddings=use_embeddings,
    )
    return await recommend(request)


@app.get("/stats")
async def stats():
    """Get dataset statistics."""
    recommender = get_recommender()
    
    if recommender._manifest is None:
        return {
            "total_tracks": 0,
            "genres": [],
            "artists": [],
        }
    
    df = recommender._manifest
    
    stats = {
        "total_tracks": len(df),
    }
    
    # Genre distribution
    if "genre" in df.columns:
        genres = df["genre"].dropna()
        if not genres.empty:
            genre_counts = genres.value_counts().head(20).to_dict()
            stats["genres"] = genre_counts
    
    # Artist distribution
    if "artist" in df.columns:
        artists = df["artist"].dropna()
        if not artists.empty:
            artist_counts = artists.value_counts().head(20).to_dict()
            stats["artists"] = artist_counts
    
    # Album distribution
    if "album" in df.columns:
        albums = df["album"].dropna()
        if not albums.empty:
            album_counts = albums.value_counts().head(20).to_dict()
            stats["albums"] = album_counts
    
    # Features
    feature_cols = [c for c in df.columns if c.startswith(("mfcc_", "chroma_", "spectral_"))]
    stats["feature_count"] = len(feature_cols)
    
    # Embeddings
    if "has_embedding" in df.columns:
        stats["tracks_with_embeddings"] = int(df["has_embedding"].sum())
    
    return stats


@app.post("/reload")
async def reload_index():
    """Reload the similarity index."""
    global _recommender
    _recommender = None
    
    recommender = get_recommender()
    
    return {
        "status": "reloaded",
        "tracks_indexed": recommender.track_count if recommender else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
