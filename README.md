# Music Recommender AI

A content-based music recommendation system that scans local music libraries, extracts audio features and metadata, builds similarity indexes, and provides recommendations through an API.

## Features

- **Audio File Scanning**: Recursively finds MP3, FLAC, WAV, and M4A files
- **Stable Track IDs**: SHA256-based track IDs from path, size, and modification time
- **Metadata Extraction**: Extracts artist, album, title, genre, year, duration, and more using mutagen
- **Feature Extraction**: Computes MFCCs, delta MFCCs, chroma, spectral features, tempo, and more using librosa
- **Embedding Support**: Optional VGGish or YAMNet embeddings (requires TensorFlow)
- **Multiple Similarity Backends**: scikit-learn, FAISS, or HNSWlib for fast similarity search
- **REST API**: FastAPI-based API for recommendations
- **CLI**: Command-line interface for all operations

## Installation

```bash
# Clone the repository
git clone https://github.com/qcomputer/music_recommender_ai.git
cd music_recommender_ai

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### System Dependencies

On Ubuntu/Debian:
```bash
sudo apt-get install ffmpeg libsndfile1
```

On macOS:
```bash
brew install ffmpeg libsndfile
```

## Quick Start

### 1. Scan your music library

```bash
python -m music_recommender.cli scan --root /path/to/your/music
```

### 2. Extract features and metadata

```bash
python -m music_recommender.cli extract
```

### 3. Build the similarity index

```bash
python -m music_recommender.cli build-index
```

### 4. Get recommendations

```bash
# By track ID
python -m music_recommender.cli recommend --track-id YOUR_TRACK_ID

# By file path
python -m music_recommender.cli recommend --file /path/to/track.mp3
```

### 5. Start the API server

```bash
python -m music_recommender.cli serve
```

The API will be available at http://localhost:8000

## Configuration

All settings can be customized in `config/config.yaml`:

```yaml
scanner:
  music_root: "./music"
  output_dir: "./data"
  supported_formats: ["mp3", "flac", "wav", "m4a"]

audio:
  sample_rate: 22050
  mono: true

features:
  n_mfcc: 20
  n_workers: 4

recommender:
  backend: "sklearn"  # sklearn, faiss, or hnswlib
  default_k: 10

api:
  host: "0.0.0.0"
  port: 8000
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API information |
| `GET /health` | Health check |
| `GET /tracks` | List all tracks |
| `GET /tracks/{track_id}` | Get track details |
| `POST /recommend` | Get recommendations |
| `GET /recommend?track_id=...&k=10` | Get recommendations (GET) |
| `GET /stats` | Dataset statistics |
| `POST /reload` | Reload the index |

## Performance Notes

### Library Size Recommendations

| Library Size | Backend | Settings |
|-------------|---------|----------|
| < 1,000 tracks | sklearn | Default |
| 1,000 - 10,000 tracks | sklearn or FAISS | Default |
| 10,000 - 50,000 tracks | FAISS (IVF) or HNSWlib | n_clusters=100, M=32 |
| 50,000+ tracks | FAISS (IVF) or HNSWlib | n_clusters=500, M=64 |

### FAISS Settings

For large libraries, use IVF index:
```yaml
faiss:
  index_type: "IVF"
  n_clusters: 100
```

### HNSWlib Settings

For fastest search with good accuracy:
```yaml
hnswlib:
  M: 32
  ef_construction: 200
  ef: 100
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
# Lint
flake8 music_recommender --max-line-length=100

# Format
black music_recommender
```

### Docker

```bash
# Build
docker build -t music-recommender:latest .

# Run
docker run -p 8000:8000 -v /path/to/music:/music music-recommender:latest
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
