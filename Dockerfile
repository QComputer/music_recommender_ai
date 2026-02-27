FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/music

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "music_recommender.cli", "serve"]

# Alternative commands:
# Scan: python -m music_recommender.cli scan --root /music
# Extract: python -m music_recommender.cli extract
# Build index: python -m music_recommender.cli build-index
# Recommend: python -m music_recommender.cli recommend
