from setuptools import setup, find_packages

setup(
    name="music-recommender-ai",
    version="0.1.0",
    description="Content-based music recommendation system",
    author="Music Recommender AI Team",
    author_email="info@musicrecommender.ai",
    url="https://github.com/yourusername/music_recommender_ai",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "mutagen>=1.47.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "embeddings": [
            "tensorflow>=2.12.0",
            "tensorflow-hub>=0.14.0",
        ],
        "large-scale": [
            "faiss-cpu>=1.7.0",
            "hnswlib>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "music-recommender=music_recommender.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.10",
)
