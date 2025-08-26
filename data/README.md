# Data Package

This package contains all data processing functionality for the itau-group4 project.

## Structure

```
data/
├── loaders/                 # Data loading from various sources
│   ├── avdeepfake.py       # AVDeepfake dataset loader
│   ├── dfd.py              # DFD dataset loader
│   ├── moviegen.py         # MovieGen dataset loader
│   └── synvta.py           # SynVTA dataset loader
├── preprocessing/           # Data preprocessing pipeline
│   ├── extractors/         # Data extraction logic
│   │   └── segment_extractor.py  # Video segment extraction
│   ├── generators/         # Embedding generation
│   │   ├── embedding_generator.py # Main embedding generation logic
│   │   └── embedding_saver.py     # Save embeddings to files
│   ├── storage/            # Storage operations
│   │   ├── dropbox_storage.py     # Dropbox upload/download
│   │   └── embedding_retriever.py # Retrieve embeddings
│   └── pipeline/           # Pipeline orchestration
│       ├── embedding_pipeline.py  # Embedding generation pipeline
│       └── batch_pipeline.py      # Batch processing pipeline
└── storage/                # Data storage
    └── database/           # Database operations
        └── setup.py        # Database setup and schema
```

## Usage

### From the root directory:

```bash
# Generate embeddings
python main.py embed 2024-01-01T00:00:00Z

# Process AVDeepfake parts in batch
python main.py batch 2 10

# Download and extract a specific part
python main.py download 001

# Extract segments from videos
python main.py extract ./videos/
```

### From Python code:

```python
from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from data.loaders.avdeepfake import download_and_extract_part
from data.preprocessing.extractors.segment_extractor import extract_and_insert_segments

# Generate embeddings
num_segments, num_uploaded = generate_for_created_at("2024-01-01T00:00:00Z")

# Download and extract
zip_path, part_out_dir, log_path = download_and_extract_part("001")

# Extract segments
num_segments = extract_and_insert_segments("./videos/", "2024-01-01T00:00:00Z")
```

## Key Features

- **Modular Design**: Each component has a single responsibility
- **Pipeline Architecture**: Clear data flow from loading to storage
- **Batch Processing**: Efficient processing of large datasets
- **Cloud Integration**: Dropbox storage for FAISS indices
- **Database Management**: SQLite storage for metadata
- **Error Handling**: Robust error handling and logging
- **CLI Interface**: Easy command-line access to all functions

## Migration from Old Structure

The old `scripts/` directory structure has been refactored into this semantic organization:

- `scripts/dataloaders/` → `data/loaders/`
- `scripts/preprocessing/` → `data/preprocessing/`
- `scripts/db/` → `data/storage/database/`
- `scripts/batch_processing/` → `data/preprocessing/pipeline/`

All functionality remains the same, but is now better organized and easier to navigate.
