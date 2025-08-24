# Data Loaders

This directory contains various data loaders for different datasets and sources.

## Available Loaders

### Core Loaders
- **`avdeepfake.py`** - AV-Deepfake1M dataset loader
- **`dfd.py`** - Deepfake Detection Challenge (DFDC) dataset loader
- **`moviegen.py** - Movie Generation dataset loader
- **`synvta.py`** - SYN-VTA dataset loader

### Instagram Loader
- **`instagram/`** - Instagram video scraping functionality
  - `instagram.py` - Main Instagram scraper module
  - `instagram_example.py` - Usage examples
  - `README_instagram.md` - Detailed documentation
  - `__init__.py` - Package initialization

## Usage

### Core Loaders
```python
from data.loaders import avdeepfake, dfd, moviegen, synvta
```

### Instagram Loader
```python
from data.loaders.instagram import scrape_public_account, scrape_multiple_accounts

# Scrape videos from a public Instagram account
result = scrape_public_account("username", "./output_dir")
```

## Installation

Most loaders use standard libraries, but the Instagram loader requires:
```bash
pip install instaloader
```

See individual loader documentation for specific requirements.
