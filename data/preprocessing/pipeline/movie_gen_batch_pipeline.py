import argparse
import os
import shutil
from datetime import datetime, timezone

from data.loaders.share_veo3 import download_and_extract_part, cleanup_files
from data.preprocessing.extractors.share_veo3_segment_extractor import extract_and_insert_share_veo3_segments
from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from utils.config_loader import load_config
from ..storage.shard_writer import ShardWriter  # kept for reference; Neon path is default


# Setup for movie gen processing