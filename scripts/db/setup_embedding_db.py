# setup_db.py

import sqlite3
import os
from utils.config_loader import load_config

config = load_config()

# ---------- Initialize Embedding DB ---------- #
embedding_db_path = config["database"]["embedding_db_path"]
os.makedirs(os.path.dirname(embedding_db_path), exist_ok=True)
conn1 = sqlite3.connect(embedding_db_path)
c1 = conn1.cursor()

c1.execute("""
CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY,        -- UUID: {source}_{video_id}_{segment_num}
    source TEXT,                        -- "AVDeepfake1M", "DFD"
    video_id TEXT,                      -- probably a video name
    video_path TEXT,                    -- absolute/relative path to full video (if available)
    start_time REAL,                    -- segment start time in seconds
    duration REAL,                      -- segment duration in seconds
    video_label TEXT,                   -- "real", "fake"
    audio_label TEXT                   -- "real", "fake"
);
""")

c1.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id TEXT PRIMARY KEY,        -- UUID or FAISS-assigned ID
    segment_id TEXT,                      -- Foreign key to segments table
    mode TEXT,                            -- "video", "audio", "video_noise", "audio_noise"
    model_name TEXT,                      -- "OpenL3", etc.
    embedding_type TEXT,                  -- "raw", "reduced", "contrasted"
    reducer_id TEXT,                      -- ID of reducer in model store (if applicable)
    contraster_id TEXT,                   -- ID of contraster in model store (if applicable)
    embedding_path TEXT,                  -- absolute/relative path to embedding file in Dropbox
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
);
""")

conn1.commit()
conn1.close()
print(f"Embedding DB initialized at: {embedding_db_path}")