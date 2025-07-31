from retriever.embedders import EMBEDDERS
from db.embedding_store_utils import get_segments_by_created_at, insert_embedding
from utils.config_loader import load_config
import numpy as np
import os
import uuid
from datetime import datetime

# Load config
config = load_config()
db_path = config["database"]["embedding_db_path"]
segment_dir = config["paths"]["segment_dir"]
embedding_out_dir = config["paths"]["embedding_dir"]

### Adjust these dates to generate embeddings for a specific time period
created_after = "2024-07-01T00:00:00"
created_before = "2024-08-01T00:00:00"

# Step 1: Fetch segment partition
segments = get_segments_by_created_at(db_path, created_after, created_before)

# Step 2: Prep accumulation containers for each embedder
accumulator = {}  # key: (model, mode) → {"embeddings": [], "segment_ids": []}

for embedder in EMBEDDERS:
    key = (embedder.model_name, embedder.mode)
    accumulator[key] = {"embeddings": [], "segment_ids": []}

# Step 3: Loop through segments and embed
for segment in segments:
    segment_id = segment["segment_id"]
    filepath = os.path.join(segment_dir, f"{segment_id}.mp4")  # adjust extension

    for embedder in EMBEDDERS:
        try:
            emb = embedder.embed(filepath)
            key = (embedder.model_name, embedder.mode)
            accumulator[key]["embeddings"].append(emb)
            accumulator[key]["segment_ids"].append(segment_id)
        except Exception as e:
            print(f"❌ Failed to embed {segment_id} with {embedder.model_name}: {e}")

# Step 4: Save batch .npy and .csv per embedder
for (model, mode), data in accumulator.items():
    embs = np.stack(data["embeddings"])  # shape: (N, D)
    seg_ids = data["segment_ids"]

    # Create output paths
    os.makedirs(embedding_out_dir, exist_ok=True)
    base = f"{model}_{mode}_{created_after[:10]}_{created_before[:10]}"
    npy_path = os.path.join(embedding_out_dir, f"{base}.npy")
    csv_path = os.path.join(embedding_out_dir, f"{base}.csv")

    np.save(npy_path, embs)
    with open(csv_path, "w") as f:
        for sid in seg_ids:
            f.write(sid + "\n")

    # Step 5 (Optional): Insert rows into embeddings table
    now = datetime.utcnow().isoformat()
    for sid in seg_ids:
        embedding_dict = {
            "embedding_id": str(uuid.uuid4()),
            "segment_id": sid,
            "mode": mode,
            "model_name": model,
            "embedding_type": "raw",
            "reducer_id": None,
            "contraster_id": None,
            "embedding_path": npy_path,  # reference the batch file
            "created_at": now
        }
        insert_embedding(db_path, embedding_dict)

print("✅ Embeddings generated and saved.")
