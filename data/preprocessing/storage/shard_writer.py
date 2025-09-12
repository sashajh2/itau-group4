from collections import defaultdict
import os, json, uuid
import numpy as np
from datetime import datetime, timezone
from dropbox_utils.dropbox_utils import get_client, upload_file
import re
from db.embedding_store_utils import insert_many_embeddings  # implement batch insert

TARGET_SHARD_BYTES = 512 * 1024 * 1024  # 512 MB
DTYPE = "float32"

def approx_bytes(n_rows, dim, dtype):
    return n_rows * dim * (4 if dtype == "float32" else 2)

def partition_dir(root, source, model, mode, noise, denoiser_name, version):
    dn = "none" if denoiser_name in (None, "", "none") else denoiser_name
    return f"{root}/{source}/raw/{mode}/{noise}/{model}/{dn}/v{version}/"

def find_existing_shards(dropbox_path):
    """Find existing shard files in Dropbox and return the highest shard index."""
    try:
        dbx = get_client()
        # List files in the partition directory
        result = dbx.files_list_folder(dropbox_path)
        
        shard_indices = []
        for entry in result.entries:
            if entry.name.endswith('.npy') and entry.name.startswith('shard_'):
                # Extract shard index from filename like "shard_000.npy"
                match = re.match(r'shard_(\d+)\.npy', entry.name)
                if match:
                    shard_indices.append(int(match.group(1)))
        
        return max(shard_indices) if shard_indices else -1
    except Exception as e:
        print(f"⚠️ Could not list existing shards in {dropbox_path}: {e}")
        return -1

def load_last_shard(dropbox_path, shard_index):
    """Load the last shard from Dropbox to continue adding to it."""
    try:
        dbx = get_client()
        shard_path = f"{dropbox_path}shard_{shard_index:03d}.npy"
        meta_path = f"{dropbox_path}shard_{shard_index:03d}.meta.json"
        
        # Download shard file
        _, response = dbx.files_download(shard_path)
        embeddings = np.load(response.content)
        
        # Download metadata
        _, meta_response = dbx.files_download(meta_path)
        metadata = json.loads(meta_response.content.decode())
        
        print(f"📥 Loaded existing shard {shard_index} with {len(embeddings)} embeddings")
        return embeddings, metadata
    except Exception as e:
        print(f"⚠️ Could not load shard {shard_index}: {e}")
        return None, None


class ShardWriter:
    def __init__(self, dropbox_root, db_path, source, version, tmp_dir="/tmp/emb"):
        self.dropbox_root = dropbox_root
        self.db_path = db_path
        self.source = source
        self.version = version
        self.tmp_dir = tmp_dir
        self.buf = defaultdict(lambda: {"embs": [], "segs": [], "dim": None, "shard_idx": 0, "loaded_from_existing": False})

    def add(self, model, mode, noise, denoiser_name, segment_id, emb):
        key = (model, mode, noise, denoiser_name or None)
        st = self.buf[key]
        emb = np.asarray(emb, dtype=np.float32)
        
        # Check for existing shards on first use
        if not st["loaded_from_existing"]:
            self._check_and_load_existing_shard(key, model, mode, noise, denoiser_name)
            st["loaded_from_existing"] = True
        
        if st["dim"] is None:
            st["dim"] = emb.shape[0]
        elif st["dim"] != emb.shape[0]:
            # skip / log mismatch
            return

        st["embs"].append(emb)
        st["segs"].append(segment_id)

        if approx_bytes(len(st["embs"]), st["dim"], DTYPE) >= TARGET_SHARD_BYTES:
            self._flush_key(key)

    def _check_and_load_existing_shard(self, key, model, mode, noise, denoiser_name):
        """Check for existing shards and load the last one if it's not full."""
        st = self.buf[key]
        
        # Get the Dropbox path for this partition
        d_dir = partition_dir(self.dropbox_root, self.source, model, mode, noise, denoiser_name, self.version)
        
        # Find the highest existing shard index
        max_shard_idx = find_existing_shards(d_dir)
        
        if max_shard_idx >= 0:
            # Load the last shard to check if it's full
            existing_embeddings, metadata = load_last_shard(d_dir, max_shard_idx)
            
            if existing_embeddings is not None and metadata is not None:
                # Check if the last shard is under the size limit
                current_size = approx_bytes(len(existing_embeddings), existing_embeddings.shape[1], DTYPE)
                
                if current_size < TARGET_SHARD_BYTES:
                    # Load the existing shard to continue adding to it
                    st["embs"] = existing_embeddings.tolist()
                    st["shard_idx"] = max_shard_idx
                    st["dim"] = existing_embeddings.shape[1]
                    print(f"📥 Continuing with existing shard {max_shard_idx} (size: {current_size/1024/1024:.1f}MB)")
                else:
                    # Last shard is full, start a new one
                    st["shard_idx"] = max_shard_idx + 1
                    print(f"📦 Last shard {max_shard_idx} is full, starting shard {st['shard_idx']}")
            else:
                # Couldn't load existing shard, start fresh
                st["shard_idx"] = max_shard_idx + 1
                print(f"⚠️ Could not load existing shard, starting fresh at index {st['shard_idx']}")
        else:
            # No existing shards, start from 0
            st["shard_idx"] = 0
            print(f"🆕 No existing shards found, starting fresh at index 0")

    def finalize(self):
        for key in list(self.buf.keys()):
            if self.buf[key]["embs"]:
                self._flush_key(key)

    def _flush_key(self, key):
        model, mode, noise, denoiser_name = key
        st = self.buf[key]
        embs = np.stack(st["embs"])
        segs = st["segs"]
        N, D = embs.shape

        # local write
        local_dir = os.path.join(self.tmp_dir, model, mode, noise, denoiser_name or "none", f"v{self.version}")
        os.makedirs(local_dir, exist_ok=True)
        shard_name = f"shard_{st['shard_idx']:03d}"
        npy_local = os.path.join(local_dir, shard_name + ".npy")
        meta_local = os.path.join(local_dir, shard_name + ".meta.json")
        np.save(npy_local, embs)
        with open(meta_local, "w") as f:
            json.dump({
                "model": model, "mode": mode, "noise": noise, "denoiser_name": denoiser_name,
                "version": self.version, "count": N, "dim": D, "dtype": DTYPE
            }, f, indent=2)

        # dropbox path
        d_dir = partition_dir(self.dropbox_root, self.source, model, mode, noise, denoiser_name, self.version)
        shard_dbx = d_dir + shard_name + ".npy"
        meta_dbx  = d_dir + shard_name + ".meta.json"

        # Upload files using the existing upload_file function with error handling
        try:
            upload_file(npy_local, shard_dbx, overwrite=True)
            upload_file(meta_local, meta_dbx, overwrite=True)
        except Exception as e:
            print(f"❌ Upload failed for shard {shard_name}: {e}")
            raise e

        # DB rows
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for i, sid in enumerate(segs):
            rows.append({
                "embedding_id": str(uuid.uuid4()),
                "segment_id": sid,
                "model_name": model,
                "mode": mode,
                "noise": noise,
                "denoiser_name": denoiser_name,
                "shard_path": shard_dbx,
                "row_index": i,
                "vector_dim": int(D),
                "dtype": DTYPE,
                "embedding_type": "raw",
                "reducer_id": None,
                "contraster_id": None,
                "version": str(self.version),
                "created_at": now
            })
        insert_many_embeddings(self.db_path, rows)

        # clear and advance
        st["embs"].clear(); st["segs"].clear(); st["shard_idx"] += 1
